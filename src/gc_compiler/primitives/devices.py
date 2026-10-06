"""Generic transistor-stack builder.

A :class:`DeviceStack` is one diffusion strip carrying a series chain of transistors.
Current flows along +y, the transistor width is the x extent of the diffusion and gates
are horizontal poly stripes. Stacks are drawn directly into a parent component so that a
bitcell stays flat and abutment-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gdsfactory as gf

from gc_compiler.primitives.contacts import place_cuts
from gc_compiler.primitives.geometry import Box, draw
from gc_compiler.tech.rules import DeviceKind, LayerRole, Technology


@dataclass(frozen=True)
class Gate:
    """A transistor gate crossing the diffusion strip."""

    net: str
    length: float
    pad: str = "none"
    """Side the gate is contacted on: ``"none"``, ``"left"`` or ``"right"``."""


@dataclass(frozen=True)
class Node:
    """A source/drain node of the stack."""

    net: str
    contact: bool = True
    boundary: bool = False
    """Contact centred on the strip end so mirrored neighbours share it by abutment."""


@dataclass(frozen=True)
class DeviceStack:
    kind: DeviceKind
    width: float
    gates: tuple[Gate, ...]
    nodes: tuple[Node, ...]
    flavor: str | None = None
    gate_pitch: float = 0.0
    """Least centre-to-centre distance between consecutive gates (what a cell routing between
    their contacts needs); the rules' own spacing where larger."""

    def __post_init__(self) -> None:
        if len(self.nodes) != len(self.gates) + 1:
            raise ValueError("a stack needs exactly one more node than it has gates")


@dataclass(frozen=True)
class StackPlan:
    """1-D plan of a stack relative to its bottom anchor (y = 0)."""

    y0: float
    y1: float
    node_y: tuple[float | None, ...]
    gate_y: tuple[float, ...]

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class StackInfo:
    """Where everything ended up, in the parent component's coordinates."""

    diff: Box
    plan: StackPlan
    node_y: list[float | None] = field(default_factory=list)
    node_li: dict[int, Box] = field(default_factory=dict)
    gate_y: list[float] = field(default_factory=list)
    gate_poly: list[Box] = field(default_factory=list)
    gate_pad_li: dict[int, Box] = field(default_factory=dict)


def gate_pad_reach(tech: Technology, kind: DeviceKind) -> float:
    """How far a side gate pad starts from its device's diffusion: past the poly endcap, and
    further if the pad's contact would otherwise be closer to that diffusion than a poly contact
    may be (farther from p+ than from n+)."""
    rules = tech.rules
    space = rules.poly_contact_to_pdiff if kind is DeviceKind.PMOS else rules.poly_contact_to_diff
    needed = space - (gate_pad_size(tech) - rules.cut(LayerRole.LICON)) / 2
    if needed <= rules.poly_endcap:
        return rules.poly_endcap
    return rules.snap(needed + rules.grid)


def gate_pad_size(tech: Technology) -> float:
    rules = tech.rules
    return rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.POLY, LayerRole.LICON)


def _gate_half_extent(tech: Technology, gate: Gate) -> float:
    """Half height of the widest poly feature of a gate, used for spacing arithmetic."""
    if gate.pad == "none":
        return gate.length / 2
    half = max(gate.length, gate_pad_size(tech)) / 2
    if tech.has(LayerRole.POLY_CUT_MARKER):
        # A contact marker (e.g. sky130's npc) can reach further than the pad it decorates,
        # and its own spacing rule can exceed poly's; stacking two padded gates needs
        # clearance for whichever of the two ends up wider.
        rules = tech.rules
        marker_half = max(
            gate_pad_size(tech) / 2 + rules.enc(LayerRole.POLY_CUT_MARKER, LayerRole.LICON),
            rules.w(LayerRole.POLY_CUT_MARKER) / 2,
        )
        half = max(half, marker_half)
    return half


def plan_stack(tech: Technology, spec: DeviceStack) -> StackPlan:
    """Resolve node and gate positions relative to the bottom anchor of the stack.

    The anchor is the centre of the bottom contact for a boundary node, and the bottom
    diffusion edge otherwise.
    """
    rules = tech.rules
    too_small = [g.length for g in spec.gates if g.length < rules.device_min_length - 1e-9]
    if spec.width < rules.device_min_width - 1e-9 or too_small:
        raise ValueError(
            f"a {spec.kind.value} device of W={spec.width:g}, L={', '.join(f'{g.length:g}' for g in spec.gates)} is "
            f"below {tech.name}'s minimum W={rules.device_min_width:g}, L={rules.device_min_length:g}: size the "
            "configuration for this technology (BitcellConfig.for_tech, LogicConfig.for_tech, ArrayConfig.for_tech)"
        )
    steps = spec.width / (2 * rules.grid)
    if abs(steps - round(steps)) > 1e-6:
        raise ValueError(
            f"a device width of {spec.width:g} is not an even number of {rules.grid:g} grid steps: its contacts, "
            "centred on it, would fall off the grid"
        )
    node_ext = rules.contacted_node_extent()
    half_cut = rules.cut(LayerRole.LICON) / 2

    bottom = spec.nodes[0]
    if bottom.contact and bottom.boundary:
        y0, cursor = -node_ext, 0.0
    elif bottom.contact:
        y0, cursor = 0.0, node_ext
    else:
        y0, cursor = 0.0, None

    node_y: list[float | None] = [cursor]
    gate_y: list[float] = []
    prev_gate_top: float | None = None

    for gate, node in zip(spec.gates, spec.nodes[1:]):
        half = _gate_half_extent(tech, gate)
        if cursor is not None:
            center = cursor + half_cut + rules.gate_to_cut_space + half
        elif prev_gate_top is not None:
            center = prev_gate_top + rules.gate_to_gate_space + half
        else:
            center = y0 + rules.diff_extension + half
        if gate_y and spec.gate_pitch and center < gate_y[-1] + spec.gate_pitch - 1e-9:
            center = rules.snap_up(gate_y[-1] + spec.gate_pitch)
            if node_y[-1] is not None:  # the contacted node between them moves to the middle
                node_y[-1] = rules.snap((gate_y[-1] + center) / 2)
        gate_y.append(center)
        prev_gate_top = center + half
        cursor = prev_gate_top + rules.gate_to_cut_space + half_cut if node.contact else None
        node_y.append(cursor)

    if spec.nodes[-1].contact:
        y1 = node_y[-1] + node_ext
    else:
        y1 = prev_gate_top + rules.diff_extension
    return StackPlan(y0=y0, y1=y1, node_y=tuple(node_y), gate_y=tuple(gate_y))


def build_device_stack(
    component: gf.Component,
    tech: Technology,
    spec: DeviceStack,
    origin: tuple[float, float] = (0.0, 0.0),
    draw_implant: bool = True,
    draw_well: bool = False,
    draw_gate_pads: bool = True,
) -> StackInfo:
    """Draw ``spec`` into ``component`` with its bottom anchor at ``origin``.

    ``draw_gate_pads=False`` leaves the gates' contact pads (and the poly that would reach them)
    to the caller, who contacts the gates elsewhere; the gates are still spaced for them."""
    rules = tech.rules
    ox, oy = origin
    plan = plan_stack(tech, spec)
    diff = Box(ox, oy + plan.y0, ox + spec.width, oy + plan.y1)
    draw(component, tech.layer(LayerRole.DIFF), diff)

    info = StackInfo(diff=diff, plan=plan)
    pad_size = gate_pad_size(tech)

    for index, y in enumerate(plan.node_y):
        if y is None:
            info.node_y.append(None)
            continue
        cy = oy + y
        info.node_y.append(cy)
        _draw_node(component, tech, diff, cy, index, info)

    for gate, y in zip(spec.gates, plan.gate_y):
        drawn = gate if draw_gate_pads else Gate(gate.net, gate.length, pad="none")
        _draw_gate(component, tech, diff, drawn, oy + y, pad_size, info, spec.kind)

    if draw_implant:
        implant = tech.implant_role(spec.kind)
        draw(component, tech.layer(implant), diff.grow(rules.enc(implant, LayerRole.DIFF)))
    if draw_well:
        well = tech.well_role(spec.kind)
        if well is not None:
            draw(component, tech.layer(well), diff.grow(rules.enc(well, LayerRole.DIFF)))
    vt = tech.vt_role(spec.flavor)
    if vt is not None:
        draw(component, tech.layer(vt), diff.grow(rules.enc(vt, LayerRole.DIFF)))

    return info


def _li_strap(tech: Technology, hull: Box) -> Box:
    """li strap over a row of cuts: enclosure along the row only, min width across it."""
    rules = tech.rules
    enc = rules.enc(LayerRole.LI, LayerRole.LICON)
    width = max(hull.height, rules.w(LayerRole.LI))
    cy = hull.center[1]
    x0, x1 = hull.x0 - enc, hull.x1 + enc
    # (its minimum area is the merged shape's: see primitives.wells.fix_min_area)
    return Box(x0, cy - width / 2, x1, cy + width / 2)


def _draw_node(
    component: gf.Component,
    tech: Technology,
    diff: Box,
    cy: float,
    index: int,
    info: StackInfo,
) -> None:
    rules = tech.rules
    cut_size = rules.cut(LayerRole.LICON)
    region = Box(diff.x0, cy - cut_size / 2, diff.x1, cy + cut_size / 2)
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.DIFF, region, axis="x")
    li = _li_strap(tech, hull)
    draw(component, tech.layer(LayerRole.LI), li)
    info.node_li[index] = li


def _draw_gate(
    component: gf.Component,
    tech: Technology,
    diff: Box,
    gate: Gate,
    center_y: float,
    pad_size: float,
    info: StackInfo,
    kind: DeviceKind = DeviceKind.NMOS,
) -> None:
    rules = tech.rules
    endcap = rules.poly_endcap
    reach = gate_pad_reach(tech, kind)
    x0, x1 = diff.x0 - endcap, diff.x1 + endcap
    if gate.pad == "left":
        x0 = diff.x0 - reach - pad_size
    elif gate.pad == "right":
        x1 = diff.x1 + reach + pad_size

    poly = Box(x0, center_y - gate.length / 2, x1, center_y + gate.length / 2)
    draw(component, tech.layer(LayerRole.POLY), poly)
    info.gate_y.append(center_y)
    info.gate_poly.append(poly)

    if gate.pad == "none":
        return

    pad_cx = (x0 + pad_size / 2) if gate.pad == "left" else (x1 - pad_size / 2)
    pad = Box.centered(pad_cx, center_y, pad_size, pad_size)
    draw(component, tech.layer(LayerRole.POLY), pad)
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.POLY, pad, axis="x")
    tech.decorate_gate_contact(component, hull.as_tuple())
    li = _li_strap(tech, hull)
    draw(component, tech.layer(LayerRole.LI), li)
    info.gate_pad_li[len(info.gate_y) - 1] = li
