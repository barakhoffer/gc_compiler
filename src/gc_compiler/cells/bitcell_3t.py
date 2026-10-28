"""3T gain-cell bitcell.

Topology (all nets lower-case, as they appear on the ports)::

    wbl ---|<o--- wwl          mw : PMOS write transistor
             |                 mr : NMOS read transistor, gate = storage node
            sn                 ms : NMOS read-select transistor
             |
    vss ---||--- sn            read current path: rbl -> ms -> mr -> vss
             |
           n_int
             |
    rbl ---||--- rwl

Layout style: both diffusion strips run vertically and sit on top of each other, so
``wwl``/``rwl`` are horizontal poly stripes shared by the whole row and the two bit lines
are vertical metal2 lines shared by the whole column. Rows are mirrored in y, which
merges the n-well bands and shares the ``wbl`` and ``rbl`` diffusion contacts between
vertically adjacent cells -- halving the junction capacitance seen by ``rbl``.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import gdsfactory as gf

from gc_compiler.cache import cached_cell
from gc_compiler.config import BitcellConfig
from gc_compiler.primitives.contacts import metal_pad, via_stack
from gc_compiler.primitives.wells import fix_min_area
from gc_compiler.primitives.devices import (
    DeviceStack,
    Gate,
    Node,
    build_device_stack,
    gate_pad_size,
    plan_stack,
)
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import LayerRole, Technology


@dataclass(frozen=True)
class BitcellGeometry:
    """Everything the array tiler and the periphery need to know about the cell."""

    pitch_x: float
    pitch_y: float
    rbl_x: float
    wbl_x: float
    vss_y: float
    wwl_y: float
    rwl_y: float
    wwl_width: float
    rwl_width: float
    nwell_y0: float
    nwell_y1: float
    rwl_poly_gap: float | None = None
    """From the ``rwl`` gate to the next poly along the strip (``mr``'s gate, which runs to the
    cell's edge): a word-line driver's landing pad, wider than the stripe, has to keep its spacing
    from it across the array's edge."""

    @property
    def area(self) -> float:
        return self.pitch_x * self.pitch_y


def _stacks(cfg: BitcellConfig) -> tuple[DeviceStack, DeviceStack]:
    read = DeviceStack(
        kind=cfg.read_kind,
        width=cfg.read_width,
        gates=(Gate("rwl", cfg.read_length), Gate("sn", cfg.effective_mr_length, pad="right")),
        nodes=(Node("rbl", boundary=True), Node("n_int", contact=False), Node("vss")),
        flavor=cfg.read_flavor,
    )
    write = DeviceStack(
        kind=cfg.write_kind,
        width=cfg.write_width,
        gates=(Gate("wwl", cfg.write_length),),
        nodes=(Node("sn"), Node("wbl", boundary=True)),
        flavor=cfg.write_flavor,
    )
    return read, write


def diff_margin(tech: Technology, cfg: BitcellConfig) -> float:
    """Distance from the cell's left edge to its diffusion: the poly endcap, or a threshold-voltage
    marker's enclosure of diffusion if that is larger, so a marker never reaches past the cell (where,
    at the array's edge, it would stick out of the n-well)."""
    rules = tech.rules
    margin = rules.poly_endcap
    for flavor in (cfg.write_flavor, cfg.read_flavor):
        vt = tech.vt_role(flavor)
        if vt is not None:
            margin = max(margin, rules.enc(vt, LayerRole.DIFF))
    return rules.snap(margin)


def bitcell_geometry(tech: Technology, cfg: BitcellConfig | None = None) -> BitcellGeometry:
    """Resolve the cell pitch and the position of every shared feature, without drawing."""
    cfg = cfg or BitcellConfig.for_tech(tech)
    rules = tech.rules
    read, write = _stacks(cfg)
    read_plan, write_plan = plan_stack(tech, read), plan_stack(tech, write)

    diff_x0 = diff_margin(tech, cfg)
    widest = max(cfg.read_width, cfg.write_width)
    min_pitch = widest + diff_x0 + rules.poly_endcap + gate_pad_size(tech) + rules.s(LayerRole.POLY)
    # an even number of grid steps: the column circuits centre themselves on a column
    step = 2 * rules.grid
    pitch_x = cfg.pitch_x or round(math.ceil(min_pitch / step - 1e-6) * step, 6)
    if pitch_x < min_pitch - rules.grid / 2:
        raise ValueError(f"pitch_x={pitch_x} is below the minimum legal pitch {min_pitch}")

    nwell_y0 = read_plan.y1 + rules.nwell_to_ndiff_space
    write_origin_y = nwell_y0 + rules.enc(LayerRole.NWELL, LayerRole.DIFF)
    pitch_y = write_origin_y + write_plan.node_y[-1]
    nwell_y1 = write_origin_y + write_plan.y1 + rules.enc(LayerRole.NWELL, LayerRole.DIFF)

    # Bit lines: rbl runs straight up from its own contact, wbl is jogged clear of it.
    rbl_x = diff_x0 + cfg.read_width / 2
    via_pad = rules.cut(LayerRole.VIA1) / 2 + rules.enc(LayerRole.M2, LayerRole.VIA1)
    wbl_x = rbl_x + via_pad + rules.s(LayerRole.M2) + rules.w(LayerRole.M2) / 2 + rules.grid
    # the column circuit below runs a third track, din, one track past wbl: the pitch holds all
    # three, clear of the next column's rbl
    if not cfg.pitch_x:
        # (the tracks carry via pads: din's, and the next column's rbl's, keep their spacing)
        track = rules.w(LayerRole.M2) + rules.s(LayerRole.M2)
        stack = tech.stack
        pads = max(
            rules.via_pad(role, cut) + rules.s(role)
            for role in (LayerRole.M1, LayerRole.M2)
            for cut in (stack.cut_between(stack.routing[stack.routing.index(role) - 1], role), stack.cut_above(role))
        )
        pitch_x = max(pitch_x, round(math.ceil((wbl_x + track + pads - rbl_x) / step - 1e-6) * step, 6))
    if wbl_x + via_pad > pitch_x - rules.s(LayerRole.M2) / 2:
        raise ValueError("cell pitch is too small to fit both bit lines on metal2")

    return BitcellGeometry(
        pitch_x=pitch_x,
        pitch_y=rules.snap(pitch_y),
        rbl_x=rules.snap(rbl_x),
        wbl_x=rules.snap(wbl_x),
        vss_y=rules.snap(read_plan.node_y[-1]),
        wwl_y=rules.snap(write_origin_y + write_plan.gate_y[0]),
        rwl_y=rules.snap(read_plan.gate_y[0]),
        wwl_width=cfg.write_length,
        rwl_width=cfg.read_length,
        nwell_y0=rules.snap(nwell_y0),
        nwell_y1=rules.snap(nwell_y1),
        rwl_poly_gap=rules.snap(
            (read_plan.gate_y[1] - cfg.effective_mr_length / 2) - (read_plan.gate_y[0] + cfg.read_length / 2)
        ),
    )


def gain_cell_3t(
    tech: Technology,
    cfg: BitcellConfig | None = None,
    name: str | None = None,
    replica: bool = False,
) -> gf.Component:
    """Build the 3T gain-cell bitcell.

    ``replica`` builds the replica-column variant, whose storage node is tied to ``wbl``: with
    ``wbl`` held at vdd the read transistor is always on, so a read always discharges ``rbl`` --
    the worst-case-fast cell that a self-timing replica bit line needs. The write transistor is
    left in place (and so is shorted, source to drain), which keeps the cell's footprint and
    its load on the word lines and bit lines exactly that of a real cell.
    """
    cfg = cfg or BitcellConfig.for_tech(tech)
    activate(tech)
    key = name or (f"{cfg.name}_rep" if replica else cfg.name)
    return cached_cell(key, lambda cell_name: _build(tech, cfg, cell_name, replica), tech)


def _build(tech: Technology, cfg: BitcellConfig, name: str, replica: bool = False) -> gf.Component:
    rules = tech.rules
    geo = bitcell_geometry(tech, cfg)
    read, write = _stacks(cfg)

    component = gf.Component(name)
    diff_x0 = diff_margin(tech, cfg)
    write_origin_y = geo.nwell_y0 + rules.enc(LayerRole.NWELL, LayerRole.DIFF)

    read_info = build_device_stack(component, tech, read, (diff_x0, 0.0))
    write_info = build_device_stack(component, tech, write, (diff_x0, write_origin_y))

    _draw_well(component, tech, geo)
    _connect_storage_node(component, tech, read_info, write_info)
    _draw_wordlines(component, tech, geo, read_info, write_info)
    _draw_supply(component, tech, geo, read_info)
    _draw_bitlines(component, tech, geo, read_info, write_info)
    if replica:
        _tie_storage_node(component, tech, geo, read_info)
    _add_ports(component, tech, cfg, geo)
    fix_min_area(component, tech, tile=(0.0, 0.0, geo.pitch_x, geo.pitch_y))

    component.info["pitch_x"] = geo.pitch_x
    component.info["replica"] = replica
    component.info["pitch_y"] = geo.pitch_y
    component.info["area"] = geo.area
    return component


def _draw_well(component: gf.Component, tech: Technology, geo: BitcellGeometry) -> None:
    """The n-well is a row-wide band: it merges across columns and across mirrored rows.

    Implants stay local to each diffusion (see ``build_device_stack``) so that tap columns
    can place the opposite implant type without implant-spacing violations.
    """
    draw(
        component,
        tech.layer(LayerRole.NWELL),
        Box(0.0, geo.nwell_y0, geo.pitch_x, geo.nwell_y1),
    )


def _connect_storage_node(component: gf.Component, tech: Technology, read_info, write_info) -> None:
    """Local interconnect from the write-side drain down to the read transistor gate."""
    li = tech.layer(LayerRole.LI)
    width = tech.rules.w(LayerRole.LI)
    pad = read_info.gate_pad_li[1]  # gate 1 of the read stack is mr, gated by sn
    node = write_info.node_li[0]  # bottom node of the write stack is sn
    draw(component, li, Box(node.x0, node.y0, pad.x1, node.y1))
    cx = pad.center[0]
    draw(component, li, Box(cx - width / 2, pad.y0, cx + width / 2, node.y1))


def _tie_storage_node(
    component: gf.Component, tech: Technology, geo: BitcellGeometry, read_info
) -> None:
    """Contact the storage node up to the ``wbl`` metal2 track that runs over it.

    The node's vertical li strip (from the read gate's pad up to the write drain) sits just
    right of the ``wbl`` track. The contact goes on the strip between the vss rail and the
    ``wbl`` contact at the cell boundary, as far from both as the spacing rules need, and its
    metal2 pad overlaps the track, which merges them.
    """
    rules = tech.rules
    pad = read_info.gate_pad_li[1]
    # pads elongated along x (relaxed enclosure in y) keep the stack short enough to fit
    via1_pad_half = (rules.cut(LayerRole.VIA1) + 2 * rules.enc_relaxed(LayerRole.M1, LayerRole.VIA1)) / 2
    rail_half = rules.cut(LayerRole.MCON) / 2 + rules.enc(LayerRole.M1, LayerRole.MCON)
    rail_top = read_info.node_li[2].center[1] + rail_half
    y = rules.snap(rail_top + rules.s(LayerRole.M1) + via1_pad_half + rules.grid)
    # the strip is as wide as the contact, so it needs no li pad of its own
    via_stack(component, tech, LayerRole.LI, LayerRole.M2, (pad.center[0], y), draw_bottom_pad=False, axis="x")


def _draw_wordlines(
    component: gf.Component, tech: Technology, geo: BitcellGeometry, read_info, write_info
) -> None:
    """Extend the two word-line gates across the full pitch so rows connect by abutment."""
    poly = tech.layer(LayerRole.POLY)
    for gate in (read_info.gate_poly[0], write_info.gate_poly[0]):
        draw(component, poly, Box(0.0, gate.y0, geo.pitch_x, gate.y1))


def _draw_supply(
    component: gf.Component, tech: Technology, geo: BitcellGeometry, read_info
) -> None:
    """vss rail on metal1, contacted at the source of the read transistor."""
    node = read_info.node_li[2]
    pads = via_stack(component, tech, LayerRole.LI, LayerRole.M1, node.center, draw_bottom_pad=False)
    rail = pads[LayerRole.M1]
    draw(component, tech.layer(LayerRole.M1), Box(0.0, rail.y0, geo.pitch_x, rail.y1))


def _draw_bitlines(
    component: gf.Component, tech: Technology, geo: BitcellGeometry, read_info, write_info
) -> None:
    """Vertical metal2 bit lines; both contacts sit on a cell boundary and are shared."""
    rules = tech.rules
    m2 = tech.layer(LayerRole.M2)
    half = rules.w(LayerRole.M2) / 2

    via_stack(component, tech, LayerRole.LI, LayerRole.M2, (geo.rbl_x, 0.0), draw_bottom_pad=False)
    draw(component, m2, Box(geo.rbl_x - half, 0.0, geo.rbl_x + half, geo.pitch_y))

    lower = via_stack(
        component, tech, LayerRole.LI, LayerRole.M1, (geo.rbl_x, geo.pitch_y), draw_bottom_pad=False
    )
    upper = via_stack(component, tech, LayerRole.M1, LayerRole.M2, (geo.wbl_x, geo.pitch_y))
    jog = Box.hull([lower[LayerRole.M1], upper[LayerRole.M1]])
    draw(component, tech.layer(LayerRole.M1), jog)
    draw(component, m2, Box(geo.wbl_x - half, 0.0, geo.wbl_x + half, geo.pitch_y))


def _add_ports(
    component: gf.Component, tech: Technology, cfg: BitcellConfig, geo: BitcellGeometry
) -> None:
    rules = tech.rules

    def add(name: str, center: tuple[float, float], orientation: float, role: LayerRole, w: float):
        add_port(component,
            name=name,
            center=center,
            orientation=orientation,
            width=w,
            layer=tech.layer(role),
            port_type="electrical",
        )
        label = tech.label_layer(role)
        if label is not None:
            component.add_label(text=name, position=center, layer=label)

    add("wwl", (0.0, geo.wwl_y), 180, LayerRole.POLY, cfg.write_length)
    add("rwl", (0.0, geo.rwl_y), 180, LayerRole.POLY, cfg.read_length)
    add("vss", (0.0, geo.vss_y), 180, LayerRole.M1, rules.w(LayerRole.M1))
    add("wbl", (geo.wbl_x, geo.pitch_y), 90, LayerRole.M2, rules.w(LayerRole.M2))
    add("rbl", (geo.rbl_x, 0.0), 270, LayerRole.M2, rules.w(LayerRole.M2))
