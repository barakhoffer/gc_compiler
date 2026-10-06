"""Fixed-height decoder gates with abutting horizontal metal1 supply rails.

Widths vary with gate count; transistor widths, well edges and supply rail positions
are shared across the family. The existing array-facing logic slices are unchanged.
"""

from __future__ import annotations

import math
from hashlib import sha256

import gdsfactory as gf

from gc_compiler.cache import cached_cell
from gc_compiler.periphery.logic import _lift, LogicConfig, _connect_supply, _draw_tap_rail, logic_geometry
from gc_compiler.primitives.wells import fix_min_area
from gc_compiler.primitives.contacts import place_cuts, via_stack
from gc_compiler.primitives.devices import DeviceStack, Gate, Node, _li_strap, build_device_stack, gate_pad_size, plan_stack
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import DeviceKind, LayerRole, Technology


_KINDS = {
    # kind: (inputs from the vss end of the NMOS stack to the output end, pull-up nodes, pull-down nodes)
    "inv": (("a",), (Node("vdd"), Node("y")), (Node("vss"), Node("y"))),
    "nand2": (
        ("b", "a"),
        (Node("y"), Node("vdd"), Node("y")),
        (Node("vss"), Node("mid"), Node("y")),
    ),
    "nand3": (
        ("c", "b", "a"),
        # the last y drain must sit opposite the NMOS y drain, so the output strip between them is
        # a straight bar rather than a bar over a vdd contact
        (Node("vdd"), Node("y"), Node("vdd"), Node("y")),
        (Node("vss"), Node("mid1"), Node("mid2"), Node("y")),
    ),
    # NORs are the dual: series PMOS, parallel NMOS whose drains are tied by a metal2 strap
    "nor2": (
        ("b", "a"),
        (Node("vdd"), Node("mid"), Node("y")),
        (Node("y"), Node("vss"), Node("y")),
    ),
    "nor3": (
        ("c", "b", "a"),
        (Node("vdd"), Node("mid1"), Node("mid2"), Node("y")),
        (Node("vss"), Node("y"), Node("vss"), Node("y")),
    ),
}


def standard_cell(
    tech: Technology, kind: str, cfg: LogicConfig | None = None, tight: bool = False,
) -> gf.Component:
    """Build an ``inv``, ``nand2``/``nand3`` or ``nor2``/``nor3`` with M2 signal pins and M1 power rails.

    Place at integer multiples of ``info['site_width']`` horizontally. Cells from
    the same configuration may abut without supply jumpers or intervening taps.
    ``LogicConfig.pitch`` is not used: row widths are derived from the devices.

    ``tight`` skips the rounding of the width up to a whole number of sites, so the cell is
    only as wide as its devices and well margins need (a ``nand2`` drops from 2.52 to 2.29
    um -- under the 2.46 um word-line pitch). Such a cell may sit at any x on the grid: its
    margins already guarantee spacing to a neighbour that abuts it.

    Pin sides: ``y`` is in the cell's lower half and ``a``/``b`` in its upper half, and the
    a NAND has an internal metal2 strap across the lower half (a NOR one across the NMOS
    side, clear of the input stubs) -- so route ``y`` out the bottom and the inputs out the
    top, never through the cell. ``a`` gates the series device next to ``y``, then ``b``
    (then ``c`` for a 3-input gate), as in the ``nand2_subckt``/``nor3_subckt`` family.
    """
    if kind not in _KINDS:
        raise ValueError(f"unsupported standard cell: {kind}")
    cfg = cfg or LogicConfig.for_tech(tech)
    if cfg.pitch is not None:
        raise ValueError("standard cells derive their own widths; pitch must be None")
    activate(tech)
    signature = sha256(repr((tech.name, cfg)).encode()).hexdigest()[:12]
    suffix = "_t" if tight else ""
    return cached_cell(f"gc_std_{kind}_{signature}{suffix}", lambda name: _build(tech, kind, cfg, name, tight), tech)


def _build(tech: Technology, kind: str, cfg: LogicConfig, name: str, tight: bool = False) -> gf.Component:
    rules = tech.rules
    geo = logic_geometry(tech, cfg)
    # The NMOS stack runs vss -> mid -> y, so the gate listed first is next to vss. The
    # schematic (``nand2_subckt``) puts ``a`` next to the output and ``b`` next to vss; the
    # inputs are logically interchangeable but not electrically, so LVS at any level above
    # the lone cell (where netgen may permute the ports) needs the same roles here.
    inputs, up_nodes, down_nodes = _KINDS[kind]
    gates = tuple(Gate(net, cfg.length, pad="right") for net in inputs)
    # a vss node between two gates of the pull-down reaches its rail on a metal stub that runs
    # between those gates' input contacts: the gates are far enough apart for the stub and the
    # two contacts' via pads (short sides relaxed), each at the layer's spacing
    pitch = 0.0
    if any(node.net == "vss" for node in down_nodes[1:-1]):
        stack = tech.stack
        m1 = stack.routing[1]
        first = stack.cut_between(stack.routing[0], m1)
        cuts = (first, stack.cut_above(m1))
        pad = max(max(rules.via_pad(m1, c, relaxed=True) for c in cuts), rules.w(m1))
        # ... grown along the gates to the layer's minimum area (its other side is the full pad)
        pad = max(pad, rules.snap_up(rules.area(m1) / max(rules.via_pad(m1, c) for c in cuts)))
        stub = max(rules.via_pad(m1, first, relaxed=True), rules.w(m1))
        pitch = pad + 2 * rules.s(m1) + stub
    pull_up = DeviceStack(DeviceKind.PMOS, cfg.pmos_width, gates, up_nodes, gate_pitch=pitch)
    pull_down = DeviceStack(DeviceKind.NMOS, cfg.nmos_width, gates, down_nodes, gate_pitch=pitch)
    plan = plan_stack(tech, pull_up)
    margin = max(rules.enc(LayerRole.NWELL, LayerRole.DIFF), rules.s(LayerRole.DIFF) / 2)
    site = rules.snap(rules.w(LayerRole.M2) + rules.s(LayerRole.M2))
    step = rules.grid if tight else site
    width = rules.snap(math.ceil((plan.height + 2 * margin) / step - 1e-9) * step)
    bottom, top = -margin, width - margin
    native = gf.Component(f"{name}_core")
    draw(native, tech.layer(LayerRole.BOUNDARY), Box(0, bottom, geo.width, top))
    draw(native, tech.layer(LayerRole.NWELL), Box(0, bottom, geo.nwell_x1, top))
    # no pads beside either stack (one would sit in the narrow gap between them, one against the
    # vss tap's implant): each gate is contacted once, where the logic slices contact theirs
    up = build_device_stack(native, tech, pull_up, (geo.pmos_x0, 0), draw_implant=False, draw_gate_pads=False)
    down = build_device_stack(native, tech, pull_down, (geo.nmos_x0, 0), draw_implant=False, draw_gate_pads=False)
    for spec, info in ((pull_up, up), (pull_down, down)):
        implant = tech.implant_role(spec.kind)
        enc = rules.enc(implant, LayerRole.DIFF)
        draw(native, tech.layer(implant), Box(info.diff.x0 - enc, bottom, info.diff.x1 + enc, top))
    for net, rail, body in (("vdd", geo.ntap_x, DeviceKind.PMOS), ("vss", geo.ptap_x, DeviceKind.NMOS)):
        _draw_tap_rail(native, tech, geo, rail, body, net, bottom, top)
    for index, node in enumerate(up_nodes):
        if node.net == "vdd":
            _connect_supply(native, tech, geo, up.node_li[index], geo.ntap_x)
    for index, node in enumerate(down_nodes):
        if node.net == "vss":
            _connect_supply(native, tech, geo, down.node_li[index], geo.ptap_x)
    y_up = [index for index, node in enumerate(up_nodes) if node.net == "y"]
    y_down = [index for index, node in enumerate(down_nodes) if node.net == "y"]
    # both stacks' last output drain sits at the same place along the gates, so the output
    # strip between them is a straight bar
    output = Box.hull([up.node_li[y_up[-1]], down.node_li[y_down[-1]]])
    draw(native, tech.layer(LayerRole.LI), output)
    signals = {}
    pad_size = gate_pad_size(tech)
    for index, net in enumerate(inputs):
        gate_y = up.gate_y[index]
        draw(native, tech.layer(LayerRole.POLY), Box(geo.pmos_x0, gate_y - cfg.length / 2, geo.input_pad_x, gate_y + cfg.length / 2))
        pad = Box.centered(geo.input_pad_x, gate_y, pad_size, pad_size)
        draw(native, tech.layer(LayerRole.POLY), pad)
        hull = place_cuts(native, tech, LayerRole.LICON, LayerRole.POLY, pad, axis="x")
        tech.decorate_gate_contact(native, hull.as_tuple())
        li = _li_strap(tech, hull)
        draw(native, tech.layer(LayerRole.LI), li)
        # the vss stubs cross this column at their nodes: the contact moves off the nearest one
        stubs = [down.node_y[i] for i, node in enumerate(down_nodes) if node.net == "vss" and down.node_y[i] is not None]
        nearest = min(stubs, key=lambda y: abs(y - gate_y)) if stubs else None
        center = _lift(native, tech, li, nearest)
        via_stack(native, tech, LayerRole.LI, LayerRole.M2, center, axis="x")
        signals[net] = center
    # a flipped cell's output wire leaves past its inputs' pins, and theirs past its own: the
    # output's via point keeps a metal2 wire's spacing from every input's pad, moved on an li
    # riser from its drain toward the cell's end if the drain is closer than that
    m2 = LayerRole.M2
    clear = max(rules.via_pad(m2, tech.stack.cut_between(LayerRole.M1, m2), relaxed=True), rules.w(m2)) / 2 + rules.w(m2) / 2 + rules.s(m2)
    drain = up.node_li[y_up[-1]]
    x, y = drain.center
    nearest = max(center[1] for center in signals.values())  # the inputs lie below the output
    if y - nearest < clear - 1e-9:
        y1 = rules.snap_up(nearest + clear)
        half = rules.w(LayerRole.LI) / 2
        draw(native, tech.layer(LayerRole.LI), Box(x - half, drain.y0, x + half, y1 + half))
        y = y1
    output_center = (x, y)
    via_stack(native, tech, LayerRole.LI, LayerRole.M2, output_center, axis="x")
    for stack, ys in ((up, y_up), (down, y_down)):
        if len(ys) > 1:  # the parallel devices' drains are tied by a metal2 strap
            start = stack.node_li[ys[0]].center
            end = output_center if stack is up else stack.node_li[ys[-1]].center
            via_stack(native, tech, LayerRole.LI, LayerRole.M2, start, axis="x")
            if stack is down:
                via_stack(native, tech, LayerRole.LI, LayerRole.M2, end, axis="x")
            half = rules.w(LayerRole.M2) / 2
            draw(native, tech.layer(LayerRole.M2), Box(start[0] - half, start[1], end[0] + half, end[1]))
    signals = {"y": output_center, **signals}
    for net, center in signals.items():
        add_port(native, name=net, center=center, width=rules.w(LayerRole.M2), orientation=0, layer=tech.layer(LayerRole.M2), port_type="electrical")
        label = tech.label_layer(LayerRole.M2)
        if label is not None:
            native.add_label(net, position=center, layer=label)
    component = gf.Component(name)
    ref = component.add_ref(native)
    ref.drotate(90)
    ref.dmove((top, 0))
    component.add_ports(ref.ports)
    component.info.update({"width": width, "height": geo.width, "site_width": site, "kind": kind})
    fix_min_area(component, tech)
    return component


def standard_cell_row(
    tech: Technology, kinds: tuple[str, ...], cfg: LogicConfig | None = None,
) -> gf.Component:
    """Abut gates with continuous supplies; leave numbered signal pins unrouted.

    This is a placement primitive, not a connected decoder. Signals are exposed
    as ``a_0``, ``y_0``, etc.; the row has one shared ``vdd``/``vss`` pair.
    """
    if not kinds:
        raise ValueError("a standard-cell row must contain at least one gate")
    cfg = cfg or LogicConfig.for_tech(tech)
    cells = [standard_cell(tech, kind, cfg) for kind in kinds]
    signature = sha256(repr(tuple(cell.name for cell in cells)).encode()).hexdigest()[:12]

    def build(name: str) -> gf.Component:
        component = gf.Component(name)
        offset = 0.0
        for index, cell in enumerate(cells):
            ref = component.add_ref(cell)
            ref.dmove((offset, 0))
            for port in ref.ports:
                if port.name in ("vdd", "vss"):
                    if index == 0:
                        add_port(component, port=port)
                else:
                    net = f"{port.name}_{index}"
                    add_port(component, name=net, port=port)
                    label = tech.label_layer(LayerRole.M2)
                    if label is not None:
                        component.add_label(net, position=port.dcenter, layer=label)
            offset = tech.rules.snap(offset + cell.info["width"])
        component.info.update({"width": offset, "height": cells[0].info["height"]})
        return component

    return cached_cell(f"gc_std_row_{signature}", build, tech)