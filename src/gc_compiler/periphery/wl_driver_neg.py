"""Word-line driver strip with a negative-rail write driver.

A PMOS write device only pulls the storage node down to about ``|Vtp|`` above its gate, so a
written '0' sits above the read device's threshold unless the write word line goes *below* vss.
Each write word line is therefore driven by a level shifter plus an inverter that swing between
``vdd`` and a negative ``vneg`` supply (see :func:`gc_compiler.netlist.spice.neg_driver_subckt`).

What that costs physically:

* the NMOS that pull down to ``vneg`` need a body at ``vneg``, i.e. their own p-well, which in
  sky130 means a **deep n-well**: an n-well ring (its inner wall 1.03 um and outer 0.4 um from the
  deep n-well's edge, ``DesignRules.isolation_ring_width``) around the p-type interior, whose
  substrate contact is the ``vneg`` rail. A well-tap-less p-substrate at ``vneg`` next to a
  ``vss`` substrate would be a forward-biased diode.
* the level shifter is ratioed (wide PMOS inputs against a long, weak NMOS latch) and, with the
  output inverter, is eight devices instead of two, so the strip is much wider.

The slice keeps the plain driver's contract -- the same word-line poly stubs on the right edge,
``<line>_in`` metal2 inputs on the left, ``vdd``/``vss`` rails -- and adds a ``vneg`` rail. It is
built to be stacked in y (every well, implant and rail spans the whole slice) and mirrored in x
for the right-hand strip. Vertical order, bottom to top::

    left slice  (rwl_0, wwl_1):  level-shifter | output driver | tracks (x, inb, in)
    right slice (wwl_0, rwl_1):  tracks (x, inb, in) | output driver | level-shifter

so that the level shifter's wires reach their tracks over the driver band, and the driver's
output leg (metal2, to the word line's poly) never crosses the read line's. The plain read
driver (an inverter) sits in the standard column at the left, next to the inverter that makes the
level shifter's complementary input.
"""

from __future__ import annotations

from dataclasses import dataclass

import gdsfactory as gf

from gc_compiler.cache import cached_cell
from gc_compiler.cells.bitcell_3t import BitcellGeometry
from gc_compiler.periphery.logic import LevelShifterConfig, LogicConfig, _lift, inverter, logic_geometry
from gc_compiler.periphery.wl_driver import (
    LEFT_WORDLINES,
    RIGHT_WORDLINES,
    WlDriverGeometry,
    _route_input,
    _route_to_wordline,
    input_exit_offset,
    strap_track_y,
    wordline_width,
    wordline_y,
)
from gc_compiler.primitives.wells import fix_min_area
from gc_compiler.primitives.contacts import min_pad, place_cuts, via_stack
from gc_compiler.primitives.devices import (
    gate_pad_reach,
    DeviceStack,
    Gate,
    Node,
    StackInfo,
    build_device_stack,
    gate_pad_size,
    plan_stack,
)
from gc_compiler.primitives.geometry import Box, cut_positions, draw, add_port
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import DeviceKind, LayerRole, Technology

def _overhang(tech: Technology, metal: LayerRole, cut: LayerRole) -> float:
    """How far a contacted node's ``metal`` pad reaches past the end of its diffusion."""
    rules = tech.rules
    return max(0.0, rules.cut(cut) / 2 + rules.enc(metal, cut) - rules.contacted_node_extent())


def _gaps(tech: Technology) -> tuple[float, float]:
    """``(band, track)``: the clear space between the level-shifter band and the driver band (their
    end diffusions, contacts' li and metal1 pads), and between a device band and a metal2 track's
    centre (the track's via1 pads, relaxed on their short sides, clear of the band's metal1)."""
    rules = tech.rules
    li_over = _overhang(tech, LayerRole.LI, LayerRole.LICON)
    m1_over = _overhang(tech, LayerRole.M1, LayerRole.MCON)
    band = max(
        rules.s(LayerRole.DIFF),
        rules.s(LayerRole.LI) + 2 * li_over,
        rules.s(LayerRole.M1) + 2 * m1_over,
    )
    # the band's own supply rows carry metal2 too (via1 pads, square): they may reach past it
    m2_pad = rules.via_pad(LayerRole.M2, LayerRole.VIA1)
    m2_over = max(0.0, m2_pad / 2 - rules.contacted_node_extent())
    track = max(
        rules.via_pad(LayerRole.M1, LayerRole.VIA1, relaxed=True) / 2 + rules.s(LayerRole.M1) + m1_over,
        m2_pad / 2 + rules.s(LayerRole.M2) + m2_over,
    )
    return rules.snap_up(band), rules.snap_up(track)


@dataclass(frozen=True)
class NegDriverGeometry:
    """Absolute positions in a negative-driver slice (origin at the slice's lower left)."""

    width: float
    height: float
    # x of the well structure, left to right
    g0: float
    """Left edge of the level shifter's n-well."""
    ntap_x: float
    pa_x0: float
    pb_x0: float
    i0: float
    """Left edge of the isolated p-well's interior: the n-well ends here."""
    nz_x0: float
    nx_x0: float
    ptap_x: float
    i1: float
    wall_x1: float
    pd_x0: float
    nd_x0: float
    tap_w: float
    pad_cx: float
    """x of the gate pad NZ shares with the output inverter (the latch's ``x`` net)."""
    # word-line landing, as in :class:`~gc_compiler.periphery.wl_driver.WlDriverGeometry`
    track_x: tuple[float, ...]
    pad_x: float


def neg_driver_geometry(
    tech: Technology,
    cell: BitcellGeometry,
    logic: LogicConfig,
    ls: LevelShifterConfig,
) -> NegDriverGeometry:
    rules = tech.rules
    lg = logic_geometry(tech, logic)
    tap_w = rules.tap_width()
    pad = gate_pad_size(tech)
    end = rules.poly_endcap
    gg = rules.s(LayerRole.POLY)
    # a P+ and an N+ diffusion side by side: each implant's enclosure, and the implants' spacing
    implant_gap = rules.enc(LayerRole.PIMP, LayerRole.DIFF) + rules.enc(LayerRole.NIMP, LayerRole.DIFF) + rules.s(LayerRole.NIMP)

    # the level shifter's n-well starts clear of the standard column's p-tap (its right edge) and
    # its n-diffusion
    g0 = lg.ptap_x + tap_w + max(rules.nwell_to_ptap_space, rules.nwell_to_ndiff_space)
    ntap_x = g0 + rules.enc(LayerRole.NWELL, LayerRole.TAP)
    # every gate here is contacted on its left: its pad starts gate_pad_reach from its own
    # diffusion (farther for PMOS), clear of the neighbour's endcap by poly spacing
    reach_p, reach_n = gate_pad_reach(tech, DeviceKind.PMOS), gate_pad_reach(tech, DeviceKind.NMOS)
    # a padded gate needs pad + reach to its left, clear of the tap and of the implant edge: the
    # pad's poly and contact keep their spacing to the tap, and its li to the tap's li strip
    cut, enc_li, w_li = rules.cut(LayerRole.LICON), rules.enc(LayerRole.LI, LayerRole.LICON), rules.w(LayerRole.LI)
    pad_left = max(
        ntap_x + tap_w + max(rules.poly_to_diff, rules.poly_contact_to_diff - (pad - cut) / 2),
        ntap_x + tap_w / 2 + w_li / 2 + rules.s(LayerRole.LI) + (cut + 2 * enc_li) / 2 - pad / 2,
    )
    pa_x0 = max(ntap_x + tap_w + implant_gap, rules.snap_up(pad_left) + pad + reach_p)
    pb_x0 = pa_x0 + ls.pmos_width + end + gg + pad + reach_p
    pb_x1 = pb_x0 + ls.pmos_width
    # The deep n-well's edge is i0 - 1.03; it must not cut a P+ diffusion (dnwell.5), so it
    # stays clear of the level shifter's PMOS, which therefore end left of it.
    i0 = pb_x1 + rules.isolation_ring_inner + rules.isolation_to_pdiff_space
    nz_x0 = max(pb_x1 + end + gg + pad + reach_n, i0 + rules.nwell_to_ndiff_space)
    nz_x1 = nz_x0 + ls.latch_width
    nx_x0 = nz_x1 + end + gg + pad + reach_n
    nx_x1 = nx_x0 + ls.latch_width
    ptap_x = nx_x1 + implant_gap
    i1 = ptap_x + tap_w + rules.nwell_to_ndiff_space
    wall_x1 = i1 + rules.isolation_ring_width

    via_pad = rules.cut(LayerRole.VIA1) + 2 * rules.enc(LayerRole.M2, LayerRole.VIA1)
    track_pitch = via_pad + rules.s(LayerRole.M2)
    # The array's n-wells must stay 4.5 um from the deep n-well (nwell.7), which ends 0.4 um inside
    # the ring's wall. They reach past the array's edge by the well's enclosure of the array's end
    # taps (the tap rows' wells end there, see primitives.wells.close_well_ends), so that much more
    # room. The word-line landing (tracks and poly pad) is put against the array, not the wall, so
    # the poly stubs stay short: they are resistance in every word line.
    # (and the ring itself keeps the n-well spacing from them: they are different nets)
    reach = max(rules.enc(LayerRole.NWELL, LayerRole.TAP), rules.enc(LayerRole.NWELL, LayerRole.DIFF))
    width = rules.snap(
        max(wall_x1 - rules.isolation_ring_outer + rules.isolation_nwell_space, wall_x1 + rules.s(LayerRole.NWELL))
        + reach + tech.policy.margin
    )
    pad_x = rules.snap(width - rules.poly_endcap - pad / 2)
    last = pad_x - pad / 2 - track_pitch / 2
    tracks = tuple(rules.snap(last - (1 - i) * track_pitch) for i in range(2))

    # The output NMOS starts under NZ, right of the gate pad the two share. Its contact li starts
    # further in the narrower it is (a centred cut array); a wide one (a strong driver) puts it at
    # the diffusion's edge, too close to the pad's li, so the device moves right by the shortfall.
    pad_cx = nz_x0 - end - pad / 2
    cut, cut_pitch = rules.cut(LayerRole.LICON), rules.cut_pitch(LayerRole.LICON)
    enc_li = rules.enc(LayerRole.LI, LayerRole.LICON)
    offsets = cut_positions(logic.nmos_width - 2 * rules.enc(LayerRole.DIFF, LayerRole.LICON), cut, cut_pitch)
    contact_li_x0 = logic.nmos_width / 2 + offsets[0] - cut / 2 - enc_li  # from the diffusion's left edge
    pad_li_x1 = pad_cx + cut / 2 + enc_li
    # the pad's via stack puts a metal1 pad beside the output's source stack: their spacing too
    stack = tech.stack
    m1 = stack.routing[1]
    m1_half = max(rules.via_pad(m1, c) for c in (stack.cut_between(stack.routing[0], m1), stack.cut_above(m1))) / 2
    contact_m1_x0 = logic.nmos_width / 2 + offsets[0] - m1_half
    nd_x0 = max(
        nz_x0,
        pad_li_x1 + rules.s(LayerRole.LI) - contact_li_x0 + rules.grid,
        pad_cx + m1_half + rules.s(m1) - contact_m1_x0,
    )

    return NegDriverGeometry(
        width=width,
        height=rules.snap(2 * cell.pitch_y),
        g0=rules.snap(g0),
        ntap_x=rules.snap(ntap_x),
        pa_x0=rules.snap(pa_x0),
        pb_x0=rules.snap(pb_x0),
        i0=rules.snap(i0),
        nz_x0=rules.snap(nz_x0),
        nx_x0=rules.snap(nx_x0),
        ptap_x=rules.snap(ptap_x),
        i1=rules.snap(i1),
        wall_x1=rules.snap(wall_x1),
        pd_x0=rules.snap(pb_x1 - logic.pmos_width),
        nd_x0=rules.snap(nd_x0),
        tap_w=tap_w,
        pad_cx=rules.snap(pad_cx),
        track_x=tracks,
        pad_x=pad_x,
    )


def neg_wl_driver_geometry(
    tech: Technology,
    cell: BitcellGeometry,
    logic: LogicConfig,
    ls: LevelShifterConfig,
    wordlines: tuple[str, ...],
) -> WlDriverGeometry:
    """The plain driver's geometry record (what the floorplan reads), for this slice."""
    geo = neg_driver_geometry(tech, cell, logic, ls)
    return WlDriverGeometry(
        width=geo.width,
        height=geo.height,
        track_x=geo.track_x,
        pad_x=geo.pad_x,
        driver_y=(),
        wordlines=tuple(wordlines),
    )


def wl_driver_pair_neg(
    tech: Technology,
    cell: BitcellGeometry,
    logic: LogicConfig | None = None,
    ls: LevelShifterConfig | None = None,
    wordlines: tuple[str, ...] = LEFT_WORDLINES,
    name: str | None = None,
    straps: bool = False,
) -> gf.Component:
    """Driver slice covering one mirrored row pair, its ``wwl`` driven from ``vneg``. ``straps``:
    as for :func:`~gc_compiler.periphery.wl_driver.wl_driver_pair`."""
    logic = logic or LogicConfig.for_tech(tech)
    ls = ls or LevelShifterConfig.for_tech(tech)
    activate(tech)
    key = name or f"gc_wl_driver_neg_{'_'.join(wordlines)}_{logic.name}_{ls.name}_{cell.pitch_y:g}" + ("_ws" if straps else "")
    return cached_cell(key, lambda n: _build(tech, cell, logic, ls, wordlines, n, straps), tech)


# ---------------------------------------------------------------------------------------
# construction


def _stack(kind, width, length, gate_net, pad, nodes) -> DeviceStack:
    return DeviceStack(kind, width, (Gate(gate_net, length, pad),), tuple(Node(n) for n in nodes))


def _tap_column(
    component: gf.Component,
    tech: Technology,
    x0: float,
    height: float,
    body_of: DeviceKind,
    net: str,
    contact_ys: list[float],
    port: bool = True,
) -> float:
    """Full-height tap strip (implant, contacts at ``contact_ys``, li, metal1 rail); returns
    the rail's centre x. ``port`` also exposes the rail at the slice's bottom edge."""
    rules = tech.rules
    tap_w = rules.tap_width()
    strip = Box(x0, 0.0, x0 + tap_w, height)
    draw(component, tech.layer(LayerRole.TAP), strip)
    implant = tech.tap_implant_role(body_of)
    draw(component, tech.layer(implant), strip.grow(rules.enc(implant, LayerRole.TAP), 0.0))
    cx = strip.center[0]
    cut = rules.cut(LayerRole.LICON)
    li_w = rules.w(LayerRole.LI)
    draw(component, tech.layer(LayerRole.LI), Box(cx - li_w / 2, 0.0, cx + li_w / 2, height))
    rail = None
    for y in contact_ys:
        place_cuts(
            component, tech, LayerRole.LICON, LayerRole.TAP,
            Box(strip.x0, y - cut / 2, strip.x1, y + cut / 2), axis="x",
        )
        pads = via_stack(component, tech, LayerRole.LI, LayerRole.M1, (cx, y), draw_bottom_pad=False)
        rail = pads[LayerRole.M1]
    draw(component, tech.layer(LayerRole.M1), Box(rail.x0, 0.0, rail.x1, height))
    label = tech.label_layer(LayerRole.M1)
    if label is not None:
        component.add_label(text=net, position=(cx, contact_ys[0]), layer=label)
    if port:
        add_port(component,
            name=net, center=(cx, 0.0), orientation=270, width=rail.width,
            layer=tech.layer(LayerRole.M1), port_type="electrical",
        )
    return cx


def _stub(component: gf.Component, tech: Technology, x: float, y_from: float, y_to: float) -> None:
    """Metal1 wire from an li pin at ``(x, y_from)`` (contacted there) to ``(x, y_to)``."""
    half = tech.rules.w(LayerRole.M1) / 2
    pads = via_stack(component, tech, LayerRole.LI, LayerRole.M1, (x, y_from), draw_bottom_pad=False)
    # a short stub is little more than its pads: it is made up to metal1's minimum area
    wire = Box.hull([pads[LayerRole.M1], Box(x - half, min(y_from, y_to) - half, x + half, max(y_from, y_to) + half)])
    draw(component, tech.layer(LayerRole.M1), min_pad(tech, wire, LayerRole.M1, "y"))


def _track_via(component: gf.Component, tech: Technology, x: float, y: float) -> None:
    """Land a metal1 wire on a metal2 track: via1 and its pads."""
    via_stack(component, tech, LayerRole.M1, LayerRole.M2, (x, y), axis="x")


def _m2_line(component: gf.Component, tech: Technology, y: float, x0: float, x1: float) -> None:
    half = tech.rules.w(LayerRole.M2) / 2
    draw(component, tech.layer(LayerRole.M2), Box(min(x0, x1), y - half, max(x0, x1), y + half))


def _build(
    tech: Technology,
    cell: BitcellGeometry,
    logic: LogicConfig,
    ls: LevelShifterConfig,
    wordlines: tuple[str, ...],
    name: str,
    straps: bool = False,
) -> gf.Component:
    rules = tech.rules
    geo = neg_driver_geometry(tech, cell, logic, ls)
    lg = logic_geometry(tech, logic)
    H = geo.height
    wl_y = wordline_y(cell)
    strap_y = strap_track_y(tech, cell) if straps else {}
    wl_w = wordline_width(cell)
    pad = gate_pad_size(tech)
    end = rules.poly_endcap
    li = tech.layer(LayerRole.LI)
    m1, m2 = tech.layer(LayerRole.M1), tech.layer(LayerRole.M2)

    neg_line = next(net for net in wordlines if net.startswith("wwl"))
    read_line = next(net for net in wordlines if net.startswith("rwl"))
    # left slice: the read line is at the bottom (rwl_0), the write driver above it
    left = tuple(wordlines) == tuple(LEFT_WORDLINES)
    if not left and tuple(wordlines) != tuple(RIGHT_WORDLINES):
        raise ValueError(f"unsupported word-line set {wordlines}")

    component = gf.Component(name)

    # --- device stacks ------------------------------------------------------------------
    pmos, nmos = DeviceKind.PMOS, DeviceKind.NMOS
    flipped = not left  # right slice: drains at the bottom of the level-shifter band
    ls_nodes_p = ("vdd", "x") if not flipped else ("x", "vdd")
    ls_nodes_pz = ("vdd", "z") if not flipped else ("z", "vdd")
    ls_nodes_nz = ("vneg", "z") if not flipped else ("z", "vneg")
    ls_nodes_nx = ("vneg", "x") if not flipped else ("x", "vneg")
    # In the level shifter's drain-at-top orientation the drains face the tracks above the
    # band; flipped, they face the tracks below it. Sources always sit on the far side.
    spec = {
        "pa": _stack(pmos, ls.pmos_width, ls.input_length, "inb", "left", ls_nodes_p),
        "pb": _stack(pmos, ls.pmos_width, ls.input_length, "in", "left", ls_nodes_pz),
        "nz": _stack(nmos, ls.latch_width, ls.latch_length, "x", "left", ls_nodes_nz),
        "nx": _stack(nmos, ls.latch_width, ls.latch_length, "z", "left", ls_nodes_nx),
    }
    plan = {k: plan_stack(tech, v) for k, v in spec.items()}
    _GAP_BAND, _GAP_TRACK = _gaps(tech)
    band_l = max(p.y1 for p in plan.values())
    drv_pmos = _stack(pmos, logic.pmos_width, logic.length, "x", "none", ("vdd", "wwl"))
    drv_nmos = _stack(nmos, logic.nmos_width, logic.length, "x", "none", ("vneg", "wwl"))
    plan_d = plan_stack(tech, drv_pmos)
    band_d = plan_d.y1

    # --- vertical plan --------------------------------------------------------------------
    track = input_exit_offset(tech)  # the tracks' pitch: an output leg and the line above it
    if left:
        # the inverter making `inb` sits by the tracks (metal2 line `in` at its exit), as low as
        # the bands below it allow: the level-shifter band then starts at the slice's bottom
        y_inb0 = rules.snap_up(band_l + _GAP_BAND + band_d + _GAP_TRACK + track - lg.out_y)
        y_read_inv = 0.0
    else:
        y_inb0 = 0.0
        y_read_inv = rules.snap(cell.pitch_y)  # the slot the plain driver used
    out_inb = y_inb0 + lg.out_y
    y_in = rules.snap(out_inb + track)  # what _route_input gives for an inverter at y_inb0
    y_inb = out_inb
    y_x = rules.snap(y_inb - track)
    if left:
        d_y0 = rules.snap(y_x - _GAP_TRACK - band_d)
        l_y0 = rules.snap(d_y0 - _GAP_BAND - band_l)
    else:
        d_y0 = rules.snap(y_in + _GAP_TRACK)
        l_y0 = rules.snap(d_y0 + band_d + _GAP_BAND)
    if l_y0 < 0 or max(l_y0 + band_l, d_y0 + band_d, y_in) > H:
        raise ValueError("the negative driver does not fit in its slice")

    # --- wells, implants, taps ----------------------------------------------------------------
    nw = tech.layer(LayerRole.NWELL)
    draw(component, nw, Box(0.0, 0.0, lg.nwell_x1, H))  # the standard column's stripe
    draw(component, nw, Box(geo.g0, 0.0, geo.i0, H))  # the level shifter's PMOS well, and the ring's left wall
    draw(component, nw, Box(geo.i1, 0.0, geo.wall_x1, H))  # the ring's right wall
    dn = rules.isolation_ring_inner
    draw(component, tech.layer(LayerRole.DNWELL), Box(geo.i0 - dn, 0.0, geo.i1 + dn, H))

    ins_p, ins_n = rules.enc(LayerRole.PIMP, LayerRole.DIFF), rules.enc(LayerRole.NIMP, LayerRole.DIFF)
    pimp, nimp = tech.layer(LayerRole.PIMP), tech.layer(LayerRole.NIMP)
    pb_x1 = geo.pb_x0 + ls.pmos_width
    nx_x1 = geo.nx_x0 + ls.latch_width
    draw(component, pimp, Box(geo.pa_x0 - ins_p, 0.0, pb_x1 + ins_p, H))  # PMOS source/drain
    draw(component, nimp, Box(geo.nz_x0 - ins_n, 0.0, nx_x1 + ins_n, H))  # isolated NMOS

    # tap contacts at each edge, as close as the next slice's (mirror images) allow and still
    # enclosed where the column ends, and one midway
    edge = rules.snap_up(max(
        [(rules.cut(c) + rules.cut_space[c]) / 2 for c in (LayerRole.LICON, tech.stack.cut_above(LayerRole.LI))]
        + [rules.cut(LayerRole.LICON) / 2 + rules.enc(LayerRole.TAP, LayerRole.LICON)]
    ))
    contact_ys = [edge, H / 2, H - edge]
    # rails: standard column (as the plain driver) ...
    for x, net in ((lg.ntap_x, "vdd"), (lg.ptap_x, "vss")):
        cx = x + geo.tap_w / 2
        draw(component, m1, Box(cx - rules.w(LayerRole.M1) / 2, 0.0, cx + rules.w(LayerRole.M1) / 2, H))
        add_port(component,
            name=net, center=(cx, 0.0), orientation=270, width=rules.w(LayerRole.M1),
            layer=m1, port_type="electrical",
        )
    # ... and the level shifter's: vdd in its PMOS well, vneg in the isolated well
    # (the second vdd rail is the standard column's net, joined to it by the driver's vdd bus below)
    rail_n = _tap_column(component, tech, geo.ntap_x, H, DeviceKind.PMOS, "vdd", contact_ys, port=False)
    rail_p = _tap_column(component, tech, geo.ptap_x, H, DeviceKind.NMOS, "vneg", contact_ys)

    # --- the standard column: read driver, and the inverter that makes inb --------------------
    inv = inverter(tech, logic)
    read_ref = component.add_ref(inv)
    read_ref.dmove((0.0, y_read_inv))
    inb_ref = component.add_ref(inv)
    inb_ref.dmove((0.0, y_inb0))
    read_out = tuple(read_ref.ports["y"].dcenter)
    _route_to_wordline(
        component, tech, out=read_out, track_x=geo.track_x[0], pad_x=geo.pad_x,
        wl_y=wl_y[read_line], right_edge=geo.width, length=wl_w[read_line],
        strap_y=strap_y.get(read_line),
    )
    read_exit = _route_input(component, tech, tuple(read_ref.ports["a"].dcenter), read_ref.ports["y"].dcenter[1])
    in_exit = _route_input(component, tech, tuple(inb_ref.ports["a"].dcenter), inb_ref.ports["y"].dcenter[1])
    if abs(in_exit - y_in) > 1e-6:
        raise AssertionError(f"input line at {in_exit}, expected {y_in}")
    add_port(component,
        name=read_line, center=(geo.width, wl_y[read_line]), orientation=0,
        width=wl_w[read_line], layer=tech.layer(LayerRole.POLY), port_type="electrical",
    )
    add_port(component,
        name=f"{read_line}_in", center=(0.0, read_exit), orientation=180,
        width=rules.w(LayerRole.M2), layer=m2, port_type="electrical",
    )
    add_port(component,
        name=f"{neg_line}_in", center=(0.0, in_exit), orientation=180,
        width=rules.w(LayerRole.M2), layer=m2, port_type="electrical",
    )

    # --- level shifter and driver devices ---------------------------------------------------------
    infos: dict[str, StackInfo] = {}
    x_of = {"pa": geo.pa_x0, "pb": geo.pb_x0, "nz": geo.nz_x0, "nx": geo.nx_x0}
    # With the drains on top the PMOS stacks (shorter: their gates are short) are raised so both
    # kinds' drain rows line up and one li strap can join them; flipped, the drains are at the
    # bottom and already line up.
    raise_p = 0.0 if flipped else rules.snap(band_l - max(plan["pa"].y1, plan["pb"].y1))
    for key, dev in spec.items():
        y0 = l_y0 + (raise_p if dev.kind is pmos else 0.0)
        infos[key] = build_device_stack(component, tech, dev, (x_of[key], y0), draw_implant=False)
    d_pmos = build_device_stack(component, tech, drv_pmos, (geo.pd_x0, d_y0), draw_implant=False)
    d_nmos = build_device_stack(component, tech, drv_nmos, (geo.nd_x0, d_y0), draw_implant=False)

    def drain(key: str) -> Box:
        return infos[key].node_li[1 if not flipped else 0]

    def source(key: str) -> Box:
        return infos[key].node_li[0 if not flipped else 1]

    def gate_pad(key: str) -> Box:
        return infos[key].gate_pad_li[0]

    # z: PB and NZ drains share a strap (both at the band's drain row), and reach NX's gate pad by
    # an li L: along the drain row to the pad's x, then along the pad
    z_strap = Box.hull([drain("pb"), drain("nz")])
    draw(component, li, z_strap)
    nx_pad = gate_pad("nx")
    bar_w = rules.w(LayerRole.LI)
    y_d = drain("pb").center[1]
    draw(component, li, Box(drain("nz").x1, y_d - bar_w / 2, nx_pad.center[0] + bar_w / 2, y_d + bar_w / 2))
    draw(
        component, li,
        Box(nx_pad.center[0] - bar_w / 2, min(y_d, nx_pad.center[1]) - bar_w / 2,
            nx_pad.center[0] + bar_w / 2, max(y_d, nx_pad.center[1]) + bar_w / 2),
    )

    # output inverter: shared gate stripe (input x) with a pad between the two devices, and an
    # output strap over both drains
    gate_y = d_y0 + plan_d.gate_y[0]
    stripe = Box(
        geo.pd_x0 - end, gate_y - logic.length / 2,
        geo.nd_x0 + logic.nmos_width + end, gate_y + logic.length / 2,
    )
    draw(component, tech.layer(LayerRole.POLY), stripe)
    # The pad sits between the two devices, right under the output strap that joins their drains,
    # so it hangs below the stripe: its li (a contact's, grown to li's minimum area) keeps li
    # spacing from the strap, while the poly pad still covers the stripe.
    out_strap = Box.hull([d_pmos.node_li[1], d_nmos.node_li[1]])
    enc_li = rules.enc(LayerRole.LI, LayerRole.LICON)
    cut = rules.cut(LayerRole.LICON)
    li_shape = min_pad(
        tech, Box.centered(0.0, 0.0, cut + 2 * enc_li, max(cut, rules.w(LayerRole.LI))), LayerRole.LI, "y"
    )
    pad_y = rules.snap_down(min(gate_y, out_strap.y0 - rules.s(LayerRole.LI) - li_shape.height / 2))
    pad_box = Box.centered(geo.pad_cx, pad_y, pad, pad)
    draw(component, tech.layer(LayerRole.POLY), pad_box)
    if pad_box.y1 < gate_y + logic.length / 2 - 1e-9:
        # hung lower than the pad reaches: a poly bridge, the pad's width, back up to the stripe
        draw(component, tech.layer(LayerRole.POLY), Box(pad_box.x0, pad_box.y1 - rules.grid, pad_box.x1, gate_y))
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.POLY, pad_box, axis="x")
    tech.decorate_gate_contact(component, hull.as_tuple())
    li_h = max(hull.height, rules.w(LayerRole.LI))
    # (grown to its minimum area along the stripe, not toward the output NMOS beside it)
    d_pad_li = min_pad(tech, Box(hull.x0 - enc_li, pad_y - li_h / 2, hull.x1 + enc_li, pad_y + li_h / 2), LayerRole.LI, "y")
    draw(component, li, d_pad_li)
    draw(component, li, out_strap)
    out_pt = d_nmos.node_li[1].center
    _route_to_wordline(
        component, tech, out=out_pt, track_x=geo.track_x[1], pad_x=geo.pad_x,
        wl_y=wl_y[neg_line], right_edge=geo.width, length=wl_w[neg_line],
        strap_y=strap_y.get(neg_line),
    )

    # --- signal wiring ---------------------------------------------------------------------------
    # x: PA and NX drains -> track; the pad_cx line joins the driver's gate pad, NZ's gate pad and
    # the track
    x_pa, x_nx = drain("pa").center[0], drain("nx").center[0]
    _stub(component, tech, x_pa, drain("pa").center[1], y_x)
    _stub(component, tech, x_nx, drain("nx").center[1], y_x)
    _track_via(component, tech, x_pa, y_x)
    _track_via(component, tech, x_nx, y_x)
    nz_pad = gate_pad("nz")
    px = nz_pad.center[0]
    # (the driver's li pad may have grown to its minimum area off-centre: its contacts are in line)
    if abs(px - pad_box.center[0]) > 1e-6:
        raise AssertionError("NZ's and the driver's gate pads are not in line")
    ys = sorted((nz_pad.center[1], pad_y, y_x))
    y_lo, y_hi = ys[0], ys[-1]
    half = rules.w(LayerRole.M1) / 2
    for y in (nz_pad.center[1], pad_y):
        via_stack(component, tech, LayerRole.LI, LayerRole.M1, (px, y), draw_bottom_pad=False)
    draw(component, m1, Box(px - half, y_lo - half, px + half, y_hi + half))
    _track_via(component, tech, px, y_x)
    _m2_line(component, tech, y_x, x_pa, x_nx)

    # in: PB's gate -> the `in` line, which _route_input started at the read inverter's input
    # (a gate pad's stack keeps clear of the source bus below it, moving up an li riser if need be;
    # the stub's stack, drawn aside, gives its metal1 pad's height)
    stub_half = via_stack(gf.Component(), tech, LayerRole.LI, LayerRole.M1, (0.0, 0.0), draw_bottom_pad=False)[LayerRole.M1].height / 2
    pb_pad = gate_pad("pb")
    pb_x, pb_y = _lift(component, tech, pb_pad, source("pb").center[1], stub_half)
    _stub(component, tech, pb_x, pb_y, y_in)
    _track_via(component, tech, pb_pad.center[0], y_in)
    _m2_line(component, tech, y_in, 0.0, pb_pad.center[0])

    # inb: the inverter's output -> PA's gate
    pa_pad = gate_pad("pa")
    inb_x = tuple(inb_ref.ports["y"].dcenter)[0]
    _stub(component, tech, inb_x, out_inb, y_inb)
    _track_via(component, tech, inb_x, y_inb)
    pa_x, pa_y = _lift(component, tech, pa_pad, source("pa").center[1], stub_half)
    _stub(component, tech, pa_x, pa_y, y_inb)
    _track_via(component, tech, pa_pad.center[0], y_inb)
    _m2_line(component, tech, y_inb, inb_x, pa_pad.center[0])

    # --- supplies ------------------------------------------------------------------------------------
    tw = geo.tap_w
    # level-shifter sources: one metal1 bus per side at the band's source row
    def bus(nodes: list[Box], rail_cx: float) -> None:
        pads = [
            via_stack(component, tech, LayerRole.LI, LayerRole.M1, n.center, draw_bottom_pad=False, axis="x")[
                LayerRole.M1
            ]
            for n in nodes
        ]
        x0 = min(min(p.x0 for p in pads), rail_cx)
        x1 = max(max(p.x1 for p in pads), rail_cx)
        y0 = min(p.y0 for p in pads)
        y1 = max(p.y1 for p in pads)
        draw(component, m1, Box(x0, y0, x1, y1))

    bus([source("pa"), source("pb")], rail_n)
    bus([source("nz"), source("nx")], rail_p)

    # driver sources ride metal2 over the level shifter's stubs: vdd to the far-left standard rail
    # (which also joins the two vdd rails), vneg to the isolated well's rail
    y_sd = d_pmos.node_li[0].center[1]
    for node, rail_cx in ((d_pmos.node_li[0], None), (d_nmos.node_li[0], rail_p)):
        cx = node.center[0]
        via_stack(component, tech, LayerRole.LI, LayerRole.M2, (cx, y_sd), draw_bottom_pad=False, axis="x")
        if rail_cx is None:
            std_vdd = lg.ntap_x + tw / 2
            _m2_line(component, tech, y_sd, std_vdd, cx)
            for rx in (std_vdd, rail_n):
                via_stack(component, tech, LayerRole.M1, LayerRole.M2, (rx, y_sd), axis="x")
        else:
            _m2_line(component, tech, y_sd, cx, rail_cx)
            via_stack(component, tech, LayerRole.M1, LayerRole.M2, (rail_cx, y_sd), axis="x")

    # --- ports for the floorplan --------------------------------------------------------------------------
    add_port(component,
        name=neg_line, center=(geo.width, wl_y[neg_line]), orientation=0,
        width=wl_w[neg_line], layer=tech.layer(LayerRole.POLY), port_type="electrical",
    )
    component.info["width"] = geo.width
    component.info["height"] = H
    component.info["y_in"] = y_in
    fix_min_area(component, tech)
    return component
