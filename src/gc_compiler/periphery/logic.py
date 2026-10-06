"""Generic CMOS logic slices, drawn in the same orientation as the bitcell.

Current flows along +y and gates are horizontal poly stripes, so a logic cell is a
"rotated" standard cell: the PMOS and NMOS columns sit side by side in x and share a gate
stripe, supplies run vertically on metal1 over the well/substrate taps, and cells stack in
y on a fixed pitch. That orientation is what lets a driver abut the array and hand its
output straight to a word-line poly stripe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gdsfactory as gf

from gc_compiler.cache import cached_cell
from gc_compiler.primitives.wells import fix_min_area
from gc_compiler.primitives.contacts import min_pad, place_cuts, via_stack
from gc_compiler.primitives.devices import (
    gate_pad_reach,
    DeviceStack,
    Gate,
    Node,
    build_device_stack,
    gate_pad_size,
    plan_stack,
)
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import DeviceKind, LayerRole, Technology


def _sizing():
    """The default technology's :class:`~gc_compiler.tech.policy.Sizing`."""
    from gc_compiler.tech import default_technology

    return default_technology().sizing


@dataclass(frozen=True)
class LogicConfig:
    """Device sizes of a logic slice."""

    pmos_width: float = field(default_factory=lambda: _sizing().logic["pmos_width"])
    nmos_width: float = field(default_factory=lambda: _sizing().logic["nmos_width"])
    length: float = field(default_factory=lambda: _sizing().logic["length"])
    """Defaults: the default technology's (``tech.sizing.logic``); for another, :meth:`for_tech`."""
    pitch: float | None = None
    """Force the slice pitch; used to pitch-match rotated slices to a column."""

    @classmethod
    def for_tech(cls, tech, **overrides) -> LogicConfig:
        return cls(**{**tech.sizing.logic, **overrides})

    @property
    def name(self) -> str:
        pitch = f"_p{self.pitch:g}" if self.pitch else ""
        return f"p{self.pmos_width:g}_n{self.nmos_width:g}_l{self.length:g}{pitch}"


def predecode_logic(tech=None) -> LogicConfig:
    """Device sizes of a banked macro's shared predecode (``tech.sizing.predecode``; on sky130 four
    times :class:`LogicConfig`'s). Its lines drive the row gates of up to four faces plus a trunk
    across the decoder, against one face in a plain macro; at the plain sizes that doubles the
    decoder delay (sky130 16x20, tt: 1.8 ns against 0.65), and since this predecode is built once
    for two banks, driving it harder costs little area per bit (see the README's banked-macro
    section for the trade-off)."""
    sizing = tech.sizing if tech is not None else _sizing()
    return LogicConfig(**{**sizing.logic, **sizing.predecode})


PREDECODE_LOGIC = predecode_logic()
"""The default technology's :func:`predecode_logic`."""


def driver_logic(logic: LogicConfig, strength: float, tech=None) -> LogicConfig:
    """The word-line drivers' sizes: ``logic``'s device widths times ``strength`` (on the
    technology's ``sizing.size_grid``), same length; ``logic`` itself at strength 1."""
    if strength == 1:
        return logic
    grid = (tech.sizing if tech is not None else _sizing()).size_grid
    return LogicConfig(
        pmos_width=round(round(logic.pmos_width * strength / grid) * grid, 6),
        nmos_width=round(round(logic.nmos_width * strength / grid) * grid, 6),
        length=logic.length,
    )


@dataclass(frozen=True)
class LevelShifterConfig:
    """Device sizes of the level shifter in a negative-rail write-word-line driver.

    Two PMOS input devices at the top rail, latched by two NMOS in an isolated p-well at
    ``vneg``. It is ratioed: the inputs have to overpower the latch NMOS (2.5 V of gate drive at
    ``vneg = -0.7``), so the latch is drawn long and the inputs wide -- the smallest pair that
    switches in every corner (ss/125 C is the worst case) is a 1.5 um input with a 0.8 um latch.
    """

    pmos_width: float = field(default_factory=lambda: _sizing().level_shifter["pmos_width"])
    latch_width: float = field(default_factory=lambda: _sizing().level_shifter["latch_width"])
    latch_length: float = field(default_factory=lambda: _sizing().level_shifter["latch_length"])
    input_length: float = field(default_factory=lambda: _sizing().level_shifter["input_length"])

    @classmethod
    def for_tech(cls, tech, **overrides) -> LevelShifterConfig:
        return cls(**{**tech.sizing.level_shifter, **overrides})

    @property
    def name(self) -> str:
        return f"ls_p{self.pmos_width:g}_n{self.latch_width:g}x{self.latch_length:g}"


@dataclass(frozen=True)
class LogicGeometry:
    """Absolute positions inside a logic slice (origin at the diffusion bottom)."""

    width: float
    pitch_y: float
    diff_top: float
    supply_y: float
    gate_y: float
    out_y: float
    ntap_x: float
    ptap_x: float
    pmos_x0: float
    pmos_x1: float
    nmos_x0: float
    nmos_x1: float
    input_pad_x: float
    nwell_x1: float


def _stacks(cfg: LogicConfig) -> tuple[DeviceStack, DeviceStack]:
    pull_up = DeviceStack(
        kind=DeviceKind.PMOS,
        width=cfg.pmos_width,
        gates=(Gate("a", cfg.length),),
        nodes=(Node("vdd"), Node("y")),
    )
    pull_down = DeviceStack(
        kind=DeviceKind.NMOS,
        width=cfg.nmos_width,
        gates=(Gate("a", cfg.length),),
        nodes=(Node("vss"), Node("y")),
    )
    return pull_up, pull_down


def _implants_apart(tech: Technology, kind: DeviceKind, tap: bool = True) -> float:
    """Space between a ``kind`` device's diffusion and a tap (or, ``tap=False``, a device) of the
    other type: their implants meet without overlapping."""
    rules = tech.rules
    other = DeviceKind.NMOS if kind is DeviceKind.PMOS else DeviceKind.PMOS
    far = rules.enc(tech.tap_implant_role(kind), LayerRole.TAP) if tap else rules.enc(tech.implant_role(other), LayerRole.DIFF)
    return rules.enc(tech.implant_role(kind), LayerRole.DIFF) + far


def logic_geometry(tech: Technology, cfg: LogicConfig) -> LogicGeometry:
    rules = tech.rules
    plan = plan_stack(tech, _stacks(cfg)[0])
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    pad = gate_pad_size(tech)
    # A gate contact must clear the diffusion it runs past.
    pad_clearance = rules.enc(LayerRole.POLY, LayerRole.LICON) + rules.s(LayerRole.POLY)

    ntap_x = rules.enc(LayerRole.NWELL, LayerRole.TAP)
    # the gate's endcap reaches back toward the tap: it keeps poly's spacing to diffusion; and the
    # two implants meet between them without overlapping
    pmos_x0 = ntap_x + tap_w + max(rules.s(LayerRole.DIFF), rules.poly_endcap + rules.poly_to_diff, _implants_apart(tech, DeviceKind.PMOS))
    pmos_x1 = pmos_x0 + cfg.pmos_width
    nwell_x1 = pmos_x1 + rules.enc(LayerRole.NWELL, LayerRole.DIFF)
    nmos_x0 = max(nwell_x1 + rules.nwell_to_ndiff_space, pmos_x1 + _implants_apart(tech, DeviceKind.PMOS, tap=False))
    nmos_x1 = nmos_x0 + cfg.nmos_width
    # the inputs' via stacks put a metal pad on each gate pad, beside the ones a drain's stack
    # puts on the pull-down (a cell with parallel drains ties them over metal2)
    stack = tech.stack
    m1 = stack.routing[1]
    cuts = (stack.cut_between(stack.routing[0], m1), stack.cut_above(m1))
    input_m1 = max(rules.via_pad(m1, c) for c in cuts)
    input_pad_x = max(
        nmos_x1 + pad_clearance + pad / 2, (nmos_x0 + nmos_x1) / 2 + input_m1 + rules.s(m1)
    )
    rail_m1 = rules.via_pad(m1, cuts[0])
    ptap_x = max(
        input_pad_x + max(pad / 2 + pad_clearance, input_m1 / 2 + rules.s(m1) + rail_m1 / 2 - tap_w / 2),
        nmos_x1 + _implants_apart(tech, DeviceKind.NMOS),
    )
    width = ptap_x + tap_w + rules.enc(LayerRole.NWELL, LayerRole.TAP)

    minimum_pitch = plan.y1 - plan.y0 + rules.s(LayerRole.DIFF)
    if cfg.pitch is not None and cfg.pitch < minimum_pitch - rules.grid / 2:
        raise ValueError(f"slice pitch {cfg.pitch} is below the minimum {minimum_pitch}")

    return LogicGeometry(
        width=rules.snap(width),
        pitch_y=rules.snap(cfg.pitch or minimum_pitch),
        diff_top=rules.snap(plan.y1),
        supply_y=rules.snap(plan.node_y[0]),
        gate_y=rules.snap(plan.gate_y[0]),
        out_y=rules.snap(plan.node_y[1]),
        ntap_x=rules.snap(ntap_x),
        ptap_x=rules.snap(ptap_x),
        pmos_x0=rules.snap(pmos_x0),
        pmos_x1=rules.snap(pmos_x1),
        nmos_x0=rules.snap(nmos_x0),
        nmos_x1=rules.snap(nmos_x1),
        input_pad_x=rules.snap(input_pad_x),
        nwell_x1=rules.snap(nwell_x1),
    )


def inverter(
    tech: Technology,
    cfg: LogicConfig | None = None,
    name: str | None = None,
) -> gf.Component:
    """CMOS inverter slice with vertical supply rails over its own taps."""
    cfg = cfg or LogicConfig.for_tech(tech)
    activate(tech)
    return cached_cell(
        name or f"gc_inv_{cfg.name}", lambda n: _build_slice(tech, cfg, n, pulldown=True), tech
    )


def pmos_switch(
    tech: Technology,
    cfg: LogicConfig | None = None,
    name: str | None = None,
) -> gf.Component:
    """PMOS pull-up slice (precharge device), same footprint as the inverter."""
    cfg = cfg or LogicConfig.for_tech(tech)
    activate(tech)
    return cached_cell(
        name or f"gc_pup_{cfg.name}", lambda n: _build_slice(tech, cfg, n, pulldown=False), tech
    )


def _build_slice(tech: Technology, cfg: LogicConfig, name: str, pulldown: bool) -> gf.Component:
    rules = tech.rules
    geo = logic_geometry(tech, cfg)
    pull_up, pull_down = _stacks(cfg)

    component = gf.Component(name)
    # Stripes span the full pitch so that stacked slices merge into continuous shapes.
    margin = (geo.pitch_y - geo.diff_top) / 2
    top = geo.diff_top + margin
    bottom = -margin
    draw(component, tech.layer(LayerRole.NWELL), Box(0.0, bottom, geo.nwell_x1, top))

    up = build_device_stack(component, tech, pull_up, (geo.pmos_x0, 0.0), draw_implant=False)
    stacks = [(pull_up, geo.pmos_x0, geo.pmos_x1)]
    down = None
    if pulldown:
        down = build_device_stack(
            component, tech, pull_down, (geo.nmos_x0, 0.0), draw_implant=False
        )
        stacks.append((pull_down, geo.nmos_x0, geo.nmos_x1))

    # Implants run the full cell height so that stacked slices merge into stripes.
    for stack, x0, x1 in stacks:
        implant = tech.implant_role(stack.kind)
        enc = rules.enc(implant, LayerRole.DIFF)
        draw(component, tech.layer(implant), Box(x0 - enc, bottom, x1 + enc, top))

    _draw_input(component, tech, cfg, geo)

    # Output: both drains sit on the same row, so a straight li strap connects them.
    out = up.node_li[1] if down is None else Box.hull([up.node_li[1], down.node_li[1]])
    draw(component, tech.layer(LayerRole.LI), out)

    _draw_tap_rail(component, tech, geo, geo.ntap_x, DeviceKind.PMOS, "vdd", bottom, top)
    _draw_tap_rail(component, tech, geo, geo.ptap_x, DeviceKind.NMOS, "vss", bottom, top)
    _connect_supply(component, tech, geo, up.node_li[0], geo.ntap_x)
    if down is not None:
        _connect_supply(component, tech, geo, down.node_li[0], geo.ptap_x)

    li_label = tech.label_layer(LayerRole.LI)
    add_port(component,
        name="a",
        center=(geo.input_pad_x, geo.gate_y),
        orientation=0,
        width=rules.w(LayerRole.LI),
        layer=tech.layer(LayerRole.LI),
        port_type="electrical",
    )
    add_port(component,
        name="y",
        center=out.center,
        orientation=0,
        width=out.height,
        layer=tech.layer(LayerRole.LI),
        port_type="electrical",
    )
    if li_label is not None:
        component.add_label(text="a", position=(geo.input_pad_x, geo.gate_y), layer=li_label)
        component.add_label(text="y", position=out.center, layer=li_label)
    component.info["pitch_y"] = geo.pitch_y
    component.info["width"] = geo.width
    fix_min_area(component, tech)
    return component


def _pad_offset(tech: Technology, kind: DeviceKind) -> float:
    """From a device's diffusion edge to the centre of a gate pad beside it: past the endcap and the
    contact's spacing to diffusion (``gate_pad_reach``), and with the pad's li clear of the device's
    own source/drain contact li, which reaches about as far as the diffusion."""
    rules = tech.rules
    pad = gate_pad_size(tech)
    cut, enc_li = rules.cut(LayerRole.LICON), rules.enc(LayerRole.LI, LayerRole.LICON)
    node_li = max(0.0, enc_li - rules.enc(LayerRole.DIFF, LayerRole.LICON))
    return rules.snap_up(max(gate_pad_reach(tech, kind) + pad / 2, node_li + rules.s(LayerRole.LI) + (cut + 2 * enc_li) / 2))


@dataclass(frozen=True)
class NandSliceGeometry:
    """A two-input NAND slice at the inverter slice's pitch (``y = NAND(a, b)``), its four devices
    side by side along x: ``b``'s PMOS, ``a``'s PMOS, ``a``'s NMOS, ``b``'s NMOS (the series pull-down,
    each NMOS twice the logic width). ``a`` is one gate across its PMOS and NMOS, contacted between
    them; ``b`` is two gates, contacted at the slice's two ends (past the n-tap, before the p-tap). In
    a column, ``a`` is ``din`` (one contact, below the output) and ``b`` is ``rbl`` (whose track runs
    past the whole slice)."""

    width: float
    pitch_y: float
    diff_top: float
    supply_y: float
    gate_y: float
    out_y: float
    ntap_x: float
    ptap_x: float
    pb_x0: float
    pa_x0: float
    na_x0: float
    nb_x0: float
    nwell_x1: float
    pmos_width: float
    """``a``'s PMOS; ``b``'s is twice as wide (see :func:`nand_slice_geometry`)."""
    nmos_width: float


def nand_slice_geometry(tech: Technology, cfg: LogicConfig) -> NandSliceGeometry:
    rules = tech.rules
    wp, wn = cfg.pmos_width, rules.snap(2 * cfg.nmos_width)
    plan = plan_stack(tech, _stacks(cfg)[0])
    tap_w = rules.tap_width()
    pad = gate_pad_size(tech)
    cut, enc_li, w_li = rules.cut(LayerRole.LICON), rules.enc(LayerRole.LI, LayerRole.LICON), rules.w(LayerRole.LI)
    end, poly_space = rules.poly_endcap, rules.s(LayerRole.POLY)
    off_p, off_n = _pad_offset(tech, DeviceKind.PMOS), _pad_offset(tech, DeviceKind.NMOS)
    stack = tech.stack
    m1 = stack.routing[1]
    cuts = (stack.cut_between(stack.routing[0], m1), stack.cut_above(m1))
    input_m1 = max(rules.via_pad(m1, c) for c in cuts)
    rail_m1 = rules.via_pad(m1, cuts[0])

    ntap_x = rules.enc(LayerRole.NWELL, LayerRole.TAP)
    # b's left pad clears the n-tap (its poly and contact, and its li the tap's li, as wide as the
    # rail's via pad where the rail is contacted). In a column the pad's li runs on sideways to its
    # track as tall as a via's li pad (periphery.column._lift_input): that clears the tap too.
    tap_li = max(w_li, cut + 2 * enc_li, rules.via_pad(LayerRole.LI, cuts[0]))
    pad_li = max(cut + 2 * enc_li, rules.via_pad(LayerRole.LI, cuts[0]))
    # ... and the metal1 pad of the via its track takes it up through, grown to metal1's minimum
    # area, keeps metal1's spacing from the tap's rail
    grown_m1 = max(input_m1, rules.area(m1) / input_m1) if input_m1 else 0.0
    rail_clear = rail_m1 / 2 + rules.s(m1) + grown_m1 / 2 + 2 * rules.grid  # (fix_min_area grows in two-step units)
    pad_left = max(
        ntap_x + tap_w + max(rules.poly_to_diff, rules.poly_contact_to_diff - (pad - cut) / 2),
        ntap_x + tap_w / 2 + tap_li / 2 + rules.s(LayerRole.LI) + pad_li / 2 - pad / 2,
        ntap_x + tap_w / 2 + rail_clear - pad / 2,
    )
    pb_x0 = max(
        pad_left + pad / 2 + off_p,
        ntap_x + tap_w + max(rules.s(LayerRole.DIFF), _implants_apart(tech, DeviceKind.PMOS)),
    )
    # b's PMOS is twice the logic width: b is a bit line a self-timed read leaves part-way down (a
    # stored '1' reads with rbl near vdd/2), and the stronger pull-up pulls the output to the rail
    # against the half-on series pull-down. a's PMOS: a separate gate beside it, the two endcaps
    # poly's spacing apart
    pb_x1 = pb_x0 + 2 * wp
    pa_x0 = pb_x1 + max(rules.s(LayerRole.DIFF), 2 * end + poly_space)
    pa_x1 = pa_x0 + wp
    nwell_x1 = pa_x1 + rules.enc(LayerRole.NWELL, LayerRole.DIFF)
    # a's pad between its two devices; the NMOS past it, the well edge and the implants
    na_x0 = max(
        pa_x1 + off_p + off_n,
        nwell_x1 + rules.nwell_to_ndiff_space,
        pa_x1 + _implants_apart(tech, DeviceKind.PMOS, tap=False),
    )
    na_x1 = na_x0 + wn
    nb_x0 = na_x1 + max(rules.s(LayerRole.DIFF), 2 * end + poly_space)
    nb_x1 = nb_x0 + wn
    # b's right pad, then the p-tap (as the inverter's input pad and tap)
    pad_right_x = max(nb_x1 + off_n, (nb_x0 + nb_x1) / 2 + input_m1 + rules.s(m1))
    pad_clearance = rules.enc(LayerRole.POLY, LayerRole.LICON) + rules.s(LayerRole.POLY)
    ptap_x = max(
        pad_right_x + max(pad / 2 + pad_clearance, input_m1 / 2 + rules.s(m1) + rail_m1 / 2 - tap_w / 2),
        pad_right_x + pad_li / 2 + rules.s(LayerRole.LI) + tap_li / 2 - tap_w / 2,
        pad_right_x + rail_clear - tap_w / 2,
        nb_x1 + _implants_apart(tech, DeviceKind.NMOS),
    )
    width = ptap_x + tap_w + rules.enc(LayerRole.NWELL, LayerRole.TAP)
    minimum_pitch = plan.y1 - plan.y0 + rules.s(LayerRole.DIFF)
    if cfg.pitch is not None and cfg.pitch < minimum_pitch - rules.grid / 2:
        raise ValueError(f"slice pitch {cfg.pitch} is below the minimum {minimum_pitch}")
    s = rules.snap
    return NandSliceGeometry(
        width=s(width), pitch_y=s(cfg.pitch or minimum_pitch), diff_top=s(plan.y1), supply_y=s(plan.node_y[0]),
        gate_y=s(plan.gate_y[0]), out_y=s(plan.node_y[1]), ntap_x=s(ntap_x), ptap_x=s(ptap_x), pb_x0=s(pb_x0),
        pa_x0=s(pa_x0), na_x0=s(na_x0), nb_x0=s(nb_x0), nwell_x1=s(nwell_x1), pmos_width=wp, nmos_width=wn,
    )


def nand_slice(tech: Technology, cfg: LogicConfig | None = None, name: str | None = None) -> gf.Component:
    """A two-input NAND slice (ports ``a``; ``b0``/``b1``: input ``b``'s two contacts; ``y``), pitch-
    matched like the inverter slice: a column's write driver with its write-back input."""
    cfg = cfg or LogicConfig.for_tech(tech)
    activate(tech)
    return cached_cell(name or f"gc_nand_{cfg.name}", lambda n: _build_nand_slice(tech, cfg, n), tech)


def _build_nand_slice(tech: Technology, cfg: LogicConfig, name: str) -> gf.Component:
    rules = tech.rules
    geo = nand_slice_geometry(tech, cfg)
    li = tech.layer(LayerRole.LI)
    component = gf.Component(name)
    margin = (geo.pitch_y - geo.diff_top) / 2
    top, bottom = geo.diff_top + margin, -margin
    draw(component, tech.layer(LayerRole.NWELL), Box(0.0, bottom, geo.nwell_x1, top))

    def stack(kind, width, net, nodes):
        # (unpadded, as the inverter's: the pads are drawn beside the stacks, which keeps every
        # stack the inverter's height)
        return DeviceStack(kind=kind, width=width, gates=(Gate(net, cfg.length),), nodes=tuple(Node(n) for n in nodes))

    pb = build_device_stack(component, tech, stack(DeviceKind.PMOS, rules.snap(2 * geo.pmos_width), "b", ("vdd", "y")), (geo.pb_x0, 0.0), draw_implant=False)
    pa = build_device_stack(component, tech, stack(DeviceKind.PMOS, geo.pmos_width, "a", ("vdd", "y")), (geo.pa_x0, 0.0), draw_implant=False)
    na = build_device_stack(component, tech, stack(DeviceKind.NMOS, geo.nmos_width, "a", ("mid", "y")), (geo.na_x0, 0.0), draw_implant=False)
    # (b's NMOS has its source on the bottom row: both supply stubs then run on the bottom row and
    # the output's jumper on the top one, so in a column neither crosses the other input's via)
    nb = build_device_stack(component, tech, stack(DeviceKind.NMOS, geo.nmos_width, "b", ("vss", "mid")), (geo.nb_x0, 0.0), draw_implant=False)
    off_p, off_n = _pad_offset(tech, DeviceKind.PMOS), _pad_offset(tech, DeviceKind.NMOS)
    # b: a pad left of its PMOS and one right of its NMOS; a: one pad right of its PMOS, its poly on
    # across to its NMOS
    b0 = _side_pad(component, tech, pb.gate_poly[0], pb.diff.x0 - off_p, cfg.length)
    b1 = _side_pad(component, tech, nb.gate_poly[0], nb.diff.x1 + off_n, cfg.length)
    a = _side_pad(component, tech, pa.gate_poly[0], pa.diff.x1 + off_p, cfg.length)
    na_poly = na.gate_poly[0]
    draw(component, tech.layer(LayerRole.POLY), Box(a.x0, na_poly.y0, na_poly.x0 + rules.grid, na_poly.y1))

    for kind, x0, x1 in ((DeviceKind.PMOS, geo.pb_x0, geo.pa_x0 + geo.pmos_width), (DeviceKind.NMOS, geo.na_x0, geo.nb_x0 + geo.nmos_width)):
        implant = tech.implant_role(kind)
        enc = rules.enc(implant, LayerRole.DIFF)
        draw(component, tech.layer(implant), Box(x0 - enc, bottom, x1 + enc, top))

    # y: the PMOS drains on li (nothing between them), on to a's NMOS drain over metal1 on the top
    # row (a's pad lies between). The series node runs from a's NMOS's bottom contact to b's NMOS's
    # top one over metal1, rising in the gap between the two. The PMOS sources go to vdd and b's
    # NMOS source to vss on metal1 along the bottom row.
    m1 = tech.layer(LayerRole.M1)
    out = Box.hull([pb.node_li[1], pa.node_li[1]])
    draw(component, li, out)

    def contact(node: Box) -> Box:
        return via_stack(component, tech, LayerRole.LI, LayerRole.M1, node.center, draw_bottom_pad=False, axis="x")[LayerRole.M1]

    draw(component, m1, Box.hull([contact(pa.node_li[1]), contact(na.node_li[1])]))
    lo, hi = contact(na.node_li[0]), contact(nb.node_li[1])
    gap_x = (na.diff.x1 + nb.diff.x0) / 2
    half = rules.w(LayerRole.M1) / 2
    draw(component, m1, Box(lo.x0, lo.y0, gap_x + half, lo.y1))  # along the bottom row into the gap
    draw(component, m1, Box(gap_x - half, lo.y0, gap_x + half, hi.y1))  # up the gap
    draw(component, m1, Box(gap_x - half, hi.y0, hi.x1, hi.y1))  # along the top row to b's NMOS
    draw(component, li, Box.hull([pb.node_li[0], pa.node_li[0]]))
    _draw_tap_rail(component, tech, geo, geo.ntap_x, DeviceKind.PMOS, "vdd", bottom, top)
    _draw_tap_rail(component, tech, geo, geo.ptap_x, DeviceKind.NMOS, "vss", bottom, top)
    _connect_supply(component, tech, geo, pb.node_li[0], geo.ntap_x)
    _connect_supply(component, tech, geo, nb.node_li[0], geo.ptap_x)

    label = tech.label_layer(LayerRole.LI)
    for port, box in (("a", a), ("b0", b0), ("b1", b1), ("y", out)):
        add_port(component, name=port, center=box.center, orientation=0, width=box.height, layer=li, port_type="electrical")
        if label is not None:
            component.add_label(text=port.rstrip("01"), position=box.center, layer=label)
    component.info["pitch_y"] = geo.pitch_y
    component.info["width"] = geo.width
    fix_min_area(component, tech)
    return component


def _side_pad(component, tech: Technology, gate: Box, pad_cx: float, length: float) -> Box:
    """A contacted poly pad centred at ``pad_cx`` on ``gate``'s line, the poly run to it from the
    gate; returns the pad's li."""
    rules = tech.rules
    size = gate_pad_size(tech)
    cy = gate.center[1]
    pad = Box.centered(pad_cx, cy, size, size)
    poly = tech.layer(LayerRole.POLY)
    draw(component, poly, pad)
    draw(component, poly, Box(min(gate.x0, pad.x0 + rules.grid), cy - length / 2, max(gate.x1, pad.x1 - rules.grid), cy + length / 2))
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.POLY, pad, axis="x")
    tech.decorate_gate_contact(component, hull.as_tuple())
    li_h = max(hull.height, rules.w(LayerRole.LI))
    enc = rules.enc(LayerRole.LI, LayerRole.LICON)
    li = Box(hull.x0 - enc, cy - li_h / 2, hull.x1 + enc, cy + li_h / 2)
    draw(component, tech.layer(LayerRole.LI), li)  # (made up to its minimum area by fix_min_area)
    return li


def _draw_input(component, tech: Technology, cfg: LogicConfig, geo: LogicGeometry):
    """Gate stripe across both columns, contacted on a pad past the pull-down column."""
    rules = tech.rules
    pad = gate_pad_size(tech)
    gate = Box(
        geo.pmos_x0 - rules.poly_endcap,
        geo.gate_y - cfg.length / 2,
        geo.nmos_x1 + rules.poly_endcap,
        geo.gate_y + cfg.length / 2,
    )
    draw(component, tech.layer(LayerRole.POLY), gate)
    pad_box = Box.centered(geo.input_pad_x, geo.gate_y, pad, pad)
    draw(component, tech.layer(LayerRole.POLY), pad_box)
    draw(
        component,
        tech.layer(LayerRole.POLY),
        Box(gate.x1, gate.y0, pad_box.x0 + rules.grid, gate.y1),
    )
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.POLY, pad_box, axis="x")
    tech.decorate_gate_contact(component, hull.as_tuple())
    li_h = max(hull.height, rules.w(LayerRole.LI))
    enc = rules.enc(LayerRole.LI, LayerRole.LICON)
    li = Box(hull.x0 - enc, geo.gate_y - li_h / 2, hull.x1 + enc, geo.gate_y + li_h / 2)
    draw(component, tech.layer(LayerRole.LI), li)  # (made up to its minimum area by fix_min_area)


def _draw_tap_rail(
    component,
    tech: Technology,
    geo: LogicGeometry,
    x0: float,
    body_of: DeviceKind,
    net: str,
    bottom: float,
    top: float,
) -> None:
    """Full-height tap strip with a vertical metal1 supply rail on top of it."""
    rules = tech.rules
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    strip = Box(x0, bottom, x0 + tap_w, top)
    draw(component, tech.layer(LayerRole.TAP), strip)

    implant = tech.tap_implant_role(body_of)
    draw(component, tech.layer(implant), strip.grow(rules.enc(implant, LayerRole.TAP), 0.0))

    cut = rules.cut(LayerRole.LICON)
    region = Box(strip.x0, geo.supply_y - cut / 2, strip.x1, geo.supply_y + cut / 2)
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.TAP, region, axis="x")
    li_w = max(hull.width, rules.w(LayerRole.LI))
    cx = strip.center[0]
    draw(component, tech.layer(LayerRole.LI), Box(cx - li_w / 2, bottom, cx + li_w / 2, top))

    pads = via_stack(
        component, tech, LayerRole.LI, LayerRole.M1, (cx, geo.supply_y), draw_bottom_pad=False
    )
    rail = pads[LayerRole.M1]
    draw(component, tech.layer(LayerRole.M1), Box(rail.x0, bottom, rail.x1, top))
    add_port(component,
        name=net,
        center=(cx, geo.supply_y),
        orientation=270,
        width=rail.width,
        layer=tech.layer(LayerRole.M1),
        port_type="electrical",
    )
    label = tech.label_layer(LayerRole.M1)
    if label is not None:
        component.add_label(text=net, position=(cx, geo.supply_y), layer=label)


def _connect_supply(component, tech: Technology, geo: LogicGeometry, node: Box, rail_x: float):
    """Metal1 stub from a source node to its supply rail."""
    rules = tech.rules
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    # Elongated in x (relaxed enclosure in y) so a long stub stays clear of anything
    # routed just above or below it, such as a neighbouring gate's input jumper.
    pads = via_stack(
        component, tech, LayerRole.LI, LayerRole.M1, node.center, draw_bottom_pad=False, axis="x"
    )
    pad = pads[LayerRole.M1]
    x0 = min(pad.x0, rail_x)
    x1 = max(pad.x1, rail_x + tap_w)
    draw(component, tech.layer(LayerRole.M1), Box(x0, pad.y0, x1, pad.y1))


@dataclass(frozen=True)
class Nand2Geometry:
    """Absolute positions inside a 2-input NAND slice (origin at the diffusion bottom)."""

    width: float
    pitch_y: float
    diff_top: float
    supply_y: float
    nwell_x1: float
    ntap_x: float
    ptap_x: float
    pmos_a_x0: float
    pmos_a_x1: float
    pmos_b_x0: float
    pmos_b_x1: float
    nmos_x0: float
    nmos_x1: float
    channel_gate_y: float
    channel_out_y: float
    """Metal2 channels, above every gate/node, that carry ``a``'s jumpers clear of ``b``."""


def _nand2_stacks(cfg: LogicConfig) -> tuple[DeviceStack, DeviceStack, DeviceStack]:
    """PMOS-A, PMOS-B (parallel pull-up) and the series NMOS pull-down.

    ``b`` is closest to the NMOS column on both the pull-up row and the pull-down's own
    gate order, so its jumpers are short local jogs; ``a`` sits behind ``b`` and is wired
    over a metal2 channel above the whole cell to avoid crossing ``b``'s own contacts.
    """
    pmos_a = DeviceStack(
        kind=DeviceKind.PMOS,
        width=cfg.pmos_width,
        gates=(Gate("a", cfg.length, pad="right"),),
        nodes=(Node("vdd"), Node("y")),
    )
    pmos_b = DeviceStack(
        kind=DeviceKind.PMOS,
        width=cfg.pmos_width,
        gates=(Gate("b", cfg.length, pad="right"),),
        nodes=(Node("vdd"), Node("y")),
    )
    pulldown = DeviceStack(
        kind=DeviceKind.NMOS,
        width=cfg.nmos_width,
        gates=(Gate("b", cfg.length, pad="left"), Gate("a", cfg.length, pad="left")),
        nodes=(Node("vss"), Node("mid"), Node("y")),
    )
    return pmos_a, pmos_b, pulldown


def nand2_geometry(tech: Technology, cfg: LogicConfig) -> Nand2Geometry:
    """Two parallel PMOS pull-ups (one column each) and one series NMOS pull-down.

    Poly can't cross an unrelated diffusion without gating it, so each gate keeps its own
    local licon/li contact and inputs are wired between pads on metal2, which is free to
    cross over anything below it.
    """
    rules = tech.rules
    pmos_a, pmos_b, pulldown = _nand2_stacks(cfg)
    down_plan = plan_stack(tech, pulldown)
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    pad = gate_pad_size(tech)
    pad_clearance = rules.enc(LayerRole.POLY, LayerRole.LICON) + rules.s(LayerRole.POLY)

    ntap_x = rules.enc(LayerRole.NWELL, LayerRole.TAP)
    pmos_a_x0 = ntap_x + tap_w + max(
        rules.s(LayerRole.DIFF), rules.poly_endcap + rules.poly_to_diff, _implants_apart(tech, DeviceKind.PMOS)
    )
    pmos_a_x1 = pmos_a_x0 + cfg.pmos_width
    # A's right-side pad must also clear B's own gate poly, which overhangs B's diffusion
    # by poly_endcap regardless of which side its own pad is on.
    pmos_b_x0 = pmos_a_x1 + pad_clearance + pad + rules.poly_endcap + rules.s(LayerRole.POLY)
    pmos_b_x1 = pmos_b_x0 + cfg.pmos_width
    nwell_x1 = pmos_b_x1 + rules.enc(LayerRole.NWELL, LayerRole.DIFF)
    pad_space = max(
        rules.s(LayerRole.POLY), rules.s(LayerRole.POLY_CUT_MARKER) if tech.has(LayerRole.POLY_CUT_MARKER) else 0.0
    )
    # B's gate pad (right of B) and the pull-down's first (left of it) face each other: each its
    # reach from its own diffusion, and their spacing
    reach_p, reach_n = gate_pad_reach(tech, DeviceKind.PMOS), gate_pad_reach(tech, DeviceKind.NMOS)
    nmos_x0 = pmos_b_x1 + max(
        pad_clearance + pad + pad_space + rules.nwell_to_ndiff_space,
        reach_p + pad + pad_space + pad + reach_n,
        _implants_apart(tech, DeviceKind.PMOS, tap=False),
    )
    nmos_x1 = nmos_x0 + cfg.nmos_width
    ptap_x = nmos_x1 + max(pad_clearance + pad + rules.s(LayerRole.DIFF), _implants_apart(tech, DeviceKind.NMOS))
    width = ptap_x + tap_w + rules.enc(LayerRole.NWELL, LayerRole.TAP)

    a_plan = plan_stack(tech, pmos_a)
    stack_top = max(a_plan.y1, down_plan.y1)
    # Two m2 channels stacked above the whole cell, clear of every gate/node pad, so `a`'s
    # jumpers never have to cross `b`'s contacts on the way to the pull-down.
    m2_pitch = rules.w(LayerRole.M2) + rules.s(LayerRole.M2)
    channel_gate_y = stack_top + rules.s(LayerRole.M2) + rules.w(LayerRole.M2) / 2
    channel_out_y = channel_gate_y + m2_pitch
    diff_top = channel_out_y + rules.w(LayerRole.M2) / 2 + rules.s(LayerRole.M2)
    minimum_pitch = diff_top - min(a_plan.y0, down_plan.y0) + rules.s(LayerRole.DIFF)
    if cfg.pitch is not None and cfg.pitch < minimum_pitch - rules.grid / 2:
        raise ValueError(f"slice pitch {cfg.pitch} is below the minimum {minimum_pitch}")

    return Nand2Geometry(
        width=rules.snap(width),
        pitch_y=rules.snap(cfg.pitch or minimum_pitch),
        diff_top=rules.snap(diff_top),
        supply_y=rules.snap(down_plan.node_y[0]),
        nwell_x1=rules.snap(nwell_x1),
        ntap_x=rules.snap(ntap_x),
        ptap_x=rules.snap(ptap_x),
        pmos_a_x0=rules.snap(pmos_a_x0),
        pmos_a_x1=rules.snap(pmos_a_x1),
        pmos_b_x0=rules.snap(pmos_b_x0),
        pmos_b_x1=rules.snap(pmos_b_x1),
        nmos_x0=rules.snap(nmos_x0),
        nmos_x1=rules.snap(nmos_x1),
        channel_gate_y=rules.snap(channel_gate_y),
        channel_out_y=rules.snap(channel_out_y),
    )


def nand2(
    tech: Technology,
    cfg: LogicConfig | None = None,
    name: str | None = None,
) -> gf.Component:
    """2-input NAND slice: parallel PMOS pull-up, series NMOS pull-down."""
    cfg = cfg or LogicConfig.for_tech(tech)
    activate(tech)
    return cached_cell(name or f"gc_nand2_{cfg.name}", lambda n: _build_nand2(tech, cfg, n), tech)


def _build_nand2(tech: Technology, cfg: LogicConfig, name: str) -> gf.Component:
    rules = tech.rules
    geo = nand2_geometry(tech, cfg)
    pmos_a, pmos_b, pulldown = _nand2_stacks(cfg)

    component = gf.Component(name)
    margin = (geo.pitch_y - geo.diff_top) / 2
    top = geo.diff_top + margin
    bottom = -margin
    draw(component, tech.layer(LayerRole.NWELL), Box(0.0, bottom, geo.nwell_x1, top))

    a = build_device_stack(component, tech, pmos_a, (geo.pmos_a_x0, 0.0), draw_implant=False)
    b = build_device_stack(component, tech, pmos_b, (geo.pmos_b_x0, 0.0), draw_implant=False)
    down = build_device_stack(component, tech, pulldown, (geo.nmos_x0, 0.0), draw_implant=False)

    for implant, x0, x1 in (
        (tech.implant_role(pmos_a.kind), geo.pmos_a_x0, geo.pmos_b_x1),
        (tech.implant_role(pulldown.kind), geo.nmos_x0, geo.nmos_x1),
    ):
        enc = rules.enc(implant, LayerRole.DIFF)
        draw(component, tech.layer(implant), Box(x0 - enc, bottom, x1 + enc, top))

    # Output: both pull-up drains and the pull-down's top node share the "y" net. B sits
    # right next to the pull-down, so its jumper is a direct jog; A's would otherwise cross
    # straight through B's own contacts, so it goes over a channel instead.
    _connect(component, tech, b.node_li[1], down.node_li[2])
    _connect_over(component, tech, a.node_li[1], down.node_li[2], geo.channel_out_y)

    _draw_tap_rail(component, tech, geo, geo.ntap_x, DeviceKind.PMOS, "vdd", bottom, top)
    _draw_tap_rail(component, tech, geo, geo.ptap_x, DeviceKind.NMOS, "vss", bottom, top)
    _connect_supply(component, tech, geo, a.node_li[0], geo.ntap_x)
    _connect_supply(component, tech, geo, b.node_li[0], geo.ntap_x)
    _connect_supply(component, tech, geo, down.node_li[0], geo.ptap_x)

    # Inputs: gate_pad_li[0] is the pull-down's bottom (b) gate, [1] its top (a) gate.
    _connect(component, tech, b.gate_pad_li[0], down.gate_pad_li[0], avoid_y=geo.supply_y)
    _connect_over(component, tech, a.gate_pad_li[0], down.gate_pad_li[1], geo.channel_gate_y, avoid_y=geo.supply_y)

    li_label = tech.label_layer(LayerRole.LI)
    for net, pad in (("b", down.gate_pad_li[0]), ("a", down.gate_pad_li[1])):
        add_port(component,
            name=net,
            center=pad.center,
            orientation=0,
            width=rules.w(LayerRole.LI),
            layer=tech.layer(LayerRole.LI),
            port_type="electrical",
        )
        if li_label is not None:
            component.add_label(text=net, position=pad.center, layer=li_label)
    add_port(component,
        name="y",
        center=b.node_li[1].center,
        orientation=0,
        width=b.node_li[1].height,
        layer=tech.layer(LayerRole.LI),
        port_type="electrical",
    )
    if li_label is not None:
        component.add_label(text="y", position=b.node_li[1].center, layer=li_label)
    component.info["pitch_y"] = geo.pitch_y
    component.info["width"] = geo.width
    fix_min_area(component, tech)
    return component


def _lift(
    component: gf.Component, tech: Technology, pad: Box, avoid_y: float | None, pad_half: float | None = None
) -> tuple[float, float]:
    """Where a via stack on the li ``pad`` goes: on it, or -- if its metal1 pad would come within
    spacing of a horizontal supply stub centred at ``avoid_y`` -- moved away from the stub on an li
    riser from the pad. ``pad_half``: the stack's metal1 pad's half height (default: its short,
    relaxed side faces the stub)."""
    x, y = pad.center
    if avoid_y is None:
        return x, y
    rules, stack = tech.rules, tech.stack
    m1 = stack.routing[1]
    stub_half = rules.via_pad(m1, stack.cut_between(stack.routing[0], m1), relaxed=True) / 2
    if pad_half is None:
        pad_half = max(rules.via_pad(m1, c, relaxed=True) for c in (stack.cut_between(stack.routing[0], m1), stack.cut_above(m1))) / 2
    need = stub_half + rules.s(m1) + pad_half
    if abs(y - avoid_y) >= need - 1e-9:
        return x, y
    y1 = rules.snap_up(avoid_y + need) if y >= avoid_y else rules.snap_down(avoid_y - need)
    half = rules.w(stack.local_interconnect) / 2
    draw(component, tech.layer(stack.local_interconnect), Box(x - half, min(y, y1) - half, x + half, max(y, y1) + half))
    return x, y1


def _connect(component: gf.Component, tech: Technology, a: Box, b: Box, avoid_y: float | None = None) -> None:
    """Route two li pads together on metal2, which is free to cross anything below it -- or, where
    they sit side by side too close for two via stacks (a cut and its spacing), join them on li."""
    rules = tech.rules
    (ax, ay), (bx, by) = a.center, b.center
    stack = tech.stack
    li = stack.local_interconnect
    cuts = [stack.cut_between(lo, hi) for lo, hi in zip(stack.routing, stack.routing[1:])][:2]
    # the two stacks' cuts, and their pads on the layer between (the top layer joins them anyway)
    middle = stack.routing[1]
    middle_pad = max(rules.via_pad(middle, c) for c in cuts)
    need = max(
        max(rules.cut(c) + rules.cut_space[c] for c in cuts),
        middle_pad + rules.s(middle), rules.area(middle) / middle_pad + rules.s(middle),
    )
    if abs(ay - by) < 1e-9 and abs(ax - bx) < need - 1e-9:
        draw(component, tech.layer(li), Box.hull([a, b]))
        return
    # Elongated in x (relaxed enclosure in y): these hops sit right above a horizontal
    # supply stub, so a smaller y footprint keeps them from needing m1 spacing there too (and
    # where even that is too close, they move away from it: _lift)
    (ax, ay), (bx, by) = _lift(component, tech, a, avoid_y), _lift(component, tech, b, avoid_y)
    pad_a = via_stack(component, tech, LayerRole.LI, LayerRole.M2, (ax, ay), draw_bottom_pad=False, axis="x")
    pad_b = via_stack(component, tech, LayerRole.LI, LayerRole.M2, (bx, by), draw_bottom_pad=False, axis="x")
    m2 = tech.layer(LayerRole.M2)
    half = rules.w(LayerRole.M2) / 2
    draw(component, m2, Box(min(ax, bx) - half, ay - half, max(ax, bx) + half, ay + half))
    draw(component, m2, Box(bx - half, min(ay, by) - half, bx + half, max(ay, by) + half))
    draw(component, m2, pad_a[LayerRole.M2])
    draw(component, m2, pad_b[LayerRole.M2])


def _connect_over(
    component: gf.Component, tech: Technology, a: Box, b: Box, channel_y: float, avoid_y: float | None = None
) -> None:
    """Route two li pads via a metal2 channel above everything else in the cell.

    Used when a direct jog between ``a`` and ``b`` would otherwise cross a third pad that
    sits on the same net level in between them.
    """
    rules = tech.rules
    (ax, ay), (bx, by) = _lift(component, tech, a, avoid_y), _lift(component, tech, b, avoid_y)
    pad_a = via_stack(component, tech, LayerRole.LI, LayerRole.M2, (ax, ay), draw_bottom_pad=False, axis="x")
    pad_b = via_stack(component, tech, LayerRole.LI, LayerRole.M2, (bx, by), draw_bottom_pad=False, axis="x")
    m2 = tech.layer(LayerRole.M2)
    half = rules.w(LayerRole.M2) / 2
    draw(component, m2, Box(ax - half, ay - half, ax + half, channel_y + half))
    draw(component, m2, Box(bx - half, by - half, bx + half, channel_y + half))
    draw(component, m2, Box(min(ax, bx) - half, channel_y - half, max(ax, bx) + half, channel_y + half))
    draw(component, m2, pad_a[LayerRole.M2])
    draw(component, m2, pad_b[LayerRole.M2])

