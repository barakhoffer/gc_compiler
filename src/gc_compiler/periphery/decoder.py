"""Physical layout of the static NAND/inverter row decoder, matched to the word-line pitch.

Mirrors ``netlist/decoder.py``'s gate list one for one (that module is the single source of
the logic: predecode *bus lines*, then one NAND per output) and builds it from the tight
``gc_std_inv``/``gc_std_nand2`` cells of ``periphery/standard_cells.py``.

Floorplan (all in the decoder's own frame; the macro turns it so the bottom faces the
word-line drivers):

* The **row strips** sit at the bottom. One holds every row's NAND, the other the write
  rows' output inverters. Their cells are *placed at the driver inputs*: each output's pin
  lands on (or a fraction of a micron from) the driver input it feeds. Everything between
  a row cell and its driver, and between a write row's NAND and its inverter, is a straight
  or lightly jogged metal2 wire -- no track, no router.
* A **bus channel** above them carries the predecode lines. Each line is one long track
  that every row NAND taps with a metal2 stub from its input pin.
* **Predecode strips** sit above the bus channel, one per logic level, with gates scheduled
  as late as possible so a bus line's producer is always in the strip just above the bus.

Routing discipline -- wires must provably be unable to short, because DRC alone cannot see a
short and only LVS can:

* Every strip abuts its gates (row strips bridge the gaps between gates with metal1 rail and
  n-well fill), so its supply rails are continuous by construction.
* A cell's ``y`` pin sits in its lower half and its ``a``/``b`` pins in its upper half (the
  ``nand2`` also has an internal metal2 strap across its lower half). So a ``y`` pin exits
  *down* and ``a``/``b`` exit *up*: a stub never runs through a cell interior.
* Vertical is always metal2 and horizontal always metal1 (a track) -- routing channels have
  no metal1 in them, so it costs no via2 and no metal3, and the track pitch is 0.465 um
  rather than 0.67. A crossing is therefore always between different layers.
* Inside one channel the pins hanging from above and the pins standing up from below sit at
  unrelated x. Two stubs from opposite sides can only touch if their tracks are ordered the
  wrong way round *and* their x are close, so tracks are ordered from a vertical-constraint
  graph, and nets with disjoint extents share a row (:mod:`periphery.channel`). A net that
  cannot be ordered consistently is split into one track per side.
* A net needing more than one track (spanning several channels, or split) ties them with a
  dedicated metal2 trunk to the right of every strip.

Inside a macro the decoder is given a :class:`DecoderInterface`: the pins it must reach (the
word-line driver inputs) and trunk positions that let a shared net (address/enable) be
extended across to the decoder on the other side of the array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from hashlib import sha256

import gdsfactory as gf
import klayout.db as kdb

from gc_compiler.primitives.wells import close_well_ends, fill_notches
from gc_compiler.cache import cached_cell
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.decoder import GateSpec, decoder_address, decoder_gates, decoder_inputs, output_net
from gc_compiler.periphery.channel import Node, Pin, assign_tracks, check_same_side_pitch
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.periphery.standard_cells import standard_cell
from gc_compiler.primitives.contacts import via_stack
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import LayerRole, Technology

_GAP = 0.2
"""Clearance between a gate strip and the tracks of its routing channel, and beyond the strips for ties."""

_RAIL_OVERLAP = 0.5
"""How far a supply reach starts inside a strip's last cell, to merge into the rail."""

_WIRE_CLEAR = 0.3
"""Clearance a straight metal2 wire keeps from any cell it passes."""

TRACK_ROLE = LayerRole.M1
"""Layer of the horizontal tracks. Routing channels have no metal1 in them, so tracks can sit
on it (via1 to the metal2 stubs) and metal3 stays free."""


@dataclass
class _Plan:
    """The gate-level netlist, before anything is placed."""

    gates: list[GateSpec] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


PARTS = ("all", "rows", "rows2", "predecode")
"""What a decoder cell holds: everything; only the row gates (the bus lines arrive as inputs,
on trunks pinned by ``DecoderInterface.trunks``); the row gates of two output groups around one
shared bus channel (``rows2``: the ``far`` group mirrored above it, its outputs on metal3); or
only the predecode (the bus lines leave on pinned trunks). A banked macro's central decoder is
one predecode shared by two ``rows2`` faces (see :mod:`gc_compiler.array.wordline_decoder`)."""


def _part_gates(cfg: ArrayConfig, outputs: list[tuple[int, str]] | None, banked: bool) -> list[GateSpec]:
    """The whole decoder's predecode, and the row gates of ``outputs`` only.

    Parts of one decoder must agree on the low/high address split -- that is what names the
    bus lines they exchange -- and the split that is cheapest for a subset of the outputs
    need not be the whole decoder's. So every part takes its gates from the whole decoder.
    """
    _bits, outputs, banked_rows = decoder_address(cfg, outputs, banked)
    wanted = {output_net(sig, r, banked_rows) for r, sig in outputs}
    return [gate for gate in decoder_gates(cfg, None, banked) if gate.role == "pre" or gate.output in wanted]


def bus_nets(cfg: ArrayConfig, outputs: list[tuple[int, str]] | None = None, banked: bool = False) -> list[str]:
    """The nets the row gates of ``outputs`` read: what a ``rows`` part takes in."""
    rows = [gate for gate in _part_gates(cfg, outputs, banked) if gate.role != "pre"]
    produced = {gate.pins["y"] for gate in rows}
    return sorted({net for gate in rows for pin, net in gate.pins.items() if pin != "y" and net not in produced})


def _plan(cfg: ArrayConfig, outputs: list[tuple[int, str]], banked: bool = False, part: str = "all") -> _Plan:
    """The netlist of ``netlist/decoder.py``, as gate objects."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    bits, outputs, banked_rows = decoder_address(cfg, outputs, banked)
    gates = decoder_gates(cfg, outputs, banked) if part == "all" else _part_gates(cfg, outputs, banked)
    plan = _Plan()
    if part == "predecode":
        plan.gates = [gate for gate in gates if gate.role == "pre"]
    elif part in ("rows", "rows2"):
        plan.gates = [gate for gate in gates if gate.role != "pre"]
    else:
        plan.gates = gates
    plan.inputs = (
        bus_nets(cfg, outputs, banked)
        if part in ("rows", "rows2")
        else decoder_inputs(cfg, banked)
    )
    plan.outputs = [] if part == "predecode" else [output_net(sig, r, banked_rows) for r, sig in outputs]
    return plan


def _predecode_levels(plan: _Plan) -> list[list[GateSpec]]:
    """Predecode gates grouped into strips, top (level 0) first.

    A gate's level is as late as its consumers allow, and every gate feeding a row gate goes
    in the last level: then a bus line's producer is always in the strip just above the bus
    channel, and a net only spans channels when it genuinely feeds several levels.
    """
    pre = [gate for gate in plan.gates if gate.role == "pre"]
    producer = {gate.pins["y"]: gate for gate in pre}
    consumers: dict[str, list[GateSpec]] = {}
    for gate in plan.gates:
        for pin, net in gate.pins.items():
            if pin != "y":
                consumers.setdefault(net, []).append(gate)
    asap: dict[int, int] = {}
    for gate in pre:
        deps = [asap[id(producer[net])] for pin, net in gate.pins.items() if pin != "y" and net in producer]
        asap[id(gate)] = 1 + max(deps) if deps else 0
    depth = max(asap.values(), default=-1)
    level: dict[int, int] = {}
    for gate in reversed(pre):
        users = consumers.get(gate.pins["y"], [])
        later = [level[id(user)] for user in users if user.role == "pre"]
        level[id(gate)] = depth if (not later or any(user.role != "pre" for user in users)) else min(later) - 1
    levels: list[list[GateSpec]] = [[] for _ in range(depth + 1)]
    for gate in pre:
        levels[level[id(gate)]].append(gate)
    return [gates for gates in levels if gates]


def _pack_levels(
    levels: list[list[GateSpec]], cells: dict[str, gf.Component], limit: float, grid: float
) -> list[list[GateSpec]]:
    """The predecode gates as strips, bottom strip first, each at most ``limit`` long.

    One strip per level wastes most of the shorter levels' length; instead the gates are laid
    bottom-up, last level first, and a strip is closed when the next gate would not fit. A
    producer is then always in the same strip as its consumer or above it: where it is not in
    the strip directly above, its net spans several channels and the router ties it with a
    trunk, as it already does for any multi-channel net.
    """
    strips: list[list[GateSpec]] = []
    current: list[GateSpec] = []
    length = 0.0
    for level in reversed(levels):
        for gate in level:
            width = cells[gate.kind].info["width"]
            if current and length + width > limit + grid / 2:
                strips.append(current)
                current, length = [], 0.0
            current.append(gate)
            length += width
    if current:
        strips.append(current)
    return strips


@dataclass(frozen=True)
class DecoderInterface:
    """How a decoder meets the macro around it.

    Coordinates are in the decoder's own frame: the bottom of the lowest strip is at
    ``y = 0`` and the strips extend upward from it.
    """

    targets: tuple[tuple[str, float, float], ...] = ()
    """``(output net, x, y)``: where each output must arrive, at or below ``y = 0``. Each
    output's row cell is placed so its pin lands on (or near) this x."""
    trunks: tuple[tuple[str, float], ...] = ()
    """``(net, x)``: nets that must have a metal2 trunk at exactly this x, so the macro can
    extend them (an address bit or enable shared with the decoder on the other side)."""
    max_strip: float | None = None
    """Longest a predecode strip may be. Default: as long as the rows (the targets' extent) or
    the longest logic level, whichever is more -- so packing never adds a strip; a smaller
    value may split a level to keep the predecode inside the rows."""
    pack: bool = True
    """Pack the predecode's logic levels into as few strips as fit (see ``max_strip``). A net
    whose producer and consumer share a strip needs a trunk, so a long predecode can come out
    smaller with one strip per level (``False``). The plain macro's half-decoders are unpacked:
    their minimum-size gates are slowed by the extra trunks more than packing saves area."""
    far: tuple[str, ...] = ()
    """``rows2`` only: the outputs of the mirrored group, above the bus channel. Each leaves
    on metal3, straight down over the near group to its target."""


@dataclass(frozen=True)
class _Geometry:
    """Widths and pitches every routing decision below derives from."""

    pad2: float
    pad_t: float
    pad_pitch: float
    tie_pitch: float
    track_pitch: float
    pad_t_short: float
    gap: float
    reach: float
    stub_pitch: float
    half1: float
    half2: float
    half_t: float
    min_len2: float
    min_len_t: float


def _geometry(tech: Technology) -> _Geometry:
    rules = tech.rules
    cut = {LayerRole.M1: LayerRole.VIA1, LayerRole.M3: LayerRole.VIA2}[TRACK_ROLE]
    pad2 = max(
        rules.w(LayerRole.M2),
        rules.cut(cut) + 2 * rules.enc(LayerRole.M2, cut),
    )
    pad_t = max(rules.w(TRACK_ROLE), rules.cut(cut) + 2 * rules.enc(TRACK_ROLE, cut))
    # Track vias use pads elongated along the track: the enclosure may be relaxed on the two
    # short sides, so a row only needs the pad's short side (and the metal2 pad's) of pitch.
    pad_short = max(
        rules.w(TRACK_ROLE),
        rules.cut(cut) + 2 * rules.enc_relaxed(TRACK_ROLE, cut),
        rules.cut(cut) + 2 * rules.enc_relaxed(LayerRole.M2, cut),
    )
    half1, half2, half_t = rules.w(LayerRole.M1) / 2, rules.w(LayerRole.M2) / 2, rules.w(TRACK_ROLE) / 2
    return _Geometry(
        pad2=pad2,
        pad_t=pad_t,
        # Closest two same-net via stacks may sit on one track without a notch between their
        # metal2 or track pads, and closest two trunks may sit side by side.
        pad_pitch=max(pad2 + rules.s(LayerRole.M2), pad_t + rules.s(TRACK_ROLE)),
        tie_pitch=max(pad2, rules.cut_pitch(cut)) + rules.s(LayerRole.M2),
        track_pitch=rules.snap(pad_short + max(rules.s(TRACK_ROLE), rules.s(LayerRole.M2)) + rules.grid),
        pad_t_short=pad_short,
        gap=max(_GAP, rules.s(LayerRole.M1), rules.s(LayerRole.M2)),
        # Closest a metal2 stub may run to another net's via pad, and to another net's stub.
        reach=half2 + pad2 / 2 + rules.s(LayerRole.M2) + rules.grid,
        stub_pitch=2 * half2 + rules.s(LayerRole.M2),
        half1=half1,
        half2=half2,
        half_t=half_t,
        # An isolated wire/pad below the layer's minimum drawn area is a real DRC violation,
        # not just a spacing nit, so every segment is padded up to it.
        min_len2=rules.area(LayerRole.M2) / (2 * half2) if half2 else 0.0,
        min_len_t=rules.area(TRACK_ROLE) / (2 * half_t) if half_t else 0.0,
    )


@dataclass
class _Layout:
    """Which gate sits where, before any routing."""

    plan: _Plan
    cells: dict[str, gf.Component]
    strips: list[list[GateSpec]]
    """Bottom strip first: the write inverters (if any), the row NANDs, then the predecode
    levels from the last to the first."""
    strip_x: list[list[float]]
    strip_right: list[float]
    """Right edge of each strip's last cell."""
    gapped: set[int]
    """Strips whose cells are not abutted (they sit at driver positions): rails need bridging."""
    targets: dict[str, float]
    band_x: float
    """Where the free metal2 trunk columns start: clear of every strip and target."""
    left_x: float = 0.0
    """The mirror image of ``band_x``: a trunk pinned on the left must be at least a column below it."""
    flipped: frozenset[int] = frozenset()
    """Strips drawn upside down (a ``y`` pin exits up, ``a``/``b`` down)."""
    shared: frozenset[int] = frozenset()
    """Channels of no height: the strips on either side share their vdd rail."""
    far: frozenset[str] = frozenset()
    """Outputs of a ``rows2`` part's mirrored group, which leave on metal3."""


def _default_targets(tech: Technology, cfg: ArrayConfig, outputs: list[str]) -> dict[str, float]:
    """Outputs at the array's word-line pitch, in the order given, for a decoder built alone."""
    from gc_compiler.cells.bitcell_3t import bitcell_geometry

    pitch = bitcell_geometry(tech, cfg.bitcell).pitch_y
    return {net: 0.5 + k * pitch for k, net in enumerate(outputs)}


def _layout(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig,
    outputs: list[tuple[int, str]],
    targets: tuple[tuple[str, float, float], ...],
    banked: bool = False,
    part: str = "all",
    max_strip: float | None = None,
    far: tuple[str, ...] = (),
    pack: bool = True,
) -> _Layout:
    rules = tech.rules
    geo = _geometry(tech)
    plan = _plan(cfg, outputs, banked, part)
    cells = {
        "inv": standard_cell(tech, "inv", logic, tight=True),
        "nand": standard_cell(tech, "nand2", logic, tight=True),
        "nand3": standard_cell(tech, "nand3", logic, tight=True),
        "nor2": standard_cell(tech, "nor2", logic, tight=True),
        "nor3": standard_cell(tech, "nor3", logic, tight=True),
    }
    target_x = {net: x for net, x, _y in targets} or _default_targets(tech, cfg, plan.outputs)
    missing = [net for net in plan.outputs if net not in target_x]
    if missing:
        raise ValueError(f"no target for outputs {missing}")

    width_n, width_i = cells["nand"].info["width"], cells["inv"].info["width"]
    y_n = cells["nand"].ports["y"].dcenter[0]
    y_i, a_i = (cells["inv"].ports[pin].dcenter[0] for pin in ("y", "a"))
    space = rules.s(LayerRole.DIFF) + rules.grid
    inv_left = y_n - a_i  # an inverter starts this far past its NAND's x (negative: before it)

    def place(nands: list[GateSpec], invs: list[GateSpec]):
        """``(inverters, their x, NANDs, their x)`` of one group of row gates, each strip sorted."""
        # A read row's NAND puts its y pin on the target. A write row's inverter puts its y pin
        # there, and its NAND sits so its y pin is right above the inverter's a pin. Where two
        # NANDs would overlap the later one slides right, and the small offset becomes a jog.
        ideal = []
        for gate in nands:
            t = target_x[gate.output]
            ideal.append((t - y_n if gate.role == "row_nand_r" else t - y_i + a_i - y_n, gate))
        ideal.sort(key=lambda item: item[0])
        nand_x: dict[int, float] = {}
        end = -math.inf
        # The write inverters share a rail with the NAND strip, so their tap diffusion sits next
        # to the NANDs': the gap between an inverter and a NAND that is not its own must be zero
        # (overlap, and merge) or at least the diffusion spacing.
        for x, gate in ideal:
            x = max(x, end)
            if math.isfinite(end):
                if 0 < x - end < space:  # tap diffusions of neighbouring NANDs: abut or keep clear
                    x = end + space
                if gate.role == "row_nand_w" and 0 < x + inv_left - end < space:
                    x = end - inv_left + space
            nand_x[id(gate)] = rules.snap(x)
            end = nand_x[id(gate)] + width_n
        nand_of = {gate.output: gate for gate in nands if gate.role == "row_nand_w"}
        inv_x = {id(gate): rules.snap(nand_x[id(nand_of[gate.output])] + y_n - a_i) for gate in invs}
        inv_order = sorted(invs, key=lambda gate: inv_x[id(gate)])
        for left, right in zip(inv_order, inv_order[1:]):
            if inv_x[id(right)] < inv_x[id(left)] + width_i:
                raise ValueError("word-line pitch is too tight for the write inverters")
        nand_order = sorted(nands, key=lambda gate: nand_x[id(gate)])
        return inv_order, [inv_x[id(g)] for g in inv_order], nand_order, [nand_x[id(g)] for g in nand_order]

    far_set = frozenset(far) if part == "rows2" else frozenset()
    if far_set - set(plan.outputs):
        raise ValueError(f"far outputs {sorted(far_set - set(plan.outputs))} are not outputs")
    row_gates = [gate for gate in plan.gates if gate.role != "pre"]
    strips: list[list[GateSpec]] = []
    strip_x: list[list[float]] = []
    gapped: set[int] = set()
    flipped: set[int] = set()
    shared: set[int] = set()
    for group, is_far in ((row_gates, False),) if not far_set else (
        ([g for g in row_gates if g.output not in far_set], False),
        ([g for g in row_gates if g.output in far_set], True),
    ):
        nands = [g for g in group if g.role in ("row_nand_r", "row_nand_w")]
        invs = [g for g in group if g.role == "row_inv"]
        if not nands:
            continue
        inv_order, inv_xs, nand_order, nand_xs = place(nands, invs)
        # near group, bottom up: write inverters (flipped, so their vdd rail meets the NANDs'),
        # then the NANDs. The far group is its mirror image above the shared bus channel.
        pair = [(nand_order, nand_xs), (inv_order, inv_xs)] if is_far else [(inv_order, inv_xs), (nand_order, nand_xs)]
        for index, (gates, xs) in enumerate(pair):
            if not gates:
                continue
            s_new = len(strips)
            strips.append(gates)
            strip_x.append(xs)
            gapped.add(s_new)
            if gates is inv_order and not is_far or gates is nand_order and is_far:
                flipped.add(s_new)
        if invs:
            shared.add(len(strips) - 1)  # the channel between the pair: one shared vdd rail
    levels = _predecode_levels(plan)
    if levels:
        widest = max(sum(cells[g.kind].info["width"] for g in level) for level in levels)
        rows_end = max(target_x.values(), default=0.0) + geo.pad2 / 2
        limit = max_strip if max_strip is not None else max(rows_end, widest)
        if pack:
            packed = _pack_levels(levels, cells, limit, rules.grid)
        else:
            # one strip per level -- except a level that only inverts a replica detector (a
            # self-timed read's rdy): one gate, which joins the next level's strip at the cost of
            # a single trunk instead of a whole strip and channel
            packed = []
            for level in reversed(levels):
                only_rdy = all(n.startswith("rdy") for g in level for p, n in g.pins.items() if p != "y")
                if packed and only_rdy:
                    packed[-1] = packed[-1] + list(level)
                else:
                    packed.append(list(level))
        for gates in packed:
            xs, x = [], 0.0
            for gate in gates:
                xs.append(x)
                x = rules.snap(x + cells[gate.kind].info["width"])
            strips.append(gates)
            strip_x.append(xs)

    strip_right = [
        max(x + cells[gate.kind].info["width"] for gate, x in zip(gates, xs))
        for gates, xs in zip(strips, strip_x)
    ]
    reach_x = max([*strip_right, *(x + geo.pad2 / 2 for x in target_x.values())])
    start_x = min([*(xs[0] for xs in strip_x if xs), *(x - geo.pad2 / 2 for x in target_x.values())])
    return _Layout(
        plan, cells, strips, strip_x, strip_right, gapped, target_x,
        rules.snap(reach_x + geo.gap), rules.snap(start_x - geo.gap),
        frozenset(flipped), frozenset(shared), far_set,
    )


def trunk_pitch(tech: Technology) -> float:
    """Spacing between two neighbouring metal2 trunks."""
    return _geometry(tech).tie_pitch


def trunk_band_x(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig | None,
    outputs: list[tuple[int, str]] | None,
    targets: tuple[tuple[str, float, float], ...],
    banked: bool = False,
    part: str = "all",
    max_strip: float | None = None,
    far: tuple[str, ...] = (),
    pack: bool = True,
) -> float:
    """Where the free trunk columns start for a decoder built with these targets; a shared
    trunk pinned by ``DecoderInterface.trunks`` must be at least one column beyond it.
    """
    activate(tech)
    return _layout(tech, cfg, logic or LogicConfig.for_tech(tech), outputs, targets, banked, part, max_strip, far, pack).band_x


def trunk_left_x(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig | None,
    outputs: list[tuple[int, str]] | None,
    targets: tuple[tuple[str, float, float], ...],
    banked: bool = False,
    part: str = "all",
    max_strip: float | None = None,
    far: tuple[str, ...] = (),
    pack: bool = True,
) -> float:
    """The left-hand counterpart of :func:`trunk_band_x`: a trunk pinned on the left of the
    strips must be at least one column below this x."""
    activate(tech)
    return _layout(tech, cfg, logic or LogicConfig.for_tech(tech), outputs, targets, banked, part, max_strip, far, pack).left_x


def row_decoder(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig | None = None,
    outputs: list[tuple[int, str]] | None = None,
    name: str | None = None,
    interface: DecoderInterface | None = None,
    banked: bool = False,
    part: str = "all",
    supply_x: float | None = None,
) -> gf.Component:
    """Build exactly the requested ``(row, "wwl_in"|"rwl_in")`` outputs.

    Ports: ``addr_<i>``, ``we``/``re`` (inputs), one per requested output (at the end of the
    wire that reaches its target), ``vdd``/``vss``, and ``bus_<net>`` at the low end of each
    trunk pinned by ``interface.trunks``. Without an interface the outputs are spaced at the
    array's word-line pitch, in the order given.

    ``banked``: ``outputs`` are logical rows of a two-bank decoder (see
    :mod:`gc_compiler.netlist.decoder`). ``part``: one of :data:`PARTS`. ``supply_x``: put the
    vss trunk at this x (vdd one trunk pitch beyond), so several parts' supply trunks line up.
    """
    logic = logic or LogicConfig.for_tech(tech)
    _bits, outputs, _banked_rows = decoder_address(cfg, outputs, banked)
    interface = interface or DecoderInterface()
    signature = sha256(
        repr((tech.name, logic, outputs, interface, banked, part, supply_x, cfg.self_timed_read)).encode("utf-8")
    ).hexdigest()[:16]
    key = name or f"gc_row_decoder_{cfg.rows}_{signature}"
    return cached_cell(key, lambda n: _build(tech, cfg, logic, outputs, interface, n, banked, part, supply_x), tech)


def _layer_bands(cell: gf.Component, tech: Technology, role: LayerRole) -> list[tuple[float, float]]:
    """``(y0, y1)`` of each separate shape a cell has on one layer, in the cell's own frame."""
    index = cell.kcl.layer(*tech.layer(role))
    region = kdb.Region(cell.kdb_cell.begin_shapes_rec(index)).merged()
    dbu = cell.kcl.dbu
    return sorted((p.bbox().bottom * dbu, p.bbox().top * dbu) for p in region.each())


def _wire(component: gf.Component, layer, half: float, points: list[tuple[float, float]]) -> None:
    """A manhattan wire through ``points`` (consecutive points share an x or a y)."""
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        draw(component, layer, Box(min(x0, x1) - half, min(y0, y1) - half, max(x0, x1) + half, max(y0, y1) + half))


def _build(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig,
    outputs: list[tuple[int, str]],
    interface: DecoderInterface,
    name: str,
    banked: bool = False,
    part: str = "all",
    supply_x: float | None = None,
) -> gf.Component:
    """Place the strips and route the channels between them."""
    activate(tech)
    rules = tech.rules
    g = _geometry(tech)
    layout = _layout(
        tech, cfg, logic, outputs, interface.targets, banked, part, interface.max_strip, interface.far,
        interface.pack,
    )
    plan, cells, strips = layout.plan, layout.cells, layout.strips
    component = gf.Component(name)
    cell_height = cells["inv"].info["height"]
    m1, m2 = tech.layer(LayerRole.M1), tech.layer(LayerRole.M2)
    track_layer = tech.layer(TRACK_ROLE)
    cut_role = {LayerRole.M1: LayerRole.VIA1, LayerRole.M3: LayerRole.VIA2}[TRACK_ROLE]
    grid = rules.grid

    def _span(lo: float, hi: float, min_len: float) -> tuple[float, float]:
        """Widen a segment to the minimum drawn length, keeping its ends on the grid."""
        if hi - lo >= min_len:
            return lo, hi
        mid = (lo + hi) / 2
        return (
            math.floor((mid - min_len / 2) / grid + 1e-6) * grid,
            math.ceil((mid + min_len / 2) / grid - 1e-6) * grid,
        )

    n_strips = len(strips)
    n_channels = n_strips + 1
    # The write inverters (strip 0) are flipped: their p-tap side faces the drivers (no n-well
    # to keep 1.27 um from the driver column's), and their vdd rail lands on the vdd rail of
    # the NAND strip above, which they share. A ``rows2`` part's far group mirrors that above
    # the bus channel: its NANDs are flipped (their inputs face down, into the bus channel),
    # and its inverters sit upright on the NANDs' vdd rail.
    flipped = layout.flipped
    # Nets that go pin to pin by a direct wire, not through a channel: everything a row gate
    # drives (the word-line inputs, and a write row's NAND -> inverter link).
    direct_nets = {gate.pins["y"] for gate in plan.gates if gate.role != "pre"}

    # --- collect the pins each channel sees. A ``y`` pin exits down (into the channel below
    # its strip), ``a``/``b`` exit up (into the one above). ---
    channel_pins: list[list[Pin]] = [[] for _ in range(n_channels)]
    for s, gates in enumerate(strips):
        for gate, gx in zip(gates, layout.strip_x[s]):
            for pin, net in gate.pins.items():
                if net in direct_nets:
                    continue
                px, py = cells[gate.kind].ports[pin].dcenter
                # a flipped cell's pins face the other way, and its local y is mirrored
                hangs = (pin == "y") != (s in flipped)
                local = cell_height - py if s in flipped else py
                channel_pins[s if hangs else s + 1].append(Pin(net, gx + px, s, local, hangs))

    # --- assign tracks channel by channel, then size the channels and stack the strips. A
    # pinned trunk left of the strips ties all of its net's tracks from the left. ---
    fixed = dict(interface.trunks)
    left = {net for net, x in fixed.items() if x < layout.band_x}
    span_count: dict[str, int] = {}
    for pins in channel_pins:
        for net in {pin.net for pin in pins}:
            span_count[net] = span_count.get(net, 0) + 1
    channels = []
    for pins in channel_pins:
        check_same_side_pitch(pins, g.stub_pitch, g.pad_pitch)
        channels.append(
            assign_tracks(
                pins, g.reach, g.pad_pitch, g.pad_t, rules.s(TRACK_ROLE) + rules.grid, g.min_len_t,
                tied=lambda net: (span_count.get(net, 0) > 1 or net in fixed) and net not in left,
                tied_left=lambda net: net in left,
            )
        )
    # A channel with no tracks only carries direct wires (and their jogs).
    channel_height = [
        ch.row_count * g.track_pitch + 2 * g.gap if ch.row_count else 2 * g.gap for ch in channels
    ]
    vdd_local = cells["inv"].ports["vdd"].dcenter[1]
    vss_local = cells["inv"].ports["vss"].dcenter[1]
    for c in layout.shared:
        # the channel between a flipped strip and an upright one has negative height: the two
        # vdd rails are the same rail, so the strips overlap by the rail's distance to the edge
        channel_height[c] = -2 * vdd_local
    channel_lo = [0.0] * n_channels
    strip_y = [0.0] * n_strips
    y = 0.0
    for c in range(n_channels):
        channel_lo[c] = y
        y += channel_height[c]
        if c < n_strips:
            strip_y[c] = y
            y += cell_height

    def rail_y(s: int, local: float) -> float:
        return strip_y[s] + (cell_height - local if s in flipped else local)

    for s, gates in enumerate(strips):
        for gate, gx in zip(gates, layout.strip_x[s]):
            ref = component.add_ref(cells[gate.kind])
            if s in flipped:
                ref.dmirror_y()
                ref.dmove((gx, strip_y[s] + cell_height))
            else:
                ref.dmove((gx, strip_y[s]))

    # --- rails and wells are continuous by abutment; where cells sit at driver positions the
    # gap between two of them needs its own rail metal and n-well ---
    rail_half = cells["inv"].ports["vdd"].dwidth / 2
    fill_bands = [
        (tech.layer(role), _layer_bands(cells["inv"], tech, role))
        for role in (LayerRole.NWELL, LayerRole.NIMP, LayerRole.PIMP)
    ]
    for s in sorted(layout.gapped):
        placed = sorted(
            (gx, gx + cells[gate.kind].info["width"]) for gate, gx in zip(strips[s], layout.strip_x[s])
        )
        for (_, end), (start, _) in zip(placed, placed[1:]):
            if start - end < grid / 2:
                continue
            for rail in (vdd_local, vss_local):
                ry = rail_y(s, rail)
                draw(component, m1, Box(end - 0.1, ry - rail_half, start + 0.1, ry + rail_half))
            for layer, bands in fill_bands:
                for y0, y1 in bands:
                    if s in flipped:
                        y0, y1 = cell_height - y1, cell_height - y0
                    draw(component, layer, Box(end, strip_y[s] + y0, start, strip_y[s] + y1))

    # --- direct wires: a row gate's output to its target, and a write row's NAND to its inverter ---
    cell_at = {id(gate): (s, gx) for s, gates in enumerate(strips) for gate, gx in zip(gates, layout.strip_x[s])}
    target_y = {net: y for net, _x, y in interface.targets}

    def pin_xy(gate: GateSpec, pin: str) -> tuple[float, float]:
        s, gx = cell_at[id(gate)]
        px, py = cells[gate.kind].ports[pin].dcenter
        return gx + px, rail_y(s, py)

    def free_below(x: float, s_from: int, s_to: int) -> None:
        """A wire at ``x`` passes strips ``s_from`` down to ``s_to`` (exclusive both ends)."""
        for s in range(s_to + 1, s_from):
            for gate, gx in zip(strips[s], layout.strip_x[s]):
                if gx - _WIRE_CLEAR < x < gx + cells[gate.kind].info["width"] + _WIRE_CLEAR:
                    raise ValueError(f"a wire at x={x:.3f} would run through a cell of strip {s}")

    nand_of_output = {gate.output: gate for gate in plan.gates if gate.role == "row_nand_w"}
    ports_out: dict[str, tuple[float, float]] = {}
    far_runs: list[tuple[str, float, float, float, float]] = []  # (net, pin x, pin y, target x, y)
    m3 = tech.layer(LayerRole.M3)
    half3 = rules.w(LayerRole.M3) / 2
    for gate in plan.gates:
        if gate.role not in ("row_nand_r", "row_inv"):
            continue
        net = gate.pins["y"]
        tx, ty = layout.targets[net], target_y.get(net, -_GAP)
        ports_out[net] = (tx, ty)
        x0, y0 = pin_xy(gate, "y")
        mid = channel_lo[0] + channel_height[0] / 2
        path = [(x0, y0), (tx, ty)] if abs(x0 - tx) < grid / 2 else [(x0, y0), (x0, mid), (tx, mid), (tx, ty)]
        if net in layout.far:
            # up to metal3 on the pin, then over everything below (the decoder has no other metal3).
            # An inverter's input link rises past its output pin on metal2: the via2 pad (wider than
            # the pin's own) keeps metal2 spacing from that leg, moving off the pin on a metal2 stub
            vx = x0
            if gate.role == "row_inv":
                ax = pin_xy(gate, "a")[0]
                need = rules.via_pad(LayerRole.M2, LayerRole.VIA2) / 2 + g.half2 + rules.s(LayerRole.M2)
                if abs(x0 - ax) < need:
                    vx = ax + (need if x0 >= ax else -need)
                    vx = rules.snap_up(vx) if x0 >= ax else rules.snap_down(vx)
                    _wire(component, m2, g.half2, [(x0, y0), (vx, y0)])
            via_stack(component, tech, LayerRole.M2, LayerRole.M3, (vx, y0))
            # metal3 from the via back over the pin, then the output's own path
            _wire(component, m3, half3, [(vx, y0)] + path if vx != x0 else path)
            far_runs.append((net, x0, y0, tx, ty))
        else:
            free_below(x0, cell_at[id(gate)][0], -1)
            _wire(component, m2, g.half2, path)
        if gate.role == "row_inv":
            (ax, ay) = pin_xy(gate, "a")
            nand = nand_of_output[gate.output]
            nx, ny = pin_xy(nand, "y")
            # the channel between the inverter's strip and the NAND's
            channel = max(cell_at[id(nand)][0], cell_at[id(gate)][0])
            if abs(nx - ax) >= grid / 2:
                if channel_height[channel] <= 0:
                    raise ValueError("the NAND -> inverter link needs a jog but the strips share a rail")
                mid = channel_lo[channel] + channel_height[channel] / 2
                _wire(component, m2, g.half2, [(nx, ny), (nx, mid), (ax, mid), (ax, ay)])
            else:
                _wire(component, m2, g.half2, [(nx, ny), (ax, ay)])

    # --- a net with several tracks (spanning channels, or split) ties them with a metal2
    # trunk off to the right of every strip; pinned trunks keep their x ---
    node_y: list[list[float]] = []
    tracks_of: dict[str, list[float]] = {}
    for c, ch in enumerate(channels):
        top = channel_lo[c] + channel_height[c]
        ys = [rules.snap(top - g.gap - (row + 0.5) * g.track_pitch) for row in ch.rows]
        node_y.append(ys)
        for node, ty in zip(ch.nodes, ys):
            tracks_of.setdefault(node.net, []).append(ty)
    multi_nets = sorted(net for net, tracks in tracks_of.items() if len(tracks) > 1 or net in fixed)
    tie_pitch = g.tie_pitch
    for net in multi_nets:
        if net not in fixed:
            continue
        inside = (
            fixed[net] > layout.left_x - tie_pitch + rules.grid / 2
            if net in left
            else fixed[net] < layout.band_x + tie_pitch - rules.grid / 2
        )
        if inside:
            raise ValueError(f"pinned trunk {net!r} at x={fixed[net]} is inside the strips/targets")
    free_nets = [net for net in multi_nets if net not in fixed]
    trunk_x = {net: fixed[net] for net in multi_nets if net in fixed}
    # free trunks take the columns beyond the strips that no pinned trunk is within a pitch of
    column = 0
    for net in free_nets:
        while True:
            column += 1
            x = rules.snap(layout.band_x + column * tie_pitch)
            if all(abs(x - pinned) >= tie_pitch - rules.grid / 2 for pinned in trunk_x.values()):
                break
        trunk_x[net] = x

    for ch, ys in zip(channels, node_y):
        for node, ty in zip(ch.nodes, ys):
            xs = [pin.x for pin in node.pins]
            if node.net in trunk_x:
                xs.append(trunk_x[node.net])
            lo, hi = _span(min(xs) - g.pad_t / 2, max(xs) + g.pad_t / 2, g.min_len_t)
            draw(component, track_layer, Box(lo, ty - g.half_t, hi, ty + g.half_t))
            for pin in node.pins:
                py = pin.local_y if pin.strip < 0 else strip_y[pin.strip] + pin.local_y
                y0, y1 = _span(*sorted((py, ty)), g.min_len2)
                draw(component, m2, Box(pin.x - g.half2, y0, pin.x + g.half2, y1))
                via_stack(component, tech, TRACK_ROLE, LayerRole.M2, (pin.x, ty), axis="x")
            if node.net in trunk_x:
                via_stack(component, tech, TRACK_ROLE, LayerRole.M2, (trunk_x[node.net], ty), axis="x")

    for net, tx in trunk_x.items():
        y0, y1 = _span(min(tracks_of[net]), max(tracks_of[net]), g.min_len2)
        draw(component, m2, Box(tx - g.half2, y0, tx + g.half2, y1))
        if net in fixed:
            add_port(component,
                name=f"bus_{net}", center=(tx, y0), orientation=270,
                width=rules.w(LayerRole.M2), layer=m2, port_type="electrical",
            )

    # --- supplies: continuous within a strip, but separate strips each need their own reach to
    # a trunk. Each reach starts inside the strip's last cell (so it merges into the rail
    # without a notch against the cell's own tap tabs) and both stay on metal1 at the rail's
    # height. vss lifts to a metal2 trunk *nearer* the strips than vdd's metal1 trunk, so
    # vdd's reach crosses vss's trunk on a different layer, and no wire ever crosses another
    # metal1 wire. (The metal1 tracks end at their own trunks, short of both.) ---
    last_x = max([x for net, x in trunk_x.items() if net not in left], default=layout.band_x)
    vss_x = rules.snap(last_x + tie_pitch)
    if supply_x is not None:
        if supply_x < vss_x - grid / 2:
            raise ValueError(f"supply trunk at x={supply_x} is inside the trunks (needs >= {vss_x})")
        vss_x = rules.snap(supply_x)
    vdd_x = rules.snap(vss_x + tie_pitch)
    vdd_ys = [rail_y(s, vdd_local) for s in range(n_strips)]
    vss_ys = [rail_y(s, vss_local) for s in range(n_strips)]
    for s in range(n_strips):
        start = layout.strip_right[s] - _RAIL_OVERLAP
        draw(component, m1, Box(start, vdd_ys[s] - g.half1, vdd_x + g.half1, vdd_ys[s] + g.half1))
        draw(component, m1, Box(start, vss_ys[s] - g.half1, vss_x, vss_ys[s] + g.half1))
        via_stack(component, tech, LayerRole.M1, LayerRole.M2, (vss_x, vss_ys[s]))
    draw(component, m1, Box(vdd_x - g.half1, min(vdd_ys) - g.half1, vdd_x + g.half1, max(vdd_ys) + g.half1))
    y0, y1 = _span(min(vss_ys), max(vss_ys), g.min_len2)
    draw(component, m2, Box(vss_x - g.half2, y0, vss_x + g.half2, y1))

    add_port(component,
        name="vdd", center=(vdd_x, vdd_ys[0]), orientation=270,
        width=rules.w(LayerRole.M1), layer=m1, port_type="electrical",
    )
    add_port(component,
        name="vss", center=(vss_x, vss_ys[0]), orientation=270,
        width=rules.w(LayerRole.M2), layer=m2, port_type="electrical",
    )
    label1 = tech.label_layer(LayerRole.M1)
    if label1 is not None:
        component.add_label("vdd", position=(vdd_x, vdd_ys[0]), layer=label1)
    label2 = tech.label_layer(LayerRole.M2)
    if label2 is not None:
        component.add_label("vss", position=(vss_x, vss_ys[0]), layer=label2)

    # a far output's jog must not cross another far output's metal3
    for net, x0, _y0, tx, _ty in far_runs:
        lo, hi = sorted((x0, tx))
        if hi - lo < grid / 2:
            continue
        for other, ox, _oy, otx, _ in far_runs:
            if other != net and any(lo - half3 - rules.s(LayerRole.M3) < x < hi + half3 + rules.s(LayerRole.M3) for x in (ox, otx)):
                raise ValueError(f"far output {net}'s jog crosses {other}'s metal3")

    _ports(component, tech, plan, [ch.nodes for ch in channels], node_y, ports_out, layout.far)
    close_well_ends(component, tech)  # the gate strips' wells end flush with their end taps
    fill_notches(component, tech)  # neighbouring cells' stubs off one rail
    bounds = component.dbbox()
    component.info["width"] = bounds.width()
    component.info["height"] = bounds.height()
    component.info["gate_count"] = len(plan.gates)
    component.info["row_count"] = n_strips
    component.info["net_count"] = len(tracks_of)
    component.info["direct_count"] = len(direct_nets)
    component.info["track_count"] = sum(len(ch.nodes) for ch in channels)
    component.info["track_row_count"] = sum(ch.row_count for ch in channels)
    component.info["tie_count"] = len(multi_nets)
    component.info["split_count"] = sum(ch.splits for ch in channels)
    component.info["channel_rows"] = ",".join(str(ch.row_count) for ch in channels)
    return component


def _ports(
    component: gf.Component,
    tech: Technology,
    plan: _Plan,
    channel_nodes: list[list[Node]],
    node_y: list[list[float]],
    output_ends: dict[str, tuple[float, float]],
    far: frozenset[str] = frozenset(),
) -> None:
    """Expose each input at the left end of its highest-channel track and each output at the
    end of the wire that reaches its target.
    """
    rules = tech.rules
    track = tech.layer(TRACK_ROLE)
    track_label = tech.label_layer(TRACK_ROLE)
    for net in plan.inputs:
        found = next(
            (
                (min(pin.x for pin in node.pins), ty)
                for c in range(len(channel_nodes) - 1, -1, -1)
                for node, ty in zip(channel_nodes[c], node_y[c])
                if node.net == net
            ),
            None,
        )
        if found is None:
            continue
        add_port(component,
            name=net, center=found, orientation=180, width=rules.w(TRACK_ROLE), layer=track,
            port_type="electrical",
        )
        if track_label is not None:
            component.add_label(text=net, position=found, layer=track_label)
    for net in plan.outputs:
        role = LayerRole.M3 if net in far else LayerRole.M2
        add_port(component,
            name=net, center=output_ends[net], orientation=270, width=rules.w(role), layer=tech.layer(role),
            port_type="electrical",
        )
        label = tech.label_layer(role)
        if label is not None:
            component.add_label(text=net, position=output_ends[net], layer=label)
