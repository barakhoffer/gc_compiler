"""Place the two half-decoders beside the word-line drivers and wire them up.

Each side of the array has a column of driver slices whose inputs face outward, so the
matching half-decoder sits beyond it, turned so that its output channel faces the
drivers. The left one is rotated a quarter turn, the right one transposed; either way
the decoder's local x axis runs along the macro's y axis, so both decoders' strips (and
trunk columns) are ordered the same way up the macro:

    left   (x, y) -> (-y + ox, x + oy)         right  (x, y) -> (y + ox, x + oy)

Each decoder routes its own outputs: every driver input is handed to it as a target
(:class:`~gc_compiler.periphery.decoder.DecoderInterface`), its row cells are placed at those
targets, and each output reaches its driver on a direct metal2 wire.

The two halves share ``addr``/``we``/``re``. Those nets get a metal2 trunk at the same
macro y in both decoders, above everything else in the macro, and a straight metal2 wire
joins the pair across the top. Supplies leave each decoder the same way and land on the
driver columns' metal1 rails, which are extended up to meet them.

**The central decoder** (:func:`central_decoder`) serves a banked macro: two banks, one
decoder between them. Each bank still needs a driver column on both of its edges (a row
pair's word lines are too close to land from one side), so each bank's *face* holds two
groups of row gates around one bus channel: the *near* group, whose outputs reach the driver
column next to the decoder on metal2, as above, and the *far* group, mirrored beyond the
channel, whose outputs rise to metal3 on their pins, pass over the near group and cross the
bank to the driver column on its other edge. Both faces read the same bus lines, from one
*predecode* in the middle -- which is the point: the predecode, most of a decoder's depth, is
built once for both banks instead of twice per bank. Left to right::

    [bank 0 face][predecode][bank 1 face]

Every bus line (and ``addr``/``we``/``re``) gets its trunk pinned at the same column in all
three parts, so each is one straight metal2 line along the decoder. They sit below the rows:
there the column-circuit band is empty beside the decoder, where above the array they would
add macro height (and did, cancelling the saving). The supply trunks are aligned the same way,
above the rows, where they reach the inner driver columns' rails.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import gdsfactory as gf
import klayout.db as kdb

from gc_compiler.array.tiler import ArrayGeometry
from gc_compiler.cache import cached_cell
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.decoder import (
    address_bits,
    banked_address_bits,
    banked_face_outputs,
    decoder_inputs,
    output_net,
    side_outputs,
)
from gc_compiler.periphery.decoder import (
    DecoderInterface,
    bus_nets,
    row_decoder,
    trunk_band_x,
    trunk_left_x,
    trunk_pitch,
)
from gc_compiler.tech.pdk import activate
from gc_compiler.periphery.logic import LogicConfig, logic_geometry
from gc_compiler.periphery.wl_driver import LEFT_WORDLINES, RIGHT_WORDLINES, far_track_y
from gc_compiler.primitives.contacts import via_stack
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.rules import LayerRole, Technology

OUTPUT_GAP = 0.6
"""Distance from a decoder's output-channel edge to the driver input line it feeds."""
_OVERLAP = 0.2
"""How far an output stub reaches into a driver's input line, so the two merge."""
_ROW_MARGIN = 0.5
"""How far the first word-line input is from the decoder's edge along the array."""
_TRUNK_OVERLAP = 0.3
"""How far a macro-level wire reaches back into the decoder trunk it extends."""
_BUS_MARGIN = 1.0
"""Clearance between the array's top edge and the lowest shared trunk."""
_RAIL_OVERLAP = 0.5
"""How far a rail extension starts inside the driver column's own strap."""
_FACE_GAP = 1.0
"""Space between neighbouring parts of the central decoder."""


def far_line_y(tech: Technology, array: ArrayGeometry, array_y: float, net: str) -> float:
    """Macro y of a far output's metal3 line on a strapped array: the spare track of its row
    (``{signal}_in_{bank}_{row}``; even rows take the pair's lower track, odd rows the upper)."""
    row = int(net.rsplit("_", 1)[1])
    low, high = far_track_y(tech, array.cell)
    return tech.rules.snap(array_y + array.row_base_y[row - row % 2] + (high if row % 2 else low))


def _bank_keys(found: dict[str, tuple[float, float]], bank: int) -> dict[str, tuple[float, float]]:
    """``rwl_in_5`` -> ``rwl_in_{bank}_5`` (the banked decoder's output names)."""
    return {f"{key.rsplit('_', 1)[0]}_{bank}_{key.rsplit('_', 1)[1]}": xy for key, xy in found.items()}


def _single_x(found: dict[str, tuple[float, float]]) -> float:
    xs = {round(x, 6) for x, _y in found.values()}
    if len(xs) != 1:
        raise ValueError("driver inputs of one column are not aligned in x")
    return next(iter(xs))


@dataclass(frozen=True)
class CentralPlan:
    """Where a banked macro's central decoder and second bank go (see :func:`plan_central_decoder`)."""

    cell: gf.Component
    x: float
    """Macro x of the decoder cell's origin (its bank 0 near targets)."""
    oy: float
    """Macro y of the decoder cell's origin."""
    bank_x: float
    """Macro x at which bank 1's driver/array/driver block starts."""


def plan_central_decoder(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig,
    array: ArrayGeometry,
    array_y: float,
    drivers: list,
    predecode_logic: LogicConfig | None = None,
) -> CentralPlan:
    """Build the central decoder for bank 0's placed ``drivers`` and say where bank 1 goes.

    Bank 1 is a copy of bank 0 further right, so its driver inputs have bank 0's y. Bank 0's
    near outputs go to its right column, the far ones to its left; bank 1 the other way round.
    """
    rules = tech.rules
    left, right = _driver_inputs(drivers)
    near = (_bank_keys(right, 0), _bank_keys(left, 1))
    far = (_bank_keys(left, 0), _bank_keys(right, 1))
    if array.wordline_straps:
        # the straps take the tracks over the word lines: far outputs cross on the spare ones
        far = tuple({net: (x, far_line_y(tech, array, array_y, net)) for net, (x, _y) in side.items()} for side in far)
    oy = rules.snap(min(y for side in (*near, *far) for _x, y in side.values()) - _ROW_MARGIN)

    def frame(side: dict[str, tuple[float, float]]) -> dict[str, float]:
        return {net: y - oy for net, (_x, y) in side.items()}

    min_supply = array_y + array.height + _BUS_MARGIN - oy  # clear of the driver columns
    cell = central_decoder(
        tech, cfg, logic, (frame(near[0]), frame(near[1])), (frame(far[0]), frame(far[1])), min_supply,
        predecode_logic=predecode_logic,
    )
    x = _single_x(right) - _OVERLAP  # bank 0's inner column takes its inputs on its right edge
    bank_x = rules.snap(x + cell.info["width"] - _OVERLAP - _single_x(left))
    return CentralPlan(cell, x, oy, bank_x)


def wire_central_decoder(
    component: gf.Component,
    tech: Technology,
    logic: LogicConfig,
    plan: CentralPlan,
    array: ArrayGeometry,
    array_y: float,
    bank_width: float,
    drivers: tuple[list, list],
) -> gf.Instance:
    """Place the planned decoder and wire it to both banks (placed as ``plan`` says).

    Far outputs cross their bank on metal3 at the driver input's y and drop onto the outer
    column's input just outside it; the supply trunks land on both inner columns' rails;
    ``addr``/``we``/``re`` become the macro's pins.
    """
    rules = tech.rules
    ref = component.add_ref(plan.cell)
    ref.dmove((plan.x, plan.oy))
    ports = {port.name: port for port in ref.ports}
    m1, m2, m3 = (tech.layer(role) for role in (LayerRole.M1, LayerRole.M2, LayerRole.M3))
    half1, half2, half3 = (rules.w(role) / 2 for role in (LayerRole.M1, LayerRole.M2, LayerRole.M3))

    # --- far outputs: across the bank on metal3, down to metal2 just outside the outer column
    # (and, on a strapped array, along it from the spare track to the driver input's y) ---
    outer = (_bank_keys(_driver_inputs(drivers[0])[0], 0), _bank_keys(_driver_inputs(drivers[1])[1], 1))
    for bank, found in enumerate(outer):
        out = -1.0 if bank == 0 else 1.0  # away from the macro
        for net, (ix, iy) in found.items():
            fx, fy = ports[net].dcenter
            expected = far_line_y(tech, array, array_y, net) if array.wordline_straps else iy
            if abs(fy - expected) > rules.grid / 2:
                raise AssertionError(f"far output {net} is not at y={expected}")
            end = ix + out * OUTPUT_GAP
            draw(component, m3, Box(min(fx, end) - half3, fy - half3, max(fx, end) + half3, fy + half3))
            via_stack(component, tech, LayerRole.M2, LayerRole.M3, (end, fy))
            if abs(fy - iy) > rules.grid / 2:
                pad2 = rules.cut(LayerRole.VIA2) / 2 + rules.enc(LayerRole.M2, LayerRole.VIA2)
                draw(component, m2, Box(end - pad2, min(fy, iy) - half2, end + pad2, max(fy, iy) + half2))
            lo, hi = sorted((ix - out * _OVERLAP, end))
            draw(component, m2, Box(lo, iy - half2, hi, iy + half2))

    # --- supplies: both ends of the decoder's trunks to the inner columns' rails ---
    lg = logic_geometry(tech, logic)
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    rail_x = {"vdd": lg.ntap_x + tap_w / 2, "vss": lg.ptap_x + tap_w / 2}
    array_top = array_y + array.height
    for net in ("vdd", "vss"):
        for port_name, rx in ((net, bank_width - rail_x[net]), (f"{net}_right", plan.bank_x + rail_x[net])):
            px, py = ports[port_name].dcenter
            x0, x1 = sorted((px, rx))
            if net == "vdd":
                draw(component, m1, Box(x0 - half1, py - half1, x1 + half1, py + half1))
            else:
                draw(component, m2, Box(x0 - half2, py - half2, x1 + half2, py + half2))
                via_stack(component, tech, LayerRole.M1, LayerRole.M2, (rx, py))
            draw(component, m1, Box(rx - half1, array_top - _RAIL_OVERLAP, rx + half1, py + half1))

    # --- the address and enables are the macro's pins ---
    label = tech.label_layer(LayerRole.M2)
    for name, port in ports.items():
        if name.startswith("addr_") or name in ("we", "re"):
            add_port(component,
                name=name, center=tuple(port.dcenter), orientation=90, width=rules.w(LayerRole.M2),
                layer=m2, port_type="electrical",
            )
            if label is not None:
                component.add_label(text=name, position=tuple(port.dcenter), layer=label)
    return ref


def central_decoder(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig | None,
    near: tuple[dict[str, float], dict[str, float]],
    far: tuple[dict[str, float], dict[str, float]],
    min_supply: float = 0.0,
    name: str | None = None,
    predecode_logic: LogicConfig | None = None,
) -> gf.Component:
    """The shared decoder of a banked macro (see the module docstring).

    ``near[b]``/``far[b]`` map each of bank ``b``'s near/far outputs (``{signal}_{b}_{row}``)
    to the y it must arrive at. The cell's x runs from bank 0's near targets (``x = 0``) to
    bank 1's (``x = info["width"]``); its y is the macro's, shifted so the targets' y are as
    given. The shared trunks are below the lowest target; the supply trunks are above the
    highest, at ``min_supply`` or beyond.

    Ports: every output at ``x = 0``/``width`` (near ones on metal2, far ones on metal3), ``addr_*``/``we``/``re`` on their trunks, and ``vdd``/``vss`` at
    both ends of the supply trunks (``vdd_right``/``vss_right`` on bank 1's side).
    """
    logic = logic or LogicConfig.for_tech(tech)
    predecode_logic = predecode_logic or logic
    signature = sha256(
        repr((tech.name, cfg.rows, cfg.bitcell, cfg.self_timed_read, logic, predecode_logic, near, far, min_supply)).encode("utf-8")
    ).hexdigest()[:16]
    key = name or f"gc_central_decoder_{cfg.rows}_{signature}"
    return cached_cell(
        key, lambda n: _central_decoder(tech, cfg, logic, near, far, min_supply, n, predecode_logic), tech
    )


def _central_decoder(tech, cfg, logic, near, far, min_supply, name, predecode_logic) -> gf.Component:
    activate(tech)
    rules = tech.rules
    faces = []  # (outputs, targets, far nets), one per bank
    for bank in (0, 1):
        outputs, targets, far_nets = [], [], []
        for is_far, group in enumerate(banked_face_outputs(cfg.rows, bank)):
            ys = (far if is_far else near)[bank]
            for r, sig in group:
                net = output_net(sig, r, cfg.rows)
                outputs.append((r, sig))
                targets.append((net, ys[net], -OUTPUT_GAP))
                if is_far:
                    far_nets.append(net)
        faces.append((outputs, tuple(targets), tuple(far_nets)))
    # The replica detectors' trunks nearest the rows: the lowest trunks run close to the metal3
    # that joins the banks' pch, and each rdy leaves its trunk on metal3, one bank's to the left
    # and the other's to the right -- so another trunk goes between the two, or each one's route
    # would pass over the other's via.
    inputs = decoder_inputs(cfg, banked=True)
    rdy = [net for net in inputs if net.startswith("rdy")]
    others = [net for net in inputs if not net.startswith("rdy")]
    shared = [rdy[0], others[-1], rdy[1], *others[:-1]] if rdy else others
    shared += [net for net in bus_nets(cfg, banked=True) if net not in shared]
    pitch = trunk_pitch(tech)
    # the shared trunks go on the strips' left: in the macro that is below the rows, where the
    # column circuits' band is empty beside the decoder, so they cost no height
    left = min(
        [trunk_left_x(tech, cfg, logic, outputs, targets, True, "rows2", far=f) for outputs, targets, f in faces]
        + [trunk_left_x(tech, cfg, predecode_logic, None, (), True, "predecode", pack=p) for p in (True, False)]
    )
    trunks = tuple((net, rules.snap(left - (k + 1) * pitch)) for k, net in enumerate(shared))

    def face_cells(supply_x: float | None) -> list[gf.Component]:
        return [
            row_decoder(
                tech, cfg, logic, outputs, interface=DecoderInterface(targets, trunks, far=f),
                banked=True, part="rows2", supply_x=supply_x,
            )
            for outputs, targets, f in faces
        ]

    def predecode_cell(max_strip: float | None, supply_x: float | None, pack: bool) -> gf.Component:
        return row_decoder(
            tech, cfg, predecode_logic, None, interface=DecoderInterface((), trunks, max_strip, pack=pack),
            banked=True, part="predecode", supply_x=supply_x,
        )

    def vss_x(cell: gf.Component) -> float:
        return cell.ports["vss"].dcenter[0]

    # The predecode's strips may be no longer than the faces' rows: past them, its supply trunks
    # (to which every part's are aligned) would lift the whole macro's top edge. Packing adds
    # free trunks beyond the strips, so shorten the strips until its supply lands no higher.
    # A packed strip's internal nets need trunks too, so one strip per level may still come out
    # narrower: keep whichever lands the supply lower, then is narrower.
    face_supply = max(vss_x(cell) for cell in face_cells(None))
    options = []
    for pack in (True, False):
        max_strip = face_supply
        for _ in range(6 if pack else 0):
            overshoot = vss_x(predecode_cell(max_strip, None, pack)) - face_supply
            if overshoot <= rules.grid / 2:
                break
            max_strip = rules.snap(max_strip - overshoot)
        cell = predecode_cell(max_strip, None, pack)
        top = rules.snap(max(face_supply, vss_x(cell), min_supply))
        options.append((top, cell.dbbox().height(), not pack, max_strip))
    supply_x, _width, unpacked, max_strip = min(options)
    built = face_cells(supply_x)
    pre = predecode_cell(max_strip, supply_x, not unpacked)
    cells = [built[0], pre, built[1]]

    # --- place, left to right. Bank 0's face (and the predecode) face left: local (x, y) ->
    # (y + ox, x); bank 1's faces right: (x, y) -> (-y + ox, x). A face's strips start at its
    # local y = 0, and its outputs end at local y = -OUTPUT_GAP, on the bank's inner column. ---
    component = gf.Component(name)
    refs = []
    edge = 0.0  # where the next part may start
    for index, cell in enumerate(cells):
        bottom, top = cell.dbbox().bottom, cell.dbbox().top
        faces_left = index < 2
        if index == 0:
            ox = OUTPUT_GAP  # its targets (local y = -OUTPUT_GAP) at x = 0
        elif index == 1:
            ox = edge + _FACE_GAP - bottom
        else:
            ox = edge + _FACE_GAP + top  # turned the other way: its top is its left edge
        ref = component.add_ref(cell)
        ref.dcplx_trans = kdb.DCplxTrans(1.0, 90, faces_left, ox, 0.0)
        refs.append(ref)
        edge = ox + top if faces_left else ox + OUTPUT_GAP
    width = edge  # bank 1's targets
    component.info["width"] = width

    m1, m2 = tech.layer(LayerRole.M1), tech.layer(LayerRole.M2)
    half1, half2 = rules.w(LayerRole.M1) / 2, rules.w(LayerRole.M2) / 2
    ports_of = [{port.name: port for port in ref.ports} for ref in refs]

    def label(net: str, role: LayerRole, xy: tuple[float, float]) -> None:
        layer = tech.label_layer(role)
        if layer is not None:
            component.add_label(text=net, position=xy, layer=layer)

    # --- each shared net: one straight metal2 line through its trunk in every part using it ---
    for net, _x in trunks:
        ends = [ports[f"bus_{net}"].dcenter for ports in ports_of if f"bus_{net}" in ports]
        if not ends:
            continue
        x0, x1 = min(x for x, _y in ends), max(x for x, _y in ends)
        y = ends[0][1]
        draw(component, m2, Box(x0 - half2, y - half2, x1 + half2, y + half2))
        if net.startswith("addr_") or net in ("we", "re"):
            add_port(component,
                name=net, center=(x0, y), orientation=90, width=rules.w(LayerRole.M2), layer=m2,
                port_type="electrical",
            )
            label(net, LayerRole.M2, (x0, y))
        elif net.startswith("rdy"):
            # a replica detector: the macro brings each bank's in from its own side
            for end, x in (("left", x0), ("right", x1)):
                add_port(component,
                    name=f"trunk_{net}_{end}", center=(x, y), orientation=180 if end == "left" else 0,
                    width=rules.w(LayerRole.M2), layer=m2, port_type="electrical",
                )

    # --- supplies: aligned trunks, joined into one metal1 (vdd) and one metal2 (vss) line ---
    for net, layer, half, role in (("vdd", m1, half1, LayerRole.M1), ("vss", m2, half2, LayerRole.M2)):
        ends = [ports[net].dcenter for ports in ports_of]
        y = ends[0][1]
        x0, x1 = min(x for x, _y in ends), max(x for x, _y in ends)
        draw(component, layer, Box(x0 - half, y - half, x1 + half, y + half))
        for port_name, x in ((net, x0), (f"{net}_right", x1)):
            add_port(component,
                name=port_name, center=(x, y), orientation=90, width=2 * half, layer=layer,
                port_type="electrical",
            )
        label(net, role, (x0, y))

    # --- outputs: all at the bank's inner column, near ones on metal2, far ones on metal3 ---
    for bank, ((outputs, _targets, far_nets), ports) in enumerate(zip(faces, (ports_of[0], ports_of[2]))):
        for r, sig in outputs:
            net = output_net(sig, r, cfg.rows)
            role = LayerRole.M3 if net in far_nets else LayerRole.M2
            xy = tuple(ports[net].dcenter)
            add_port(component,
                name=net, center=xy, orientation=180 if bank == 0 else 0,
                width=rules.w(role), layer=tech.layer(role), port_type="electrical",
            )
            label(net, role, xy)
    component.info["predecode_width"] = cells[1].dbbox().height()
    component.info["face_widths"] = ",".join(f"{cells[i].dbbox().height():.2f}" for i in (0, 2))
    return component


def _driver_inputs(drivers: list) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Absolute ``(x, y)`` of every driver input, per side, keyed by decoder output name."""
    sides: tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]] = ({}, {})
    for pair_index, left_ref, right_ref in drivers:
        for ref, found in ((left_ref, sides[0]), (right_ref, sides[1])):
            ports = {port.name: port for port in ref.ports}
            for net in LEFT_WORDLINES + RIGHT_WORDLINES:
                port = ports.get(f"{net}_in")
                if port is None:
                    continue
                signal, relative_row = net.rsplit("_", 1)
                found[f"{signal}_in_{2 * pair_index + int(relative_row)}"] = tuple(port.dcenter)
    return sides


def place_decoders(
    component: gf.Component,
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig,
    array: ArrayGeometry,
    array_y: float,
    macro_width: float,
    drivers: list,
    driver_logic: LogicConfig | None = None,
) -> tuple[gf.Instance, gf.Instance]:
    """Add both half-decoders, their bus, and their supply hookup; return the two instances.

    ``macro_width`` is the width of the array plus both driver columns (what the drivers'
    mirrored supply rails are reflected about); ``drivers`` is as returned for the macro.
    ``driver_logic``: the drivers' sizes (default ``logic``), which place their supply rails.
    """
    rules = tech.rules
    m1, m2 = tech.layer(LayerRole.M1), tech.layer(LayerRole.M2)
    half1, half2 = rules.w(LayerRole.M1) / 2, rules.w(LayerRole.M2) / 2
    left_outputs, right_outputs = side_outputs(cfg.rows)
    inputs = _driver_inputs(drivers)
    shared = decoder_inputs(cfg)
    pitch = trunk_pitch(tech)
    array_top = array_y + array.height

    def arrange():
        """Both half-decoders' cells, placements and bus trunks."""
        # --- where each decoder sits along the macro's y axis, and what it has to reach ---
        frames = []
        for mirror, outputs, found in ((False, left_outputs, inputs[0]), (True, right_outputs, inputs[1])):
            xs = {x for x, _y in found.values()}
            if len(xs) != 1:
                raise ValueError("driver inputs of one side are not aligned in x")
            (port_x,) = xs
            # the stub reaches _OVERLAP into the driver's input line, which runs away from its edge
            target_x = port_x - _OVERLAP if mirror else port_x + _OVERLAP
            ys = [found[f"{signal}_{row}"][1] for row, signal in outputs]
            oy = rules.snap(min(ys) - _ROW_MARGIN)
            targets = tuple(
                (f"{signal}_{row}", rules.snap(found[f"{signal}_{row}"][1] - oy), -OUTPUT_GAP)
                for row, signal in outputs
            )
            ox = rules.snap(target_x + OUTPUT_GAP if mirror else target_x - OUTPUT_GAP)
            band = trunk_band_x(tech, cfg, logic, outputs, targets, pack=False)
            frames.append((mirror, outputs, targets, ox, oy, band))

        # --- shared nets: one trunk each at the same macro y in both decoders, above everything ---
        first = rules.snap(
            max(*(oy + band for _m, _o, _t, _ox, oy, band in frames), array_top + _BUS_MARGIN) + pitch + rules.grid
        )
        bus_y = {net: rules.snap(first + i * pitch) for i, net in enumerate(shared)}
        cells = []
        for mirror, outputs, targets, ox, oy, _band in frames:
            interface = DecoderInterface(
                targets=targets, trunks=tuple((net, rules.snap(bus_y[net] - oy)) for net in shared), pack=False
            )
            cells.append((row_decoder(tech, cfg, logic, outputs, interface=interface), mirror, ox, oy))
        return cells, bus_y

    # The predecode stays one strip per level: packed, its minimum-size gates drive the extra
    # trunks packing adds, and a plain macro reads slower for a few percent of area (32x32: 4%
    # smaller, 8% slower; 64x32: 9% smaller, 24% slower). Only the central decoder's sized-up
    # predecode is packed.
    cells, bus_y = arrange()
    refs = []
    for cell, mirror, ox, oy in cells:
        ref = component.add_ref(cell)
        ref.dcplx_trans = kdb.DCplxTrans(1.0, 90, mirror, ox, oy)
        refs.append(ref)
    left_ref, right_ref = refs

    for net in shared:
        p_left, p_right = left_ref.ports[f"bus_{net}"], right_ref.ports[f"bus_{net}"]
        y = bus_y[net]
        if abs(p_left.dcenter[1] - y) > 1e-6 or abs(p_right.dcenter[1] - y) > 1e-6:
            raise AssertionError(f"shared trunk {net!r} is not at the same y in both decoders")
        x0, x1 = p_left.dcenter[0] - _TRUNK_OVERLAP, p_right.dcenter[0] + _TRUNK_OVERLAP
        draw(component, m2, Box(x0, y - half2, x1, y + half2))
        if net.startswith("rdy"):
            continue  # the replica's own output, already a macro pin: the macro routes it here
        add_port(component,
            name=net, center=((x0 + x1) / 2, y), orientation=90, width=rules.w(LayerRole.M2),
            layer=m2, port_type="electrical",
        )
        label = tech.label_layer(LayerRole.M2)
        if label is not None:
            component.add_label(text=net, position=((x0 + x1) / 2, y), layer=label)

    # --- supplies: each decoder's trunks meet the extended rails of its driver column ---
    lg = logic_geometry(tech, driver_logic or logic)  # the driver columns' rails
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    rail_x = {"vdd": lg.ntap_x + tap_w / 2, "vss": lg.ptap_x + tap_w / 2}
    for (_cell, mirror, *_rest), ref in zip(cells, refs):
        for net in ("vdd", "vss"):
            port = ref.ports[net]
            px, py = port.dcenter
            rx = macro_width - rail_x[net] if mirror else rail_x[net]
            # a decoder-side trunk end, then across to the rail, then up the rail from the array
            reach_from = px + _TRUNK_OVERLAP if mirror else px - _TRUNK_OVERLAP
            x0, x1 = sorted((reach_from, rx - half1 if mirror else rx + half1))
            if net == "vdd":
                draw(component, m1, Box(x0, py - half1, x1, py + half1))
            else:
                draw(component, m2, Box(x0, py - half2, x1, py + half2))
                via_stack(component, tech, LayerRole.M1, LayerRole.M2, (rx, py))
            draw(component, m1, Box(rx - half1, array_top - _RAIL_OVERLAP, rx + half1, py + half1))
    return left_ref, right_ref
