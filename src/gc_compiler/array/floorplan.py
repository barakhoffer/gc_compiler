"""Macro floorplan: array, word-line drivers on both sides, column circuits below.

With ``decoded=True`` a row decoder is placed beyond each driver column too
(see :mod:`gc_compiler.array.wordline_decoder`) and wired to the driver inputs, so the
macro takes ``addr``/``we``/``re`` instead of the raw per-row ``wwl_in``/``rwl_in``.
"""

from __future__ import annotations

from dataclasses import dataclass

import gdsfactory as gf

from gc_compiler.primitives.wells import close_implant_ends, close_well_ends, fill_notches, fix_min_area
from gc_compiler.array.tiler import ArrayGeometry, array_geometry, gc_array
from gc_compiler.array.macro_supply import ColumnPlacement, connect_supplies, place_columns
from gc_compiler.array.wordline_decoder import place_decoders, plan_central_decoder, wire_central_decoder
from gc_compiler.config import ArrayConfig
from gc_compiler.periphery.column import column_geometry
from gc_compiler.periphery.logic import predecode_logic as _predecode
from gc_compiler.periphery.logic import (
    LevelShifterConfig,
    LogicConfig,
    driver_logic,
    logic_geometry,
)
from gc_compiler.periphery.wl_driver import (
    LEFT_WORDLINES,
    RIGHT_WORDLINES,
    wl_driver_geometry,
    wl_driver_pair,
)
from gc_compiler.periphery.wl_driver_neg import (
    neg_driver_geometry,
    neg_wl_driver_geometry,
    wl_driver_pair_neg,
)
from gc_compiler.primitives.contacts import via_stack
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import LayerRole, Technology


@dataclass(frozen=True)
class MacroGeometry:
    array: ArrayGeometry
    driver_width: float
    column_height: float
    width: float
    height: float
    array_x: float
    array_y: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def efficiency(self) -> float:
        return self.array.cell.area * self.array.bits / self.area


def macro_geometry(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig | None = None,
    ls: LevelShifterConfig | None = None,
) -> MacroGeometry:
    if cfg.negative_write_wl and not tech.has(LayerRole.DNWELL):
        raise ValueError(
            f"negative_write_wl needs an isolated p-well (a deep n-well) for the driver's "
            f"negative-rail NMOS, which {tech.name} does not have"
        )
    logic = logic or LogicConfig.for_tech(tech)
    array = array_geometry(tech, cfg)
    drv = driver_logic(logic, cfg.wordline_driver_strength, tech)
    if cfg.negative_write_wl:
        driver = neg_wl_driver_geometry(tech, array.cell, drv, ls or LevelShifterConfig.for_tech(tech), LEFT_WORDLINES)
    else:
        driver = wl_driver_geometry(tech, array.cell, drv)
    column = column_geometry(tech, array.cell, logic, cfg.refresh_in_macro)
    return MacroGeometry(
        array=array,
        driver_width=driver.width,
        column_height=column.height,
        width=2 * driver.width + array.width,
        height=array.height + column.height,
        array_x=driver.width,
        array_y=column.height,
    )


def _place_drivers(
    component: gf.Component,
    tech: Technology,
    geo: MacroGeometry,
    logic: LogicConfig,
    cells: tuple[gf.Component, gf.Component] | None = None,
    ls: LevelShifterConfig | None = None,
    x0: float = 0.0,
    straps: bool = False,
) -> list:
    """Both driver columns; returns ``(pair_index, left_ref, right_ref)`` per mirrored row pair.

    ``cells`` substitutes the left/right slice cells (they need the same ports); it exists
    so the wiring around the drivers can be checked without what is inside them. ``ls`` (the
    level shifter's sizing) selects the negative-rail write drivers instead of plain inverters.
    ``x0`` shifts the whole thing right (a banked macro's second bank). ``straps`` feeds each
    word line's metal3 strap from its driver.
    """
    if cells is None and ls is not None:
        cells = (
            wl_driver_pair_neg(tech, geo.array.cell, logic, ls, LEFT_WORDLINES, straps=straps),
            wl_driver_pair_neg(tech, geo.array.cell, logic, ls, RIGHT_WORDLINES, straps=straps),
        )
    left, right = cells or (
        wl_driver_pair(tech, geo.array.cell, logic, LEFT_WORDLINES, straps=straps),
        wl_driver_pair(tech, geo.array.cell, logic, RIGHT_WORDLINES, straps=straps),
    )
    drivers = []
    for pair_index, base in enumerate(geo.array.row_base_y[::2]):
        ref = component.add_ref(left)
        ref.dmove((x0, geo.array_y + base))
        mirrored = component.add_ref(right)
        mirrored.dmirror_x()
        mirrored.dmove((x0 + geo.array_x + geo.array.width + geo.driver_width, geo.array_y + base))
        drivers.append((pair_index, ref, mirrored))
    return drivers


def wordline_periphery(
    tech: Technology,
    cfg: ArrayConfig | None = None,
    logic: LogicConfig | None = None,
    name: str | None = None,
    driver_cells: tuple[gf.Component, gf.Component] | None = None,
    ls: LevelShifterConfig | None = None,
) -> tuple[gf.Component, list, tuple[gf.Instance, gf.Instance]]:
    """The word-line drivers and both half-decoders exactly as :func:`gc_macro` places them,
    without the array or the column circuits -- the part of a decoded macro whose wiring
    is checkable on its own (the drivers' outputs are left as poly stubs).

    ``driver_cells`` substitutes the slice cells, as in :func:`_place_drivers`. Returns the
    component, the driver instances ``(pair_index, left, right)`` and the ``(left, right)``
    decoder instances.
    """
    cfg = cfg or ArrayConfig.for_tech(tech)
    logic = logic or LogicConfig.for_tech(tech)
    ls = (ls or LevelShifterConfig.for_tech(tech)) if cfg.negative_write_wl else None
    activate(tech)
    geo = macro_geometry(tech, cfg, logic, ls)
    drv = driver_logic(logic, cfg.wordline_driver_strength, tech)
    component = gf.Component(name or f"{cfg.name}_wordline_periphery")
    drivers = _place_drivers(component, tech, geo, drv, driver_cells, ls, straps=cfg.wordline_straps)
    _supply_straps(component, tech, geo, drv)
    if ls is not None:
        _negative_rail(component, tech, geo, drv, ls)
    decoders = place_decoders(component, tech, cfg, logic, geo.array, geo.array_y, geo.width, drivers, drv)
    close_well_ends(component, tech)
    close_implant_ends(component, tech)
    fill_notches(component, tech)  # same-net shapes of abutting cells (stacked slices' tap pads)
    fix_min_area(component, tech)
    return component, drivers, decoders


def gc_macro(
    tech: Technology,
    cfg: ArrayConfig | None = None,
    logic: LogicConfig | None = None,
    name: str | None = None,
    decoded: bool = False,
    ls: LevelShifterConfig | None = None,
) -> gf.Component:
    """Full memory macro: array plus word-line drivers and column circuits.

    With ``cfg.negative_write_wl`` the write drivers are negative-rail drivers (see
    :mod:`gc_compiler.periphery.wl_driver_neg`), and the macro gains a ``vneg`` pin.

    ``decoded`` also places and wires a row decoder on each side; the macro's ports are
    then ``addr_*``/``we``/``re`` rather than the per-row ``wwl_in_*``/``rwl_in_*``.
    ``info['width']``/``['height']`` then include the decoders (``macro_geometry`` does not).
    """
    cfg = cfg or ArrayConfig.for_tech(tech)
    logic = logic or LogicConfig.for_tech(tech)
    ls = (ls or LevelShifterConfig.for_tech(tech)) if cfg.negative_write_wl else None
    activate(tech)
    geo = macro_geometry(tech, cfg, logic, ls)

    component = gf.Component(name or f"{cfg.name}_macro")
    array = component.add_ref(gc_array(tech, cfg))
    array.dmove((geo.array_x, geo.array_y))

    drv = driver_logic(logic, cfg.wordline_driver_strength, tech)
    drivers = _place_drivers(component, tech, geo, drv, ls=ls, straps=cfg.wordline_straps)

    columns = place_columns(component, tech, cfg, logic, geo.array, geo.array_x)

    _supply_straps(component, tech, geo, drv)
    if ls is not None:
        _negative_rail(component, tech, geo, drv, ls)
    connect_supplies(component, tech, drv, columns, geo.array_y, geo.width)
    _add_ports(component, tech, geo, cfg, logic, drivers, columns, wordline_inputs=not decoded, driver=drv)
    # self_timed_read gates the decoder's read enable, so without a decoder there is nothing to
    # gate: the macro's rwl_in pins are driven directly and rdy is only an output
    if decoded:
        decoders = place_decoders(component, tech, cfg, logic, geo.array, geo.array_y, geo.width, drivers, drv)
        if cfg.self_timed_read:
            _route_rdy_around(component, tech, decoders[1].ports["bus_rdy"], component.ports["rdy"])

    close_well_ends(component, tech)  # the strips' wells end flush with their end taps
    close_implant_ends(component, tech)
    fill_notches(component, tech)  # same-net shapes of abutting cells (stacked slices' tap pads)
    fix_min_area(component, tech)
    bounds = component.dbbox()
    width, height = (bounds.width(), bounds.height()) if decoded else (geo.width, geo.height)
    component.info["width"] = width
    component.info["height"] = height
    component.info["bits"] = geo.array.bits
    component.info["area_per_bit"] = width * height / geo.array.bits
    component.info["array_efficiency"] = geo.array.cell.area * geo.array.bits / (width * height)
    component.info["decoded"] = decoded
    return component


def gc_banked_macro(
    tech: Technology,
    cfg: ArrayConfig | None = None,
    logic: LogicConfig | None = None,
    name: str | None = None,
    ls: LevelShifterConfig | None = None,
    predecode_logic: LogicConfig | None = None,
) -> gf.Component:
    """Two banks of ``cfg`` around one central decoder (always decoded).

    Each bank is exactly a plain macro's array, driver columns and column circuits; the
    decoder between them (:func:`~gc_compiler.array.wordline_decoder.central_decoder`) takes
    one more address bit than a bank needs -- the top bit picks the bank. Pins: ``addr_*``,
    ``we``/``re``, ``din_{bank}_{col}``/``dout_{bank}_{col}``, ``rdy_{bank}`` with a replica,
    and one shared ``pch``/``vdd``/``vss`` (and ``vneg``). ``info`` is as for :func:`gc_macro`,
    with ``bits`` counting both banks. ``predecode_logic`` sizes the shared predecode (default
    :func:`~gc_compiler.periphery.logic.predecode_logic`); the netlist must use the same.
    """
    cfg = cfg or ArrayConfig.for_tech(tech)
    logic = logic or LogicConfig.for_tech(tech)
    ls = (ls or LevelShifterConfig.for_tech(tech)) if cfg.negative_write_wl else None
    activate(tech)
    geo = macro_geometry(tech, cfg, logic, ls)
    drv = driver_logic(logic, cfg.wordline_driver_strength, tech)
    component = gf.Component(name or f"{cfg.name}_banked_macro")
    array = gc_array(tech, cfg)

    def place_bank(bank: int, x0: float):
        component.add_ref(array).dmove((x0 + geo.array_x, geo.array_y))
        drivers = _place_drivers(component, tech, geo, drv, ls=ls, x0=x0, straps=cfg.wordline_straps)
        columns = place_columns(component, tech, cfg, logic, geo.array, x0 + geo.array_x)
        _supply_straps(component, tech, geo, drv, x0)
        link = _negative_rail(component, tech, geo, drv, ls, x0, port=False) if ls is not None else None
        connect_supplies(component, tech, drv, columns, geo.array_y, geo.width, x0)
        _add_ports(
            component, tech, geo, cfg, logic, drivers, columns, wordline_inputs=False, offset=x0, bank=bank,
            driver=drv,
        )
        return drivers, columns, link

    drivers0, columns0, link0 = place_bank(0, 0.0)
    plan = plan_central_decoder(
        tech, cfg, logic, geo.array, geo.array_y, drivers0, predecode_logic or _predecode(tech)
    )
    drivers1, columns1, link1 = place_bank(1, plan.bank_x)
    decoder = wire_central_decoder(component, tech, drv, plan, geo.array, geo.array_y, geo.width, (drivers0, drivers1))
    if cfg.self_timed_read:
        # each bank's hop runs above the metal3 that joins the banks' pch (and well below the
        # first row's word-line straps), bank 1's a metal3 pitch above bank 0's
        pch_y = next(y for name, y, _w in columns0.rails if name == "pch")
        pitch3 = tech.rules.w(LayerRole.M3) + tech.rules.s(LayerRole.M3)
        ends = [decoder.ports[f"trunk_rdy_{bank}_{end}"].dcenter for bank, end in ((0, "left"), (1, "right"))]
        # bank 0 leaves to the left of both trunk ends, bank 1 to the right, so their metal3 never
        # crosses (where the predecode's gates put the two ends is its own business)
        (x0, _), (x1, _) = ends
        exits = (x0, x1) if x0 < x1 else (x1 - pitch3, x0 + pitch3)
        for bank in (0, 1):
            hop_y = tech.rules.snap(pch_y + (1 + bank) * pitch3 + 0.1)
            _route_rdy_under(
                component, tech, ends[bank], tech.rules.snap(exits[bank]), component.ports[f"rdy_{bank}"], hop_y
            )
    _join_banks(component, tech, columns0, columns1, link0, link1)
    close_well_ends(component, tech)  # the strips' wells end flush with their end taps
    close_implant_ends(component, tech)
    fill_notches(component, tech)  # same-net shapes of abutting cells (stacked slices' tap pads)
    fix_min_area(component, tech)

    width, height = component.dbbox().width(), component.dbbox().height()
    bits = 2 * geo.array.bits
    component.info["width"] = width
    component.info["height"] = height
    component.info["bits"] = bits
    component.info["area_per_bit"] = width * height / bits
    component.info["array_efficiency"] = geo.array.cell.area * bits / (width * height)
    component.info["decoded"] = True
    component.info["banked"] = True
    component.info["bank_x"] = plan.bank_x
    return component


_RDY_GAP = 0.5
"""Clearance from the macro's outline of the metal3 that brings ``rdy`` round to the decoders: half
a via2 landing (0.185) plus metal3 spacing (0.3), so the landing clears whatever metal reaches the
outline."""


def _route_rdy_around(component: gf.Component, tech: Technology, trunk, pin) -> None:
    """Join the replica's ``rdy`` (a metal2 track leaving the column band's bottom edge) to the
    decoders' ``rdy`` trunk line across the top of the macro, for a self-timed read.

    Over the array metal3 is taken (word-line straps), so the route goes round: the trunk line
    runs on past the right decoder, drops on metal3 outside the macro's right edge, runs back
    under its bottom edge, and comes up into the replica's track.
    """
    rules = tech.rules
    m2, m3 = tech.layer(LayerRole.M2), tech.layer(LayerRole.M3)
    half2, half3 = rules.w(LayerRole.M2) / 2, rules.w(LayerRole.M3) / 2
    box = component.dbbox()
    x_edge = rules.snap(box.right + _RDY_GAP)
    y_low = rules.snap(box.bottom - _RDY_GAP)
    (tx, ty), (px, py) = trunk.dcenter, pin.dcenter
    draw(component, m2, Box(tx - half2, ty - half2, x_edge + half2, ty + half2))
    via_stack(component, tech, LayerRole.M2, LayerRole.M3, (x_edge, ty))
    draw(component, m3, Box(x_edge - half3, y_low - half3, x_edge + half3, ty + half3))
    draw(component, m3, Box(px - half3, y_low - half3, x_edge + half3, y_low + half3))
    via_stack(component, tech, LayerRole.M2, LayerRole.M3, (px, y_low))
    draw(component, m2, Box(px - half2, y_low, px + half2, py + half2))


def _route_rdy_under(
    component: gf.Component, tech: Technology, end: tuple[float, float], exit_x: float, pin, hop_y: float
) -> None:
    """A banked macro's self-timed read: join one bank's ``rdy`` (a metal2 track leaving its
    column band's bottom edge) to its trunk beside the central decoder. The trunk's metal2 line
    runs on from its ``end`` to ``exit_x``; metal3 from there down to ``hop_y``, along to the bank's
    replica column, down through that bank's column band (clear of the ``pch`` bridge, which only
    spans the gap between the banks), and up into the track from below the macro's edge."""
    rules = tech.rules
    m2, m3 = tech.layer(LayerRole.M2), tech.layer(LayerRole.M3)
    half2, half3 = rules.w(LayerRole.M2) / 2, rules.w(LayerRole.M3) / 2
    y_low = rules.snap(component.dbbox().bottom - _RDY_GAP)
    (ex, ty), (px, py) = end, pin.dcenter
    tx = exit_x
    draw(component, m2, Box(min(ex, tx) - half2, ty - half2, max(ex, tx) + half2, ty + half2))
    via_stack(component, tech, LayerRole.M2, LayerRole.M3, (tx, ty))
    draw(component, m3, Box(tx - half3, min(ty, hop_y) - half3, tx + half3, max(ty, hop_y) + half3))
    draw(component, m3, Box(min(tx, px) - half3, hop_y - half3, max(tx, px) + half3, hop_y + half3))
    draw(component, m3, Box(px - half3, y_low - half3, px + half3, hop_y + half3))
    via_stack(component, tech, LayerRole.M2, LayerRole.M3, (px, y_low))
    draw(component, m2, Box(px - half2, y_low, px + half2, py + half2))


_PCH_REACH = 0.6
"""How far the precharge rail is extended past a bank's column circuits to its bridge via."""


def _join_banks(component, tech, columns0, columns1, link0, link1) -> None:
    """The nets the two banks share beyond the decoder: ``pch`` (and ``vneg``).

    The precharge rails meet on metal3 under the decoder: at the rail's height the driver
    columns' supply rails run down past the column circuits on metal1, and the column
    circuits have metal2 all along it, so each rail is extended on metal1 into the free
    space just past its bank's column circuits and lifted there.
    """
    rules = tech.rules
    m1, m3 = tech.layer(LayerRole.M1), tech.layer(LayerRole.M3)
    half3 = rules.w(LayerRole.M3) / 2
    pch_y, pch_w = next((y, w) for name, y, w in columns0.rails if name == "pch")
    lift0, lift1 = columns0.x1 + _PCH_REACH - 0.2, columns1.x0 - _PCH_REACH + 0.2
    draw(component, m1, Box(columns0.x1 - 0.05, pch_y - pch_w / 2, columns0.x1 + _PCH_REACH, pch_y + pch_w / 2))
    draw(component, m1, Box(columns1.x0 - _PCH_REACH, pch_y - pch_w / 2, columns1.x0 + 0.05, pch_y + pch_w / 2))
    for x in (lift0, lift1):
        via_stack(component, tech, LayerRole.M1, LayerRole.M3, (x, pch_y))
    draw(component, m3, Box(lift0 - half3, pch_y - half3, lift1 + half3, pch_y + half3))
    if link0 is None:
        return
    (_l0, right0), y = link0
    (left1, _r1), _y = link1
    draw(component, m3, Box(right0, y - half3, left1, y + half3))
    mid = (right0 + left1) / 2
    add_port(component,
        name="vneg", center=(mid, y), orientation=90, width=rules.w(LayerRole.M3), layer=m3,
        port_type="electrical",
    )
    label = tech.label_layer(LayerRole.M3)
    if label is not None:
        component.add_label(text="vneg", position=(mid, y), layer=label)


def _negative_rail(
    component: gf.Component,
    tech: Technology,
    geo: MacroGeometry,
    logic: LogicConfig,
    ls: LevelShifterConfig,
    x0: float = 0.0,
    port: bool = True,
) -> tuple[tuple[float, float], float]:
    """What a column of negative-rail drivers needs beyond its slices.

    * The isolated p-well's n-well ring and deep n-well are continuous over the whole column --
      including the tap-row gaps between driver slices, which would otherwise split the well into
      islands closer than the deep n-well spacing -- and closed at both ends by a cap.
    * The two columns' ``vneg`` rails are joined over the top of the array on metal3 (the
      region under the strips is taken by the column circuits' supply rails), where the macro's
      ``vneg`` pin is (unless ``port`` is off).

    Returns the metal3 link's ends ``(x_left, x_right)`` and its y.
    """
    rules = tech.rules
    ng = neg_driver_geometry(tech, geo.array.cell, logic, ls)
    y0, y1 = geo.array_y, geo.array_y + geo.array.height
    ring, inner = rules.isolation_ring_width, rules.isolation_ring_inner
    nwell, dnwell = tech.layer(LayerRole.NWELL), tech.layer(LayerRole.DNWELL)
    m1, m3 = tech.layer(LayerRole.M1), tech.layer(LayerRole.M3)
    half1 = rules.w(LayerRole.M1) / 2
    link_y = rules.snap(y1 + 0.5)
    rail = ng.ptap_x + ng.tap_w / 2

    xs = []
    for mirror in (False, True):

        def x(a: float) -> float:
            return x0 + (geo.width - a if mirror else a)

        def box(xa: float, xb: float, ya: float, yb: float) -> Box:
            return Box(min(x(xa), x(xb)), ya, max(x(xa), x(xb)), yb)

        for xa, xb in ((ng.g0, ng.i0), (ng.i1, ng.wall_x1)):  # the ring's side walls
            draw(component, nwell, box(xa, xb, y0, y1))
        draw(component, dnwell, box(ng.i0 - inner, ng.i1 + inner, y0 - inner, y1 + inner))
        for ya, yb in ((y1, y1 + ring), (y0 - ring, y0)):  # the caps closing the ring
            draw(component, nwell, box(ng.i0 - ring, ng.wall_x1, ya, yb))
        xs.append(x(rail))
        draw(component, m1, Box(xs[-1] - half1, y0, xs[-1] + half1, link_y))
    via_stack(component, tech, LayerRole.M1, LayerRole.M3, (xs[0], link_y))
    via_stack(component, tech, LayerRole.M1, LayerRole.M3, (xs[1], link_y))
    half3 = rules.w(LayerRole.M3) / 2
    draw(component, m3, Box(xs[0] - half3, link_y - half3, xs[1] + half3, link_y + half3))
    if port:
        mid = (xs[0] + xs[1]) / 2
        add_port(component,
            name="vneg", center=(mid, link_y), orientation=90, width=rules.w(LayerRole.M3),
            layer=m3, port_type="electrical",
        )
        label = tech.label_layer(LayerRole.M3)
        if label is not None:
            component.add_label(text="vneg", position=(mid, link_y), layer=label)
    return (xs[0], xs[1]), link_y


def _supply_straps(
    component: gf.Component, tech: Technology, geo: MacroGeometry, cfg: LogicConfig, x0: float = 0.0
) -> None:
    """Continuous metal1 rails over the driver taps, bridging the tap-row gaps."""
    rules = tech.rules
    logic = logic_geometry(tech, cfg)
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    width = rules.w(LayerRole.M1)
    y0, y1 = geo.array_y, geo.array_y + geo.array.height
    for x in (logic.ntap_x + tap_w / 2, logic.ptap_x + tap_w / 2):
        mirror_x = x0 + geo.width - x
        x = x0 + x
        draw(component, tech.layer(LayerRole.M1), Box(x - width / 2, y0, x + width / 2, y1))
        draw(
            component,
            tech.layer(LayerRole.M1),
            Box(mirror_x - width / 2, y0, mirror_x + width / 2, y1),
        )


def _add_ports(
    component: gf.Component,
    tech: Technology,
    geo: MacroGeometry,
    cfg: ArrayConfig,
    logic: LogicConfig,
    drivers: list,
    columns: ColumnPlacement,
    wordline_inputs: bool = True,
    offset: float = 0.0,
    bank: int | None = None,
    driver: LogicConfig | None = None,
) -> None:
    """The macro's pins. ``bank`` (with its ``offset``) names one bank's data pins of a banked
    macro (``din_{bank}_{col}``, ``rdy_{bank}``); ``pch`` and the supplies are bank 0's.
    ``driver``: the word-line drivers' sizes (default ``logic``), whose rails carry the supplies."""
    rules = tech.rules
    column = column_geometry(tech, geo.array.cell, logic, cfg.refresh_in_macro)
    m2_label = tech.label_layer(LayerRole.M2)
    m1_label_layer = tech.label_layer(LayerRole.M1)
    tag = "" if bank is None else f"_{bank}"

    for col in range(cfg.cols):
        x0 = offset + geo.array_x + geo.array.column_x(col)
        for net, x in (("din", column.din_x), ("dout", column.dout_x)):
            name = f"{net}{tag}_{col}"
            add_port(component,
                name=name,
                center=(x0 + x, 0.0),
                orientation=270,
                width=rules.w(LayerRole.M2),
                layer=tech.layer(LayerRole.M2),
                port_type="electrical",
            )
            if m2_label is not None:
                component.add_label(text=name, position=(x0 + x, 0.0), layer=m2_label)

    if cfg.replica_column:  # the replica detector's output, next to the last data column's
        x = offset + geo.array_x + geo.array.column_x(cfg.cols) + column.dout_x
        add_port(component,
            name=f"rdy{tag}", center=(x, 0.0), orientation=270, width=rules.w(LayerRole.M2),
            layer=tech.layer(LayerRole.M2), port_type="electrical",
        )
        if m2_label is not None:
            component.add_label(text=f"rdy{tag}", position=(x, 0.0), layer=m2_label)

    if bank:  # bank 1 of a banked macro: pch and the supplies are already bank 0's pins
        return
    # the shared precharge gate rail, and the supplies, named where they leave the macro
    pch_y, pch_w = next((y, w) for name, y, w in columns.rails if name == "pch")
    add_port(component,
        name="pch", center=(columns.x0, pch_y), orientation=180, width=pch_w,
        layer=tech.layer(LayerRole.M1), port_type="electrical",
    )
    if m1_label_layer is not None:
        component.add_label(text="pch", position=(columns.x0, pch_y), layer=m1_label_layer)
        lg = logic_geometry(tech, driver or logic)
        tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
        for net, x in (("vdd", lg.ntap_x + tap_w / 2), ("vss", lg.ptap_x + tap_w / 2)):
            y = geo.array_y + 1.0
            component.add_label(text=net, position=(x, y), layer=m1_label_layer)
            add_port(component,
                name=net, center=(x, y), orientation=180, width=rules.w(LayerRole.M1),
                layer=tech.layer(LayerRole.M1), port_type="electrical",
            )

    # Each driver strip covers one mirrored row pair; its own "rwl_0"/"wwl_1"-style port
    # names are relative to that pair, so translate them to absolute row indices here.
    m1_label = tech.label_layer(LayerRole.M2)
    for pair_index, left_ref, right_ref in drivers if wordline_inputs else ():
        for ref in (left_ref, right_ref):
            port_names = {p.name for p in ref.ports}
            for net in LEFT_WORDLINES + RIGHT_WORDLINES:
                port_name = f"{net}_in"
                if port_name not in port_names:
                    continue
                signal, relative_row = net.rsplit("_", 1)
                row = 2 * pair_index + int(relative_row)
                name = f"{signal}_in_{row}"
                port = ref.ports[port_name]
                center = tuple(port.dcenter)
                add_port(component,
                    name=name,
                    center=center,
                    orientation=port.orientation,
                    width=port.width,
                    layer=port.layer,
                    port_type="electrical",
                )
                if m1_label is not None:
                    component.add_label(text=name, position=center, layer=m1_label)
