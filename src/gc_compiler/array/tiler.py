"""Bitcell array assembly.

Rows are mirrored in pairs about their shared ``wbl`` boundary, which merges the n-well
bands and shares both bit-line contacts between vertically adjacent cells. Tap rows are
inserted at ``rbl`` boundaries every ``tap_row_interval`` pairs, and vertical strap
columns (see :mod:`gc_compiler.array.strap_column`) are inserted every
``strap_column_interval`` bitcell columns to tie every row's ``vss`` rail and every
row-pair's n-well to the tap rows' actual contacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import gdsfactory as gf

from gc_compiler.cache import cached_cell
from gc_compiler.primitives.wells import close_well_ends, fix_min_area
from gc_compiler.cells.bitcell_3t import BitcellGeometry, bitcell_geometry, gain_cell_3t
from gc_compiler.cells.tap_row import tap_row_cell, tap_row_geometry
from gc_compiler.config import ArrayConfig
from gc_compiler.periphery.wl_driver import strap_track_y, strap_width, wordline_width, wordline_y
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.array.strap_column import strap_column_geometry, strap_column_segment
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import LayerRole, Technology


@dataclass(frozen=True)
class ArrayGeometry:
    """Absolute positions of every array-level feature."""

    cell: BitcellGeometry
    rows: int
    cols: int
    width: float
    height: float
    row_base_y: tuple[float, ...]
    row_mirrored: tuple[bool, ...]
    tap_row_y: tuple[float, ...]
    col_x: tuple[float, ...]
    strap_col_x: tuple[float, ...]
    replica_col: int | None = None
    wordline_straps: bool = False
    """Every word line has a metal3 strap (see :func:`strap_track_y`), tied down in each strap column."""
    """Index into ``col_x`` of the replica column (right after the last data column)."""

    def column_x(self, col: int) -> float:
        return self.col_x[col]

    def _row_feature(self, row: int, y: float) -> float:
        base = self.row_base_y[row]
        if self.row_mirrored[row]:
            return base + 2 * self.cell.pitch_y - y
        return base + y

    def wwl_y(self, row: int) -> float:
        return self._row_feature(row, self.cell.wwl_y)

    def rwl_y(self, row: int) -> float:
        return self._row_feature(row, self.cell.rwl_y)

    def vss_y(self, row: int) -> float:
        return self._row_feature(row, self.cell.vss_y)

    def rbl_x(self, col: int) -> float:
        return self.column_x(col) + self.cell.rbl_x

    def wbl_x(self, col: int) -> float:
        return self.column_x(col) + self.cell.wbl_x

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bits(self) -> int:
        return self.rows * self.cols


def _row_pair(tech: Technology, bitcell: gf.Component, pitch_y: float) -> gf.Component:
    def build(name: str) -> gf.Component:
        pair = gf.Component(name)
        pair.add_ref(bitcell)
        mirrored = pair.add_ref(bitcell)
        mirrored.dmirror_y(pitch_y)
        pair.info["pitch_y"] = 2 * pitch_y
        return pair

    return cached_cell(f"{bitcell.name}_pair", build, tech)


def _column_layout(tech: Technology, cfg: ArrayConfig, cell: BitcellGeometry) -> tuple[list[float], list[float], float]:
    """x of every bitcell column and every strap column, interleaved: the first strap column
    half an interval in, then one every interval, and one at the end -- so no column is more than
    half an interval from one (the row pairs' n-wells and the p-strips between them are tapped
    only there, and latch-up allows 15 um)."""
    strap_width = strap_column_geometry(tech).width if cfg.strap_column_interval else 0.0
    col_x: list[float] = []
    strap_x: list[float] = []

    x = 0.0
    for col in range(cfg.total_cols):
        col_x.append(x)
        x += cell.pitch_x
        is_last = col == cfg.total_cols - 1
        interval = cfg.strap_column_interval
        if interval and not is_last and (col + 1 - interval // 2) % interval == 0 and col + 1 >= interval // 2:
            strap_x.append(x)
            x += strap_width
    if cfg.strap_column_interval:
        strap_x.append(x)
        x += strap_width
    return col_x, strap_x, x


def array_geometry(tech: Technology, cfg: ArrayConfig) -> ArrayGeometry:
    """Plan the vertical stacking of row pairs and tap rows."""
    cell = bitcell_geometry(tech, cfg.bitcell)
    tap_height = tap_row_geometry(tech, cell).height
    pairs = cfg.rows // 2
    col_x, strap_x, width = _column_layout(tech, cfg, cell)

    base_y: list[float] = []
    mirrored: list[bool] = []
    tap_y: list[float] = []

    y = 0.0
    if cfg.edge_tap_rows:
        tap_y.append(y)
        y += tap_height
    for pair in range(pairs):
        base_y += [y, y]
        mirrored += [False, True]
        y += 2 * cell.pitch_y
        last = pair == pairs - 1
        if not last and cfg.tap_row_interval and (pair + 1) % cfg.tap_row_interval == 0:
            tap_y.append(y)
            y += tap_height
    if cfg.edge_tap_rows:
        tap_y.append(y)
        y += tap_height

    return ArrayGeometry(
        cell=cell,
        rows=cfg.rows,
        cols=cfg.cols,
        width=width,
        height=y,
        row_base_y=tuple(base_y),
        row_mirrored=tuple(mirrored),
        tap_row_y=tuple(tap_y),
        col_x=tuple(col_x),
        strap_col_x=tuple(strap_x),
        replica_col=cfg.cols if cfg.replica_column else None,
        wordline_straps=cfg.wordline_straps,
    )


def _array_cell_name(cfg: ArrayConfig) -> str:
    spacing = (cfg.tap_row_interval, cfg.edge_tap_rows, cfg.strap_column_interval)
    fields = ArrayConfig.__dataclass_fields__
    if spacing == tuple(fields[f].default for f in ("tap_row_interval", "edge_tap_rows", "strap_column_interval")):
        return cfg.name
    return f"{cfg.name}_t{cfg.tap_row_interval}{'' if cfg.edge_tap_rows else 'n'}_s{cfg.strap_column_interval}"


def gc_array(
    tech: Technology,
    cfg: ArrayConfig | None = None,
    name: str | None = None,
) -> gf.Component:
    """Build a ``rows`` x ``cols`` gain-cell array.

    Without a ``name`` the cell is cached (a macro reuses its array, so two macros of one
    configuration -- say one bank and two -- can be built in one session); its name is the
    configuration's, plus the tap-row and strap-column spacing when those are not the defaults."""
    cfg = cfg or ArrayConfig.for_tech(tech)
    if name is None:
        return cached_cell(_array_cell_name(cfg), lambda cell_name: gc_array(tech, cfg, cell_name), tech)
    activate(tech)
    geo = array_geometry(tech, cfg)

    bitcell = gain_cell_3t(tech, cfg.bitcell)
    pair = _row_pair(tech, bitcell, geo.cell.pitch_y)
    replica_pair = (
        _row_pair(tech, gain_cell_3t(tech, cfg.bitcell, replica=True), geo.cell.pitch_y)
        if cfg.replica_column
        else None
    )
    tap = tap_row_cell(tech, geo.cell)

    component = gf.Component(name or cfg.name)
    for pair_index in range(cfg.rows // 2):
        y = geo.row_base_y[2 * pair_index]
        for col, x in enumerate(geo.col_x):
            ref = component.add_ref(replica_pair if col == geo.replica_col else pair)
            ref.dmove((x, y))
    for y in geo.tap_row_y:
        for x in geo.col_x:
            ref = component.add_ref(tap)
            ref.dmove((x, y))

    _add_strap_columns(component, tech, geo)
    if geo.wordline_straps:
        _add_wordline_straps(component, tech, geo)
    _add_ports(component, tech, geo)
    close_well_ends(component, tech)  # the rows' wells end flush with their end taps
    fix_min_area(component, tech)
    component.info["rows"] = cfg.rows
    component.info["cols"] = cfg.cols
    component.info["bits"] = geo.bits
    component.info["width"] = geo.width
    component.info["height"] = geo.height
    component.info["area_per_bit"] = geo.area / geo.bits
    return component


def _add_strap_columns(component: gf.Component, tech: Technology, geo: ArrayGeometry) -> None:
    """Vertical vss/n-well straps: one metal2 pair per strap column, full array height."""
    if not geo.strap_col_x:
        return
    sgeo = strap_column_geometry(tech)
    wl = wordline_y(geo.cell)
    pair_h = 2 * geo.cell.pitch_y
    straps = strap_track_y(tech, geo.cell) if geo.wordline_straps else {}

    pair_seg = strap_column_segment(
        tech,
        sgeo,
        pair_h,
        vss_bridge_y=(geo.cell.vss_y, pair_h - geo.cell.vss_y),
        nwell_taps=((geo.cell.nwell_y0, pair_h - geo.cell.nwell_y0),),
        poly_y=tuple(wl.values()),
        poly_widths=tuple(wordline_width(geo.cell)[net] for net in wl),
        strap_y=tuple(straps[net] for net in wl) if straps else (),
        name=f"gc_strap_pair_{pair_h:g}_w{geo.cell.rwl_width:g}_{geo.cell.wwl_width:g}" + ("_ws" if straps else ""),
    )
    tap_geo = tap_row_geometry(tech, geo.cell)
    tap_seg = strap_column_segment(
        tech,
        sgeo,
        tap_geo.height,
        tap_row_bridges=((tap_geo.ptap_y, "vss"), (tap_geo.ntap_y, "vnw")),
        tap_row_nwell=(tap_geo.nwell_y0, tap_geo.nwell_y1),
        name=f"gc_strap_tap_{tap_geo.height:g}",
    )

    for sx in geo.strap_col_x:
        for pair_index in range(geo.rows // 2):
            ref = component.add_ref(pair_seg)
            ref.dmove((sx, geo.row_base_y[2 * pair_index]))
        for y in geo.tap_row_y:
            ref = component.add_ref(tap_seg)
            ref.dmove((sx, y))


def _add_wordline_straps(component: gf.Component, tech: Technology, geo: ArrayGeometry) -> None:
    """One metal3 line per word line, the array's full width (its drivers continue it outward)."""
    half = strap_width(tech) / 2
    m3 = tech.layer(LayerRole.M3)
    straps = strap_track_y(tech, geo.cell)
    for pair_index in range(geo.rows // 2):
        base = geo.row_base_y[2 * pair_index]
        for y in straps.values():
            draw(component, m3, Box(0.0, base + y - half, geo.width, base + y + half))


def _add_ports(component: gf.Component, tech: Technology, geo: ArrayGeometry) -> None:
    rules = tech.rules

    def add(name: str, center, orientation: float, role: LayerRole, width: float, label: str | None = None) -> None:
        add_port(component,
            name=name,
            center=center,
            orientation=orientation,
            width=width,
            layer=tech.layer(role),
            port_type="electrical",
        )
        label_layer = tech.label_layer(role)
        if label_layer is not None:
            component.add_label(text=label or name, position=center, layer=label_layer)

    for row in range(geo.rows):
        add(f"wwl_{row}", (0.0, geo.wwl_y(row)), 180, LayerRole.POLY, rules.w(LayerRole.POLY))
        add(f"rwl_{row}", (0.0, geo.rwl_y(row)), 180, LayerRole.POLY, rules.w(LayerRole.POLY))
        # Same label text everywhere: once the strap columns tie these together, they are
        # electrically one net, matching the single "vss" pin in the generated netlist.
        add(f"vss_{row}", (0.0, geo.vss_y(row)), 180, LayerRole.M1, rules.w(LayerRole.M1), label="vss")
    for col in range(geo.cols):
        add(f"wbl_{col}", (geo.wbl_x(col), geo.height), 90, LayerRole.M2, rules.w(LayerRole.M2))
        add(f"rbl_{col}", (geo.rbl_x(col), 0.0), 270, LayerRole.M2, rules.w(LayerRole.M2))
    if geo.replica_col is not None:
        add("wbl_rep", (geo.wbl_x(geo.replica_col), geo.height), 90, LayerRole.M2, rules.w(LayerRole.M2))
        add("rbl_rep", (geo.rbl_x(geo.replica_col), 0.0), 270, LayerRole.M2, rules.w(LayerRole.M2))
