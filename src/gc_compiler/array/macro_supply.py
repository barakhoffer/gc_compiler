"""Column-circuit placement and the macro's supply distribution.

The column circuits sit below the array, one per bit-line column, at the array's own column
positions -- not at a uniform pitch, because the array inserts strap columns every so often
and every column after one is shifted by a strap's width. Where a strap sits, the column
circuits' horizontal rails (the three vdd/vss pairs and ``pch``) are bridged across the gap.

The three vdd and three vss rails of a column circuit run across the whole array width. Each
side's driver column has a vertical vdd and vss rail; they are extended down beside the column
circuits and tied to the rails:

* vss on metal1 straight across (the vss rail is the nearer one to the array);
* vdd on metal2 from the rail's end, over the vss rail, to the outer vdd rail, so that no
  metal1 wire crosses another.

Everything else that needs vdd -- the decoders -- already reaches a driver rail, so this makes
vdd and vss single nets across the macro (the array's own ``vnw`` stays its own net, as the
netlist has it).
"""

from __future__ import annotations

from dataclasses import dataclass

import gdsfactory as gf

from gc_compiler.array.tiler import ArrayGeometry
from gc_compiler.config import ArrayConfig
from gc_compiler.periphery.column import column_circuit
from gc_compiler.periphery.logic import LogicConfig, logic_geometry
from gc_compiler.primitives.contacts import via_stack
from gc_compiler.primitives.geometry import Box, draw
from gc_compiler.tech.rules import LayerRole, Technology

_RAILS = ("pch", "vdd_0", "vss_0", "vdd_1", "vss_1", "vdd_2", "vss_2")
_HOP = 0.4
"""How far a vdd wire runs on metal1 from a rail's end before it changes to metal2."""


@dataclass(frozen=True)
class ColumnPlacement:
    """Where the column circuits went: runs of adjacent columns, ``(x, count)`` each."""

    runs: tuple[tuple[float, int], ...]
    pitch: float
    rails: tuple[tuple[str, float, float], ...]
    """``(name, y, width)`` of each horizontal rail."""

    @property
    def x0(self) -> float:
        return self.runs[0][0]

    @property
    def x1(self) -> float:
        x, count = self.runs[-1]
        return x + count * self.pitch


def place_columns(
    component: gf.Component,
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig,
    array: ArrayGeometry,
    array_x: float,
) -> ColumnPlacement:
    """One column circuit under every bit-line column, rails bridged across the strap gaps."""
    rules = tech.rules
    cell = column_circuit(tech, array.cell, logic, write_back=cfg.refresh_in_macro)
    pitch = array.cell.pitch_x
    runs: list[list[float]] = []
    for col in range(cfg.total_cols):  # the replica column, if any, is the last
        x = array_x + array.column_x(col)
        if runs and abs(x - (runs[-1][0] + runs[-1][1] * pitch)) < rules.grid / 2:
            runs[-1][1] += 1
        else:
            runs.append([x, 1])
    for x, count in runs:
        ref = component.add_ref(cell, columns=int(count), rows=1, column_pitch=pitch)
        ref.dmove((x, 0.0))

    ports = {port.name: port for port in cell.ports}
    rails = tuple((name, ports[name].dcenter[1], ports[name].dwidth) for name in _RAILS)
    if cfg.replica_column:
        # The replica column's write driver has its input tied to vss, so it holds wbl_rep -- and
        # with it every replica storage node -- at vdd. din is a metal2 track that runs up from
        # the bottom edge; the vss rail passes across it just above.
        from gc_compiler.periphery.column import column_geometry

        din_x = array_x + array.column_x(cfg.cols) + column_geometry(tech, array.cell, logic, cfg.refresh_in_macro).din_x
        vss_y = next(y for name, y, _w in rails if name == "vss_0")
        via_stack(component, tech, LayerRole.M1, LayerRole.M2, (din_x, vss_y))
    m1 = tech.layer(LayerRole.M1)
    for (x0, n0), (x1, _) in zip(runs, runs[1:]):
        end = x0 + n0 * pitch
        for _name, y, width in rails:
            draw(component, m1, Box(end - 0.1, y - width / 2, x1 + 0.1, y + width / 2))
    return ColumnPlacement(tuple((x, int(n)) for x, n in runs), pitch, rails)


def connect_supplies(
    component: gf.Component,
    tech: Technology,
    logic: LogicConfig,
    columns: ColumnPlacement,
    array_y: float,
    macro_width: float,
    x0: float = 0.0,
) -> None:
    """Tie the column circuits' rails to the driver columns' rails on both sides.

    ``macro_width`` is the width of the array plus its two driver columns, starting at ``x0``
    (a banked macro's second bank sits further right)."""
    rules = tech.rules
    m1, m2 = tech.layer(LayerRole.M1), tech.layer(LayerRole.M2)
    half1, half2 = rules.w(LayerRole.M1) / 2, rules.w(LayerRole.M2) / 2
    lg = logic_geometry(tech, logic)
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    rail_x = {"vdd": lg.ntap_x + tap_w / 2, "vss": lg.ptap_x + tap_w / 2}
    ys = {net: [y for name, y, _w in columns.rails if name.startswith(net)] for net in ("vdd", "vss")}

    for right in (False, True):
        edge = columns.x1 if right else columns.x0  # the column rails' end on this side
        sign = 1.0 if right else -1.0  # away from the array
        x_of = {net: x0 + (macro_width - x if right else x) for net, x in rail_x.items()}
        # the driver rails, extended down past the column circuits
        for net in ("vdd", "vss"):
            draw(
                component, m1,
                Box(x_of[net] - half1, min(ys[net]) - half1, x_of[net] + half1, array_y + 0.5),
            )
        for y in ys["vss"]:  # straight across on metal1
            lo, hi = sorted((x_of["vss"], edge))
            draw(component, m1, Box(lo - half1, y - half1, hi + half1, y + half1))
        for y in ys["vdd"]:  # short metal1 hop, then metal2 over the vss rail to the vdd rail
            hop = edge + sign * _HOP
            lo, hi = sorted((edge, hop))
            draw(component, m1, Box(lo - 0.05, y - half1, hi + 0.05, y + half1))
            via_stack(component, tech, LayerRole.M1, LayerRole.M2, (hop, y))
            lo, hi = sorted((hop, x_of["vdd"]))
            draw(component, m2, Box(lo, y - half2, hi, y + half2))
            via_stack(component, tech, LayerRole.M1, LayerRole.M2, (x_of["vdd"], y))
