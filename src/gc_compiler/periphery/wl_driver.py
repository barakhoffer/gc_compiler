"""Word-line driver strip.

Word lines are landed on alternating sides of the array: sorted by y, every other word
line is driven from the left strip and the rest from the right one. Two word lines of a
mirrored row pair are only ~0.4 um apart, which leaves no room for a poly landing pad plus
the stripe that has to run past it, so single-sided landing is not legal at this pitch.

Each strip cell covers one mirrored row pair and drives two of its four word lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import gdsfactory as gf

from gc_compiler.cache import cached_cell
from gc_compiler.cells.bitcell_3t import BitcellGeometry
from gc_compiler.periphery.logic import LogicConfig, inverter, logic_geometry
from gc_compiler.primitives.wells import fix_min_area
from gc_compiler.primitives.contacts import place_cuts, via_stack
from gc_compiler.primitives.devices import gate_pad_size
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import LayerRole, Technology

#: Word lines of a mirrored row pair, bottom to top.
WORDLINES = ("rwl_0", "wwl_0", "wwl_1", "rwl_1")
#: Word lines landed by the left strip and by the right strip.
LEFT_WORDLINES = ("rwl_0", "wwl_1")
RIGHT_WORDLINES = ("wwl_0", "rwl_1")


def wordline_y(cell: BitcellGeometry) -> dict[str, float]:
    """Absolute y of the four word lines of a mirrored row pair."""
    pair = 2 * cell.pitch_y
    return {
        "rwl_0": cell.rwl_y,
        "wwl_0": cell.wwl_y,
        "wwl_1": pair - cell.wwl_y,
        "rwl_1": pair - cell.rwl_y,
    }


def wordline_width(cell: BitcellGeometry) -> dict[str, float]:
    """Poly width of each word line, so the landing stub meets it flush."""
    return {
        "rwl_0": cell.rwl_width,
        "rwl_1": cell.rwl_width,
        "wwl_0": cell.wwl_width,
        "wwl_1": cell.wwl_width,
    }


def strap_width(tech: Technology) -> float:
    """Width of a metal3 word-line strap: as wide as a via2 landing, so the landings on it
    need no pads of their own (which would poke out toward the next strap)."""
    rules = tech.rules
    return rules.snap(max(rules.w(LayerRole.M3), rules.cut(LayerRole.VIA2) + 2 * rules.enc(LayerRole.M3, LayerRole.VIA2)))


def strap_track_y(tech: Technology, cell: BitcellGeometry) -> dict[str, float]:
    """y of each word line's metal3 strap within a mirrored row pair.

    A strap sits over its word line where it can; the two write lines of a pair, and a read line
    and the next pair's, are closer than a strap pitch, so those straps are pushed apart,
    symmetrically about the pair's middle and its edges.
    """
    rules = tech.rules
    half_pitch = (strap_width(tech) + rules.s(LayerRole.M3)) / 2
    pair = 2 * cell.pitch_y
    rwl = rules.snap(max(cell.rwl_y, half_pitch + rules.grid))
    wwl = rules.snap(min(cell.wwl_y, cell.pitch_y - half_pitch - rules.grid))
    if wwl - rwl < 2 * half_pitch:
        raise ValueError("a row pair is too short for four word-line straps")
    return {"rwl_0": rwl, "wwl_0": wwl, "wwl_1": pair - wwl, "rwl_1": pair - rwl}


def far_track_y(tech: Technology, cell: BitcellGeometry) -> tuple[float, float]:
    """y within a row pair of the two metal3 tracks left between the straps (one per row): where
    a banked macro's far decoder outputs cross the array when its word lines are strapped."""
    rules = tech.rules
    straps = strap_track_y(tech, cell)
    low = rules.snap((straps["rwl_0"] + straps["wwl_0"]) / 2)
    high = rules.snap((straps["wwl_1"] + straps["rwl_1"]) / 2)
    pitch = strap_width(tech) + rules.s(LayerRole.M3)
    if low - straps["rwl_0"] < pitch - rules.grid / 2:
        raise ValueError("no room for a far-output track between the straps")
    return low, high


@dataclass(frozen=True)
class WlDriverGeometry:
    width: float
    height: float
    track_x: tuple[float, ...]
    pad_x: float
    driver_y: tuple[float, ...]
    wordlines: tuple[str, ...]


def wl_driver_geometry(
    tech: Technology,
    cell: BitcellGeometry,
    cfg: LogicConfig,
    wordlines: tuple[str, ...] = LEFT_WORDLINES,
) -> WlDriverGeometry:
    rules = tech.rules
    logic = logic_geometry(tech, cfg)
    pad = gate_pad_size(tech)

    via_pad = rules.cut(LayerRole.VIA1) + 2 * rules.enc(LayerRole.M2, LayerRole.VIA1)
    track_pitch = via_pad + rules.s(LayerRole.M2)
    track0 = logic.width + rules.s(LayerRole.M2) + via_pad / 2
    tracks = tuple(rules.snap(track0 + i * track_pitch) for i in range(len(wordlines)))
    pad_x = rules.snap(tracks[-1] + track_pitch / 2 + pad / 2)

    height = 2 * cell.pitch_y
    return WlDriverGeometry(
        width=rules.snap(pad_x + pad / 2 + rules.poly_endcap),
        height=rules.snap(height),
        track_x=tracks,
        pad_x=pad_x,
        driver_y=tuple(rules.snap(i * height / len(wordlines)) for i in range(len(wordlines))),
        wordlines=tuple(wordlines),
    )


def wl_driver_pair(
    tech: Technology,
    cell: BitcellGeometry,
    cfg: LogicConfig | None = None,
    wordlines: tuple[str, ...] = LEFT_WORDLINES,
    name: str | None = None,
    straps: bool = False,
) -> gf.Component:
    """Driver slice covering one mirrored row pair. ``straps``: each output also rises to its word
    line's metal3 strap (:func:`strap_track_y`), which leaves the slice on the array's side."""
    cfg = cfg or LogicConfig.for_tech(tech)
    activate(tech)
    key = name or f"gc_wl_driver_{'_'.join(wordlines)}_{cfg.name}_{cell.pitch_y:g}" + ("_ws" if straps else "")
    return cached_cell(key, lambda n: _build(tech, cell, cfg, wordlines, n, straps), tech)


def _build(
    tech: Technology,
    cell: BitcellGeometry,
    cfg: LogicConfig,
    wordlines: tuple[str, ...],
    name: str,
    straps: bool = False,
) -> gf.Component:
    rules = tech.rules
    geo = wl_driver_geometry(tech, cell, cfg, wordlines)
    logic = logic_geometry(tech, cfg)
    wl_y = wordline_y(cell)
    strap_y = strap_track_y(tech, cell) if straps else {}
    wl_w = wordline_width(cell)
    inv = inverter(tech, cfg)

    component = gf.Component(name)
    # Well and supply rails span the whole slice so stacked slices stay connected.
    draw(component, tech.layer(LayerRole.NWELL), Box(0.0, 0.0, logic.nwell_x1, geo.height))
    tap_w = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    for x in (logic.ntap_x, logic.ptap_x):
        draw(
            component,
            tech.layer(LayerRole.M1),
            Box.centered(x + tap_w / 2, geo.height / 2, rules.w(LayerRole.M1), geo.height),
        )

    for index, (net, y_drv) in enumerate(zip(geo.wordlines, geo.driver_y)):
        ref = component.add_ref(inv)
        ref.dmove((0.0, y_drv))
        _route_to_wordline(
            component,
            tech,
            out=tuple(ref.ports["y"].dcenter),
            track_x=geo.track_x[index],
            pad_x=geo.pad_x,
            wl_y=wl_y[net],
            right_edge=geo.width,
            length=wl_w[net],
            strap_y=strap_y.get(net),
        )
        exit_y = _route_input(
            component, tech, tuple(ref.ports["a"].dcenter), ref.ports["y"].dcenter[1]
        )
        add_port(component,
            name=net,
            center=(geo.width, wl_y[net]),
            orientation=0,
            width=wl_w[net],
            layer=tech.layer(LayerRole.POLY),
            port_type="electrical",
        )
        add_port(component,
            name=f"{net}_in",
            center=(0.0, exit_y),
            orientation=180,
            width=rules.w(LayerRole.M2),
            layer=tech.layer(LayerRole.M2),
            port_type="electrical",
        )

    for net, x in (("vdd", logic.ntap_x), ("vss", logic.ptap_x)):
        add_port(component,
            name=net,
            center=(x + tap_w / 2, 0.0),
            orientation=270,
            width=rules.w(LayerRole.M1),
            layer=tech.layer(LayerRole.M1),
            port_type="electrical",
        )

    component.info["width"] = geo.width
    component.info["height"] = geo.height
    fix_min_area(component, tech)
    return component


def _route_to_wordline(
    component: gf.Component,
    tech: Technology,
    out: tuple[float, float],
    track_x: float,
    pad_x: float,
    wl_y: float,
    right_edge: float,
    length: float,
    strap_y: float | None = None,
) -> None:
    """Inverter output -> metal2 dog-leg -> metal1 -> poly landing pad -> word line; with a
    ``strap_y``, the dog-leg also rises to the metal3 strap there, which runs to ``right_edge``."""
    rules = tech.rules
    m2, m1, poly = (tech.layer(r) for r in (LayerRole.M2, LayerRole.M1, LayerRole.POLY))
    out_x, out_y = out
    half = rules.w(LayerRole.M2) / 2
    via_pad = rules.cut(LayerRole.VIA1) / 2 + rules.enc(LayerRole.M2, LayerRole.VIA1)
    ys = (out_y, wl_y) if strap_y is None else (out_y, wl_y, strap_y)

    via_stack(component, tech, LayerRole.LI, LayerRole.M2, out, draw_bottom_pad=False)
    # The vertical leg is drawn at via-pad width so the via pads leave no notch.
    draw(component, m2, Box(out_x, out_y - half, track_x + via_pad, out_y + half))
    draw(
        component,
        m2,
        Box(
            track_x - via_pad,
            min(ys) - via_pad,
            track_x + via_pad,
            max(ys) + via_pad,
        ),
    )
    if strap_y is not None:
        via_stack(component, tech, LayerRole.M2, LayerRole.M3, (track_x, strap_y))
        # the strap's via2 pad is wider than the leg: where the output line or the corner pad on
        # the leg comes within spacing of it, the metal2 between them is filled at its width, so
        # no notch is left (they are all one net)
        pad2 = rules.via_pad(LayerRole.M2, LayerRole.VIA2) / 2
        for y in (out_y, wl_y):
            if abs(y - strap_y) < pad2 + via_pad + rules.s(LayerRole.M2):
                draw(component, m2, Box(track_x - pad2, min(y, strap_y), track_x + pad2, max(y, strap_y)))
        half3 = strap_width(tech) / 2
        draw(component, tech.layer(LayerRole.M3), Box(track_x - half3, strap_y - half3, right_edge, strap_y + half3))

    corner = via_stack(component, tech, LayerRole.M1, LayerRole.M2, (track_x, wl_y))
    landing = via_stack(component, tech, LayerRole.LI, LayerRole.M1, (pad_x, wl_y))
    draw(component, m1, Box.hull([corner[LayerRole.M1], landing[LayerRole.M1]]))

    pad_size = gate_pad_size(tech)
    pad_box = Box.centered(pad_x, wl_y, pad_size, pad_size)
    draw(component, poly, pad_box)
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.POLY, pad_box, axis="x")
    tech.decorate_gate_contact(component, hull.as_tuple())
    draw(
        component,
        poly,
        Box(pad_box.x1 - rules.grid, wl_y - length / 2, right_edge, wl_y + length / 2),
    )


def input_exit_offset(tech: Technology) -> float:
    """How far above an inverter's output leg its input's metal2 line runs: the leg's half-width,
    metal2 spacing and the line's via1 pad (plus a grid step)."""
    rules = tech.rules
    pad_half = rules.via_pad(LayerRole.M2, LayerRole.VIA1) / 2
    return rules.w(LayerRole.M2) / 2 + rules.s(LayerRole.M2) + pad_half + rules.grid


def _route_input(
    component: gf.Component, tech: Technology, pad: tuple[float, float], out_y: float
) -> float:
    """Bring a driver input out to the left edge on metal2; return the y of that line.

    The gate pad sits between the source contact and the output, where a via stack has no room:
    its metal1 would touch the inverter's own vss strap (which runs past the pad to the tap) and
    its metal2 the output leg that runs along ``out_y``. Either overlap merges into the shape
    it touches -- DRC cannot see it -- and shorts the input to vss or to the output. So the
    contact is lifted on a vertical li strip to just above the output leg, where both are clear.
    Returns the y of the metal2 line, which is where the input is reached.
    """
    rules = tech.rules
    m2 = tech.layer(LayerRole.M2)
    x, y = pad
    half = rules.w(LayerRole.M2) / 2
    exit_y = rules.snap(out_y + input_exit_offset(tech))
    li_half = rules.w(LayerRole.LI) / 2
    draw(component, tech.layer(LayerRole.LI), Box(x - li_half, y, x + li_half, exit_y))
    via_stack(component, tech, LayerRole.LI, LayerRole.M2, (x, exit_y))
    draw(component, m2, Box(0.0, exit_y - half, x + half, exit_y + half))
    return exit_y
