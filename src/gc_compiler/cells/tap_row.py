"""Tap row: ties the substrate to ``vss`` and the n-well to ``vnw``.

The row is one bitcell pitch wide and is arrayed across the columns, so its tap strips,
implants, well and supply rails merge into continuous stripes. It is inserted at ``rbl``
cell boundaries, where the neighbouring rows present only n+ diffusion, which keeps the
well spacing arithmetic simple.
"""

from __future__ import annotations

from dataclasses import dataclass

import gdsfactory as gf

from gc_compiler.cache import cached_cell
from gc_compiler.cells.bitcell_3t import BitcellGeometry
from gc_compiler.primitives.wells import fix_min_area
from gc_compiler.primitives.contacts import place_cuts, via_stack
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.rules import DeviceKind, LayerRole, Technology


@dataclass(frozen=True)
class TapRowGeometry:
    height: float
    ptap_y: float
    ntap_y: float
    nwell_y0: float
    nwell_y1: float


def tap_row_geometry(tech: Technology, geo: BitcellGeometry) -> TapRowGeometry:
    rules = tech.rules
    # n+ diffusion of the adjacent rows pokes this far into the tap row.
    protrusion = rules.contacted_node_extent()
    strip = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)

    # ... and its implant, which the p tap's must not overlap
    to_ptap = max(rules.s(LayerRole.DIFF), rules.enc(LayerRole.NIMP, LayerRole.DIFF) + rules.enc(LayerRole.PIMP, LayerRole.TAP))
    ptap_y = protrusion + to_ptap + strip / 2
    nwell_y0 = ptap_y + strip / 2 + rules.nwell_to_ndiff_space
    nwell_y1 = nwell_y0 + max(
        rules.w(LayerRole.NWELL), strip + 2 * rules.enc(LayerRole.NWELL, LayerRole.TAP)
    )
    ntap_y = (nwell_y0 + nwell_y1) / 2

    # The neighbouring rows push diffusion, and its implant, into the tap row.
    implant_protrusion = protrusion + rules.enc(LayerRole.NIMP, LayerRole.DIFF)
    ntap_implant_top = ntap_y + strip / 2 + rules.enc(LayerRole.NIMP, LayerRole.TAP)
    height = max(
        nwell_y1 + rules.nwell_to_ndiff_space + protrusion,
        ntap_implant_top + rules.s(LayerRole.NIMP) + implant_protrusion,
    )
    return TapRowGeometry(
        height=rules.snap(height),
        ptap_y=rules.snap(ptap_y),
        ntap_y=rules.snap(ntap_y),
        nwell_y0=rules.snap(nwell_y0),
        nwell_y1=rules.snap(nwell_y1),
    )


def tap_row_cell(
    tech: Technology,
    geo: BitcellGeometry,
    name: str | None = None,
) -> gf.Component:
    """One bitcell-pitch slice of a tap row."""
    name = name or f"gc_tap_row_{geo.pitch_x:g}"
    return cached_cell(name, lambda cell_name: _build(tech, geo, cell_name), tech)


def _build(tech: Technology, geo: BitcellGeometry, name: str) -> gf.Component:
    rules = tech.rules
    tap_geo = tap_row_geometry(tech, geo)
    component = gf.Component(name)

    draw(
        component,
        tech.layer(LayerRole.NWELL),
        Box(0.0, tap_geo.nwell_y0, geo.pitch_x, tap_geo.nwell_y1),
    )

    _tap_strip(component, tech, geo, tap_geo.ptap_y, DeviceKind.NMOS, "vss")
    _tap_strip(component, tech, geo, tap_geo.ntap_y, DeviceKind.PMOS, "vnw")

    # Bit lines run straight through.
    half = rules.w(LayerRole.M2) / 2
    for x in (geo.rbl_x, geo.wbl_x):
        draw(component, tech.layer(LayerRole.M2), Box(x - half, 0.0, x + half, tap_geo.height))

    for name_, x in (("rbl", geo.rbl_x), ("wbl", geo.wbl_x)):
        for suffix, y, orientation in (("_n", tap_geo.height, 90), ("_s", 0.0, 270)):
            port_name = f"{name_}{suffix}"
            add_port(component,
                name=port_name,
                center=(x, y),
                orientation=orientation,
                width=rules.w(LayerRole.M2),
                layer=tech.layer(LayerRole.M2),
                port_type="electrical",
            )
            label = tech.label_layer(LayerRole.M2)
            if label is not None:
                component.add_label(text=name_, position=(x, y), layer=label)

    component.info["pitch_x"] = geo.pitch_x
    component.info["height"] = tap_geo.height
    fix_min_area(component, tech)
    return component


def _tap_strip(
    component: gf.Component,
    tech: Technology,
    geo: BitcellGeometry,
    cy: float,
    body_of: DeviceKind,
    net: str,
) -> None:
    """Tap strip plus its implant, li strap and metal1 rail, drawn full pitch."""
    rules = tech.rules
    strip_h = rules.cut(LayerRole.LICON) + 2 * rules.enc(LayerRole.TAP, LayerRole.LICON)
    strip = Box(0.0, cy - strip_h / 2, geo.pitch_x, cy + strip_h / 2)
    draw(component, tech.layer(LayerRole.TAP), strip)

    implant = tech.tap_implant_role(body_of)
    draw(component, tech.layer(implant), strip.grow(rules.enc(implant, LayerRole.TAP)))

    cut_region = Box(
        strip.x0, cy - rules.cut(LayerRole.LICON) / 2, strip.x1, cy + rules.cut(LayerRole.LICON) / 2
    )
    hull = place_cuts(component, tech, LayerRole.LICON, LayerRole.TAP, cut_region, axis="x")
    li_h = max(hull.height, rules.w(LayerRole.LI))
    draw(component, tech.layer(LayerRole.LI), Box(0.0, cy - li_h / 2, geo.pitch_x, cy + li_h / 2))

    pads = via_stack(
        component, tech, LayerRole.LI, LayerRole.M1, (geo.pitch_x / 2, cy), draw_bottom_pad=False
    )
    rail = pads[LayerRole.M1]
    draw(component, tech.layer(LayerRole.M1), Box(0.0, rail.y0, geo.pitch_x, rail.y1))
    add_port(component,
        name=net,
        center=(0.0, cy),
        orientation=180,
        width=rail.height,
        layer=tech.layer(LayerRole.M1),
        port_type="electrical",
    )
    label = tech.label_layer(LayerRole.M1)
    if label is not None:
        component.add_label(text=net, position=(0.0, cy), layer=label)
