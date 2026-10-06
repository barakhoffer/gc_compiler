"""Column periphery: precharge device, read buffer and write driver.

The slices are the same logic cells used by the word-line drivers, rotated 90 degrees so
that their supply rails run horizontally and their footprint pitch-matches the column.
Signal flow is vertical: the bit lines come down from the array on metal2, ``dout`` and
``din`` leave at the bottom edge, and ``pch`` is a horizontal metal1 rail shared by every
column.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import gdsfactory as gf

from gc_compiler.cache import cached_cell
from gc_compiler.cells.bitcell_3t import BitcellGeometry
from gc_compiler.periphery.logic import LogicConfig, inverter, logic_geometry, nand_slice, nand_slice_geometry, pmos_switch
from gc_compiler.primitives.wells import fix_min_area
from gc_compiler.primitives.contacts import min_pad, via_stack
from gc_compiler.primitives.geometry import Box, draw, add_port
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import LayerRole, Technology


@dataclass(frozen=True)
class ColumnGeometry:
    width: float
    height: float
    slice_pitch: float
    precharge_y: float
    read_y: float
    write_y: float
    din_x: float
    dout_x: float


def column_geometry(tech: Technology, cell: BitcellGeometry, cfg: LogicConfig, write_back: bool = False) -> ColumnGeometry:
    """``write_back``: the write driver is a NAND of ``din`` and ``rbl`` (a refresh writes the sensed
    bit back, see :func:`column_circuit`), above the read buffer instead of below it."""
    rules = tech.rules
    logic = logic_geometry(tech, cfg)
    # Stacked slices face each other tap-to-well, so they need the well-to-diffusion gap.
    gap = rules.nwell_to_ndiff_space
    pitch = logic.width + gap

    half = rules.w(LayerRole.M2) / 2
    din_x = rules.snap(cell.wbl_x + half + rules.s(LayerRole.M2) + half)
    dout_x = cell.rbl_x
    if write_back:
        nand = nand_slice_geometry(tech, replace(cfg, pitch=cell.pitch_x)).width + gap
        return ColumnGeometry(
            width=cell.pitch_x,
            height=rules.snap(2 * pitch + nand),
            slice_pitch=rules.snap(pitch),
            read_y=0.0,
            write_y=rules.snap(pitch),
            precharge_y=rules.snap(pitch + nand),
            din_x=din_x,
            dout_x=dout_x,
        )
    return ColumnGeometry(
        width=cell.pitch_x,
        height=rules.snap(3 * pitch),
        slice_pitch=rules.snap(pitch),
        write_y=0.0,
        read_y=rules.snap(pitch),
        precharge_y=rules.snap(2 * pitch),
        din_x=din_x,
        dout_x=dout_x,
    )


def column_circuit(
    tech: Technology,
    cell: BitcellGeometry,
    cfg: LogicConfig | None = None,
    name: str | None = None,
    write_back: bool = False,
) -> gf.Component:
    """One column's periphery slice, pitch-matched to the bitcell column.

    ``write_back``: the write driver is ``wbl = NAND(din, rbl)``. A write has the bit line
    precharged (``rbl`` high), so it is the plain driver's ``wbl = ~din``; a refresh keeps the
    precharge off after its read and holds ``din`` high, so each column writes back ``~rbl``, the
    bit it just sensed."""
    cfg = cfg or LogicConfig.for_tech(tech)
    # Rotated slices tile along the column pitch, so that is their pitch.
    cfg = replace(cfg, pitch=cell.pitch_x)
    activate(tech)
    key = name or f"gc_column_{cfg.name}_{cell.pitch_x:g}" + ("_wb" if write_back else "")
    return cached_cell(key, lambda n: _build(tech, cell, cfg, n, write_back), tech)


def _place(component: gf.Component, slice_cell: gf.Component, y: float, clockwise: bool, mirror: bool = False):
    """Rotate a logic slice into the column orientation and drop it at ``y``; ``mirror`` flips it
    across the column's axis."""
    ref = component.add_ref(slice_cell)
    ref.drotate(-90 if clockwise else 90)
    if mirror:
        ref.dmirror_x()
    box = ref.dbbox()
    ref.dmove((-box.left, y - box.bottom))
    return ref


def _nwell_band(ref, tech: Technology) -> tuple[float, float]:
    """y extent of a placed slice's n-well."""
    box = ref.dbbox(gf.get_layer(tech.layer(LayerRole.NWELL)))
    return (box.bottom, box.top)


def _merge_wells(component: gf.Component, tech: Technology, geo: ColumnGeometry, bands) -> None:
    """Bridge n-well bands that face each other across a slice gap."""
    rules = tech.rules
    bands = sorted(bands)
    for (_, top), (bottom, _) in zip(bands, bands[1:]):
        if 0 < bottom - top < rules.s(LayerRole.NWELL):
            draw(component, tech.layer(LayerRole.NWELL), Box(0.0, top, geo.width, bottom))


def _build(tech: Technology, cell: BitcellGeometry, cfg: LogicConfig, name: str, write_back: bool = False) -> gf.Component:
    rules = tech.rules
    geo = column_geometry(tech, cell, cfg, write_back)
    component = gf.Component(name)
    m2 = tech.layer(LayerRole.M2)
    half = rules.w(LayerRole.M2) / 2
    if write_back:
        return _build_write_back(tech, cell, cfg, geo, component)

    write = _place(component, inverter(tech, cfg), geo.write_y, clockwise=True)
    read = _place(component, inverter(tech, cfg), geo.read_y, clockwise=False)
    precharge = _place(component, pmos_switch(tech, cfg), geo.precharge_y, clockwise=False)

    logic = logic_geometry(tech, cfg)
    _merge_wells(
        component,
        tech,
        geo,
        [_nwell_band(ref, tech) for ref in (write, read, precharge)],
    )

    pre_out = tuple(precharge.ports["y"].dcenter)
    read_in = tuple(read.ports["a"].dcenter)
    read_out = tuple(read.ports["y"].dcenter)
    write_in = tuple(write.ports["a"].dcenter)
    write_out = tuple(write.ports["y"].dcenter)

    # rbl: down from the array, into the read buffer, with the precharge device tapped on.
    _drop(component, tech, pre_out, cell.rbl_x)
    _lift_input(component, tech, read_in, cell.rbl_x)
    draw(component, m2, Box(cell.rbl_x - half, read_in[1], cell.rbl_x + half, geo.height))

    # wbl: write driver output up to the array.
    _drop(component, tech, write_out, cell.wbl_x)
    draw(component, m2, Box(cell.wbl_x - half, write_out[1], cell.wbl_x + half, geo.height))

    # dout leaves below the read buffer, on the track the bit line has vacated.
    _drop(component, tech, read_out, geo.dout_x)
    draw(component, m2, Box(geo.dout_x - half, 0.0, geo.dout_x + half, read_out[1]))

    # din enters at the bottom edge and runs up to the write driver.
    _lift_input(component, tech, write_in, geo.din_x)
    draw(component, m2, Box(geo.din_x - half, 0.0, geo.din_x + half, write_in[1]))

    _rail(component, tech, geo, tuple(precharge.ports["a"].dcenter), "pch")
    _extend_supply_rails(component, tech, geo, (write, read, precharge))

    for net, x, orientation in (("dout", geo.dout_x, 270), ("din", geo.din_x, 270)):
        add_port(component,
            name=net,
            center=(x, 0.0),
            orientation=orientation,
            width=rules.w(LayerRole.M2),
            layer=m2,
            port_type="electrical",
        )
    for net, x in (("rbl", cell.rbl_x), ("wbl", cell.wbl_x)):
        add_port(component,
            name=net,
            center=(x, geo.height),
            orientation=90,
            width=rules.w(LayerRole.M2),
            layer=m2,
            port_type="electrical",
        )

    component.info["width"] = geo.width
    component.info["height"] = geo.height
    fix_min_area(component, tech, tile=(0.0, 0.0, geo.width, geo.height))
    return component


def _build_write_back(tech: Technology, cell: BitcellGeometry, cfg: LogicConfig, geo: ColumnGeometry, component: gf.Component) -> gf.Component:
    """The ``write_back`` column, bottom up: read buffer, the write driver's NAND, precharge. ``rbl``
    comes down past the NAND (its ``b`` input) to the read buffer; ``din`` rises past the read buffer
    to the NAND's two ``a`` contacts; ``dout`` leaves below the read buffer as before."""
    rules = tech.rules
    m2 = tech.layer(LayerRole.M2)
    half = rules.w(LayerRole.M2) / 2
    read = _place(component, inverter(tech, cfg), geo.read_y, clockwise=False)
    # (mirrored: its metal1 jumper from the output to the pull-down, over din's contact, then runs on
    # rbl's side of the column, clear of din's via)
    write = _place(component, nand_slice(tech, cfg), geo.write_y, clockwise=True, mirror=True)
    precharge = _place(component, pmos_switch(tech, cfg), geo.precharge_y, clockwise=False)
    _merge_wells(component, tech, geo, [_nwell_band(ref, tech) for ref in (read, write, precharge)])

    read_in, read_out = tuple(read.ports["a"].dcenter), tuple(read.ports["y"].dcenter)
    # rbl: the precharge device, the NAND's two b contacts and the read buffer's input, on its track
    _drop(component, tech, tuple(precharge.ports["y"].dcenter), cell.rbl_x)
    for p in ("b0", "b1"):
        _lift_input(component, tech, tuple(write.ports[p].dcenter), cell.rbl_x)
    _lift_input(component, tech, read_in, cell.rbl_x)
    draw(component, m2, Box(cell.rbl_x - half, read_in[1], cell.rbl_x + half, geo.height))
    # wbl: the NAND's output up to the array
    write_out = tuple(write.ports["y"].dcenter)
    _drop(component, tech, write_out, cell.wbl_x)
    draw(component, m2, Box(cell.wbl_x - half, write_out[1], cell.wbl_x + half, geo.height))
    # dout below the read buffer, on the track rbl ends above
    _drop(component, tech, read_out, geo.dout_x)
    draw(component, m2, Box(geo.dout_x - half, 0.0, geo.dout_x + half, read_out[1]))
    # din up to the NAND's a contact, which is below its output: din's track ends before wbl's
    # starts (the two tracks are a minimum pitch apart, too close for either to pass the other's via)
    write_in = tuple(write.ports["a"].dcenter)
    if write_in[1] >= write_out[1]:
        raise AssertionError("the NAND's input is above its output: din would run past wbl's via")
    _lift_input(component, tech, write_in, geo.din_x)
    draw(component, m2, Box(geo.din_x - half, 0.0, geo.din_x + half, write_in[1]))

    _rail(component, tech, geo, tuple(precharge.ports["a"].dcenter), "pch")
    _extend_supply_rails(component, tech, geo, (read, write, precharge))
    for net, x, orientation in (("dout", geo.dout_x, 270), ("din", geo.din_x, 270)):
        add_port(component, name=net, center=(x, 0.0), orientation=orientation, width=rules.w(LayerRole.M2), layer=m2, port_type="electrical")
    for net, x in (("rbl", cell.rbl_x), ("wbl", cell.wbl_x)):
        add_port(component, name=net, center=(x, geo.height), orientation=90, width=rules.w(LayerRole.M2), layer=m2, port_type="electrical")
    component.info["width"] = geo.width
    component.info["height"] = geo.height
    fix_min_area(component, tech, tile=(0.0, 0.0, geo.width, geo.height))
    return component


def _drop(component: gf.Component, tech: Technology, port: tuple[float, float], track_x: float):
    """Connect a slice port to a vertical metal2 track.

    The hop runs on metal1 so that the wide via pads stay clear of the neighbouring bit
    line, which is only a track away.
    """
    _, y = port
    local = via_stack(
        component, tech, LayerRole.LI, LayerRole.M1, port, draw_bottom_pad=False, axis="x"
    )
    corner = via_stack(component, tech, LayerRole.M1, LayerRole.M2, (track_x, y), axis="x")
    # (its minimum area is made up on the cell, by fix_min_area, which keeps the spacing)
    draw(component, tech.layer(LayerRole.M1), Box.hull([local[LayerRole.M1], corner[LayerRole.M1]]))


def _lift_input(component: gf.Component, tech: Technology, port: tuple[float, float], track_x: float):
    """Connect a slice's gate input to a vertical metal2 track.

    A gate pad sits close to the inverter's own supply stub, so a metal1 contact *on* the pad
    overlaps the stub: the shapes merge into one polygon (DRC cannot see it) and the input is
    shorted to the supply. The contact is made at the track instead, on a li strip that runs
    sideways from the pad, well clear of the stub.
    """
    x, y = port
    pads = via_stack(component, tech, LayerRole.LI, LayerRole.M2, (track_x, y), axis="x")
    # as tall as the pads at both ends, so the strip leaves no li notch between them
    half = pads[LayerRole.LI].height / 2
    draw(
        component,
        tech.layer(LayerRole.LI),
        Box(min(x, track_x), y - half, max(x, track_x), y + half),
    )


def _rail(
    component: gf.Component,
    tech: Technology,
    geo: ColumnGeometry,
    port: tuple[float, float],
    net: str,
) -> None:
    """Horizontal metal1 control rail, drawn full pitch so columns share it."""
    rules = tech.rules
    pads = via_stack(component, tech, LayerRole.LI, LayerRole.M1, port, draw_bottom_pad=False)
    rail = pads[LayerRole.M1]
    draw(component, tech.layer(LayerRole.M1), Box(0.0, rail.y0, geo.width, rail.y1))
    add_port(component,
        name=net,
        center=(0.0, rail.center[1]),
        orientation=180,
        width=rail.height,
        layer=tech.layer(LayerRole.M1),
        port_type="electrical",
    )


def _extend_supply_rails(component: gf.Component, tech: Technology, geo: ColumnGeometry, refs):
    """Stretch each slice's supply rails across the full column pitch."""
    rules = tech.rules
    width = rules.w(LayerRole.M1)
    for index, ref in enumerate(refs):
        for net in ("vdd", "vss"):
            _, y = ref.ports[net].dcenter
            draw(
                component,
                tech.layer(LayerRole.M1),
                Box(0.0, y - width / 2, geo.width, y + width / 2),
            )
            add_port(component,
                name=f"{net}_{index}",
                center=(0.0, y),
                orientation=180,
                width=width,
                layer=tech.layer(LayerRole.M1),
                port_type="electrical",
            )
