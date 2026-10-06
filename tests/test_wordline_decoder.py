"""The two half-decoders placed beside the word-line drivers, and their wiring."""

from __future__ import annotations

import shutil
from functools import lru_cache

import gdsfactory as gf
import pytest

from gc_compiler.array.floorplan import gc_macro, macro_geometry, wordline_periphery
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.decoder import (
    address_bits,
    gate_subckts,
    row_decoder_pins,
    row_decoder_subckt,
    side_outputs,
)
from gc_compiler.netlist.spice import inverter_subckt
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.periphery.wl_driver import LEFT_WORDLINES, RIGHT_WORDLINES, wl_driver_pair
from gc_compiler.primitives.geometry import Box, draw
from gc_compiler.tech.rules import LayerRole
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.connectivity import trace_nets
from gc_compiler.verify.drc import default_deck, run_drc
from gc_compiler.verify.lvs import default_magicrc, default_netgen_setup, lvs_layout_vs_netlist


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


requires_drc = pytest.mark.skipif(
    shutil.which("klayout") is None or not default_deck().exists(),
    reason="klayout or the sky130 deck is missing",
)
requires_lvs = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("magic", "netgen"))
    or not default_magicrc().exists()
    or not default_netgen_setup().exists(),
    reason="magic, netgen or the sky130 setup files are missing",
)


@lru_cache(maxsize=None)
def _standin(tech: Sky130, wordlines: tuple[str, ...], rows: int) -> gf.Component:
    """A driver slice with the real ports and input lines and none of the real insides."""
    real = wl_driver_pair(tech, _cell(tech), LogicConfig(), wordlines)
    cell = gf.Component(f"standin_{'_'.join(wordlines)}_{rows}")
    m1, m2 = tech.layer(LayerRole.M1), tech.layer(LayerRole.M2)
    for port in real.ports:
        x, y = port.dcenter
        if port.name.endswith("_in"):
            draw(cell, m2, Box(x, y - 0.07, x + 2.9, y + 0.07))
            cell.add_port(name=port.name, center=(x, y), width=0.14, orientation=180, layer=m2, port_type="electrical")
        elif port.name in ("vdd", "vss"):
            draw(cell, m1, Box(x - 0.07, 0.0, x + 0.07, real.dbbox().top))
            cell.add_port(name=port.name, center=(x, 0.0), width=0.14, orientation=270, layer=m1, port_type="electrical")
    return cell


def _cell(tech):
    from gc_compiler.array.tiler import array_geometry

    return array_geometry(tech, ArrayConfig.basic()).cell


def _periphery(tech, rows, name, real_drivers):
    cfg = ArrayConfig.basic(rows=rows, cols=4)
    cells = None if real_drivers else (_standin(tech, LEFT_WORDLINES, rows), _standin(tech, RIGHT_WORDLINES, rows))
    return cfg, wordline_periphery(tech, cfg, name=name, driver_cells=cells)


# gc_macro builds its array under the array's own name, so each macro in one session needs
# a distinct array size (hence the odd column counts below).
def test_decoded_macro_exposes_the_address_interface(tech):
    cfg = ArrayConfig.basic(rows=8, cols=6)
    macro = gc_macro(tech, cfg, name="wld_decoded_ports", decoded=True)
    names = {port.name for port in macro.ports}
    assert {f"addr_{i}" for i in range(address_bits(cfg.rows))} | {"we", "re"} <= names
    assert not any(name.startswith(("wwl_in_", "rwl_in_")) for name in names)
    assert {f"din_{c}" for c in range(cfg.cols)} <= names
    base = macro_geometry(tech, cfg)
    assert macro.info["decoded"] is True
    assert macro.info["width"] > base.width and macro.info["height"] >= base.height


def test_default_macro_is_still_the_undecoded_one(tech):
    cfg = ArrayConfig.basic(rows=8, cols=5)
    macro = gc_macro(tech, cfg, name="wld_plain")
    base = macro_geometry(tech, cfg)
    assert macro.info["decoded"] is False
    assert macro.info["width"] == pytest.approx(base.width)
    assert macro.info["height"] == pytest.approx(base.height)
    assert any(port.name == "wwl_in_0" for port in macro.ports)


@pytest.mark.parametrize("rows", [2, 6, 16, 32])
def test_decoder_outputs_reach_their_drivers_and_nothing_else(tech, tmp_path, rows):
    """Metal connectivity (the driver slices are stood in for, see the xfail below): each of
    the ``2 * rows`` word-line inputs is fed by exactly its own decoder output, the shared
    address/enable trunks are one net across both decoders, and no signal touches a supply.
    """
    cfg, (component, drivers, (left, right)) = _periphery(tech, rows, f"wld_conn_{rows}", real_drivers=False)
    gds = component.write_gds(tmp_path / "periphery.gds")
    nets = trace_nets(tech, gds, metals_only=True)
    M1, M2 = LayerRole.M1, LayerRole.M2

    outputs = {}
    for ref, outs in ((left, side_outputs(rows)[0]), (right, side_outputs(rows)[1])):
        for row, signal in outs:
            outputs[f"{signal}_{row}"] = nets.net_at(M2, tuple(ref.ports[f"{signal}_{row}"].dcenter))
    inputs = {}
    for pair_index, left_ref, right_ref in drivers:
        for ref in (left_ref, right_ref):
            for port in ref.ports:
                if port.name.endswith("_in"):
                    signal, relative = port.name[: -len("_in")].rsplit("_", 1)
                    inputs[f"{signal}_in_{2 * pair_index + int(relative)}"] = nets.net_at(M2, tuple(port.dcenter))
    assert set(outputs) == set(inputs) and len(outputs) == 2 * rows
    assert {k for k in outputs if outputs[k] is None or outputs[k] != inputs[k]} == set()
    assert len(set(outputs.values())) == len(outputs)

    shared = [f"addr_{i}" for i in range(address_bits(rows))] + ["we", "re"]
    bus = {}
    for net in shared:
        a = nets.net_at(M2, tuple(left.ports[f"bus_{net}"].dcenter))
        b = nets.net_at(M2, tuple(right.ports[f"bus_{net}"].dcenter))
        assert a is not None and a == b, net
        bus[net] = a

    geo = macro_geometry(tech, cfg)
    top = geo.array_y + geo.array.height
    supplies = {}
    for side, ref, (_pair, driver_left, driver_right) in (("L", left, drivers[0]), ("R", right, drivers[0])):
        driver = driver_left if side == "L" else driver_right
        vdd = nets.net_at(M1, tuple(ref.ports["vdd"].dcenter))
        vss = nets.net_at(M2, tuple(ref.ports["vss"].dcenter))
        rail_vdd = nets.net_at(M1, (driver.ports["vdd"].dcenter[0], top + 1.0))
        rail_vss = nets.net_at(M1, (driver.ports["vss"].dcenter[0], top + 1.0))
        assert vdd is not None and vdd == rail_vdd and vss is not None and vss == rail_vss, side
        supplies[side + "vdd"], supplies[side + "vss"] = vdd, vss
    everything = [*outputs.values(), *bus.values(), *supplies.values()]
    assert len(set(everything)) == len(everything)


@requires_drc
@pytest.mark.parametrize("rows", [2, 8, 32])
def test_periphery_is_drc_clean(tech, tmp_path, rows):
    _cfg, (component, _drivers, _decoders) = _periphery(tech, rows, f"wld_drc_{rows}", real_drivers=True)
    report = run_drc(component.write_gds(tmp_path / "periphery.gds"))
    assert report.clean, str(report)


@requires_drc
def test_decoded_macro_is_drc_clean(tech, tmp_path):
    macro = gc_macro(tech, ArrayConfig.basic(rows=8, cols=7), name="wld_drc_macro", decoded=True)
    report = run_drc(macro.write_gds(tmp_path / "macro.gds"))
    assert report.clean, str(report)


@requires_lvs
@pytest.mark.parametrize("rows", [4, 16])
def test_each_half_decoder_matches_its_schematic(tech, tmp_path, rows):
    """The decoder exactly as the macro builds it (targets, pinned trunks and all)."""
    logic = LogicConfig()
    cfg, (_component, _drivers, decoders) = _periphery(tech, rows, f"wld_lvs_{rows}", real_drivers=False)
    for ref, outputs in zip(decoders, side_outputs(rows)):
        cell = ref.cell
        gds = tmp_path / f"{cell.name}.gds"
        cell.kdb_cell.write(str(gds))
        reference = tmp_path / f"{cell.name}.spice"
        reference.write_text("\n\n".join([
            inverter_subckt(tech, logic),
            *gate_subckts(tech, logic),
            row_decoder_subckt(tech, cfg, logic, outputs=outputs),
        ]))
        result = lvs_layout_vs_netlist(gds, cell.name, reference, "gc_row_decoder", workdir=tmp_path / f"lvs_{cell.name}")
        assert result.matched, str(result)
        assert result.mismatched_nets == []
        assert result.net_count[0] == result.net_count[1]


@pytest.mark.parametrize("wordlines", [LEFT_WORDLINES, RIGHT_WORDLINES])
def test_driver_slice_nets_are_all_separate(tech, tmp_path, wordlines):
    """A driver's input pad used to overlap the inverter's own vss strap and its output leg, so
    every input was shorted to vss (and to the output). DRC cannot see it -- the shapes merge
    into one polygon -- so trace the nets: inputs, word-line outputs and both rails must all
    be different nets, in the metal wiring alone and through the contacts and devices.
    """
    cell = wl_driver_pair(tech, _cell(tech), LogicConfig(), wordlines, name=f"wld_driver_probe_{wordlines[0]}")
    gds = cell.write_gds(tmp_path / "driver.gds")
    for metals_only in (True, False):
        nets = trace_nets(tech, gds, metals_only=metals_only)
        found = {
            f"{net}_in": nets.net_at(LayerRole.M2, tuple(cell.ports[f"{net}_in"].dcenter)) for net in wordlines
        }
        for rail in ("vdd", "vss"):
            found[rail] = nets.net_at(LayerRole.M1, (cell.ports[rail].dcenter[0], cell.dbbox().top / 2))
        if not metals_only:
            for net in wordlines:
                x, y = cell.ports[net].dcenter
                found[net] = nets.net_at(LayerRole.POLY, (x - 0.05, y))
        assert all(v is not None for v in found.values()), found
        assert len(set(found.values())) == len(found), found


@requires_lvs
@pytest.mark.parametrize("rows", [4, 16])
def test_decoders_and_real_drivers_match_their_schematic(tech, tmp_path, rows):
    """Both decoders and the actual driver slices, extracted and compared to gates: every
    word-line input is fed by exactly its decoder output and drives exactly its inverter.

    Without the array the two driver columns share no vdd, so the reference gives them one
    each (the vss taps are joined by the substrate).
    """
    logic = LogicConfig()
    cfg = ArrayConfig.basic(rows=rows, cols=4)
    name = f"wld_real_{rows}"
    component, _drivers, _decoders = wordline_periphery(tech, cfg, logic, name=name)
    gds = component.write_gds(tmp_path / "periphery.gds")
    left, right = side_outputs(rows)
    pins = [f"addr_{i}" for i in range(address_bits(rows))] + ["we", "re", "vdd_l", "vdd_r", "vss"]
    lines = [f".subckt {name} {' '.join(pins)}"]
    for side, outputs, vdd in (("left", left, "vdd_l"), ("right", right, "vdd_r")):
        lines.append(
            f"Xdecoder_{side} {' '.join(vdd if p == 'vdd' else p for p in row_decoder_pins(cfg, outputs))} gc_row_decoder_{side}"
        )
    for outputs, vdd in ((left, "vdd_l"), (right, "vdd_r")):
        for row, signal in outputs:
            lines.append(f"Xdrv_{signal}_{row} {signal}_{row} {signal.removesuffix('_in')}_{row} {vdd} vss gc_inv")
    lines.append(f".ends {name}")
    reference = tmp_path / "periphery.spice"
    reference.write_text("\n\n".join([
        inverter_subckt(tech, logic),
        *gate_subckts(tech, logic),
        row_decoder_subckt(tech, cfg, logic, name="gc_row_decoder_left", outputs=left),
        row_decoder_subckt(tech, cfg, logic, name="gc_row_decoder_right", outputs=right),
        "\n".join(lines),
    ]))
    result = lvs_layout_vs_netlist(gds, name, reference, name, workdir=tmp_path / "lvs")
    assert result.matched, str(result)
    assert result.mismatched_nets == []
    assert result.net_count[0] == result.net_count[1]


def test_column_circuit_nets_are_all_separate(tech, tmp_path):
    """The write driver's and read buffer's inputs used to be contacted on a pad that overlapped
    the inverter's own vss stub, shorting ``din`` and ``rbl`` to vss (DRC cannot see it)."""
    from gc_compiler.periphery.column import column_circuit

    cell = column_circuit(tech, _cell(tech), LogicConfig(), name="wld_column_probe")
    gds = cell.write_gds(tmp_path / "column.gds")
    for metals_only in (True, False):
        nets = trace_nets(tech, gds, metals_only=metals_only)
        found = {}
        for port in cell.ports:
            x, y = port.dcenter
            if port.name in ("dout", "din"):
                found[port.name] = nets.net_at(LayerRole.M2, (x, y + 0.3))
            elif port.name in ("rbl", "wbl"):
                found[port.name] = nets.net_at(LayerRole.M2, (x, y - 0.3))
            else:  # the horizontal metal1 rails: pch, vdd_i, vss_i
                found[port.name] = nets.net_at(LayerRole.M1, (x + 0.3, y))
        assert all(v is not None for v in found.values()), found
        assert len(set(found.values())) == len(found), found


def test_column_circuits_follow_the_array_columns(tech):
    """Past a strap column the array's columns shift; the column circuits must shift with them
    or every later bit line misses its circuit."""
    from gc_compiler.array.tiler import array_geometry

    cfg = ArrayConfig.basic(rows=8, cols=20)
    geo = array_geometry(tech, cfg)
    assert len(geo.strap_col_x) > 1  # the case this guards
    macro = gc_macro(tech, cfg, name="wld_columns_probe")
    ports = {p.name: p for p in macro.ports}
    for col in range(cfg.cols):
        shift = geo.column_x(col) - geo.column_x(0)
        assert ports[f"din_{col}"].dcenter[0] - ports["din_0"].dcenter[0] == pytest.approx(shift)
        assert ports[f"dout_{col}"].dcenter[0] - ports["dout_0"].dcenter[0] == pytest.approx(shift)
