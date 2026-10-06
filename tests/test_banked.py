"""The banked macro: two sub-arrays on either side of one central decoder."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gc_compiler.array.floorplan import gc_banked_macro, macro_geometry
from gc_compiler.array.wordline_decoder import central_decoder
from gc_compiler.cells.bitcell_3t import bitcell_geometry
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.decoder import (
    address_bits,
    bank_and_row,
    banked_face_outputs,
    gate_subckts,
    output_net,
    row_decoder_subckt,
)
from gc_compiler.netlist.spice import inverter_subckt, macro_netlist, macro_pins
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.tech.rules import LayerRole
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.drc import default_deck, run_drc
from gc_compiler.verify.lvs import default_magicrc, default_netgen_setup, lvs_layout_vs_netlist

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


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


def _pitch_targets(tech, cfg) -> tuple[tuple[dict, dict], tuple[dict, dict]]:
    """Near and far targets of both banks at the word-line pitch, as a decoder built alone. The
    far ones sit half a pitch up, as the two driver columns' inputs do in a macro (a near and a
    far output at one point would make their pins indistinguishable)."""
    pitch = bitcell_geometry(tech, cfg.bitcell).pitch_y
    near, far = ({}, {}), ({}, {})
    for bank in (0, 1):
        for shift, (side, outputs) in enumerate(zip((near, far), banked_face_outputs(cfg.rows, bank))):
            for logical, sig in outputs:
                y = 0.5 + (bank_and_row(cfg.rows, logical)[1] + shift / 2) * pitch
                side[bank][output_net(sig, logical, cfg.rows)] = tech.rules.snap(y)
    return near, far


def test_central_decoder_lands_its_outputs_where_asked(tech):
    """Every output at the two edges (x = 0 for bank 0, x = width for bank 1): near ones on
    metal2, far ones on metal3; every one at the y it was given."""
    cfg = ArrayConfig.basic(rows=8, cols=2)
    near, far = _pitch_targets(tech, cfg)
    decoder = central_decoder(tech, cfg, LogicConfig(), near, far)
    ports = {port.name: port for port in decoder.ports}
    width = decoder.info["width"]
    m2, m3 = tech.layer(LayerRole.M2), tech.layer(LayerRole.M3)
    for bank in (0, 1):
        for net, y in near[bank].items():
            x, py = ports[net].dcenter
            assert py == pytest.approx(y) and x == pytest.approx(0.0 if bank == 0 else width)
            assert ports[net].layer == decoder.kcl.layer(*m2)
        for net, y in far[bank].items():
            x, py = ports[net].dcenter
            assert py == pytest.approx(y) and x == pytest.approx(0.0 if bank == 0 else width)
            assert ports[net].layer == decoder.kcl.layer(*m3)
    # one bus channel per bank: each face is a merged near/far part
    assert len(decoder.info["face_widths"].split(",")) == 2
    # one predecode for both banks: narrower than two faces' worth of it would be
    assert decoder.info["predecode_width"] < width / 2


@requires_drc
@requires_lvs
@pytest.mark.parametrize("rows", [6, 16])
def test_central_decoder_drc_and_lvs(tech, tmp_path, rows):
    """The two faces and the shared predecode, joined by their pinned trunks, are exactly the
    flat banked decoder -- pin for pin, including a non-power-of-two bank, where a face's own
    cheapest address split would differ from the whole decoder's."""
    cfg = ArrayConfig.basic(rows=rows, cols=2)
    logic = LogicConfig()
    near, far = _pitch_targets(tech, cfg)
    name = f"central_dec_{rows}"
    decoder = central_decoder(tech, cfg, logic, near, far, name=name)
    gds = tmp_path / f"{name}.gds"
    decoder.write_gds(gds)
    report = run_drc(gds)
    assert report.clean, str(report)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(
        "\n".join([inverter_subckt(tech, logic), *gate_subckts(tech, logic),
                   row_decoder_subckt(tech, cfg, logic, name=name, banked=True)]) + "\n"
    )
    ports = {p.name: tuple(p.dcenter) for p in decoder.ports if not p.name.endswith("_right")}
    result = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs", ports=ports)
    assert result.matched and not result.mismatched_nets, str(result)
    assert not result.port_errors, str(result)


# --- the whole banked macro ------------------------------------------------------------------
# Sizes no other test file builds: gc_array names its cell after the config, so building the
# same array config twice in one session collides.


def test_banked_macro_pins(tech):
    """One address bit more than a bank needs, shared enables and supplies, and every data pin
    once per bank -- no raw word-line inputs."""
    cfg = ArrayConfig.basic(rows=10, cols=3, replica_column=True)
    macro = gc_banked_macro(tech, cfg, name="banked_pins_10x3")
    names = {port.name for port in macro.ports}
    bits = address_bits(cfg.rows)
    assert {f"addr_{i}" for i in range(bits + 1)} | {"we", "re", "pch", "vdd", "vss"} <= names
    assert f"addr_{bits + 1}" not in names
    for bank in (0, 1):
        assert {f"din_{bank}_{c}" for c in range(cfg.cols)} | {f"dout_{bank}_{c}" for c in range(cfg.cols)} <= names
        assert f"rdy_{bank}" in names
    assert not any(name.startswith(("wwl_in", "rwl_in", "din_2", "rdy_2")) for name in names)
    assert macro.info["bits"] == 2 * cfg.bits
    assert macro.info["banked"] is True
    assert macro.ports["din_1_0"].dcenter[0] > macro.info["bank_x"] > macro.ports["din_0_2"].dcenter[0]
    assert set(macro_pins(cfg, banked=True)) - names == {"vnw_0", "vnw_1"}
    # the shared predecode's strips stay within the rows, so nothing lifts the top edge beyond
    # the supply trunks just above the array
    geo = macro_geometry(tech, cfg)
    assert macro.dbbox().top < geo.array_y + geo.array.height + 2.5


@requires_drc
@requires_lvs
@pytest.mark.parametrize(
    "rows,cols,replica,neg",
    [(10, 6, False, False), (14, 5, True, False), (6, 7, False, True)],
)
def test_banked_macro_drc_and_lvs(tech, tmp_path, rows, cols, replica, neg):
    """Two banks, the central decoder, the metal3 lines to the outer driver columns, the
    precharge bridge (and the vneg link) are DRC clean and match ``macro_netlist(banked=True)``
    pin for pin -- except each array's own n-well bias, which never reaches the macro edge."""
    cfg = ArrayConfig.basic(rows=rows, cols=cols, replica_column=replica, negative_write_wl=neg)
    name = f"banked_{rows}x{cols}{'_r' if replica else ''}{'_n' if neg else ''}"
    macro = gc_banked_macro(tech, cfg, LogicConfig(), name=name)
    gds = tmp_path / f"{name}.gds"
    macro.write_gds(gds)
    report = run_drc(gds)
    assert report.clean, str(report)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(macro_netlist(tech, cfg, LogicConfig(), name=name, banked=True))
    ports = {p.name: tuple(p.dcenter) for p in macro.ports}
    result = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs", ports=ports)
    assert result.matched and not result.mismatched_nets, str(result)
    assert result.net_count[0] == result.net_count[1]
    assert sorted(result.pin_mismatches, key=str) == [(None, "vnw_0"), (None, "vnw_1")], str(result)


_PLAIN = """
from gc_compiler.array.floorplan import gc_macro
from gc_compiler.config import ArrayConfig
from gc_compiler.tech.sky130 import Sky130
print(gc_macro(Sky130(), ArrayConfig.basic(rows={rows}, cols={cols}), decoded=True).info["area_per_bit"])
"""


def test_banked_macro_beats_two_plain_decoded_macros(tech):
    """The point of it: one predecode for two banks, with the shared bus lines below the rows
    where the column band is empty, costs less area per bit than a plain decoded macro of the
    same bank size (13% at 32x32 when this was written)."""
    rows, cols = 32, 30
    plain = subprocess.run(
        [sys.executable, "-c", _PLAIN.format(rows=rows, cols=cols)], capture_output=True, text=True, check=True
    )
    plain_area = float(plain.stdout.strip().splitlines()[-1])
    banked = gc_banked_macro(tech, ArrayConfig.basic(rows=rows, cols=cols), name="banked_area_32x30")
    assert banked.info["area_per_bit"] < 0.92 * plain_area


def test_compiler_and_cli_build_a_banked_macro(tech, tmp_path):
    """``build(banked=True)`` writes the banked layout, its matching netlist and metrics that
    count both banks' bits; banking needs the decoder (it is what picks the bank)."""
    from gc_compiler.cli import main
    from gc_compiler.compiler import GainCellCompiler

    cfg = ArrayConfig.basic(rows=4, cols=5)
    compiler = GainCellCompiler(cfg, tech)
    with pytest.raises(ValueError, match="decoded"):
        compiler.build(tmp_path, macro=True, decoded=False, banked=True)
    result = compiler.build(tmp_path, name="cli_banked_4x5", macro=True, decoded=True, banked=True)
    assert result.banked and result.metrics["banked"]
    assert result.metrics["bits"] == 2 * cfg.bits
    assert result.metrics["macro_area_per_bit_um2"] == pytest.approx(
        result.component.info["width"] * result.component.info["height"] / (2 * cfg.bits)
    )
    assert "gc_row_decoder_banked" in result.spice.read_text()
    assert "2 banks of 4x5" in str(result)
    with pytest.raises(SystemExit):
        main(["--rows", "4", "--cols", "5", "--no-decoded", "--banked", "--out", str(tmp_path)])


_model_file = Path(Sky130().model_library()).parent / "corners" / "tt.spice"
requires_timing = pytest.mark.skipif(
    shutil.which("magic") is None
    or shutil.which("ngspice") is None
    or not default_magicrc().exists()
    or not _model_file.exists(),
    reason="magic, ngspice or the sky130 model files are missing",
)


@requires_timing
def test_each_bank_reads_through_its_own_pins(tech, tmp_path):
    """``build_model(bank=b)`` keeps bank b's array, drops the other's cells, and the benches
    drive ``addr_n`` (the bank bit) and bank b's ``din``/``dout``/``rdy``: a fresh '1' reads
    back in both banks, the replica tracks it, and a write stores it."""
    from gc_compiler.verify.timing import (
        BenchSettings, access_energy, build_model, simulate_reads, simulate_writes, standard_corners,
    )
    from gc_compiler.verify.parasitics import extract_flat

    cfg = ArrayConfig.basic(rows=6, cols=4, replica_column=True)
    name = "banked_timing_6x4"
    macro = gc_banked_macro(tech, cfg, name=name)
    gds = tmp_path / f"{name}.gds"
    macro.write_gds(gds)
    flat = extract_flat(gds, name, tmp_path / "extract")
    tt = next(c for c in standard_corners() if c.process == "tt")
    for bank in (0, 1):
        model = build_model(tech, cfg, gds, name, tmp_path / f"b{bank}", bank=bank, flat=flat)
        assert model.pin("din_2") == f"din_{bank}_2" and model.pin("rdy") == f"rdy_{bank}"
        assert len({key for key in model.sn}) == cfg.rows * 3  # two kept columns and the replica
        read = simulate_reads(tech, model, tt, cfg.rows - 1, {"3": (1, tt.vdd)}, tmp_path / f"r{bank}", BenchSettings())["3"]
        assert read.correct and read.t_dout is not None and read.t_rdy is not None
        assert 0.8 < read.t_dout / read.t_rdy < 1.3
        write = simulate_writes(tech, model, tt, 0, {"0": 1}, tmp_path / f"w{bank}", BenchSettings())["0"]
        assert write.sn_final > 0.9 * tt.vdd and write.t_write is not None
    # energy per access: a read of '1's discharges bit lines a read of '0's leaves alone
    energy = access_energy(tech, model, tt, tmp_path / "energy")
    assert energy.read1_j > energy.read0_j > 0 and energy.write_j > 0 and energy.idle_w > 0
