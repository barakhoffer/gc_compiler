from __future__ import annotations

import shutil

import pytest

from gc_compiler.compiler import GainCellCompiler
from gc_compiler.config import ArrayConfig, BitcellConfig
from gc_compiler.netlist.spice import array_netlist, bitcell_subckt
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.drc import LEAF_RULES, default_deck, run_drc


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


def test_bitcell_subckt_wires_the_3t_topology(tech):
    text = bitcell_subckt(tech, BitcellConfig.plain())
    assert ".subckt gc3t wwl wbl rwl rbl vss vnw" in text
    assert "Xmw wbl wwl sn vnw sky130_fd_pr__pfet_01v8" in text
    assert "Xmr n_int sn vss vss sky130_fd_pr__nfet_01v8" in text
    assert "Xms rbl rwl n_int vss sky130_fd_pr__nfet_01v8" in text


def test_array_netlist_instantiates_every_bit(tech):
    cfg = ArrayConfig.basic(rows=4, cols=3)
    text = array_netlist(tech, cfg)
    assert text.count("gc3t\n") + text.count("gc3t") >= cfg.bits
    assert "Xbit_3_2 wwl_3 wbl_2 rwl_3 rbl_2 vss vnw gc3t" in text


def test_compiler_writes_all_outputs(tech, tmp_path):
    result = GainCellCompiler(ArrayConfig.basic(rows=4, cols=4), tech).build(tmp_path, name="tiny_array", macro=False)
    assert result.gds.exists() and result.spice.exists() and result.metrics_file.exists()
    assert result.metrics["bits"] == 16
    assert 0 < result.metrics["array_efficiency"] <= 1


drc_available = shutil.which("klayout") is not None and default_deck().exists()
requires_drc = pytest.mark.skipif(not drc_available, reason="klayout or the sky130 deck is missing")


@requires_drc
def test_bitcell_is_drc_clean(tech, tmp_path):
    from gc_compiler.cells.bitcell_3t import gain_cell_3t

    gds = tmp_path / "gc3t.gds"
    gain_cell_3t(tech, name="drc_gc3t").write_gds(gds)
    report = run_drc(gds, waive=LEAF_RULES)  # a lone cell: its taps are its neighbours'
    assert report.clean, str(report)


@requires_drc
def test_small_array_is_drc_clean(tech, tmp_path):
    result = GainCellCompiler(ArrayConfig.basic(rows=8, cols=4), tech).build(tmp_path, name="drc_array", macro=False)
    report = run_drc(result.gds)
    assert report.clean, str(report)


def test_compiler_builds_a_decoded_macro(tech, tmp_path):
    cfg = ArrayConfig.basic(rows=4, cols=3)
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(tmp_path, name="decoded_macro_test", macro=True, decoded=True, banked=False)
    assert result.metrics["decoded"] is True
    assert result.metrics["macro_width_um"] == pytest.approx(result.component.info["width"])
    text = result.spice.read_text()
    assert "Xdecoder_left" in text and "Xdecoder_right" in text
    assert {p.name for p in result.component.ports} >= {"addr_0", "addr_1", "we", "re"}


def test_decoded_needs_a_macro(tech, tmp_path):
    with pytest.raises(ValueError, match="macro"):
        GainCellCompiler(ArrayConfig.basic(rows=4, cols=3), tech).build(tmp_path, macro=False, decoded=True, banked=False)


@requires_drc
def test_replica_cell_is_drc_clean(tech, tmp_path):
    from gc_compiler.cells.bitcell_3t import gain_cell_3t

    gds = tmp_path / "gc3t_rep.gds"
    gain_cell_3t(tech, name="drc_gc3t_rep", replica=True).write_gds(gds)
    report = run_drc(gds, waive=LEAF_RULES)
    assert report.clean, str(report)


@requires_drc
def test_replica_macro_is_drc_clean(tech, tmp_path):
    cfg = ArrayConfig.basic(rows=8, cols=4, replica_column=True)
    result = GainCellCompiler(cfg, tech).build(tmp_path, name="drc_rep_macro", macro=True, decoded=True, banked=False)
    report = run_drc(result.gds)
    assert report.clean, str(report)


def test_replica_netlist_has_the_extra_column(tech, tmp_path):
    cfg = ArrayConfig.basic(rows=4, cols=3, replica_column=True)
    result = GainCellCompiler(cfg, tech).build(tmp_path, name="rep_netlist_test", macro=True, decoded=True, banked=False)
    text = result.spice.read_text()
    assert text.count("Xrep_") == cfg.rows
    assert ".subckt gc3t_rep" in text
    assert "Xrd_rep rbl_rep rdy" in text
    assert "rdy" in {p.name for p in result.component.ports}
    assert result.metrics["replica_column"] is True
    assert result.metrics["bits"] == 12  # the replica column stores nothing


def test_characterise_needs_a_decoded_macro(tech, tmp_path):
    compiler = GainCellCompiler(ArrayConfig.basic(rows=6, cols=5), tech)
    result = compiler.build(tmp_path, name="char_undecoded", macro=True, decoded=False)
    with pytest.raises(ValueError, match="decoded"):
        compiler.characterise(result)


requires_lvs = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("magic", "netgen")),
    reason="magic or netgen is missing",
)


@requires_drc
@requires_lvs
def test_the_default_decoded_macro_is_banked_self_timed_and_clean(tech, tmp_path):
    """With nothing chosen, a decoded macro is two banks of the LVT cell with a replica column and
    self-timed reads -- and passes DRC and LVS."""
    from gc_compiler.verify.lvs import lvs_layout_vs_netlist

    cfg = ArrayConfig(rows=6, cols=13)
    assert cfg.bitcell == BitcellConfig() and cfg.bitcell.write_flavor is None and cfg.bitcell.mr_length == 1.0
    assert cfg.replica_column and cfg.self_timed_read
    result = GainCellCompiler(cfg, tech).build(tmp_path, name="default_decoded", macro=True, decoded=True)
    assert result.banked and result.metrics["bits"] == 2 * cfg.bits
    assert {"rdy_0", "rdy_1"} <= {p.name for p in result.component.ports}
    report = run_drc(result.gds)
    assert report.clean, str(report)
    ports = {p.name: tuple(p.dcenter) for p in result.component.ports}
    lvs = lvs_layout_vs_netlist(result.gds, "default_decoded", result.spice, "default_decoded",
                                workdir=tmp_path / "lvs", ports=ports)
    assert lvs.matched and not lvs.mismatched_nets, str(lvs)
    assert all((l or r or "").startswith("vnw") for l, r in lvs.pin_mismatches), str(lvs)


def test_an_undecoded_macro_ignores_self_timing(tech, tmp_path):
    """Self-timing gates the decoder, so a default config still builds an undecoded macro (whose
    rwl_in pins are driven directly)."""
    result = GainCellCompiler(ArrayConfig(rows=4, cols=5), tech).build(tmp_path, name="default_undecoded", macro=True, decoded=False)
    assert not result.banked and "rwl_in_0" in {p.name for p in result.component.ports}


def test_the_default_build_is_the_decoded_banked_macro(tech, tmp_path):
    from gc_compiler.cli import build_parser

    args = build_parser().parse_args([])
    assert args.macro and args.decoded is None  # None: follows --macro
    result = GainCellCompiler(ArrayConfig(rows=4, cols=5), tech).build(tmp_path, name="default_build")
    assert result.decoded and result.banked
    assert any(p.name.startswith("addr") for p in result.component.ports)
    array = GainCellCompiler(ArrayConfig(rows=4, cols=5), tech).build(tmp_path, name="default_array_only", macro=False)
    assert not array.decoded and "macro_width_um" not in array.metrics
