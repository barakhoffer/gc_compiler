"""A supply-free alternative to the negative-rail driver: an LVT write device (leaves a lower
'0') paired with a longer read gate (tolerates a higher one). Both are plain ``BitcellConfig``
fields -- no new device-stack code, just a couple of new values -- so this file mostly checks
that the combination actually reads a written '0' back correctly, on the extracted macro, at
every corner. ``mr_length`` (below) is the one genuinely new field: it lengthens only ``mr``
(the storage-node-gated device the tolerance depends on), not ``ms`` (the rwl-gated select
device, which lengthening does nothing for), so it is worth its own DRC/LVS/timing checks."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gc_compiler.array.floorplan import gc_macro
from gc_compiler.cells.bitcell_3t import bitcell_geometry
from gc_compiler.compiler import GainCellCompiler
from gc_compiler.config import ArrayConfig, BitcellConfig
from gc_compiler.netlist.spice import bitcell_subckt, macro_netlist
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.drc import default_deck, run_drc
from gc_compiler.verify.lvs import default_magicrc, default_netgen_setup, lvs_layout_vs_netlist
from gc_compiler.verify.timing import standard_corners

LOGIC = LogicConfig()
LVT_WRITE = BitcellConfig.plain(write_flavor="lvt", write_length=0.35, read_length=0.5)
# Same tolerated '0', but only mr (the storage-node-gated device) pays for it; ms (the rwl-gated
# select device, and the word line's own poly width) stays at the default length.
LVT_WRITE_MR = BitcellConfig.plain(write_flavor="lvt", write_length=0.35, mr_length=0.5)

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
_model_file = Path(Sky130().model_library()).parent / "corners" / "tt.spice"
requires_timing = pytest.mark.skipif(
    shutil.which("magic") is None
    or shutil.which("ngspice") is None
    or not default_magicrc().exists()
    or not _model_file.exists(),
    reason="magic, ngspice or the sky130 model files are missing",
)


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


# --- configuration and cost -----------------------------------------------------------------


def test_lvt_write_is_a_plain_bitcell_config(tech):
    """No dedicated option: an LVT write device and a longer read gate are ``BitcellConfig``
    fields any bitcell can set, so a macro built from one needs no new driver or ``pb`` line."""
    cfg = ArrayConfig.basic(rows=4, cols=2, bitcell=LVT_WRITE)
    assert cfg.negative_write_wl is False
    assert cfg.name != ArrayConfig.basic(rows=4, cols=2).name  # a different bitcell is a different array
    assert "gc_wwl_neg" not in macro_netlist(tech, cfg, LOGIC, name="p", decoded=False)


def test_compiler_reports_the_cost_of_lvt_write(tech, tmp_path):
    compiler = GainCellCompiler(ArrayConfig.basic(rows=4, cols=3, bitcell=LVT_WRITE), tech)
    result = compiler.build(tmp_path, name="lvt_metrics", macro=True, decoded=False)
    plain = GainCellCompiler(ArrayConfig.basic(rows=4, cols=3), tech).metrics(macro=True)
    m = result.metrics
    assert m["transistors"] == plain["transistors"]  # same device count, just resized/reflavoured
    assert m["macro_area_um2"] > plain["macro_area_um2"]  # the LVT marker widens the cell


def _device_params(subckt_text: str, ref: str) -> str:
    line = next(line for line in subckt_text.splitlines() if line.startswith(f"X{ref} "))
    return line


def test_mr_length_defaults_to_read_length(tech):
    assert BitcellConfig.plain().effective_mr_length == BitcellConfig.plain().read_length
    assert BitcellConfig.plain(read_length=0.5).effective_mr_length == 0.5
    cfg = BitcellConfig.plain(read_length=0.15, mr_length=0.5)
    assert cfg.effective_mr_length == 0.5 and cfg.read_length == 0.15


def test_mr_length_only_lengthens_mr_in_the_netlist(tech):
    """``ms``, the rwl-gated select device, keeps ``read_length`` even when ``mr_length`` grows;
    only ``mr``, gated by the storage node, sees the longer channel."""
    text = bitcell_subckt(tech, LVT_WRITE_MR)
    assert "l=0.5" in _device_params(text, "mr")
    assert "l=0.15" in _device_params(text, "ms")  # BitcellConfig.read_length's default


def test_mr_length_grows_the_cell_less_than_lengthening_both_gates(tech):
    """Both options tolerate the same '0' (same ``mr`` length); ``LVT_WRITE`` pays for it on
    ``ms`` too, ``LVT_WRITE_MR`` does not, so the cell (and the rwl word line's own poly width)
    should be smaller."""
    both = bitcell_geometry(tech, LVT_WRITE)
    mr_only = bitcell_geometry(tech, LVT_WRITE_MR)
    assert mr_only.pitch_y < both.pitch_y
    assert mr_only.rwl_width == pytest.approx(BitcellConfig.plain().read_length)
    assert both.rwl_width == pytest.approx(LVT_WRITE.read_length)


# --- DRC / LVS -------------------------------------------------------------------------------


@requires_drc
@requires_lvs
@pytest.mark.parametrize(
    "rows,cols,decoded,replica",
    [(8, 4, False, False), (12, 6, True, False), (16, 20, True, True)],
)
def test_lvt_write_macro_drc_and_lvs(tech, tmp_path, rows, cols, decoded, replica):
    cfg = ArrayConfig.basic(rows=rows, cols=cols, bitcell=LVT_WRITE, replica_column=replica)
    name = f"lvt_m_{rows}x{cols}{'_d' if decoded else '_p'}{'_r' if replica else ''}"
    gds = tmp_path / f"{name}.gds"
    component = gc_macro(tech, cfg, LOGIC, name=name, decoded=decoded)
    component.write_gds(gds)
    report = run_drc(gds)
    assert report.clean, str(report)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(macro_netlist(tech, cfg, LOGIC, name=name, decoded=decoded))
    ports = {p.name: tuple(p.dcenter) for p in component.ports}
    result = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs", ports=ports)
    assert result.matched and not result.mismatched_nets, str(result)
    assert result.net_count[0] == result.net_count[1]
    # every declared pin is verified by name except vnw, which has no macro-level pin at all
    # (see magic_extract's docstring) -- anything more here would be a real regression.
    assert result.pin_mismatches == [(None, "vnw")], str(result)


@requires_drc
@requires_lvs
def test_lvt_write_mr_only_macro_drc_and_lvs(tech, tmp_path):
    """The one physically new thing ``mr_length`` produces: two different-length gates in series
    on one diffusion strip. Everything that draws and spaces them is generic (``plan_stack``), but
    this is the first bitcell to actually exercise it, so it gets its own DRC/LVS check."""
    cfg = ArrayConfig.basic(rows=12, cols=6, bitcell=LVT_WRITE_MR, replica_column=True)
    name = "lvt_mr_m_12x6_d_r"
    gds = tmp_path / f"{name}.gds"
    component = gc_macro(tech, cfg, LOGIC, name=name, decoded=True)
    component.write_gds(gds)
    report = run_drc(gds)
    assert report.clean, str(report)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(macro_netlist(tech, cfg, LOGIC, name=name, decoded=True))
    ports = {p.name: tuple(p.dcenter) for p in component.ports}
    result = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs", ports=ports)
    assert result.matched and not result.mismatched_nets, str(result)
    assert result.net_count[0] == result.net_count[1]
    assert result.pin_mismatches == [(None, "vnw")], str(result)


# --- the point of it: a written '0' now reads back correctly, with no extra supply -----------


@requires_timing
def test_a_written_zero_reads_back_in_every_corner_without_a_negative_supply(tech, tmp_path):
    cfg = ArrayConfig.basic(rows=8, cols=5, replica_column=True, bitcell=LVT_WRITE)
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(tmp_path, name="lvt_char", macro=True, decoded=True, banked=False)
    report = compiler.characterise(result, corners=standard_corners(), levels=(1.0,), monte_carlo=3)

    assert report.vneg is None  # no negative rail: nothing to report

    # local mismatch on top: every die's own write leaves a '0' that still reads as one
    for c in report.corners:
        assert c.mc.n == 3 and c.mc.zero_failures == 0, c.corner.name

    for c in report.corners:
        assert c.reads_zero_correctly, c.corner.name
        assert c.sn_one == pytest.approx(c.corner.vdd, abs=0.03)  # writing a '1' is unaffected


@requires_timing
def test_mr_length_alone_reads_back_a_written_zero_just_like_lengthening_both_gates(tech, tmp_path):
    """Same claim as above, for the cheaper option: only ``mr`` grows."""
    cfg = ArrayConfig.basic(rows=8, cols=5, replica_column=True, bitcell=LVT_WRITE_MR)
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(tmp_path, name="lvt_mr_char", macro=True, decoded=True, banked=False)
    report = compiler.characterise(result, corners=standard_corners(), levels=(1.0,), monte_carlo=3)

    assert report.vneg is None
    for c in report.corners:
        assert c.mc.n == 3 and c.mc.zero_failures == 0, c.corner.name
    for c in report.corners:
        assert c.reads_zero_correctly, c.corner.name
        assert c.sn_one == pytest.approx(c.corner.vdd, abs=0.03)
