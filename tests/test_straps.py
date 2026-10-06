"""Word-line straps: metal3 over every word line, tied to its poly in each strap column."""

from __future__ import annotations

import shutil
from pathlib import Path

import klayout.db as kdb
import pytest

from gc_compiler.array.floorplan import gc_banked_macro, gc_macro, macro_geometry
from gc_compiler.array.tiler import array_geometry, gc_array
from gc_compiler.cells.bitcell_3t import bitcell_geometry
from gc_compiler.config import ArrayConfig, BitcellConfig
from gc_compiler.netlist.spice import macro_netlist
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.periphery.wl_driver import far_track_y, strap_track_y, strap_width, wordline_y
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
_model_file = Path(Sky130().model_library()).parent / "corners" / "tt.spice"
requires_timing = pytest.mark.skipif(
    shutil.which("magic") is None or shutil.which("ngspice") is None or not _model_file.exists(),
    reason="magic, ngspice or the sky130 model files are missing",
)
LVT_MR = BitcellConfig.plain(write_flavor="lvt", write_length=0.35, mr_length=0.5)


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


def test_straps_need_strap_columns():
    with pytest.raises(ValueError, match="strap columns"):
        ArrayConfig.basic(wordline_straps=True, strap_column_interval=0)
    assert ArrayConfig.basic(rows=4, cols=2, wordline_straps=True).name == "gc_array_4x2_ws_plain"


@pytest.mark.parametrize("bitcell", [BitcellConfig.plain(), LVT_MR], ids=["plain", "lvt_mr"])
def test_straps_and_far_tracks_keep_metal3_spacing(tech, bitcell):
    """Four straps and two far-output tracks per row pair, all a metal3 pitch apart -- also
    across the pair's edges, where the next pair's (mirrored) tracks are."""
    cell = bitcell_geometry(tech, bitcell)
    pair = 2 * cell.pitch_y
    straps = strap_track_y(tech, cell)
    tracks = sorted([*straps.values(), *far_track_y(tech, cell)])
    pitch = strap_width(tech) + tech.rules.s(LayerRole.M3)
    wrapped = [*tracks, pair + tracks[0]]  # the next pair starts with the same track
    assert all(b - a >= pitch - 1e-9 for a, b in zip(wrapped, wrapped[1:]))
    # each strap stays near its word line (its via stack jogs only a little on metal2)
    wl = wordline_y(cell)
    assert all(abs(straps[net] - wl[net]) < 0.2 for net in wl)


def test_array_draws_one_strap_per_word_line(tech):
    cfg = ArrayConfig.basic(rows=4, cols=20, wordline_straps=True)
    geo = array_geometry(tech, cfg)
    array = gc_array(tech, cfg, name="ws_array_probe")
    m3 = array.kcl.layer(*tech.layer(LayerRole.M3))
    region = kdb.Region(array.kdb_cell.begin_shapes_rec(m3)).merged()
    full = [p for p in region.each() if p.bbox().width() * array.kcl.dbu >= geo.width - 1e-6]
    assert len(full) == 2 * cfg.rows  # a read and a write word line per row


def _check(tech, cfg, tmp_path, name, banked=False):
    logic = LogicConfig()
    macro = gc_banked_macro(tech, cfg, logic, name=name) if banked else gc_macro(tech, cfg, logic, name=name, decoded=True)
    gds = tmp_path / f"{name}.gds"
    macro.write_gds(gds)
    report = run_drc(gds)
    assert report.clean, str(report)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(macro_netlist(tech, cfg, logic, name=name, decoded=True, banked=banked))
    ports = {p.name: tuple(p.dcenter) for p in macro.ports}
    result = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs", ports=ports)
    assert result.matched and not result.mismatched_nets, str(result)
    # every pin checked but the arrays' n-well bias (never a macro pin by design)
    assert all((l or r or "").startswith("vnw") for l, r in result.pin_mismatches), str(result)
    return macro


@requires_drc
@requires_lvs
@pytest.mark.parametrize(
    "rows, cols, extra",
    [(4, 20, {}), (4, 18, {"negative_write_wl": True, "replica_column": True}), (4, 22, {"bitcell": LVT_MR})],
    ids=["plain", "neg_rep", "lvt_mr"],
)
def test_strapped_macro_drc_and_lvs(tech, tmp_path, rows, cols, extra):
    """Straps add no devices and no nets: a strapped macro matches the same schematic, pin for pin."""
    cfg = ArrayConfig.basic(rows=rows, cols=cols, wordline_straps=True, **extra)
    macro = _check(tech, cfg, tmp_path, f"ws_macro_{rows}x{cols}")
    plain = macro_geometry(tech, ArrayConfig.basic(rows=rows, cols=cols, **extra), LogicConfig())
    assert macro_geometry(tech, cfg, LogicConfig()).width == pytest.approx(plain.width)  # no area


@requires_drc
@requires_lvs
def test_strapped_banked_macro_drc_and_lvs(tech, tmp_path):
    """The far outputs move to the spare tracks between the straps and jog down to their
    drivers outside the outer columns."""
    _check(tech, ArrayConfig.basic(rows=6, cols=18, wordline_straps=True), tmp_path, "ws_banked_6x18", banked=True)


@requires_timing
def test_straps_speed_up_the_far_column(tech, tmp_path):
    """The timing model ties each word line to its strap at every strap column, and the far
    column's word line then rises sooner."""
    from gc_compiler.verify.timing import BenchSettings, build_model, simulate_reads, standard_corners

    tt = next(c for c in standard_corners() if c.process == "tt")
    driver = {}
    for straps in (False, True):
        cfg = ArrayConfig.basic(rows=4, cols=48, wordline_straps=straps)
        name = f"ws_timing_{int(straps)}"
        gds = tmp_path / f"{name}.gds"
        gc_macro(tech, cfg, name=name, decoded=True).write_gds(gds)
        model = build_model(tech, cfg, gds, name, tmp_path / name)
        text = model.netlist.read_text()
        taps = len(array_geometry(tech, cfg).strap_col_x) if straps else 0
        assert text.count("__s") >= 2 * cfg.rows * taps  # a strap node per word line per strap column
        far = str(cfg.cols - 1)
        reads = simulate_reads(tech, model, tt, 0, {far: (1, 1.8), "0": (1, 1.8)}, tmp_path / name / "r", BenchSettings())
        driver[straps] = max(t.t_driver for t in reads.values())
    assert driver[True] < 0.9 * driver[False]
