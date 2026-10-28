"""IHP SG13G2: macros signed off with IHP's own KLayout decks (main and extra rules) and extracted
with IHP's magic setup for netgen LVS. Skipped without the PDK (``ciel enable --pdk-family ihp-sg13g2``)
or the tools; the rules check of every structure is in ``test_portability.py``."""

from __future__ import annotations

import re
import shutil

import pytest

from gc_compiler.compiler import GainCellCompiler
from gc_compiler.config import ArrayConfig
from gc_compiler.tech.sg13g2 import SG13G2
from gc_compiler.verify.drc import default_deck, run_drc
from gc_compiler.verify.lvs import lvs_layout_vs_netlist

IHP = SG13G2()

installed = default_deck(tech=IHP).exists() and IHP.tools.magicrc_path().exists()
tools = all(shutil.which(tool) for tool in ("klayout", "magic", "netgen"))
requires_ihp = pytest.mark.skipif(not (installed and tools), reason="the IHP SG13G2 PDK or klayout/magic/netgen is missing")


def test_the_netlist_uses_the_ihp_devices(tmp_path):
    result = GainCellCompiler(ArrayConfig.for_tech(IHP, rows=4, cols=4), IHP).build(
        tmp_path, name="ihp_netlist", macro=False
    )
    text = result.spice.read_text()
    assert "sg13_lv_pmos" in text and "sg13_lv_nmos" in text
    assert "w=0.3u l=0.13u" in text


@requires_ihp
@pytest.mark.parametrize("kind", ["one_bank", "banked", "negative_rail", "refresh_in_macro"])
def test_decoded_macro_is_drc_and_lvs_clean(tmp_path, kind):
    banked = kind == "banked"
    cfg = ArrayConfig.for_tech(IHP, rows=8, cols=8, negative_write_wl=kind == "negative_rail",
                               refresh_in_macro=kind == "refresh_in_macro")
    name = f"ihp_macro_{kind}"
    result = GainCellCompiler(cfg, IHP).build(tmp_path, name=name, macro=True, decoded=True, banked=banked)
    drc = run_drc(result.gds, tech=IHP)
    assert drc.clean, str(drc)
    ports = {p.name: tuple(p.dcenter) for p in result.component.ports}
    lvs = lvs_layout_vs_netlist(
        result.gds, result.component.name, result.spice, result.component.name, workdir=tmp_path / "lvs",
        ports=ports, tech=IHP,
    )
    assert lvs.matched, lvs.report[-3000:]


simulates = installed and all(shutil.which(tool) for tool in ("ngspice", "openvaf"))
requires_sim = pytest.mark.skipif(not simulates, reason="the IHP SG13G2 PDK, ngspice or OpenVAF is missing")


@requires_sim
def test_the_psp_models_compile_and_simulate(tmp_path):
    """IHP ships its MOSFET models as Verilog-A: the adapter compiles them (OpenVAF) and a bench
    loads them. A stored '1' reads (the read bit line discharges) and a stored '0' does not."""
    from gc_compiler.verify.sim import RetentionTestbench, run_retention

    one = run_retention(IHP, tb=RetentionTestbench(data=1, wwl_low=0.0, hold_time=20e-6, reads=2), workdir=tmp_path)
    zero = run_retention(IHP, tb=RetentionTestbench(data=0, wwl_low=0.0, hold_time=20e-6, reads=2), workdir=tmp_path)
    assert all(level < 0.2 for _t, level in one.read_levels())
    assert all(level > 0.8 for _t, level in zero.read_levels())


@requires_sim
def test_the_mismatch_library_varies_the_devices(tmp_path):
    """IHP turns mismatch on per instance (mm_ok=1): the adapter's mismatch library makes it the
    default, so two identical devices differ, and differ again with another seed."""
    import subprocess

    currents = []
    for seed in (1, 2):
        deck = tmp_path / f"mm{seed}.sp"
        deck.write_text("\n".join([
            "* mismatch", *IHP.model_header("tt", mismatch=True), f".option seed={seed}",
            "X1 d1 g 0 0 sg13_lv_nmos w=0.3u l=0.13u", "X2 d2 g 0 0 sg13_lv_nmos w=0.3u l=0.13u",
            "v1 d1 0 1.2", "v2 d2 0 1.2", "vg g 0 0.6",
            ".control", "op", "print i(v1)", "print i(v2)", ".endc", ".end", "",
        ]))
        out = subprocess.run(["ngspice", "-b", str(deck)], capture_output=True, text=True, cwd=tmp_path).stdout
        currents.append([float(line.split("=")[1]) for line in out.splitlines() if line.startswith("i(v")])
    assert currents[0][0] != pytest.approx(currents[0][1], rel=1e-3)
    assert currents[0] != pytest.approx(currents[1], rel=1e-3)


blocks = installed and all(shutil.which(tool) for tool in ("librelane", "iverilog", "klayout"))
requires_block = pytest.mark.skipif(not blocks, reason="the IHP SG13G2 PDK, librelane, iverilog or klayout is missing")


@requires_block
@pytest.mark.parametrize("banked,rim", [(False, False), (True, False), (False, True)], ids=["one_bank", "banked", "refresh_in_macro"])
def test_memory_block_is_clean_timed_and_passes_its_testbench(tmp_path, banked, rim):
    """The macro with its refresh controller, placed and routed by LibreLane on SG13G2: its power
    pins on TopMetal1 under IHP's grid (a strap of each net aligned onto the macro's stripes), DRC
    (IHP's KLayout decks, the extra rules too) and LVS clean, timing met, testbench passing."""
    cfg = ArrayConfig.for_tech(IHP, rows=8, cols=8, refresh_in_macro=rim)
    compiler = GainCellCompiler(cfg, IHP)
    memory = compiler.memory(tmp_path, banked=banked)
    m = memory.metrics
    assert (m["loopback"] == {}) == rim  # the macro's own write-back leaves no loop-back to time
    assert m["drc_errors"] == 0 and m["lvs_errors"] == 0
    assert all(t["setup_ns"] >= 0 and t["hold_ns"] >= 0 for t in m["timing"].values())
    # no copy of the user's request: control flops and a dout latch a column; the flow settings took
    # (no 16x clock buffer at the tree roots), and the cells near the library's area per flop
    assert m["flops"] <= cfg.cols + 40  # (control, with the handshake's synchronisers, about 40)
    assert re.search(r"\b\w*buf_16\b", memory.file(".v").read_text()) is None
    assert m["controller_cell_area_um2"] / m["flops"] <= 1.1 * IHP.standard_cells().area_per_flop_um2
    drc = run_drc(memory.file(".gds"), tech=IHP)
    assert drc.clean, str(drc)


def test_the_grid_is_aligned_onto_the_macros_stripes():
    """A vdd strap and the vss strap after it land right on a vdd and a vss stripe of the macro."""
    from pathlib import Path

    from gc_compiler.flow.librelane import LEFT_MARGIN_SITES, MacroViews, _aligned_grid

    lib = IHP.standard_cells()
    stripes = (("vdd", 2.6), ("vss", 25.1), ("vdd", 46.1), ("vss", 52.7), ("vnw", 75.6))
    views = MacroViews("m", Path("m.gds"), Path("m.lef"), Path("bb"), Path("pbb"), Path("s"), {}, 80.0, 50.0,
                       ("vdd", "vnw"), ("vss",), stripes)
    macro_x = 6.24
    grid = _aligned_grid(lib, views, macro_x)
    core_x0 = LEFT_MARGIN_SITES * lib.site_width
    vdd = core_x0 + grid["PDN_VOFFSET"]
    vss = vdd + grid["PDN_VWIDTH"] + grid["PDN_VSPACING"]
    pitch = lib.pdn.vertical_pitch
    on = lambda x, net: any(abs((x - (macro_x + sx)) % pitch) < 1e-6 or abs((x - (macro_x + sx)) % pitch - pitch) < 1e-6  # noqa: E731
                           for n, sx in stripes if n == net)
    assert on(vdd, "vdd") and on(vss, "vss")
    assert grid["PDN_VSPACING"] >= lib.pdn.vertical_min_spacing


@requires_sim
def test_in_macro_refresh_restores_on_sg13g2(tmp_path):
    """The macro's own write-back (refresh_in_macro) on an extracted SG13G2 macro, tt and ss: a '1'
    decayed to 0.7 vdd and a '0' at 0.5 V both come back, with the controller's write-back time
    (a PMOS write needs about 10 ns to pull a '0' down). At ss a '0' at 0.55 V is already above
    vdd/2, and the write-back leaves it there."""
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.verify.timing import build_model, simulate_refresh, standard_corners

    if shutil.which("magic") is None:
        pytest.skip("magic is missing")
    cfg = ArrayConfig.for_tech(IHP, rows=4, cols=20, refresh_in_macro=True)
    gds = tmp_path / "rim.gds"
    gc_macro(IHP, cfg, name="ihp_rim_test", decoded=True).write_gds(gds)
    model = build_model(IHP, cfg, gds, "ihp_rim_test", tmp_path / "model")
    near, far = model.kept[0], model.kept[-1]
    for corner in (c for c in standard_corners(IHP) if c.process in ("tt", "ss")):
        for cells in ({near: (1, 0.7 * corner.vdd), far: (0, 0.5)}, {near: (0, 0.5), far: (1, 0.7 * corner.vdd)}):
            result = simulate_refresh(
                IHP, model, corner, 0, cells, tmp_path / f"{corner.process}_{cells[near][0]}",
                write_time=IHP.characterised.controller.write_ns * 1e-9,
            )
            for col, r in result.items():
                assert r.restored, (corner.name, col, r)
