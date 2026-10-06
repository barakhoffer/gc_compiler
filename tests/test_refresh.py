"""Refresh: the single-access write-back on the extracted macro, and the generated controller."""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.spice import macro_pins
from gc_compiler.rtl import SYNC_CYCLES, ControllerTiming, controller_params, refresh_rtl
from gc_compiler.tech.sky130 import Sky130

_model_file = Path(Sky130().model_library()).parent / "corners" / "tt.spice"
requires_timing = pytest.mark.skipif(
    shutil.which("magic") is None or shutil.which("ngspice") is None or not _model_file.exists(),
    reason="magic, ngspice or the sky130 model files are missing",
)
requires_iverilog = pytest.mark.skipif(
    shutil.which("iverilog") is None or shutil.which("vvp") is None, reason="iverilog is missing"
)
FAST = ControllerTiming(clock_ns=10, read_ns=1.8, write_ns=1.0, precharge_ns=1.2, interval_us=10.0)


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


def test_controller_params_follow_the_clock_and_the_interval():
    p = controller_params(ArrayConfig.basic(rows=16, cols=8), FAST)
    assert (p.read_cycles, p.write_cycles, p.precharge_cycles) == (1, 1, 1)
    assert p.refresh_cycles == int(10_000 * 0.5 / 16 / 10)  # every row within half the interval
    banked = controller_params(ArrayConfig.basic(rows=16, cols=8), FAST, banked=True)
    assert banked.refresh_cycles == p.refresh_cycles // 2 and banked.addr_bits == p.addr_bits + 1
    with pytest.raises(ValueError, match="cannot refresh"):
        controller_params(ArrayConfig.basic(rows=256, cols=8), ControllerTiming(clock_ns=50, interval_us=10.0))


def test_timing_from_a_characterisation():
    data = {"corners": [
        {"access_s": 1.0e-9, "write_s": 0.2e-9, "precharge_s": 0.7e-9, "reads": [{"t_rdy": 0.9e-9}]},
        {"access_s": 1.9e-9, "write_s": 0.3e-9, "precharge_s": 1.1e-9, "reads": [{"t_rdy": 1.8e-9}]},
    ]}
    t = ControllerTiming.from_characterisation(data, clock_ns=5.0, interval_us=100.0)
    assert t.read_ns == pytest.approx(1.25 * 1.8)  # the strobe of the slowest corner
    assert t.write_ns == pytest.approx(1.0)  # three times the slowest write, at least 1 ns
    assert t.precharge_ns == pytest.approx(1.2 * 1.1)


@pytest.mark.parametrize("banked", [False, True], ids=["plain", "banked"])
def test_generated_verilog_matches_the_macro(banked):
    cfg = ArrayConfig.basic(rows=6, cols=4)
    files = refresh_rtl("mac", cfg, FAST, banked)
    assert set(files) == {"mac_ctrl.v", "mac_model.v", "mac_top.v", "mac_tb.v", "mac_ctrl.sdc"}
    assert "set_multicycle_path -setup" in files["mac_ctrl.sdc"]
    model = files["mac_model.v"]
    for pin in macro_pins(cfg, True, banked):
        if pin in ("vdd", "vss", "vnw", "vnw_0", "vnw_1"):
            assert f" {pin}" not in model.split(");")[0]  # no supplies in the model
        else:
            assert pin in model
    ctrl = files["mac_ctrl.v"]
    assert "module mac_ctrl" in ctrl and "refreshing ? ~macro_dout : ~din" in ctrl
    assert ("m_dout_1" in ctrl) == banked
    # no copy of the user's address and write data: they drive the macro directly; dout is a latch
    assert "reg [COLS-1:0] wdata" not in ctrl and "assign m_addr = refreshing ?" in ctrl
    assert "~din;" in ctrl and "always @(*) if (dout_open) dout = macro_dout;" in ctrl
    assert "wire dout_open = !clk && dout_open_q;" in ctrl  # a flop's output gates it: no glitch, half a cycle


def _simulate(tmp_path: Path, files: dict[str, str]) -> str:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    vvp = tmp_path / "sim.vvp"
    verilog = [str(tmp_path / f) for f in files if f.endswith(".v")]
    subprocess.run(["iverilog", "-g2005", "-o", str(vvp), *verilog], check=True)
    return subprocess.run(["vvp", str(vvp)], capture_output=True, text=True, timeout=600, check=True).stdout


@requires_iverilog
@pytest.mark.parametrize("banked", [False, True], ids=["plain", "banked"])
def test_controller_keeps_every_row(tmp_path, banked):
    """Random traffic, then two intervals with no user access: every row still reads back what
    was written -- and without refresh (the tick disabled) the same bench loses them."""
    cfg = ArrayConfig.basic(rows=6 if banked else 16, cols=4 if banked else 8)
    files = refresh_rtl("mac", cfg, FAST, banked)
    (tmp_path / "on").mkdir()
    out = _simulate(tmp_path / "on", files)
    assert "PASS" in out, out
    files["mac_tb.v"] = files["mac_tb.v"].replace(
        "defparam dut.macro.RETENTION_NS", "defparam dut.ctrl.REFRESH_CYCLES = 1 << 30;\n    defparam dut.macro.RETENTION_NS"
    )
    (tmp_path / "off").mkdir()
    assert "FAIL" in _simulate(tmp_path / "off", files)


@requires_timing
def test_one_access_restores_decayed_ones_and_zeros(tech, tmp_path):
    """The loop-back access on the extracted macro: a '1' decayed to 0.7 vdd comes back to the
    rail and a '0' decayed to 0.75 V back to its written level, near and far from the drivers."""
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.verify.timing import build_model, simulate_refresh, standard_corners

    cfg = ArrayConfig.basic(rows=4, cols=20, replica_column=True, self_timed_read=True)
    gds = tmp_path / "rf.gds"
    gc_macro(tech, cfg, name="rf_test", decoded=True).write_gds(gds)
    model = build_model(tech, cfg, gds, "rf_test", tmp_path / "model")
    near, far = model.kept[0], model.kept[-1]
    for corner in (c for c in standard_corners() if c.process in ("tt", "ss")):
        for cells in ({near: (1, 0.7 * corner.vdd), far: (0, 0.75)}, {near: (0, 0.75), far: (1, 0.7 * corner.vdd)}):
            result = simulate_refresh(tech, model, corner, 0, cells, tmp_path / f"{corner.process}_{len(cells)}_{cells[near][0]}")
            for col, r in result.items():
                assert r.read_ok and r.restored, (corner.name, col, r)
                if r.data:
                    assert r.sn_final > 0.95 * corner.vdd
                else:
                    assert r.sn_final < 0.75
                assert r.t_read < r.t_access < 10e-9


def test_controller_needs_a_self_timed_macro(tmp_path):
    """A fixed read time cannot be right at every corner, so the controller is only made for a
    macro that ends its reads itself."""
    from gc_compiler.compiler import GainCellCompiler

    compiler = GainCellCompiler(ArrayConfig.basic(rows=4, cols=13, replica_column=True))
    result = compiler.build(tmp_path, name="rf_fixed", macro=True, decoded=True, banked=False)
    with pytest.raises(ValueError, match="self_timed_read"):
        compiler.refresh_rtl(result)


def test_cli_writes_the_verilog(tmp_path):
    from gc_compiler.cli import main

    # a size no other test builds: gc_array names its cell after the config
    assert main(["--rows", "4", "--cols", "11", "--macro", "--decoded", "--replica", "--self-timed", "--refresh",
                 "--refresh-interval", "20", "--out", str(tmp_path), "--name", "rf_cli"]) == 0
    for suffix in ("ctrl", "model", "top", "tb"):
        assert (tmp_path / f"rf_cli_{suffix}.v").exists()
    assert "--refresh needs" not in (tmp_path / "rf_cli_ctrl.v").read_text()


def test_the_default_refresh_interval_follows_the_cell():
    from gc_compiler.config import BitcellConfig
    from gc_compiler.rtl import default_refresh_interval_us

    assert default_refresh_interval_us(ArrayConfig.basic(bitcell=BitcellConfig())) == 614.0
    assert default_refresh_interval_us(ArrayConfig.basic()) == 45.0
    assert default_refresh_interval_us(ArrayConfig.basic(negative_write_wl=True)) == 88.0
    assert default_refresh_interval_us(ArrayConfig.basic(negative_write_wl=True, write_wl_low=-0.2)) == 45.0
    lvt_both = BitcellConfig.plain(write_flavor="lvt", write_length=0.35, read_length=0.5)
    assert default_refresh_interval_us(ArrayConfig.basic(bitcell=lvt_both)) == 571.0
    assert default_refresh_interval_us(ArrayConfig.basic(bitcell=BitcellConfig(mr_length=0.6))) == 45.0


HANDSHAKE = dataclasses.replace(FAST, handshake=True)


def test_the_handshake_needs_no_read_time():
    cfg = ArrayConfig.basic(rows=16, cols=8, replica_column=True, self_timed_read=True)
    p = controller_params(cfg, HANDSHAKE)
    assert p.handshake and p.read_cycles == 5 + SYNC_CYCLES  # the timeout, not read_ns
    assert controller_params(cfg, dataclasses.replace(HANDSHAKE, read_ns=40.0)) == p
    with pytest.raises(ValueError, match="too fast"):  # dout must follow rdy within two cycles
        controller_params(cfg, dataclasses.replace(HANDSHAKE, clock_ns=1.0))
    files = refresh_rtl("mac", cfg, HANDSHAKE)
    assert "input  m_rdy" in files["mac_ctrl.v"] and ".rdy(m_rdy)" in files["mac_top.v"]
    assert "set_false_path -from [get_ports m_rdy*]" in files["mac_ctrl.sdc"]
    with pytest.raises(ValueError, match="replica"):
        refresh_rtl("mac", ArrayConfig.basic(rows=16, cols=8), HANDSHAKE)


def test_the_rdy_lag_from_a_characterisation():
    data = {"corners": [
        {"write_s": 0.2e-9, "reads": [{"t_rdy": 1.0e-9, "t_dout": 1.1e-9, "t_cutoff": 1.3e-9}]},
        {"write_s": 0.3e-9, "reads": [{"t_rdy": 1.8e-9, "t_dout": 2.3e-9, "t_cutoff": 2.2e-9}]},
    ]}
    t = ControllerTiming.from_characterisation(data, clock_ns=5.0, interval_us=100.0, handshake=True)
    assert t.handshake and t.rdy_lag_ns == pytest.approx(1.2 * 0.5)


@requires_iverilog
@pytest.mark.parametrize("banked", [False, True], ids=["plain", "banked"])
def test_the_handshake_waits_for_a_slow_macro(tmp_path, banked):
    """The model's replica fires 25 ns after re -- far beyond read_ns: the handshake controller
    still keeps every row; the fixed-time one, against the same macro, reads wrong; and with a
    replica that never fires the handshake's reads time out and say so."""
    cfg = ArrayConfig.basic(rows=6 if banked else 16, cols=4 if banked else 8, replica_column=True, self_timed_read=True)
    files = refresh_rtl("mac", cfg, HANDSHAKE, banked)
    assert "RDY_NS = 25" in files["mac_tb.v"]
    (tmp_path / "hs").mkdir()
    out = _simulate(tmp_path / "hs", files)
    assert "PASS" in out, out
    fixed = refresh_rtl("mac", cfg, FAST, banked)
    fixed["mac_tb.v"] = fixed["mac_tb.v"].replace(
        "defparam dut.macro.RETENTION_NS", "defparam dut.macro.DOUT_NS = 26;\n    defparam dut.macro.RETENTION_NS"
    )
    (tmp_path / "fixed").mkdir()
    assert "FAIL" in _simulate(tmp_path / "fixed", fixed)
    files["mac_tb.v"] = files["mac_tb.v"].replace("RDY_NS = 25", "RDY_NS = 1e9")
    (tmp_path / "stuck").mkdir()
    out = _simulate(tmp_path / "stuck", files)
    assert "a read timed out" in out and "FAIL" in out, out


def _rim(**kw) -> ArrayConfig:
    return ArrayConfig.basic(replica_column=True, self_timed_read=True, refresh_in_macro=True, **kw)


def test_refresh_in_macro_needs_self_timed_reads():
    with pytest.raises(ValueError, match="self-timed"):
        ArrayConfig.basic(rows=4, cols=4, refresh_in_macro=True)


@requires_iverilog
@pytest.mark.parametrize("banked", [False, True], ids=["plain", "banked"])
def test_controller_with_refresh_in_macro(tmp_path, banked):
    """With the macro writing a refresh back itself, the controller drives din high in a refresh
    (no loop-back from dout, nor its timing constraint) and drops pch a cycle after we; the bench
    keeps every row. A macro model that did not write back would lose them."""
    cfg = _rim(rows=6 if banked else 16, cols=4 if banked else 8)
    files = refresh_rtl("mac", cfg, FAST, banked)
    ctrl, sdc = files["mac_ctrl.v"], files["mac_ctrl.sdc"]
    assert "refreshing ? {COLS{1'b1}} : din_on ? ~din" in ctrl and "~macro_dout" not in ctrl
    assert "din_on ? ~din : {COLS{1'b0}}" in ctrl  # din low through user reads: the write bit lines stay still
    assert "-to [get_ports m_din*]" not in sdc
    loop = controller_params(dataclasses.replace(cfg, refresh_in_macro=False), FAST, banked)
    assert controller_params(cfg, FAST, banked).refresh_in_macro and not loop.refresh_in_macro
    (tmp_path / "on").mkdir()
    out = _simulate(tmp_path / "on", files)
    assert "PASS" in out, out
    model = files["mac_model.v"]
    assert "pch ? held" in model
    files["mac_model.v"] = model.replace("| (pch ? held", "& ~(pch ? held")  # no write-back: din high stores 0s
    (tmp_path / "no_write_back").mkdir()
    assert "FAIL" in _simulate(tmp_path / "no_write_back", files)


@requires_timing
def test_in_macro_refresh_restores_decayed_ones_and_zeros(tech, tmp_path):
    """The macro's own write-back on the extracted macro (din held high, pch released after we): a
    '1' decayed to 0.7 vdd comes back to the rail and a '0' decayed to 0.75 V back down, near and far."""
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.verify.timing import build_model, simulate_refresh, standard_corners

    cfg = _rim(rows=4, cols=20)
    gds = tmp_path / "rim.gds"
    gc_macro(tech, cfg, name="rim_test", decoded=True).write_gds(gds)
    model = build_model(tech, cfg, gds, "rim_test", tmp_path / "model")
    assert model.refresh_in_macro
    near, far = model.kept[0], model.kept[-1]
    for corner in (c for c in standard_corners() if c.process in ("tt", "ss")):
        for cells in ({near: (1, 0.7 * corner.vdd), far: (0, 0.75)}, {near: (0, 0.75), far: (1, 0.7 * corner.vdd)}):
            result = simulate_refresh(tech, model, corner, 0, cells, tmp_path / f"{corner.process}_{cells[near][0]}")
            for col, r in result.items():
                assert r.restored, (corner.name, col, r)


_drc = shutil.which("klayout") is not None and shutil.which("magic") is not None and shutil.which("netgen") is not None


@pytest.mark.skipif(not (_drc and _model_file.exists()), reason="klayout, magic, netgen or the sky130 PDK is missing")
@pytest.mark.parametrize("banked", [False, True], ids=["plain", "banked"])
def test_refresh_in_macro_is_drc_and_lvs_clean(tech, tmp_path, banked):
    from gc_compiler.compiler import GainCellCompiler
    from gc_compiler.verify.drc import run_drc
    from gc_compiler.verify.lvs import lvs_layout_vs_netlist

    cfg = ArrayConfig.for_tech(tech, rows=8, cols=8, refresh_in_macro=True)
    result = GainCellCompiler(cfg, tech).build(tmp_path, name=f"rim_{int(banked)}", macro=True, decoded=True, banked=banked)
    drc = run_drc(result.gds, tech=tech)
    assert drc.clean, str(drc)
    ports = {p.name: tuple(p.dcenter) for p in result.component.ports}
    lvs = lvs_layout_vs_netlist(result.gds, result.component.name, result.spice, result.component.name,
                                workdir=tmp_path / "lvs", ports=ports, tech=tech)
    assert lvs.matched, lvs.report[-3000:]
