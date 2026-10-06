"""The memory as one hard block: the gain-cell macro's power pins and abstract views, and the
macro with its refresh controller placed and routed by LibreLane."""

from __future__ import annotations

import re
import shutil

import pytest

from gc_compiler.array.floorplan import gc_banked_macro, gc_macro
from gc_compiler.config import ArrayConfig
from gc_compiler.flow.abstract import hard_macro, pin_use, write_blackbox, write_gds, write_lef, write_liberty
from gc_compiler.netlist.spice import macro_netlist, macro_pins
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.rtl import ControllerTiming, controller_params, controller_sdc, top_sdc
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.drc import default_deck, run_drc
from gc_compiler.verify.lvs import default_magicrc, default_netgen_setup, lvs_layout_vs_netlist

requires_drc = pytest.mark.skipif(
    shutil.which("klayout") is None or not default_deck().exists(), reason="klayout or the sky130 deck is missing"
)
requires_lvs = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("magic", "netgen"))
    or not default_magicrc().exists()
    or not default_netgen_setup().exists(),
    reason="magic, netgen or the sky130 setup files are missing",
)
requires_librelane = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("librelane", "iverilog")), reason="librelane or iverilog is missing"
)


@pytest.fixture(scope="module")
def tech() -> Sky130:
    return Sky130()


def _hard(tech, rows, cols, banked, name):
    cfg = ArrayConfig(rows=rows, cols=cols)
    logic = LogicConfig()
    core = gc_banked_macro(tech, cfg, logic, name=f"{name}_core") if banked else gc_macro(
        tech, cfg, logic, name=f"{name}_core", decoded=True
    )
    return cfg, hard_macro(tech, core, cfg, name, banked)


@pytest.mark.parametrize("banked", [False, True], ids=["one_bank", "banked"])
def test_views_describe_every_pin(tech, tmp_path, banked):
    """The LEF, both blackboxes and the timing library list exactly the netlist's pins: the
    supplies as metal4 stripes (one per n-well of each bank), every signal on its own metal."""
    cfg, hard = _hard(tech, 6, 7 if banked else 9, banked, f"views_{int(banked)}")
    pins = macro_pins(cfg, decoded=True, banked=banked)
    assert hard.pins == pins
    assert hard.component.dbbox().left == 0 and hard.component.dbbox().bottom == 0
    assert {s.net for s in hard.stripes} == {p for p in pins if pin_use(p) != "SIGNAL"}
    lef = write_lef(tech, hard, tmp_path / "m.lef").read_text()
    assert re.findall(r"^  PIN (\S+)", lef, re.M) == pins
    for pin in pins:
        body = lef.split(f"  PIN {pin}\n")[1].split(f"  END {pin}")[0]
        assert ("LAYER met4" in body) == (pin_use(pin) != "SIGNAL"), pin
    assert "  OBS" in lef and "LAYER met2 ;" in lef.split("  OBS")[1]
    unpowered = write_blackbox(hard, tmp_path / "bb.v").read_text()
    powered = write_blackbox(hard, tmp_path / "pbb.v", powered=True).read_text()
    for pin in pins:
        assert (re.search(rf"\b{pin}\b", unpowered) is not None) == (pin_use(pin) == "SIGNAL")
        assert re.search(rf"\b{pin}\b", powered)
    # every pin marker lies on its metal: magic reads a pin shape as metal, and one reaching past
    # the metal the LEF describes lets the router come too close (met2.2 in a routed block)
    import klayout.db as kdb

    from gc_compiler.tech.rules import LayerRole

    comp = hard.component
    for role in tech.stack.lef_names:
        pin_layer = tech.pin_layer(role)
        if pin_layer is None or comp.kcl.layout.find_layer(*pin_layer) is None:
            continue
        markers = kdb.Region(comp.begin_shapes_rec(comp.kcl.layout.find_layer(*pin_layer))).merged()
        metal = kdb.Region(comp.begin_shapes_rec(comp.kcl.layout.find_layer(*tech.layer(role)))).merged()
        assert (markers - metal).is_empty(), role
    lib = write_liberty(hard, tmp_path / "m.lib", "tt", 3.5).read_text()
    assert lib.count('related_pin : "re"') == sum(p.startswith(("dout", "rdy")) for p in pins)


@requires_drc
@requires_lvs
@pytest.mark.parametrize("banked", [False, True], ids=["one_bank", "banked"])
def test_power_pins_keep_the_macro_clean(tech, tmp_path, banked):
    """The metal4 stripes and their via stacks down to the rails: DRC clean, and each stripe on
    its own net (a stripe on the wrong rail would short two supplies and fail LVS)."""
    rows, cols = (6, 7) if banked else (6, 9)
    name = f"pw_{int(banked)}"
    cfg, hard = _hard(tech, rows, cols, banked, name)
    gds = write_gds(hard.component, tmp_path / f"{name}.gds")
    report = run_drc(gds)
    assert report.clean, str(report)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(macro_netlist(tech, cfg, LogicConfig(), name=name, decoded=True, banked=banked))
    ports = {p.name: tuple(p.dcenter) for p in hard.component.ports if "_m4_" not in p.name}
    result = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs", ports=ports)
    assert result.matched and not result.mismatched_nets, str(result)


def test_constraints_follow_the_read_protocol():
    """Both SDCs sample a read READ_CYCLES edges after re rises, the loop-back one edge later,
    and the controller's read budgets its own sampling on top of the macro's read."""
    cfg = ArrayConfig(rows=6, cols=5, refresh_in_macro=False)  # the loop-back
    timing = ControllerTiming(clock_ns=5.0, read_ns=4.0, interval_us=100)
    params = controller_params(cfg, timing, banked=True)
    assert params.read_cycles == 2  # 4 ns + 1.5 ns of its own does not fit one 5 ns cycle
    block = controller_sdc(params, timing)
    assert "set_multicycle_path -setup 2 -from [get_ports m_dout*]" in block
    assert "set_multicycle_path -setup 3 -from [get_ports m_dout*] -to [get_ports m_din*]" in block
    top = top_sdc(params, timing)
    assert "set_multicycle_path -setup 2 -through [get_pins macro/dout*]" in top
    # the loop-back ends on pins without a check, so it is measured after routing, not constrained
    assert "set_max_delay" not in top and "loopback_arrival" in top
    # with the handshake the macro's rdy only reaches the synchroniser, and the read is sampled
    # two edges after the one that catches it
    hs = ControllerTiming(clock_ns=5.0, read_ns=4.0, interval_us=100, handshake=True)
    params = controller_params(cfg, hs, banked=True)
    top = top_sdc(params, hs)
    assert "set_false_path -through [get_pins macro/rdy*]" in top
    assert "set_multicycle_path -setup 3 -through [get_pins macro/dout*]" in top
    assert "set_multicycle_path -setup 2 -from [get_ports m_dout*]" in controller_sdc(params, hs)
    # with the macro writing a refresh back itself (the default) there is no loop-back to constrain or measure
    rim = ArrayConfig(rows=6, cols=5)
    assert rim.refresh_in_macro and not ArrayConfig.basic(rows=6, cols=5).refresh_in_macro
    params = controller_params(rim, timing, banked=True)
    assert "-to [get_ports m_din*]" not in controller_sdc(params, timing)
    assert "loopback_arrival" not in top_sdc(params, timing)


@requires_librelane
@pytest.mark.parametrize(
    "banked,handshake,rim", [(False, False, False), (True, False, False), (True, True, False), (False, False, True)],
    ids=["one_bank", "banked", "handshake", "refresh_in_macro"],
)
def test_memory_block_is_clean_timed_and_passes_its_testbench(tech, tmp_path, banked, handshake, rim):
    """The macro with its controller, placed and routed by LibreLane: DRC (KLayout and magic, the
    macro included) and LVS clean, timing met at tt/ss/ff, and the generated testbench
    passing on the routed netlist -- build_memory raises otherwise."""
    from gc_compiler.compiler import GainCellCompiler

    cfg = ArrayConfig(rows=6, cols=5 if banked else 7, refresh_in_macro=rim)
    memory = GainCellCompiler(cfg, tech).memory(
        tmp_path, name=f"blk{int(banked)}{'h' if handshake else ''}{'r' if rim else ''}", banked=banked, handshake=handshake
    )
    m = memory.metrics
    assert m["handshake"] == handshake
    assert m["drc_errors"] == 0 and m["lvs_errors"] == 0 and m["clean"]
    assert "PASS" in memory.gate_level
    if rim:
        assert m["loopback"] == {}  # the macro writes a refresh back itself
    else:
        assert set(m["loopback"]) == {"tt", "ss", "ff"} and all(v["slack_ns"] > 0 for v in m["loopback"].values())
    # no copy of the user's request: control flops and a dout latch a column; the flow settings took
    # (no 16x clock buffer at the tree roots), and the cells near the library's area per flop
    assert m["flops"] <= cfg.cols + 40  # (control, with the handshake's synchronisers, about 40)
    assert re.search(r"\b\w*buf_16\b", memory.file(".v").read_text()) is None
    assert m["controller_cell_area_um2"] / m["flops"] <= 1.1 * tech.standard_cells().area_per_flop_um2
    assert m["bits"] == cfg.bits * (2 if banked else 1)
    for suffix in (".gds", ".lef", ".v", "_tt.lib", "_ss.lib", "_ff.lib", ".json", ".sdc"):
        assert memory.file(suffix).exists(), suffix
