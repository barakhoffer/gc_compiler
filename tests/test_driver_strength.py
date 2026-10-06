"""Stronger word-line drivers: scaled inverters (and negative-rail output stages) in wider driver columns."""

from __future__ import annotations

import shutil

import pytest

from gc_compiler.array.floorplan import gc_banked_macro, gc_macro, macro_geometry
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.spice import macro_netlist
from gc_compiler.periphery.logic import LevelShifterConfig, LogicConfig, driver_logic
from gc_compiler.periphery.wl_driver_neg import neg_driver_geometry
from gc_compiler.cells.bitcell_3t import bitcell_geometry
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


def test_strength_must_be_positive_and_scales_the_widths():
    with pytest.raises(ValueError, match="positive"):
        ArrayConfig.basic(wordline_driver_strength=0)
    logic = LogicConfig()
    assert driver_logic(logic, 1) is logic
    drv = driver_logic(logic, 4)
    assert (drv.pmos_width, drv.nmos_width, drv.length) == (2.56, 1.68, logic.length)


def test_netlist_uses_a_scaled_driver_inverter(tech):
    """Drivers get their own inverter subcircuit; the column circuits keep the logic one."""
    plain = macro_netlist(tech, ArrayConfig.basic(rows=4, cols=2), decoded=False)
    assert "gc_inv_drv" not in plain
    text = macro_netlist(tech, ArrayConfig.basic(rows=4, cols=2, wordline_driver_strength=4), decoded=False)
    assert ".subckt gc_inv_drv" in text and "w=2.56" in text
    assert "Xrwl_drv_0 rwl_in_0 rwl_0 vdd vss gc_inv_drv" in text
    assert "Xrd_0 rbl_0 dout_0 vdd vss gc_inv" in text
    neg = macro_netlist(tech, ArrayConfig.basic(rows=4, cols=2, negative_write_wl=True, wordline_driver_strength=4), decoded=False)
    assert "Xinb in inb vdd vss gc_inv_drv" in neg  # the level shifter's inverter is the driver's


def test_stronger_drivers_widen_only_the_driver_columns(tech):
    base = macro_geometry(tech, ArrayConfig.basic(rows=4, cols=8))
    strong = macro_geometry(tech, ArrayConfig.basic(rows=4, cols=8, wordline_driver_strength=4))
    assert strong.driver_width > base.driver_width and strong.height == pytest.approx(base.height)
    assert strong.width - base.width == pytest.approx(2 * (strong.driver_width - base.driver_width))


def test_negative_driver_keeps_its_output_nmos_clear_of_the_shared_pad(tech):
    """A wide output NMOS's contact li reaches its diffusion's edge; the device moves right so
    that li clears the gate pad's (li spacing), and a narrow one stays under NZ."""
    cell = bitcell_geometry(tech)
    for strength, moved in ((1, False), (2, False), (4, True)):
        g = neg_driver_geometry(tech, cell, driver_logic(LogicConfig(), strength), LevelShifterConfig())
        assert (g.nd_x0 > g.nz_x0 + 1e-9) == moved, strength


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
    assert all((l or r or "").startswith("vnw") for l, r in result.pin_mismatches), str(result)


@requires_drc
@requires_lvs
@pytest.mark.parametrize(
    "rows, cols, extra, banked",
    [
        (4, 12, {}, False),
        (4, 14, {"negative_write_wl": True, "replica_column": True, "wordline_straps": True}, False),
        (4, 10, {"wordline_straps": True}, True),
    ],
    ids=["plain", "neg_rep_straps", "banked_straps"],
)
def test_strong_driver_macro_drc_and_lvs(tech, tmp_path, rows, cols, extra, banked):
    cfg = ArrayConfig.basic(rows=rows, cols=cols, wordline_driver_strength=4, **extra)
    _check(tech, cfg, tmp_path, f"drv4_{rows}x{cols}{'_b' if banked else ''}", banked)
