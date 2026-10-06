from __future__ import annotations

import shutil

import gdsfactory as gf
import pytest

from gc_compiler.array.floorplan import gc_macro, macro_geometry
from gc_compiler.cells.bitcell_3t import bitcell_geometry
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.spice import inverter_subckt, macro_netlist, macro_pins
from gc_compiler.periphery.column import column_circuit, column_geometry
from gc_compiler.periphery.logic import LogicConfig, inverter, logic_geometry, pmos_switch
from gc_compiler.periphery.wl_driver import (
    LEFT_WORDLINES,
    RIGHT_WORDLINES,
    wl_driver_pair,
    wordline_y,
)
from gc_compiler.tech.rules import LayerRole
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.drc import default_deck, run_drc


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


def test_logic_slice_ports(tech):
    slice_cell = inverter(tech, LogicConfig(), name="test_inv")
    assert {p.name for p in slice_cell.ports} == {"a", "y", "vdd", "vss"}


def test_pmos_switch_has_no_pulldown(tech):
    cfg = LogicConfig()
    diff = gf.get_layer(tech.layer(LayerRole.DIFF))
    switch = pmos_switch(tech, cfg, name="test_pup")
    inv = inverter(tech, cfg, name="test_inv2")
    assert switch.dbbox(diff).width() < inv.dbbox(diff).width()


def test_slice_pitch_can_be_forced(tech):
    forced = logic_geometry(tech, LogicConfig(pitch=1.22))
    assert forced.pitch_y == pytest.approx(1.22)
    with pytest.raises(ValueError):
        logic_geometry(tech, LogicConfig(pitch=0.4))


def test_wordlines_are_split_between_the_two_strips():
    assert set(LEFT_WORDLINES) | set(RIGHT_WORDLINES) == {"rwl_0", "wwl_0", "wwl_1", "rwl_1"}
    assert not set(LEFT_WORDLINES) & set(RIGHT_WORDLINES)


def test_wordline_landings_are_far_enough_apart(tech):
    """Each strip must keep its landings clear of the stripe that runs past them."""
    cell = bitcell_geometry(tech)
    ys = wordline_y(cell)
    pair = 2 * cell.pitch_y
    for wordlines in (LEFT_WORDLINES, RIGHT_WORDLINES):
        landed = sorted(ys[net] for net in wordlines)
        landed.append(landed[0] + pair)  # the next row pair repeats the pattern
        gaps = [b - a for a, b in zip(landed, landed[1:])]
        assert min(gaps) > 0.5


def test_driver_drives_every_wordline_of_a_pair(tech):
    cell = bitcell_geometry(tech)
    left = wl_driver_pair(tech, cell, wordlines=LEFT_WORDLINES, name="test_wl_left")
    names = {p.name for p in left.ports}
    assert {"rwl_0", "wwl_1", "rwl_0_in", "wwl_1_in", "vdd", "vss"} <= names


def test_column_circuit_pitch_matches_the_bitcell(tech):
    cell = bitcell_geometry(tech)
    column = column_circuit(tech, cell, name="test_column")
    geo = column_geometry(tech, cell, LogicConfig(pitch=cell.pitch_x))
    assert geo.width == pytest.approx(cell.pitch_x)
    assert {"din", "dout", "rbl", "wbl", "pch"} <= {p.name for p in column.ports}


def test_macro_geometry_wraps_the_array(tech):
    cfg = ArrayConfig.basic(rows=8, cols=8)
    geo = macro_geometry(tech, cfg)
    assert geo.width == pytest.approx(geo.array.width + 2 * geo.driver_width)
    assert geo.height == pytest.approx(geo.array.height + geo.column_height)
    assert 0 < geo.efficiency < 1


def test_macro_netlist_has_periphery(tech):
    cfg = ArrayConfig.basic(rows=4, cols=4)
    text = macro_netlist(tech, cfg)
    assert "Xwwl_drv_3 wwl_in_3 wwl_3 vdd vss gc_inv" in text
    assert "Xrd_0 rbl_0 dout_0 vdd vss gc_inv" in text
    assert "Xwr_2 din_2 wbl_2 vdd vss gc_inv" in text
    assert "pch" in macro_pins(cfg)
    assert ".subckt gc_inv a y vdd vss" in inverter_subckt(tech, LogicConfig())


drc_available = shutil.which("klayout") is not None and default_deck().exists()
requires_drc = pytest.mark.skipif(not drc_available, reason="klayout or the sky130 deck is missing")


@requires_drc
def test_macro_is_drc_clean(tech, tmp_path):
    macro = gc_macro(tech, ArrayConfig.basic(rows=8, cols=4), name="drc_macro")
    gds = tmp_path / "macro.gds"
    macro.write_gds(gds)
    report = run_drc(gds)
    assert report.clean, str(report)
