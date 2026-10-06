from __future__ import annotations

import pytest

from gc_compiler.array.tiler import array_geometry, gc_array
from gc_compiler.cells.bitcell_3t import bitcell_geometry, gain_cell_3t
from gc_compiler.config import ArrayConfig, BitcellConfig
from gc_compiler.tech.sky130 import Sky130


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


@pytest.fixture(scope="session")
def bitcell(tech):
    return gain_cell_3t(tech, BitcellConfig.plain(), name="test_gc3t")


def test_ports_are_lowercase_nets(bitcell):
    names = {port.name for port in bitcell.ports}
    assert names == {"wwl", "rwl", "wbl", "rbl", "vss"}
    assert all(name == name.lower() for name in names)


def test_cell_is_on_grid(tech, bitcell):
    geo = bitcell_geometry(tech, BitcellConfig.plain())
    grid = tech.rules.grid
    for value in (geo.pitch_x, geo.pitch_y, geo.rbl_x, geo.wbl_x, geo.wwl_y, geo.rwl_y):
        assert abs(value / grid - round(value / grid)) < 1e-9


def test_bit_line_contacts_sit_on_the_cell_boundary(tech):
    geo = bitcell_geometry(tech, BitcellConfig.plain())
    rbl = next(p for p in gain_cell_3t(tech, name="boundary_gc3t").ports if p.name == "rbl")
    assert rbl.dcenter[1] == pytest.approx(0.0)
    assert geo.pitch_y > 0


def test_bit_lines_fit_the_pitch(tech):
    geo = bitcell_geometry(tech, BitcellConfig.plain())
    assert geo.rbl_x < geo.wbl_x < geo.pitch_x


def test_too_small_pitch_is_rejected(tech):
    with pytest.raises(ValueError):
        bitcell_geometry(tech, BitcellConfig.plain(pitch_x=0.5))


def test_array_geometry_stacks_rows_and_taps(tech):
    cfg = ArrayConfig.basic(rows=8, cols=4, tap_row_interval=2, strap_column_interval=0)
    geo = array_geometry(tech, cfg)
    assert geo.bits == 32
    assert len(geo.tap_row_y) == 3  # bottom edge, middle, top edge
    assert geo.width == pytest.approx(4 * geo.cell.pitch_x)
    # mirrored rows share their wbl boundary, so their word lines are symmetric
    assert geo.wwl_y(1) - geo.wwl_y(0) == pytest.approx(
        2 * (geo.cell.pitch_y - geo.cell.wwl_y)
    )


def test_array_geometry_includes_strap_columns(tech):
    from gc_compiler.array.strap_column import strap_column_geometry

    cfg = ArrayConfig.basic(rows=8, cols=4, tap_row_interval=2, strap_column_interval=2)
    geo = array_geometry(tech, cfg)
    strap_width = strap_column_geometry(tech).width
    # the first strap half an interval in (after column 1), then every 2nd (after column 3 is the
    # end), plus the mandatory right-edge strap: no column is more than one from a strap
    assert len(geo.strap_col_x) == 2 + 1
    assert geo.strap_col_x[0] == pytest.approx(geo.cell.pitch_x)
    assert geo.width == pytest.approx(4 * geo.cell.pitch_x + 3 * strap_width)


def test_odd_row_count_is_rejected():
    with pytest.raises(ValueError):
        ArrayConfig.basic(rows=7)


def test_array_builds_with_all_ports(tech):
    cfg = ArrayConfig.basic(rows=4, cols=3)
    component = gc_array(tech, cfg, name="test_array_4x3")
    names = {port.name for port in component.ports}
    assert {"wwl_0", "wwl_3", "rwl_2", "wbl_2", "rbl_0", "vss_1"} <= names
    assert component.info["bits"] == 12
