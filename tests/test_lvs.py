from __future__ import annotations

import shutil

import pytest

from gc_compiler.cells.bitcell_3t import gain_cell_3t
from gc_compiler.compiler import GainCellCompiler
from gc_compiler.config import ArrayConfig, BitcellConfig
from gc_compiler.netlist.spice import array_netlist, bitcell_subckt
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.lvs import default_magicrc, default_netgen_setup, lvs_layout_vs_netlist

lvs_available = (
    shutil.which("magic") is not None
    and shutil.which("netgen") is not None
    and default_magicrc().exists()
    and default_netgen_setup().exists()
)
requires_lvs = pytest.mark.skipif(not lvs_available, reason="magic, netgen or the sky130 setup files are missing")


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


@requires_lvs
def test_bitcell_devices_match_layout(tech, tmp_path):
    """A standalone bitcell has no adjacent tap, so its supply nodes are legitimately open.

    LVS should still confirm the three devices, their types and their wwl/wbl/rwl/rbl/sn
    connectivity are exactly what the netlist describes.
    """
    cfg = BitcellConfig.plain()
    name = "lvs_gc3t"
    gds = tmp_path / f"{name}.gds"
    gain_cell_3t(tech, cfg, name=name).write_gds(gds)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(bitcell_subckt(tech, cfg, name="gc3t"))

    result = lvs_layout_vs_netlist(gds, name, spice, "gc3t", workdir=tmp_path / "lvs")

    # wwl/wbl/rwl/rbl are matched by name (labels are added in _add_ports); the only
    # disagreement is the supply nets, which is expected: a standalone, un-tapped cell has
    # no path from its "vss" metal to the substrate, and the write PMOS's n-well is a
    # floating island with no strap at all.
    assert result.net_count == (9, 8)
    assert result.mismatched_nets == [("vss", "vss"), ("VSUBS", None)]


@requires_lvs
def test_lvs_detects_a_real_mismatch(tech, tmp_path):
    """Sanity check: LVS must fail when the reference netlist is wrong."""
    cfg = BitcellConfig.plain()
    name = "lvs_gc3t_bad"
    gds = tmp_path / f"{name}.gds"
    gain_cell_3t(tech, cfg, name=name).write_gds(gds)

    wrong = tmp_path / f"{name}.spice"
    wrong.write_text(
        ".subckt gc3t wwl wbl rwl rbl vss vnw\n"
        "Xmw wbl wwl sn vnw sky130_fd_pr__pfet_01v8 w=0.42 l=0.18\n"
        "Xmr n_int sn vss vss sky130_fd_pr__nfet_01v8 w=0.42 l=0.15\n"
        "Xms rbl wwl n_int vss sky130_fd_pr__nfet_01v8 w=0.42 l=0.15\n"  # bug: rwl -> wwl
        ".ends\n"
    )
    result = lvs_layout_vs_netlist(gds, name, wrong, "gc3t", workdir=tmp_path / "lvs_bad")
    assert not result.matched


@requires_lvs
def test_strapped_array_lvs_matches(tech, tmp_path):
    """The vss/n-well strap columns (array/strap_column.py) fix the floating-well gap that
    test_bitcell_devices_match_layout documents for a standalone cell: once rows are tiled
    into a full array with strap columns, every net -- including vss and the n-well -- must
    line up exactly with the reference netlist.
    """
    from gc_compiler.array import gc_array

    cfg = ArrayConfig.basic(rows=8, cols=8, tap_row_interval=2, strap_column_interval=4)
    name = "lvs_strap_array"
    gds = tmp_path / f"{name}.gds"
    gc_array(tech, cfg, name=name).write_gds(gds)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(array_netlist(tech, cfg, name=name))

    result = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs")

    assert result.matched
    assert result.mismatched_nets == []
    assert result.net_count == (162, 162)


@requires_lvs
@pytest.mark.parametrize(
    "rows,cols,decoded",
    [
        (12, 4, False),
        (12, 6, True),
        # more than 16 columns: the array inserts a strap column, shifting every later column
        (16, 20, False),
        (16, 22, True),
    ],
)
def test_macro_lvs_matches(tech, tmp_path, rows, cols, decoded):
    """The whole macro -- array, word-line drivers, column circuits and (optionally) both
    decoders -- extracted and compared to ``macro_netlist``. This is the check that found the
    driver and column-circuit inputs shorted to vss, the column circuits placed at the wrong
    pitch past a strap column, and the missing vdd distribution: none of them is visible to DRC.
    """
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.netlist.spice import macro_netlist
    from gc_compiler.periphery.logic import LogicConfig

    cfg = ArrayConfig.basic(rows=rows, cols=cols)
    name = f"lvs_macro_{rows}x{cols}{'_d' if decoded else ''}"
    gds = tmp_path / f"{name}.gds"
    component = gc_macro(tech, cfg, LogicConfig(), name=name, decoded=decoded)
    component.write_gds(gds)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(macro_netlist(tech, cfg, LogicConfig(), name=name, decoded=decoded))

    ports = {p.name: tuple(p.dcenter) for p in component.ports}
    result = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs", ports=ports)

    assert result.matched, str(result)
    assert result.mismatched_nets == []
    assert result.net_count[0] == result.net_count[1]
    # every declared pin is verified by name except vnw, which has no macro-level pin at all
    # (see magic_extract's docstring) -- anything more here would be a real regression.
    assert result.pin_mismatches == [(None, "vnw")], str(result)


@requires_lvs
def test_macro_lvs_catches_a_pin_wired_to_the_wrong_name(tech, tmp_path):
    """Without ``ports``, netgen has no top-level pin list to check at all -- it reports a
    topology match with the whole pin list unverified (``LvsResult.port_errors``), which
    would have silently passed a macro whose write-word-line pin was wired to the wrong row
    (still the same *set* of nets overall, just mislabelled). Passing ``ports`` catches
    exactly that: a real bug that pure net-topology matching cannot see, because the
    topology genuinely is still isomorphic -- only the pin *names* disagree."""
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.netlist.spice import macro_netlist
    from gc_compiler.periphery.logic import LogicConfig

    # a size not used by any other plain ArrayConfig macro in the suite: component names are
    # keyed by config, not by the `name=` passed to build(), so a size collision reuses the
    # wrong cell (see test_retention.py's macro_refresh_interval test for the same issue).
    cfg = ArrayConfig.basic(rows=8, cols=11)
    name = "lvs_macro_8x11_badpin"
    gds = tmp_path / f"{name}.gds"
    component = gc_macro(tech, cfg, LogicConfig(), name=name, decoded=False)
    component.write_gds(gds)
    spice_text = macro_netlist(tech, cfg, LogicConfig(), name=name, decoded=False)
    bad = spice_text.replace("wwl_in_0", "wwl_in_WRONG")
    assert bad != spice_text
    spice = tmp_path / f"{name}.spice"
    spice.write_text(bad)

    ports = {p.name: tuple(p.dcenter) for p in component.ports}
    without_ports = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs_noports")
    with_ports = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs_ports", ports=ports)

    assert without_ports.matched  # the bug is invisible to topology-only matching
    assert with_ports.matched  # still a topology match -- ports just adds pin-name checking
    assert ("wwl_in_0", "wwl_in_WRONG") in with_ports.pin_mismatches, str(with_ports)


@requires_lvs
@pytest.mark.parametrize("rows,cols,decoded", [(10, 4, False), (12, 6, True), (16, 20, True)])
def test_replica_macro_lvs_matches(tech, tmp_path, rows, cols, decoded):
    """The replica column -- cells with the storage node tied to the write bit line, a write
    driver held at vdd and a read buffer that drives ``rdy`` -- must extract to what
    ``macro_netlist`` says, including when it lands past a strap column."""
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.netlist.spice import macro_netlist
    from gc_compiler.periphery.logic import LogicConfig

    cfg = ArrayConfig.basic(rows=rows, cols=cols, replica_column=True)
    name = f"lvs_rep_{rows}x{cols}{'_d' if decoded else '_p'}"
    gds = tmp_path / f"{name}.gds"
    component = gc_macro(tech, cfg, LogicConfig(), name=name, decoded=decoded)
    component.write_gds(gds)
    spice = tmp_path / f"{name}.spice"
    spice.write_text(macro_netlist(tech, cfg, LogicConfig(), name=name, decoded=decoded))

    ports = {p.name: tuple(p.dcenter) for p in component.ports}
    result = lvs_layout_vs_netlist(gds, name, spice, name, workdir=tmp_path / "lvs", ports=ports)

    assert result.matched, str(result)
    assert result.mismatched_nets == []
    assert result.pin_mismatches == [(None, "vnw")], str(result)
