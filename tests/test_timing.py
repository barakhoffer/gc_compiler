from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from gc_compiler.compiler import GainCellCompiler
from gc_compiler.config import ArrayConfig
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.flatnet import FlatNetlist, parse_si
from gc_compiler.verify.lvs import default_magicrc
from gc_compiler.verify.timing import (
    BenchSettings,
    _read_correct,
    crossing,
    standard_corners,
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


# --- measurements ----------------------------------------------------------------------


def test_crossing_interpolates_between_samples():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert crossing(t, y, 0.5, rising=True) == pytest.approx(1.5)
    assert crossing(t, 1 - y, 0.5, rising=False) == pytest.approx(1.5)


def test_crossing_ignores_what_happens_before_after_and_missing_edges():
    t = np.linspace(0, 10, 11)
    y = np.where(t < 2, 0.0, 1.0) * (t < 6)  # a pulse from t=2 to t=6
    assert crossing(t, y, 0.5, rising=True, after=3.0) is None
    assert crossing(t, y, 0.5, rising=False, after=3.0) == pytest.approx(5.5)
    assert crossing(t, np.zeros_like(t), 0.5, rising=True) is None


def test_read_correctness_is_judged_at_the_strobe():
    settings = BenchSettings(strobe_factor=1.25)
    # rdy fires 1.0 ns after re; the strobe is at 1.25 ns
    args = dict(t_re=0.0, t_rdy=1.0e-9, dout_max=1.8, half=0.9, settings=settings)
    assert _read_correct(1, t_dout=1.2e-9, **args)
    assert not _read_correct(1, t_dout=1.4e-9, **args)  # a '1' that arrives after the strobe
    assert _read_correct(0, t_dout=None, **args)
    assert _read_correct(0, t_dout=2.0e-9, **args)  # a '0' that flips only after the strobe
    assert not _read_correct(0, t_dout=1.05e-9, **args)


def test_read_correctness_without_a_replica_uses_the_whole_window():
    settings = BenchSettings()
    args = dict(t_re=0.0, t_rdy=None, half=0.9, settings=settings)
    assert _read_correct(1, t_dout=1e-9, dout_max=1.8, **args)
    assert not _read_correct(1, t_dout=None, dout_max=0.1, **args)
    assert _read_correct(0, t_dout=None, dout_max=0.2, **args)
    assert not _read_correct(0, t_dout=3e-9, dout_max=1.7, **args)


def test_standard_corners_cover_the_process_sections():
    corners = standard_corners()
    assert {c.process for c in corners} == set(Sky130().model_corners)
    assert len({c.name for c in corners}) == len(corners)


def test_sky130_model_header_loads_only_the_corner_file():
    lines = Sky130().model_header("ss")
    assert any(l.startswith(".include") and "corners/ss.spice" in l for l in lines)
    assert not any("sky130.lib.spice" in l for l in lines)  # the slow, all-inclusive library


# --- the flat netlist ----------------------------------------------------------------------


def _cell(prefix: str, row: int, col: str) -> str:
    """One extracted-style bitcell: ms, mr, mw plus a cap from its storage node to the word line."""
    sn, mid = f"{prefix}c{row}_{col}/sn#", f"{prefix}c{row}_{col}/mid#"
    return "\n".join(
        [
            f"Xms{row}{col} {prefix}rbl_{col} {prefix}rwl_{row} {mid} vss sky130_fd_pr__nfet_01v8 w=0.42 l=0.15",
            f"Xmr{row}{col} {mid} {sn} vss vss sky130_fd_pr__nfet_01v8 w=0.42 l=0.15",
            f"Xmw{row}{col} {prefix}wbl_{col} {prefix}wwl_{row} {sn} vnw sky130_fd_pr__pfet_01v8 w=0.42 l=0.18",
            f"C{row}{col}a {prefix}wwl_{row} {sn} 0.05f",
            f"C{row}{col}b {prefix}rwl_{row} {prefix}rwl_{row} 0.01f",
            f"C{row}{col}c {prefix}rwl_{row} {sn} 0.02f",
            f"C{row}{col}d {prefix}rbl_{col} vss 0.3f",
        ]
    )


def _tiny_array(rows: int = 2, cols: int = 4) -> str:
    prefix = "arr/"
    lines = [_cell(prefix, r, str(c)) for r in range(rows) for c in range(cols)]
    for c in range(cols):
        lines.append(f"Xrd{c} {prefix}rbl_{c} dout_{c} vdd vss sky130_fd_pr__pfet_01v8 w=1 l=0.15")
        lines.append(f"Xwr{c} din_{c} {prefix}wbl_{c} vdd vss sky130_fd_pr__nfet_01v8 w=1 l=0.15")
    return "\n".join(lines) + "\n"


def test_parse_si():
    assert parse_si("0.02546f") == pytest.approx(2.546e-17)
    assert parse_si("3.5") == 3.5
    assert parse_si("12p") == pytest.approx(12e-12)


def test_parse_handles_continuations_and_turns_varactors_into_capacitors():
    text = (
        "Xv1 a b c sky130_fd_pr__cap_var_lvt w=0.42 l=0.06\n"
        "Xm1 d g s b sky130_fd_pr__nfet_01v8\n+ w=0.42 l=0.15\n"
        "C1 a b 2f $ extra\n"
        "R1 a b 10\n"
        "* comment\n"
    )
    net = FlatNetlist.parse(text)
    # the varactor is a fixed gate-to-well capacitor (its model is out of range for a sliver that
    # small in some corners), not a device
    assert [d.name for d in net.devices] == ["Xm1"] and net.devices[0].params == "w=0.42 l=0.15"
    varactor = next(c for c in net.caps if c.name == "CXv1")
    assert (varactor.a, varactor.b) == ("a", "b") and varactor.value == pytest.approx(0.42 * 0.06 * 8.5e-15)
    assert next(c for c in net.caps if c.name == "C1").value == pytest.approx(2e-15)
    assert net.extra == ["R1 a b 10"]


def test_gate_loads_use_the_cells_own_device_sizes():
    net = FlatNetlist.parse(_tiny_array(1, 4))
    loads = net.reduce({"0", "3"}, group=2)
    net.add_wordline_rc(
        columns=["0", "1", "3"], x_of={"0": 5.0, "1": 25.0, "3": 55.0},
        left_driven={("rwl", 0): True, ("wwl", 0): True},
        resistance_per_um={"rwl": 2.0, "wwl": 3.0}, loads=loads,
        gate_load={"n": ("sky130_fd_pr__nfet_01v8", "w=0.42 l=0.5"), "p": ("sky130_fd_pr__pfet_01v8_lvt", "w=0.42 l=0.35")},
        edges=(0.0, 60.0),
    )
    added = {d.model: d.params for d in net.devices if d.name.startswith("Xld")}
    assert added == {"sky130_fd_pr__nfet_01v8": "w=0.42 l=0.5 m=2", "sky130_fd_pr__pfet_01v8_lvt": "w=0.42 l=0.35 m=2"}


def test_gate_loads_are_written_in_the_technologys_units():
    from gc_compiler.tech.sg13g2 import SG13G2
    from gc_compiler.tech.sky130 import Sky130
    from gc_compiler.tech.rules import DeviceKind
    from gc_compiler.verify.timing import _device_params

    assert _device_params(Sky130(), DeviceKind.NMOS, 0.42, 0.5)[1] == "w=0.42 l=0.5"
    assert _device_params(SG13G2(), DeviceKind.NMOS, 0.3, 0.13) == ("sg13_lv_nmos", "w=0.3u l=0.13u")


def test_find_cells_recovers_every_bitcell_and_its_nodes():
    net = FlatNetlist.parse(_tiny_array(2, 4))
    cells = net.find_cells()
    assert set(cells) == {(r, str(c)) for r in range(2) for c in range(4)}
    cell = cells[(1, "2")]
    assert cell.sn == "arr/c1_2/sn#" and cell.n_int == "arr/c1_2/mid#"
    assert cell.mw.g == "arr/wwl_1" and cell.ms.g == "arr/rwl_1"
    assert net.wordline_prefix() == "arr/"


def test_net_matching_needs_exactly_one_net():
    net = FlatNetlist.parse(_tiny_array(2, 2))
    assert net.net_matching(r"arr/rbl_1$") == "arr/rbl_1"
    with pytest.raises(ValueError):
        net.net_matching(r"arr/rbl_\d$")
    with pytest.raises(ValueError):
        net.net_matching(r"nothing_like_this")


def test_reduce_drops_unkept_columns_and_returns_groups():
    net = FlatNetlist.parse(_tiny_array(2, 6))
    loads = net.reduce({"0", "5"}, group=2)
    assert loads == {"1": ["1", "2"], "3": ["3", "4"]}
    remaining = net.find_cells()
    assert {c for _r, c in remaining} == {"0", "5"}
    nets = {n for d in net.devices for n in d.nodes}
    assert not any(f"rbl_{c}" in n or f"dout_{c}" in n for n in nets for c in "1234")
    # the dropped columns' bit-line caps are gone, the kept columns' stay
    assert all("rbl_1" not in (c.a + c.b) for c in net.caps)
    assert any("rbl_0" in (c.a + c.b) for c in net.caps)


def test_reduce_turns_coupling_to_a_dropped_net_into_capacitance_to_ground():
    net = FlatNetlist.parse(_tiny_array(1, 3))
    net.caps.append(type(net.caps[0])("Cx", "arr/rwl_0", "arr/c0_1/sn#", 0.7e-15))
    net.reduce({"0", "2"}, group=1)
    cx = next(c for c in net.caps if c.name == "Cx")
    assert {cx.a, cx.b} == {"arr/rwl_0", "vss"}


def _ladder(net: FlatNetlist, columns: list[str], loads: dict[str, list[str]], driven_left: bool):
    x_of = {"0": 5.0, "1": 25.0, "3": 55.0}
    rows = sorted({r for r, _ in net.find_cells()})
    return net.add_wordline_rc(
        columns=columns,
        x_of=x_of,
        left_driven={(k, r): driven_left for k in ("rwl", "wwl") for r in rows},
        resistance_per_um={"rwl": 2.0, "wwl": 3.0},
        loads=loads,
        gate_load={"n": ("sky130_fd_pr__nfet_01v8", "w=0.42 l=0.15"), "p": ("sky130_fd_pr__pfet_01v8", "w=0.42 l=0.18")},
        edges=(0.0, 60.0),
    )


def test_wordline_ladder_resistance_and_load_devices():
    net = FlatNetlist.parse(_tiny_array(2, 4))
    loads = net.reduce({"0", "3"}, group=2)
    assert loads == {"1": ["1", "2"]}
    nodes = _ladder(net, ["0", "1", "3"], loads, driven_left=True)

    assert nodes[("rwl", 0, "3")] == "arr/rwl_0__c3"
    # R segments run driver end -> 0 (5 um) -> group 1 (20 um) -> 3 (30 um), at 2 ohm/um
    values = {}
    for line in net.extra:
        name, a, b, r = line.split()
        values[(a, b)] = float(r)
    assert values[("arr/rwl_0", "arr/rwl_0__c0")] == pytest.approx(10.0)
    assert values[("arr/rwl_0__c0", "arr/rwl_0__c1")] == pytest.approx(40.0)
    assert values[("arr/rwl_0__c1", "arr/rwl_0__c3")] == pytest.approx(60.0)
    # the write word line uses its own resistance per um
    assert values[("arr/wwl_0", "arr/wwl_0__c0")] == pytest.approx(15.0)
    # gates of kept cells sit on their ladder node; the group is a multiplied gate load there
    ms = next(d for d in net.devices if d.name == "Xms13")
    mw = next(d for d in net.devices if d.name == "Xmw13")
    assert ms.g == "arr/rwl_1__c3" and mw.g == "arr/wwl_1__c3"
    loads_added = [d for d in net.devices if d.name.startswith("Xld")]
    assert len(loads_added) == 2 * 2  # rwl and wwl, per row
    assert all("m=2" in d.params for d in loads_added)
    assert {d.nodes[1] for d in loads_added} >= {"arr/rwl_0__c1", "arr/wwl_1__c1"}


def test_wordline_ladder_runs_from_the_right_when_the_driver_is_there():
    net = FlatNetlist.parse(_tiny_array(1, 4))
    loads = net.reduce({"0", "3"}, group=2)
    _ladder(net, ["0", "1", "3"], loads, driven_left=False)
    values = {}
    for line in net.extra:
        _, a, b, r = line.split()
        values[(a, b)] = float(r)
    assert values[("arr/rwl_0", "arr/rwl_0__c3")] == pytest.approx(10.0)  # 60 -> 55 um at 2 ohm/um
    assert ("arr/rwl_0__c1", "arr/rwl_0__c0") in values


def test_wordline_capacitance_is_conserved_when_moved_onto_the_ladder():
    net = FlatNetlist.parse(_tiny_array(1, 4))
    loads = net.reduce({"0", "3"}, group=2)
    before = sum(c.value for c in net.caps if "rwl_0" in (c.a + c.b) and c.a != c.b)
    _ladder(net, ["0", "1", "3"], loads, driven_left=True)
    after = sum(c.value for c in net.caps if "rwl_0__c" in (c.a + c.b))
    assert after == pytest.approx(before)
    assert not any(c.a == "arr/rwl_0" or c.b == "arr/rwl_0" for c in net.caps)


def test_drop_static_caps_removes_only_rail_to_rail_and_self_caps():
    net = FlatNetlist.parse("Ca vdd vss 1f\nCb a a 1f\nCc a vss 1f\nCd vnw vdd 1f\n")
    assert net.drop_static_caps({"vdd", "vss", "vnw"}) == 3
    assert [c.name for c in net.caps] == ["Cc"]


def test_emit_round_trips_devices_and_caps():
    net = FlatNetlist.parse(_tiny_array(1, 2))
    again = FlatNetlist.parse(net.emit())
    assert [d.line() for d in again.devices] == [d.line() for d in net.devices]
    assert [round(c.value, 20) for c in again.caps] == [round(c.value, 20) for c in net.caps]


# --- end to end (needs magic and ngspice) ---------------------------------------------------


@requires_timing
def test_characterise_a_small_decoded_replica_macro(tech, tmp_path):
    cfg = ArrayConfig.basic(rows=8, cols=6, replica_column=True)
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(tmp_path, name="char_small", macro=True, decoded=True, banked=False)
    tt = tuple(c for c in standard_corners() if c.process == "tt")
    report = compiler.characterise(result, corners=tt, levels=(1.0,))

    (corner,) = report.corners
    # a write drives the storage node to the rail through the PMOS; a '0' stops at ~|Vtp|
    assert corner.sn_one == pytest.approx(1.8, abs=0.02)
    assert 0.3 < corner.sn_zero < 1.0
    assert 0.1e-9 < corner.access < 3e-9
    assert corner.write_time is not None and 0 < corner.write_time < 1e-9
    # At tt the plain cell's written '0' (~0.69 V) is read correctly: the read device is only
    # weakly on and dout flips well after the strobe. (Only sf, where the write leaves ~0.88 V,
    # fails.) zero_level_limit, given the written level as its bound, then echoes it back.
    assert corner.reads_zero_correctly
    assert corner.zero_limit == pytest.approx(corner.sn_zero, abs=0.02)
    # the replica's rdy tracks the array's own read: within tens of percent of the data
    ratio = report.replica_ratios()[corner.corner.name]
    assert 0.8 < ratio < 1.5
    assert report.recommended_margin() >= 1.0
    assert (tmp_path / "char_small.timing.json").exists()
    assert "tt_1.8V_25C" in report.table()
