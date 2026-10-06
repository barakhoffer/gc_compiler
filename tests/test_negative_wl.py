"""The negative-rail write word-line driver: netlist, isolated-well geometry, and the macro."""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

import gdsfactory as gf
import klayout.db as kdb
import pytest

from gc_compiler.array.floorplan import gc_macro
from gc_compiler.array.tiler import array_geometry
from gc_compiler.compiler import GainCellCompiler
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.spice import (
    inverter_subckt,
    macro_netlist,
    macro_pins,
    neg_driver_subckt,
)
from gc_compiler.periphery.logic import LevelShifterConfig, LogicConfig
from gc_compiler.periphery.wl_driver import LEFT_WORDLINES, RIGHT_WORDLINES
from gc_compiler.periphery.wl_driver_neg import neg_driver_geometry, wl_driver_pair_neg
from gc_compiler.tech.pdk import activate
from gc_compiler.tech.rules import LayerRole
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.connectivity import trace_nets
from gc_compiler.verify.drc import LEAF_RULES, default_deck, run_drc
from gc_compiler.verify.lvs import default_magicrc, default_netgen_setup, lvs_layout_vs_netlist
from gc_compiler.verify.timing import standard_corners

LOGIC = LogicConfig()
LS = LevelShifterConfig()

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


def _cell(tech):
    return array_geometry(tech, ArrayConfig.basic()).cell


# --- configuration and netlist ------------------------------------------------------------


def test_negative_write_wl_needs_a_negative_level():
    assert ArrayConfig.basic(negative_write_wl=True).write_wl_low < 0
    with pytest.raises(ValueError, match="negative"):
        ArrayConfig.basic(negative_write_wl=True, write_wl_low=0.2)
    ArrayConfig.basic(write_wl_low=0.2)  # unused unless the option is on
    assert ArrayConfig.basic(rows=4, cols=2, negative_write_wl=True).name == "gc_array_4x2_neg_plain"
    assert ArrayConfig.basic(rows=4, cols=2).name == "gc_array_4x2_plain"
    assert ArrayConfig(rows=4, cols=2).name == "gc_array_4x2_rep_st_rim"  # the default cell adds nothing
    assert ArrayConfig(rows=4, cols=2, refresh_in_macro=False).name == "gc_array_4x2_rep_st"


def _devices(subckt: str) -> dict[str, list[str]]:
    return {
        line.split()[0][1:]: line.split() for line in subckt.splitlines() if line.startswith("X")
    }


def test_driver_subckt_is_a_level_shifter_and_an_output_inverter(tech):
    text = neg_driver_subckt(tech, LOGIC, LS)
    assert text.splitlines()[0] == ".subckt gc_wwl_neg in wwl vdd vss vneg"
    devices = _devices(text)
    assert set(devices) == {"inb", "pa", "pb", "nx", "nz", "pd", "nd"}
    # (drain gate source bulk): the NMOS all sit on the negative rail with their body there too
    for name in ("nx", "nz", "nd"):
        _d, _g, source, bulk = devices[name][1:5]
        assert (source, bulk) == ("vneg", "vneg")
    for name in ("pa", "pb", "pd"):
        assert devices[name][3:5] == ["vdd", "vdd"]
    # the latch cross-couples: each NMOS is gated by the other's drain
    assert devices["nx"][1:3] == ["x", "z"] and devices["nz"][1:3] == ["z", "x"]
    # inputs are PMOS at the top rail (fully off at either rail), not NMOS at vneg
    assert devices["pa"][2] == "inb" and devices["pb"][2] == "in"
    # the output stage is an inverter driven by the latch's true output
    assert devices["pd"][1:3] == ["wwl", "x"] and devices["nd"][1:3] == ["wwl", "x"]
    # the latch NMOS are long and the inputs wide -- it is a ratioed circuit
    assert f"l={LS.latch_length:g}" in " ".join(devices["nx"])
    assert f"w={LS.pmos_width:g}" in " ".join(devices["pa"])


def test_macro_netlist_adds_vneg_and_the_driver(tech):
    plain, neg = ArrayConfig.basic(rows=4, cols=2), ArrayConfig.basic(rows=4, cols=2, negative_write_wl=True)
    assert "vneg" not in macro_pins(plain) and macro_pins(neg)[-1] == "vneg"
    text = macro_netlist(tech, neg, LOGIC, name="n", decoded=False)
    assert ".subckt gc_wwl_neg" in text
    for row in range(4):
        assert f"Xwwl_drv_{row} wwl_in_{row} wwl_{row} vdd vss vneg gc_wwl_neg" in text
        assert f"Xrwl_drv_{row} rwl_in_{row} rwl_{row} vdd vss gc_inv" in text  # reads are unchanged
    assert "gc_wwl_neg" not in macro_netlist(tech, plain, LOGIC, name="p", decoded=False)


def test_compiler_reports_the_cost_of_the_negative_rail(tech, tmp_path):
    compiler = GainCellCompiler(ArrayConfig.basic(rows=4, cols=3, negative_write_wl=True), tech)
    result = compiler.build(tmp_path, name="neg_metrics", macro=True, decoded=False)
    m = result.metrics
    assert m["negative_write_wl"] is True and m["write_wl_low_v"] == compiler.cfg.write_wl_low
    assert m["write_oxide_stress_v"] == pytest.approx(1.8 - compiler.cfg.write_wl_low)
    plain = GainCellCompiler(ArrayConfig.basic(rows=4, cols=3), tech).metrics(macro=True)
    assert m["transistors"] - plain["transistors"] == 6 * 4  # six more devices per row
    assert m["driver_width_um"] > 3 * plain["driver_width_um"]
    assert "vneg" in result.spice.read_text()
    with pytest.raises(ValueError, match="macro"):
        compiler.build(tmp_path, name="neg_array_only", macro=False)


# --- geometry: the isolated well obeys the deep n-well rules -------------------------------------


def test_isolated_well_geometry_follows_the_deep_nwell_rules(tech):
    r = tech.rules
    g = neg_driver_geometry(tech, _cell(tech), LOGIC, LS)
    inner, ring = r.isolation_ring_inner, r.isolation_ring_width
    pb_x1 = g.pb_x0 + LS.pmos_width
    dn_left, dn_right = g.i0 - inner, g.i1 + inner
    assert dn_left > pb_x1  # dnwell.5: the deep n-well edge never cuts a P+ diffusion
    assert g.i0 - g.g0 >= ring - 1e-9 and g.wall_x1 - g.i1 >= ring - 1e-9  # both walls are wide enough
    assert g.nz_x0 - g.i0 >= r.nwell_to_ndiff_space - 1e-9  # NMOS diffusion clear of the n-well
    # nwell.7: the array's n-wells start at the strip's right edge, 4.5 um from the deep n-well
    assert g.width - dn_right >= r.isolation_nwell_space - 1e-9
    # the word-line landing sits against the array, not the wall (poly stubs are resistance)
    assert g.width - g.pad_x < 0.6 and g.track_x[0] > g.wall_x1


@pytest.mark.parametrize("wordlines", [LEFT_WORDLINES, RIGHT_WORDLINES])
def test_slice_keeps_the_plain_drivers_contract(tech, wordlines):
    cell = _cell(tech)
    sl = wl_driver_pair_neg(tech, cell, LOGIC, LS, wordlines)
    ports = {p.name: p for p in sl.ports}
    assert {"vdd", "vss", "vneg", *wordlines, *(f"{n}_in" for n in wordlines)} <= set(ports)
    g = neg_driver_geometry(tech, cell, LOGIC, LS)
    assert sl.info["width"] == pytest.approx(g.width) and sl.info["height"] == pytest.approx(2 * cell.pitch_y)
    for net in wordlines:  # the poly stubs leave on the right edge at the array's word-line heights
        assert ports[net].dcenter[0] == pytest.approx(g.width)
        assert ports[f"{net}_in"].dcenter[0] == pytest.approx(0.0)
    # the decoder's rows are ~2.29 um tall, so successive inputs down a strip must be that far apart
    height = 2 * cell.pitch_y
    ys = sorted(ports[f"{n}_in"].dcenter[1] for n in wordlines)
    assert ys[1] - ys[0] >= 2.29 and height - (ys[1] - ys[0]) >= 2.29
    # every well spans the whole slice, so slices stack into continuous wells
    region = kdb.Region(sl.kdb_cell.begin_shapes_rec(sl.kcl.layout.layer(*tech.layer(LayerRole.DNWELL))))
    box = region.bbox()
    assert (box.bottom, box.top) == (0, round(height * 1000))


# --- DRC / LVS -----------------------------------------------------------------------------------


@requires_drc
@pytest.mark.parametrize("wordlines", [LEFT_WORDLINES, RIGHT_WORDLINES])
def test_slice_is_drc_clean(tech, tmp_path, wordlines):
    gds = tmp_path / "slice.gds"
    wl_driver_pair_neg(tech, _cell(tech), LOGIC, LS, wordlines).write_gds(gds)
    report = run_drc(gds, waive=LEAF_RULES)  # the macro tests check it assembled
    assert report.clean, str(report)


def _labelled_slice(tech, wordlines, name):
    """A slice with a label on each port, so magic names the nets the netlist uses."""
    sl = wl_driver_pair_neg(tech, _cell(tech), LOGIC, LS, wordlines)
    top = gf.Component(name)
    top.add_ref(sl)
    label = {
        LayerRole.M2: tech.label_layer(LayerRole.M2),
        LayerRole.M1: tech.label_layer(LayerRole.M1),
        LayerRole.POLY: tech.label_layer(LayerRole.POLY),
    }
    for port in sl.ports:
        role = (
            LayerRole.POLY if port.name in wordlines
            else LayerRole.M2 if port.name.endswith("_in") else LayerRole.M1
        )
        top.add_label(text=port.name, position=tuple(port.dcenter), layer=label[role])
    return top


def _slice_netlist(tech, wordlines, name, body_of_nd="vneg"):
    read, neg = (wordlines[0], wordlines[1]) if wordlines == LEFT_WORDLINES else (wordlines[1], wordlines[0])
    driver = neg_driver_subckt(tech, LOGIC, LS)
    if body_of_nd != "vneg":
        driver = driver.replace(f"nfet_01v8 w={LOGIC.nmos_width:g} l={LOGIC.length:g}", "nfet_01v8 w=0.42 l=0.15")
        lines = driver.splitlines()
        i = next(k for k, l in enumerate(lines) if l.startswith("Xnd"))
        lines[i] = lines[i].replace("vneg vneg", f"vneg {body_of_nd}")
        driver = "\n".join(lines)
    pins = f"{wordlines[0]}_in {wordlines[0]} {wordlines[1]}_in {wordlines[1]} vdd vss vneg"
    return (
        f"{inverter_subckt(tech, LOGIC)}\n{driver}\n.subckt {name} {pins}\n"
        f"Xr {read}_in {read} vdd vss gc_inv\nXw {neg}_in {neg} vdd vss vneg gc_wwl_neg\n.ends\n"
    )


@requires_lvs
@pytest.mark.parametrize("wordlines", [LEFT_WORDLINES, RIGHT_WORDLINES])
def test_slice_lvs_matches_and_sees_the_isolated_body(tech, tmp_path, wordlines):
    activate(tech)
    name = f"lvs_neg_slice_{wordlines[0]}"
    gds = tmp_path / f"{name}.gds"
    _labelled_slice(tech, wordlines, name).write_gds(gds)
    good = tmp_path / "good.spice"
    good.write_text(_slice_netlist(tech, wordlines, name))
    result = lvs_layout_vs_netlist(gds, name, good, name, workdir=tmp_path / "lvs")
    assert result.matched and not result.mismatched_nets, str(result)
    assert result.net_count == (10, 10)  # 2 inputs, 2 outputs, 3 rails, and inb, x, z

    # an output NMOS whose body is on vss instead of vneg is a different circuit: the
    # isolated body is really part of what LVS compares
    bad = tmp_path / "bad.spice"
    bad.write_text(_slice_netlist(tech, wordlines, name, body_of_nd="vss"))
    assert not lvs_layout_vs_netlist(gds, name, bad, name, workdir=tmp_path / "lvs_bad").matched


@requires_drc
@requires_lvs
@pytest.mark.parametrize(
    "rows,cols,decoded,replica",
    [(8, 4, False, False), (12, 6, True, False), (16, 20, True, True)],
)
def test_negative_macro_drc_and_lvs(tech, tmp_path, rows, cols, decoded, replica):
    """The whole macro -- deep n-well ring closed and bridged over the tap-row gaps, the two strips'
    ``vneg`` rails joined -- is DRC clean and matches ``macro_netlist``, with and without the
    decoders, the replica, and a strap column (past 16 columns)."""
    cfg = ArrayConfig.basic(rows=rows, cols=cols, negative_write_wl=True, replica_column=replica)
    name = f"neg_m_{rows}x{cols}{'_d' if decoded else '_p'}{'_r' if replica else ''}"
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


# --- the macro's wells and rail, geometrically -----------------------------------------------------


@lru_cache(maxsize=None)
def _macro_gds(rows: int, cols: int) -> Path:
    tech = Sky130()
    cfg = ArrayConfig.basic(rows=rows, cols=cols, negative_write_wl=True)
    name = f"neg_geo_{rows}x{cols}"
    path = Path("/tmp") / f"{name}.gds"
    gc_macro(tech, cfg, LOGIC, name=name, decoded=False).write_gds(path)
    return path


def _region(layout: kdb.Layout, tech, role: LayerRole) -> kdb.Region:
    return kdb.Region(layout.top_cell().begin_shapes_rec(layout.layer(*tech.layer(role))))


def test_deep_nwell_rings_are_closed_and_enclose_their_interior(tech):
    """Each strip's n-well ring encloses exactly one hole (the isolated p-well), inside the deep
    n-well by the rule's margin, and the deep n-wells stay clear of every unrelated n-well."""
    gds = _macro_gds(10, 3)
    layout = kdb.Layout()
    layout.read(str(gds))
    r = tech.rules
    nwell, dnwell = _region(layout, tech, LayerRole.NWELL).merged(), _region(layout, tech, LayerRole.DNWELL).merged()
    unit = 1000
    holes = nwell.holes()
    assert holes.count() == 2  # one isolated p-well per driver strip
    assert dnwell.count() == 2 and dnwell.merged().count() == 2  # and one deep n-well each
    # dnwell.6 / nwell.6: the deep n-well encloses each hole by the ring's inner margin ...
    inner = round(r.isolation_ring_inner * unit)
    assert (holes.sized(inner) - dnwell).is_empty()
    # ... nwell.7: and no n-well outside it comes within 4.5 um of it
    space = round(r.isolation_nwell_space * unit)
    unrelated = nwell.not_interacting(dnwell)
    assert unrelated.count() > 0
    assert dnwell.separation_check(unrelated, space).count() == 0
    # dnwell.5: no P+ diffusion straddles a deep n-well edge
    pdiff = _region(layout, tech, LayerRole.DIFF) & _region(layout, tech, LayerRole.PIMP)
    assert (pdiff.interacting(dnwell) - pdiff.inside(dnwell)).is_empty()


@requires_drc
def test_vneg_is_one_net_apart_from_vss_and_vdd(tech):
    gds = _macro_gds(10, 3)
    nets = trace_nets(tech, gds)
    g = neg_driver_geometry(tech, _cell(tech), LOGIC, LS)
    layout = kdb.Layout()
    layout.read(str(gds))
    width = layout.top_cell().dbbox().width()
    y = layout.top_cell().dbbox().center().y
    rail = g.ptap_x + g.tap_w / 2
    left, right = (LayerRole.M1, (rail, y)), (LayerRole.M1, (width - rail, y))
    vss, vdd = (LayerRole.M1, (0.385 + 3.17, y)), (LayerRole.M1, (0.385, y))
    assert nets.same_net(left, right)  # the two strips' rails are joined over the array
    assert not nets.same_net(left, vss) and not nets.same_net(left, vdd)
    assert not nets.same_net(vss, vdd)


# --- the point of it: a written '0' now reads as a '0' --------------------------------------------------


@requires_timing
def test_a_written_zero_reads_back_in_every_corner(tech, tmp_path):
    from gc_compiler.verify.timing import characterise

    cfg = ArrayConfig.basic(rows=8, cols=5, replica_column=True, negative_write_wl=True)
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(tmp_path, name="neg_char", macro=True, decoded=True, banked=False)
    corners = tuple(c for c in standard_corners() if c.process in ("tt", "sf"))  # sf is the worst
    report = compiler.characterise(result, corners=corners, levels=(1.0,), monte_carlo=3)

    # the '0' a plain PMOS write leaves (measurements/data/characterise.json); the rail lowers it by
    # about 0.25 V at the default -0.3 V (0.6 V at -0.8 V)
    plain_zero = {"tt": 0.69, "sf": 0.88}

    # local mismatch on top: every die's own write leaves a '0' that still reads as one
    for c in report.corners:
        assert c.mc.n == 3 and c.mc.zero_failures == 0 and c.mc.zero_level is None, c.corner.name
        assert all(s.write0.sn_final < plain_zero[c.corner.process] - 0.1 for s in c.mc.samples), c.corner.name

    assert report.vneg == cfg.write_wl_low
    for c in report.corners:
        assert c.sn_zero < plain_zero[c.corner.process] - 0.15, c.corner.name
        assert c.reads_zero_correctly, c.corner.name
        assert c.sn_one == pytest.approx(c.corner.vdd, abs=0.03)  # writing a '1' is unaffected
        # a level shifter that is properly off leaks like any logic gate: microamps for the macro
        assert c.supply.vneg_a < 20e-6
    assert report.write_stress() == pytest.approx(1.8 - cfg.write_wl_low)
    assert "vneg peak" in report.supply_table()
    assert report.to_dict()["vneg_v"] == cfg.write_wl_low
