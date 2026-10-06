"""Self-timed reads: the replica's rdy gates the decoder's read enable, so a read ends by itself."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gc_compiler.array.floorplan import gc_banked_macro, gc_macro
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.decoder import decoder_gates, decoder_inputs, row_decoder_pins
from gc_compiler.netlist.spice import macro_netlist
from gc_compiler.periphery.logic import LogicConfig
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


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


def test_self_timed_needs_the_replica():
    with pytest.raises(ValueError, match="replica"):
        ArrayConfig.basic(self_timed_read=True)
    ArrayConfig.basic(self_timed_read=True, replica_column=True)


@pytest.mark.parametrize("banked", [False, True], ids=["plain", "banked"])
def test_rdy_is_one_more_term_of_every_read_bus_line(banked):
    """rdy (both banks' in a banked decoder) is an input of the decoder, and every read row's
    output depends on it -- it is one more term of each read bus line -- while no write row's does."""
    cfg = ArrayConfig.basic(rows=8, cols=4, replica_column=True, self_timed_read=True)
    rdy = ("rdy_0", "rdy_1") if banked else ("rdy",)
    assert decoder_inputs(cfg, banked)[-len(rdy):] == list(rdy)
    assert row_decoder_pins(cfg, banked=banked)[: len(decoder_inputs(cfg, banked))] == decoder_inputs(cfg, banked)
    gates = decoder_gates(cfg, banked=banked)
    for net in rdy:
        reached, grew = {net}, True
        while grew:
            grew = False
            for g in gates:
                if g.pins["y"] not in reached and any(n in reached for p, n in g.pins.items() if p != "y"):
                    reached.add(g.pins["y"])
                    grew = True
        assert all(g.pins["y"] in reached for g in gates if g.role == "row_nand_r")
        assert not any(g.pins["y"] in reached for g in gates if g.role in ("row_nand_w", "row_inv"))
    plain = ArrayConfig.basic(rows=8, cols=4, replica_column=True)
    assert not any("rdy" in n for g in decoder_gates(plain, banked=banked) for n in g.pins.values())


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
        (4, 19, {}, False),
        (4, 21, {"wordline_straps": True, "negative_write_wl": True, "wordline_driver_strength": 4}, False),
        (6, 17, {}, True),
        (6, 19, {"wordline_straps": True}, True),
    ],
    ids=["plain", "plain_straps_neg_x4", "banked", "banked_straps"],
)
def test_self_timed_macro_drc_and_lvs(tech, tmp_path, rows, cols, extra, banked):
    """rdy reaches the decoders' trunk: round the plain macro's right side on metal3, and under
    each bank to the central decoder in a banked one."""
    cfg = ArrayConfig.basic(rows=rows, cols=cols, replica_column=True, self_timed_read=True, **extra)
    _check(tech, cfg, tmp_path, f"st_{rows}x{cols}{'_b' if banked else ''}", banked)


@requires_timing
def test_a_read_ends_by_itself(tech, tmp_path):
    """With re held for the whole bench window, a stored '0' still reads '0' at the end of it,
    near and far from the drivers, at tt and ff (where a fixed-time read of the same macro lets
    it flip after about a nanosecond): the read word line has turned itself off."""
    from gc_compiler.verify.timing import build_model, simulate_reads, simulate_writes, standard_corners

    cfg = ArrayConfig.basic(rows=4, cols=18, replica_column=True, self_timed_read=True)
    gds = tmp_path / "st_read.gds"
    gc_macro(tech, cfg, name="st_read", decoded=True).write_gds(gds)
    model = build_model(tech, cfg, gds, "st_read", tmp_path / "model")
    near, far = model.kept[0], model.kept[-1]
    for corner in (c for c in standard_corners() if c.process in ("tt", "ff")):
        written = simulate_writes(tech, model, corner, 0, {near: 0, far: 0}, tmp_path / f"{corner.process}_w")
        for zero, one in ((near, far), (far, near)):
            reads = simulate_reads(
                tech, model, corner, 0, {zero: (0, written[zero].sn_final), one: (1, corner.vdd)},
                tmp_path / f"{corner.process}_{zero}",
            )
            for col, r in reads.items():
                assert r.correct, (corner.name, col, r)
                assert r.t_cutoff is not None and r.t_cutoff < 5e-9, (corner.name, col, r)
            assert reads[one].t_dout < reads[one].t_cutoff  # the data was there before the cutoff
