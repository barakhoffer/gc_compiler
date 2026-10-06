from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from gc_compiler.compiler import GainCellCompiler
from gc_compiler.config import ArrayConfig
from gc_compiler.tech.rules import Technology
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.lvs import default_magicrc
from gc_compiler.verify.montecarlo import (
    MonteCarlo,
    Sample,
    Spread,
    _ends,
    monte_carlo,
    table,
    upper_bound,
)
from gc_compiler.verify.timing import (
    BenchSettings,
    ReadTiming,
    TimingModel,
    WriteTiming,
    _deck,
    build_model,
    driven_from_left,
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

TT = standard_corners()[0]


@pytest.fixture(scope="session")
def tech() -> Sky130:
    return Sky130()


# --- statistics ------------------------------------------------------------------------


def test_spread_of_known_values():
    spread = Spread.of([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spread.n == 5 and spread.mean == 3.0
    assert spread.std == pytest.approx(1.5811, abs=1e-3)  # sample (n-1) standard deviation
    assert (spread.minimum, spread.p50, spread.maximum) == (1.0, 3.0, 5.0)
    assert spread.at(3.0) == pytest.approx(3.0 + 3 * spread.std)


def test_spread_of_nothing_and_of_one_value():
    assert Spread.of([]) is None
    one = Spread.of([2.5])
    assert one.mean == 2.5 and one.std == 0.0


def test_upper_bound_of_a_failure_rate():
    assert upper_bound(0, 0) == 1.0
    assert 0.03 < upper_bound(0, 100) < 0.045  # zero failures in 100 dies: "below ~4%"
    assert upper_bound(0, 100) < upper_bound(1, 100) < upper_bound(5, 100)
    assert upper_bound(0, 1000) < upper_bound(0, 100)  # more dies, tighter bound
    assert upper_bound(10, 10) == pytest.approx(1.0)


def _read(data: int, t_dout: float | None, t_rdy: float = 1e-9, correct: bool = True) -> ReadTiming:
    return ReadTiming(
        row=1, col="0", data=data, sn_level=1.0, t_decoder=5e-10, t_driver=1e-10, t_bitline=2e-10,
        t_dout=t_dout, t_rdy=t_rdy, rbl_min=0.0, dout_max=1.8, correct=correct,
    )


def _write(sn_final: float, t_write: float = 1.5e-10) -> WriteTiming:
    return WriteTiming(
        row=1, col="0", data=1, t_decoder=5e-10, t_driver=1e-10, t_write=t_write,
        sn_start=0.4, sn_final=sn_final,
    )


def _sample(seed: int, tracking: float, flip: float | None, one_ok=True, zero_ok=True) -> Sample:
    return Sample(
        seed,
        read1=_read(1, tracking * 1e-9, correct=one_ok),
        read0=_read(0, None if flip is None else flip * 1e-9, correct=zero_ok),
        write1=_write(1.8),
        write0=_write(0.7 + 0.01 * seed),
    )


def _mc(samples: list[Sample]) -> MonteCarlo:
    return MonteCarlo(TT, row=1, one_level=0.7, zero_level=0.2, strobe_factor=1.25, samples=samples)


def test_sample_ratios_handle_missing_edges():
    assert _sample(1, 1.1, 3.0).tracking == pytest.approx(1.1)
    assert _sample(1, 1.1, 3.0).flip == pytest.approx(3.0)
    assert _sample(1, 1.1, None).flip is None  # a '0' that never flips
    assert Sample(1, error="boom").tracking is None and Sample(1, error="boom").flip is None
    assert Sample(1, read1=_read(1, 1e-9, t_rdy=0.0)).tracking is None  # rdy never fired


def test_monte_carlo_statistics_and_margin():
    mc = _mc([_sample(i, t, 3.0) for i, t in enumerate((1.00, 1.04, 1.08, 1.12, 1.06), 1)])
    assert mc.n == 5 and mc.errors == 0
    spread = mc.tracking
    assert spread.mean == pytest.approx(1.06)
    assert mc.margin(3.0) == pytest.approx(spread.mean + 3 * spread.std)
    # (1.25 - mean) / std standard deviations lie between the average die and the strobe
    assert mc.sigma_to_strobe() == pytest.approx((1.25 - spread.mean) / spread.std)
    assert mc.write_time.mean == pytest.approx(1.5e-10)
    assert mc.sn_zero.n == 5 and mc.sn_zero.mean == pytest.approx(0.73)


def test_margin_is_never_below_one():
    mc = _mc([_sample(i, t, 3.0) for i, t in enumerate((0.95, 0.96, 0.97), 1)])
    assert mc.margin(3.0) == 1.0


def test_failures_are_counted_and_failed_runs_left_out():
    mc = _mc(
        [
            _sample(1, 1.0, 3.0),
            _sample(2, 1.4, 3.0, one_ok=False),
            _sample(3, 1.0, 1.1, zero_ok=False),
            Sample(4, error="ngspice produced no output"),
        ]
    )
    assert (mc.n, mc.errors) == (3, 1)
    assert (mc.one_failures, mc.zero_failures) == (1, 1)
    assert mc.tracking.n == 3  # the errored die has no read


def test_flip_spread_leaves_out_dies_that_never_flip():
    mc = _mc([_sample(1, 1.0, 3.0), _sample(2, 1.0, None), _sample(3, 1.0, 5.0)])
    assert mc.flip.n == 2 and mc.flip.mean == pytest.approx(4.0)


def test_summary_table_and_json_round_trip():
    mc = _mc([_sample(i, 1.0 + 0.02 * i, 3.0) for i in range(1, 5)])
    text = table([mc])
    assert "tt_1.8V_25C" in text and "0/4" in text and "σ" in text
    assert text.count("\n") == 2  # header, rule, one corner
    data = json.loads(json.dumps(mc.to_dict()))
    assert data["n"] == 4 and data["corner"] == TT.name
    assert len(data["samples"]) == 4 and data["margin_3sigma"] >= 1.0
    assert data["sn_zero_v"]["n"] == 4


# --- plumbing --------------------------------------------------------------------------------


def test_model_header_switches_mismatch(tech):
    nominal = tech.model_header("tt")
    varied = tech.model_header("tt", mismatch=True)
    assert ".param mc_mm_switch=0" in nominal and ".param mc_mm_switch=1" in varied
    assert [l for l in nominal if l.startswith(".include")] == [l for l in varied if l.startswith(".include")]


def test_default_header_names_the_libraries_mismatch_section(tech):
    class Stock(Sky130):
        model_header = Technology.model_header

    assert Stock().model_header("ss", mismatch=True)[0].endswith(" ss_mm")
    assert Stock().model_header("ss")[0].endswith(" ss")


def _stub_model(tmp_path: Path) -> TimingModel:
    return TimingModel(
        netlist=tmp_path / "timing.spice", rows=4, cols=3, replica=False, kept=("0", "2"),
        prefix="a/", sn={}, wordline={}, decoder={}, vnw=("vnw",), read_length=0.15, write_length=0.18,
    )


def test_deck_seeds_mismatch_only_when_asked(tech, tmp_path):
    args = (tech, _stub_model(tmp_path), TT, [], {}, {}, 1e-9)
    nominal = _deck(*args, BenchSettings(), "w.csv")
    varied = _deck(*args, replace(BenchSettings(), seed=7), "w.csv")
    assert ".option seed" not in nominal and "mc_mm_switch=0" in nominal
    assert ".option seed=7" in varied and "mc_mm_switch=1" in varied


def test_word_line_sides_alternate_by_row_pair():
    assert [driven_from_left("rwl", r) for r in range(4)] == [True, False, True, False]
    assert [driven_from_left("wwl", r) for r in range(4)] == [False, True, False, True]


def test_ends_are_nearest_and_farthest_from_the_driver(tech, tmp_path):
    model = _stub_model(tmp_path)
    assert _ends(model, "rwl", 0) == ("0", "2")  # driven from the left: column 2 is far
    assert _ends(model, "rwl", 1) == ("2", "0")
    assert _ends(model, "wwl", 1) == ("0", "2")


def test_monte_carlo_argument_checks(tech, tmp_path):
    model = _stub_model(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        monte_carlo(tech, model, TT, 0, tmp_path)
    with pytest.raises(ValueError, match="two columns"):
        monte_carlo(tech, replace(model, kept=("0",)), TT, 2, tmp_path)


# --- end to end (needs magic and ngspice) -------------------------------------------------------


@requires_timing
def test_mismatch_makes_every_die_different_and_every_seed_repeatable(tech, tmp_path):
    cfg = ArrayConfig.basic(rows=8, cols=5, replica_column=True)
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(tmp_path, name="mc_small", macro=True, decoded=True, banked=False)
    model = build_model(tech, cfg, result.gds, result.component.name, tmp_path / "extract")

    first = monte_carlo(tech, model, TT, 3, tmp_path / "a", seed=11, jobs=3)
    again = monte_carlo(tech, model, TT, 2, tmp_path / "b", seed=11, jobs=2)  # seeds 11 and 12
    assert first.errors == 0 and first.n == 3

    # the same seed is the same die: identical waveforms, identical measurements
    for a, b in zip(first.samples[:2], again.samples):
        assert a.seed == b.seed
        assert a.read1.t_dout == b.read1.t_dout and a.read0.t_dout == b.read0.t_dout
        assert a.write0.sn_final == b.write0.sn_final
    # different seeds are different dies, and mismatch moves the result off the nominal one
    douts = {s.read1.t_dout for s in first.samples}
    assert len(douts) == 3
    assert len({s.write0.sn_final for s in first.samples}) == 3
    assert first.tracking.std > 0
    assert 0.1e-9 < first.access.mean < 3e-9  # a plausible read, not a broken bench
    assert first.margin(3.0) >= 1.0

    # a die's '0' starts from the level its own write left (0.6-0.9 V for this plain PMOS write),
    # unless a fixed level is asked for. At tt that '0' turns the read device only weakly on:
    # its bit line falls too slowly to flip dout before the strobe, so it reads correctly.
    for sample in first.samples:
        assert sample.read0.sn_level == pytest.approx(sample.write0.sn_final)
        assert 0.4 < sample.read0.sn_level < 1.0 and sample.read0.correct
    fixed = monte_carlo(tech, model, TT, 1, tmp_path / "c", seed=11, jobs=1, zero_level=0.2)
    assert fixed.samples[0].read0.sn_level == 0.2 and fixed.zero_level == 0.2
    assert first.zero_level is None and first.to_dict()["zero_from_write"] is True


@requires_timing
def test_characterise_runs_monte_carlo_and_saves_it(tech, tmp_path):
    cfg = ArrayConfig.basic(rows=8, cols=3, replica_column=True)
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(tmp_path, name="mc_char", macro=True, decoded=True, banked=False)
    tt = tuple(c for c in standard_corners() if c.process == "tt")
    report = compiler.characterise(result, corners=tt, levels=(1.0,), monte_carlo=2, mc_seed=3)

    (corner,) = report.corners
    assert corner.mc is not None and corner.mc.n == 2
    assert [s.seed for s in corner.mc.samples] == [3, 4]
    assert report.statistical_margin(3.0) >= 1.0
    assert "tt_1.8V_25C" in report.mc_table()
    saved = json.loads((tmp_path / "mc_char.timing.json").read_text())
    assert saved["corners"][0]["monte_carlo"]["n"] == 2
    assert saved["statistical_margin_3sigma"] >= 1.0
