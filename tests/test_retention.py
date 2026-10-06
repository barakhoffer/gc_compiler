"""Single-cell retention: how long a written level holds before it drifts (``verify/sim.py``).

The comparative check here is the point of it: an earlier, less careful single-corner estimate
in the README claimed the LVT-write option (see ``test_lvt_write.py``) had *shorter* retention
than the plain cell. A proper sweep across all five corners found the opposite at four of them --
this file is the regression guard for that finding.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from gc_compiler.config import ArrayConfig, BitcellConfig
from gc_compiler.compiler import GainCellCompiler
from gc_compiler.tech.sky130 import Sky130
from gc_compiler.verify.lvs import default_magicrc
from gc_compiler.verify.sim import RetentionTestbench, refresh_interval, run_retention
from gc_compiler.verify.timing import (
    BenchSettings,
    build_model,
    macro_refresh_interval,
    standard_corners,
    zero_level_limit,
)

LVT_WRITE = BitcellConfig.plain(write_flavor="lvt", write_length=0.35, read_length=0.5)

_model_file = Path(Sky130().model_library()).parent / "corners" / "tt.spice"
requires_ngspice = pytest.mark.skipif(
    shutil.which("ngspice") is None or not _model_file.exists(),
    reason="ngspice or the sky130 model files are missing",
)
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


def _drift_time(tech, cfg, corner, tmp_path, drift: float = 0.2, hold_time: float = 2e-3) -> tuple[float, float | None]:
    """Write a '0' (no negative rail: ``wwl_low=0``, the real macro's actual swing), then time
    how long it takes to drift up by ``drift`` volts from where it settled. Returns
    ``(settled_level, drift_time)`` -- ``drift_time`` is ``None`` if it never crosses within
    ``hold_time``."""
    tb = RetentionTestbench(
        data=0, vdd=corner.vdd, temperature=corner.temp, corner=corner.process,
        wwl_low=0.0, reads=0, hold_time=hold_time,
    )
    result = run_retention(tech, cfg, tb, workdir=tmp_path)
    t, sn = result.time, result["sn"]
    write_end = 100e-9 + tb.write_width
    settle_at = write_end + tb.write_width
    settled = float(sn[min(np.searchsorted(t, settle_at), len(sn) - 1)])
    return settled, result.retention_time(settled + drift, rising=True, after=settle_at)


@requires_ngspice
def test_writing_a_zero_with_no_underdrive_settles_below_a_one(tech, tmp_path):
    """Sanity check of the bench itself, independent of the comparative claim below."""
    tb0 = RetentionTestbench(data=0, wwl_low=0.0, hold_time=10e-6, reads=0)
    tb1 = RetentionTestbench(data=1, wwl_low=0.0, hold_time=10e-6, reads=0)
    zero = run_retention(tech, BitcellConfig.plain(), tb0, workdir=tmp_path)
    one = run_retention(tech, BitcellConfig.plain(), tb1, workdir=tmp_path)
    write_end = 100e-9 + tb0.write_width
    at = write_end + 20e-9
    sn0 = float(zero["sn"][np.searchsorted(zero.time, at)])
    sn1 = float(one["sn"][np.searchsorted(one.time, at)])
    assert 0.3 < sn0 < 1.2  # a plain PMOS write's '0': nowhere near a full rail-to-rail swing
    assert 1.65 < sn1 < 1.95  # writing a '1' is not threshold-limited (close to vdd, +/- overshoot)


@requires_ngspice
@pytest.mark.parametrize("process", ["tt", "ss", "ff", "sf", "fs"])
def test_lvt_write_does_not_shorten_zero_retention(tech, tmp_path, process):
    """The regression guard: at every corner but ``ss``, the LVT write's '0' should drift the
    same 0.2 V no faster than the plain cell's -- and even at ``ss`` not by much."""
    corner = next(c for c in standard_corners() if c.process == process)
    _, plain_drift = _drift_time(tech, BitcellConfig.plain(), corner, tmp_path / "plain")
    _, lvt_drift = _drift_time(tech, LVT_WRITE, corner, tmp_path / "lvt")
    assert plain_drift is not None and lvt_drift is not None
    # ss is the one corner where the LVT write is worse (by ~6% in the full characterisation);
    # everywhere else it should be a real improvement, not just "not worse".
    if process == "ss":
        assert lvt_drift > 0.85 * plain_drift
    else:
        assert lvt_drift > plain_drift


# --- refresh_interval: the true-threshold measurement behind the README's numbers ------------

# The true read-tolerance thresholds (verify.timing.zero_level_limit, bisected on the extracted
# 16x20 macro with a generous ``high`` instead of the written level -- see that function's
# docstring) that produced the README's refresh-interval table. Hardcoded here rather than
# re-extracted per test run: re-deriving them needs magic and a macro build, which these tests
# are not otherwise paying for, and the point of this file is refresh_interval's own logic, not
# re-proving the macro's read tolerance (test_lvt_write.py's DRC/LVS/timing tests already cover
# the macro itself).
PLAIN_THRESHOLD_TT = 0.45
LVT_THRESHOLD_TT = 0.6375


@requires_ngspice
def test_refresh_interval_is_zero_when_the_write_already_exceeds_the_threshold(tech, tmp_path):
    """The plain cell's written '0' at tt (~0.65 V) is already past its own read tolerance
    (0.45 V) -- matches the original "reads as a '1' in every corner but fs" finding: there is
    no interval during which the data is good, not a small or negative one."""
    result = refresh_interval(tech, BitcellConfig.plain(), PLAIN_THRESHOLD_TT, workdir=tmp_path)
    assert result.margin_v < 0
    assert result.refresh_time_s == 0.0


@requires_ngspice
def test_refresh_interval_finds_the_real_crossing_when_there_is_margin(tech, tmp_path):
    """The LVT write's '0' at tt clears its own (higher) read tolerance with real margin, and
    the measured refresh interval is the same order of magnitude as the README's (~292 us)."""
    result = refresh_interval(tech, LVT_WRITE, LVT_THRESHOLD_TT, workdir=tmp_path)
    assert result.margin_v > 0
    assert result.refresh_time_s is not None
    assert 100e-6 < result.refresh_time_s < 600e-6


@requires_ngspice
def test_refresh_interval_gives_up_after_max_doublings(tech, tmp_path):
    """A threshold nothing will ever reach (past vdd) returns ``None``, not an infinite loop."""
    result = refresh_interval(
        tech, LVT_WRITE, 5.0, workdir=tmp_path, hold0=10e-6, max_doublings=2,
    )
    assert result.refresh_time_s is None
    assert result.attempts == 2
    assert result.hold_time_s == pytest.approx(40e-6)  # 10 us, doubled (x4) once for the 2nd try


# --- macro_refresh_interval: the same measurement on the real extracted macro ----------------


@requires_timing
def test_macro_refresh_interval_matches_the_single_cell_estimate_in_order_of_magnitude(tech, tmp_path):
    """``macro_refresh_interval`` writes and holds on the real extracted macro (real wordline/
    bitline RC) instead of a bare cell. This is also the regression guard for two bugs found
    building it: ``BenchSettings.step`` (tens of ps, sized for a ~10-30 ns bench) left uncapped
    over a multi-us to multi-ms hold asked ngspice for tens of millions of print points and it
    would hang; and the settled-level reading, taken from whichever attempt happened to find the
    crossing, came out wrong once that attempt's step had been coarsened for a long hold."""
    # a size not used by any other LVT_WRITE macro in the suite: component names are keyed by
    # config, not by the `name=` passed to build(), so a size collision reuses the wrong cell.
    cfg = ArrayConfig.basic(rows=8, cols=9, replica_column=True, bitcell=LVT_WRITE)
    result = GainCellCompiler(cfg, tech).build(tmp_path, name="refresh_macro_test", macro=True, decoded=True, banked=False)
    model = build_model(tech, cfg, result.gds, result.component.name, tmp_path / "extract")
    corner = next(c for c in standard_corners() if c.process == "tt")
    settings = BenchSettings()
    far_row, far_col = 0, model.kept[-1]

    threshold = zero_level_limit(
        tech, model, corner, far_row, far_col, high=1.2, workdir=tmp_path / "limit", settings=settings,
    )
    r = macro_refresh_interval(
        tech, model, corner, far_row, far_col, threshold, workdir=tmp_path / "refresh", settings=settings,
    )
    assert 0.2 < r.settled_v < 0.6  # a real LVT '0' at tt, not the near-zero a stray .ic would give
    assert r.margin_v > 0
    assert r.refresh_time_s is not None
    assert 50e-6 < r.refresh_time_s < 5e-3  # same order of magnitude as every other corner measured
