"""Monte-Carlo (local mismatch) analysis of a macro's read and write timing.

:mod:`gc_compiler.verify.timing` characterises the nominal devices at each PVT corner. That
says how long a read takes; it does not say how far the replica bit line can be trusted, because
what limits a self-timed read is not the average cell but the *difference* between the replica's
path and the data column's. Both are built of minimum-size devices, and for those the spread of
threshold voltage between two neighbours (Pelgrom: ``sigma ~ 1/sqrt(W*L)``) is a large fraction
of the gate overdrive -- and a gain cell's read pair is operated in exactly that regime.

Each **sample** is one die: a seed that gives every transistor of the reduced macro (the
decoders, drivers, the two kept end columns, the replica, all column circuits) its own random
threshold, subthreshold-slope and oxide offsets from the PDK's mismatch model. The same seed is
used for both accesses of a sample (a read and a write), so they see the same devices:

* a write of a '1' and of a '0' -> the write time and the level the '0' settles at;
* a read of a decayed '1' at the far cell (and, in the same access, of a '0' at the near cell)
  -> ``dout`` and ``rdy`` times, hence the tracking ratio ``t_dout / t_rdy`` that sets how long
  ``rdy`` has to be delayed, and whether the '0' survives until the data is sampled. The '0' starts
  from the level *that die's own write* left, unless a fixed ``zero_level`` is given (what a write
  underdrive to that level would give): so a macro whose write cannot reach a readable '0' fails
  here, and one with a negative write word line passes with the margin its mismatch leaves.

Mismatch is *local*: it is superimposed on the corner, so the sweep over corners (which stands
for die-to-die variation) and the sweep over seeds (within-die) are independent.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from gc_compiler.tech.rules import Technology
from gc_compiler.verify.timing import (
    BenchSettings,
    Corner,
    ReadTiming,
    TimingModel,
    WriteTiming,
    driven_from_left,
    simulate_reads,
    simulate_writes,
)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


@dataclass
class Sample:
    """One die: the read and the write that share a mismatch seed."""

    seed: int
    read1: ReadTiming | None = None
    read0: ReadTiming | None = None
    write1: WriteTiming | None = None
    write0: WriteTiming | None = None
    error: str | None = None
    """Why the sample has no results (a simulation that did not run to the end)."""

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def tracking(self) -> float | None:
        """``t_dout / t_rdy`` of the '1' read: how much later than ``rdy`` the data is valid."""
        return None if self.read1 is None else _ratio(self.read1.t_dout, self.read1.t_rdy)

    @property
    def flip(self) -> float | None:
        """When the '0' read's ``dout`` flips wrongly, as a multiple of ``rdy`` (``None`` if it
        does not flip at all)."""
        return None if self.read0 is None else _ratio(self.read0.t_dout, self.read0.t_rdy)


@dataclass(frozen=True)
class Spread:
    """Sample statistics of one quantity."""

    n: int
    mean: float
    std: float
    minimum: float
    maximum: float
    p01: float
    p50: float
    p99: float

    @classmethod
    def of(cls, values: list[float]) -> Spread | None:
        if not values:
            return None
        v = np.asarray(values, dtype=float)
        p01, p50, p99 = np.percentile(v, [1, 50, 99])
        return cls(
            n=len(v),
            mean=float(v.mean()),
            std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            minimum=float(v.min()),
            maximum=float(v.max()),
            p01=float(p01),
            p50=float(p50),
            p99=float(p99),
        )

    def at(self, sigma: float) -> float:
        """``mean + sigma * std``: the value that many standard deviations out, assuming the
        quantity is Gaussian (a fit worth trusting only as far as the histogram looks it)."""
        return self.mean + sigma * self.std


def upper_bound(failures: int, n: int, z: float = 1.96) -> float:
    """Upper limit (Wilson, ~97.5% one-sided) on a failure probability seen ``failures`` times
    in ``n`` trials: with zero failures in 100 dies, all one can say is 'below ~4%'."""
    if n == 0:
        return 1.0
    p = failures / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + spread) / (1 + z * z / n))


@dataclass
class MonteCarlo:
    corner: Corner
    row: int
    one_level: float
    """The '1' the reads start from, as a fraction of the supply."""
    zero_level: float | None
    """The '0' the reads start from, in volts; ``None`` if it is each die's own written level."""
    strobe_factor: float
    samples: list[Sample] = field(default_factory=list)

    @property
    def good(self) -> list[Sample]:
        return [s for s in self.samples if s.ok]

    @property
    def n(self) -> int:
        return len(self.good)

    @property
    def errors(self) -> int:
        return len(self.samples) - self.n

    def _spread(self, values) -> Spread | None:
        return Spread.of([v for v in values if v is not None])

    # --- distributions ---------------------------------------------------------------

    @property
    def access(self) -> Spread | None:
        """``re`` to ``dout`` of the '1' read, seconds."""
        return self._spread(s.read1.t_dout for s in self.good)

    @property
    def rdy(self) -> Spread | None:
        return self._spread(s.read1.t_rdy for s in self.good)

    @property
    def tracking(self) -> Spread | None:
        """``t_dout / t_rdy``: the delay ``rdy`` needs to be a valid strobe for this die."""
        return self._spread(s.tracking for s in self.good)

    @property
    def flip(self) -> Spread | None:
        """When the '0' read flips, in units of ``rdy`` time (dies where it never does are left
        out of this and counted as passing)."""
        return self._spread(s.flip for s in self.good)

    @property
    def write_time(self) -> Spread | None:
        return self._spread(s.write1.t_write for s in self.good)

    @property
    def sn_zero(self) -> Spread | None:
        return self._spread(s.write0.sn_final for s in self.good)

    @property
    def sn_one(self) -> Spread | None:
        return self._spread(s.write1.sn_final for s in self.good)

    # --- yield -----------------------------------------------------------------------

    @property
    def one_failures(self) -> int:
        """Dies whose '1' was not on ``dout`` at the strobe."""
        return sum(not s.read1.correct for s in self.good)

    @property
    def zero_failures(self) -> int:
        """Dies whose '0' had flipped ``dout`` by the strobe."""
        return sum(not s.read0.correct for s in self.good)

    def margin(self, sigma: float = 3.0) -> float | None:
        """The factor ``rdy`` has to be delayed by to be valid for dies up to ``sigma``
        standard deviations out (never below 1)."""
        spread = self.tracking
        return None if spread is None else max(1.0, spread.at(sigma))

    def sigma_to_strobe(self) -> float | None:
        """How many standard deviations of the tracking ratio lie between its mean and the
        strobe (``strobe_factor``): the figure of merit for the delay chosen."""
        spread = self.tracking
        if spread is None or spread.std == 0:
            return None
        return (self.strobe_factor - spread.mean) / spread.std

    def to_dict(self) -> dict:
        def spread(s: Spread | None) -> dict | None:
            return None if s is None else asdict(s)

        return {
            "corner": self.corner.name,
            "row": self.row,
            "one_level": self.one_level,
            "zero_level_v": self.zero_level,
            "zero_from_write": self.zero_level is None,
            "strobe_factor": self.strobe_factor,
            "n": self.n,
            "errors": self.errors,
            "access_s": spread(self.access),
            "rdy_s": spread(self.rdy),
            "tracking": spread(self.tracking),
            "margin_3sigma": self.margin(3.0),
            "sigma_to_strobe": self.sigma_to_strobe(),
            "flip_over_rdy": spread(self.flip),
            "write_s": spread(self.write_time),
            "sn_zero_v": spread(self.sn_zero),
            "sn_one_v": spread(self.sn_one),
            "one_failures": self.one_failures,
            "zero_failures": self.zero_failures,
            "samples": [asdict(s) for s in self.samples],
        }


def _ends(model: TimingModel, kind: str, row: int) -> tuple[str, str]:
    """The kept columns nearest to and farthest from a word line's driver."""
    first, last = model.kept[0], model.kept[-1]
    return (first, last) if driven_from_left(kind, row) else (last, first)


def _sample(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    seed: int,
    row: int,
    one_level: float,
    zero_level: float | None,
    workdir: Path,
    settings: BenchSettings,
) -> Sample:
    settings = replace(settings, seed=seed)
    sample = Sample(seed)
    try:
        near, far = _ends(model, "wwl", row)
        writes = simulate_writes(tech, model, corner, row, {far: 1, near: 0}, workdir / "write", settings)
        sample.write1, sample.write0 = writes[far], writes[near]
        # One access serves both polarities: a '1' is slowest to arrive at the end of the word
        # line farthest from its driver, and the '0' goes at the other end (on the default row
        # that is the last column, whose '0' the nominal runs show flipping soonest against rdy).
        near, far = _ends(model, "rwl", row)
        zero = sample.write0.sn_final if zero_level is None else zero_level
        reads = simulate_reads(
            tech, model, corner, row,
            {far: (1, one_level * corner.vdd), near: (0, zero)},
            workdir / "read", settings,
        )
        sample.read1, sample.read0 = reads[far], reads[near]
    except RuntimeError as error:  # ngspice gave no result: keep going, count it
        sample.error = str(error).splitlines()[0]
    return sample


def monte_carlo(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    samples: int,
    workdir: str | Path,
    row: int = 1,
    one_level: float | None = None,
    zero_level: float | None = None,
    seed: int = 1,
    jobs: int | None = None,
    settings: BenchSettings | None = None,
) -> MonteCarlo:
    """``samples`` dies at ``corner``, seeds ``seed .. seed + samples - 1``.

    Each die is two simulations of row ``row``. The read starts a '1' at the end of the read
    word line farthest from its driver and a '0' at the other end -- the two columns kept at the
    ends of the array -- and the write stores a '1' at the far end of the write word line and a
    '0' at the near end. Row 1 (the default) is driven from the right, so its '1' is read at
    column 0 and its '0' at the last column. ``one_level`` scales the supply to the decayed '1'
    the read starts from (0.7 is the weakest '1' the retention target allows); ``zero_level`` is
    the level a '0' has been written to (volts) -- by default the level that die's own write left.
    """
    if samples < 1:
        raise ValueError("need at least one sample")
    if len(model.kept) < 2:
        raise ValueError("Monte Carlo reads two columns at once: keep the two end columns")
    settings = settings or BenchSettings()
    if one_level is None:
        from gc_compiler.rtl import _measured

        one_level = _measured(tech).decayed_one_fraction
    workdir = Path(workdir)
    jobs = jobs or min(os.cpu_count() or 1, 8)
    result = MonteCarlo(corner, row, one_level, zero_level, settings.strobe_factor)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                _sample, tech, model, corner, seed + i, row, one_level, zero_level,
                workdir / f"s{seed + i}", settings,
            )
            for i in range(samples)
        ]
        result.samples = [f.result() for f in futures]
    return result


def table(runs: list[MonteCarlo]) -> str:
    """A markdown summary, one line per corner (times in ns)."""

    def fmt(spread: Spread | None, scale: float = 1.0, digits: int = 2) -> str:
        if spread is None:
            return "-"
        return f"{spread.mean * scale:.{digits}f} ± {spread.std * scale:.{digits}f}"

    head = (
        "| corner | dies | access (ns) | dout/rdy | 3σ strobe | strobe at | '1' fails "
        "| '0' flips at (x rdy) | '0' fails | '0' level (V) |"
    )
    rule = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [head, rule]
    for mc in runs:
        margin = mc.margin(3.0)
        sigma = mc.sigma_to_strobe()
        lines.append(
            f"| {mc.corner.name} | {mc.n} | {fmt(mc.access, 1e9)} | {fmt(mc.tracking)} "
            f"| {'-' if margin is None else f'x{margin:.2f}'} "
            f"| {'-' if sigma is None else f'{sigma:.1f}σ'} "
            f"| {mc.one_failures}/{mc.n} | {fmt(mc.flip)} | {mc.zero_failures}/{mc.n} "
            f"| {fmt(mc.sn_zero)} |"
        )
    return "\n".join(lines)
