"""ngspice testbenches for the gain cell.

The interesting property of a gain cell is that reads are non-destructive but the storage
node leaks, so the natural demonstration is a single transient that writes once and then
reads periodically while the stored level decays.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gc_compiler.config import BitcellConfig
from gc_compiler.netlist.spice import bitcell_subckt
from gc_compiler.tech.rules import Technology


def default_model_lib(pdk_root: str | os.PathLike[str] | None = None, tech: Technology | None = None) -> Path:
    """The technology's SPICE model library (:meth:`~gc_compiler.tech.rules.Technology.model_library`)."""
    if tech is None:
        from gc_compiler.tech import default_technology

        tech = default_technology()
    return Path(tech.model_library(pdk_root))


@dataclass(frozen=True)
class RetentionTestbench:
    """Write once, then read periodically while the storage node leaks away."""

    data: int = 1
    vdd: float | None = None
    """Supply (default: the technology's nominal)."""
    wwl_low: float = -0.4
    """Write word-line low level; a negative level writes a full '0' through a PMOS."""
    hold_time: float = 400e-6
    reads: int = 8
    """Number of periodic reads during the hold; 0 holds undisturbed (pure leakage)."""
    read_width: float = 5e-9
    write_width: float = 50e-9
    rbl_cap: float = 50e-15
    """Read bit-line capacitance contributed by the rest of the column."""
    precharge_width: float = 1.0
    """Precharge device width."""
    temperature: float = 85.0
    corner: str | None = None
    """Model-library corner (default: the technology's typical one)."""
    step: float = 20e-9

    @property
    def read_period(self) -> float:
        return self.hold_time / self.reads if self.reads else self.hold_time


def retention_netlist(
    tech: Technology,
    cfg: BitcellConfig,
    tb: RetentionTestbench,
    model_lib: Path | None = None,
) -> str:
    # The full sky130.lib.spice takes ~40 s to parse (the resistor/capacitor corners and the
    # several-thousand-bin special-cell library, neither used here); Technology.model_header()
    # loads only the MOSFET corner file, in ~0.2 s, for identical waveforms. ``model_lib``
    # overrides this with a direct ``.lib`` include, for a corner file this technology adapter
    # does not know how to find on its own.
    corner = tb.corner or tech.model_corners[0]
    vdd = tb.vdd if tb.vdd is not None else tech.electrical.vdd_nominal
    model_lines = tech.model_header(corner) if model_lib is None else [f'.lib "{model_lib}" {corner}']
    write_start = 100e-9
    write_end = write_start + tb.write_width
    data_level = vdd if tb.data else 0.0

    if tb.reads:
        first_read = write_end + tb.read_period / 2
        stop = first_read + tb.reads * tb.read_period
        read_sources = [
            f"vrwl rwl 0 pulse(0 {vdd} {first_read:g} 100p 100p {tb.read_width:g} "
            f"{tb.read_period:g})",
            f"vpch pch 0 pulse(0 {vdd} {first_read - 1e-9:g} 100p 100p "
            f"{tb.read_width + 2e-9:g} {tb.read_period:g})",
        ]
    else:
        # Undisturbed hold: every read couples the read node back onto sn through the
        # gate-drain capacitance of mr, which would show up as extra 'leakage'.
        stop = write_end + tb.hold_time
        read_sources = ["vrwl rwl 0 0", "vpch pch 0 0"]

    lines = [
        "* 3T gain-cell retention and read testbench",
        *model_lines,
        "",
        bitcell_subckt(tech, cfg),
        "",
        "xcell wwl wbl rwl rbl vss vdd gc3t",
        "vvss vss 0 0",
        f"vvdd vdd 0 {vdd}",
        f"crbl rbl 0 {tb.rbl_cap}",
        "* precharge device: rbl is pulled high whenever pch is low",
        tech.spice_device("pre", "vdd", "pch", "rbl", "vdd", cfg.write_kind, tb.precharge_width, cfg.write_length),
        "",
        "* write pulse: wwl goes low for a PMOS write port",
        f"vwwl wwl 0 pwl(0 {vdd} {write_start:g} {vdd} "
        f"{write_start + 1e-10:g} {tb.wwl_low:g} {write_end:g} {tb.wwl_low:g} "
        f"{write_end + 1e-10:g} {vdd})",
        f"vwbl wbl 0 pwl(0 {data_level:g} {write_end:g} {data_level:g} "
        f"{write_end + 1e-9:g} 0)",
        *read_sources,
        "",
        f".temp {tb.temperature:g}",
        ".options savecurrents",
        f".tran {tb.step:g} {stop:g}",
        ".control",
        "run",
        "wrdata sim.csv v(xcell.sn) v(rbl) v(rwl) v(wwl)",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


@dataclass
class SimResult:
    time: np.ndarray
    signals: dict[str, np.ndarray]
    workdir: Path

    def __getitem__(self, key: str) -> np.ndarray:
        return self.signals[key]

    def read_levels(self, rwl_threshold: float | None = None) -> list[tuple[float, float]]:
        """(time, rbl level) at the end of every read pulse (``rwl`` above ``rwl_threshold``,
        by default half its swing)."""
        rwl, rbl = self.signals["rwl"], self.signals["rbl"]
        active = rwl > (rwl_threshold if rwl_threshold is not None else rwl.max() / 2)
        edges = np.flatnonzero(active[:-1] & ~active[1:])
        return [(float(self.time[i]), float(rbl[i])) for i in edges]

    def retention_time(
        self, threshold: float, rising: bool = True, after: float = 1e-6
    ) -> float | None:
        """When the storage node crosses ``threshold`` after ``after``, interpolated.

        ``rising`` selects the stored '0', which well leakage pulls up; use
        ``rising=False`` for a stored '1'. ``None`` if it never crosses. If it is already past
        ``threshold`` by ``after`` (a write that lands on the wrong side of it, say), returns
        ``after`` itself rather than extrapolating backward through the excluded region before
        it (which for a level that only touches ``threshold`` right at the edge of that region
        used to give an arbitrary, sometimes negative, answer).
        """
        sn = self.signals["sn"]
        mask = self.time > after
        if not mask.any():
            return None
        start = int(np.flatnonzero(mask)[0])
        beyond = (sn > threshold) if rising else (sn < threshold)
        if beyond[start]:
            return float(after)
        tail = np.flatnonzero(beyond[start:])
        if not len(tail):
            return None
        i = start + int(tail[0])
        t0, t1 = self.time[i - 1], self.time[i]
        v0, v1 = sn[i - 1], sn[i]
        if v1 == v0:
            return float(t1)
        return float(t0 + (threshold - v0) * (t1 - t0) / (v1 - v0))


def run_retention(
    tech: Technology,
    cfg: BitcellConfig | None = None,
    tb: RetentionTestbench | None = None,
    workdir: str | Path = "build/sim",
    model_lib: Path | None = None,
) -> SimResult:
    """Write the testbench, run ngspice and return the parsed waveforms."""
    cfg = cfg or BitcellConfig.for_tech(tech)
    tb = tb or RetentionTestbench()
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        raise RuntimeError("ngspice is not on PATH (enter the nix shell first)")

    workdir = Path(workdir) / f"data{tb.data}"
    workdir.mkdir(parents=True, exist_ok=True)
    deck = workdir / "tb.spice"
    deck.write_text(retention_netlist(tech, cfg, tb, model_lib))

    process = subprocess.run(
        [ngspice, "-b", deck.name],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output = workdir / "sim.csv"
    if not output.exists():
        raise RuntimeError(f"ngspice produced no output:\n{process.stdout[-3000:]}")

    raw = np.loadtxt(output)
    names = ["sn", "rbl", "rwl", "wwl"]
    signals = {name: raw[:, 2 * i + 1] for i, name in enumerate(names)}
    return SimResult(time=raw[:, 0], signals=signals, workdir=workdir)


@dataclass(frozen=True)
class RefreshResult:
    """How long a written '0' takes to drift up to a read-tolerance threshold: the actual
    refresh interval, not a proxy (a fixed drift amount says whether one option leaks faster
    than another; this says how long the data is actually good for)."""

    settled_v: float
    """The storage node shortly after the write, before any drift."""
    threshold_v: float
    """The highest '0' that still reads correctly -- typically ``verify.timing.zero_level_limit``
    on the extracted macro, called with a generous bound instead of the written level (which is
    what makes that function's own use inside ``characterise()`` degenerate once a write already
    passes: it only ever bisects downward from the level actually written)."""
    refresh_time_s: float | None
    """Seconds until a fresh write drifts up to ``threshold_v``; ``0.0`` if it starts there or
    past it already; ``None`` if it never got there within ``hold_time_s`` (retention is at
    least that good, not "infinite")."""
    hold_time_s: float
    """The longest hold this ended up simulating (``hold0`` after doubling, if it needed to)."""
    attempts: int

    @property
    def margin_v(self) -> float:
        """Positive: the write already clears the threshold. Negative: it starts over it, and
        ``refresh_time_s`` is ``0.0`` -- there is no interval during which the data is good."""
        return self.threshold_v - self.settled_v


def refresh_interval(
    tech: Technology,
    cfg: BitcellConfig,
    threshold: float,
    vdd: float | None = None,
    temperature: float = 25.0,
    corner: str | None = None,
    wwl_low: float = 0.0,
    workdir: str | Path = "build/sim/refresh",
    hold0: float = 1e-3,
    max_doublings: int = 8,
) -> RefreshResult:
    """Write a '0' (``wwl_low=0``: no negative rail, the real macro's actual swing, unless
    ``cfg`` is meant for one) and time how long it takes to drift up to ``threshold`` -- see
    :class:`RefreshResult`. Retries with the hold time doubled (up to ``max_doublings`` times)
    if the first attempt is not long enough to see the crossing.
    """
    workdir = Path(workdir)
    hold = hold0
    settled = 0.0
    for attempt in range(max_doublings):
        tb = RetentionTestbench(
            data=0, vdd=vdd, temperature=temperature, corner=corner,
            wwl_low=wwl_low, reads=0, hold_time=hold,
        )
        result = run_retention(tech, cfg, tb, workdir=workdir / f"a{attempt}")
        write_end = 100e-9 + tb.write_width
        settle_at = write_end + tb.write_width
        idx = np.searchsorted(result.time, settle_at)
        settled = float(result["sn"][min(idx, len(result["sn"]) - 1)])
        fail_time = result.retention_time(threshold, rising=True, after=settle_at)
        if fail_time is not None:
            refresh_time = max(0.0, fail_time - settle_at)
            return RefreshResult(settled, threshold, refresh_time, hold, attempt + 1)
        if attempt + 1 < max_doublings:
            hold *= 4
    return RefreshResult(settled, threshold, None, hold, max_doublings)


def decoder_op_netlist(
    tech: Technology,
    cfg,
    logic,
    row: int,
    we: bool,
    re: bool,
    vdd: float,
    model_lib: Path | None = None,
) -> str:
    """A standalone decoder instance with fixed address/control levels, for a single .op."""
    from gc_compiler.netlist.decoder import address_bits, gate_subckts, row_decoder_pins, row_decoder_subckt
    from gc_compiler.netlist.spice import inverter_subckt

    bits = address_bits(cfg.rows)
    lines = [
        "* Static row decoder operating-point testbench",
        *(tech.model_header(tech.model_corners[0]) if model_lib is None else [f'.lib "{model_lib}" {tech.model_corners[0]}']),
        inverter_subckt(tech, logic),
        *gate_subckts(tech, logic),
        row_decoder_subckt(tech, cfg, logic),
        f"Xdec {' '.join(row_decoder_pins(cfg))} gc_row_decoder",
        f"vdd_src vdd 0 {vdd:g}",
        "vss_src vss 0 0",
        f"vwe we 0 {vdd if we else 0:g}",
        f"vre re 0 {vdd if re else 0:g}",
    ]
    for i in range(bits):
        level = vdd if (row >> i) & 1 else 0.0
        lines.append(f"vaddr{i} addr_{i} 0 {level:g}")
    outputs = [f"v(wwl_in_{r})" for r in range(cfg.rows)] + [f"v(rwl_in_{r})" for r in range(cfg.rows)]
    lines += [
        ".op",
        ".control",
        "run",
        f"wrdata out.csv {' '.join(outputs)}",
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def run_decoder_op(
    tech: Technology,
    cfg,
    logic=None,
    row: int = 0,
    we: bool = True,
    re: bool = False,
    workdir: str | Path = "build/sim/decoder",
    model_lib: Path | None = None,
    vdd: float | None = None,
) -> dict[str, float]:
    """Run one decoder operating point and return every ``wwl_in``/``rwl_in`` level (at the
    technology's nominal supply unless ``vdd`` says otherwise)."""
    from gc_compiler.periphery.logic import LogicConfig

    logic = logic or LogicConfig.for_tech(tech)
    lib = model_lib
    vdd = vdd if vdd is not None else tech.electrical.vdd_nominal
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        raise RuntimeError("ngspice is not on PATH (enter the nix shell first)")

    workdir = Path(workdir) / f"row{row}_we{int(we)}_re{int(re)}"
    workdir.mkdir(parents=True, exist_ok=True)
    deck = workdir / "tb.spice"
    deck.write_text(decoder_op_netlist(tech, cfg, logic, row, we, re, vdd, lib))

    process = subprocess.run(
        [ngspice, "-b", deck.name], cwd=workdir, capture_output=True, text=True, timeout=300
    )
    output = workdir / "out.csv"
    if not output.exists():
        raise RuntimeError(f"ngspice produced no output:\n{process.stdout[-3000:]}")

    raw = np.loadtxt(output)
    values = raw[1::2] if raw.ndim == 1 else raw[0, 1::2]
    names = [f"wwl_in_{r}" for r in range(cfg.rows)] + [f"rwl_in_{r}" for r in range(cfg.rows)]
    return dict(zip(names, values))
