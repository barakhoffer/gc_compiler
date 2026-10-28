"""Post-layout timing characterisation of a compiled macro.

The flow, end to end:

1. **Extract** the macro's layout with parasitics (:mod:`gc_compiler.verify.parasitics`): every
   device and every wiring capacitance, one flat netlist, nets named after the layout's labels.
2. **Reduce** it (:mod:`gc_compiler.verify.flatnet`): keep the row decoders, drivers and column
   circuits, the cells of a few columns (both ends of the array and the replica column), and
   stand in for every other column with a gate load on the word lines -- which is all it does
   to a read or write elsewhere. Each word line becomes a distributed RC ladder (extraction gives
   its capacitance; its poly resistance is analytic), driven from the end its driver is at.
3. **Simulate** reads and writes of chosen cells with ngspice at each process/voltage/
   temperature corner, and measure where the time goes.

What comes out (per corner): the write path (``we`` -> decoder -> driver -> word line -> storage
node), the read path (``re`` -> ... -> read bit line -> read buffer -> ``dout``) split into its
stages, the level a written '0' and '1' settle at, and -- for a macro with a replica column --
when its ``rdy`` output fires against when the slowest legitimate read is valid, which is what
decides whether ``rdy`` can be used as a self-timed data-valid strobe.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gc_compiler.array.tiler import array_geometry
from gc_compiler.config import ArrayConfig
from gc_compiler.netlist.decoder import address_bits
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.tech.electrical import Corner


def _measured(tech=None):
    """What was measured on ``tech`` (the default technology): the benches' defaults."""
    from gc_compiler.rtl import _measured as measured

    return measured(tech)
from gc_compiler.tech.rules import DeviceKind, Technology
from gc_compiler.verify.flatnet import FlatNetlist
from gc_compiler.verify.parasitics import extract_flat
from gc_compiler.verify.sim import RefreshResult

if TYPE_CHECKING:
    from gc_compiler.verify.montecarlo import MonteCarlo


def standard_corners(tech: Technology | None = None) -> tuple[Corner, ...]:
    """The technology's corners (``tech.electrical.corners``): for sky130 the usual five --
    typical, slow (low supply, hot), fast (high supply, cold), and the two skewed ones."""
    if tech is None:
        from gc_compiler.tech import default_technology

        tech = default_technology()
    return tech.electrical.corners


@dataclass
class TimingModel:
    """A reduced, RC-annotated netlist of a macro, ready to be wrapped in a testbench."""

    netlist: Path
    rows: int
    cols: int
    replica: bool
    kept: tuple[str, ...]
    """The data columns kept as real cells."""
    prefix: str
    sn: dict[tuple[int, str], str]
    wordline: dict[tuple[str, int, str], str]
    decoder: dict[tuple[str, int], str]
    vnw: tuple[str, ...]
    read_length: float
    write_length: float
    vneg_net: str | None = None
    """The net of the negative supply, if the macro has negative-rail write drivers."""
    vneg: float | None = None
    """The level it is supplied at (volts)."""
    bank: int | None = None
    """For one bank of a banked macro: which one (its pins are ``din_{bank}_{col}``,
    ``rdy_{bank}``, and the address has the bank bit on top)."""
    self_timed: bool = False
    """The macro ends its reads itself (``ArrayConfig.self_timed_read``): ``dout`` holds from the
    replica's ``rdy`` on, however long ``re`` stays up."""
    refresh_in_macro: bool = False
    """The macro writes a refresh's data back itself (``ArrayConfig.refresh_in_macro``)."""

    def column_probe(self, kind: str, row: int, col: str) -> str:
        return self.wordline[(kind, row, col)]

    def pin(self, name: str) -> str:
        """A macro pin as this model's macro names it: ``din_3`` -> ``din_{bank}_3`` and
        ``rdy`` -> ``rdy_{bank}`` in one bank of a banked macro, unchanged otherwise."""
        if self.bank is None:
            return name
        signal, _, index = name.rpartition("_")
        return f"{signal}_{self.bank}_{index}" if signal and index.isdigit() else f"{name}_{self.bank}"


def driven_from_left(kind: str, row: int) -> bool:
    """Whether a word line's driver is at the left end of the array. A mirrored row pair's
    drivers alternate sides: rwl_0/wwl_1 on the left, wwl_0/rwl_1 on the right."""
    return (row % 2 == 0) if kind == "rwl" else (row % 2 == 1)


def _device_params(tech: Technology, kind, width: float, length: float, flavor=None) -> tuple[str, str]:
    """``(model, parameters)`` of a device as the technology writes one (its units: sky130's um,
    IHP's metres with a ``u`` suffix)."""
    tokens = tech.spice_device("x", "d", "g", "s", "b", kind, width, length, flavor).split()
    return tokens[5], " ".join(tokens[6:])


def build_model(
    tech: Technology,
    cfg: ArrayConfig,
    gds: str | Path,
    cell: str,
    workdir: str | Path,
    keep: tuple[int, ...] | None = None,
    group: int = 4,
    cthresh: float = 0.02,
    bank: int | None = None,
    flat: str | Path | None = None,
) -> TimingModel:
    """Extract ``gds`` and build the reduced timing netlist (see the module docstring).

    ``keep`` lists the data columns kept as real cells (default: both ends of the array, which
    between them are the nearest and farthest from every word-line driver).

    ``bank`` models one bank of a banked macro: that bank's array is reduced as above and the
    other bank's cells and column circuits are dropped (they share only the decoder with it;
    its row gates stay, loading the shared predecode as they really do). ``flat`` reuses an
    extraction already made (both banks of one macro come from the same one).
    """
    workdir = Path(workdir)
    keep_cols = tuple(dict.fromkeys(str(c) for c in (keep if keep is not None else (0, cfg.cols - 1))))
    flat = Path(flat) if flat is not None else extract_flat(gds, cell, workdir, cthresh, tech=tech)
    workdir.mkdir(parents=True, exist_ok=True)
    net = FlatNetlist.parse(flat.read_text(), tech)
    tag = "" if bank is None else f"_{bank}"

    decoder = {}
    for kind in ("rwl", "wwl"):
        for row in range(cfg.rows):
            try:
                decoder[(kind, row)] = net.net_matching(rf"/{kind}_in{tag}_{row}$")
            except ValueError as error:
                raise RuntimeError(
                    "timing characterisation needs the decoded macro (decoded=True): " + str(error)
                ) from error

    arrays = net.array_prefixes()
    vnw_all = {c.mw.b for prefix in arrays for c in _cells_under(net, prefix).values()}
    if bank is not None:
        # this bank's array is the one its row-0 read word-line driver drives
        dec = decoder[("rwl", 0)]
        out = next(t for d in net.devices if d.g == dec for t in (d.d, d.s) if re.search(r"(?:^|/)rwl_\d+$", t))
        mine = out[: out.rindex("rwl_")]
        for prefix in arrays:
            if prefix != mine:
                net.drop_array(prefix)
        net.array_prefix = mine
        if cfg.self_timed_read:
            # the other bank's replica went with its column circuits; its rdy still enters the
            # decoder, and in this bank's reads it never fires: hold it low
            idle = net.net_matching(rf"(?:^|/)rdy_{1 - bank}$")
            net.extra.append(f"Rrdy_idle {idle} vss 1")
    elif len(arrays) != 1:
        raise RuntimeError(f"found {len(arrays)} arrays in the extracted netlist: pass bank= for a banked macro")
    prefix = net.wordline_prefix()
    cells = net.find_cells()
    expected = cfg.rows * cfg.total_cols
    if len(cells) != expected:
        raise RuntimeError(f"found {len(cells)} bitcells in the extracted netlist, expected {expected}")

    sn = {key: c.sn for key, c in cells.items() if key[1] in keep_cols or key[1] == "rep"}
    vnw = tuple(sorted(vnw_all))
    loads = net.reduce(set(keep_cols), group)

    geo = array_geometry(tech, cfg)
    half = geo.cell.pitch_x / 2
    x_of: dict[str, float] = {c: geo.col_x[int(c)] + half for c in keep_cols}
    for first, members in loads.items():
        x_of[first] = float(np.mean([geo.col_x[int(c)] + half for c in members]))
    if cfg.replica_column:
        x_of["rep"] = geo.col_x[geo.replica_col] + half
    columns = sorted(x_of, key=lambda c: x_of[c])

    rsh = tech.poly_sheet_resistance
    left_driven = {(kind, row): driven_from_left(kind, row) for kind in ("rwl", "wwl") for row in range(cfg.rows)}
    nodes = net.add_wordline_rc(
        columns=columns,
        x_of=x_of,
        left_driven=left_driven,
        resistance_per_um={"rwl": rsh / cfg.bitcell.read_length, "wwl": rsh / cfg.bitcell.write_length},
        loads=loads,
        gate_load={
            "n": _device_params(tech, cfg.bitcell.read_kind, cfg.bitcell.read_width, cfg.bitcell.read_length, cfg.bitcell.read_flavor),
            "p": _device_params(tech, cfg.bitcell.write_kind, cfg.bitcell.write_width, cfg.bitcell.write_length, cfg.bitcell.write_flavor),
        },
        edges=(0.0, geo.width),
        **_strap_ladder(tech, geo),
    )
    net.drop_static_caps({"vss", "vdd", *vnw})
    out = workdir / "timing.spice"
    out.write_text(net.emit())
    vneg_net = net.net_matching(r"(?:^|/)vneg$") if cfg.negative_write_wl else None
    return TimingModel(
        netlist=out, rows=cfg.rows, cols=cfg.cols, replica=cfg.replica_column, kept=keep_cols,
        prefix=prefix, sn=sn, wordline=nodes, decoder=decoder, vnw=vnw,
        read_length=cfg.bitcell.read_length, write_length=cfg.bitcell.write_length,
        vneg_net=vneg_net, vneg=cfg.write_wl_low if cfg.negative_write_wl else None, bank=bank,
        self_timed=cfg.self_timed_read,
        refresh_in_macro=cfg.refresh_in_macro,
    )


def _strap_ladder(tech: Technology, geo) -> dict:
    """The word-line ladder's strap arguments: where each word line meets its metal3 strap (the
    strap columns' middles) and the strap's resistance, or nothing for an unstrapped array."""
    if not geo.wordline_straps:
        return {}
    from gc_compiler.array.strap_column import strap_column_geometry
    from gc_compiler.periphery.wl_driver import strap_width

    half = strap_column_geometry(tech).width / 2
    return {
        "taps": tuple(x + half for x in geo.strap_col_x),
        "strap_per_um": tech.m3_sheet_resistance / strap_width(tech),
        "tap_resistance": tech.strap_contact_resistance,
    }


def _cells_under(net: FlatNetlist, prefix: str):
    selected, net.array_prefix = net.array_prefix, prefix
    try:
        return net.find_cells()
    finally:
        net.array_prefix = selected


# --- waveforms and measurements ---------------------------------------------------------


def crossing(t: np.ndarray, y: np.ndarray, level: float, rising: bool, after: float = 0.0) -> float | None:
    """First time after ``after`` that ``y`` crosses ``level`` in the given direction (linearly
    interpolated between samples), or ``None``."""
    passed = (y >= level) if rising else (y <= level)
    start = int(np.searchsorted(t, after))
    # the crossing is where the condition first becomes true after having been false
    for i in range(max(start, 1), len(t)):
        if passed[i] and not passed[i - 1]:
            y0, y1 = y[i - 1], y[i]
            return float(t[i - 1] + (level - y0) * (t[i] - t[i - 1]) / (y1 - y0))
    return None


@dataclass
class Waveforms:
    t: np.ndarray
    v: dict[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.v[name]


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


@dataclass
class ReadTiming:
    """One read of one cell, times in seconds from the ``re`` edge unless noted."""

    row: int
    col: str
    data: int
    sn_level: float
    """Storage-node voltage the cell started with."""
    t_decoder: float | None
    """``re`` to the decoder's output for the row."""
    t_driver: float | None
    """Decoder output to the word line at the cell (driver plus word-line RC)."""
    t_bitline: float | None
    """Word line at the cell to the read bit line crossing vdd/2."""
    t_dout: float | None
    """``re`` to ``dout`` crossing vdd/2."""
    t_rdy: float | None
    """``re`` to the replica detector's ``rdy`` crossing vdd/2 (macros with a replica column)."""
    rbl_min: float
    dout_max: float
    t_precharge: float | None = None
    """From ``pch`` falling to the bit line back at 90% of vdd."""
    correct: bool = True
    """Whether ``dout`` ended up at the stored data: a '1' pulls it high, a '0' leaves it low --
    at the strobe, or, for a self-timed macro, at the end of the ``re`` window (it must hold)."""
    t_cutoff: float | None = None
    """A self-timed read: ``re`` to the read word line at the cell falling again, by itself."""
    supply_charge: float | None = None
    """Charge drawn from ``vdd`` over the whole access, precharge back included (coulombs; the
    same for every cell of one access). See :func:`access_energy`."""
    supply_window: float | None = None
    """How long that window is (seconds), to take the leakage over it back out."""


@dataclass
class WriteTiming:
    row: int
    col: str
    data: int
    t_decoder: float | None
    """``we`` to the decoder's output for the row."""
    t_driver: float | None
    """Decoder output to the write word line at the cell."""
    t_write: float | None
    """Write word line at the cell to the storage node at 90% of its settled swing."""
    sn_start: float
    sn_final: float
    vneg_peak_a: float | None = None
    """Peak current drawn from the ``vneg`` supply during the write (negative-rail macros)."""
    supply_charge: float | None = None
    """Charge drawn from ``vdd`` over the write (coulombs; see :func:`access_energy`)."""
    supply_window: float | None = None


@dataclass
class SupplyCurrents:
    """What a macro at rest draws from its supplies, in amperes (magnitudes)."""

    vdd_a: float
    vneg_a: float | None = None


# --- decks ---------------------------------------------------------------------------------


def _pwl(name: str, node: str, points: list[tuple[float, float]]) -> str:
    return f"v{name} {node} 0 pwl(" + " ".join(f"{t:.6g} {v:.6g}" for t, v in points) + ")"


@dataclass(frozen=True)
class BenchSettings:
    """Timing of the stimulus, in seconds."""

    settle: float = 8e-9
    """Static settling (precharge, decoder) before the operation starts."""
    slew: float = 100e-12
    window: float = 10e-9
    """How long the enable is held."""
    recovery: float = 8e-9
    """Simulated after the enable falls (precharge recovery)."""
    step: float = 40e-12
    sn_idle: float = field(default_factory=lambda: _measured().idle_storage_node_v)
    """Storage-node level of every cell that is not the one under test."""
    strobe_factor: float = field(default_factory=lambda: _measured().strobe_factor)
    """Where the data is sampled, as a multiple of the replica's ``rdy`` time: a '1' must have
    pulled ``dout`` high by then and a '0' must not yet have. Without a replica column, the
    end of the enable window."""
    vneg: float | None = None
    """Override the level of a negative-rail macro's ``vneg`` supply (default: the level it was
    compiled for, ``ArrayConfig.write_wl_low``)."""
    uic: bool = False
    """Start the transient from the ``.ic`` values with every other node at 0 V instead of from an
    operating point (faster, but see :func:`_deck`: it can corrupt the initial state)."""
    seed: int | None = None
    """``None`` simulates the nominal devices. An integer turns local mismatch on and seeds
    its draws: the same seed on the same netlist is the same set of devices (so a read and a
    write with one seed are of the same die), and every device instance draws independently."""


def _deck(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    sources: list[str],
    initial: dict[str, float],
    probes: dict[str, str],
    stop: float,
    settings: BenchSettings,
    csv: str,
) -> str:
    lines = [
        f"* timing bench, corner {corner.name}",
        *tech.model_header(corner.process, mismatch=settings.seed is not None),
        f".include {model.netlist.resolve()}",
        f"vvdd vdd 0 {corner.vdd:g}",
        "vvss vss 0 0",
        # the charge vdd delivers, integrated by the simulator at its own timesteps: vvdd's
        # current mirrored into a 1 F capacitor (the printed waveform is far too coarse to
        # integrate switching spikes from)
        "fqvdd 0 qvdd vvdd 1",
        "cqvdd qvdd 0 1",
        "rqvdd qvdd 0 1e15",
        ".ic v(qvdd)=0",  # else the operating point parks it at the leakage current x 1e15 ohm
    ]
    if model.vneg_net is not None:
        lines.append(f"vvneg {model.vneg_net} 0 {settings.vneg if settings.vneg is not None else model.vneg:g}")
    # an exact short (a milliohm tie leaves IHP's PSP models an ill-conditioned operating point)
    lines += [f"vvnw{i} {net} vdd 0" for i, net in enumerate(model.vnw)]
    lines += sources
    lines += [f".ic v({node})={level:g}" for node, level in initial.items()]
    lines += [
        f".temp {corner.temp:g}",
        *([f".option seed={settings.seed}"] if settings.seed is not None else []),
        ".option method=gear",
        # no ``uic``: start from an operating point with the .ic nodes held there. From all-zero
        # node voltages, a negative-rail driver's level shifter latch powers up in contention and
        # pulls its write word line below vss for the first ~0.5 ns -- enough to empty a stored
        # '1' through the write device before the bench has even started.
        f".tran {settings.step:g} {stop:g}" + (" uic" if settings.uic else ""),
        ".control",
        "run",
        f"wrdata {csv} " + " ".join(node if node.startswith("i(") else f"v({node})" for node in probes.values()),
        "quit",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def _address_sources(model: TimingModel, row: int, vdd: float) -> list[str]:
    """The address pins for ``row`` (and, in a banked macro, the bank bit above them)."""
    bits = address_bits(model.rows)
    lines = [f"vaddr{i} addr_{i} 0 {vdd if (row >> i) & 1 else 0:g}" for i in range(bits)]
    if model.bank is not None:
        lines.append(f"vaddr{bits} addr_{bits} 0 {vdd if model.bank else 0:g}")
    return lines


def _din_sources(model: TimingModel, level) -> list[str]:
    """One source per kept column's ``din``; ``level(col)`` gives its voltage."""
    return [f"vdin_{c} {model.pin(f'din_{c}')} 0 {level(c):g}" for c in _data_columns(model)]


def _run(deck: str, workdir: Path, probes: dict[str, str], csv: str = "wave.csv") -> Waveforms:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        raise RuntimeError("ngspice is not on PATH (enter the nix shell first)")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "bench.spice").write_text(deck)
    (workdir / csv).unlink(missing_ok=True)
    process = subprocess.run(
        [ngspice, "-b", "bench.spice"], cwd=workdir, capture_output=True, text=True, timeout=1800
    )
    if not (workdir / csv).exists():
        raise RuntimeError(f"ngspice produced no output:\n{process.stdout[-2500:]}")
    log = process.stdout + process.stderr
    if "singular matrix" in log or "stepping failed" in log:
        # the operating point fell back to something that no longer holds the .ic nodes (a
        # stored '1' quietly becomes whatever the solver settled on): refuse the result
        raise RuntimeError(f"ngspice could not solve the operating point in {workdir}:\n{log[-2500:]}")
    raw = np.loadtxt(workdir / csv)
    return Waveforms(raw[:, 0], {name: raw[:, 2 * i + 1] for i, name in enumerate(probes)})


def _data_columns(model: TimingModel) -> list[str]:
    return list(model.kept)


def _charge(wave: Waveforms, t_start: float, t_stop: float) -> tuple[float, float]:
    """Charge drawn from ``vdd`` between two instants (the deck's ``qvdd`` integrator, whose
    value in volts is the charge in coulombs) and the window's length."""
    q = np.interp([t_start, t_stop], wave.t, wave["qvdd"])
    return float(abs(q[1] - q[0])), float(t_stop - t_start)


@dataclass
class AccessEnergy:
    """Energy per access of a macro at one corner, leakage over the access taken out (joules).

    Measured on a model that keeps every column (``build_model(keep=range(cols))``): a reduced
    one drops most bit lines and column circuits, which is most of a read's energy.
    """

    corner: str
    read1_j: float
    """Reading a row whose every cell holds a '1' (every read bit line discharges): the worst read."""
    read0_j: float
    """Reading a row of '0's at 0 V (no bit line discharges): the best read."""
    write_j: float
    """Writing a '1' into every cell of a row that held '0's, every write bit line swinging."""
    idle_w: float
    """Leakage power at rest."""
    bits_per_access: int
    read_window_s: float | None = None
    """How long ``re`` was held for the reads: until the strobe (``strobe_factor`` times the
    replica's ``rdy``), as a self-timed macro would, when there is a replica."""
    refresh_j: float | None = None
    """One refresh access (:func:`simulate_refresh`) of a row whose every cell holds a '1' decayed
    to 0.7 vdd: every read bit line discharges and every write bit line swings (the worst row)."""

    def refresh_power(self, rows: int, interval_s: float, banks: int = 1) -> float:
        """Power to refresh every row once per ``interval_s``, plus leakage: the standing cost of
        keeping the data. Each row costs one refresh access (``refresh_j``), or a worst-case read
        plus a write where that was not measured."""
        per_row = self.refresh_j if self.refresh_j is not None else self.read1_j + self.write_j
        return banks * rows * per_row / interval_s + self.idle_w

    def to_dict(self) -> dict:
        return asdict(self)


def access_energy(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    workdir: str | Path,
    settings: BenchSettings | None = None,
    row: int = 0,
) -> AccessEnergy:
    """Read and write energy per access, and leakage, at ``corner`` (see :class:`AccessEnergy`).

    A read's energy is mostly the bit lines', and how far they fall depends on how long the word
    line is up. With a replica column the reads end where a self-timed macro would end them, at
    the strobe (``settings.strobe_factor`` x ``rdy``, measured by a first read); without one they
    use the bench's fixed window, which lets every bit line of a fast cell fall all the way.
    """
    settings = settings or BenchSettings()
    workdir = Path(workdir)
    vdd = corner.vdd
    idle = simulate_idle(tech, model, corner, workdir / "idle", settings)
    columns = _data_columns(model)
    window = None
    if model.replica:
        probe = simulate_reads(tech, model, corner, row, {columns[-1]: (1, vdd)}, workdir / "probe", settings)
        t_rdy = probe[columns[-1]].t_rdy
        if t_rdy is not None:
            window = settings.strobe_factor * t_rdy + settings.slew
    read_settings = replace(settings, window=window) if window is not None else settings

    def net(charge: float, window: float) -> float:
        return vdd * charge - vdd * idle.vdd_a * window

    reads = {}
    for data in (1, 0):
        level = vdd if data else 0.0  # a solid '0': no bit line moves (the best case)
        timing = simulate_reads(
            tech, model, corner, row, {c: (data, level) for c in columns}, workdir / f"read{data}", read_settings
        )
        first = timing[columns[0]]
        reads[data] = net(first.supply_charge, first.supply_window)
    written = simulate_writes(
        tech, model, corner, row, {c: 1 for c in columns}, workdir / "write", settings, toggle=True
    )
    first = written[columns[0]]
    refreshed = simulate_refresh(
        tech, model, corner, row, {c: (1, 0.7 * vdd) for c in columns}, workdir / "refresh", settings,
        read_time=(settings.strobe_factor * t_rdy + settings.slew) if model.replica and t_rdy is not None else None,
    )
    ref = refreshed[columns[0]]
    return AccessEnergy(
        corner=corner.name, read1_j=reads[1], read0_j=reads[0],
        write_j=net(first.supply_charge, first.supply_window),
        idle_w=vdd * idle.vdd_a, bits_per_access=len(columns), read_window_s=window,
        refresh_j=net(ref.supply_charge, ref.supply_window),
    )


def simulate_read(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    row: int,
    col: str,
    data: int,
    sn_level: float,
    workdir: str | Path,
    settings: BenchSettings | None = None,
) -> ReadTiming:
    """Read the cell at ``(row, col)``, whose storage node starts at ``sn_level``."""
    reads = simulate_reads(tech, model, corner, row, {col: (data, sn_level)}, workdir, settings)
    return reads[col]


def simulate_reads(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    row: int,
    cells: dict[str, tuple[int, float]],
    workdir: str | Path,
    settings: BenchSettings | None = None,
) -> dict[str, ReadTiming]:
    """Read row ``row`` once and measure several of its columns at the same time.

    ``cells`` maps a kept column to ``(data, storage-node level)``. Columns share nothing but
    the word line, so one access serves them all -- which is what makes it cheap to read a '1'
    at the far end of the word line and a '0' at the near end (each one's worst case) together.
    """
    settings = settings or BenchSettings()
    vdd = corner.vdd
    t0 = settings.settle
    t1 = t0 + settings.window
    s = settings.slew
    stop = t1 + settings.recovery

    probes = {"re": "re", "pch": "pch", "dec": model.decoder[("rwl", row)], "qvdd": "qvdd"}
    for col in cells:
        probes[f"wl_{col}"] = model.column_probe("rwl", row, col)
        probes[f"rbl_{col}"] = f"{model.prefix}rbl_{col}"
        probes[f"dout_{col}"] = model.pin(f"dout_{col}")
    if model.replica:
        probes["rdy"] = model.pin("rdy")
    edge = [(0, 0), (t0, 0), (t0 + s, vdd), (t1, vdd), (t1 + s, 0)]
    sources = [
        _pwl("re", "re", edge),
        _pwl("pch", "pch", edge),
        "vwe we 0 0",
        *_address_sources(model, row, vdd),
        # (the controller holds din high outside a write -- the write bit lines low, so they do not
        # leak into the stored '0's (rtl.controller) -- so with refresh_in_macro a read bit line's
        # fall raises its write bit line, NAND(1, rbl))
        *_din_sources(model, lambda c: vdd),
    ]
    initial = {
        node: (cells[key[1]][1] if key[0] == row and key[1] in cells else settings.sn_idle)
        for key, node in model.sn.items()
        if key[1] != "rep"
    }

    workdir = Path(workdir)
    wave = _run(_deck(tech, model, corner, sources, initial, probes, stop, settings, "wave.csv"), workdir, probes)
    t, half = wave.t, vdd / 2
    t_re = crossing(t, wave["re"], half, True)
    t_dec = crossing(t, wave["dec"], half, False, t_re or 0.0)  # active-low
    t_rdy = crossing(t, wave["rdy"], half, True, t_re or 0.0) if "rdy" in wave.v else None
    t_pch = crossing(t, wave["pch"], half, False, t1)
    window = (t > t0) & (t < t1)
    charge, span = _charge(wave, t0 - s, stop)
    out = {}
    for col, (data, sn_level) in cells.items():
        t_wl = crossing(t, wave[f"wl_{col}"], half, True, t_re or 0.0)
        t_rbl = crossing(t, wave[f"rbl_{col}"], half, False, t_re or 0.0)
        t_dout = crossing(t, wave[f"dout_{col}"], half, True, t_re or 0.0)
        t_back = crossing(t, wave[f"rbl_{col}"], 0.9 * vdd, True, t1)
        dout_max = float(wave[f"dout_{col}"][window].max())
        t_off = crossing(t, wave[f"wl_{col}"], half, False, t_wl) if t_wl is not None else None
        if model.self_timed:
            held = float(np.interp(t1, t, wave[f"dout_{col}"]))
            correct = (held > half) == bool(data)
        else:
            correct = _read_correct(data, t_re, t_dout, t_rdy, dout_max, half, settings)
        out[col] = ReadTiming(
            row=row, col=col, data=data, sn_level=sn_level,
            t_decoder=_delta(t_re, t_dec), t_driver=_delta(t_dec, t_wl), t_bitline=_delta(t_wl, t_rbl),
            t_dout=_delta(t_re, t_dout), t_rdy=_delta(t_re, t_rdy),
            rbl_min=float(wave[f"rbl_{col}"][window].min()), dout_max=dout_max,
            t_precharge=_delta(t_pch, t_back),
            correct=correct,
            t_cutoff=_delta(t_re, t_off) if t_off is not None and t_off < t1 else None,
            supply_charge=charge, supply_window=span,
        )
    return out


def _read_correct(
    data: int,
    t_re: float | None,
    t_dout: float | None,
    t_rdy: float | None,
    dout_max: float,
    half: float,
    settings: BenchSettings,
) -> bool:
    """Whether ``dout`` shows ``data`` at the sampling instant (see ``strobe_factor``)."""
    if t_rdy is not None and t_re is not None:
        strobe = t_re + settings.strobe_factor * (t_rdy - t_re)
        arrived = t_dout is not None and t_dout <= strobe
        return arrived if data else not arrived
    return (t_dout is not None) if data else dout_max < half


def simulate_write(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    row: int,
    col: str,
    data: int,
    workdir: str | Path,
    settings: BenchSettings | None = None,
) -> WriteTiming:
    """Write ``data`` into the cell at ``(row, col)``, which held the opposite level."""
    return simulate_writes(tech, model, corner, row, {col: data}, workdir, settings)[col]


def simulate_writes(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    row: int,
    cells: dict[str, int],
    workdir: str | Path,
    settings: BenchSettings | None = None,
    toggle: bool = False,
) -> dict[str, WriteTiming]:
    """Write row ``row`` once, storing ``cells[col]`` in each listed kept column (each cell held
    the opposite level before); the other columns write a '0' onto their idle level.

    ``toggle`` switches each listed column's ``din`` from the opposite data to the new at the
    ``we`` edge, so its bit line swings in the write (what a write's energy has to include);
    otherwise ``din`` is steady and the bit line already carries the data."""
    settings = settings or BenchSettings()
    vdd = corner.vdd
    t0 = settings.settle
    t1 = t0 + settings.window
    s = settings.slew
    stop = t1 + settings.recovery

    probes = {"we": "we", "dec": model.decoder[("wwl", row)], "qvdd": "qvdd"}
    for col in cells:
        probes[f"wl_{col}"] = model.column_probe("wwl", row, col)
        probes[f"sn_{col}"] = model.sn[(row, col)]
    if model.vneg_net is not None:
        probes["ineg"] = "i(vvneg)"
    edge = [(0, 0), (t0, 0), (t0 + s, vdd), (t1, vdd), (t1 + s, 0)]
    # the column's write driver inverts: din low writes a '1' (wbl high)
    sources = [
        _pwl("we", "we", edge),
        "vre re 0 0",
        "vpch pch 0 0",
        *_address_sources(model, row, vdd),
    ]
    for c in _data_columns(model):
        level = (0 if cells[c] else vdd) if c in cells else vdd  # din low writes a '1'
        if toggle and c in cells:
            sources.append(_pwl(f"din_{c}", model.pin(f"din_{c}"), [(0, vdd - level), (t0, vdd - level), (t0 + s, level)]))
        else:
            sources.append(f"vdin_{c} {model.pin(f'din_{c}')} 0 {level:g}")
    initial = {
        node: ((settings.sn_idle if cells[key[1]] else vdd) if key[0] == row and key[1] in cells else settings.sn_idle)
        for key, node in model.sn.items()
        if key[1] != "rep"
    }

    workdir = Path(workdir)
    wave = _run(_deck(tech, model, corner, sources, initial, probes, stop, settings, "wave.csv"), workdir, probes)
    t, half = wave.t, vdd / 2
    t_we = crossing(t, wave["we"], half, True)
    t_dec = crossing(t, wave["dec"], half, True, t_we or 0.0)
    window = (t > t1 - 1e-9) & (t <= t1)
    charge, span = _charge(wave, t0 - s, stop)
    out = {}
    for col, data in cells.items():
        t_wl = crossing(t, wave[f"wl_{col}"], half, False, t_we or 0.0)
        sn = wave[f"sn_{col}"]
        sn_final = float(sn[window].mean())
        sn_start = float(sn[t > t0 - 1e-9][0])
        swing = sn_final - sn_start
        t_sn = None
        if abs(swing) > 0.05:
            t_sn = crossing(t, sn, sn_start + 0.9 * swing, swing > 0, t_wl or 0.0)
        out[col] = WriteTiming(
            row=row, col=col, data=data, t_decoder=_delta(t_we, t_dec), t_driver=_delta(t_dec, t_wl),
            t_write=_delta(t_wl, t_sn), sn_start=sn_start, sn_final=sn_final,
            vneg_peak_a=(
                float(np.abs(wave["ineg"][(t > t0) & (t < t1)]).max()) if "ineg" in wave.v else None
            ),
            supply_charge=charge, supply_window=span,
        )
    return out


@dataclass
class RefreshTiming:
    """One refresh access of one cell (see :func:`simulate_refresh`); times in seconds from ``re``."""

    row: int
    col: str
    data: int
    sn_start: float
    """Storage-node level the cell started at (a decayed '1' or '0')."""
    sn_final: float
    """Level the access leaves the cell at: at the end of the bench, after the write word line
    (which opens and closes some time after ``we``) and the precharge are off."""
    t_read: float
    """When ``we`` rose: the read was taken as resolved."""
    t_access: float
    """When ``we`` fell: the whole refresh access."""
    restored: bool
    """A '1' back within 5% of vdd; a '0' at or below where it started and still a '0' (below
    vdd/2 -- the read tolerance is above that everywhere)."""
    read_ok: bool = True
    """Whether the read at the strobe got the data: the read bit line below vdd/2 for a '1',
    above it for a '0' (it leads ``dout``). A cell whose bit line is past that at the strobe is
    past its read margin -- a read would get it wrong too -- and its bit line is frozen near the
    read buffer's threshold, so the write-back puts an in-between level back."""
    rbl_strobe: float = 0.0
    """The read bit line at the strobe (V)."""
    supply_charge: float | None = None
    supply_window: float | None = None


def simulate_refresh(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    row: int,
    cells: dict[str, tuple[int, float]],
    workdir: str | Path,
    settings: BenchSettings | None = None,
    logic: LogicConfig | None = None,
    read_time: float | None = None,
    write_time: float = 2e-9,
    pch_delay: float = 1e-9,
) -> dict[str, RefreshTiming]:
    """One refresh access of row ``row``: read it, then write every cell's own value back.

    A macro with ``refresh_in_macro`` writes the bits back itself: every ``din`` is held high
    through the access (as the controller does in a refresh) and ``pch`` falls ``pch_delay`` after ``we`` (the controller's extra cycle, short here), so the bit
    lines hold until the write word line has closed. Otherwise:

    ``cells`` maps a kept column to ``(data, storage-node level)`` -- typically a '1' decayed
    below vdd and a '0' decayed up toward the read tolerance. ``re`` and ``pch`` rise; at
    ``read_time`` (default: ``strobe_factor`` x the replica's ``rdy``, measured by a probe read, or
    half the bench window without a replica) -- the strobe of a normal read -- ``re`` falls but
    ``pch`` stays up: the read word line is off and the precharge still off, so each read bit
    line holds the level it had at the strobe, and ``dout`` with it. Then ``we`` pulses for
    ``write_time``, and only after that does ``pch`` fall. Every kept column's ``din`` is ``NOT
    dout`` through a real logic inverter (``logic``'s sizes): the controller's loop-back; the write
    driver inverts ``din`` again, so ``wbl`` carries the stored value.

    Holding ``re`` up through the write instead is wrong: a '0' keeps discharging its bit line
    slowly (that is why a read samples at the strobe), and once ``dout`` flips mid-write the
    loop-back writes a '1', which turns the cell on harder -- a decayed '0' comes back a '1'.
    """
    settings = settings or BenchSettings()
    logic = logic or LogicConfig.for_tech(tech)
    vdd = corner.vdd
    workdir = Path(workdir)
    if read_time is None:
        if model.replica:
            first = next(iter(cells))
            probe = simulate_reads(tech, model, corner, row, {first: (1, vdd)}, workdir / "probe", settings)[first]
            if model.self_timed and probe.t_cutoff is not None:
                # the macro ends the read itself; re only has to outlast that
                read_time = 1.2 * probe.t_cutoff + settings.slew
            elif probe.t_rdy is not None:
                read_time = settings.strobe_factor * probe.t_rdy + settings.slew
            else:
                read_time = settings.window / 2
        else:
            read_time = settings.window / 2
    t0 = settings.settle
    s = settings.slew
    tr = t0 + read_time  # the strobe: re falls here
    tw = tr + s  # we rises once the read word line is off
    tw1 = tw + write_time
    t1 = tw1 + (pch_delay if model.refresh_in_macro else s)  # precharge back on after the write
    stop = t1 + settings.recovery

    probes = {"re": "re", "we": "we", "qvdd": "qvdd"}
    for col in cells:
        probes[f"sn_{col}"] = model.sn[(row, col)]
        probes[f"dout_{col}"] = model.pin(f"dout_{col}")
        probes[f"rbl_{col}"] = f"{model.prefix}rbl_{col}"
        probes[f"wbl_{col}"] = f"{model.prefix}wbl_{col}"
    sources = [
        _pwl("re", "re", [(0, 0), (t0, 0), (t0 + s, vdd), (tr, vdd), (tr + s, 0)]),
        _pwl("pch", "pch", [(0, 0), (t0, 0), (t0 + s, vdd), (t1, vdd), (t1 + s, 0)]),
        _pwl("we", "we", [(0, 0), (tw, 0), (tw + s, vdd), (tw1, vdd), (tw1 + s, 0)]),
        *_address_sources(model, row, vdd),
    ]
    for c in _data_columns(model):
        din, dout = model.pin(f"din_{c}"), model.pin(f"dout_{c}")
        if model.refresh_in_macro:
            sources.append(f"vdin_{c} {din} 0 {vdd:g}")  # the macro writes its sensed bits back
            continue
        sources += [
            tech.spice_device(f"lbp_{c}", din, dout, "vdd", "vdd", DeviceKind.PMOS, logic.pmos_width, logic.length),
            tech.spice_device(f"lbn_{c}", din, dout, "vss", "vss", DeviceKind.NMOS, logic.nmos_width, logic.length),
        ]
    initial = {
        node: (cells[key[1]][1] if key[0] == row and key[1] in cells else settings.sn_idle)
        for key, node in model.sn.items()
        if key[1] != "rep"
    }
    wave = _run(_deck(tech, model, corner, sources, initial, probes, stop, settings, "wave.csv"), workdir, probes)
    t = wave.t
    t_re = crossing(t, wave["re"], vdd / 2, True) or t0
    t_we = crossing(t, wave["we"], vdd / 2, True, t_re) or tw
    t_off = crossing(t, wave["we"], vdd / 2, False, t_we) or tw1
    # what the access leaves behind, not the level at we's fall: the write word line trails we
    # through the decoder (1.4 ns on SG13G2 at ss), so the node is still rising when we falls
    end = t > stop - 0.3e-9
    charge, span = _charge(wave, t0 - s, stop)
    out = {}
    for col, (data, level) in cells.items():
        sn = wave[f"sn_{col}"]
        sn_start = float(sn[t > t0 - 1e-9][0])
        sn_final = float(sn[end].mean())
        restored = sn_final > 0.95 * vdd if data else (sn_final <= sn_start + 0.02 and sn_final < vdd / 2)
        rbl_strobe = float(np.interp(tr, t, wave[f"rbl_{col}"]))
        out[col] = RefreshTiming(
            row=row, col=col, data=data, sn_start=sn_start, sn_final=sn_final,
            t_read=t_we - t_re, t_access=t_off - t_re, restored=restored,
            read_ok=(rbl_strobe < vdd / 2) == bool(data), rbl_strobe=rbl_strobe,
            supply_charge=charge, supply_window=span,
        )
    return out


def macro_refresh_interval(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    row: int,
    col: str,
    threshold: float,
    workdir: str | Path,
    settings: BenchSettings | None = None,
    hold0: float = 10e-6,
    max_doublings: int = 10,
) -> RefreshResult:
    """Write a '0' into ``(row, col)`` on the real extracted macro, then hold idle and time how
    long ``sn`` takes to drift up to ``threshold`` -- typically ``zero_level_limit``'s own,
    write-independent value. The real-word-line/bit-line-RC counterpart of
    ``verify.sim.refresh_interval``'s single-cell bench (see :class:`RefreshResult`).

    Retries with the hold doubled (up to ``max_doublings`` times) if the first attempt is not
    long enough to see the crossing.
    """
    settings = settings or BenchSettings()
    vdd = corner.vdd
    t0 = settings.settle
    t1 = t0 + settings.window
    s = settings.slew

    probes = {"we": "we", "sn": model.sn[(row, col)]}
    edge = [(0, 0), (t0, 0), (t0 + s, vdd), (t1, vdd), (t1 + s, 0)]
    sources = [
        _pwl("we", "we", edge),
        "vre re 0 0",
        "vpch pch 0 0",
        *_address_sources(model, row, vdd),
        *_din_sources(model, lambda c: vdd),  # every column writes a '0'
    ]
    initial = {
        node: (vdd if key == (row, col) else settings.sn_idle)  # (row, col) held the opposite '1'
        for key, node in model.sn.items()
        if key[1] != "rep"
    }

    workdir = Path(workdir)
    after = t1 + 2 * s  # past the word line's own fall transient

    # The settled level, from its own short, fine-step run: decoupled from the long-hold
    # attempts below, whose step is coarsened to the hold (see the loop) and would otherwise
    # land on a stale or still-settling sample instead of the write's real settled value.
    settle_wave = _run(
        _deck(tech, model, corner, sources, initial, probes, t1 + 50 * s, settings, "settle.csv"),
        workdir / "settle", probes, csv="settle.csv",
    )
    idx = int(np.searchsorted(settle_wave.t, after))
    settled = float(settle_wave["sn"][min(idx, len(settle_wave.t) - 1)])
    if settled >= threshold:
        # already past threshold once settled: no interval during which the data is good
        return RefreshResult(settled, threshold, 0.0, 0.0, 0)

    hold = hold0
    for attempt in range(max_doublings):
        stop = t1 + hold
        # settings.step (tens of ps) is for a ~10-30 ns read/write bench; at a multi-us to
        # multi-ms hold it would ask ngspice for tens to hundreds of millions of print points
        # -- scale it to the hold instead, capped at ~20k points regardless of how long it is.
        deck_settings = replace(settings, step=max(settings.step, hold / 20_000))
        wave = _run(
            _deck(tech, model, corner, sources, initial, probes, stop, deck_settings, "wave.csv"),
            workdir / f"a{attempt}", probes,
        )
        fail_time = crossing(wave.t, wave["sn"], threshold, True, after)
        if fail_time is not None:
            return RefreshResult(settled, threshold, max(0.0, fail_time - after), hold, attempt + 1)
        if attempt + 1 < max_doublings:
            hold *= 4
    return RefreshResult(settled, threshold, None, hold, max_doublings)


# --- the characterisation ------------------------------------------------------------------


def simulate_idle(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    workdir: str | Path,
    settings: BenchSettings | None = None,
) -> SupplyCurrents:
    """The macro at rest -- enables low, every storage node at its idle level -- for a few ns:
    the average current from ``vdd`` and (if the macro has one) ``vneg``. This is leakage plus
    whatever a level shifter that is not fully off burns."""
    settings = settings or BenchSettings()
    vdd = corner.vdd
    stop = 10e-9
    probes = {"qvdd": "qvdd"}
    if model.vneg_net is not None:
        probes["ivneg"] = "i(vvneg)"
    sources = [
        "vre re 0 0",
        "vwe we 0 0",
        "vpch pch 0 0",
        *_address_sources(model, 0, vdd),
        *_din_sources(model, lambda c: vdd),
    ]
    initial = {node: settings.sn_idle for key, node in model.sn.items() if key[1] != "rep"}
    wave = _run(
        _deck(tech, model, corner, sources, initial, probes, stop, settings, "wave.csv"),
        Path(workdir), probes,
    )
    window = wave.t > stop - 4e-9
    charge, span = _charge(wave, stop - 4e-9, stop)
    return SupplyCurrents(
        vdd_a=charge / span,
        vneg_a=float(np.abs(wave["ivneg"][window]).mean()) if "ivneg" in wave.v else None,
    )


def zero_level_limit(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    row: int,
    col: str,
    high: float,
    workdir: str | Path,
    settings: BenchSettings | None = None,
    resolution: float = 0.02,
) -> float:
    """Highest storage-node level from which a '0' is still read correctly, by bisection in
    ``[0, high]`` (0.0 if even a perfect '0' misreads; ``high`` if the written '0' level is fine).

    This is the design target for the write path: it says how low a write has to pull the
    storage node, hence how far below vss a PMOS write device's word line has to go.

    ``characterise()`` calls this with ``high`` set to the level the write actually settled at,
    which makes it answer "does the write already clear its own target" rather than "what is the
    read device's true tolerance" -- once a write already passes, it just echoes ``high`` back,
    since bisection never runs. For the true, write-independent threshold (to size a refresh
    interval against, say -- see ``verify.sim.refresh_interval``), pass a generous ``high`` (e.g.
    ``corner.vdd``) instead of the written level.
    """
    workdir = Path(workdir)

    def reads_zero(level: float) -> bool:
        return simulate_read(
            tech, model, corner, row, col, 0, level, workdir / f"l{level:.3f}", settings
        ).correct

    if reads_zero(high):
        return high
    if not reads_zero(0.0):
        return 0.0
    low = 0.0
    while high - low > resolution:
        mid = (low + high) / 2
        low, high = (mid, high) if reads_zero(mid) else (low, mid)
    return low


def refresh_zero_limit(
    tech: Technology,
    model: TimingModel,
    corner: Corner,
    row: int,
    col: str,
    high: float,
    workdir: str | Path,
    settings: BenchSettings | None = None,
    write_time: float = 2e-9,
    floor: float | None = None,
    resolution: float = 0.02,
) -> tuple[float, float]:
    """``(limit, after)``: the highest storage-node level from which a refresh
    (:func:`simulate_refresh`, its write ``write_time`` long) leaves a '0' no higher than it was
    or than ``floor`` (where a written '0' settles, if above it: a refresh brings a '0' toward that,
    from below too; default the idle level), by bisection between the idle level and ``high``, and
    the level that refresh leaves it at --
    :func:`zero_level_limit`'s counterpart for a macro that refreshes itself
    (``refresh_in_macro``). There the write driver is ``NAND(din, rbl)``: it writes back the level
    the read bit line reached, not a sensed bit, so a decayed '0' that pulls its bit line near the
    NAND's threshold comes back in between, or a '1', while a read of it is still right; and a
    short write pulls even a well-read '0' only part of the way back down. A refresh interval is
    the drift from ``after`` up to the lower of the two limits (the idle level as ``limit``: a
    refresh does not restore a '0' at all)."""
    workdir = Path(workdir)
    settings = settings or BenchSettings()

    def refresh(level: float) -> RefreshTiming:
        return simulate_refresh(
            tech, model, corner, row, {col: (0, level)}, workdir / f"l{level:.3f}", settings, write_time=write_time
        )[col]

    low = settings.sn_idle
    floor = max(low, floor or 0.0)
    best = refresh(low)

    def restores_zero(r: RefreshTiming) -> bool:
        # (no higher than it was: high is at most the read's limit, so it still reads as a '0'; not
        # read_ok, whose vdd/2 on the bit line is not where the write-back NAND switches)
        return r.sn_final <= max(r.sn_start, floor)

    if not restores_zero(best):
        return low, best.sn_final
    top = refresh(high)
    if restores_zero(top):
        return high, top.sn_final
    while high - low > resolution:
        mid = (low + high) / 2
        r = refresh(mid)
        if restores_zero(r):
            low, best = mid, r
        else:
            high = mid
    return low, best.sn_final


@dataclass
class CornerResult:
    corner: Corner
    write1: WriteTiming
    write0: WriteTiming
    reads: list[ReadTiming] = field(default_factory=list)
    zero_limit: float | None = None
    """See :func:`zero_level_limit`."""
    mc: MonteCarlo | None = None
    """Local-mismatch statistics at this corner, if the characterisation ran them."""
    supply: SupplyCurrents | None = None
    """Idle supply currents (negative-rail macros)."""
    refresh: list[RefreshTiming] = field(default_factory=list)
    """In-place refreshes (``refresh_in_macro``): a decayed '1' and a '0' at the read limit, near
    and far, each written back by the macro itself."""

    @property
    def refresh_ok(self) -> bool | None:
        """Every in-place refresh read its bit and wrote it back; ``None`` without any."""
        return all(r.restored for r in self.refresh) if self.refresh else None

    @property
    def sn_one(self) -> float:
        return self.write1.sn_final

    @property
    def sn_zero(self) -> float:
        return self.write0.sn_final

    def reads_of(self, data: int) -> list[ReadTiming]:
        return [r for r in self.reads if r.data == data]

    @property
    def access(self) -> float | None:
        """Slowest ``re``-to-``dout`` of a freshly written '1'."""
        times = [r.t_dout for r in self.reads_of(1) if r.sn_level >= 0.98 * self.sn_one and r.t_dout is not None]
        return max(times) if times else None

    @property
    def reads_zero_correctly(self) -> bool:
        """Whether every read of a stored '0' still showed a '0' when the data was sampled.

        A PMOS write device only pulls the storage node down to about |Vtp|, so with a
        word line that goes no lower than vss the '0' sits above the read device's threshold
        and reads as a '1'.
        """
        return all(r.correct for r in self.reads_of(0))

    @property
    def write_time(self) -> float | None:
        return self.write1.t_write

    @property
    def precharge(self) -> float | None:
        times = [r.t_precharge for r in self.reads if r.t_precharge is not None]
        return max(times) if times else None


@dataclass
class Characterisation:
    rows: int
    cols: int
    replica: bool
    corners: list[CornerResult]
    levels: tuple[float, ...]
    """Fractions of the settled '1' level the reads started from."""
    vneg: float | None = None
    """The level of the negative supply, for a negative-write-word-line macro."""
    bank: int | None = None
    """Which bank of a banked macro this characterises."""

    # --- derived -----------------------------------------------------------------------

    def replica_ratios(self) -> dict[str, float]:
        """Per corner, the slowest valid read over the replica's ``rdy``: how much later than
        ``rdy`` the data of the weakest legitimate '1' is actually good."""
        out = {}
        for result in self.corners:
            ratios = [
                r.t_dout / r.t_rdy
                for r in result.reads_of(1)
                if r.t_dout is not None and r.t_rdy
            ]
            if ratios:
                out[result.corner.name] = max(ratios)
        return out

    def recommended_margin(self) -> float | None:
        """A single factor by which ``rdy`` must be delayed to be valid in every corner, for
        every level and cell tested (>= 1)."""
        ratios = self.replica_ratios()
        return max(1.0, max(ratios.values())) if ratios else None

    def write_stress(self) -> float | None:
        """The largest voltage a write puts across a gate oxide (the driver's and the write
        device's): the highest supply of any corner minus ``vneg``. Volts; ``None`` without one."""
        if self.vneg is None:
            return None
        return max(c.corner.vdd for c in self.corners) - self.vneg

    def supply_table(self) -> str:
        """What the negative supply costs: idle currents per corner (empty without one)."""
        rows = [c for c in self.corners if c.supply is not None and c.supply.vneg_a is not None]
        if not rows:
            return ""
        lines = ["| corner | idle vdd (uA) | idle vneg (uA) | vneg peak in a write (uA) |", "|---|---|---|---|"]
        for c in rows:
            peak = max((w.vneg_peak_a for w in (c.write0, c.write1) if w.vneg_peak_a is not None), default=None)
            lines.append(
                f"| {c.corner.name} | {c.supply.vdd_a * 1e6:.1f} | {c.supply.vneg_a * 1e6:.1f} "
                f"| {'-' if peak is None else f'{peak * 1e6:.0f}'} |"
            )
        return "\n".join(lines)

    def statistical_margin(self, sigma: float = 3.0) -> float | None:
        """Like :meth:`recommended_margin`, but for the dies ``sigma`` standard deviations out of
        the mismatch distribution instead of the nominal ones: the factor ``rdy`` has to be
        delayed by, over every corner that was simulated with Monte Carlo."""
        margins = [c.mc.margin(sigma) for c in self.corners if c.mc is not None and c.mc.margin(sigma)]
        return max(margins) if margins else None

    def mc_table(self) -> str:
        """The Monte-Carlo summary, one line per corner (empty if there was none)."""
        from gc_compiler.verify.montecarlo import table

        runs = [c.mc for c in self.corners if c.mc is not None]
        return table(runs) if runs else ""

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "replica": self.replica,
            "bank": self.bank,
            "levels": list(self.levels),
            "recommended_margin": self.recommended_margin(),
            "statistical_margin_3sigma": self.statistical_margin(3.0),
            "vneg_v": self.vneg,
            "write_oxide_stress_v": self.write_stress(),
            "corners": [
                {
                    "corner": asdict(c.corner) | {"name": c.corner.name},
                    "sn_one_v": c.sn_one,
                    "sn_zero_v": c.sn_zero,
                    "access_s": c.access,
                    "write_s": c.write_time,
                    "reads_zero_correctly": c.reads_zero_correctly,
                    "zero_limit_v": c.zero_limit,
                    "precharge_s": c.precharge,
                    "write1": asdict(c.write1),
                    "write0": asdict(c.write0),
                    "reads": [asdict(r) for r in c.reads],
                    "monte_carlo": None if c.mc is None else c.mc.to_dict(),
                    "supply": None if c.supply is None else asdict(c.supply),
                    **({"refresh_ok": c.refresh_ok, "refresh": [asdict(r) for r in c.refresh]} if c.refresh else {}),
                }
                for c in self.corners
            ],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        return path

    def table(self) -> str:
        """A markdown summary, one line per corner (times in ns, levels in V)."""

        def ns(x: float | None) -> str:
            return "-" if x is None else f"{x * 1e9:.2f}"

        head = "| corner | access | write | precharge | '1' level | '0' level | '0' reads | '0' limit |"
        rule = "|---|---|---|---|---|---|---|---|"
        if self.replica:
            head += " rdy (near/far) | dout/rdy |"
            rule += "---|---|"
        refreshes = any(c.refresh for c in self.corners)
        if refreshes:
            head += " refresh |"
            rule += "---|"
        rows = [head, rule]
        ratios = self.replica_ratios()
        for c in self.corners:
            line = (
                f"| {c.corner.name} | {ns(c.access)} | {ns(c.write_time)} | {ns(c.precharge)} "
                f"| {c.sn_one:.2f} | {c.sn_zero:.2f} | {'ok' if c.reads_zero_correctly else 'FAIL'} "
                f"| {'-' if c.zero_limit is None else f'{c.zero_limit:.2f}'} |"
            )
            if self.replica:
                rdys = sorted(r.t_rdy for r in c.reads if r.t_rdy is not None)
                span = f"{ns(rdys[0])} / {ns(rdys[-1])}" if rdys else "-"
                line += f" {span} | {ratios.get(c.corner.name, float('nan')):.2f} |"
            if refreshes:
                slowest = max((r.t_access for r in c.refresh), default=None)
                line += f" {'ok' if c.refresh_ok else 'FAIL'} {ns(slowest)} |"
            rows.append(line)
        return "\n".join(rows)


def characterise(
    tech: Technology,
    cfg: ArrayConfig,
    gds: str | Path,
    cell: str,
    workdir: str | Path = "build/timing",
    corners: tuple[Corner, ...] | None = None,
    levels: tuple[float, ...] = (1.0, 0.7),
    jobs: int | None = None,
    keep: tuple[int, ...] | None = None,
    settings: BenchSettings | None = None,
    monte_carlo: int = 0,
    mc_seed: int = 1,
    mc_zero_level: float | None = None,
    bank: int | None = None,
    flat: str | Path | None = None,
) -> Characterisation:
    """Characterise the compiled macro ``cell`` in ``gds`` (see the module docstring), or one
    ``bank`` of a banked macro.

    Per corner: write a '1' and a '0' (which give the levels they settle at), then read cells
    at the two rows that put the far end of the array at opposite ends of the word line (the
    first and last, whose drivers are on opposite sides) and both end columns, from each of
    ``levels`` (fractions of the settled '1' level -- what a decayed '1' looks like) plus a '0'.

    ``monte_carlo`` > 0 also samples that many dies with local device mismatch at each corner
    (:mod:`gc_compiler.verify.montecarlo`; seeds ``mc_seed`` upward, the same dies at every
    corner), with '0's starting from each die's own written level (or ``mc_zero_level`` volts).
    """
    workdir = Path(workdir)
    corners = corners or standard_corners(tech)
    model = build_model(tech, cfg, gds, cell, workdir / "extract", keep=keep, bank=bank, flat=flat)
    last = cfg.rows - 1
    columns = list(model.kept)
    sites = [(row, col) for row in (0, last) for col in columns]
    settings = settings or BenchSettings()
    jobs = jobs or min(os.cpu_count() or 1, 16)
    far_row, far_col = 0, columns[-1]

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # phase 1: the levels a '1' and a '0' settle at, per corner
        writes = {
            (corner.name, data): pool.submit(
                simulate_write, tech, model, corner, far_row, far_col, data,
                workdir / "runs" / corner.name / f"write{data}", settings,
            )
            for corner in corners
            for data in (1, 0)
        }
        settled = {key: future.result() for key, future in writes.items()}

        # phase 2: every read, from those levels
        reads: dict[str, list] = {corner.name: [] for corner in corners}
        for corner in corners:
            w1, w0 = settled[(corner.name, 1)], settled[(corner.name, 0)]
            base = workdir / "runs" / corner.name
            for row, col in sites:
                for level in levels:
                    reads[corner.name].append(pool.submit(
                        simulate_read, tech, model, corner, row, col, 1, level * w1.sn_final,
                        base / f"read1_r{row}_c{col}_l{level:g}", settings,
                    ))
                reads[corner.name].append(pool.submit(
                    simulate_read, tech, model, corner, row, col, 0, w0.sn_final,
                    base / f"read0_r{row}_c{col}", settings,
                ))
        limits = {
            corner.name: pool.submit(
                zero_level_limit, tech, model, corner, far_row, far_col,
                settled[(corner.name, 0)].sn_final,
                workdir / "runs" / corner.name / "zero_limit", settings,
            )
            for corner in corners
        }
        idle = {
            corner.name: pool.submit(
                simulate_idle, tech, model, corner, workdir / "runs" / corner.name / "idle", settings
            )
            for corner in corners
        }
        results = [
            CornerResult(
                corner, settled[(corner.name, 1)], settled[(corner.name, 0)],
                [future.result() for future in reads[corner.name]],
                limits[corner.name].result(),
                supply=idle[corner.name].result(),
            )
            for corner in corners
        ]
        if model.refresh_in_macro:
            # the in-place refresh: a '1' decayed as far as a refresh lets it, and a '0' at the
            # highest level that still reads, near and far -- the write driver's NAND must judge
            # them as the read buffer does
            near = columns[0]
            decayed = _measured(tech).decayed_one_fraction
            write_time = _measured(tech).controller.write_ns * 1e-9  # the controller's write-back
            refreshes = {}
            for result in results:
                one = decayed * result.sn_one
                zero = result.zero_limit if result.zero_limit is not None else result.sn_zero
                base = workdir / "runs" / result.corner.name
                refreshes[result.corner.name] = [
                    pool.submit(
                        simulate_refresh, tech, model, result.corner, far_row, cells, base / f"refresh{i}", settings,
                        write_time=write_time,
                    )
                    for i, cells in enumerate(({near: (1, one), far_col: (0, zero)}, {near: (0, zero), far_col: (1, one)}))
                ]
            for result in results:
                result.refresh = [r for future in refreshes[result.corner.name] for r in future.result().values()]
    if monte_carlo:
        from gc_compiler.verify.montecarlo import monte_carlo as run_monte_carlo

        for result in results:
            result.mc = run_monte_carlo(
                tech, model, result.corner, monte_carlo,
                workdir / "runs" / result.corner.name / "mc",
                zero_level=mc_zero_level, seed=mc_seed, jobs=min(jobs, 8), settings=settings,
            )
    return Characterisation(
        cfg.rows, cfg.cols, cfg.replica_column, results, levels,
        vneg=model.vneg if model.vneg_net is not None else None, bank=bank,
    )
