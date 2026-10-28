"""What was *measured* on a technology rather than read off its rules: the refresh intervals of its
cells, how a replica tracks a read, the controller's timing. A technology without these (``None``)
makes the compiler refuse to guess them -- a refresh interval measured on another process could
lose data."""

from __future__ import annotations

from dataclasses import dataclass, field


class NotCharacterised(ValueError):
    """The technology has no measurement for what was asked: characterise it (or pass the value)."""


@dataclass(frozen=True)
class RefreshInterval:
    bitcell: dict
    """The cell, as :class:`~gc_compiler.config.BitcellConfig` field values (on top of the
    technology's default cell)."""
    microseconds: float
    negative_rail: float | None = None
    """With a negative write rail at this level or lower (volts); ``None`` without one."""
    refresh_in_macro: bool | None = None
    """Only for macros with (``True``) or without (``False``) the in-macro refresh, whose
    write-back restores a decayed '0' from a lower level than a read tolerates
    (:func:`~gc_compiler.verify.timing.refresh_zero_limit`); ``None``: either."""


@dataclass(frozen=True)
class ControllerDefaults:
    """The refresh controller's timing when no characterisation is given (ns), and its
    standard-cell library's own share of a read."""

    read_ns: float
    write_ns: float
    precharge_ns: float
    sample_ns: float
    macro_min_ns: float
    rdy_lag_ns: float
    timeout_ns: float
    write_settle_ns: float = 0.0
    """The shortest write that settles a '0' where the refresh intervals assume it is: a
    characterisation's write time (three times the slowest write's crossing) is never shorter."""


@dataclass(frozen=True)
class Characterised:
    refresh_intervals: tuple[RefreshInterval, ...]
    """The worst die's interval at the worst corner, per measured cell."""
    unmeasured_refresh_us: float
    """For a cell nobody measured: the shortest interval of any measured one."""
    strobe_factor: float
    """How much later than the replica's ``rdy`` a read's data is valid, at worst (non-self-timed
    reads sample there)."""
    controller: ControllerDefaults
    idle_storage_node_v: float
    """A storage node's level between accesses, what timing benches start a cell at."""
    decayed_one_fraction: float
    """How far (of vdd) a stored '1' may decay before a refresh must restore it."""
    macro_input_pf: float
    """A macro input pin's load in its timing library."""
    macro_output_transition_ns: float
    notes: dict = field(default_factory=dict)
    refresh_max_rows: int | None = None
    """The most rows per bank a refresh was measured to restore a decayed '1' in (``None``: no
    limit measured). A read bit line runs beside its write bit line the whole column; on a taller
    one the write bit line's swing, as the write driver follows what the read senses, couples the
    read bit line back toward the driver's threshold, and the '1' is written back lower than it
    was. Measured with the default cell; it holds every cell to it."""
    loopback_marginal: tuple[dict, ...] = ()
    """Cells (as :attr:`RefreshInterval.bitcell`) whose written '0' a loop-back column (no in-macro
    refresh) misreads at some corner under mismatch: the compiler warns when one is built that way."""
