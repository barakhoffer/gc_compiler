"""Configuration objects for the gain-cell compiler."""

from __future__ import annotations

from dataclasses import dataclass, field

from gc_compiler.tech.rules import DeviceKind


def _default_tech():
    from gc_compiler.tech import default_technology

    return default_technology()


def _sized(group: str, key: str):
    """A default from the default technology's :class:`~gc_compiler.tech.policy.Sizing`."""
    return field(default_factory=lambda: getattr(_default_tech().sizing, group)[key])


@dataclass(frozen=True)
class BitcellConfig:
    """Geometry of one 3T gain cell.

    The default topology is a PMOS write port with an NMOS read pair: the write
    transistor ``mw`` charges the storage node ``sn`` from ``wbl`` under ``wwl``
    control, ``mr`` converts ``sn`` into a current and the ``rwl``-gated ``ms``
    selects the cell onto ``rbl``.
    """

    write_kind: DeviceKind = DeviceKind.PMOS
    read_kind: DeviceKind = DeviceKind.NMOS
    write_width: float = _sized("bitcell", "write_width")
    write_length: float = _sized("bitcell", "write_length")
    """The default technology's (``tech.sizing.bitcell``; for another, :meth:`for_tech`). On sky130
    the LVT write device's 0.35 um (its LVT PMOS model is characterised from there); the plain
    cell's is 0.18 (:meth:`plain`)."""
    read_width: float = _sized("bitcell", "read_width")
    read_length: float = _sized("bitcell", "read_length")
    """Gate length of ``ms``, the ``rwl``-gated read-select device (and the default for ``mr``
    below); this is also the word line's own poly width, so it sets ``rwl``'s resistance."""
    mr_length: float | None = _sized("bitcell", "mr_length")
    """Override ``mr``'s gate length -- the device gated by the storage node, whose threshold
    sets how high a stored '0' still reads as one. Defaults to ``read_length``; splitting it out
    lets a higher '0' tolerance be bought without also lengthening ``ms`` (which does not affect
    the tolerated level, only costs it the same access-time and area penalty)."""
    write_flavor: str | None = _sized("bitcell", "write_flavor")
    """The write device's implant. The default cell is the LVT write with a 0.5 um ``mr``: its
    written '0' reads back at every corner (the plain cell's fails at sf) and it keeps its data
    about twelve times longer, for 5% more area (see the README)."""
    read_flavor: str | None = None
    pitch_x: float | None = None
    """Override the column pitch; the minimum legal pitch is used when ``None``."""

    @classmethod
    def for_tech(cls, tech, **overrides) -> BitcellConfig:
        """``tech``'s default cell (``tech.sizing.bitcell``), with ``overrides`` applied."""
        return cls(**{**tech.sizing.bitcell, **overrides})

    @classmethod
    def plain(cls, tech=None, **overrides) -> BitcellConfig:
        """The plain 3T cell: a standard-threshold write device and ``mr`` as long as ``ms`` (the
        compiler's default before the LVT cell; ``tech.sizing.plain_bitcell``, the default
        technology's without ``tech``), with ``overrides`` applied."""
        tech = tech or _default_tech()
        return cls(**{**tech.sizing.bitcell, **tech.sizing.plain_bitcell, **overrides})

    @property
    def effective_mr_length(self) -> float:
        """``mr_length`` if set, else ``read_length``."""
        return self.mr_length if self.mr_length is not None else self.read_length

    @property
    def name(self) -> str:
        mr = f"_mr{self.mr_length:g}" if self.mr_length is not None else ""
        return (
            f"gc3t_w{self.write_width:g}x{self.write_length:g}"
            f"_r{self.read_width:g}x{self.read_length:g}{mr}"
        )


WIDE_ARRAY = _default_tech().sizing.wide_array_cols
"""The default technology's (``tech.sizing.wide_array_cols``): from this many columns word lines are
strapped and driven harder by default. On sky130, 64: at 64 columns that cuts a read from 1.78 to
1.09 ns, at 128 from 3.42 to 1.25, for 2.6-4% area (README)."""


def _policy(key: str):
    return field(default_factory=lambda: getattr(_default_tech().policy, key))


@dataclass(frozen=True)
class ArrayConfig:
    """Bitcell array organisation."""

    rows: int = 32
    cols: int = 32
    bitcell: BitcellConfig = field(default_factory=BitcellConfig)
    tap_row_interval: int = _policy("tap_row_interval")
    """Insert a tap row every N mirrored row pairs (0 keeps only the edge tap rows)."""
    edge_tap_rows: bool = True
    strap_column_interval: int = _policy("strap_column_interval")
    """Insert a vertical vss/n-well strap column every N bitcell columns (0 disables): the first
    after N/2 columns, then every N, and one at the end, so none is further than N/2 from one.

    Each row's ``vss`` rail and each row-pair's n-well merge horizontally across columns
    by abutment, but nothing ties them vertically to each other or to the tap rows'
    actual p+/n+ contacts without this column.
    """

    wordline_straps: bool | None = None
    """Strap every word line in metal3 (default: on from ``WIDE_ARRAY`` columns).

    Word lines are poly (about 48 ohm per square), so on a wide array their far end lags the
    driver end. With straps, each word line gets a metal3 line over it, fed by its driver and
    tied down to the poly in every strap column, so no stretch of poly is longer than
    ``strap_column_interval`` columns. Needs strap columns.
    """

    self_timed_read: bool | None = None
    """End every read by itself when the replica column says the data is there (default: on
    whenever there is a replica column; it applies to decoded macros).

    The decoder's read enable becomes ``re AND NOT rdy``, so the read word line turns off one
    decoder delay after the replica's ``rdy``; with ``pch`` still up the bit lines -- and ``dout``
    -- then hold. A read no longer depends on how long ``re`` is held: at every corner it ends
    where that corner's replica says, which a fixed read time cannot do (a slow corner needs
    longer than a fast corner's stored '0's last). Needs the replica column and the decoder.
    """

    refresh_in_macro: bool | None = None
    """Refresh without the data leaving the macro (default: on whenever reads are self-timed):
    each column's write driver is ``NAND(din, rbl)``, so a refresh -- a read, ``re`` dropped with
    the precharge still off, then a write with every ``din`` high -- writes each column's sensed bit
    straight back. The controller then drives ``din`` high in a refresh instead of looping ``dout``
    back, so no ``dout -> din`` path has to be timed, and holds ``din`` low through user reads. It
    costs a NAND slice of column height (+4-9% macro area) and some read energy (+9-17% at tt
    against the loop-back, ``measurements/data/refresh_in_macro.json``); ``False`` keeps the
    loop-back. Needs self-timed reads."""

    wordline_driver_strength: float | None = None
    """Word-line driver size, as a multiple of the logic cells' device widths (default: 4 from
    ``WIDE_ARRAY`` columns, 1 below).

    A driver charges its word line's poly (and strap) and every cell's gate on it, so on a wide
    array the minimum-size inverter is most of the access. Its devices are laid out along the
    driver column's width, so a stronger driver makes the column wider, not taller.
    """

    replica_column: bool = True
    """Add a replica bit-line column after the last data column (on by default: it is what times
    a read, and a self-timed read needs it).

    Its cells share every row's word lines with the array, so a read through them sees the same
    decoder, driver and word-line delay, but their storage nodes are tied to the write bit
    line (held at vdd), so the replica read bit line always discharges. It is not counted in
    ``cols`` or ``bits``.
    """

    negative_write_wl: bool = False
    """Drive the write word lines below vss.

    A PMOS write device only pulls the storage node down to about ``|Vtp|`` above its gate, so
    with a word line that stops at vss a written '0' sits above the read device's threshold and
    reads as a '1'. This replaces each write-word-line driver with a level shifter plus a driver
    whose pull-down NMOS sits in an isolated p-well (deep n-well) at a new ``vneg`` supply.
    """
    write_wl_low: float = field(default_factory=lambda: _default_tech().sizing.write_wl_low)
    """The level ``vneg`` is supplied at (volts, negative); only simulation uses it.

    The slow-PMOS/fast-NMOS corner sets it. On the extracted 16x20 macro a written '0' already
    reads back there at -0.2 V (0.14 V of margin, no failing mismatch die); -0.3 V gives 0.23 V.
    More does little for retention, which the write device's leakage and drain junction set: the
    worst refresh interval grows only from 88 us at -0.3 V to 122 us at -0.8 V
    (``measurements/data/vneg.json``). A write puts ``vdd - write_wl_low`` across the write device's and
    the driver's gate oxide: 2.1 V here (2.28 V at the ff supply), still above the 1.8 V devices'
    rating, against 2.6 V (2.78 V) at -0.8 V.
    """

    tech: object = field(default=None, repr=False, compare=False)
    """The technology :meth:`for_tech` gave the defaults of (``None``: the default technology's);
    only names use it (a bitcell that is the technology's default or plain one adds no hash)."""

    def __post_init__(self) -> None:
        if self.negative_write_wl and self.write_wl_low >= 0:
            raise ValueError("write_wl_low must be negative when negative_write_wl is set")
        if self.rows % 2:
            raise ValueError("rows must be even: cells are mirrored in pairs")
        # the automatic choices, resolved once so everything downstream reads plain values
        sizing = _default_tech().sizing
        wide = self.cols >= sizing.wide_array_cols
        if self.wordline_straps is None:
            object.__setattr__(self, "wordline_straps", wide and bool(self.strap_column_interval))
        if self.wordline_driver_strength is None:
            object.__setattr__(self, "wordline_driver_strength", sizing.wide_driver_strength if wide else 1.0)
        if self.self_timed_read is None:
            object.__setattr__(self, "self_timed_read", self.replica_column)
        if self.self_timed_read and not self.replica_column:
            raise ValueError("self_timed_read needs the replica column (its rdy ends the read)")
        if self.refresh_in_macro is None:
            object.__setattr__(self, "refresh_in_macro", self.self_timed_read)
        if self.refresh_in_macro and not self.self_timed_read:
            raise ValueError("refresh_in_macro needs self-timed reads (the read must end before the write-back)")
        if self.wordline_driver_strength <= 0:
            raise ValueError("wordline_driver_strength must be positive")
        if self.wordline_straps and not self.strap_column_interval:
            raise ValueError("wordline_straps needs strap columns (strap_column_interval > 0)")

    @classmethod
    def for_tech(cls, tech, **overrides) -> ArrayConfig:
        """The array with ``tech``'s defaults (cell, tap and strap intervals, when a wide array gets
        straps and stronger drivers, the negative rail), with ``overrides`` applied."""
        cols = overrides.get("cols", cls.__dataclass_fields__["cols"].default)
        interval = overrides.get("strap_column_interval", tech.policy.strap_column_interval)
        wide = cols >= tech.sizing.wide_array_cols
        base = {
            "bitcell": BitcellConfig.for_tech(tech),
            "tap_row_interval": tech.policy.tap_row_interval,
            "strap_column_interval": interval,
            "wordline_straps": wide and bool(interval),
            "wordline_driver_strength": tech.sizing.wide_driver_strength if wide else 1.0,
            "write_wl_low": tech.sizing.write_wl_low,
            "tech": tech,
        }
        return cls(**{**base, **overrides})

    @classmethod
    def basic(cls, tech=None, **overrides) -> ArrayConfig:
        """The array with none of the defaults' extras: the plain bitcell, no replica column, no
        self-timing, no straps and minimum-size drivers (the compiler's defaults before they
        changed), with ``overrides`` applied -- ``tech``'s, or the default technology's."""
        base = {
            "bitcell": BitcellConfig.plain(tech),
            "replica_column": False,
            "self_timed_read": False,
            "wordline_straps": False,
            "wordline_driver_strength": 1.0,
        }
        if tech is not None:
            return cls.for_tech(tech, **{**base, **overrides})
        return cls(**{**base, **overrides})

    def _bitcell_suffix(self) -> str:
        """Distinguishes arrays of the same size built from different bitcells (a cell name must be
        unique in a session); the technology's default bitcell adds nothing and its plain one
        ``_plain``."""
        if self.bitcell == (BitcellConfig.for_tech(self.tech) if self.tech else BitcellConfig()):
            return ""
        if self.bitcell == BitcellConfig.plain(self.tech):
            return "_plain"
        import hashlib

        return "_c" + hashlib.sha1(repr(self.bitcell).encode()).hexdigest()[:6]

    @property
    def bits(self) -> int:
        return self.rows * self.cols

    @property
    def total_cols(self) -> int:
        """Physical bit-line columns: the data columns plus the replica column, if any."""
        return self.cols + int(self.replica_column)

    @property
    def name(self) -> str:
        return (
            f"gc_array_{self.rows}x{self.cols}"
            + ("_rep" if self.replica_column else "")
            + ("_neg" if self.negative_write_wl else "")
            + ("_ws" if self.wordline_straps else "")
            + ("_st" if self.self_timed_read else "")
            + ("_rim" if self.refresh_in_macro else "")
            + self._bitcell_suffix()
        )
