"""Verilog for refreshing a macro: a controller, a behavioural model of the macro, a top level that
joins them, and a self-checking testbench.

**The refresh access.** The macro has no refresh logic of its own; the controller refreshes one
row with one ordinary access (see :func:`gc_compiler.verify.timing.simulate_refresh`): ``re`` and
``pch`` rise on the row; the macro is self-timed (``ArrayConfig.self_timed_read``), so its
replica's ``rdy`` turns the read word line off by itself once the data is there, and with ``pch``
still up each read bit line -- and ``dout`` -- holds the row's data; the controller then drops
``re`` and pulses ``we`` with ``din = ~dout`` fed back (the macro's write driver inverts ``din``,
so that rewrites each cell's own value); last ``pch`` falls. No data is stored anywhere but in
the row.

**The controller** (``<name>_ctrl``) is synchronous. The user side is a request/busy handshake: a
request (``req`` with ``we``, ``addr``, ``din``) is taken on a clock edge where ``busy`` is low,
and ``we``, ``addr`` and ``din`` must then hold until ``busy`` is low again -- the controller keeps
no copy of them: they drive the macro's address and data inputs directly (a refresh switches the
address to its own row a cycle before it starts). A read's data appears on ``dout`` with
``dout_valid`` for one cycle and holds until the next read: ``dout`` is a latch, open in the
second half of a read's last cycle and closing on the edge that ends the read. It hides the macro's
inversion (the macro's ``dout`` is the inverse of the ``din`` that wrote it): user data reads
back as written. A timer requests a refresh of the next row every ``REFRESH_CYCLES``; a pending
refresh takes the macro at the next idle cycle, ahead of the user, and ``busy`` is high while it
runs. Every phase lasts a whole number of clock cycles, from the macro's characterised timing
at the given clock period. By default no asynchronous input is sampled: the controller does not
watch ``rdy`` -- it holds ``re`` for at least the slowest corner's self-timed cutoff
(``read_ns``), which is safe because holding ``re`` longer than a read needs changes nothing in
a self-timed macro. With ``ControllerTiming(handshake=True)`` it ends each read on the replica
instead: ``rdy`` is synchronised and the data taken two edges after it is caught, so the read
lasts what the macro needs, at any corner and size, and ``read_ns`` only sets the timing
library; a read whose ``rdy`` never comes ends at ``timeout_ns`` and sets ``read_timeout``.

**The model** (``<macro name>``) has the macro's pins without the supplies, and keeps retention:
a row not rewritten within ``RETENTION_NS`` reads ``x``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gc_compiler.config import ArrayConfig
from gc_compiler.tech.characterised import NotCharacterised


def _measured(tech=None):
    """``tech``'s (the default technology's) :class:`~gc_compiler.tech.characterised.Characterised`."""
    if tech is None:
        from gc_compiler.tech import default_technology

        tech = default_technology()
    if tech.characterised is None:
        raise NotCharacterised(
            f"{tech.name} has no measured refresh intervals or controller timing: characterise it, or "
            "give them (ControllerTiming(...), interval_us=...)"
        )
    return tech.characterised


def _controller(key: str):
    return field(default_factory=lambda: getattr(_measured().controller, key))
from gc_compiler.netlist.decoder import address_bits, banked_address_bits
from gc_compiler.netlist.spice import macro_pins


@dataclass(frozen=True)
class ControllerTiming:
    """What the controller's cycle counts are derived from (nanoseconds, microseconds)."""

    clock_ns: float = 10.0
    read_ns: float = _controller("read_ns")
    """How long ``re`` is held: at least the slowest corner's self-timed cutoff (see
    :meth:`from_characterisation`)."""
    write_ns: float = _controller("write_ns")
    """``we`` pulse of a write or of a refresh's write-back."""
    precharge_ns: float = _controller("precharge_ns")
    """``pch`` low after a read before the next access."""
    interval_us: float = field(default_factory=lambda: _measured().unmeasured_refresh_us)
    """Refresh interval: every row rewritten at least this often (the macro's worst corner,
    with mismatch; :func:`default_refresh_interval_us` for a given cell)."""
    margin: float = 0.5
    """The controller refreshes each row within ``margin`` x ``interval_us``."""
    sample_ns: float = _controller("sample_ns")
    """The controller's own share of a read: from ``dout`` valid at its pins, through its input mux,
    to the capturing flop's setup, with the clock tree's skew and uncertainty (static timing
    checks it; 1 ns failed at the slow corner). A read lasts ``read_ns + sample_ns``, in whole
    cycles."""
    macro_min_ns: float = _controller("macro_min_ns")
    """The earliest the macro's outputs move after the controller's do (its fastest decoder and
    driver): the hold budget of the paths back into the controller."""
    handshake: bool = False
    """End each read on the macro's own ``rdy`` instead of after a fixed ``read_ns``: the
    controller synchronises ``rdy`` (two flops) and takes the data on the second edge after it
    rises, so a read lasts what that die, corner and bank need and ``read_ns`` no longer has to
    cover the slowest one. It costs the synchroniser's latency: at least three cycles a read."""
    rdy_lag_ns: float = _controller("rdy_lag_ns")
    """With ``handshake``: how much later than ``rdy`` the macro's ``dout`` is valid (the replica
    tracks the data within a few percent; see :meth:`from_characterisation`). It has two clock
    cycles, less ``sample_ns``."""
    timeout_ns: float = _controller("timeout_ns")
    """With ``handshake``: a read whose ``rdy`` has not risen after this long (plus the
    synchroniser) ends anyway and sets the sticky ``read_timeout`` output, so a stuck replica
    cannot stop refresh. Far beyond any read the macro takes."""

    @classmethod
    def default(cls, tech, **overrides) -> ControllerTiming:
        """``tech``'s measured defaults (``tech.characterised.controller``), with ``overrides``;
        raises :class:`~gc_compiler.tech.characterised.NotCharacterised` if it has none."""
        measured = _measured(tech)
        base = {**vars(measured.controller), "interval_us": measured.unmeasured_refresh_us}
        return cls(**{**base, **overrides})

    @classmethod
    def from_characterisation(
        cls, data: dict, clock_ns: float, interval_us: float, margin: float = 0.5, handshake: bool = False
    ) -> ControllerTiming:
        """From a characterisation's ``to_dict()`` (``<name>.timing.json``): the slowest corner of
        each quantity. A self-timed macro's read lasts until it has cut itself off (the slowest
        cutoff, plus 20%): holding ``re`` longer is harmless, so one time serves every corner.
        Otherwise the read strobe is ``strobe_factor`` x ``rdy`` (the access time without a
        replica). The write pulse is three times the slowest write, at least a nanosecond. With
        ``handshake``, ``rdy_lag_ns`` is the latest ``dout`` after ``rdy`` of any read, plus 20%
        (at least 0.2 ns)."""
        corners = data["corners"]
        strobe = data.get("strobe_factor") or _measured().strobe_factor
        cutoff = [r["t_cutoff"] for c in corners for r in c.get("reads", []) if r.get("t_cutoff")]
        rdy = [r["t_rdy"] for c in corners for r in c.get("reads", []) if r.get("t_rdy")]
        if cutoff:
            read_s = 1.2 * max(cutoff)
        elif rdy:
            read_s = strobe * max(rdy)
        else:
            read_s = max(c["access_s"] for c in corners if c.get("access_s"))
        write_s = max(c["write_s"] for c in corners if c.get("write_s"))
        pre = [c["precharge_s"] for c in corners if c.get("precharge_s")]
        lags = [
            r["t_dout"] - r["t_rdy"] for c in corners for r in c.get("reads", []) if r.get("t_dout") and r.get("t_rdy")
        ]
        return cls(
            handshake=handshake,
            rdy_lag_ns=max(0.2, 1.2 * max(lags) * 1e9) if lags else _measured().controller.rdy_lag_ns,
            clock_ns=clock_ns,
            read_ns=read_s * 1e9,
            write_ns=max(3 * write_s * 1e9, 1.0),
            precharge_ns=1.2 * max(pre) * 1e9 if pre else 2.0,
            interval_us=interval_us,
            margin=margin,
        )


def default_refresh_interval_us(cfg: ArrayConfig, tech=None) -> float:
    """The refresh interval measured for ``cfg``'s cell on ``tech`` (the default technology): its
    ``characterised.refresh_intervals`` -- on sky130 the worst of 20 mismatch dies at the worst
    corner (ss, 1.62 V, 125 C) on the extracted 16x20 macro (``measurements/data/retention_mc_ss.json``,
    ``measurements/data/vneg.json``): 614 us for the default cell (the LVT write with its 0.5 um ``mr``),
    571 us for the LVT write with both read gates at 0.5 um, 88 us for the plain cell with a -0.3 V
    (or lower) write rail. Any other cell gets the shortest measured interval (45 us on sky130):
    measure it (:func:`~gc_compiler.verify.timing.macro_refresh_interval`) for a better one. A
    technology with no measurements raises
    :class:`~gc_compiler.tech.characterised.NotCharacterised` -- an interval from another process
    could lose data."""
    from gc_compiler.config import BitcellConfig

    if tech is None:
        from gc_compiler.tech import default_technology

        tech = default_technology()
    measured = _measured(tech)
    for entry in measured.refresh_intervals:
        if cfg.bitcell != BitcellConfig.for_tech(tech, **entry.bitcell):
            continue
        if entry.negative_rail is None and not cfg.negative_write_wl:
            return entry.microseconds
        if entry.negative_rail is not None and cfg.negative_write_wl and cfg.write_wl_low <= entry.negative_rail:
            return entry.microseconds
    return measured.unmeasured_refresh_us


def _cycles(ns: float, clock_ns: float) -> int:
    return max(1, math.ceil(ns / clock_ns - 1e-9))


@dataclass(frozen=True)
class ControllerParams:
    rows: int
    """Rows per bank."""
    banks: int
    addr_bits: int
    cols: int
    read_cycles: int
    write_cycles: int
    precharge_cycles: int
    refresh_cycles: int
    """Clock cycles between two refreshes (one row each)."""
    handshake: bool = False
    """Reads end on the synchronised ``rdy``; ``read_cycles`` is then the timeout."""
    refresh_in_macro: bool = False
    """The macro writes a refresh's data back itself (``ArrayConfig.refresh_in_macro``): a refresh
    drives every ``din`` high, and releases ``pch`` a cycle after ``we``."""


SYNC_CYCLES = 3
"""With the handshake, the edges from ``rdy`` rising to the data taken (at most: the edge that
may just miss it, and the synchroniser's two)."""


def controller_params(cfg: ArrayConfig, timing: ControllerTiming, banked: bool = False) -> ControllerParams:
    banks = 2 if banked else 1
    rows = banks * cfg.rows
    refresh = math.floor(timing.interval_us * 1e3 * timing.margin / rows / timing.clock_ns)
    if timing.handshake:
        if 2 * timing.clock_ns < timing.rdy_lag_ns + timing.sample_ns:
            raise ValueError(
                f"a {timing.clock_ns} ns clock is too fast for the handshake: the data is taken two edges "
                f"after rdy is caught, and needs {timing.rdy_lag_ns:.3g} ns after rdy plus {timing.sample_ns} ns"
            )
        reads = _cycles(timing.timeout_ns, timing.clock_ns) + SYNC_CYCLES
    else:
        reads = _cycles(timing.read_ns + timing.sample_ns, timing.clock_ns)
    # one refresh access must fit between two ticks, with room for the user
    per_refresh = (
        1 + reads + 1 + _cycles(timing.write_ns, timing.clock_ns) + _cycles(timing.precharge_ns, timing.clock_ns) + 1
        + (1 if cfg.refresh_in_macro else 0)  # pch released a cycle after we
    )
    if refresh < 2 * per_refresh:
        raise ValueError(
            f"a {timing.clock_ns} ns clock cannot refresh {rows} rows within "
            f"{timing.margin} x {timing.interval_us} us (a tick every {refresh} cycles, a refresh "
            f"takes {per_refresh})"
        )
    return ControllerParams(
        rows=cfg.rows, banks=banks,
        addr_bits=banked_address_bits(cfg.rows) if banked else address_bits(cfg.rows),
        cols=cfg.cols,
        read_cycles=reads,
        write_cycles=_cycles(timing.write_ns, timing.clock_ns),
        precharge_cycles=_cycles(timing.precharge_ns, timing.clock_ns),
        refresh_cycles=refresh,
        handshake=timing.handshake,
        refresh_in_macro=cfg.refresh_in_macro,
    )


def _width(n: int) -> int:
    return max(1, math.ceil(math.log2(n + 1)))


def controller(name: str, p: ControllerParams, timing: ControllerTiming) -> str:
    """The synchronous refresh controller (see the module docstring)."""
    if p.refresh_in_macro:
        # the macro writes the bit it just sensed back itself (its write driver is NAND(din, rbl), the
        # bit line held since the read): a refresh drives every din high; pch stays up until a
        # cycle after we falls, so the bit lines hold until the write word line has closed. Outside
        # refreshes and writes din is held low: the write bit lines then stay high through a user
        # read instead of following every read bit line down (and up again with the precharge). A
        # refresh keeps din high from its start: its write bit lines settle while the read still
        # drives the read bit lines (rising only once those float, a write bit line's fall couples
        # back into its read bit line through the NAND's gate and can turn a marginal '0' into a '1')
        loopback = (
            "    // the macro's write driver inverts din: a write stores ~din, a read returns what is stored.\n"
            "    // A refresh drives din high: the macro writes each column's sensed bit back itself.\n"
            "    // Otherwise din is low (NAND(0, rbl) holds each write bit line still through a read) but\n"
            "    // in a user write: din_q, a flop, opens it from the write's start until its precharge ends --\n"
            "    // steady while we falls -- and the data goes out already in the cycle that takes the write.\n"
            "    reg din_q;\n"
            "    wire din_on = din_q || (state == S_IDLE && !pending && req && we);\n"
            "    wire [COLS-1:0] macro_din = refreshing ? {COLS{1'b1}} : din_on ? ~din : {COLS{1'b0}};"
        )
        write_end = "                count <= refreshing ? PRECHARGE_CYCLES : PRECHARGE_CYCLES - 1;  // a refresh drops pch next cycle"
        pre_release = "\n                m_pch <= 1'b0;"
        din_open = "\n                        din_q <= 1'b1;"
        hold_open = ""
        din_close = "\n                din_q <= 1'b0;"
        din_reset = "\n            din_q <= 1'b0;"
    else:
        loopback = (
            "    // the macro's write driver inverts din: a write stores ~din, a read returns what is stored.\n"
            "    // Refresh loops the held read data back (din = ~dout rewrites each cell's own value).\n"
            "    wire [COLS-1:0] macro_din = refreshing ? ~macro_dout : ~din;"
        )
        write_end = "                m_pch <= 1'b0;\n                count <= PRECHARGE_CYCLES - 1;"
        pre_release = ""
        din_open = hold_open = din_close = din_reset = ""
    banked = p.banks == 2
    row_bits = address_bits(p.rows)
    cnt_w = _width(max(p.read_cycles, p.write_cycles, p.precharge_cycles))
    tick_w = _width(p.refresh_cycles)
    macro_data = (
        "    output [COLS-1:0] m_din_0, output [COLS-1:0] m_din_1,\n"
        "    input  [COLS-1:0] m_dout_0, input  [COLS-1:0] m_dout_1\n"
        if banked
        else "    output [COLS-1:0] m_din,\n    input  [COLS-1:0] m_dout\n"
    )
    hs = p.handshake
    if hs:
        macro_data = macro_data.rstrip("\n") + (",\n    input  m_rdy_0, input m_rdy_1\n" if banked else ",\n    input  m_rdy\n")
    bank = "m_addr[ADDR_BITS-1]"
    # the handshake: each replica's rdy through three flops (two to synchronise, one to see it
    # rise), reset with the rest; a read ends on the first rise of its bank's rdy
    rdy_banks = ["m_rdy_0", "m_rdy_1"] if banked else ["m_rdy"]
    sync = "".join(
        f"    reg [2:0] rdy_s{b};\n"
        f"    always @(posedge clk or negedge rst_q)\n"
        f"        if (!rst_q) rdy_s{b} <= 3'b000; else rdy_s{b} <= {{rdy_s{b}[1:0], {pin}}};\n"
        for b, pin in enumerate(rdy_banks)
    )
    rise = f"({bank} ? rdy_s1[1] & ~rdy_s1[2] : rdy_s0[1] & ~rdy_s0[2])" if banked else "(rdy_s0[1] & ~rdy_s0[2])"
    sync += f"    wire rdy_rise = {rise};\n"
    # rdy's rise one edge ahead (what rdy_rise will be after the next edge), for the dout latch's
    # registered enable; a user read's bank straight from the user's address
    next_rise = rise.replace("[1]", "[0]").replace("[2]", "[1]").replace(bank, "addr[ADDR_BITS-1]")
    sync += f"    wire next_rdy_rise = {next_rise};\n"
    last_soon = " || next_rdy_rise" if hs else ""
    read_step = (
        "            S_READ: if (!rdy_rise && count != 0) begin\n"
        "                count <= count - 1'b1;\n"
        f"                dout_open_q <= !refreshing && (count == 1{last_soon});\n"
        "            end else begin\n"
        "                // rdy rose two edges ago: the data is there (or no rdy came: flag it and go on)\n"
        "                if (!rdy_rise) read_timeout <= 1'b1;\n"
        if hs
        else "            S_READ: if (count != 0) begin\n"
        "                count <= count - 1'b1;\n"
        "                dout_open_q <= !refreshing && count == 1;\n"
        "            end else begin\n"
    )
    dout_sel = f"({bank} ? m_dout_1 : m_dout_0)" if banked else "m_dout"
    din_drive = (
        "    assign m_din_0 = macro_din;\n    assign m_din_1 = macro_din;\n" if banked else "    assign m_din = macro_din;\n"
    )
    next_row = (
        f"            if (ref_row == ROWS - 1) begin ref_row <= 0; ref_bank <= ~ref_bank; end\n"
        f"            else ref_row <= ref_row + 1'b1;\n"
        if banked
        else "            ref_row <= (ref_row == ROWS - 1) ? 0 : ref_row + 1'b1;\n"
    )
    ref_addr = (
        f"{{ref_bank, ref_row}}" if banked else "ref_row"
    )
    return f"""// Refresh controller for {name} (generated by gc_compiler.rtl).
// Clock {timing.clock_ns:g} ns; read {f"on rdy (timeout {timing.timeout_ns:g} ns)" if hs else f"{timing.read_ns:.3g} ns"}, write {timing.write_ns:.3g} ns,
// precharge {timing.precharge_ns:.3g} ns; every row refreshed within {timing.margin:g} x {timing.interval_us:g} us.
`timescale 1ns/1ps
module {name}_ctrl #(
    parameter ADDR_BITS = {p.addr_bits},
    parameter COLS = {p.cols},
    parameter ROWS = {p.rows},              // rows per bank
    parameter READ_CYCLES = {p.read_cycles},
    parameter WRITE_CYCLES = {p.write_cycles},
    parameter PRECHARGE_CYCLES = {p.precharge_cycles},
    parameter REFRESH_CYCLES = {p.refresh_cycles}
) (
    input  clk,
    input  rst_n,
    // user side: a request is taken on a clock edge where busy is low
    input  req,
    input  we,
    input  [ADDR_BITS-1:0] addr,
    input  [COLS-1:0] din,
    output reg [COLS-1:0] dout,
    output reg dout_valid,
    output busy,
{"    output reg read_timeout,       // sticky: a read whose rdy never came" + chr(10) if hs else ""}    // macro side
    output reg m_re,
    output reg m_we,
    output reg m_pch,
    output [ADDR_BITS-1:0] m_addr,
{macro_data});
    localparam S_IDLE = 3'd0, S_READ = 3'd1, S_HOLD = 3'd2, S_WRITE = 3'd3, S_PRE = 3'd4, S_ARM = 3'd5;
    reg [2:0] state;
    reg [{cnt_w - 1}:0] count;
    reg [{tick_w - 1}:0] tick;
    reg pending, refreshing, writing;
    reg [{row_bits - 1}:0] ref_row;
{"    reg ref_bank;" if banked else ""}
    // the user's address and data go to the macro as they are (held until busy falls); a refresh
    // switches the address to its row a cycle before re rises (S_ARM), and back after it ends
    assign m_addr = refreshing ? {ref_addr} : addr;
    wire [COLS-1:0] macro_dout = {dout_sel};
{loopback}
{din_drive}
    // reset: asserted at once, released on the clock (two flops), so no flop sees rst_n's
    // release close to an edge; busy until then
    reg [1:0] rst_sync;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_sync <= 2'b00;
        else rst_sync <= {{rst_sync[0], 1'b1}};
    end
    wire rst_q = rst_sync[1];
    assign busy = !rst_q || (state != S_IDLE) || pending;
{sync if hs else ""}
    always @(posedge clk or negedge rst_q) begin
        if (!rst_q) begin
            tick <= 0;
            pending <= 1'b0;
        end else begin
            // a tick arriving as a refresh starts is the next row's, so it wins
            if (tick == REFRESH_CYCLES - 1) begin
                tick <= 0;
                pending <= 1'b1;
            end else begin
                tick <= tick + 1'b1;
                if (state == S_IDLE && pending) pending <= 1'b0;
            end
        end
    end

    // dout: a latch, open while clk is low in a user read's last cycle and closing on the edge that
    // ends the read -- where a flop would take the data (the macro's dout holds until pch falls,
    // after that edge); it then holds until the next read. Its enable is a flop, which changes only
    // after a rising edge, while clk is high: it cannot glitch open, and it has half a cycle.
    reg dout_open_q;  // high through a user read's last cycle (set by the edge that starts it)
    wire dout_open = !clk && dout_open_q;
    // verilator lint_off LATCH
    always @(*) if (dout_open) dout = macro_dout;  // intended
    // verilator lint_on LATCH

    always @(posedge clk or negedge rst_q) begin
        if (!rst_q) begin
            state <= S_IDLE;
            count <= 0;
            refreshing <= 1'b0;
            writing <= 1'b0;
            ref_row <= 0;
{"            ref_bank <= 1'b0;" if banked else ""}
            dout_valid <= 1'b0;
            dout_open_q <= 1'b0;{din_reset}
{"            read_timeout <= 1'b0;" + chr(10) if hs else ""}            m_re <= 1'b0;
            m_we <= 1'b0;
            m_pch <= 1'b0;
        end else begin
            dout_valid <= 1'b0;
            dout_open_q <= 1'b0;
            case (state)
            S_IDLE: begin
                if (pending) begin
                    // the refresh row onto the address a cycle ahead of re
                    refreshing <= 1'b1;
                    writing <= 1'b0;
                    state <= S_ARM;
                end else if (req) begin
                    refreshing <= 1'b0;
                    writing <= we;
                    if (we) begin
                        m_we <= 1'b1;{din_open}
                        count <= WRITE_CYCLES - 1;
                        state <= S_WRITE;
                    end else begin
                        m_re <= 1'b1;
                        m_pch <= 1'b1;
                        count <= READ_CYCLES - 1;
                        dout_open_q <= READ_CYCLES == 1{last_soon};
                        state <= S_READ;
                    end
                end
            end
{read_step}                // the read is over (the macro cut its word line off itself); precharge still off, so the bit lines hold
                m_re <= 1'b0;
                if (refreshing) begin
                    state <= S_HOLD;{hold_open}
                end
                else begin
                    dout_valid <= 1'b1;
                    m_pch <= 1'b0;
                    count <= PRECHARGE_CYCLES - 1;
                    state <= S_PRE;
                end
            end
            S_ARM: begin
                m_re <= 1'b1;
                m_pch <= 1'b1;
                count <= READ_CYCLES - 1;
                state <= S_READ;
            end
            S_HOLD: begin  // one cycle between the read word line falling and the write's rising
                m_we <= 1'b1;
                count <= WRITE_CYCLES - 1;
                state <= S_WRITE;
            end
            S_WRITE: if (count != 0) count <= count - 1'b1; else begin
                m_we <= 1'b0;
{write_end}
                state <= S_PRE;
            end
            S_PRE: if (count != 0) begin
                count <= count - 1'b1;{pre_release}
            end else begin
                // (the next refresh row only now: re and we are down, so the address may move)
                if (refreshing) begin
{next_row}                end
                refreshing <= 1'b0;{din_close}
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
"""


def _sdc(sdc):
    """The standard-cell library's clock and I/O assumptions (the default technology's by default)."""
    if sdc is None:
        from gc_compiler.tech import default_technology

        sdc = default_technology().standard_cells().sdc
    return sdc


def controller_sdc(params: ControllerParams, timing: ControllerTiming, io_fraction: float = 0.2, sdc=None) -> str:
    """Timing constraints for ``<name>_ctrl`` (``<name>_ctrl.sdc``; LibreLane reads it for place and
    route and for sign-off).

    The user side gets ``io_fraction`` of the clock as input and output delay. The macro side
    follows the controller's own protocol: a read's ``m_dout`` is valid ``read_ns`` after the edge
    that raised ``m_re`` and is sampled ``READ_CYCLES`` edges later; the refresh loop-back
    (``m_dout`` to ``m_din``) has one edge more, since ``m_we`` rises only after the hold cycle.
    ``rst_n`` only reaches the reset synchroniser, whose flops are asynchronous to it.

    With the handshake the read has no fixed length: ``m_rdy`` is asynchronous (it only reaches
    its synchroniser), and ``m_dout`` is timed from the edge that catches ``rdy`` -- valid
    ``rdy_lag_ns`` after it, and taken two edges later (three for the loop-back).
    """
    period = timing.clock_ns
    io = io_fraction * period
    sdc = _sdc(sdc)
    reads = SYNC_CYCLES - 1 if params.handshake else params.read_cycles
    dout_max = timing.rdy_lag_ns if params.handshake else timing.read_ns
    header = (
        f"macro read on rdy, data {timing.rdy_lag_ns:.3g} ns after it" if params.handshake
        else f"macro read {timing.read_ns:.3g} ns in {reads} cycle(s)"
    )
    loopback = "" if params.refresh_in_macro else (
        f"set_multicycle_path -setup {reads + 1} -from [get_ports m_dout*] -to [get_ports m_din*]\n"
        f"set_multicycle_path -hold {reads} -from [get_ports m_dout*] -to [get_ports m_din*]\n"
    )
    return f"""# {timing.clock_ns:g} ns clock; {header} (generated by gc_compiler.rtl)
create_clock -name clk -period {period:g} [get_ports clk]
set_clock_uncertainty -setup {sdc.setup_uncertainty:g} [get_clocks clk]
set_clock_uncertainty -hold {sdc.hold_uncertainty:g} [get_clocks clk]
set_clock_transition {sdc.clock_transition:g} [get_clocks clk]
set_timing_derate -early {sdc.derate_early:g}
set_timing_derate -late {sdc.derate_late:g}
set_input_transition {sdc.input_transition:g} [all_inputs]
set_load {sdc.output_load:g} [all_outputs]
set_input_delay {io:g} -clock clk [get_ports {{req we addr* din*}}]
set_output_delay {io:g} -clock clk [get_ports {{dout* dout_valid busy{" read_timeout" if params.handshake else ""}}}]
set_false_path -from [get_ports rst_n]
set_output_delay 0 -clock clk [get_ports {{m_re m_we m_pch m_addr* m_din*}}]
{"set_false_path -from [get_ports m_rdy*]" + chr(10) if params.handshake else ""}set_input_delay -max {dout_max:g} -clock clk [get_ports m_dout*]
set_input_delay -min {timing.macro_min_ns:g} -clock clk [get_ports m_dout*]
set_multicycle_path -setup {reads} -from [get_ports m_dout*]
set_multicycle_path -hold {reads - 1} -from [get_ports m_dout*]
{loopback}# dout is a latch, open in the second half of a read's last cycle: it is taken by the user an edge
# later, with dout_valid (as a flop's would be), not on the edge the latch closes
set_multicycle_path -setup 2 -from [all_registers -level_sensitive]
set_multicycle_path -hold 1 -from [all_registers -level_sensitive]
if {{ [info exists ::env(OPENLANE_SDC_IDEAL_CLOCKS)] && $::env(OPENLANE_SDC_IDEAL_CLOCKS) }} {{
    unset_propagated_clock [all_clocks]
}} else {{
    set_propagated_clock [all_clocks]
}}
"""


def top_sdc(
    params: ControllerParams, timing: ControllerTiming, instance: str = "macro", io_fraction: float = 0.2, sdc=None
) -> str:
    """Timing constraints for ``<name>_top`` built as one block with the macro (the macro placed
    as a hard macro called ``instance``, its timing from a library whose read is an arc from
    ``re`` to every ``dout``): the user side as in :func:`controller_sdc`; a read through the
    macro is sampled ``READ_CYCLES`` edges after ``re`` rises, and the refresh loop-back reaches
    the macro's ``din`` within one edge more.

    With the handshake the macro's ``rdy`` only reaches the synchroniser (a false path), and a
    read is sampled two edges after the edge that catches ``rdy``: from ``re``, the library's read
    rounded up to whole cycles, plus two. (The library gives ``dout`` and ``rdy`` the same arc, so
    this cannot see how far ``dout`` trails ``rdy``: :func:`controller_params` holds that lag to
    two cycles, from the characterisation.)"""
    period = timing.clock_ns
    io = io_fraction * period
    sdc = _sdc(sdc)
    if params.handshake:
        reads = _cycles(timing.read_ns, period) + SYNC_CYCLES - 1
        header = f"macro read on rdy, sampled {reads} edges after re at the library's {timing.read_ns:.3g} ns"
    else:
        reads = params.read_cycles
        header = f"macro read {timing.read_ns:.3g} ns in {reads} cycle(s)"
    loopback_note = "" if params.refresh_in_macro else (
        "# the refresh loop-back (dout through the controller back to din, before we rises (reads + 1)\n"
        "# edges after re) ends on the macro's din pins, which have no timing check to constrain:\n"
        "# build_memory measures it on the routed block instead (flow.sta.loopback_arrival)\n"
    )
    return f"""# {timing.clock_ns:g} ns clock; {header} (generated by gc_compiler.rtl)
create_clock -name clk -period {period:g} [get_ports clk]
set_clock_uncertainty -setup {sdc.setup_uncertainty:g} [get_clocks clk]
set_clock_uncertainty -hold {sdc.hold_uncertainty:g} [get_clocks clk]
set_clock_transition {sdc.clock_transition:g} [get_clocks clk]
set_timing_derate -early {sdc.derate_early:g}
set_timing_derate -late {sdc.derate_late:g}
set_input_transition {sdc.input_transition:g} [all_inputs]
set_load {sdc.output_load:g} [all_outputs]
set_input_delay {io:g} -clock clk [get_ports {{req we addr* din*}}]
set_output_delay {io:g} -clock clk [get_ports {{dout* dout_valid busy{" read_timeout" if params.handshake else ""}}}]
set_false_path -from [get_ports rst_n]
{f"set_false_path -through [get_pins {instance}/rdy*]" + chr(10) if params.handshake else ""}set_multicycle_path -setup {reads} -through [get_pins {instance}/dout*]
set_multicycle_path -hold {reads - 1} -through [get_pins {instance}/dout*]
{loopback_note}# dout is a latch, open in the second half of a read's last cycle: it is taken by the user an edge
# later, with dout_valid (as a flop's would be), not on the edge the latch closes
set_multicycle_path -setup 2 -from [all_registers -level_sensitive]
set_multicycle_path -hold 1 -from [all_registers -level_sensitive]
if {{ [info exists ::env(OPENLANE_SDC_IDEAL_CLOCKS)] && $::env(OPENLANE_SDC_IDEAL_CLOCKS) }} {{
    unset_propagated_clock [all_clocks]
}} else {{
    set_propagated_clock [all_clocks]
}}
"""


def _data_pins(cfg: ArrayConfig, banked: bool) -> tuple[list[str], list[str]]:
    if banked:
        return [f"din_{b}_{c}" for b in (0, 1) for c in range(cfg.cols)], [f"dout_{b}_{c}" for b in (0, 1) for c in range(cfg.cols)]
    return [f"din_{c}" for c in range(cfg.cols)], [f"dout_{c}" for c in range(cfg.cols)]


_SUPPLIES = ("vdd", "vss", "vnw", "vnw_0", "vnw_1", "vneg")


def macro_model(name: str, cfg: ArrayConfig, banked: bool = False, retention_ns: float = 40_000.0) -> str:
    """A behavioural model of the macro ``name`` (its pins without the supplies): what it stores,
    what it reads, and when a row's data has leaked away. A read's data reaches ``dout``
    ``DOUT_NS`` after ``re`` rises, and the bank's replica raises ``rdy`` ``RDY_NS`` after it; both
    hold while ``pch`` stays up and fall with it."""
    pins = [p for p in macro_pins(cfg, True, banked) if p not in _SUPPLIES]
    bits = banked_address_bits(cfg.rows) if banked else address_bits(cfg.rows)
    banks = 2 if banked else 1
    din, dout = _data_pins(cfg, banked)
    rdy = [p for p in pins if p.startswith("rdy")]
    outputs = set(dout) | set(rdy)
    decl = ",\n    ".join(("output reg " if p in outputs else "input ") + p for p in pins)
    addr = "{" + ", ".join(f"addr_{i}" for i in reversed(range(bits))) + "}"
    lines = [
        f"// Behavioural model of {name} (generated by gc_compiler.rtl): the macro's pins, no supplies.",
        "`timescale 1ns/1ps",
        f"module {name} #(parameter RETENTION_NS = {retention_ns:g}, parameter DOUT_NS = 0, parameter RDY_NS = 0) (",
        f"    {decl}",
        ");",
        f"    localparam COLS = {cfg.cols}, BANKS = {banks}, WORDS = {banks * 2 ** address_bits(cfg.rows)};",
        f"    wire [{bits - 1}:0] a = {addr};",
        "    reg [COLS-1:0] mem [0:WORDS-1];",
        "    realtime written [0:WORDS-1];",
        "    reg [COLS-1:0] held [0:BANKS-1];",
        "    integer i;",
        "    initial for (i = 0; i < WORDS; i = i + 1) begin mem[i] = {COLS{1'bx}}; written[i] = 0; end",
        "    initial for (i = 0; i < BANKS; i = i + 1) held[i] = 0;",
        *(f"    initial {p} = 1'b0;" for p in rdy),
    ]
    for b in range(banks):
        bank_pins = din[b * cfg.cols:(b + 1) * cfg.cols]
        vec = "{" + ", ".join(reversed(bank_pins)) + "}"
        sel = f" && a[{bits - 1}] == 1'b{b}" if banked else ""
        lines += [
            f"    wire [COLS-1:0] din{b} = {vec};",
            *([
                f"    // a write stores ~(din & rbl): ~din with the bit line precharged; with pch up after a",
                f"    // read, the sensed bit wherever din is high (the macro's in-place refresh); taken as we falls",
                f"    always @(negedge we) if (1{sel}) begin mem[a] = ~din{b} | (pch ? held[{b}] : {{COLS{{1'b0}}}}); written[a] = $realtime; end",
            ] if cfg.refresh_in_macro else [
                f"    // a write stores ~din (din low writes a '1'); taken as we falls",
                f"    always @(negedge we) if (1{sel}) begin mem[a] = ~din{b}; written[a] = $realtime; end",
            ]),
        ]
    lines += [
        "    // a read: the row reaches the bank's bit lines DOUT_NS after re rises and its replica fires",
        "    // RDY_NS after it; both hold while pch stays up (the macro ends the read itself)",
        "    wire #(DOUT_NS) re_dout = re;",
        "    wire #(RDY_NS) re_rdy = re;",
        "    always @(*) begin",
    ]
    for b in range(banks):
        sel = f"a[{bits - 1}] == 1'b{b}" if banked else "1"
        lines += [
            f"        if (re && re_dout && ({sel}))",
            f"            held[{b}] = ($realtime - written[a] > RETENTION_NS) ? {{COLS{{1'bx}}}} : mem[a];",
            f"        else if (!pch) held[{b}] = {{COLS{{1'b0}}}};",
        ]
        if rdy:
            lines += [
                f"        if (re && re_rdy && ({sel})) {rdy[b]} = 1'b1;",
                f"        else if (!pch) {rdy[b]} = 1'b0;",
            ]
    lines.append("    end")
    for b in range(banks):
        for c in range(cfg.cols):
            lines.append(f"    always @(*) {dout[b * cfg.cols + c]} = held[{b}][{c}];")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def top(name: str, cfg: ArrayConfig, p: ControllerParams, banked: bool = False) -> str:
    """``<name>_top``: the controller wired to the macro (the model in simulation)."""
    pins = [q for q in macro_pins(cfg, True, banked) if q not in _SUPPLIES]
    din, dout = _data_pins(cfg, banked)
    conn = []
    for q in pins:
        if q.startswith("addr_"):
            conn.append(f".{q}(m_addr[{q.split('_')[1]}])")
        elif q in ("we", "re", "pch"):
            conn.append(f".{q}(m_{q})")
        elif q in din or q in dout:
            signal, *rest = q.split("_")
            b, c = (rest[0], rest[1]) if banked else (None, rest[0])
            vec = f"m_{signal}" + (f"_{b}" if banked else "")
            conn.append(f".{q}({vec}[{c}])")
        elif q.startswith("rdy"):
            conn.append(f".{q}({'m_' + q if p.handshake else ''})")
    rdy = [q for q in pins if q.startswith("rdy")]
    if p.handshake and not rdy:
        raise ValueError("the handshake ends reads on the replica's rdy: build the macro with a replica column")
    wires = (
        "    wire [COLS-1:0] m_din_0, m_din_1, m_dout_0, m_dout_1;\n" if banked else "    wire [COLS-1:0] m_din, m_dout;\n"
    )
    data = (
        ".m_din_0(m_din_0), .m_din_1(m_din_1), .m_dout_0(m_dout_0), .m_dout_1(m_dout_1)"
        if banked
        else ".m_din(m_din), .m_dout(m_dout)"
    )
    timeout_port = timeout_conn = ""
    if p.handshake:
        wires += f"    wire {', '.join('m_' + q for q in rdy)};\n"
        data += "".join(f", .m_{q}(m_{q})" for q in rdy)
        timeout_port, timeout_conn = ", output read_timeout", ", .read_timeout(read_timeout)"
    return f"""// {name} with its refresh controller (generated by gc_compiler.rtl).
`timescale 1ns/1ps
module {name}_top #(parameter ADDR_BITS = {p.addr_bits}, parameter COLS = {p.cols}) (
    input clk, input rst_n,
    input req, input we, input [ADDR_BITS-1:0] addr, input [COLS-1:0] din,
    output [COLS-1:0] dout, output dout_valid, output busy{timeout_port}
);
    wire m_re, m_we, m_pch;
    wire [ADDR_BITS-1:0] m_addr;
{wires}    {name}_ctrl ctrl (
        .clk(clk), .rst_n(rst_n), .req(req), .we(we), .addr(addr), .din(din),
        .dout(dout), .dout_valid(dout_valid), .busy(busy){timeout_conn},
        .m_re(m_re), .m_we(m_we), .m_pch(m_pch), .m_addr(m_addr), {data}
    );
    {name} macro (
        {", ".join(conn)}
    );
endmodule
"""


def testbench(name: str, cfg: ArrayConfig, p: ControllerParams, timing: ControllerTiming, banked: bool = False,
              intervals: float = 1.0, idle: float = 2.0, seed: int = 1) -> str:
    """A self-checking testbench of ``<name>_top``: every row written, ``intervals`` refresh
    intervals of random reads and writes against a golden copy, then ``idle`` intervals with no
    user access at all -- only refresh keeps the data then -- and a read of every row. Any
    mismatch or ``x`` (a row not refreshed in time) fails. Prints ``PASS`` or ``FAIL``. Each
    request's ``we``, ``addr`` and ``din`` hold until ``busy`` falls, and wander between requests.

    With the handshake the model's replica fires only halfway to the timeout -- several cycles,
    far later than any ``read_ns`` -- and the data trails it by ``rdy_lag_ns``: the reads must
    wait for ``rdy``, and none may time out."""
    row_bits = address_bits(p.rows)
    slow = ""
    if p.handshake:
        rdy_ns = timing.timeout_ns / 2
        slow = (
            f"\n    defparam dut.macro.RDY_NS = {rdy_ns:g};"
            f"\n    defparam dut.macro.DOUT_NS = {rdy_ns + timing.rdy_lag_ns:g};"
        )
    words = p.banks * 2 ** row_bits  # the address space (a bank's rows need not fill it)
    duration = intervals * timing.interval_us * 1e3
    quiet = idle * timing.interval_us * 1e3
    busy_limit = p.read_cycles + 1 + p.write_cycles + p.precharge_cycles + 2
    address = (
        f"{{bank, row[{row_bits - 1}:0]}}" if banked else f"row[{row_bits - 1}:0]"
    )
    return f"""`timescale 1ns/1ps
module tb;
    localparam ADDR_BITS = {p.addr_bits}, COLS = {p.cols}, ROWS = {p.rows}, WORDS = {words};
    reg clk = 0, rst_n = 0, req = 0, we = 0;
    reg [ADDR_BITS-1:0] addr = 0;
    reg [COLS-1:0] din = 0;
    wire [COLS-1:0] dout;
    wire dout_valid, busy{", read_timeout" if p.handshake else ""};
    {name}_top dut (.clk(clk), .rst_n(rst_n), .req(req), .we(we), .addr(addr), .din(din),
        .dout(dout), .dout_valid(dout_valid), .busy(busy){", .read_timeout(read_timeout)" if p.handshake else ""});
    defparam dut.macro.RETENTION_NS = {timing.interval_us * 1e3:g};{slow}
    always #({timing.clock_ns / 2:g}) clk = ~clk;

    reg [COLS-1:0] golden [0:WORDS-1];
    reg [ADDR_BITS-1:0] expect_addr;
    integer errors = 0, reads = 0, writes = 0, busy_run = 0, busy_max = 0, i, row, bank, seed = {seed};

    always @(posedge clk) if (rst_n) begin
        busy_run = busy ? busy_run + 1 : 0;
        if (busy_run > busy_max) busy_max = busy_run;
        if (dout_valid) begin
            reads = reads + 1;
            if (dout !== golden[expect_addr]) begin
                errors = errors + 1;
                if (errors < 5) $display("read %0d: got %h expected %h at %0t", expect_addr, dout, golden[expect_addr], $time);
            end
        end
    end

    task access(input w, input [ADDR_BITS-1:0] a, input [COLS-1:0] d);
        begin
            @(negedge clk);
            req = 1; we = w; addr = a; din = d;
            @(posedge clk);
            while (busy) @(posedge clk);
            @(negedge clk);
            req = 0;
            if (w) begin golden[a] = d; writes = writes + 1; end
            else expect_addr = a;
            // wait for the access to finish (we, addr and din held until then)
            @(posedge clk);
            while (busy) @(posedge clk);
            // then let them wander: with no request, nothing may depend on them (a refresh uses its own row)
            @(negedge clk);
            we = $random(seed); addr = $random(seed); din = $random(seed);
        end
    endtask

    initial begin
        #({5 * timing.clock_ns:g}) rst_n = 1;
        // fill every row once
        for (bank = 0; bank < {p.banks}; bank = bank + 1)
            for (row = 0; row < ROWS; row = row + 1)
                access(1, {address}, $random(seed));
        // then random traffic until {intervals:g} refresh intervals have passed
        while ($time < {duration:.0f}) begin
            bank = {"$random(seed) & 1" if banked else "0"};
            row = ($random(seed) & 32'h7fffffff) % ROWS;
            if ($random(seed) & 1) access(1, {address}, $random(seed));
            else access(0, {address}, 0);
            repeat (($random(seed) & 32'hf)) @(posedge clk);
        end
        // no user access for {idle:g} intervals: only refresh keeps the data
        #({quiet:.0f});
        // a last read of every row: all must still hold their data
        for (bank = 0; bank < {p.banks}; bank = bank + 1)
            for (row = 0; row < ROWS; row = row + 1)
                access(0, {address}, 0);
        if (busy_max > {busy_limit} + {p.write_cycles + p.precharge_cycles + p.read_cycles + 2})
            begin errors = errors + 1; $display("busy held %0d cycles", busy_max); end
{"        if (read_timeout !== 1'b0) begin errors = errors + 1; $display(" + '"a read timed out"' + "); end" + chr(10) if p.handshake else ""}        $display("reads %0d writes %0d busy_max %0d errors %0d", reads, writes, busy_max, errors);
        if (errors == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end
endmodule
"""


def refresh_rtl(
    name: str, cfg: ArrayConfig, timing: ControllerTiming, banked: bool = False, tech=None
) -> dict[str, str]:
    """Every file for a macro named ``name``: ``{filename: text}`` (the constraints with ``tech``'s
    standard-cell library's clock assumptions; the default technology's without)."""
    p = controller_params(cfg, timing, banked)
    return {
        f"{name}_ctrl.v": controller(name, p, timing),
        f"{name}_model.v": macro_model(name, cfg, banked, timing.interval_us * 1e3),
        f"{name}_top.v": top(name, cfg, p, banked),
        f"{name}_tb.v": testbench(name, cfg, p, timing, banked),
        f"{name}_ctrl.sdc": controller_sdc(p, timing, sdc=tech.standard_cells().sdc if tech is not None else None),
    }
