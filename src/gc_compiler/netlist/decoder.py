"""Static NAND/inverter address decoder: turns an address bus into per-row word-line inputs.

The row gate is a single 2-input NAND per output, which is what lets the physical decoder
match the word-line pitch (see :mod:`gc_compiler.periphery.decoder`). The address is split
into a low and a high part and both are predecoded to one-hot *bus lines* that run past every
row: ``L`` lines from the low bits, and ``H`` lines from the high bits already ANDed with an
enable -- one set with ``we`` and one with ``re``. Row ``r`` is then

    rwl_in_r = NAND(L[low(r)], Hre[high(r)])           active-low
    wwl_in_r = INV(NAND(L[low(r)], Hwe[high(r)]))      active-high

so the word-line drivers (which invert) select the NMOS read gate with a high word line and
the PMOS write gate with a low one. Idle rows drive ``rwl_in`` high and ``wwl_in`` low.

Decoding a group of address bits is one NOR over the bits' *complement* literals (an AND
is a NOR of inverted inputs), or for more than three bits NANDs of groups feeding a NOR --
so a bus line comes straight out of a gate, with no inverter behind it. Every line is
shared between the rows that need it and only the lines the requested outputs use are built.
The low/high split is whichever needs the fewest gates.

A *banked* decoder serves two identical sub-arrays from one predecode. The bank is one more
address bit, above the row bits, so bank ``b``'s row ``r`` is decoded as the *logical* row
``b * 2**address_bits(rows) + r`` -- the bank bit lands in the high part and is folded into
the ``H`` lines with the enable, exactly like the row bits, and nothing else changes. Its
outputs are named ``{signal}_{bank}_{row}``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gc_compiler.config import ArrayConfig
from gc_compiler.periphery.logic import LogicConfig
from gc_compiler.tech.rules import DeviceKind, Technology

NAND2_PINS = ("a", "b", "y", "vdd", "vss")
NAND3_PINS = ("a", "b", "c", "y", "vdd", "vss")
NOR2_PINS = NAND2_PINS
NOR3_PINS = NAND3_PINS


def address_bits(rows: int) -> int:
    """Number of address lines needed to select one of ``rows`` rows."""
    return max(1, math.ceil(math.log2(rows)))


def all_row_outputs(rows: int) -> list[tuple[int, str]]:
    """Both word lines of every row -- the full, non-split decoder."""
    return [(r, sig) for r in range(rows) for sig in ("wwl_in", "rwl_in")]


def side_outputs(rows: int) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Split outputs the way the array actually lands them: each row has one word line on
    the left strip and the other on the right (see periphery.wl_driver.LEFT_WORDLINES).
    """
    left = [(r, "rwl_in" if r % 2 == 0 else "wwl_in") for r in range(rows)]
    right = [(r, "wwl_in" if r % 2 == 0 else "rwl_in") for r in range(rows)]
    return left, right


def banked_address_bits(rows: int) -> int:
    """Address lines of a two-bank decoder: the row bits plus the bank bit on top."""
    return address_bits(rows) + 1


def logical_row(rows: int, bank: int, row: int) -> int:
    """The row number a banked decoder decodes for ``bank``'s ``row``."""
    return (bank << address_bits(rows)) | row


def bank_and_row(rows: int, logical: int) -> tuple[int, int]:
    return logical >> address_bits(rows), logical & ((1 << address_bits(rows)) - 1)


def output_net(signal: str, row: int, banked_rows: int | None = None) -> str:
    """Net name of an output: ``{signal}_{row}``, or ``{signal}_{bank}_{row}`` for a banked
    decoder of ``banked_rows`` rows per bank (``row`` is then the logical row)."""
    if banked_rows is None:
        return f"{signal}_{row}"
    bank, r = bank_and_row(banked_rows, row)
    return f"{signal}_{bank}_{r}"


def banked_outputs(rows: int) -> list[tuple[int, str]]:
    """Both word lines of every row of both banks, as logical rows."""
    return [(logical_row(rows, bank, r), sig) for bank in (0, 1) for r, sig in all_row_outputs(rows)]


def banked_face_outputs(rows: int, bank: int) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """``(near, far)`` outputs of one bank, as logical rows.

    A bank is laid out like a plain macro (left driver column, array, right driver column)
    and the central decoder sits on bank 0's right and bank 1's left. So bank 0's *near*
    word lines are the ones its right column lands and its *far* ones go to its left
    column; for bank 1 it is the other way round (see :func:`side_outputs`).
    """
    left, right = side_outputs(rows)
    near, far = (right, left) if bank == 0 else (left, right)
    return (
        [(logical_row(rows, bank, r), sig) for r, sig in near],
        [(logical_row(rows, bank, r), sig) for r, sig in far],
    )


def nand2_subckt(tech: Technology, cfg: LogicConfig, name: str = "gc_nand2") -> str:
    return "\n".join(
        [
            f".subckt {name} {' '.join(NAND2_PINS)}",
            tech.spice_device(
                "mpa", "y", "a", "vdd", "vdd", DeviceKind.PMOS, cfg.pmos_width, cfg.length
            ),
            tech.spice_device(
                "mpb", "y", "b", "vdd", "vdd", DeviceKind.PMOS, cfg.pmos_width, cfg.length
            ),
            tech.spice_device(
                "mna", "y", "a", "mid", "vss", DeviceKind.NMOS, cfg.nmos_width, cfg.length
            ),
            tech.spice_device(
                "mnb", "mid", "b", "vss", "vss", DeviceKind.NMOS, cfg.nmos_width, cfg.length
            ),
            ".ends",
        ]
    )


def nand3_subckt(tech: Technology, cfg: LogicConfig, name: str = "gc_nand3") -> str:
    """Three-input NAND; ``a`` gates the NMOS next to ``y``, then ``b``, then ``c`` next to vss."""
    return "\n".join(
        [
            f".subckt {name} {' '.join(NAND3_PINS)}",
            *(
                tech.spice_device(
                    f"mp{pin}", "y", pin, "vdd", "vdd", DeviceKind.PMOS, cfg.pmos_width, cfg.length
                )
                for pin in "abc"
            ),
            tech.spice_device("mna", "y", "a", "mid1", "vss", DeviceKind.NMOS, cfg.nmos_width, cfg.length),
            tech.spice_device("mnb", "mid1", "b", "mid2", "vss", DeviceKind.NMOS, cfg.nmos_width, cfg.length),
            tech.spice_device("mnc", "mid2", "c", "vss", "vss", DeviceKind.NMOS, cfg.nmos_width, cfg.length),
            ".ends",
        ]
    )


def nor2_subckt(tech: Technology, cfg: LogicConfig, name: str = "gc_nor2") -> str:
    """Two-input NOR; ``a`` gates the PMOS next to ``y``, ``b`` the one next to vdd."""
    return "\n".join(
        [
            f".subckt {name} {' '.join(NOR2_PINS)}",
            tech.spice_device("mpa", "y", "a", "mid", "vdd", DeviceKind.PMOS, cfg.pmos_width, cfg.length),
            tech.spice_device("mpb", "mid", "b", "vdd", "vdd", DeviceKind.PMOS, cfg.pmos_width, cfg.length),
            tech.spice_device("mna", "y", "a", "vss", "vss", DeviceKind.NMOS, cfg.nmos_width, cfg.length),
            tech.spice_device("mnb", "y", "b", "vss", "vss", DeviceKind.NMOS, cfg.nmos_width, cfg.length),
            ".ends",
        ]
    )


def nor3_subckt(tech: Technology, cfg: LogicConfig, name: str = "gc_nor3") -> str:
    """Three-input NOR; ``a`` gates the PMOS next to ``y``, then ``b``, then ``c`` next to vdd."""
    return "\n".join(
        [
            f".subckt {name} {' '.join(NOR3_PINS)}",
            tech.spice_device("mpa", "y", "a", "mid1", "vdd", DeviceKind.PMOS, cfg.pmos_width, cfg.length),
            tech.spice_device("mpb", "mid1", "b", "mid2", "vdd", DeviceKind.PMOS, cfg.pmos_width, cfg.length),
            tech.spice_device("mpc", "mid2", "c", "vdd", "vdd", DeviceKind.PMOS, cfg.pmos_width, cfg.length),
            *(
                tech.spice_device(f"mn{pin}", "y", pin, "vss", "vss", DeviceKind.NMOS, cfg.nmos_width, cfg.length)
                for pin in "abc"
            ),
            ".ends",
        ]
    )


def gate_subckts(tech: Technology, cfg: LogicConfig, suffix: str = "") -> list[str]:
    """Every multi-input gate a decoder can instantiate (the inverter lives in ``spice``);
    ``suffix`` names a second set, sized by ``cfg`` (a banked decoder's predecode)."""
    return [
        fn(tech, cfg, name=f"gc_{kind}{suffix}")
        for fn, kind in ((nand2_subckt, "nand2"), (nand3_subckt, "nand3"), (nor2_subckt, "nor2"), (nor3_subckt, "nor3"))
    ]


_CELL_OF = {"inv": "gc_inv", "nand": "gc_nand2", "nand3": "gc_nand3", "nor2": "gc_nor2", "nor3": "gc_nor3"}


@dataclass
class GateSpec:
    """One gate of the decoder: ``kind`` is ``inv`` (pins ``a``, ``y``), ``nand``/``nor2``
    (``a``, ``b``, ``y``) or ``nand3``/``nor3`` (``a``, ``b``, ``c``, ``y``), ``pins`` maps each
    pin to its net (in that order), and ``role`` tells the
    physical implementation where the gate goes: ``pre`` (predecode logic), ``row_nand_r``
    (a read row's output gate), ``row_nand_w`` (a write row's first stage) or ``row_inv``
    (a write row's output gate). ``output`` is the decoder output a row gate belongs to.
    """

    kind: str
    pins: dict[str, str]
    role: str = "pre"
    output: str | None = None


def _gates(
    bits: int, low: int, outputs: list[tuple[int, str]], banked_rows: int | None = None,
    rdy: tuple[str, ...] = (),
) -> list[GateSpec]:
    """The gate list for a given low/high split of ``bits`` address bits.

    ``rdy`` (the replica detectors' nets, for a self-timed read) ends reads: each read bus line
    is also ANDed with NOT ``rdy``. Where that still fits one NOR3 it is one more term of the NOR
    the line already is, whose input for it is ``rdy`` itself -- no gate at all. Where it would
    not (a line with two or more high address bits), ``re`` is gated once instead:
    ``re_i_n = NAND(re, rdy_n)`` is the complement every read line takes in place of ``re_n``, so
    the lines stay NOR3 and the predecode no deeper. (A banked decoder takes both banks' ``rdy``,
    ``rdy_n`` then being their NOR: only the bank being read fires its replica.)"""
    gates: list[GateSpec] = []
    made: set[str] = set()
    low_bits = tuple(range(low))
    high_bits = tuple(range(low, bits))

    def inv(a: str, y: str, role: str = "pre", output: str | None = None) -> None:
        gates.append(GateSpec("inv", {"a": a, "y": y}, role, output))

    def nand(a: str, b: str, y: str, role: str = "pre", output: str | None = None) -> None:
        gates.append(GateSpec("nand", {"a": a, "b": b, "y": y}, role, output))

    def ensure(net: str) -> str:
        """``net``, creating its inverter first if it is the complement of an input (or, for a
        self-timed read, the gated read enable and its complement)."""
        if net in made:
            return net
        if net == "rdy_n" and len(rdy) > 1:
            made.add(net)
            gates.append(GateSpec("nor2", {"a": rdy[0], "b": rdy[1], "y": net}))
        elif net == "re_i_n":
            made.add(net)
            nand("re", ensure("rdy_n" if len(rdy) > 1 else f"{rdy[0]}_n"), net)
        elif net == "re_i":
            made.add(net)
            inv(ensure("re_i_n"), net)
        elif net.endswith("_n") and (net[:-2].startswith("addr_") or net[:-2] in ("we", "re", *rdy)):
            made.add(net)
            inv(net[:-2], net)
        return net

    def literal(bit: int, value: int) -> tuple[str, str]:
        """(net that is high when address bit ``bit`` reads ``value``, its complement net)."""
        true, false = f"addr_{bit}", f"addr_{bit}_n"
        return (true, false) if value else (false, true)

    def and_of(leaves: list[tuple[str, str]]) -> str:
        """The AND of ``leaves`` (each a net and its complement), active-high.

        Up to three leaves are one NOR over their *complements* (AND is NOR of the inverted
        inputs) -- no inverter after it. More are split into groups of at most three: each
        group is a NAND (active-low), and one NOR over the groups' outputs ANDs them.
        """
        if len(leaves) == 1:
            return ensure(leaves[0][0])
        name = "and_" + "_".join(true for true, _false in leaves)
        if name in made:
            return name
        if len(leaves) <= 3:
            complements = [ensure(false) for _true, false in leaves]
        else:
            size = -(-len(leaves) // 3)
            complements = []
            for i in range(0, len(leaves), size):
                group = leaves[i : i + size]
                if len(group) == 1:
                    complements.append(ensure(group[0][1]))
                    continue
                nand_net = "nand_" + "_".join(true for true, _false in group)
                if nand_net not in made:
                    made.add(nand_net)
                    pins = dict(zip("abc", (ensure(true) for true, _false in group))) | {"y": nand_net}
                    gates.append(GateSpec("nand" if len(group) == 2 else "nand3", pins))
                complements.append(nand_net)
        made.add(name)
        gates.append(
            GateSpec("nor2" if len(complements) == 2 else "nor3", dict(zip("abc", complements)) | {"y": name})
        )
        return name

    def onehot(idxs: tuple[int, ...], value: int) -> str:
        """The net that is high exactly when address bits ``idxs`` read ``value``."""
        return and_of([literal(bit, (value >> k) & 1) for k, bit in enumerate(idxs)])

    def enabled(enable: str, high: int) -> str:
        """The ``H`` bus line: high bits ``== high`` and ``enable`` high (and, for reads of a
        self-timed macro, no replica detector fired yet)."""
        leaves = [literal(bit, (high >> k) & 1) for k, bit in enumerate(high_bits)]
        if enable == "re" and rdy:
            if len(leaves) + 1 + len(rdy) <= 3:
                return and_of(leaves + [(enable, f"{enable}_n")] + [(f"{r}_n", r) for r in rdy])
            return and_of(leaves + [("re_i", "re_i_n")])
        return and_of(leaves + [(enable, f"{enable}_n")])

    for row, signal in outputs:
        line_low = onehot(low_bits, row & ((1 << low) - 1))
        line_high = enabled("we" if signal == "wwl_in" else "re", row >> low)
        out = output_net(signal, row, banked_rows)
        if signal == "wwl_in":
            write_n = "write_n" + out[len(signal):]
            nand(line_low, line_high, write_n, "row_nand_w", out)
            inv(write_n, out, "row_inv", out)
        else:
            nand(line_low, line_high, out, "row_nand_r", out)
    return gates


def split_address(
    bits: int, outputs: list[tuple[int, str]], banked_rows: int | None = None, rdy: tuple[str, ...] = ()
) -> int:
    """Number of low address bits (at least one) that need the fewest gates."""
    best = min(range(1, bits + 1), key=lambda low: (len(_gates(bits, low, outputs, banked_rows, rdy)), low))
    return best


def replica_detectors(cfg: ArrayConfig, banked: bool = False) -> tuple[str, ...]:
    """The ``rdy`` nets a self-timed read's decoder takes (none unless ``cfg.self_timed_read``)."""
    if not cfg.self_timed_read:
        return ()
    return ("rdy_0", "rdy_1") if banked else ("rdy",)


def decoder_inputs(cfg: ArrayConfig, banked: bool = False) -> list[str]:
    """Every input of the decoder: the address, ``we``, ``re``, and ``rdy`` for a self-timed read."""
    bits = banked_address_bits(cfg.rows) if banked else address_bits(cfg.rows)
    return [f"addr_{i}" for i in range(bits)] + ["we", "re", *replica_detectors(cfg, banked)]


def decoder_address(cfg: ArrayConfig, outputs: list[tuple[int, str]] | None, banked: bool = False):
    """``(bits, outputs, banked_rows)`` of a decoder, with the default outputs filled in."""
    if banked:
        default, bits, banked_rows = banked_outputs(cfg.rows), banked_address_bits(cfg.rows), cfg.rows
    else:
        default, bits, banked_rows = all_row_outputs(cfg.rows), address_bits(cfg.rows), None
    return bits, (outputs if outputs is not None else default), banked_rows


def decoder_gates(
    cfg: ArrayConfig, outputs: list[tuple[int, str]] | None = None, banked: bool = False
) -> list[GateSpec]:
    """Every gate of the decoder for ``outputs`` (default: both word lines of every row, of
    both banks if ``banked``), producers before consumers, predecode logic first and row
    gates last.
    """
    bits, outputs, banked_rows = decoder_address(cfg, outputs, banked)
    rdy = replica_detectors(cfg, banked)
    return _gates(bits, split_address(bits, outputs, banked_rows, rdy), outputs, banked_rows, rdy)


def row_decoder_pins(
    cfg: ArrayConfig, outputs: list[tuple[int, str]] | None = None, banked: bool = False
) -> list[str]:
    _bits, outputs, banked_rows = decoder_address(cfg, outputs, banked)
    pins = decoder_inputs(cfg, banked)
    pins += [output_net(sig, r, banked_rows) for r, sig in outputs]
    return pins + ["vdd", "vss"]


def row_decoder_subckt(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig | None = None,
    name: str = "gc_row_decoder",
    outputs: list[tuple[int, str]] | None = None,
    banked: bool = False,
    predecode_suffix: str = "",
) -> str:
    """The decoder as one flat subcircuit of ``gc_inv``/``gc_nand2``/``gc_nand3`` instances.

    ``predecode_suffix`` instantiates the predecode gates from a second, separately sized set of
    cells (``gc_inv{suffix}``, ...; see :func:`gate_subckts`).

    ``outputs`` restricts which (row, "wwl_in"|"rwl_in") signals are actually produced --
    used to build the two half-decoders in :mod:`gc_compiler.periphery.decoder` that each
    feed one side of the array, without generating gates for the other side's rows.
    Defaults to every row's both word lines (of both banks, as logical rows, if ``banked``).
    """
    lines = [f".subckt {name} {' '.join(row_decoder_pins(cfg, outputs, banked))}"]
    lines.append("* predecode lines (low bits: L, high bits with enable: H), then one NAND per row")
    for index, gate in enumerate(decoder_gates(cfg, outputs, banked)):
        cell = _CELL_OF[gate.kind] + (predecode_suffix if gate.role == "pre" else "")
        lines.append(f"X{gate.kind}_{index} {' '.join(gate.pins.values())} vdd vss {cell}")
    lines.append(".ends")
    return "\n".join(lines)
