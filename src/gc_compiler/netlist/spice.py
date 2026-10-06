"""Hierarchical SPICE netlist generation.

The netlist is built from the same configuration objects as the layout, so the two stay
consistent by construction; LVS then compares it against the extracted layout.
"""

from __future__ import annotations

from gc_compiler.config import ArrayConfig, BitcellConfig
from gc_compiler.netlist.decoder import (
    address_bits,
    banked_address_bits,
    gate_subckts,
    row_decoder_pins,
    row_decoder_subckt,
    side_outputs,
)
from gc_compiler.periphery.logic import LevelShifterConfig, LogicConfig, driver_logic
from gc_compiler.periphery.logic import predecode_logic as _predecode
from gc_compiler.tech.rules import DeviceKind, Technology

BITCELL_PINS = ("wwl", "wbl", "rwl", "rbl", "vss", "vnw")
INVERTER_PINS = ("a", "y", "vdd", "vss")


REPLICA_CELL = "gc3t_rep"


def bitcell_subckt(
    tech: Technology, cfg: BitcellConfig, name: str | None = None, replica: bool = False
) -> str:
    """3T gain cell: PMOS write device, NMOS read pair.

    ``replica`` is the replica-column cell: its storage node is the ``wbl`` net (the layout
    ties them together), so the write device is shorted and the read device's gate is ``wbl``.
    """
    name = name or (REPLICA_CELL if replica else "gc3t")
    sn = "wbl" if replica else "sn"
    write_bulk = "vnw" if cfg.write_kind is DeviceKind.PMOS else "vss"
    read_bulk = "vss" if cfg.read_kind is DeviceKind.NMOS else "vnw"
    lines = [
        f".subckt {name} {' '.join(BITCELL_PINS)}",
        "* mw: write transistor, wbl -> sn under wwl control",
        tech.spice_device(
            "mw",
            "wbl",
            "wwl",
            sn,
            write_bulk,
            cfg.write_kind,
            cfg.write_width,
            cfg.write_length,
            cfg.write_flavor,
        ),
        "* mr: read transistor, gated by the storage node",
        tech.spice_device(
            "mr",
            "n_int",
            sn,
            "vss",
            read_bulk,
            cfg.read_kind,
            cfg.read_width,
            cfg.effective_mr_length,
            cfg.read_flavor,
        ),
        "* ms: read select",
        tech.spice_device(
            "ms",
            "rbl",
            "rwl",
            "n_int",
            read_bulk,
            cfg.read_kind,
            cfg.read_width,
            cfg.read_length,
            cfg.read_flavor,
        ),
        ".ends",
    ]
    return "\n".join(lines)


def array_pins(cfg: ArrayConfig) -> list[str]:
    pins = [f"wwl_{r}" for r in range(cfg.rows)]
    pins += [f"rwl_{r}" for r in range(cfg.rows)]
    pins += [f"wbl_{c}" for c in range(cfg.cols)]
    pins += [f"rbl_{c}" for c in range(cfg.cols)]
    if cfg.replica_column:
        pins += ["wbl_rep", "rbl_rep"]
    return pins + ["vss", "vnw"]


def array_subckt(cfg: ArrayConfig, name: str | None = None, cell: str = "gc3t") -> str:
    name = name or cfg.name
    lines = [f".subckt {name} {' '.join(array_pins(cfg))}"]
    for row in range(cfg.rows):
        for col in range(cfg.cols):
            nets = f"wwl_{row} wbl_{col} rwl_{row} rbl_{col} vss vnw"
            lines.append(f"Xbit_{row}_{col} {nets} {cell}")
        if cfg.replica_column:
            nets = f"wwl_{row} wbl_rep rwl_{row} rbl_rep vss vnw"
            lines.append(f"Xrep_{row} {nets} {REPLICA_CELL}")
    lines.append(".ends")
    return "\n".join(lines)


def array_netlist(tech: Technology, cfg: ArrayConfig, name: str | None = None) -> str:
    """Complete netlist: bitcell definition followed by the array."""
    header = [
        f"* gain-cell array {cfg.rows}x{cfg.cols} ({cfg.bits} bits)",
        f"* technology: {tech.name}",
        "",
    ]
    cells = [bitcell_subckt(tech, cfg.bitcell), ""]
    if cfg.replica_column:
        cells += [bitcell_subckt(tech, cfg.bitcell, replica=True), ""]
    return "\n".join(header + cells + [array_subckt(cfg, name), ""])


def inverter_subckt(tech: Technology, cfg: LogicConfig, name: str = "gc_inv") -> str:
    return "\n".join(
        [
            f".subckt {name} {' '.join(INVERTER_PINS)}",
            tech.spice_device(
                "mp", "y", "a", "vdd", "vdd", DeviceKind.PMOS, cfg.pmos_width, cfg.length
            ),
            tech.spice_device(
                "mn", "y", "a", "vss", "vss", DeviceKind.NMOS, cfg.nmos_width, cfg.length
            ),
            ".ends",
        ]
    )


NEG_DRIVER_PINS = ("in", "wwl", "vdd", "vss", "vneg")


def neg_driver_subckt(
    tech: Technology,
    logic: LogicConfig,
    ls: LevelShifterConfig | None = None,
    name: str = "gc_wwl_neg",
    inverter: str = "gc_inv",
) -> str:
    """Write-word-line driver that pulls the word line to ``vneg`` (below vss).

    The word line is ``not in`` as for the plain inverter driver, but between ``vdd`` and
    ``vneg``. A plain inverter cannot do that: its NMOS would see a 0 V gate against a
    source at ``vneg``, so it is never fully off. A level shifter first moves the input to the
    ``vdd``/``vneg`` domain: two PMOS at the top rail take ``in`` and its complement (which are
    fully off at the rails and turn on with a full swing), and two cross-coupled NMOS at ``vneg``
    latch the result. The output stage is an inverter in that domain::

        inb = not in           x, z: the latch nodes (x follows in, z its complement)
        pa: inb -> x (vdd)     pb: in -> z (vdd)      nx: z gates x -> vneg    nz: x gates z -> vneg
        pd/nd: inverter, input x, output wwl
    """
    ls = ls or LevelShifterConfig.for_tech(tech)
    n, p = DeviceKind.NMOS, DeviceKind.PMOS
    dev = tech.spice_device
    return "\n".join(
        [
            f".subckt {name} {' '.join(NEG_DRIVER_PINS)}",
            f"Xinb in inb vdd vss {inverter}",
            "* level shifter: PMOS inputs at vdd, latched by NMOS in an isolated well at vneg",
            dev("pa", "x", "inb", "vdd", "vdd", p, ls.pmos_width, ls.input_length),
            dev("pb", "z", "in", "vdd", "vdd", p, ls.pmos_width, ls.input_length),
            dev("nx", "x", "z", "vneg", "vneg", n, ls.latch_width, ls.latch_length),
            dev("nz", "z", "x", "vneg", "vneg", n, ls.latch_width, ls.latch_length),
            "* output inverter in the vdd/vneg domain",
            dev("pd", "wwl", "x", "vdd", "vdd", p, logic.pmos_width, logic.length),
            dev("nd", "wwl", "x", "vneg", "vneg", n, logic.nmos_width, logic.length),
            ".ends",
        ]
    )


def driver_inverter(cfg: ArrayConfig) -> str:
    """The word-line drivers' inverter subcircuit: the logic inverter, or its scaled copy."""
    return "gc_inv" if cfg.wordline_driver_strength == 1 else "gc_inv_drv"


def bank_net(net: str, bank: int) -> str:
    """An array/column net of one bank of a banked macro: ``wwl_3`` -> ``wwl_{bank}_3``,
    ``wbl_rep`` -> ``wbl_rep_{bank}``, ``vnw`` -> ``vnw_{bank}`` (each array's n-well bias is
    its own net); ``vss`` is shared."""
    if net == "vss":
        return net
    if net == "vnw" or net.endswith("_rep"):
        return f"{net}_{bank}"
    signal, index = net.rsplit("_", 1)
    return f"{signal}_{bank}_{index}"


def macro_pins(cfg: ArrayConfig, decoded: bool = True, banked: bool = False) -> list[str]:
    if banked:
        pins = [f"addr_{i}" for i in range(banked_address_bits(cfg.rows))] + ["we", "re"]
        pins += [f"din_{b}_{c}" for b in (0, 1) for c in range(cfg.cols)]
        pins += [f"dout_{b}_{c}" for b in (0, 1) for c in range(cfg.cols)]
        if cfg.replica_column:
            pins += ["rdy_0", "rdy_1"]
        pins += ["pch", "vdd", "vss", "vnw_0", "vnw_1"]
        return pins + (["vneg"] if cfg.negative_write_wl else [])
    if decoded:
        bits = address_bits(cfg.rows)
        pins = [f"addr_{i}" for i in range(bits)] + ["we", "re"]
    else:
        pins = [f"wwl_in_{r}" for r in range(cfg.rows)]
        pins += [f"rwl_in_{r}" for r in range(cfg.rows)]
    pins += [f"din_{c}" for c in range(cfg.cols)]
    pins += [f"dout_{c}" for c in range(cfg.cols)]
    if cfg.replica_column:
        pins.append("rdy")
    pins += ["pch", "vdd", "vss", "vnw"]
    return pins + (["vneg"] if cfg.negative_write_wl else [])


def _write_driver(tech: Technology, cfg: ArrayConfig, logic: LogicConfig, tag: str, din: str, wbl: str, rbl: str) -> list[str]:
    """A column's write driver: an inverter, or with ``cfg.refresh_in_macro`` ``wbl = NAND(din, rbl)``
    (``rbl``'s PMOS and both series NMOS twice the logic width, as laid out) -- see
    :func:`gc_compiler.periphery.column.column_circuit`."""
    if not cfg.refresh_in_macro:
        return [f"Xwr_{tag} {din} {wbl} vdd vss gc_inv"]
    wn, mid = round(2 * logic.nmos_width, 6), f"wmid_{tag}"
    return [
        tech.spice_device(f"wpa_{tag}", wbl, din, "vdd", "vdd", DeviceKind.PMOS, logic.pmos_width, logic.length),
        tech.spice_device(f"wpb_{tag}", wbl, rbl, "vdd", "vdd", DeviceKind.PMOS, round(2 * logic.pmos_width, 6), logic.length),
        tech.spice_device(f"wna_{tag}", wbl, din, mid, "vss", DeviceKind.NMOS, wn, logic.length),
        tech.spice_device(f"wnb_{tag}", mid, rbl, "vss", "vss", DeviceKind.NMOS, wn, logic.length),
    ]


def macro_subckt(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig,
    name: str,
    array_name: str,
    decoded: bool = True,
    ls: LevelShifterConfig | None = None,
    banked: bool = False,
) -> str:
    """Array plus word-line drivers, precharge devices, read buffers and write drivers.

    ``decoded`` wires a static NAND/inverter row decoder (see
    :mod:`gc_compiler.netlist.decoder`) between the ``addr``/``we``/``re`` pins and the
    array's word lines; set it to ``False`` to expose the raw per-row ``wwl_in``/``rwl_in``
    pins instead. ``banked``: two arrays and one banked decoder (see :func:`_banked_body`).
    """
    if banked:
        return "\n".join(
            [f".subckt {name} {' '.join(macro_pins(cfg, True, True))}", *_banked_body(tech, cfg, logic, array_name), ".ends"]
        )
    lines = [f".subckt {name} {' '.join(macro_pins(cfg, decoded))}"]
    lines.append(f"Xarray {' '.join(array_pins(cfg))} {array_name}")
    if decoded:
        left, right = side_outputs(cfg.rows)
        lines.append("* address decoder: addr/we/re -> per-row wwl_in/rwl_in, split the way")
        lines.append("* the array actually lands word lines (one per row on each side)")
        lines.append(f"Xdecoder_left {' '.join(row_decoder_pins(cfg, left))} gc_row_decoder_left")
        lines.append(f"Xdecoder_right {' '.join(row_decoder_pins(cfg, right))} gc_row_decoder_right")
    lines.append("* word-line drivers (inverting; landed from alternating sides)")
    drv = driver_inverter(cfg)
    for row in range(cfg.rows):
        if cfg.negative_write_wl:
            lines.append(f"Xwwl_drv_{row} wwl_in_{row} wwl_{row} vdd vss vneg gc_wwl_neg")
        else:
            lines.append(f"Xwwl_drv_{row} wwl_in_{row} wwl_{row} vdd vss {drv}")
        lines.append(f"Xrwl_drv_{row} rwl_in_{row} rwl_{row} vdd vss {drv}")
    lines.append("* column circuits")
    for col in range(cfg.cols):
        lines.append(
            tech.spice_device(
                f"pre_{col}",
                f"rbl_{col}",
                "pch",
                "vdd",
                "vdd",
                DeviceKind.PMOS,
                logic.pmos_width,
                logic.length,
            )
        )
        lines.append(f"Xrd_{col} rbl_{col} dout_{col} vdd vss gc_inv")
        lines += _write_driver(tech, cfg, logic, str(col), f"din_{col}", f"wbl_{col}", f"rbl_{col}")
    if cfg.replica_column:
        lines.append("* replica column: the same slices, but the write driver's input is tied low so")
        lines.append("* wbl_rep (and with it every replica storage node) is held at vdd, and the read")
        lines.append("* buffer is the detector: rdy rises when the replica bit line has discharged")
        lines.append(
            tech.spice_device(
                "pre_rep", "rbl_rep", "pch", "vdd", "vdd", DeviceKind.PMOS, logic.pmos_width, logic.length
            )
        )
        lines.append("Xrd_rep rbl_rep rdy vdd vss gc_inv")
        lines += _write_driver(tech, cfg, logic, "rep", "vss", "wbl_rep", "rbl_rep")
    lines.append(".ends")
    return "\n".join(lines)


def _banked_body(tech: Technology, cfg: ArrayConfig, logic: LogicConfig, array_name: str) -> list[str]:
    """Two banks, each exactly a plain macro's array, drivers and column circuits (its nets
    renamed by :func:`bank_net`), and one banked decoder driving both banks' word lines."""
    lines = []
    for bank in (0, 1):
        lines.append(f"Xarray_{bank} {' '.join(bank_net(net, bank) for net in array_pins(cfg))} {array_name}")
    lines.append("* one decoder for both banks: the bank is the top address bit")
    lines.append(f"Xdecoder {' '.join(row_decoder_pins(cfg, banked=True))} gc_row_decoder_banked")
    for bank in (0, 1):
        lines.append(f"* bank {bank}: word-line drivers and column circuits")
        for row in range(cfg.rows):
            wwl_in, wwl = f"wwl_in_{bank}_{row}", bank_net(f"wwl_{row}", bank)
            if cfg.negative_write_wl:
                lines.append(f"Xwwl_drv_{bank}_{row} {wwl_in} {wwl} vdd vss vneg gc_wwl_neg")
            else:
                lines.append(f"Xwwl_drv_{bank}_{row} {wwl_in} {wwl} vdd vss {driver_inverter(cfg)}")
            lines.append(
                f"Xrwl_drv_{bank}_{row} rwl_in_{bank}_{row} {bank_net(f'rwl_{row}', bank)} vdd vss {driver_inverter(cfg)}"
            )
        columns = [(str(col), bank_net(f"rbl_{col}", bank), bank_net(f"wbl_{col}", bank),
                    f"din_{bank}_{col}", f"dout_{bank}_{col}") for col in range(cfg.cols)]
        if cfg.replica_column:
            columns.append(("rep", f"rbl_rep_{bank}", f"wbl_rep_{bank}", "vss", f"rdy_{bank}"))
        for tag, rbl, wbl, din, dout in columns:
            lines.append(
                tech.spice_device(
                    f"pre_{bank}_{tag}", rbl, "pch", "vdd", "vdd", DeviceKind.PMOS, logic.pmos_width, logic.length
                )
            )
            lines.append(f"Xrd_{bank}_{tag} {rbl} {dout} vdd vss gc_inv")
            lines += _write_driver(tech, cfg, logic, f"{bank}_{tag}", din, wbl, rbl)
    return lines


def macro_netlist(
    tech: Technology,
    cfg: ArrayConfig,
    logic: LogicConfig | None = None,
    name: str | None = None,
    decoded: bool = True,
    ls: LevelShifterConfig | None = None,
    banked: bool = False,
    predecode_logic: LogicConfig | None = None,
) -> str:
    """Full macro netlist: bitcell, array, logic slices, periphery and (by default) decoder.

    ``ls`` sizes the level shifters of a negative-write-word-line macro; it must be the one the
    layout was built with. ``banked`` (which implies ``decoded``): two banks of ``cfg`` sharing
    one decoder, whose predecode is sized by ``predecode_logic`` (default
    :func:`~gc_compiler.periphery.logic.predecode_logic`, as the layout's).
    """
    logic = logic or LogicConfig.for_tech(tech)
    array_name = cfg.name
    name = name or f"{cfg.name}_macro"
    bits = 2 * cfg.bits if banked else cfg.bits
    header = [
        f"* gain-cell macro {'2x ' if banked else ''}{cfg.rows}x{cfg.cols} ({bits} bits)",
        f"* technology: {tech.name}",
        "",
    ]
    body = [bitcell_subckt(tech, cfg.bitcell), ""]
    if cfg.replica_column:
        body += [bitcell_subckt(tech, cfg.bitcell, replica=True), ""]
    body += [array_subckt(cfg, array_name), "", inverter_subckt(tech, logic), ""]
    drv = driver_logic(logic, cfg.wordline_driver_strength, tech)
    if driver_inverter(cfg) != "gc_inv":
        body += [inverter_subckt(tech, drv, name=driver_inverter(cfg)), ""]
    if cfg.negative_write_wl:
        body += [neg_driver_subckt(tech, drv, ls, inverter=driver_inverter(cfg)), ""]
    if banked:
        predecode_logic = predecode_logic or _predecode(tech)
        body += [
            *[text for subckt in gate_subckts(tech, logic) for text in (subckt, "")],
            inverter_subckt(tech, predecode_logic, name="gc_inv_pre"),
            "",
            *[text for subckt in gate_subckts(tech, predecode_logic, "_pre") for text in (subckt, "")],
            row_decoder_subckt(
                tech, cfg, logic, name="gc_row_decoder_banked", banked=True, predecode_suffix="_pre"
            ),
            "",
            macro_subckt(tech, cfg, logic, name, array_name, True, ls, banked=True),
            "",
        ]
        return "\n".join(header + body)
    if decoded:
        left, right = side_outputs(cfg.rows)
        body += [
            *[text for subckt in gate_subckts(tech, logic) for text in (subckt, "")],
            row_decoder_subckt(tech, cfg, logic, name="gc_row_decoder_left", outputs=left),
            "",
            row_decoder_subckt(tech, cfg, logic, name="gc_row_decoder_right", outputs=right),
            "",
        ]
    body += [macro_subckt(tech, cfg, logic, name, array_name, decoded, ls), ""]
    return "\n".join(header + body)
