"""Command line entry point: ``gc-compiler --rows 32 --cols 32`` (a decoded, banked macro)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gc_compiler.compiler import GainCellCompiler
from gc_compiler.config import WIDE_ARRAY, ArrayConfig, BitcellConfig
from gc_compiler.tech import TECHNOLOGIES, default_technology, technology


def build_parser() -> argparse.ArgumentParser:
    cell = BitcellConfig()  # the default technology's, for the help texts
    plain = BitcellConfig.plain()
    parser = argparse.ArgumentParser(description="Generate a 3T gain-cell memory array")
    parser.add_argument("--rows", type=int, default=32, help="number of rows (must be even)")
    parser.add_argument("--cols", type=int, default=32, help="number of columns")
    parser.add_argument("--out", type=Path, default=Path("build"), help="output directory")
    parser.add_argument(
        "--tech",
        default=None,
        help=f"the technology: {', '.join(TECHNOLOGIES)} (default {default_technology().name}), or "
        "package.module:Class for an adapter of your own",
    )
    parser.add_argument("--name", default=None, help="top cell name")
    parser.add_argument(
        "--plain-cell",
        action="store_true",
        help=f"the plain 3T cell (mr as long as ms) instead of the default with a {cell.mr_length:g} um mr; "
        "its written '0' needs --neg-wl to read back at every corner",
    )
    parser.add_argument("--write-width", type=float, default=None, help=f"default {cell.write_width:g} (the technology's)")
    parser.add_argument(
        "--write-length",
        type=float,
        default=None,
        help=f"default {cell.write_length:g}, {plain.write_length:g} with --plain-cell (the technology's "
        "sizing; these are the default technology's)",
    )
    parser.add_argument("--read-width", type=float, default=None, help=f"default {cell.read_width:g}")
    parser.add_argument(
        "--read-length",
        type=float,
        default=None,
        help=f"length of ms, the rwl-gated select device (default {cell.read_length:g}); "
        "also mr's with --plain-cell",
    )
    parser.add_argument(
        "--mr-length",
        type=float,
        default=None,
        help="length of mr, the storage-node-gated read device: its threshold sets how high a "
        f"stored '0' still reads as one (default {cell.mr_length:g}; with --plain-cell, "
        "the same as --read-length)",
    )
    parser.add_argument(
        "--write-flavor",
        default=None,
        choices=["lvt", "svt", "hvt"],
        help=f"write device implant (default {cell.write_flavor or 'svt'}, the technology's; lvt writes a "
        "lower '0' but leaks into the column's other '0's while its write bit line is high: see the README)",
    )
    parser.add_argument(
        "--tap-row-interval", type=int, default=None, help="row pairs between tap rows (default: the technology's)"
    )
    parser.add_argument(
        "--macro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generate the periphery (word-line drivers and column circuits) around the array; "
        "--no-macro builds the bitcell array alone (default: on)",
    )
    parser.add_argument(
        "--decoded",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="place and wire the row decoder, so the macro takes addr/we/re; --no-decoded leaves "
        "raw per-row word-line inputs (default: on with the macro)",
    )
    parser.add_argument(
        "--banked",
        action="store_true",
        help="two banks of --rows x --cols around one central decoder (one more address bit; "
        "less decoder area per bit) -- the default for a decoded macro",
    )
    parser.add_argument(
        "--one-bank",
        action="store_true",
        help="build a single bank (a half-decoder on each side) instead of two",
    )
    parser.add_argument(
        "--replica",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="a replica bit-line column (adds a 'rdy' output that tracks the read delay; on by default)",
    )
    parser.add_argument(
        "--wl-straps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="strap every word line in metal3, tied to its poly in every strap column (faster "
        f"word lines on wide arrays; default: on from {WIDE_ARRAY} columns)",
    )
    parser.add_argument(
        "--wl-driver-strength",
        type=float,
        default=None,
        help="word-line driver size as a multiple of the logic cells' (default: 4 from "
        f"{WIDE_ARRAY} columns, 1 below)",
    )
    parser.add_argument(
        "--neg-wl",
        action="store_true",
        help="drive the write word lines below vss (level shifter + isolated-well driver, adds a "
        "'vneg' pin): a written '0' then reads back as a '0'",
    )
    parser.add_argument(
        "--refresh-in-macro",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="each column writes a refresh's data back itself (its write driver is NAND(din, rbl)): "
        "the controller drives din high in a refresh instead of looping dout back (default: on with "
        "self-timed reads; --no-refresh-in-macro loops the data through the controller, which the "
        "sky130 default cell's '0' does not survive at sf)",
    )
    parser.add_argument(
        "--vneg",
        type=float,
        default=None,
        help="level (V) the vneg supply is simulated at with --neg-wl (default: the technology's)",
    )
    parser.add_argument(
        "--characterise",
        action="store_true",
        help="extract parasitics and simulate read/write timing of the decoded macro "
        "(needs magic and ngspice)",
    )
    parser.add_argument(
        "--corners",
        default=None,
        help="comma-separated process corners for --characterise (default: tt,ss,ff,sf,fs)",
    )
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=0,
        metavar="N",
        help="with --characterise, also simulate N dies with local device mismatch at each corner",
    )
    parser.add_argument("--mc-seed", type=int, default=1, help="first mismatch seed (default 1)")
    parser.add_argument(
        "--mc-zero-level",
        type=float,
        default=None,
        help="storage-node level (V) the Monte-Carlo '0's start from (default: the level each "
        "die's own write leaves)",
    )
    parser.add_argument(
        "--self-timed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="the replica's rdy ends every read by itself (the read word line turns off, the bit "
        "lines hold), so reads work at every corner however long re is held (default: on with "
        "the replica column)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="also write a refresh controller, a behavioural model of the "
        "macro, a top level and a testbench (Verilog); timing from --characterise when given",
    )
    parser.add_argument(
        "--controller",
        action="store_true",
        help="also build the whole memory as one hard block (<name>_top): the macro with its refresh "
        "controller synthesized, placed and routed under it by LibreLane (GDS, LEF, timing "
        "libraries, netlists), DRC/LVS checked, timed at every corner and simulated at gate level",
    )
    parser.add_argument("--clock-ns", type=float, default=10.0, help="the controller's clock period (default 10)")
    parser.add_argument(
        "--handshake",
        action="store_true",
        help="the controller ends each read on the replica's rdy (synchronised) instead of after a "
        "fixed read time: no read time to get right, at least three cycles a read",
    )
    parser.add_argument(
        "--read-ns",
        type=float,
        default=None,
        help="how long the controller holds a read, without --characterise (default 4, sized for "
        "small macros: a large one's slowest corner needs more -- holding longer is harmless, the "
        "macro ends each read itself)",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=None,
        metavar="US",
        help="refresh interval to keep, in us (default: the one measured for the cell at its worst "
        "corner and die -- 154 for the default cell, 45 for one not measured)",
    )
    parser.add_argument("--drc", action="store_true", help="run the PDK DRC deck on the result")
    parser.add_argument(
        "--lvs", action="store_true", help="extract the GDS with magic and compare it to the netlist"
    )
    return parser


def resolve_bitcell(args: argparse.Namespace, tech=None) -> BitcellConfig:
    """``tech``'s default cell, or its plain one with ``--plain-cell``, with each geometry flag given
    overriding its own field."""
    overrides = {
        field: value
        for field, value in (
            ("write_width", args.write_width),
            ("write_length", args.write_length),
            ("read_width", args.read_width),
            ("read_length", args.read_length),
            ("mr_length", args.mr_length),
        )
        if value is not None
    }
    if args.write_flavor is not None:
        overrides["write_flavor"] = None if args.write_flavor == "svt" else args.write_flavor
    tech = tech or technology(getattr(args, "tech", None))
    return BitcellConfig.plain(tech, **overrides) if args.plain_cell else BitcellConfig.for_tech(tech, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tech = technology(args.tech)
    bitcell = resolve_bitcell(args, tech)
    if args.decoded and not args.macro:
        build_parser().error("--decoded needs the macro (drop --no-macro)")
    if args.decoded is None:
        args.decoded = args.macro
    if args.neg_wl and not args.macro:
        build_parser().error("--neg-wl needs the macro (drop --no-macro)")
    if args.characterise and not (args.macro and args.decoded):
        build_parser().error("--characterise needs the decoded macro (drop --no-macro / --no-decoded)")
    if args.banked and args.one_bank:
        build_parser().error("--banked and --one-bank exclude each other")
    if args.banked and not (args.macro and args.decoded):
        build_parser().error("--banked needs the decoded macro (drop --no-macro / --no-decoded)")
    args.banked = args.macro and args.decoded and not args.one_bank
    if args.self_timed and not args.replica:
        build_parser().error("--self-timed needs the replica column (drop --no-replica)")
    if args.refresh and not (args.macro and args.decoded and args.replica and args.self_timed is not False):
        build_parser().error("--refresh needs the decoded macro and self-timed reads (no --no-macro, --no-decoded, --no-replica or --no-self-timed)")
    if args.controller and not (args.macro and args.decoded and args.replica and args.self_timed is not False):
        build_parser().error("--controller needs the decoded macro and self-timed reads (no --no-macro, --no-decoded, --no-replica or --no-self-timed)")
    if args.controller and args.neg_wl:
        build_parser().error("--controller does not support --neg-wl (the block has no vneg supply)")
    if args.read_ns is not None and args.characterise:
        build_parser().error("--read-ns and --characterise exclude each other (the measured read sets it)")
    if args.monte_carlo and not args.characterise:
        build_parser().error("--monte-carlo needs --characterise")
    if args.monte_carlo < 0:
        build_parser().error("--monte-carlo needs a positive number of dies")
    given = {
        "tap_row_interval": args.tap_row_interval,
        "self_timed_read": args.self_timed,
        "wordline_straps": args.wl_straps,
        "wordline_driver_strength": args.wl_driver_strength,
        "write_wl_low": args.vneg,
    }
    cfg = ArrayConfig.for_tech(
        tech, rows=args.rows, cols=args.cols, bitcell=bitcell, replica_column=args.replica,
        negative_write_wl=args.neg_wl, refresh_in_macro=args.refresh_in_macro,
        **{k: v for k, v in given.items() if v is not None},
    )
    from gc_compiler.rtl import loopback_warning, rows_warning

    if args.controller and rows_warning(cfg, tech):
        build_parser().error(f"--controller: {rows_warning(cfg, tech)}")
    for warning in (rows_warning(cfg, tech), loopback_warning(cfg, tech)):
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(
        args.out, name=args.name, macro=args.macro, decoded=args.decoded, banked=args.banked
    )
    print(result)
    print(f"  gds:     {result.gds}")
    print(f"  spice:   {result.spice}")
    print(f"  metrics: {result.metrics_file}")

    if args.drc:
        from gc_compiler.verify.drc import run_drc

        print(run_drc(result.gds, tech=tech))

    if args.lvs:
        from gc_compiler.verify.lvs import lvs_layout_vs_netlist

        top_cell = result.component.name
        ports = {p.name: tuple(p.dcenter) for p in result.component.ports}
        lvs = lvs_layout_vs_netlist(
            result.gds, top_cell, result.spice, top_cell, workdir=args.out / "lvs", ports=ports, tech=tech
        )
        print(lvs)

    if args.characterise:
        from gc_compiler.verify.timing import standard_corners

        corners = standard_corners(tech)
        if args.corners:
            wanted = [name.strip() for name in args.corners.split(",")]
            unknown = [name for name in wanted if name not in {c.process for c in corners}]
            if unknown:
                build_parser().error(f"unknown corner(s): {', '.join(unknown)}")
            corners = tuple(c for c in corners if c.process in wanted)
        characterised = report = compiler.characterise(
            result,
            workdir=args.out / "timing",
            corners=corners,
            monte_carlo=args.monte_carlo,
            mc_seed=args.mc_seed,
            mc_zero_level=args.mc_zero_level,
        )
        reports = report if isinstance(report, tuple) else (report,)
        for report in reports:
            suffix = "" if report.bank is None else f".bank{report.bank}"
            if report.bank is not None:
                print(f"\nbank {report.bank}:")
            print(report.table())
            if report.supply_table():
                print()
                print(f"Negative supply {report.vneg:g} V: gate-oxide stress in a write up to {report.write_stress():.2f} V")
                print(report.supply_table())
            if args.monte_carlo:
                print()
                print(f"Monte Carlo, {args.monte_carlo} dies per corner (local mismatch):")
                print(report.mc_table())
                margin = report.statistical_margin(3.0)
                if margin is not None:
                    print(f"  3σ statistical margin: rdy delayed by x{margin:.2f}")
            margin = report.recommended_margin()
            if margin is not None:
                print(f"  rdy must be delayed by x{margin:.2f} to be valid in every corner")
            print(f"  timing:  {result.gds.with_name(result.component.name + suffix + '.timing.json')}")

    timing = None
    if args.read_ns is not None and not args.characterise:
        from gc_compiler.rtl import ControllerTiming, default_refresh_interval_us

        interval = args.refresh_interval if args.refresh_interval is not None else default_refresh_interval_us(cfg, tech)
        timing = ControllerTiming.default(
            tech, clock_ns=args.clock_ns, read_ns=args.read_ns, interval_us=interval, handshake=args.handshake
        )

    if args.refresh:
        paths = compiler.refresh_rtl(
            result, clock_ns=args.clock_ns, interval_us=args.refresh_interval, timing=timing,
            characterisation=characterised if args.characterise else None, handshake=args.handshake,
        )
        for path in paths:
            print(f"  verilog: {path}")

    if args.controller:
        memory = compiler.memory(
            args.out, banked=args.banked, clock_ns=args.clock_ns, interval_us=args.refresh_interval, timing=timing,
            characterisation=characterised if args.characterise else None, handshake=args.handshake,
        )
        m = memory.metrics
        print(
            f"\n{memory.name}: {m['bits']} bits, {m['die_area_um2']:.0f} um2 ({m['area_per_bit_um2']:.2f} um2/bit); "
            f"macro {m['macro_area_um2']:.0f} um2, controller cells {m['controller_cell_area_um2']:.0f} um2 "
            f"({m['logic_cells']} cells, {m['flops']} flops)"
        )
        for corner, t in m["timing"].items():
            print(f"  {corner}: setup slack {t['setup_ns']:.2f} ns, hold slack {t['hold_ns']:.2f} ns")
        print(f"  DRC {m['drc_errors']}, LVS {m['lvs_errors']}, gate-level testbench: PASS")
        print(f"  gds:     {memory.file('.gds')}")
        print(f"  metrics: {memory.file('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
