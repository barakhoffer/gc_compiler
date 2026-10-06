"""LibreLane runs: a config in, the RTL-to-GDS flow (synthesis, floorplan, power grid, placement,
clock tree, routing, sign-off timing, DRC, LVS, GDS) run, its results read back."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from gc_compiler.flow import FlowError, require, run
from gc_compiler.tech.rules import StdCellLibrary, Technology


@dataclass(frozen=True)
class FlowRun:
    design: str
    run_dir: Path
    metrics: dict
    corners: dict[str, str]
    """SPICE process corner -> the library's corner name (the sign-off timing reports these)."""
    rc_corner: str = "nom"  # tech-neutral: LibreLane's name for its nominal RC corner
    """The flow's parasitic corner (the library's ``rc_corner``)."""

    @property
    def final(self) -> Path:
        return self.run_dir / "final"

    def output(self, kind: str, suffix: str) -> Path:
        """One of the flow's final views, e.g. ``output("gds", "gds")`` or ``output("nl", "nl.v")``."""
        return self.final / kind / f"{self.design}.{suffix}"

    @property
    def gds(self) -> Path:
        return self.output("gds", "gds")

    @property
    def lef(self) -> Path:
        return self.output("lef", "lef")

    @property
    def netlist(self) -> Path:
        """Gate-level Verilog without power pins (what a testbench simulates)."""
        return self.output("nl", "nl.v")

    def liberty(self, corner: str) -> Path:
        """The block's own timing model at one process corner (``tt``/``ss``/``ff``)."""
        lib_corner = self.corners[corner]
        rc = self.rc_corner
        return self.final / "lib" / f"{rc}_{lib_corner}" / f"{self.design}__{rc}_{lib_corner}.lib"

    def _m(self, key: str, default=None):
        return self.metrics.get(key, default)

    @property
    def drc_errors(self) -> int:
        """Both sign-off decks: KLayout's and magic's."""
        return int(self._m("magic__drc_error__count", 0)) + int(self._m("klayout__drc_error__count", 0))

    @property
    def lvs_errors(self) -> int:
        return int(self._m("design__lvs_error__count", 0))

    @property
    def clean(self) -> bool:
        """DRC, LVS and antenna clean, and timing met at every corner -- slew and load limits too
        (a weak driver on a long wire meets setup and hold and still misses its slew)."""
        timing = all(
            c["setup_ns"] >= 0 and c["hold_ns"] >= 0 and c["max_slew_violations"] == 0 and c["max_cap_violations"] == 0
            for c in self.timing().values()
        )
        antenna = int(self._m("route__antenna_violation__count", 0)) == 0
        return self.drc_errors == 0 and self.lvs_errors == 0 and antenna and timing

    def timing(self) -> dict[str, dict[str, float]]:
        """Worst setup and hold slack (ns) and slew/capacitance violations at each process corner."""
        out = {}
        for corner, lib_corner in self.corners.items():
            tag = f"corner:{self.rc_corner}_{lib_corner}"
            out[corner] = {
                "setup_ns": float(self._m(f"timing__setup__ws__{tag}", 0.0)),
                "hold_ns": float(self._m(f"timing__hold__ws__{tag}", 0.0)),
                "max_slew_violations": int(self._m(f"design__max_slew_violation__count__{tag}", 0)),
                "max_cap_violations": int(self._m(f"design__max_cap_violation__count__{tag}", 0)),
            }
        return out

    def summary(self) -> dict:
        m = self._m
        # LibreLane's standard-cell totals leave the fill out but count the well taps
        return {
            "design": self.design,
            "die_area_um2": float(m("design__die__area", 0.0)),
            "cell_area_um2": float(m("design__instance__area__stdcell", 0.0))
            - float(m("design__instance__area__class:tap_cell", 0.0)),
            "logic_cells": int(m("design__instance__count__stdcell", 0))
            - int(m("design__instance__count__class:tap_cell", 0)),
            "flops": int(m("design__instance__count__class:sequential_cell", 0)),
            "utilization": float(m("design__instance__utilization", 0.0)),
            "power_w": float(m("power__total", 0.0)),
            "drc_errors": self.drc_errors,
            "lvs_errors": self.lvs_errors,
            "antenna_violations": int(m("route__antenna_violation__count", 0)),
            "timing": self.timing(),
            "clean": self.clean,
            "run_dir": str(self.run_dir),
        }


def _flow_error(log: Path) -> str:
    """LibreLane's own account of why it stopped: its log from "The following error was
    encountered" on, with the terminal's colour and hyperlink escapes removed (depending on where
    it runs, they can make every line hundreds of characters long)."""
    text = log.read_text(errors="replace")
    text = re.sub(r"\x1b\]8;[^\x1b\x07]*(?:\x1b\\|\x07)", "", text)  # hyperlinks (OSC 8)
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)  # colours and cursor control
    marker = text.rfind("The following error was encountered")
    return (text[marker:] if marker >= 0 else text[-3000:]).strip()


def write_config(lib: StdCellLibrary, config: dict, workdir: str | Path) -> Path:
    """``workdir/config.json``: ``config`` with the library's synthesis settings under it (the
    block's own values win), and its list of cells synthesis may not use written next to it."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    config = {**lib.flow_config, **config}
    if lib.synth_excluded_cells is not None and "SYNTH_EXCLUDED_CELL_FILE" not in config:
        excluded = workdir / "synth_exclude.cells"
        excluded.write_text("".join(f"{cell}\n" for cell in lib.synth_excluded_cells))
        config["SYNTH_EXCLUDED_CELL_FILE"] = str(excluded)
    (workdir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    return workdir


def run_librelane(tech: Technology, config: dict, workdir: str | Path, tag: str = "run") -> FlowRun:
    """Run LibreLane's classic flow on ``config`` (a LibreLane config; paths absolute) in
    ``workdir``. Raises :class:`~gc_compiler.flow.FlowError` if the flow stops; a finished run
    can still carry violations -- see :attr:`FlowRun.clean`."""
    librelane = require("librelane")
    lib = tech.standard_cells()
    workdir = write_config(lib, config, workdir)
    argv = [
        librelane, "--manual-pdk", "--pdk-root", lib.pdk_root, "--pdk", lib.pdk, "--scl", lib.name,
        "--run-tag", tag, "--overwrite", "config.json",
    ]
    try:
        run(argv, workdir, "librelane.log", timeout=6 * 3600)
    except FlowError:
        raise FlowError(f"LibreLane stopped in {workdir / 'runs' / tag}:\n{_flow_error(workdir / 'librelane.log')}") from None
    run_dir = workdir / "runs" / tag
    metrics = run_dir / "final" / "metrics.json"
    if not metrics.exists():
        raise FlowError(f"LibreLane finished without final metrics ({run_dir})")
    return FlowRun(config["DESIGN_NAME"], run_dir, json.loads(metrics.read_text()), dict(lib.corners), lib.rc_corner)


def controller_config(
    design: str, verilog: Path, sdc: Path, clock_ns: float, utilization: float = 55.0
) -> dict:
    """The refresh controller alone, as a block. Its ports face the macro and the user's logic
    directly, so they are not buffered; placement density a little above the core utilisation."""
    return {
        "DESIGN_NAME": design,
        "VERILOG_FILES": [str(Path(verilog).resolve())],
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": clock_ns,
        "PNR_SDC_FILE": str(Path(sdc).resolve()),
        "SIGNOFF_SDC_FILE": str(Path(sdc).resolve()),
        "FP_CORE_UTIL": utilization,
        "PL_TARGET_DENSITY_PCT": utilization + 10,
        "DESIGN_REPAIR_BUFFER_INPUT_PORTS": False,
        "DESIGN_REPAIR_BUFFER_OUTPUT_PORTS": False,
    }


@dataclass(frozen=True)
class MacroViews:
    """The gain-cell macro's views for a flow that places it (see :mod:`gc_compiler.flow.abstract`)."""

    name: str
    gds: Path
    lef: Path
    blackbox: Path
    powered_blackbox: Path
    spice: Path
    liberty: dict[str, Path]
    """Per SPICE process corner (``tt``/``ss``/``ff``)."""
    width: float
    height: float
    power_pins: tuple[str, ...]
    """Pins tied to the power net (``vdd`` and the n-well pins)."""
    ground_pins: tuple[str, ...]
    stripes: tuple[tuple[str, float], ...] = ()
    """The macro's power stripes: ``(pin, x of the centre)`` from its left edge."""


#: The core's left edge, in placement sites from the die's (LibreLane's default, set explicitly
#: where the grid is aligned to the macro).
LEFT_MARGIN_SITES = 12


def _aligned_grid(lib: StdCellLibrary, macro: MacroViews, macro_x: float) -> dict:
    """``PDN_VOFFSET``/``PDN_VSPACING`` that put a power strap and the ground strap after it right
    on a power and a ground stripe of the macro placed at ``macro_x`` (see
    :attr:`~gc_compiler.tech.rules.PowerGrid.align_to_macro`): the closest such pair far enough
    apart for the straps' spacing, and close enough that the next power strap still fits."""
    grid = lib.pdn
    width, space, pitch = grid.vertical_width, grid.vertical_min_spacing, grid.vertical_pitch
    powers = [x for pin, x in macro.stripes if pin in macro.power_pins]
    grounds = [x for pin, x in macro.stripes if pin in macro.ground_pins]
    pairs = sorted(
        (g - p, p) for p in powers for g in grounds if width + space <= g - p <= pitch - width - space
    )
    if not pairs:
        raise ValueError("no power and ground stripe of the macro a grid strap pair can land on")
    distance, power_x = pairs[0]
    core_x0 = LEFT_MARGIN_SITES * lib.site_width
    offset = (macro_x + power_x - core_x0) % pitch
    snap = lambda v: round(round(v / grid.grid) * grid.grid, 6) if grid.grid else v  # noqa: E731
    return {
        "LEFT_MARGIN_MULT": LEFT_MARGIN_SITES,
        "PDN_VOFFSET": snap(offset),
        "PDN_VSPACING": snap(distance - width),
        "PDN_VWIDTH": width,
    }


#: The core's bottom edge, in rows from the die's, and the halo around the macro that placement
#: keeps clear (LibreLane's defaults, set explicitly: the band is sized between them).
BOTTOM_MARGIN_ROWS = 4
MACRO_HALO = 10.0


def user_pin_order(addr_bits: int, cols: int, handshake: bool = False) -> str:
    """A LibreLane pin-order file for a memory block's user pins, all on the bottom edge: the
    control pins first (at the left, under the macro's decoder), then each column's ``din`` and
    ``dout`` in column order -- under the macro's own columns, so every data wire runs straight down
    through the controller band (left to LibreLane's placer they went to the side edges, a user's
    bus split across both)."""
    control = ["clk", "rst_n", "req", "we", *(f"addr\\[{i}\\]" for i in range(addr_bits)), "busy", "dout_valid"]
    if handshake:
        control.append("read_timeout")
    data = [f"{pin}\\[{c}\\]" for c in range(cols) for pin in ("din", "dout")]
    return "#S\n" + "".join(f"{line}\n" for line in control + data)


def memory_config(
    design: str,
    verilog: list[Path],
    sdc: Path,
    clock_ns: float,
    macro: MacroViews,
    controller_area_um2: float,
    lib: StdCellLibrary,
    instance: str = "macro",
    margin: float = 6.0,
    utilization: float = 0.9,
    magic_drc: bool = True,
    pin_order: Path | None = None,
    crossing_nets: int = 0,
    band_scale: float = 1.0,
) -> dict:
    """The macro with its controller as one hard block: the macro fixed at the top, a band of
    standard cells under it (where its data pins are) sized for the controller at ``utilization``,
    the user's pins along the bottom edge, and the core's horizontal power straps (the library's
    ``pdn``) crossing the macro so they land on its supply stripes. Edges and the macro's position
    snap to the library's placement site and rows."""
    corners, site, row = lib.corners, lib.site_width, lib.row_height
    width = macro.width + 2 * margin
    # the band is what placement can use: from the core's bottom edge (BOTTOM_MARGIN_ROWS up) to
    # the macro's halo -- sized as the gap between them, the margin and halo ate ~9 um of it, half
    # of a wide macro's band (its controller is spread thin), which then overfilled
    band = max(10.0, controller_area_um2 / utilization / (width - 2 * margin)) * band_scale
    snap = lambda v: round(v / site) * site  # noqa: E731
    y = round((BOTTOM_MARGIN_ROWS * row + band + MACRO_HALO) / row) * row
    # and tall enough for the nets that run the block's width (a banked macro's din and dout)
    y = max(y, math.ceil(crossing_nets / lib.crossing_nets_per_um * band_scale / row) * row)
    height = y + macro.height + margin
    width, height = snap(width + site), round((height + row) / row) * row
    x = snap(margin)
    power, ground = lib.power_pin, lib.ground_pin
    hooks = [f"{instance} {power} {ground} {p} {macro.ground_pins[0]}" for p in macro.power_pins]
    aligned = _aligned_grid(lib, macro, x) if lib.pdn.align_to_macro else {}
    return {
        "DESIGN_NAME": design,
        "VERILOG_FILES": [str(Path(v).resolve()) for v in verilog],
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": clock_ns,
        "PNR_SDC_FILE": str(Path(sdc).resolve()),
        "SIGNOFF_SDC_FILE": str(Path(sdc).resolve()),
        "FP_SIZING": "absolute",
        "BOTTOM_MARGIN_MULT": BOTTOM_MARGIN_ROWS,
        "FP_MACRO_HORIZONTAL_HALO": MACRO_HALO,
        "FP_MACRO_VERTICAL_HALO": MACRO_HALO,
        "DIE_AREA": [0, 0, round(width, 3), round(height, 3)],
        "PL_TARGET_DENSITY_PCT": 75,
        "MACROS": {
            macro.name: {
                "gds": [str(macro.gds)],
                "lef": [str(macro.lef)],
                "nl": [str(macro.blackbox)],
                "pnl": [str(macro.powered_blackbox)],
                "spice": [str(macro.spice)],
                "lib": {f"*_{corners[c]}": [str(p)] for c, p in macro.liberty.items()},
                "instances": {instance: {"location": [round(x, 3), round(y, 3)], "orientation": "N"}},
            }
        },
        "PDN_MACRO_CONNECTIONS": hooks,
        "PDN_HPITCH": lib.pdn.horizontal_pitch,
        "PDN_HOFFSET": lib.pdn.horizontal_offset,
        "PDN_VPITCH": lib.pdn.vertical_pitch,
        **aligned,
        # (a process whose magic deck is not part of its sign-off: KLayout's alone)
        **({} if magic_drc else {"RUN_MAGIC_DRC": False}),
        **({"IO_PIN_ORDER_CFG": str(Path(pin_order).resolve()), "ERRORS_ON_UNMATCHED_IO": "both"} if pin_order else {}),
        # a long net's route is longer than placement guessed (the cross-bank din and dout of a
        # wide banked macro): repair with a wider margin. (A congested global route still stops the
        # run, for a taller band: handed on, a badly congested one kept the detailed router at
        # thousands of violations for hours, and then failed sign-off anyway.)
        "DESIGN_REPAIR_MAX_CAP_PCT": 50,
        "DESIGN_REPAIR_MAX_SLEW_PCT": 50,
        # the global route's antenna repair with a wider margin: its detailed routes come out longer
        # (a wide macro's cross-bank wires), and at 10% some 30-40 nets were over the limit again,
        # more than the detailed router's diode passes could fix
        "GRT_ANTENNA_REPAIR_MARGIN": 30,
        # and the detailed router given six diode passes, not three (the 2 x 128x128 block's went
        # 29, 17, 15, 6, 3 nets), and the resizer a pass on the global route's own parasitics for
        # the slew and load its estimate missed: so routed, that block signs off clean
        "DRT_ANTENNA_REPAIR_ITERS": 6,
        "RUN_POST_GRT_DESIGN_REPAIR": True,
        "VDD_NETS": [power],
        "GND_NETS": [ground],
        "MAGIC_EXT_USE_GDS": True,
        "LVS_INCLUDE_MARCO_NETLISTS": True,
        "DESIGN_REPAIR_BUFFER_INPUT_PORTS": False,
        "DESIGN_REPAIR_BUFFER_OUTPUT_PORTS": False,
    }
