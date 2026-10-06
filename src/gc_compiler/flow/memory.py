"""The whole memory as one hard block: the gain-cell macro placed as a hard macro, and its refresh
controller synthesized, placed and routed under it, by LibreLane (:func:`build_memory`)."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from gc_compiler.config import ArrayConfig
from gc_compiler.flow.abstract import hard_macro, write_blackbox, write_gds, write_lef, write_liberty
from gc_compiler.flow.gatesim import simulate_gates
from gc_compiler.flow.librelane import FlowRun, MacroViews, memory_config, run_librelane, user_pin_order
from gc_compiler.flow.sta import loopback_arrival
from gc_compiler.netlist.decoder import address_bits
from gc_compiler.netlist.spice import macro_netlist
from gc_compiler.periphery.logic import LevelShifterConfig, LogicConfig
from gc_compiler.rtl import SYNC_CYCLES, ControllerTiming, controller_params, refresh_rtl, top_sdc
from gc_compiler.tech.rules import Technology

BAND_UTILIZATION = 0.9
"""The controller band's first sizing, of the area placement can use (the 8x8 blocks of the earlier
sizing had placed at up to ~90% of it)."""

BAND_GROWTH = 0.8
"""Each retry's band against the last's: 1/0.8 taller."""


@dataclass
class MemoryResult:
    name: str
    """The block's name: ``<name>_top`` (the macro inside it is ``<name>``)."""
    outdir: Path
    run: FlowRun
    macro: MacroViews
    gate_level: str
    """The gate-level testbench's output (it passed)."""
    metrics: dict

    def file(self, suffix: str) -> Path:
        return self.outdir / f"{self.name}{suffix}"


#: Place-and-route attempts, each with the band 1/0.8 taller than the last (a wide macro's band
#: is short, and its routing -- every column's data to the user's pins -- can congest it).
ATTEMPTS = 6


def estimate_flops(cfg: ArrayConfig, banked: bool, timing: ControllerTiming) -> int:
    """The controller's flops (and latches): the read data latches, the refresh row and tick
    counters, the phase counter and the state (see ``rtl.controller``)."""
    params = controller_params(cfg, timing, banked)
    tick = math.ceil(math.log2(params.refresh_cycles + 1))
    handshake = 3 * (2 if banked else 1) + 1 if params.handshake else 0  # rdy synchronisers, read_timeout
    return cfg.cols + address_bits(cfg.rows) + tick + 4 + 3 + 6 + handshake


def _read_ns(characterisation, corner: str, default: float) -> float:
    """The slowest read (``dout``, or the cutoff for a self-timed macro) at one process corner."""
    if characterisation is None:
        return default
    reports = characterisation if isinstance(characterisation, tuple) else (characterisation,)
    times = []
    for report in reports:
        for entry in report.to_dict()["corners"]:
            if entry["corner"]["process"] != corner:
                continue
            for read in entry.get("reads", []):
                times += [t for t in (read.get("t_cutoff"), read.get("t_dout")) if t]
    return max(times) * 1e9 if times else default


def build_memory(
    tech: Technology,
    cfg: ArrayConfig,
    name: str,
    outdir: str | Path,
    timing: ControllerTiming,
    banked: bool = True,
    logic: LogicConfig | None = None,
    ls: LevelShifterConfig | None = None,
    characterisation=None,
) -> MemoryResult:
    """Build ``<name>_top``: the decoded (by default banked) macro ``<name>`` with metal4 power
    pins, and its refresh controller, placed and routed around it by LibreLane; then run the
    generated testbench on the routed gate-level netlist (with the macro's behavioural model).

    Writes, next to each other in ``outdir``: the block's GDS, LEF, timing libraries (one per
    corner), gate-level and powered netlists, SDC and parasitics; the macro's own views; the RTL;
    and ``<name>_top.json`` with the metrics. Raises unless the result is DRC and LVS clean, meets
    timing at every corner and passes its testbench.
    """
    from gc_compiler.array.floorplan import gc_banked_macro, gc_macro
    from gc_compiler.flow import FlowError

    if cfg.negative_write_wl:
        raise ValueError("the negative-rail macro's vneg pin has no supply in the block: not supported")
    if not cfg.self_timed_read:
        raise ValueError("the refresh controller needs self-timed reads (self_timed_read)")
    logic = logic or LogicConfig.for_tech(tech)
    outdir = Path(outdir).resolve()
    views_dir = outdir / f"{name}_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    lib = tech.standard_cells()

    # the macro and its views
    core = f"{name}_core"
    macro = gc_banked_macro(tech, cfg, logic, name=core, ls=ls) if banked else gc_macro(
        tech, cfg, logic, name=core, decoded=True, ls=ls
    )
    hard = hard_macro(tech, macro, cfg, name, banked)
    gds = write_gds(hard.component, views_dir / f"{name}.gds")
    # (the replica's rdy is left unconnected unless the controller ends its reads on it)
    unused = () if timing.handshake else tuple(p for p in hard.pins if p.startswith("rdy"))
    lef = write_lef(tech, hard, views_dir / f"{name}.lef", logic, unused=unused)
    blackbox = write_blackbox(hard, views_dir / f"{name}.bb.v")
    powered = write_blackbox(hard, views_dir / f"{name}.pbb.v", powered=True)
    spice = views_dir / f"{name}.spice"
    spice.write_text(macro_netlist(tech, cfg, logic, name=name, decoded=True, banked=banked, ls=ls))
    reads = {c: _read_ns(characterisation, c, timing.read_ns) for c in lib.corners}
    liberty = {c: write_liberty(hard, views_dir / f"{name}_{c}.lib", c, reads[c], tech=tech) for c in lib.corners}
    wells = tuple(p for p in hard.pins if p.startswith("vnw"))
    views = MacroViews(
        name, gds, lef, blackbox, powered, spice, liberty, hard.width, hard.height, ("vdd", *wells), ("vss",),
        tuple((s.net, s.x) for s in hard.stripes),
    )

    # the controller's RTL and the block's constraints
    rtl = {}
    for filename, text in refresh_rtl(name, cfg, timing, banked, tech).items():
        rtl[filename] = outdir / filename
        rtl[filename].write_text(text)
    params = controller_params(cfg, timing, banked)
    sdc = outdir / f"{name}_top.sdc"
    sdc.write_text(top_sdc(params, timing, sdc=lib.sdc))
    pins = outdir / f"{name}_top.pins"
    pins.write_text(user_pin_order(params.addr_bits, cfg.cols, params.handshake))

    # the library's placed area per flop (its flops, their logic, clock tree and timing repair)
    area = lib.area_per_flop_um2 * estimate_flops(cfg, banked, timing)
    # the band under the macro is sized from an estimate of the controller's area; on a narrow
    # macro the fixed overheads (pins, taps, clock tree) can overfill it, and in a wide banked one
    # every column's din and dout span both banks, wires that crowd it -- so on an overfull
    # placement (global, or detailed once the clock tree and timing repair have added their
    # cells), a congested global route, or a routed block that misses only slew, load or antenna
    # limits (its detours), it is rebuilt taller
    # (each retry scales the whole band -- the cells' share and the height the crossing nets need --
    # so a retry is a different layout, and each keeps its run and why it was retried)
    scale, attempts = 1.0, []
    for attempt in range(ATTEMPTS):
        last = attempt == ATTEMPTS - 1
        config = memory_config(
            f"{name}_top", [rtl[f"{name}_ctrl.v"], rtl[f"{name}_top.v"]], sdc, timing.clock_ns, views, area,
            lib, utilization=BAND_UTILIZATION, magic_drc=tech.tools.magic_drc, pin_order=pins,
            crossing_nets=2 * cfg.cols if banked else 0,  # each column's din and dout reach both banks
            band_scale=scale,
        )
        try:
            run = run_librelane(tech, config, outdir / f"{name}_flow", tag=f"try{attempt}")
        except FlowError as error:
            # (and a router that left a violation or two, e.g. at a crowded macro pin: placed again
            # in a taller band, the routes differ)
            retry = ("GPL-0301", "DPL-0036", "GRT-0116", "Routing DRC errors found")
            found = next((code for code in retry if code in str(error)), None)
            attempts.append({"band_scale": scale, "die": config["DIE_AREA"], "result": found or "failed"})
            if found is None or last:
                raise
            scale /= BAND_GROWTH
            continue
        misses = {c: {k: v for k, v in t.items() if k.startswith("max_") and v} for c, t in run.timing().items()}
        attempts.append({"band_scale": scale, "die": config["DIE_AREA"], "result": "clean" if run.clean else {
            "drc": run.drc_errors, "lvs": run.lvs_errors, "antenna": run.summary()["antenna_violations"],
            "limits": {c: m for c, m in misses.items() if m},
            "timing": {c: (t["setup_ns"], t["hold_ns"]) for c, t in run.timing().items()},
        }})
        if run.clean or last or run.drc_errors or run.lvs_errors:
            break
        scale /= BAND_GROWTH
    summary = run.summary()
    summary["attempts"] = attempts
    if not run.clean:
        raise FlowError(f"{name}_top is not clean: {json.dumps(summary, indent=1)}")

    # the refresh loop-back, which no constraint can end on: dout back to din before we rises,
    # (READ_CYCLES + 1) edges after the edge that raised re, less the clock's setup uncertainty.
    # With the handshake, we rises three edges after the one that caught rdy, and dout trails
    # rdy by rdy_lag_ns: the path is then timed from rdy (the arrival less the macro's read)
    uncertainty = lib.sdc.setup_uncertainty
    budget = (params.read_cycles + 1) * timing.clock_ns - uncertainty
    if params.handshake:
        budget = SYNC_CYCLES * timing.clock_ns - uncertainty
    loopback = {}
    for corner, lib_corner in (() if params.refresh_in_macro else lib.corners.items()):
        # (none with refresh_in_macro: a refresh's data never leaves the macro)
        spef = run.final / "spef" / lib.rc_corner / f"{name}_top.{lib.rc_corner}.spef"
        arrival = loopback_arrival(
            tech, corner, run.netlist, f"{name}_top", sdc, spef, liberty[corner], outdir / f"{name}_sta"
        )
        if params.handshake:
            arrival += timing.rdy_lag_ns - reads[corner]
        loopback[corner] = {"arrival_ns": arrival, "budget_ns": budget, "slack_ns": budget - arrival}
    summary["loopback"] = loopback
    if any(v["slack_ns"] < 0 for v in loopback.values()):
        raise FlowError(f"the refresh loop-back misses its cycle: {loopback}")

    # the routed netlist under the generated testbench, the macro as its behavioural model
    gate_level = simulate_gates(
        tech, run.netlist, [rtl[f"{name}_model.v"], rtl[f"{name}_tb.v"]], outdir / f"{name}_gatesim"
    )

    # the deliverables, next to the RTL
    top = f"{name}_top"
    copies = {
        f"{top}.gds": run.gds, f"{top}.lef": run.lef, f"{top}.v": run.netlist,
        f"{top}.pnl.v": run.output("pnl", "pnl.v"),
    }
    for corner in lib.corners:
        copies[f"{top}_{corner}.lib"] = run.liberty(corner)
    for filename, source in copies.items():
        if source.exists():
            shutil.copyfile(source, outdir / filename)
    bits = cfg.bits * (2 if banked else 1)
    die = summary["die_area_um2"]
    metrics = {
        **summary,
        "bits": bits,
        "macro_area_um2": hard.width * hard.height,
        "controller_cell_area_um2": summary["cell_area_um2"],
        "area_per_bit_um2": die / bits,
        "clock_ns": timing.clock_ns,
        "refresh_interval_us": timing.interval_us,
        "read_ns": reads,
        "handshake": timing.handshake,
        "read_cycles": params.read_cycles,
    }
    (outdir / f"{top}.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    return MemoryResult(top, outdir, run, views, gate_level, metrics)
