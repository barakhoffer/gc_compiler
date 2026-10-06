"""The refresh controller's area against its flow settings, on sky130 and SG13G2 -- what the
adapters' ``StdCellLibrary.flow_config`` / ``synth_excluded_cells`` come from.

    python sweep_controller_area.py synth [sky130A] [ihp-sg13g2]                      -> data/controller_area_synth.json
    python sweep_controller_area.py block [sky130A] [ihp-sg13g2] [variant ...] [wide] -> data/controller_area_block.json

``synth``: the controller alone, synthesized (LibreLane to Yosys.Synthesis) for an 8x8 one-bank, an
8x8 banked and a 16x64 banked macro, under each of ``VARIANTS``: cell area, flops, latches and
multiplexers from Yosys' statistics.

``block``: the 8x8 one-bank and banked hard blocks (with ``wide``, an 8x64 banked one too),
placed and routed, under each of ``VARIANTS``: controller cell area (by class: logic, flops, clock
tree), die area, timing, DRC/LVS, testbench.

The data files also keep the entries measured with the earlier controller, whose address, write
data and read data were flops (``before``, ``after*``, ``*_root*``, ``noreset*``, ``final*``; in
``synth`` ``reset``/``noreset``/``area1-3``/``clockgate*``/``drive1``): the 4x clock-tree root and
sky130's drive-1 cells come from those (area-optimising synthesis gained nothing), and the clock
gating they measured has nothing left to gate. The current controller's entries are ``latch*``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "controller_area"

TECHS = ("sky130A", "ihp-sg13g2")
BLOCKS = {"one_bank_8x8": (8, 8, False), "banked_8x8": (8, 8, True), "banked_16x64": (16, 64, True)}
ROUTED = {"one_bank_8x8": (8, 8, False), "banked_8x8": (8, 8, True)}
WIDE = {"banked_8x64": (8, 64, True)}

#: ``latch``: the adapter's flow settings; ``latch_pdk``: the PDK's.
VARIANTS = ("latch_pdk", "latch")


def _library(lib, variant: str):
    if variant == "latch":
        return lib
    if variant == "latch_pdk":
        return replace(lib, flow_config={}, synth_excluded_cells=None)
    raise ValueError(f"unknown variant {variant!r}")


def _stat(run_dir: Path) -> dict:
    step = next(run_dir.glob("*-yosys-synthesis"))
    stat = json.loads((step / "reports" / "stat.json").read_text())
    top = next(m for m in stat["modules"].values() if "area" in m)
    cells = top.get("num_cells_by_type", {})
    count = lambda *keys: sum(n for c, n in cells.items() if any(k in c for k in keys))  # noqa: E731
    return {
        "cell_area_um2": top["area"],
        "sequential_um2": top.get("sequential_area"),
        "flops": count("df"),
        "latches": count("dlx", "dlh", "dll"),
        "muxes": count("mux2"),
        "cells": cells,
    }


def synth(name: str, only: list[str]) -> dict:
    from gc_compiler.config import ArrayConfig
    from gc_compiler.flow.librelane import controller_config, write_config
    from gc_compiler.rtl import ControllerTiming, controller, controller_params, controller_sdc
    from gc_compiler.tech import technology

    tech = technology(name)
    base = tech.standard_cells()
    timing = ControllerTiming.default(tech)
    jobs = {}
    for block, (rows, cols, banked) in BLOCKS.items():
        cfg = ArrayConfig.for_tech(tech, rows=rows, cols=cols)
        params = controller_params(cfg, timing, banked)
        for variant in only or VARIANTS:
            work = WORK / name / block / variant
            work.mkdir(parents=True, exist_ok=True)
            (work / "c_ctrl.v").write_text(controller("c", params, timing))
            (work / "c_ctrl.sdc").write_text(controller_sdc(params, timing, sdc=base.sdc))
            config = controller_config("c_ctrl", work / "c_ctrl.v", work / "c_ctrl.sdc", timing.clock_ns)
            write_config(_library(base, variant), config, work)
            jobs[(block, variant)] = work
    argv = [
        "librelane", "--manual-pdk", "--pdk-root", base.pdk_root, "--pdk", base.pdk, "--scl", base.name,
        "--run-tag", "synth", "--overwrite", "--to", "Yosys.Synthesis", "config.json",
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        done = {key: pool.submit(subprocess.run, argv, cwd=work, capture_output=True, text=True) for key, work in jobs.items()}
        out: dict = {}
        for (block, variant), future in done.items():
            proc = future.result()
            entry = out.setdefault(block, {})
            try:
                entry[variant] = _stat(jobs[(block, variant)] / "runs" / "synth")
            except (StopIteration, FileNotFoundError):
                entry[variant] = {"error": proc.stdout[-1500:] + proc.stderr[-1500:]}
            print(name, block, variant, {k: v for k, v in entry[variant].items() if k != "cells"}, flush=True)
    return out


def block(name: str, only: list[str]) -> dict:
    from gc_compiler.compiler import GainCellCompiler
    from gc_compiler.config import ArrayConfig
    from gc_compiler.tech import technology

    blocks = {**ROUTED, **WIDE} if "wide" in only else ROUTED
    only = [v for v in only if v != "wide"]
    tech = technology(name)
    adapted = tech.standard_cells()
    path = DATA / "controller_area_block.json"
    done = {k for k, v in (json.loads(path.read_text()).get(name, {}) if path.exists() else {}).items() if "error" not in v}
    out = {}
    for variant in only or VARIANTS:
        lib = _library(adapted, variant)
        for key, (rows, cols, banked) in blocks.items():
            if f"{key}/{variant}" in done:  # measured already
                continue
            tech.standard_cells = lambda root=None, lib=lib: lib  # noqa: E731
            work = WORK / "block" / name / f"{key}_{variant}"
            try:
                cfg = ArrayConfig.for_tech(tech, rows=rows, cols=cols)
                # (a cell name of its own each: one process may build several variants)
                result = GainCellCompiler(cfg, tech).memory(work, name=f"ca_{key}_{variant}", banked=banked)
                m = result.metrics
                out[f"{key}/{variant}"] = {
                    k: m[k] for k in ("controller_cell_area_um2", "die_area_um2", "macro_area_um2", "flops", "logic_cells",
                                      "timing", "drc_errors", "lvs_errors", "area_per_bit_um2")
                }
                metrics = json.loads((Path(m["run_dir"]) / "final" / "metrics.json").read_text())
                out[f"{key}/{variant}"]["area_by_class"] = {
                    k.split("class:")[1]: v for k, v in metrics.items() if k.startswith("design__instance__area__class:")
                }
            except Exception as error:  # noqa: BLE001 -- recorded, the sweep goes on
                out[f"{key}/{variant}"] = {"error": str(error)[-3000:]}
            finally:
                tech.__dict__.pop("standard_cells", None)
            print(name, key, variant, json.dumps(out[f"{key}/{variant}"])[:600], flush=True)
    return out


def main(mode: str, names: list[str]) -> None:
    path = DATA / f"controller_area_{mode}.json"
    techs = [n for n in names if n in TECHS] or list(TECHS)
    only = [n for n in names if n not in TECHS]
    for name in techs:
        result = synth(name, only) if mode == "synth" else block(name, only)
        # (re-read and merge: one process per technology or variant can share the file)
        results = json.loads(path.read_text()) if path.exists() else {}
        if mode == "synth":
            for key, entries in result.items():
                results.setdefault(name, {}).setdefault(key, {}).update(entries)
        else:
            results.setdefault(name, {}).update(result)
        DATA.mkdir(exist_ok=True)
        path.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
