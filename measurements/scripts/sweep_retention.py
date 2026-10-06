"""E3/E4 and the retention table: refresh interval on the extracted 16x20 macro.

    python sweep_retention.py corners   # every option at the five standard corners -> data/retention.json
    python sweep_retention.py temp      # every option at tt, -40..125 C             -> data/retention_temp.json
    python sweep_retention.py mc [N]    # N mismatch dies at sf (default 20)         -> data/retention_mc.json
    python sweep_retention.py mc_ss [N] # N mismatch dies at ss, the worst nominal corner -> data/retention_mc_ss.json

A point is the true read-tolerance threshold (``zero_level_limit`` bisected from a generous
bound, not the written level) and the time a written '0' takes to drift up to it on the real
macro (``macro_refresh_interval``), at the far cell (row 0, last column).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "retention"

OPTIONS = {
    # name: (ArrayConfig overrides, bitcell overrides)
    "plain": ({}, {}),
    "negwl": ({"negative_write_wl": True, "write_wl_low": -0.8}, {}),
    "lvt_w": ({}, {"write_flavor": "lvt", "write_length": 0.35}),
    "lvt": ({}, {"write_flavor": "lvt", "write_length": 0.35, "read_length": 0.5}),
    "lvt_mr": ({}, {"write_flavor": "lvt", "write_length": 0.35, "mr_length": 0.5}),
    "lvt_r1": ({}, {"write_flavor": "lvt", "write_length": 0.35, "read_length": 1.0}),
}
TEMPS = (-40.0, 0.0, 25.0, 55.0, 85.0, 125.0)


def _models():
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.tech.sky130 import Sky130
    from gc_compiler.verify.timing import build_model

    tech = Sky130()
    models = {}
    for name, (array, bitcell) in OPTIONS.items():
        cfg = ArrayConfig(rows=16, cols=20, replica_column=True, self_timed_read=False, wordline_straps=False, wordline_driver_strength=1.0, bitcell=BitcellConfig.plain(**bitcell), **array)
        work = WORK / name
        work.mkdir(parents=True, exist_ok=True)
        cell = f"ret_{name}_16x20"
        gds = work / f"{cell}.gds"
        gc_macro(tech, cfg, name=cell, decoded=True).write_gds(gds)
        models[name] = build_model(tech, cfg, gds, cell, work / "model")
    return tech, models


def _point(tech, model, corner, workdir: Path, settings) -> dict:
    from gc_compiler.verify.timing import macro_refresh_interval, zero_level_limit

    row, col = 0, model.kept[-1]
    threshold = zero_level_limit(tech, model, corner, row, col, corner.vdd, workdir / "limit", settings)
    result = macro_refresh_interval(tech, model, corner, row, col, threshold, workdir / "refresh", settings)
    return {
        "corner": corner.name, "process": corner.process, "vdd": corner.vdd, "temp": corner.temp,
        "seed": settings.seed, "threshold_v": threshold, "settled_v": result.settled_v,
        "margin_v": result.margin_v, "refresh_s": result.refresh_time_s,
    }


def main(mode: str, dies: int = 20) -> None:
    from gc_compiler.verify.timing import BenchSettings, Corner, standard_corners

    tech, models = _models()
    base = BenchSettings()
    jobs = []
    if mode == "corners":
        jobs = [(name, corner, base) for name in models for corner in standard_corners()]
    elif mode == "temp":
        jobs = [(name, Corner("tt", 1.8, temp), base) for name in models for temp in TEMPS]
    elif mode in ("mc", "mc_ss"):
        process = "sf" if mode == "mc" else "ss"
        corner = next(c for c in standard_corners() if c.process == process)
        jobs = [(name, corner, replace(base, seed=seed)) for name in models for seed in range(1, dies + 1)]
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [
            (name, pool.submit(_point, tech, models[name], corner, WORK / name / mode / f"{corner.name}_s{s.seed}", s))
            for name, corner, s in jobs
        ]
        results: dict[str, list] = {name: [] for name in models}
        for name, future in futures:
            results[name].append(future.result())
            print(name, results[name][-1], flush=True)
    DATA.mkdir(exist_ok=True)
    suffix = {"corners": "", "temp": "_temp", "mc": "_mc", "mc_ss": "_mc_ss"}[mode]
    (DATA / f"retention{suffix}.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "corners", int(sys.argv[2]) if len(sys.argv) > 2 else 20)
