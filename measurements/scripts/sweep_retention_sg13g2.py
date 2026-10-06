"""SG13G2's refresh intervals, measured as sky130's are (``sweep_retention.py``): the true
read-tolerance threshold (``zero_level_limit`` from a generous bound) and the time a written '0'
takes to drift up to it on the extracted 16x20 macro (``macro_refresh_interval``), at the far cell.

    python sweep_retention_sg13g2.py options      # cell options at tt and ss  -> data/retention_sg13g2.json
    python sweep_retention_sg13g2.py mc_ss [N]    # N mismatch dies at ss      -> data/retention_mc_ss_sg13g2.json
    python sweep_retention_sg13g2.py neg          # with a -0.12 V write rail, ss -> data/retention_neg_sg13g2.json
    python sweep_retention_sg13g2.py mc_neg [N]   # ... N mismatch dies at ss  -> data/retention_mc_neg_sg13g2.json

What the adapter's ``CHARACTERISED`` and default cell come from (mr 1.0 um: 121 us nominal at ss,
100 us the worst of 20 dies; the plain cell 35 / 26.9 us).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "retention_sg13g2"

#: The negative write rail's level: vdd - vneg at the LV devices' 1.32 V rating.
VNEG = -0.12

OPTIONS = {
    # name: bitcell overrides on the technology's default cell
    "plain": {"mr_length": None},
    "mr0.5": {"mr_length": 0.5},
    "default": {},  # mr 1.0
    "w0.3_mr0.5": {"write_length": 0.3, "mr_length": 0.5},
    "w0.5_mr1.0": {"write_length": 0.5, "mr_length": 1.0},
}


def _model(tech, name: str, bitcell: dict, negative: bool = False):
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.verify.timing import build_model

    cfg = ArrayConfig.for_tech(
        tech, rows=16, cols=20, replica_column=True, self_timed_read=False, wordline_straps=False,
        wordline_driver_strength=1.0, bitcell=BitcellConfig.for_tech(tech, **bitcell),
        negative_write_wl=negative, write_wl_low=VNEG if negative else tech.sizing.write_wl_low,
    )
    work = WORK / name
    work.mkdir(parents=True, exist_ok=True)
    cell = f"ret_sg13_{name.replace('.', 'p')}"
    gds = work / f"{cell}.gds"
    gc_macro(tech, cfg, name=cell, decoded=True).write_gds(gds)
    return build_model(tech, cfg, gds, cell, work / "model")


def _point(tech, model, corner, workdir: Path, settings) -> dict:
    from gc_compiler.verify.timing import macro_refresh_interval, zero_level_limit

    row, col = 0, model.kept[-1]
    threshold = zero_level_limit(tech, model, corner, row, col, corner.vdd, workdir / "limit", settings)
    result = macro_refresh_interval(tech, model, corner, row, col, threshold, workdir / "refresh", settings)
    return {
        "corner": corner.name, "seed": settings.seed, "threshold_v": threshold, "settled_v": result.settled_v,
        "margin_v": result.margin_v,
        "refresh_us": None if result.refresh_time_s is None else result.refresh_time_s * 1e6,
    }


def main(mode: str, dies: int = 20) -> None:
    from gc_compiler.tech import technology
    from gc_compiler.verify.timing import BenchSettings, standard_corners

    tech = technology("ihp-sg13g2")
    base = BenchSettings()
    corners = {c.process: c for c in standard_corners(tech)}
    if mode == "options":
        models = {name: _model(tech, name, bitcell) for name, bitcell in OPTIONS.items()}
        jobs = [(name, corners[p], base) for name in models for p in ("tt", "ss")]
    elif mode == "mc_ss":
        models = {name: _model(tech, name, OPTIONS[name]) for name in ("default", "plain")}
        jobs = [(name, corners["ss"], replace(base, seed=seed)) for name in models for seed in range(1, dies + 1)]
    elif mode in ("neg", "mc_neg"):
        models = {f"{name}_neg": _model(tech, f"{name}_neg", OPTIONS[name], negative=True) for name in ("default", "plain")}
        seeds = [None] if mode == "neg" else range(1, dies + 1)
        jobs = [(name, corners["ss"], replace(base, seed=seed)) for name in models for seed in seeds]
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
    out = {
        "options": "retention_sg13g2.json", "mc_ss": "retention_mc_ss_sg13g2.json",
        "neg": "retention_neg_sg13g2.json", "mc_neg": "retention_mc_neg_sg13g2.json",
    }[mode]
    (DATA / out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "options", int(sys.argv[2]) if len(sys.argv) > 2 else 20)
