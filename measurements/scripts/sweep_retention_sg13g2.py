"""SG13G2's refresh intervals, measured as sky130's are (``sweep_retention.py``): the true
read-tolerance threshold (``zero_level_limit`` from a generous bound) and the time a written '0'
takes to drift up to it on the extracted 16x20 macro (``macro_refresh_interval``), at the far cell.

    python sweep_retention_sg13g2.py options      # cell options at tt and ss  -> data/retention_sg13g2.json
    python sweep_retention_sg13g2.py mc_ss [N]    # N mismatch dies at ss      -> data/retention_mc_ss_sg13g2.json
    python sweep_retention_sg13g2.py neg          # with a -0.12 V write rail, ss -> data/retention_neg_sg13g2.json
    python sweep_retention_sg13g2.py mc_neg [N]   # ... N mismatch dies at ss  -> data/retention_mc_neg_sg13g2.json
    python sweep_retention_sg13g2.py rim          # with the in-macro refresh, tt and ss -> data/retention_rim_sg13g2.json
    python sweep_retention_sg13g2.py mc_rim [N]   # ... N mismatch dies at ss, with and without the -0.12 V rail
                                                  #                            -> data/retention_mc_rim_sg13g2.json

What the adapter's ``CHARACTERISED`` and default cell come from (mr 1.0 um: 121 us nominal at ss,
100 us the worst of 20 dies; the plain cell 35 / 26.9 us).

With the in-macro refresh (``rim``) a refresh writes back the level its read bit line reached, and
in the controller's 2 ns write only part of the way: its interval is the drift from the level a
refresh leaves a '0' at up to the highest level a refresh still brings back down
(``refresh_zero_limit``), when that is below the read's own limit.
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

#: A refresh's write with the in-macro refresh (``rim``), ns: the controller's shortest (its
#: ``write_ns``) and one clock cycle at 100 MHz. SG13G2's PMOS write device, its word line 0.12 V below
#: ground, takes ~10 ns to settle a '0': in 2 ns it pulls a decayed one only part of the way down.
WRITE_NS = (2.0, 10.0)

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


def _model(tech, name: str, bitcell: dict, negative: bool = False, rim: bool = False):
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.verify.timing import build_model

    cfg = ArrayConfig.for_tech(
        tech, rows=16, cols=20, replica_column=True, self_timed_read=rim, refresh_in_macro=rim, wordline_straps=False,
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


def _point_rim(tech, model, corner, workdir: Path, settings) -> dict:
    """With the in-macro refresh: a refresh writes back the level the read bit line reached, and
    in a short write only part of the way, so its interval is not the drift from the written level
    to the read's limit but, for a threshold ``L``, the drift from where a refresh leaves a '0' at
    ``L`` up to ``L`` -- the best ``L`` on a 20 mV grid below the read's limit, for each write time."""
    import numpy as np

    from gc_compiler.verify.timing import crossing, macro_refresh_interval, simulate_refresh, zero_level_limit

    row, col = 0, model.kept[-1]
    read_limit = zero_level_limit(tech, model, corner, row, col, corner.vdd, workdir / "limit", settings)
    drift = macro_refresh_interval(tech, model, corner, row, col, read_limit, workdir / "drift", settings)
    # the whole drift curve from the longest hold's waveform (t, we, t, sn)
    wave = np.loadtxt(workdir / "drift" / f"a{drift.attempts - 1}" / "wave.csv")
    t, sn = wave[:, 0], wave[:, 3]
    after = settings.settle + settings.window + 2 * settings.slew

    def time_to(level: float) -> float | None:
        if level <= drift.settled_v:
            return 0.0
        hit = crossing(t, sn, level, True, after)
        return None if hit is None else hit - after

    levels = [round(v, 3) for v in np.arange(drift.settled_v + 0.02, read_limit + 1e-9, 0.02)]
    curve = {str(v): time_to(v) for v in levels}
    best = {}
    for write_ns in WRITE_NS:
        points = []
        for level in levels:
            r = simulate_refresh(
                tech, model, corner, row, {col: (0, level)}, workdir / f"rf_{level:.3f}_{write_ns:g}", settings,
                write_time=write_ns * 1e-9,
            )[col]
            reach, leave = time_to(level), time_to(r.sn_final) if r.sn_final < level else None
            interval = None if reach is None else (reach - leave if leave is not None else 0.0)
            points.append({"threshold_v": level, "after_refresh_v": r.sn_final, "interval_us": None if interval is None else interval * 1e6})
        known = [p for p in points if p["interval_us"] is not None]
        top = max(known, key=lambda p: p["interval_us"]) if known else None
        best[f"{write_ns:g}"] = {"best": top, "points": points}
    return {
        "corner": corner.name, "seed": settings.seed, "read_limit_v": read_limit, "settled_v": drift.settled_v,
        "read_refresh_us": None if drift.refresh_time_s is None else drift.refresh_time_s * 1e6,
        "drift_us": {k: None if v is None else v * 1e6 for k, v in curve.items()}, "write_ns": best,
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
    elif mode in ("rim", "mc_rim"):
        names = ("default", "plain") if mode == "rim" else ("default", "plain", "default_neg", "plain_neg")
        models = {
            f"{name}_rim": _model(tech, f"{name}_rim", OPTIONS[name.removesuffix("_neg")], negative=name.endswith("_neg"), rim=True)
            for name in names
        }
        if mode == "rim":
            jobs = [(name, corners[p], base) for name in models for p in ("tt", "ss")]
        else:
            jobs = [(name, corners["ss"], replace(base, seed=seed)) for name in models for seed in range(1, dies + 1)]
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    point = _point_rim if mode in ("rim", "mc_rim") else _point
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [
            (name, pool.submit(point, tech, models[name], corner, WORK / name / mode / f"{corner.name}_s{s.seed}", s))
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
        "rim": "retention_rim_sg13g2.json", "mc_rim": "retention_mc_rim_sg13g2.json",
    }[mode]
    (DATA / out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "options", int(sys.argv[2]) if len(sys.argv) > 2 else 20)
