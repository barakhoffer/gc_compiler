"""Refresh written back in the macro (``ArrayConfig.refresh_in_macro``) against the controller's
loop-back, on sky130 and SG13G2 -- what decides whether it becomes the default.

    python compare_refresh_in_macro.py [sky130A] [ihp-sg13g2]   -> data/refresh_in_macro.json
    python compare_refresh_in_macro.py din [sky130A] [ihp-sg13g2]   (the in-place refresh's
        characterisation, energy and mismatch dies again, the rest kept: after the controller came
        to hold din low outside writes)

Per technology, flag off and on: the area of a 16x20 one-bank and an 8x8 banked macro; the 16x20
macro's characterisation at the five corners (read, write, precharge; with the flag, the in-place
refresh at each corner); energy per access at tt with every column kept (a refresh against a read
plus a write); and, with the flag, the in-place refresh under 20 local-mismatch dies at ss (the
write driver's NAND must judge each sensed bit as the read buffer does).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "refresh_in_macro"
DIES = 20


def _macro(tech, rim: bool, rows: int, cols: int, banked: bool, name: str):
    from gc_compiler.array.floorplan import gc_banked_macro, gc_macro
    from gc_compiler.config import ArrayConfig

    cfg = ArrayConfig.for_tech(tech, rows=rows, cols=cols, refresh_in_macro=rim)
    macro = gc_banked_macro(tech, cfg, name=name) if banked else gc_macro(tech, cfg, name=name, decoded=True)
    return cfg, macro


def _measure(tech, rim: bool, din_only: bool = False, out: dict | None = None) -> dict:
    from gc_compiler.verify.timing import (
        BenchSettings, access_energy, build_model, characterise, simulate_refresh, standard_corners,
    )

    tag = f"{tech.name.replace('-', '_')}_{'rim' if rim else 'loop'}"
    work = WORK / tag
    work.mkdir(parents=True, exist_ok=True)
    out = dict(out or {})
    out.update({"technology": tech.name, "refresh_in_macro": rim})

    for key, rows, cols, banked in (() if din_only else (("one_bank_16x20", 16, 20, False), ("banked_8x8", 8, 8, True))):
        cfg, macro = _macro(tech, rim, rows, cols, banked, f"cmp_{tag}_{key}")
        box = macro.dbbox()
        bits = cfg.bits * (2 if banked else 1)
        out[key] = {"width_um": box.width(), "height_um": box.height(), "um2_per_bit": box.width() * box.height() / bits}

    cfg, macro = _macro(tech, rim, 16, 20, False, f"cmp_{tag}_char")
    gds = work / f"{macro.name}.gds"
    macro.write_gds(gds)
    report = characterise(tech, cfg, gds, macro.name, work / "char")
    report.save(work / "char.json")
    out["corners"] = {
        c.corner.name: {
            "access_ns": None if c.access is None else c.access * 1e9,
            "write_ns": None if c.write_time is None else c.write_time * 1e9,
            "precharge_ns": None if c.precharge is None else c.precharge * 1e9,
            "zero_limit_v": c.zero_limit,
            "refresh_ok": c.refresh_ok,
            "refresh_ns": max(r.t_access for r in c.refresh) * 1e9 if c.refresh else None,
        }
        for c in report.corners
    }

    corners = {c.process: c for c in standard_corners(tech)}
    full = build_model(tech, cfg, gds, macro.name, work / "energy_model", keep=tuple(range(cfg.cols)))
    energy = access_energy(tech, full, corners["tt"], work / "energy")
    out["energy_tt"] = {
        "read1_pj": energy.read1_j * 1e12, "write_pj": energy.write_j * 1e12,
        "refresh_pj": None if energy.refresh_j is None else energy.refresh_j * 1e12,
    }

    if rim:
        # the in-place refresh under mismatch at ss: a '1' decayed to 0.7 vdd and a '0' at the read
        # limit, near and far, die by die
        ss = corners["ss"]
        zero = next(c for c in report.corners if c.corner.process == "ss").zero_limit
        model = build_model(tech, cfg, gds, macro.name, work / "mc_model")
        near, far = model.kept[0], model.kept[-1]
        jobs = []
        with ThreadPoolExecutor(max_workers=12) as pool:
            for seed in range(1, DIES + 1):
                settings = replace(BenchSettings(), seed=seed)
                for i, cells in enumerate(({near: (1, 0.7 * ss.vdd), far: (0, zero)}, {near: (0, zero), far: (1, 0.7 * ss.vdd)})):
                    jobs.append((seed, pool.submit(simulate_refresh, tech, model, ss, 0, cells, work / "mc" / f"s{seed}_{i}", settings)))
            dies = {}
            for seed, future in jobs:
                for r in future.result().values():
                    dies.setdefault(seed, []).append({"col": r.col, "data": r.data, "sn_final": r.sn_final, "restored": r.restored})
        out["mc_ss"] = {
            "dies": DIES, "zero_limit_v": zero,
            "failing_dies": sorted(s for s, rs in dies.items() if not all(r["restored"] for r in rs)),
            "worst_one_v": min(r["sn_final"] for rs in dies.values() for r in rs if r["data"]),
            "worst_zero_v": max(r["sn_final"] for rs in dies.values() for r in rs if not r["data"]),
        }
    return out


def main(names: list[str], din_only: bool = False) -> None:
    from gc_compiler.tech import technology

    path = DATA / "refresh_in_macro.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        tech = technology(name)
        for rim in ((True,) if din_only else (False, True)):
            key = f"{name}/{'rim' if rim else 'loop'}"
            results[key] = _measure(tech, rim, din_only, results.get(key))
            print(key, json.dumps(results[key])[:400], flush=True)
            DATA.mkdir(exist_ok=True)
            path.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    din_only = bool(args) and args[0] == "din"
    main([a for a in args if a != "din"] or ["sky130A", "ihp-sg13g2"], din_only)
