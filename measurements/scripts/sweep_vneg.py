"""The negative rail's level: every corner plus local-mismatch dies, and the refresh interval per
corner, per vneg, on the extracted 16x20 macro with replica (the corrected bench; the -0.8 V
default was chosen on the old one).

    python sweep_vneg.py            # every level -> data/vneg.json
    python sweep_vneg.py <vneg> [N] # one level (N Monte-Carlo dies per corner, default 20)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "vneg"

LEVELS = (-0.2, -0.3, -0.4, -0.5, -0.6)  # -0.8 is the negwl option in characterise.json


def run_one(vneg: float, dies: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from gc_compiler.compiler import GainCellCompiler
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.tech.sky130 import Sky130
    from gc_compiler.verify.timing import BenchSettings, build_model, standard_corners

    tech = Sky130()
    cfg = ArrayConfig(
        rows=16, cols=20, replica_column=True, negative_write_wl=True, write_wl_low=vneg,
        bitcell=BitcellConfig.plain(), self_timed_read=False, wordline_straps=False, wordline_driver_strength=1.0,
    )
    tag = f"vneg{abs(vneg):.1f}".replace(".", "p")
    work = WORK / tag
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(work, name=f"char_{tag}", macro=True, decoded=True, banked=False)
    out = compiler.characterise(result, workdir=work / "timing", monte_carlo=dies).to_dict()
    # the read tolerance and the refresh interval per corner, as sweep_retention measures them
    sys.path.insert(0, str(HERE))
    from sweep_retention import _point

    model = build_model(tech, cfg, result.gds, result.component.name, work / "retention")
    with ThreadPoolExecutor(max_workers=5) as pool:
        points = [
            pool.submit(_point, tech, model, c, work / "retention" / c.name, BenchSettings())
            for c in standard_corners()
        ]
        for entry, future in zip(out["corners"], points):
            point = future.result()
            entry["read_tolerance_v"] = point["threshold_v"]
            entry["retention"] = point
    out["vneg"] = vneg
    out["area_per_bit_um2"] = result.metrics["macro_area_per_bit_um2"]
    return out


def main(dies: int = 20) -> None:
    DATA.mkdir(exist_ok=True)
    path = DATA / "vneg.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for vneg in LEVELS:
        key = f"{vneg:.1f}"
        if key in results:
            continue
        proc = subprocess.run([sys.executable, __file__, key, str(dies)], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"{key}: FAILED\n{proc.stderr[-3000:]}", flush=True)
            continue
        results[key] = json.loads(proc.stdout.strip().splitlines()[-1])
        path.write_text(json.dumps(results, indent=1) + "\n")
        print(f"{key}: done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        print(json.dumps(run_one(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 20)))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
