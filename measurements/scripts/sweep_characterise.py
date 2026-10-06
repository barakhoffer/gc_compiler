"""The corner tables and Monte Carlo of every option, on the extracted 16x20 macro with replica.

    python sweep_characterise.py              # every option -> data/characterise.json
    python sweep_characterise.py <option> [N] # one option (N Monte-Carlo dies per corner, default 50)

Per option and corner: the full ``characterise()`` (writes, reads from fresh and decayed '1's and
the written '0', the '0' limit, idle supply) and ``N`` local-mismatch dies, plus the *true*
read tolerance (``zero_level_limit`` bisected from vdd, not from the written level).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "characterise"

OPTIONS = {
    "plain": ({}, {}),
    "negwl": ({"negative_write_wl": True, "write_wl_low": -0.8}, {}),
    "lvt": ({}, {"write_flavor": "lvt", "write_length": 0.35, "read_length": 0.5}),
    "lvt_mr": ({}, {"write_flavor": "lvt", "write_length": 0.35, "mr_length": 0.5}),
    "lvt_r1": ({}, {"write_flavor": "lvt", "write_length": 0.35, "read_length": 1.0}),
    "lvt_w": ({}, {"write_flavor": "lvt", "write_length": 0.35}),
}


def run_one(option: str, dies: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from gc_compiler.compiler import GainCellCompiler
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.tech.sky130 import Sky130
    from gc_compiler.verify.timing import BenchSettings, build_model, standard_corners, zero_level_limit

    array, bitcell = OPTIONS[option]
    tech = Sky130()
    cfg = ArrayConfig(rows=16, cols=20, replica_column=True, self_timed_read=False, wordline_straps=False, wordline_driver_strength=1.0, bitcell=BitcellConfig.plain(**bitcell), **array)
    work = WORK / option
    compiler = GainCellCompiler(cfg, tech)
    result = compiler.build(work, name=f"char_{option}", macro=True, decoded=True, banked=False)
    report = compiler.characterise(result, workdir=work / "timing", monte_carlo=dies)
    out = report.to_dict()
    # the true read tolerance per corner, independent of what the write left
    model = build_model(tech, cfg, result.gds, result.component.name, work / "tolerance")
    corners = standard_corners()
    with ThreadPoolExecutor(max_workers=5) as pool:
        limits = [
            pool.submit(zero_level_limit, tech, model, c, 0, model.kept[-1], c.vdd, work / "tolerance" / c.name, BenchSettings())
            for c in corners
        ]
        for entry, future in zip(out["corners"], limits):
            entry["read_tolerance_v"] = future.result()
    out["option"] = option
    out["area_per_bit_um2"] = result.metrics["macro_area_per_bit_um2"]
    return out


def main(dies: int = 50) -> None:
    DATA.mkdir(exist_ok=True)
    path = DATA / "characterise.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for option in OPTIONS:
        if option in results:
            continue
        proc = subprocess.run([sys.executable, __file__, option, str(dies)], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"{option}: FAILED\n{proc.stderr[-3000:]}", flush=True)
            continue
        results[option] = json.loads(proc.stdout.strip().splitlines()[-1])
        path.write_text(json.dumps(results, indent=1) + "\n")
        print(f"{option}: done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in OPTIONS:
        print(json.dumps(run_one(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
