"""E1: energy per access, leakage and refresh power of every variant, on the extracted macro.

    python sweep_energy.py            # every variant, one subprocess each -> data/energy.json
    python sweep_energy.py <variant>  # one variant (what the loop runs)

Every column is kept as a real cell (a reduced model drops most of a read's energy), at tt.
Refresh intervals are the extracted-macro ones from data/retention.json when present.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "energy"

VARIANTS = {
    # name: (ArrayConfig overrides, bitcell overrides, banked)
    "plain_16x20": ({"rows": 16, "cols": 20}, {}, False),
    "negwl_16x20": ({"rows": 16, "cols": 20, "negative_write_wl": True, "write_wl_low": -0.8}, {}, False),
    "lvt_16x20": ({"rows": 16, "cols": 20}, {"write_flavor": "lvt", "write_length": 0.35, "read_length": 0.5}, False),
    "lvt_mr_16x20": ({"rows": 16, "cols": 20}, {"write_flavor": "lvt", "write_length": 0.35, "mr_length": 0.5}, False),
    "lvt_w_16x20": ({"rows": 16, "cols": 20}, {"write_flavor": "lvt", "write_length": 0.35}, False),
    "plain_32x32": ({"rows": 32, "cols": 32}, {}, False),
    "banked_32x32": ({"rows": 32, "cols": 32}, {}, True),
}


def run_one(name: str) -> dict:
    from gc_compiler.array.floorplan import gc_banked_macro, gc_macro
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.tech.sky130 import Sky130
    from gc_compiler.verify.parasitics import extract_flat
    from gc_compiler.verify.timing import access_energy, build_model, standard_corners

    array, bitcell, banked = VARIANTS[name]
    tech = Sky130()
    cfg = ArrayConfig(**array, replica_column=True, self_timed_read=False, wordline_straps=False, wordline_driver_strength=1.0, bitcell=BitcellConfig.plain(**bitcell))
    work = WORK / name
    work.mkdir(parents=True, exist_ok=True)
    macro = gc_banked_macro(tech, cfg, name=name) if banked else gc_macro(tech, cfg, name=name, decoded=True)
    gds = work / f"{name}.gds"
    macro.write_gds(gds)
    flat = extract_flat(gds, name, work / "extract")
    tt = next(c for c in standard_corners() if c.process == "tt")
    banks = []
    for bank in ((0, 1) if banked else (None,)):
        model = build_model(tech, cfg, gds, name, work / f"model{bank}", keep=tuple(range(cfg.cols)), bank=bank, flat=flat)
        banks.append(access_energy(tech, model, tt, work / f"bank{bank}").to_dict())
    return {
        "variant": name, "rows": cfg.rows, "cols": cfg.cols, "banked": banked,
        "bits": macro.info["bits"], "area_per_bit_um2": macro.info["area_per_bit"], "banks": banks,
    }


def main() -> None:
    DATA.mkdir(exist_ok=True)
    out = DATA / "energy.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for name in VARIANTS:
        if name in results:
            continue
        proc = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"{name}: FAILED\n{proc.stderr[-3000:]}")
            continue
        results[name] = json.loads(proc.stdout.strip().splitlines()[-1])
        out.write_text(json.dumps(results, indent=1) + "\n")
        print(f"{name}: done")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(run_one(sys.argv[1])))
    else:
        main()
