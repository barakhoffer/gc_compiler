"""E2: area and tt timing of every variant across sizes, plain and banked.

    python sweep_scaling.py         # every point, one subprocess each -> data/scaling.json
    python sweep_scaling.py <key>   # one point: "<variant>:<rows>x<cols>:<plain|banked>"

Timing (worst read of a fresh '1' over the first/last row and both end columns, and a write
into the last row's far cell, at tt, on the reduced extracted model) up to TIMING_MAX_ROWS.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "scaling"

BITCELLS = {
    "plain": ({}, {}),
    "negwl": ({"negative_write_wl": True, "write_wl_low": -0.8}, {}),
    "lvt": ({}, {"write_flavor": "lvt", "write_length": 0.35, "read_length": 0.5}),
    "lvt_mr": ({}, {"write_flavor": "lvt", "write_length": 0.35, "mr_length": 0.5}),
    "lvt_w": ({}, {"write_flavor": "lvt", "write_length": 0.35}),
}
SIZES = ((16, 20), (32, 32), (64, 32), (128, 64))
TIMING_MAX_ROWS = 64


def keys() -> list[str]:
    return [f"{v}:{r}x{c}:{kind}" for v in BITCELLS for r, c in SIZES for kind in ("plain", "banked")]


def run_one(key: str) -> dict:
    from gc_compiler.array.floorplan import gc_banked_macro, gc_macro, macro_geometry
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.tech.sky130 import Sky130
    from gc_compiler.verify.parasitics import extract_flat
    from gc_compiler.verify.timing import BenchSettings, build_model, simulate_reads, simulate_writes, standard_corners

    variant, size, kind = key.split(":")
    rows, cols = (int(x) for x in size.split("x"))
    array, bitcell = BITCELLS[variant]
    tech = Sky130()
    cfg = ArrayConfig(rows=rows, cols=cols, replica_column=True, self_timed_read=False, wordline_straps=False, wordline_driver_strength=1.0, bitcell=BitcellConfig.plain(**bitcell), **array)
    name = key.replace(":", "_")
    work = WORK / name
    work.mkdir(parents=True, exist_ok=True)
    banked = kind == "banked"
    macro = gc_banked_macro(tech, cfg, name=name) if banked else gc_macro(tech, cfg, name=name, decoded=True)
    info = macro.info
    out = {
        "key": key, "variant": variant, "rows": rows, "cols": cols, "banked": banked,
        "bits": info["bits"], "width_um": info["width"], "height_um": info["height"],
        "area_per_bit_um2": info["area_per_bit"], "array_efficiency": info["array_efficiency"],
        "banks": [],
    }
    # E5: where the area goes. Array, driver columns and column circuits are the bank's own
    # (times two when banked); the rest of the bounding box is decoder and its routing.
    geo = macro_geometry(tech, cfg)
    n = 2 if banked else 1
    parts = {
        "array_um2": n * geo.array.width * geo.array.height,
        "drivers_um2": n * 2 * geo.driver_width * geo.array.height,
        "columns_um2": n * geo.width * geo.column_height,
    }
    parts["decoder_and_routing_um2"] = info["width"] * info["height"] - sum(parts.values())
    out["area_breakdown"] = parts
    if rows > TIMING_MAX_ROWS:
        return out
    gds = work / f"{name}.gds"
    macro.write_gds(gds)
    flat = extract_flat(gds, name, work / "extract")
    tt = next(c for c in standard_corners() if c.process == "tt")
    far, settings = str(cols - 1), BenchSettings()
    for bank in ((0, 1) if banked else (None,)):
        model = build_model(tech, cfg, gds, name, work / f"b{bank}", bank=bank, flat=flat)
        worst = None
        for row in (0, rows - 1):
            reads = simulate_reads(tech, model, tt, row, {far: (1, tt.vdd), "0": (1, tt.vdd)}, work / f"b{bank}" / f"r{row}", settings)
            for t in reads.values():
                if worst is None or (t.t_dout or 1) > (worst.t_dout or 1):
                    worst = t
        write = simulate_writes(tech, model, tt, rows - 1, {far: 1}, work / f"b{bank}" / "w", settings)[far]
        out["banks"].append({
            "bank": bank, "access_s": worst.t_dout, "decoder_s": worst.t_decoder, "driver_s": worst.t_driver,
            "bitline_s": worst.t_bitline, "rdy_s": worst.t_rdy, "write_s": write.t_write,
            "write_decoder_s": write.t_decoder,
        })
    return out


def main() -> None:
    DATA.mkdir(exist_ok=True)
    path = DATA / "scaling.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for key in keys():
        if key in results:
            continue
        proc = subprocess.run([sys.executable, __file__, key], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"{key}: FAILED\n{proc.stderr[-2000:]}", flush=True)
            continue
        results[key] = json.loads(proc.stdout.strip().splitlines()[-1])
        path.write_text(json.dumps(results, indent=1) + "\n")
        print(f"{key}: done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(run_one(sys.argv[1])))
    else:
        main()
