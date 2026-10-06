"""How accurate and how fast the reduced timing model is, against keeping every column.

    python sweep_reduction.py   -> data/reduction.json

The same reads (fresh '1' at the far and near end of the first and last rows) and write, at tt,
on the plain 16x20 macro with replica, once on the reduced model and once on a model that keeps
every column as real cells.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "reduction"


def main() -> None:
    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.tech.sky130 import Sky130
    from gc_compiler.verify.parasitics import extract_flat
    from gc_compiler.verify.timing import BenchSettings, build_model, simulate_reads, simulate_writes, standard_corners

    tech = Sky130()
    cfg = ArrayConfig(
        rows=16, cols=20, replica_column=True, bitcell=BitcellConfig.plain(), self_timed_read=False, wordline_straps=False, wordline_driver_strength=1.0
    )
    WORK.mkdir(parents=True, exist_ok=True)
    gds = WORK / "red16.gds"
    gc_macro(tech, cfg, name="red16", decoded=True).write_gds(gds)
    flat = extract_flat(gds, "red16", WORK / "extract")
    tt = next(c for c in standard_corners() if c.process == "tt")
    out = {}
    for label, keep in (("reduced", None), ("full", tuple(range(cfg.cols)))):
        model = build_model(tech, cfg, gds, "red16", WORK / label, keep=keep, flat=flat)
        t0 = time.time()
        reads = {}
        for row in (0, cfg.rows - 1):
            r = simulate_reads(tech, model, tt, row, {"0": (1, tt.vdd), "19": (1, tt.vdd)}, WORK / label / f"r{row}", BenchSettings())
            reads.update({f"r{row}_c{col}": {"dout_s": t.t_dout, "rdy_s": t.t_rdy} for col, t in r.items()})
        read_time = (time.time() - t0) / 2
        w = simulate_writes(tech, model, tt, cfg.rows - 1, {"19": 1}, WORK / label / "w", BenchSettings())["19"]
        out[label] = {"reads": reads, "write_s": w.t_write, "seconds_per_read": read_time}
        print(label, out[label], flush=True)
    DATA.mkdir(exist_ok=True)
    (DATA / "reduction.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
