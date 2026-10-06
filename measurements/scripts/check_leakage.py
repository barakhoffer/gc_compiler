"""What sets the refresh interval's drop at 125 C? The same retention point with the PDK's 1.8 V
FET models as shipped, with their GIDL coefficients zeroed, and with their junction saturation
currents zeroed.

    python check_leakage.py                       # every variant and option -> data/leakage.json
    python check_leakage.py <option> <variant>    # one run (what the loop calls)

Each modified model set lives in a private PDK root (build/leakage/<variant>/pdk): symlinks to
the real PDK, except the 1.8 V FET model files, copied with the named parameters set to zero.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "leakage"
REAL_ROOT = Path(os.environ.get("PDK_ROOT", Path.home() / ".ciel"))
TEMPS = (85.0, 125.0)
OPTIONS = ("plain", "lvt_mr")
VARIANTS = {
    "real": (),
    "nogidl": ("agidl", "agisl"),
    "nojunction": ("jss", "jsd", "jsws", "jswd", "jswgs", "jswgd"),
}


def private_root(variant: str) -> Path:
    """A PDK root whose 1.8 V FET models have the variant's parameters zeroed."""
    root = WORK / variant / "pdk"
    marker = root / ".done"
    if marker.exists():
        return root
    real = (REAL_ROOT / "sky130A").resolve()
    fake = root / "sky130A"
    for sub in ("libs.tech", "libs.ref"):
        for path in (real / sub).iterdir():
            target = fake / sub / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(path)
    # the corner files include the device models by a relative path into libs.ref, so replace
    # the fd_pr spice directory by one of per-file symlinks, with the 1.8 V FETs rewritten
    spice = fake / "libs.ref" / "sky130_fd_pr"
    spice.unlink()
    (spice / "spice").mkdir(parents=True)
    for path in (real / "libs.ref" / "sky130_fd_pr").iterdir():
        if path.name != "spice":
            (spice / path.name).symlink_to(path)
    names = "|".join(VARIANTS[variant])
    zero = re.compile(rf"^(\+\s*(?:{names})\s*=\s*).*$", re.IGNORECASE | re.MULTILINE)
    for path in (real / "libs.ref" / "sky130_fd_pr" / "spice").iterdir():
        target = spice / "spice" / path.name
        if re.search(r"[np]fet_01v8", path.name) and path.suffix == ".spice":
            target.write_text(zero.sub(lambda m: m.group(1) + "0", path.read_text()))
        else:
            target.symlink_to(path)
    # ngspice's corner files reach libs.ref through ../../../, so libs.tech/ngspice must be a
    # real directory inside the private root, not a symlink back into the real PDK
    ngspice = fake / "libs.tech" / "ngspice"
    ngspice.unlink()
    subprocess.run(["cp", "-r", str(real / "libs.tech" / "ngspice"), str(ngspice)], check=True)
    marker.write_text("")
    return root


def run_one(option: str, variant: str) -> list[dict]:
    if VARIANTS[variant]:
        os.environ["PDK_ROOT"] = str(private_root(variant))
    sys.path.insert(0, str(HERE))
    from sweep_retention import OPTIONS as ALL, _point

    from gc_compiler.array.floorplan import gc_macro
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.tech.sky130 import Sky130
    from gc_compiler.verify.timing import BenchSettings, Corner, build_model

    array, bitcell = ALL[option]
    tech = Sky130()
    cfg = ArrayConfig(rows=16, cols=20, replica_column=True, self_timed_read=False, wordline_straps=False, wordline_driver_strength=1.0, bitcell=BitcellConfig.plain(**bitcell), **array)
    work = WORK / option
    work.mkdir(parents=True, exist_ok=True)
    cell = f"leak_{option}_16x20"
    gds = work / f"{cell}.gds"
    if not gds.exists():
        gc_macro(tech, cfg, name=cell, decoded=True).write_gds(gds)
    model = build_model(tech, cfg, gds, cell, work / f"model_{variant}")
    return [
        {"models": variant, **_point(tech, model, Corner("tt", 1.8, t), work / variant / f"t{t:g}", BenchSettings())}
        for t in TEMPS
    ]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    path = DATA / "leakage.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for option in OPTIONS:
        done = {p["models"] for p in results.get(option, [])}
        for variant in VARIANTS:
            if variant in done:
                continue
            proc = subprocess.run([sys.executable, __file__, option, variant], capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"{option} {variant}: FAILED\n{proc.stderr[-3000:]}", flush=True)
                continue
            results.setdefault(option, []).extend(json.loads(proc.stdout.strip().splitlines()[-1]))
            path.write_text(json.dumps(results, indent=1) + "\n")
            print(f"{option} {variant}: done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(json.dumps(run_one(sys.argv[1], sys.argv[2])))
    else:
        main()
