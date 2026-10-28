"""The write device's leakage into a stored '0', and what it does to a macro.

    python compare_write_device.py leak     # off-state current of each write device, every corner
    python compare_write_device.py macro    # the extracted macro of each cell, idle and hammered
                                            # -> data/write_device.json

An off write device (gate at vdd) still leaks from its write bit line into the storage node. While
the write bit line is high -- every read of a '1' in the column with the in-macro refresh's
``NAND(din, rbl)``, every write of a '1' -- a stored '0' leaks up, and its write device's
gate-source voltage stays at zero as it does, so it keeps leaking. ``leak`` is that current (the
storage node held at 0.3 V, the write bit line and the gate at vdd).

``macro`` builds one bank (16x8, decoded) of each cell, extracts it flat (with capacitances) and
drives its pins the way the controller does (``din`` high outside a write): every row written '1'
(the replica cells too), two rows overwritten with a pattern and its complement, then either 2 us
idle or 300 short reads of an all-'1' row (the "hammer", about 2 us of read traffic), then both rows
read back (self-timed: the read ends on ``rdy``), one refreshed (a read, then the write-back with
``pch`` held up) and read again. A read is right when every ``dout`` is.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
WORK = HERE.parent / "build" / "write_device"

ROWS, COLS, PATTERN = 16, 8, 0b10100101
WRITE_NS, READ_NS, HAMMER_READ_NS, PRECHARGE_NS, GAP_NS, SLEW = 3.0, 12.0, 3.0, 2.0, 1.0, 0.1
IDLE_NS, HAMMER_READS = 2000.0, 300

CELLS = {
    # name: BitcellConfig.for_tech overrides
    "lvt_mr0.5": {"write_flavor": "lvt", "write_length": 0.35, "mr_length": 0.5},  # the default before
    "std_mr0.5": {"write_flavor": None, "write_length": 0.18, "mr_length": 0.5},
    "std_mr1.0": {"write_flavor": None, "write_length": 0.18, "mr_length": 1.0},  # the default
}


def leak() -> dict:
    """Off-state write-device current into a '0' (A), per corner, LVT and standard-Vt."""
    from gc_compiler.tech import technology
    from gc_compiler.tech.rules import DeviceKind

    tech = technology("sky130A")
    out: dict[str, dict[str, float]] = {}
    work = WORK / "leak"
    work.mkdir(parents=True, exist_ok=True)
    for corner in tech.electrical.corners:
        row = {}
        for flavor, length in (("lvt", 0.35), (None, 0.18)):
            vdd = corner.vdd
            deck = [
                "* write-device leakage", *tech.model_header(corner.process),
                f"vw wbl 0 {vdd}", f"vg wwl 0 {vdd}", f"vb vnw 0 {vdd}", "vs sn 0 0.3",
                tech.spice_device("w", "wbl", "wwl", "sn", "vnw", DeviceKind.PMOS, 0.42, length, flavor),
                f".temp {corner.temp:g}", ".op", ".control", "run", "print i(vs)", "quit", ".endc", ".end", "",
            ]
            path = work / f"{corner.process}_{flavor or 'std'}.spice"
            path.write_text("\n".join(deck))
            text = subprocess.run(["ngspice", "-b", path.name], cwd=work, capture_output=True, text=True).stdout
            row[flavor or "standard"] = float(re.search(r"i\(vs\)\s*=\s*(\S+)", text).group(1))
        out[corner.process] = row
        print(corner.process, row, flush=True)
    return out


def _schedule(scenario: str) -> list[tuple[str, int, int]]:
    full = (1 << COLS) - 1
    ops = [("write", r, full) for r in range(ROWS)]
    ops += [("write", 3, PATTERN), ("write", 10, ~PATTERN & full)]
    ops += [("idle", 0, 0)] if scenario == "idle" else [("hammer", 0, full)] * HAMMER_READS
    ops += [("read", 3, PATTERN), ("read", 10, ~PATTERN & full)]
    ops += [("refresh", 3, PATTERN), ("read", 3, PATTERN)]
    return ops


def _bench(tech, corner, netlist: Path, wells: list[str], scenario: str, out: Path) -> dict:
    from gc_compiler.netlist.decoder import address_bits

    vdd = corner.vdd
    bits = address_bits(ROWS)
    waves = {p: [(0.0, 0.0)] for p in ["we", "re", "pch"] + [f"addr_{i}" for i in range(bits)] + [f"din_{c}" for c in range(COLS)]}

    def drive(pin, t, v):
        last = waves[pin][-1][1]
        if last != v:
            waves[pin] += [(t, last), (t + SLEW, v)]

    for c in range(COLS):  # the controller holds din high outside a write
        drive(f"din_{c}", 1.0, vdd)
    t, reads = 2.0, []
    for kind, row, data in _schedule(scenario):
        for i in range(bits):
            drive(f"addr_{i}", t, vdd * ((row >> i) & 1))
        if kind == "idle":
            t += IDLE_NS
        elif kind == "write":  # pch low (precharging); din = ~data while we is up
            for c in range(COLS):
                drive(f"din_{c}", t, vdd * (1 - ((data >> c) & 1)))
            drive("we", t + 0.5, vdd)
            drive("we", t + 0.5 + WRITE_NS, 0.0)
            t += 0.5 + WRITE_NS + GAP_NS
            for c in range(COLS):
                drive(f"din_{c}", t, vdd)
            t += GAP_NS
        else:  # a read: pch up, then re; self-timed, so the read ends by itself
            refresh = kind == "refresh"
            if refresh:
                t += 1.0  # (din is already high: the write bit lines settle while the precharge holds)
            drive("pch", t, vdd)
            length = HAMMER_READ_NS if kind == "hammer" else READ_NS
            drive("re", t + 0.5, vdd)
            drive("re", t + 0.5 + length, 0.0)
            if kind != "hammer":
                reads.append((kind, row, data, t + 0.5, t + 0.5 + length))
            t += 1.0 + length
            if refresh:  # the write-back, pch still up
                drive("we", t, vdd)
                drive("we", t + WRITE_NS, 0.0)
                t += WRITE_NS + 0.5
            drive("pch", t, 0.0)
            t += PRECHARGE_NS
    stop = t + 1.0
    deck = [f"* write-device macro bench {scenario} {corner.process}", *tech.model_header(corner.process),
            f".include {netlist}", f"vvdd vdd 0 {vdd}", "vvss vss 0 0"]
    for pin, points in waves.items():
        points = points + [(stop, points[-1][1])]
        deck.append(f"v{pin} {pin} 0 pwl(" + " ".join(f"{a:.4g}n {b:g}" for a, b in points) + ")")
    deck += [f"vwell{i} {net} 0 {vdd}" for i, net in enumerate(wells)]  # the n-wells (no macro pin)
    probes = ["rdy"] + [f"dout_{c}" for c in range(COLS)]
    deck += [f".temp {corner.temp:g}", ".options method=gear", f".tran 20p {stop:g}n", ".control", "run",
             "wrdata sim.csv " + " ".join(f"v({p})" for p in probes), "quit", ".endc", ".end", ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.spice").write_text("\n".join(deck))
    subprocess.run(["ngspice", "-b", "bench.spice"], cwd=out, capture_output=True, timeout=7200)
    raw = np.loadtxt(out / "sim.csv")
    time = raw[:, 0] * 1e9
    signal = {p: raw[:, 1 + 2 * i] for i, p in enumerate(probes)}
    result = []
    for kind, row, data, t_re, t_end in reads:
        window = (time > t_re) & (time < t_end)
        fired = np.nonzero(window & (signal["rdy"] > vdd / 2))[0]
        got = 0
        for c in range(COLS):
            high = np.interp(t_end - 0.1, time, signal[f"dout_{c}"]) > vdd / 2
            got |= int(high) << c
        result.append({"kind": kind, "row": row, "want": data, "got": got, "ok": got == data,
                       "rdy_ns": round(float(time[fired[0]] - t_re), 3) if len(fired) else None})
    return {"corner": corner.process, "scenario": scenario, "ok": all(r["ok"] for r in result), "reads": result}


def macro() -> dict:
    from gc_compiler.compiler import GainCellCompiler
    from gc_compiler.config import ArrayConfig, BitcellConfig
    from gc_compiler.tech import technology
    from gc_compiler.verify.parasitics import extract_flat

    tech = technology("sky130A")
    jobs = []
    for name, fields in CELLS.items():
        bitcell = BitcellConfig.for_tech(tech, **fields)
        cfg = ArrayConfig.for_tech(tech, rows=ROWS, cols=COLS, bitcell=bitcell)
        work = WORK / name
        built = GainCellCompiler(cfg, tech).build(work, name=f"wd_{name.replace('.', 'p')}", macro=True, decoded=True, banked=False)
        netlist = extract_flat(built.gds, built.component.name, work / "ext", tech=tech)
        wells = sorted(set(re.findall(r"(?:^|\s)(\S*vnw\w*)(?=\s)", netlist.read_text(), re.M)))
        jobs += [(name, corner, netlist, wells, scenario)
                 for scenario in ("idle", "hammer") for corner in tech.electrical.corners]
    results: dict[str, list] = {name: [] for name in CELLS}
    with ThreadPoolExecutor(8) as pool:
        futures = [(name, pool.submit(_bench, tech, corner, netlist, wells, scenario,
                                      WORK / name / f"{scenario}_{corner.process}"))
                   for name, corner, netlist, wells, scenario in jobs]
        for name, future in futures:
            results[name].append(future.result())
            r = results[name][-1]
            print(name, r["scenario"], r["corner"], "ok" if r["ok"] else "FAIL", flush=True)
    return results


def main(mode: str) -> None:
    path = DATA / "write_device.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[mode] = leak() if mode == "leak" else macro()
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "leak")
