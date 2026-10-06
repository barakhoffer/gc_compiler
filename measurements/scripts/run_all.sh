#!/usr/bin/env bash
# Regenerate every measurement data file (measurements/data), in sequence (they share the machine's
# cores). Run from the repository root inside the nix shell:  bash measurements/scripts/run_all.sh
set -u
PY=.venv/bin/python
cd "$(dirname "$0")/../.."
for step in \
    "sweep_characterise.py" \
    "sweep_retention.py corners" \
    "sweep_retention.py temp" \
    "sweep_retention.py mc 20" \
    "sweep_retention.py mc_ss 20" \
    "sweep_vneg.py" \
    "sweep_reduction.py" \
    "check_leakage.py" \
    "sweep_energy.py" \
    "sweep_scaling.py"; do
    echo "=== $(date +%H:%M:%S) $step"
    $PY measurements/scripts/$step || echo "!!! $step failed"
done
echo "=== $(date +%H:%M:%S) all done"
