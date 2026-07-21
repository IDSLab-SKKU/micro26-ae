#!/usr/bin/env bash
# Figure 11 — run the FP8 then NVFP4 decode sweeps in sequence.
#
# Any arguments are passed through to both runs, e.g.  ./run_figure11.sh --gpu 1
#
# It stops at the first failure (set -e), so a broken run does not silently skip
# the rest of the sweep.
set -euo pipefail

cd "$(dirname "$0")/.."         # → artifact/ (where scripts/run_experiment.py lives)

# Check the GPU is SM120 (compute capability 12.0); warn up front if it is not.
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
       | head -1 | tr -d ' .' || true)"
if [[ "$cap" != "120" ]]; then
    echo "WARNING: compute capability is '${cap:-unknown}', not 12.0 (SM120)."
fi

echo "=== Figure 11(a): FP8 CoFDA decode sweep (9 runs) ==="
python3 scripts/run_experiment.py --exp exp3_figure11_decode/fp8_cofda "$@"

echo "=== Figure 11(b): NVFP4 GDFS decode sweep (16 runs) ==="
python3 scripts/run_experiment.py --exp exp3_figure11_decode/nvfp4_gdfs "$@"

echo "=== Figure 11 done ==="
