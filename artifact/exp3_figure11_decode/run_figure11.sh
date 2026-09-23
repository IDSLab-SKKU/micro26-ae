#!/usr/bin/env bash
# Figure 11 — run the FP8 and NVFP4 decode sweeps.
#
# By default both sweeps run in sequence (FP8 then NVFP4). To run just one, pass
# --fp8 or --nvfp4.
#
# Re-running resumes: combinations that already have a complete result in their
# results/ directory are skipped, so an interrupted sweep picks up where it
# stopped. Pass --overwrite to re-run them all.
#
#   ./run_figure11.sh              # both sweeps (default)
#   ./run_figure11.sh --fp8        # only the FP8 CoFDA sweep
#   ./run_figure11.sh --nvfp4      # only the NVFP4 GDFS sweep
#
# Any other arguments pass through to the runs, e.g.  ./run_figure11.sh --nvfp4 --gpu 1
# or  ./run_figure11.sh --overwrite
#
# It stops at the first failure (set -e), so a broken run does not silently skip
# the rest of the sweep.
set -euo pipefail

cd "$(dirname "$0")/.."         # → artifact/ (where scripts/run_experiment.py lives)

# Pick which sweeps to run; strip our flags out and pass the rest through.
run_fp8=false
run_nvfp4=false
passthrough=()
for arg in "$@"; do
    case "$arg" in
        --fp8)   run_fp8=true ;;
        --nvfp4) run_nvfp4=true ;;
        *)       passthrough+=("$arg") ;;
    esac
done
# No selector given → run both.
if [[ "$run_fp8" == false && "$run_nvfp4" == false ]]; then
    run_fp8=true
    run_nvfp4=true
fi

# Check the GPU is SM120 (compute capability 12.0); warn up front if it is not.
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
       | head -1 | tr -d ' .' || true)"
if [[ "$cap" != "120" ]]; then
    echo "WARNING: compute capability is '${cap:-unknown}', not 12.0 (SM120)."
fi

if [[ "$run_fp8" == true ]]; then
    echo "=== Figure 11(a): FP8 CoFDA decode sweep (9 runs) ==="
    python3 scripts/run_experiment.py --exp exp3_figure11_decode/fp8_cofda "${passthrough[@]}"
fi

if [[ "$run_nvfp4" == true ]]; then
    echo "=== Figure 11(b): NVFP4 GDFS decode sweep (16 runs) ==="
    python3 scripts/run_experiment.py --exp exp3_figure11_decode/nvfp4_gdfs "${passthrough[@]}"
fi

echo "=== Figure 11 done ==="
