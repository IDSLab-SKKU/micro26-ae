#!/usr/bin/env bash
# Figure 6(a) — run the FP8 CoFDA design-space sweep (36 runs).
#
# 2 algorithms x 2 chunk sizes x 9 F values, all emulated on the CUDA cores, so
# the numbers do not depend on the host GPU's native tensor cores. Any arguments
# pass through, e.g.  ./run.sh --gpu 1
#
# It stops at the first failure (set -e). Plot the result afterwards with:
#   python3 figure6a.py
set -euo pipefail

cd "$(dirname "$0")/.."         # → artifact/ (where scripts/run_experiment.py lives)

echo "=== Figure 6(a): FP8 CoFDA design-space sweep (36 runs) ==="
python3 scripts/run_experiment.py --exp exp2_figure6a_fp8_cofda "$@"

echo "=== Figure 6(a) done — plot with: python3 figure6a.py ==="
