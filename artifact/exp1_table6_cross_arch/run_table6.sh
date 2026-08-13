#!/usr/bin/env bash
# Table 6 — run the side of the cross-architecture check that matches THIS GPU.
#
# The experiment is a swap test across two machines: run this once on a
# Blackwell GPU and once on an H100 (Hopper). The script detects the GPU and
# runs the matching config:
#   Blackwell (SM120/SM100) → rtx_pro6000/emulate_hopper (Hopper's F=13, emulated)
#   Hopper    (SM90)        → h100/native                (native tensor cores)
# The emulation kernels are plain scalar arithmetic with no arch-specific
# instructions, so either Blackwell — the RTX PRO 6000 (SM120) we report in the
# paper, or a data-center B200 (SM100) — runs the same emulated_hopper config.
# Afterwards, bring both result directories onto one machine and compare:
#   python3 scripts/compare.py
#
# Any arguments pass through, e.g.  ./run_table6.sh --gpu 1.  Stops on failure.
set -euo pipefail

cd "$(dirname "$0")/.."         # → artifact/ (where scripts/run_experiment.py lives)

cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
       | head -1 | tr -d ' .' || true)"
case "$cap" in
    120|100) exp="exp1_table6_cross_arch/rtx_pro6000/emulate_hopper"
         [ "$cap" = 120 ] && gpu="SM120, e.g. RTX PRO 6000" || gpu="SM100, e.g. B200"
         echo "=== Table 6: Blackwell ($gpu) — Hopper's F=13, emulated (→ should match H100) ===" ;;
    90)  exp="exp1_table6_cross_arch/h100/native"
         echo "=== Table 6: H100 — native Hopper tensor cores ===" ;;
    *)   echo "ERROR: compute capability '${cap:-unknown}' is neither Blackwell (12.0 or 10.0) nor Hopper (9.0)." >&2
         echo "ERROR: Table 6 is a swap test that needs one Blackwell and one Hopper GPU." >&2
         exit 1 ;;
esac

python3 scripts/run_experiment.py --exp "$exp" "$@"

echo "=== done — copy both machines' results together, then: python3 scripts/compare.py ==="
