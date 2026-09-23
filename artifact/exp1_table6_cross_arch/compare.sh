#!/usr/bin/env bash
# Table 6 — compare the two machines' results: benchmark scores and per-sample
# logprobs, checked for bit-exact agreement. No GPU needed; run it once both
# machines' result directories are present in this clone.
#
# Exit status: 0 match, 1 mismatch, 2 cannot compare (a result or samples file
# is missing, or a side ran on the wrong architecture).
#
# Any arguments pass through, e.g.  ./compare.sh --out cmp.md
set -euo pipefail

cd "$(dirname "$0")/.."         # → artifact/ (where scripts/compare.py lives)

python3 scripts/compare.py "$@"
