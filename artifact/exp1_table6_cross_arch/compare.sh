#!/usr/bin/env bash
# Table 6 — compare the two machines' results: benchmark scores and per-sample
# logprobs, checked for bit-exact agreement. No GPU needed; run it once both
# machines' result directories are present in this clone.
#
# Any arguments pass through, e.g.  ./compare_table6.sh --out cmp.md
set -euo pipefail

cd "$(dirname "$0")/.."         # → artifact/ (where scripts/compare.py lives)

python3 scripts/compare.py "$@"
