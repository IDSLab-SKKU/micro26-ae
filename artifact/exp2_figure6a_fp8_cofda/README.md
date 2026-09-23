# exp2 — Figure 6(a)

## Overview

This reproduces **Figure 6(a)** from **§5.2 (*Effect of Accumulation Bitwidth and
CS*)**, which sweeps the FP8 CoFDA accumulation configuration on
**LLaMA-3.1-8B-Instruct**, measuring **WikiText-2 perplexity** at each
configuration; the paper's **Table 8** lists the full sweep ranges.

It compares two accumulation modes — **C-fused** (`cofda`) and **C-decoupled**
(`cofda_decoupled`) — at chunk sizes 16 and 32: 2 modes × 2 chunk sizes × 9
values of **F** = **36 runs**, all emulated on the CUDA cores.

## Run the experiment

Run everything below on the **RTX PRO 6000 or B200 (Blackwell)**, from this experiment's
directory `exp2_figure6a_fp8_cofda/`.

**Approx. run time:** ~30 h (36 emulated runs).

**1. Sweep** the 36 configurations:

```bash
./run_figure6a.sh
```

Each combination is saved under `results/<combo>.json`.

If the sweep is interrupted, run `./run_figure6a.sh` again: it resumes, skipping
combinations that already have a complete result. `--dry-run` shows what would
be skipped, and `--overwrite` re-runs all 36.

**2. Plot:**

```bash
python3 figure6a.py
```

Expected — the figure reproduces the **§5.2** finding. Under **C-fused**, FP8
stays iso-accuracy (≤1% above the dashed baseline) only for **F≥13**, then slips
through the transition band into catastrophic collapse at low F — off the chart
by F=7. Under **C-decoupled**, iso-accuracy holds down to **F=7**, and even F=3
stays in the transition region rather than collapsing. The `cofda` / F=25 / CS=32
point is FP8's native baseline.
