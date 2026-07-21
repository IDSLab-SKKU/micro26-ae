# exp3 — Figure 11

## Overview

This reproduces **Figure 11** from **§5.4 (*Trends Across Model Families, Scales,
Modalities, and Generation Tasks*)**, which measures **decode-phase** accuracy on
**LLaMA-3.1-8B-Instruct** — **GSM8K** (exact match, chain-of-thought) and
**HumanEval** (pass@1), both under greedy decoding.

It has two panels, both emulated on the CUDA cores:

| Panel | Fixed | Swept | Runs |
| --- | --- | --- | --- |
| **(a) FP8-CoFDA** | CS=32 | fractional bits F ∈ {25, 21, 17, 13, 12, 11, 10, 9, 7} | 9 |
| **(b) NVFP4-GDFS** | GS=16 | group bits G ∈ {6, 5, 4, 3} × fractional bits F ∈ {35, 25, 13, 10} | 16 |

## Run the experiment

Run everything below on the **RTX PRO 6000 (Blackwell)**, from this experiment's
directory, `exp3_figure11_decode/`.

**Approx. run time:** ~10 h (FP8) + ~20 h (NVFP4).

**1. Sweep** — FP8 then NVFP4:

```bash
./run_figure11.sh
```

Runs the FP8 sweep (9 runs) then the NVFP4 sweep (16 runs), each saved under its
own `results/` (`fp8_cofda/results/`, `nvfp4_gdfs/results/`).

**2. Plot:**

```bash
python3 figure11.py
```

Expected — the figure reproduces the **§5.4** decode finding. Under
**FP8-CoFDA (a)**, both tasks track their F=25 baseline at high F but fall off as
F drops, collapsing to near zero at **F=7**. Under
**NVFP4-GDFS (b)**, intra-group precision **G** governs accuracy while inter-group
precision **F** has little effect.
