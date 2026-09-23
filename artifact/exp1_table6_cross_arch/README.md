# exp1 — Table 6

## Overview

This reproduces **Table 6**-style cross-architecture emulation experiments from
**§4.3 (*Cross-Architecture Correctness Validation*)**, which isolates the MMA
accumulation configuration as the sole source of the Hopper↔Blackwell output
divergence — ruling out other microarchitectural differences.

This artifact evaluation covers the **Hopper-native direction** of Table 6 —
emulating Hopper's config on Blackwell to match a native H100.

It spans the **multiple-choice benchmarks** — ARC-Challenge, ARC-Easy, PIQA,
WinoGrande — and checks each at two levels:

- **Benchmark scores** — the accuracy metrics agree when the emulated config
  matches the architecture.
- **Per-sample log-probabilities** — the emulated arithmetic reproduces the
  hardware *bit-for-bit*, sample by sample.

## Run the experiment

Run everything below from this experiment's directory, `exp1_table6_cross_arch/`.

`./run_table6.sh` detects the GPU and runs the config matching it, so the same
command runs on each machine. Do the Hopper side first, then Blackwell, then
compare.

**Approx. run time:** ~5 min on the H100, ~3 h on the RTX PRO 6000.

**1. On the H100 (Hopper native).**

```bash
./run_table6.sh
```

The native-Hopper results are saved under `h100/native/results/`.

**2. On the RTX PRO 6000 (Blackwell).** A B200 (SM100) works here too:
`run_table6.sh` detects either Blackwell and runs the same emulated-Hopper
config. Copy the H100's `h100/native/results/`
into this clone first — each result JSON records the GPU it ran on, so the two
machines stay distinguishable — then run:

```bash
./run_table6.sh
```

The emulated-Hopper results are saved under `rtx_pro6000/emulate_hopper/results/`.

**3. Compare** (on the Blackwell machine, where both results now live — no GPU
needed):

```bash
./compare.sh
```

Expected — the emulated Hopper config (on Blackwell) and the native Hopper
hardware (H100) agree on every task: every score is **identical** and every
logprob **bit-identical**.

`compare.sh` ends with a one-line `Result:` and sets its exit status accordingly,
so it can be checked from a script:

| Exit | Result | Meaning |
| --- | --- | --- |
| 0 | `MATCH` | every score identical, every logprob bit-identical |
| 1 | `MISMATCH` | some score or logprob differs |
| 2 | `CANNOT COMPARE` | a result or `_samples.json` is missing, or a side ran on the wrong architecture |

## Optional — more tasks

This experiment intentionally evaluates the four multiple-choice benchmarks
above. The runner can also check two more tasks, for up to **six**:

| Task key | Benchmark | Metric |
| --- | --- | --- |
| `wikitext` | WikiText-2 | word perplexity |
| `lambada_openai` | LAMBADA | accuracy |

To add them, append the keys to the `tasks:` list in **both** configs,
`h100/native/config.yaml` and `rtx_pro6000/emulate_hopper/config.yaml`.
`compare.sh` only compares tasks that ran on both sides:

```yaml
eval:
  tasks:
    - arc_challenge
    - arc_easy
    - piqa
    - winogrande
    - wikitext         # optional
    - lambada_openai   # optional
```

Then run steps 1–3 as above. A result that lacks a configured task is not
reused, so `./run_table6.sh` re-runs the whole task list on each machine,
which adds run time. `compare.sh` checks the new tasks the same way, including
per-sample logprobs.

`scripts/tasks.py` also defines GSM8K-CoT and HumanEval. They are generation
tasks used in exp3 (Figure 11) and are not part of this experiment.

## Optional — the emulation kernels

The kernels we implemented live in
[`csrc/quantization/mma_emu/`](../../csrc/quantization/mma_emu), symlinked here as
`../kernels` for convenience.

This experiment's emulation uses the **FP8-CoFDA** kernel. The same directory
also carries the **GDFS** algorithm (which supports FP8 too) and the **MXFP4**
and **NVFP4** kernels, which you can explore there.

See that folder's [README](../../csrc/quantization/mma_emu/README.md) for the details.
