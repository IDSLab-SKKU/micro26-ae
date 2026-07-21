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

**2. On the RTX PRO 6000 (Blackwell).** Copy the H100's `h100/native/results/`
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

## Optional — the emulation kernels

The kernels we implemented live in
[`csrc/quantization/mma_emu/`](../../csrc/quantization/mma_emu), symlinked here as
`../kernels` for convenience.

This experiment's emulation uses the **FP8-CoFDA** kernel. The same directory
also carries the **GDFS** algorithm (which supports FP8 too) and the **MXFP4**
and **NVFP4** kernels, which you can explore there.

See that folder's [README](../../csrc/quantization/mma_emu/README.md) for the details.
