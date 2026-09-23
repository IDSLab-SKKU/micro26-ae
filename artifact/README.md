# Running the Experiments

## Directory Structure

```text
artifact/
├── run-docker.sh              start the container (prebuilt image)
├── build.sh                   optional: build vLLM from source with uv
├── scripts/                   shared tooling
│   ├── run_experiment.py      experiment runner (reads a config.yaml, drives lm-eval)
│   ├── compare.py             Table 6 cross-architecture comparison
│   └── tasks.py               lm-eval task definitions
├── kernels/                   MMA-Emu CUDA kernels (symlink → ../csrc/quantization/mma_emu)
├── exp1_table6_cross_arch/    Table 6     — cross-architecture correctness
├── exp2_figure6a_fp8_cofda/   Figure 6(a) — FP8 CoFDA accuracy design space
└── exp3_figure11_decode/      Figure 11   — decode-phase accuracy
```

## Installation

All experiments run inside a container built from our Docker image. Please follow
the procedure below to evaluate our artifact.

### Start the container

```bash
git clone https://github.com/IDSLab-SKKU/micro26-ae
cd micro26-ae/artifact
./run-docker.sh
```

`run-docker.sh` pulls the image, mounts `artifact/` into the container.

For reference, the image
([`jongyeop1999/micro26-ae`](https://hub.docker.com/repository/docker/jongyeop1999/micro26-ae/general))
ships the paper's software environment, so nothing needs to be built:

- **vLLM prebuilt**, with the MMA-Emu kernels compiled for **Hopper (SM90)** and
  **Blackwell (SM120)**.
- The native GEMM path runs on the tensor cores via **CUTLASS v4.0.0**.
- Pinned to the environment the paper's numbers came from: torch 2.8.0 / CUDA
  12.8.1, `lm_eval==0.4.9.1`, `transformers==4.55.2`, and matplotlib for the plots.

### Verify the image

Inside the container, confirm the prebuilt vLLM imports cleanly:

```bash
python3 -c "import vllm; print('vLLM', vllm.__version__)"
```

Printing a version (e.g. `vLLM 0.1.dev...`) means the image was built correctly —
the vLLM wheel with the MMA-Emu kernels is installed and loads.

### Optional: build from source with uv

We recommend the prebuilt Docker image above. If you would rather build from
source, run `./build.sh` from `artifact/` — it builds vLLM and the MMA-Emu
kernels with [uv](https://docs.astral.sh/uv/), pinned to the same environment as
the image, and compiled for Hopper (SM90) and Blackwell (SM120) (~30 min).

**Host prerequisites:**

- **CUDA Toolkit 12.8.1** (`nvcc` on `PATH`)
- **C++ compiler** (g++ 10 or newer)
- **Python 3.12**
- **uv** (<https://docs.astral.sh/uv/>)

```bash
./build.sh          # build vLLM + MMA-Emu kernels from source (~30 min)
```

The build creates a `.venv/` at the repo root. Activate it and verify:

```bash
source ../.venv/bin/activate   # .venv is at the repo root
python3 -c "import vllm; print('vLLM', vllm.__version__)"
```

Afterwards the experiments run directly, without Docker.

## Evaluation and Expected Results

Each experiment is run from its own directory (`cd exp<N>_… && ./run_*.sh`); its
README has the full step-by-step.

> **Total run time ≈ 65 h (~3 days)** on a single RTX PRO 6000 Blackwell WS Edition.
> We highly recommend the **Workstation Edition (WS)** for the fastest runs — the
> Max-Q edition runs the same experiments but is power-limited and slower.

### [exp1](exp1_table6_cross_arch/) — cross-architecture correctness

- **Machine:** H100 (Hopper) + RTX PRO 6000 (Blackwell).
- **Run** — same script on each machine, then compare:
  1. On the **H100**: `./run_table6.sh` (native Hopper)
  2. On the **RTX PRO 6000**: copy the H100 results over, then `./run_table6.sh` (emulated Hopper)
  3. `./compare.sh`
- **Expected** (Table 6, §4.3): every task's score is identical and every logprob bit-identical.
- **Time:** ~3 h on the RTX PRO 6000, ~5 min on the H100.

### [exp2](exp2_figure6a_fp8_cofda/) — FP8 CoFDA design space

- **Machine:** RTX PRO 6000 (Blackwell).
- **Run:**
  1. `./run_figure6a.sh` — the 36-run sweep
  2. `python3 figure6a.py` — plot
- **Expected** (Figure 6(a), §5.2): C-fused collapses at low F; C-decoupled holds near baseline down to F=3.
- **Time:** ~30 h.

### [exp3](exp3_figure11_decode/) — decode-phase analysis

- **Machine:** RTX PRO 6000 (Blackwell).
- **Run:**
  1. `./run_figure11.sh` — the FP8 then NVFP4 sweeps
  2. `python3 figure11.py` — plot
- **Expected** (Figure 11, §5.4): FP8-CoFDA collapses at F=7; NVFP4-GDFS accuracy is governed by G, not F.
- **Time:** ~10 h (FP8) + ~20 h (NVFP4).

## Resuming an interrupted sweep

A re-run resumes by default: `run_experiment.py` skips any combination whose
result JSON already exists in `results/` (same sweep parameters, every task
evaluated without error) and runs only the rest. Pass `--overwrite` to re-run
everything and replace existing results; `--dry-run` marks each combination
`[SKIP]` or `[RUN]`:

```bash
./run_figure6a.sh               # continue an interrupted exp2 sweep
./run_figure6a.sh --overwrite   # start it over
```

## GPU device selection

Each `run_*.sh` passes its arguments through to the runner — append `--gpu N` to
pin a run to a single CUDA device (the default is GPU 0):

```bash
./run_figure6a.sh --gpu 1   # example: run exp2 on GPU 1
```
