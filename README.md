# Artifact Evaluation for MICRO'26

## 1. Abstract

This is the artifact of *Not All Dot Products Are Equal: The Hidden MMA Arithmetic
Design Space Drives Cross-Architecture LLM Inference Gaps*, accepted at MICRO'26.

This artifact provides the MMA emulation kernels and the evaluation
framework to reproduce three of the paper's main results:

1. **Cross-architecture correctness:** emulating one architecture's accumulation
   on the other reproduces its native benchmark scores and per-sample
   log-probabilities bit-for-bit (§4.3, Table 6).
2. **Prefill-phase MMA design-space analysis:** sweeping the **FP8-CoFDA**
   accumulation configuration, measured as WikiText-2 perplexity (§5.2, Figure 6(a)).
3. **Decode-phase MMA design-space analysis:** sweeping the **FP8-CoFDA** and
   **NVFP4-GDFS** accumulation configurations on the generation tasks GSM8K and
   HumanEval (§5.4, Figure 11).

## 2. Artifact Checklist (meta-information)

- **Algorithm:** MMA accumulation algorithm
- **Program:** vLLM with the MMA Emulation kernels; lm-eval-harness for evaluation
- **Compilation:** Not required by default — prebuilt in the Docker image; optionally build from source with uv (needs CUDA Toolkit 12.8.1, g++ 10+)
- **Model:** LLaMA-3.1-8B-Instruct (FP8 and NVFP4 checkpoints from HuggingFace)
- **Data set:** WikiText-2, ARC-Challenge, ARC-Easy, PIQA, WinoGrande, GSM8K, HumanEval (via lm-eval-harness)
- **Run-time environment:** Linux with Docker (≥ 19.03) and a recent NVIDIA Container Toolkit; NVIDIA driver ≥ 570.124.06 for the image's CUDA 12.8.1
- **Hardware:** x86_64 CPU; NVIDIA H100 (Hopper) and RTX PRO 6000 (Blackwell)
- **Metrics:** WikiText-2 perplexity, task accuracy (exact-match, pass@1, acc, acc_norm), and per-sample log-probabilities
- **Output:** Cross-architecture correctness validation results (Table 6) and reproduced figures (Figure 6(a) and Figure 11)
- **Experiments:** Each reproduction includes a run script (`run_*.sh`) and a step-by-step README
- **Disk space (approx.):** ~30 GB
- **Setup time (approx.):** ~10 min
- **Experiment time (approx.):** ~65 h (~3 days) on a single RTX PRO 6000, plus a ~5 min run on an H100
- **Publicly available?:** Yes
- **Code license (if publicly available)?:** Apache 2.0
- **Archived (provide DOI)?:** TBA

## 3. Description

### 3.1 How to access

- **GitHub:** <https://github.com/IDSLab-SKKU/micro26-ae>

### 3.2 Hardware dependencies

Full reproduction needs an **x86_64 CPU** and **two NVIDIA GPU architectures**,
because Table 6 is a cross-architecture swap test:

- **NVIDIA H100** (Hopper, SM90)
- **NVIDIA RTX PRO 6000** (Blackwell, SM120)
    - **Minimum:** Max-Q Edition
    - **Preferred:** Workstation (WS) Edition

### 3.3 Software dependencies

All software is packaged in the prebuilt Docker image
([`jongyeop1999/micro26-ae`](https://hub.docker.com/repository/docker/jongyeop1999/micro26-ae));
the host requires only **Docker** and the **NVIDIA Container Toolkit**. The image
includes:

- the forked **vLLM** with the **MMA-Emu** kernels, prebuilt for SM90 / SM120
- **CUDA 12.8.1**, **PyTorch 2.8.0**
- **lm-eval-harness 0.4.9.1**, **transformers 4.55.2**, **matplotlib**

### 3.4 Models and data sets

Both the model and the datasets are pulled from **Hugging Face** at run time.

- **Model:** LLaMA-3.1-8B-Instruct, quantized to FP8
  (`nvidia/Llama-3.1-8B-Instruct-FP8`) and NVFP4
  (`nvidia/Llama-3.1-8B-Instruct-NVFP4`).
- **Data sets:** WikiText-2, ARC-Challenge, ARC-Easy, PIQA, WinoGrande, GSM8K, and
  HumanEval — downloaded automatically by lm-eval-harness.

## 4. Installation

```bash
git clone https://github.com/IDSLab-SKKU/micro26-ae
cd micro26-ae/artifact
./run-docker.sh
```

This pulls the Docker image and drops you into the container. See
[`artifact/README.md`](artifact/README.md) for the full setup — the image, GPU
requirements, and a quick verification step.

The Docker image is recommended, but you can also build from source: run
`./build.sh` from `artifact/` — see [Optional: build from source with uv](artifact/README.md#optional-build-from-source-with-uv).

## 5. Experiments and Expected Results

Each experiment is run from its own directory under [`artifact/`](artifact/):
`cd` into it and run its `run_*.sh`. Each directory's README has the exact steps,
the machine it needs, the expected result, and the run time.

- [**exp1 — Table 6**](artifact/exp1_table6_cross_arch/): cross-architecture correctness (bit-exact scores and log-probabilities)
- [**exp2 — Figure 6(a)**](artifact/exp2_figure6a_fp8_cofda/): prefill-phase (FP8-CoFDA) analysis
- [**exp3 — Figure 11**](artifact/exp3_figure11_decode/): decode-phase (FP8-CoFDA, NVFP4-GDFS) analysis

> **Total run time ≈ 65 h (~3 days)** on a single RTX PRO 6000 Blackwell, plus a
> ~5 min native run on an H100 for exp1 (Table 6). The Workstation Edition is recommended
> for the fastest runs.

## 6. Built on vLLM

This repository is a fork of [vLLM](https://github.com/vllm-project/vllm). Our
additions are the **MMA-Emu** emulation kernels (`csrc/quantization/mma_emu`),
their integration into vLLM's GEMM path — which routes an LLM's linear layers
through an arbitrary MMA accumulation configuration — and the `artifact/`
evaluation harness. Everything else is stock vLLM.

## 7. License

Apache License 2.0 (see [LICENSE](LICENSE)), inherited from vLLM. The MMA-Emu
kernels and the artifact added in this fork are released under the same license.
