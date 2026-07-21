# MMA-Emu — Configurable MMA Accumulation Emulation Kernels

Bit-accurate emulation of the internal accumulation arithmetic of commercial
matrix multiply-accumulate (MMA) units, running on CUDA cores so that every
alignment and truncation step is under software control.

Tensor cores are fixed-function: their accumulation precision and their order of
operations cannot be changed from software. These kernels reproduce that
arithmetic on CUDA cores instead, which makes the accumulation algorithm and its
bitwidths configurable parameters. That is what allows an LLM to be run
end-to-end under an arbitrary MMA accumulation configuration.

This directory implements the emulation kernel of the paper
*"Not All Dot Products Are Equal: The Hidden MMA Arithmetic Design Space Drives
Cross-Architecture LLM Inference Gaps"*.

## Accumulation algorithms

These algorithms and the accumulator-coupling modes below are the paper's MMA
arithmetic design space; refer to **§2.2 (*MMA Arithmetic Design Space*)** and
**Figure 1** (block diagrams of the accumulation algorithms).

| Algorithm | Description | Parameters |
| --- | --- | --- |
| **FDA** (Fused-Dot-Add) | Aligns all `K` partial products to the max exponent, truncates to `F` fractional bits (round-to-zero), sums in fixed point | `F` |
| **CoFDA** (Chain-of-FDA) | Chains FDA over chunks of `CS` products. `CS = K` reduces to FDA | `F`, `CS` |
| **GDFS** (Group-Dot-Fused-Sum) | Two-level: intra-group accumulation at `G` bits over groups of `GS`, then inter-group accumulation at `F` bits | `F`, `G`, `GS` |

### Accumulator coupling

How the running accumulator `C` is integrated with each Dot result is a separate
axis of the design space:

| `algorithm` | Mode | Behavior |
| --- | --- | --- |
| `cofda` | **C-fused** | `C` participates in the reduced-precision datapath: aligned to the same max exponent and truncated to `F` fractional bits alongside the `K` products |
| `cofda_decoupled` | **C-decoupled** | `C` is excluded from the reduced-precision datapath: the `K` products are summed at `F` bits, then merged with `C` in a second alignment stage at wider precision |

## Directory Structure

```text
mma_emu/
├── core/                     Format-agnostic accumulation arithmetic
│   ├── types.cuh             Operand / Product / fixed-point types
│   ├── fp32_utils.cuh        IEEE-754 bit manipulation, alignment, round-to-zero
│   ├── accumulator.cuh       FDA · CoFDA · GDFS accumulators
│   ├── gdfs_group.cuh        GroupResult, group_accumulate (GDFS STP3)
│   ├── design_space.cuh      Accepted F / G / CS / GS values; enum Algorithm
│   └── tiling.cuh            Tile shapes per format
│
├── formats/                  Per-format element and scale arithmetic
│   ├── fp8_e4m3.cuh          FP8 E4M3 elements, per-tensor scales
│   ├── fp4_e2m1.cuh          E2M1 elements — shared by NVFP4 and MXFP4
│   ├── nvfp4_ue4m3.cuh       NVFP4 block scales: UE4M3
│   ├── mxfp4_e8m0.cuh        MXFP4 block scales: E8M0
│   └── scale_swizzle.cuh     Swizzled block-scale memory layout
│
├── gemm/                     GEMM kernels
│   ├── scaled_fp8_mm.cuh
│   ├── scaled_nvfp4_mm.cuh
│   └── scaled_mxfp4_mm.cuh
│
└── *_gemm_kernels.cu         Torch operator entry points
```

Dependencies flow strictly downward: `core/` → `formats/` → `gemm/`.

`core/design_space.cuh` is the single source of truth for which `F`, `G`, `CS`
and `GS` values the kernels accept. Every dispatch table and every validation
check derives from it; nothing restates those values.

## Torch operators

| Operator | Arch | Built when |
| --- | --- | --- |
| `mma_emu_scaled_fp8_mm` | SM89+ (Ada, Hopper, Blackwell) | `ENABLE_MMAEMU_FP8` |
| `mma_emu_scaled_nvfp4_mm` | SM100+ (Blackwell), CUDA 12.8+ | `ENABLE_MMAEMU_NVFP4` |
| `mma_emu_scaled_mxfp4_mm` | SM100+, CUDA 12.8+ | `ENABLE_MMAEMU_MXFP4` |

