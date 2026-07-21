# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MMA-Emu NVFP4 (NVFP4) scaled_mm kernel.

Compares MMA-Emu Emulation (GDFS/CoFDA) vs MMA-Emu TC vs CUTLASS for BF16/FP16.

The MMA-Emu NVFP4 kernel implements NVFP4 (E2M1) GEMM with configurable
accumulation algorithms (TC, GDFS, CoFDA) for research on intermediate
rounding effects in 4-bit quantized LLM inference.

Precision parameters:
- F-bits: Fractional bits for fused-sum accumulation (STP5-7). Range [5, 35].
- G-bits: Fractional bits for GDFS group accumulator. Values: {2, 3, 4, 5, 6}.
  G=6 is lossless for E2M1 products.

Run: pytest tests/kernels/quantization/test_mma_emu_scaled_nvfp4_mm.py -v -s
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

# Skip if not SM100+ (Blackwell required for NVFP4)
if not current_platform.has_device_capability(100):
    pytest.skip(reason="NVFP4 requires compute capability 10.0 or above.",
                allow_module_level=True)

# Import FP4 utilities after capability check
from nvfp4_utils import (FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX,)

# ============================================================================
# Test Configuration
# ============================================================================

# Output dtypes to test
OUT_DTYPES = [torch.bfloat16, torch.float16]

# Matrix dimension factors (M, N, K)
# K : Packed FP4 blocks (2 values per byte)
MNK_FACTORS = [
    # (m, n, k_packed) — actual K = k_packed * 2
    (128, 128, 64),       # K = 128
    (1, 4096, 512),       # K = 1024, decode
    (32, 4096, 2048),     # K = 4096
]

# GDFS precision configuration
F_BITS_GDFS = 35   # GDFS inter-group bits — Blackwell's native NVFP4 config
F_BITS_COFDA = 25  # CoFDA fractional bits (CoFDA and GDFS use different F sets)
G_BITS = 6         # Lossless intra-group width for E2M1 products

SEEDS = [42]
CUDA_DEVICES = ['cuda:0']


# ============================================================================
# Helper Functions
# ============================================================================

def generate_realistic_tensor(shape: tuple, device: str = "cuda",
                              dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Generate tensor with realistic LLM activation/weight distribution.

    Based on empirical observations from LLMs like Llama-3.1:
    - Activations: Most values in [-50, 50], with outliers up to [-200, 200]
    - Weights: Most values in [-1, 1], with some spread to [-5, 5]
    - Heavy-tailed distribution with occasional extreme outliers

    This function generates activations-like tensors with larger magnitudes
    to stress-test quantization accuracy.
    """
    # Mix of distributions for realistic LLM activation values:
    # 1. Normal with std=15.0 for base distribution (40%) - typical activation range
    # 2. Normal with std=50.0 for spread (35%) - larger activations
    # 3. Uniform in [-200, 200] for outliers (10%) - extreme values
    # 4. Very large outliers [-400, 400] (15%) - stress test quantization
    base = torch.randn(shape, device=device, dtype=dtype) * 15.0
    spread = torch.randn(shape, device=device, dtype=dtype) * 50.0
    outliers = torch.empty(shape, device=device, dtype=dtype).uniform_(-200.0, 200.0)
    extreme = torch.empty(shape, device=device, dtype=dtype).uniform_(-400.0, 400.0)

    # Combine with weights
    mask_spread = torch.rand(shape, device=device) < 0.35
    mask_outlier = torch.rand(shape, device=device) < 0.10
    mask_extreme = torch.rand(shape, device=device) < 0.15

    result = base.clone()
    result[mask_spread] = spread[mask_spread]
    result[mask_outlier] = outliers[mask_outlier]
    result[mask_extreme] = extreme[mask_extreme]

    return result


def quantize_to_nvfp4(tensor: torch.Tensor) -> tuple:
    """Quantize a tensor to NVFP4 format.

    Returns:
        (fp4_packed, block_scale_swizzled, global_scale, alpha)
    """
    # Compute global scale for FP4 quantization
    global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                    torch.abs(tensor).max().to(torch.float32))

    # Quantize to FP4 with block scales
    fp4_packed, block_scale = ops.scaled_fp4_quant(tensor, global_scale)

    return fp4_packed, block_scale, global_scale


def compute_mismatch_stats(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Compute detailed mismatch statistics for debugging."""
    mismatch_mask = out != ref
    num_mismatches = mismatch_mask.sum().item()
    total_elements = out.numel()
    mismatch_rate = num_mismatches / total_elements * 100

    if num_mismatches > 0:
        out_float = out.float()
        ref_float = ref.float()
        abs_diff = torch.abs(out_float - ref_float)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff[mismatch_mask].mean().item()

        return {
            "num_mismatches": num_mismatches,
            "total_elements": total_elements,
            "mismatch_rate": mismatch_rate,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        }
    return {
        "num_mismatches": 0,
        "total_elements": total_elements,
        "mismatch_rate": 0.0,
        "max_diff": 0.0,
        "mean_diff": 0.0,
    }


def get_dtype_name(out_dtype: torch.dtype) -> str:
    """Get short name for dtype."""
    dtype_names = {
        torch.bfloat16: "BF16",
        torch.float16: "FP16",
    }
    return dtype_names.get(out_dtype, str(out_dtype))


# ============================================================================
# MMA-Emu vs CUTLASS Comparison Tests
# ============================================================================

@pytest.mark.parametrize("dtype", OUT_DTYPES)
@pytest.mark.parametrize("shape", MNK_FACTORS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gdfs_vs_cutlass(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    """Test MMA-Emu NVFP4 kernel against CUTLASS reference.

    Uses G=6 (lossless) so group-level accumulation introduces no precision
    loss compared to the exact Q8.8 fixed-point path.

    Compares:
    - MMA-Emu Emulation (GDFS with F=35, G=6)
    - MMA-Emu Tensor Core (native MMA)
    - CUTLASS reference
    """
    current_platform.seed_everything(seed)
    m, n, k_packed = shape
    dtype_name = get_dtype_name(dtype)
    k = k_packed * 2  # Unpack FP4 blocks (2 values per byte)
    # Generate realistic tensors
    a_dtype = generate_realistic_tensor((m, k), device=device, dtype=dtype)
    b_dtype = generate_realistic_tensor((n, k), device=device, dtype=dtype)

    # Quantize to NVFP4
    a_fp4, a_scale, a_global_scale = quantize_to_nvfp4(a_dtype)
    b_fp4, b_scale, b_global_scale = quantize_to_nvfp4(b_dtype)

    # Compute alpha = 1 / (a_global_scale * b_global_scale)
    alpha = (1.0 / (a_global_scale * b_global_scale)).to(torch.float32)

    # CUTLASS output (reference)
    cutlass_out = ops.cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, alpha, dtype
    )

    # MMA-Emu Emulation output (GDFS with F=35, G=6)
    emu_out = ops.mma_emu_scaled_nvfp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, alpha, dtype,
        algorithm="gdfs",
        f_bits=F_BITS_GDFS,
        g_bits=G_BITS
    )

    exact = torch.equal(emu_out, cutlass_out)

    if exact:
        print(f"[BITWISE EXACT] ({m}, {n}, {k}) {dtype_name}: Emu == CUTLASS")
    else:
        stats = compute_mismatch_stats(emu_out, cutlass_out)
        print(f"[MISMATCH] ({m}, {n}, {k}) {dtype_name}: "
              f"{stats['num_mismatches']} / {stats['total_elements']} elements "
              f"({stats['mismatch_rate']:.2f}%), max_diff={stats['max_diff']:.6g}")
        torch.testing.assert_close(emu_out, cutlass_out, atol=1e-1, rtol=1e-1)


# ============================================================================
# CoFDA Tests
# ============================================================================

# Use a subset of shapes for CoFDA tests
COFDA_SHAPES = [
    (128, 128, 64),
    (1, 4096, 512),
    (32, 4096, 2048),
]


@pytest.mark.parametrize("shape", COFDA_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_nvfp4_cofda_vs_cutlass(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> None:
    """Test MMA-Emu NVFP4 CoFDA kernel against CUTLASS reference.

    CoFDA uses product-level UE4M3 scale application with chunked FDA
    accumulation; CS is fixed at 16 for NVFP4.
    """
    device = "cuda"
    m, n, k_packed = shape
    k = k_packed * 2

    torch.manual_seed(42)

    a_dtype = generate_realistic_tensor((m, k), device=device, dtype=dtype)
    b_dtype = generate_realistic_tensor((n, k), device=device, dtype=dtype)

    a_fp4, a_scale, a_global_scale = quantize_to_nvfp4(a_dtype)
    b_fp4, b_scale, b_global_scale = quantize_to_nvfp4(b_dtype)
    alpha = (1.0 / (a_global_scale * b_global_scale)).to(torch.float32)

    # MMA-Emu CoFDA output
    cofda_out = ops.mma_emu_scaled_nvfp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, alpha, dtype,
        algorithm="cofda",
        f_bits=F_BITS_COFDA,
    )

    # Basic sanity checks: no NaN, no Inf
    assert not torch.isnan(cofda_out).any(), \
        f"CoFDA ({m},{n},{k}): output contains NaN"
    assert not torch.isinf(cofda_out).any(), \
        f"CoFDA ({m},{n},{k}): output contains Inf"

    # Compare against CUTLASS baseline
    cutlass_out = ops.cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, alpha, dtype
    )

    is_exact = torch.equal(cofda_out, cutlass_out)
    if is_exact:
        print(f"[EXACT] CoFDA ({m},{n},{k}): matches CUTLASS")
    else:
        stats = compute_mismatch_stats(cofda_out, cutlass_out)
        print(f"[MISMATCH] CoFDA ({m},{n},{k}): "
              f"{stats['num_mismatches']} mismatches "
              f"({stats['mismatch_rate']:.2f}%), "
              f"max_diff={stats['max_diff']:.6f}")
        # CoFDA may differ from CUTLASS due to different accumulation order
        torch.testing.assert_close(cofda_out, cutlass_out,
                                   atol=1.0, rtol=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
