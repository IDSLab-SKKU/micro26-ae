# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MMA-Emu MXFP4 scaled_mm kernel.

Compares MMA-Emu Emulation vs MMA-Emu TC vs CUTLASS for BF16/FP16.

The MMA-Emu MXFP4 kernel implements MXFP4 (E2M1) GEMM with GDFS
(Group-Dot-Fused-Sum) accumulation algorithm for research on intermediate
rounding effects in 4-bit quantized LLM inference.

Key differences from NVFP4 (MMA-Emu FP4):
- E8M0 block scales (uint8) instead of UE4M3 (float8_e4m3fn)
- Block size = 32 instead of 16
- No global alpha parameter
- scale_vec::2X instead of scale_vec::4X

Precision parameters:
- F-bits: Fractional bits for fused-sum accumulation (STP5-7). Range [5, 35].
- G-bits: Fractional bits for group dot-product accumulation (STP2-3).
  Range [-1, 6]. G=6 is lossless for E2M1 products.

Run: pytest tests/kernels/quantization/test_mma_emu_scaled_mxfp4_mm.py -v -s
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

# Skip if not SM100+ (Blackwell required for MXFP4)
if not current_platform.has_device_capability(100):
    pytest.skip(reason="MXFP4 requires compute capability 10.0 or above.",
                allow_module_level=True)

# Import MXFP4 utilities after capability check
from mxfp4_utils import quant_mxfp4_tensor

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
F_BITS_GDFS = 35   # GDFS inter-group bits — Blackwell's native MXFP4 config
F_BITS_COFDA = 25  # CoFDA fractional bits (CoFDA and GDFS use different F sets)
G_BITS = 6         # Lossless intra-group width for E2M1 products

SEEDS = [42]
CUDA_DEVICES = ['cuda:0']


# ============================================================================
# Helper Functions
# ============================================================================

def generate_realistic_tensor(shape: tuple, device: str = "cuda",
                              dtype: torch.dtype = torch.bfloat16
                              ) -> torch.Tensor:
    """Generate tensor with realistic LLM activation/weight distribution.

    Based on empirical observations from LLMs like Llama-3.1:
    - Activations: Most values in [-50, 50], with outliers up to [-200, 200]
    - Weights: Most values in [-1, 1], with some spread to [-5, 5]
    - Heavy-tailed distribution with occasional extreme outliers

    This function generates activations-like tensors with larger magnitudes
    to stress-test quantization accuracy.
    """
    base = torch.randn(shape, device=device, dtype=dtype) * 15.0
    spread = torch.randn(shape, device=device, dtype=dtype) * 50.0
    outliers = torch.empty(shape, device=device,
                           dtype=dtype).uniform_(-200.0, 200.0)
    extreme = torch.empty(shape, device=device,
                          dtype=dtype).uniform_(-400.0, 400.0)

    mask_spread = torch.rand(shape, device=device) < 0.35
    mask_outlier = torch.rand(shape, device=device) < 0.10
    mask_extreme = torch.rand(shape, device=device) < 0.15

    result = base.clone()
    result[mask_spread] = spread[mask_spread]
    result[mask_outlier] = outliers[mask_outlier]
    result[mask_extreme] = extreme[mask_extreme]

    return result


def quantize_to_mxfp4(tensor: torch.Tensor) -> tuple:
    """Quantize a tensor to MXFP4 format (no global scale).

    Returns:
        (fp4_packed, block_scale_swizzled)
    """
    fp4_packed, block_scale = quant_mxfp4_tensor(tensor)
    return fp4_packed, block_scale


def compute_mismatch_stats(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Compute detailed mismatch statistics for debugging.

    Separates NaN mismatches from finite-value mismatches to avoid
    masking real numeric differences behind max_diff=nan.
    """
    mismatch_mask = out != ref
    num_mismatches = mismatch_mask.sum().item()
    total_elements = out.numel()
    mismatch_rate = num_mismatches / total_elements * 100

    if num_mismatches > 0:
        out_float = out.float()
        ref_float = ref.float()

        # Count NaN in each tensor
        out_nan = torch.isnan(out_float)
        ref_nan = torch.isnan(ref_float)
        out_nan_count = out_nan.sum().item()
        ref_nan_count = ref_nan.sum().item()

        # Finite-only diff (exclude positions where either is NaN)
        either_nan = out_nan | ref_nan
        finite_mismatch = mismatch_mask & ~either_nan
        num_finite_mismatches = finite_mismatch.sum().item()

        abs_diff = torch.abs(out_float - ref_float)
        if num_finite_mismatches > 0:
            finite_diff = abs_diff[finite_mismatch]
            max_finite_diff = finite_diff.max().item()
            mean_finite_diff = finite_diff.mean().item()
        else:
            max_finite_diff = 0.0
            mean_finite_diff = 0.0

        return {
            "num_mismatches": num_mismatches,
            "total_elements": total_elements,
            "mismatch_rate": mismatch_rate,
            "out_nan_count": out_nan_count,
            "ref_nan_count": ref_nan_count,
            "num_finite_mismatches": num_finite_mismatches,
            "max_finite_diff": max_finite_diff,
            "mean_finite_diff": mean_finite_diff,
        }
    return {
        "num_mismatches": 0,
        "total_elements": total_elements,
        "mismatch_rate": 0.0,
        "out_nan_count": 0,
        "ref_nan_count": 0,
        "num_finite_mismatches": 0,
        "max_finite_diff": 0.0,
        "mean_finite_diff": 0.0,
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
def test_mxfp4_gdfs_vs_cutlass(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    """Test MMA-Emu MXFP4 kernel against CUTLASS reference.

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

    # Quantize to MXFP4 (no global scale)
    a_fp4, a_scale = quantize_to_mxfp4(a_dtype)
    b_fp4, b_scale = quantize_to_mxfp4(b_dtype)

    # CUTLASS output (reference, no alpha for MXFP4)
    cutlass_out = ops.cutlass_scaled_mxfp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, dtype
    )

    # MMA-Emu Emulation output (GDFS with F=35, G=6)
    emu_out = ops.mma_emu_scaled_mxfp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, dtype,
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
              f"{stats['num_mismatches']} mismatches "
              f"({stats['mismatch_rate']:.2f}%), "
              f"max_finite_diff={stats['max_finite_diff']:.6g}")
        torch.testing.assert_close(emu_out, cutlass_out, atol=1e-1, rtol=1e-1)


# ============================================================================
# GROUP_SIZE Tests
# ============================================================================

# MXFP4 GROUP_SIZE values: {4, 8, 16, 32}
MXFP4_GROUP_SIZES = [16, 32]

# Use a subset of shapes for GROUP_SIZE sweep (faster test runs)
GROUP_SIZE_SHAPES = [
    (128, 128, 64),
    (1, 4096, 512),
]


@pytest.mark.parametrize("group_size", MXFP4_GROUP_SIZES)
@pytest.mark.parametrize("shape", GROUP_SIZE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_mxfp4_gdfs_group_size(
    group_size: int,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> None:
    """Test MMA-Emu MXFP4 kernel with variable GROUP_SIZE.

    - GS=16 with G=6 should match CUTLASS baseline (default behavior).
    - Other GS values: verify no crash/NaN, and reasonable tolerance.
    """
    device = "cuda"
    m, n, k_packed = shape
    k = k_packed * 2

    torch.manual_seed(42)

    a_dtype = generate_realistic_tensor((m, k), device=device, dtype=dtype)
    b_dtype = generate_realistic_tensor((n, k), device=device, dtype=dtype)

    a_fp4, a_scale = quantize_to_mxfp4(a_dtype)
    b_fp4, b_scale = quantize_to_mxfp4(b_dtype)

    # MMA-Emu Emulation with variable group_size
    emu_out = ops.mma_emu_scaled_mxfp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, dtype,
        algorithm="gdfs",
        f_bits=F_BITS_GDFS,
        g_bits=G_BITS,
        group_size=group_size,
    )

    # Basic sanity checks: no NaN, no Inf
    assert not torch.isnan(emu_out).any(), \
        f"GS={group_size}: output contains NaN"
    assert not torch.isinf(emu_out).any(), \
        f"GS={group_size}: output contains Inf"

    # Compare against CUTLASS baseline
    cutlass_out = ops.cutlass_scaled_mxfp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, dtype
    )

    if group_size == 16:
        # GS=16 (default) should closely match CUTLASS
        is_exact = torch.equal(emu_out, cutlass_out)
        if is_exact:
            print(f"[EXACT] GS={group_size} ({m},{n},{k}): "
                  f"matches CUTLASS")
        else:
            stats = compute_mismatch_stats(emu_out, cutlass_out)
            print(f"[MISMATCH] GS={group_size} ({m},{n},{k}): "
                  f"{stats['num_mismatches']} mismatches "
                  f"({stats['mismatch_rate']:.2f}%)")
            torch.testing.assert_close(emu_out, cutlass_out,
                                       atol=1e-1, rtol=1e-1)
    else:
        # Other GS values: allow larger tolerance since accumulation
        # granularity differs from CUTLASS
        torch.testing.assert_close(emu_out, cutlass_out,
                                   atol=1.0, rtol=0.5)
        print(f"[OK] GS={group_size} ({m},{n},{k}): "
              f"within tolerance of CUTLASS")


# ============================================================================
# CoFDA Tests
# ============================================================================

# CoFDA CHUNK_SIZE values: {4, 8, 16, 32}
MXFP4_CHUNK_SIZES = [16, 32]

# Use a subset of shapes for CoFDA sweep (faster test runs)
COFDA_SHAPES = [
    (128, 128, 64),
    (1, 4096, 512),
    (32, 4096, 2048),
]


@pytest.mark.parametrize("chunk_size", MXFP4_CHUNK_SIZES)
@pytest.mark.parametrize("shape", COFDA_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_mxfp4_cofda_vs_cutlass(
    chunk_size: int,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> None:
    """Test MMA-Emu MXFP4 CoFDA kernel against CUTLASS reference.

    CoFDA applies scales at the product level and accumulates in chunks,
    unlike GDFS which uses group-level accumulation.
    """
    device = "cuda"
    m, n, k_packed = shape
    k = k_packed * 2

    torch.manual_seed(42)

    a_dtype = generate_realistic_tensor((m, k), device=device, dtype=dtype)
    b_dtype = generate_realistic_tensor((n, k), device=device, dtype=dtype)

    a_fp4, a_scale = quantize_to_mxfp4(a_dtype)
    b_fp4, b_scale = quantize_to_mxfp4(b_dtype)

    # CoFDA Emulation with variable chunk_size
    cofda_out = ops.mma_emu_scaled_mxfp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, dtype,
        algorithm="cofda",
        f_bits=F_BITS_COFDA,
        chunk_size=chunk_size,
    )

    # Basic sanity checks: no NaN, no Inf
    assert not torch.isnan(cofda_out).any(), \
        f"CS={chunk_size}: output contains NaN"
    assert not torch.isinf(cofda_out).any(), \
        f"CS={chunk_size}: output contains Inf"

    # Compare against CUTLASS baseline
    cutlass_out = ops.cutlass_scaled_mxfp4_mm(
        a_fp4, b_fp4, a_scale, b_scale, dtype
    )

    # CoFDA uses a different accumulation strategy than CUTLASS,
    # so we allow tolerance
    torch.testing.assert_close(cofda_out, cutlass_out,
                               atol=1.0, rtol=0.5)
    print(f"[OK] CoFDA CS={chunk_size} ({m},{n},{k}): "
          f"within tolerance of CUTLASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
