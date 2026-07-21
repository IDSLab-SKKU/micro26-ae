# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GEMM-level tests for the MMA-Emu FP8 kernel.

The emulation kernel is compared against CUTLASS, which runs the GPU's native
tensor-core MMA. On Blackwell the native FP8 configuration is FDA with F=25 and
a K-tile of 32, so CoFDA at F=25 / CS=32 must reproduce CUTLASS bit for bit.
GDFS is a different algorithm and is checked within tolerance.

Run: pytest tests/kernels/quantization/test_mma_emu_scaled_fp8_mm.py -v -s
"""

import pytest
import torch

from tests.kernels.utils import to_fp8
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

# ============================================================================
# Test Configuration
# ============================================================================

# Matrix dimension factors from test_cutlass_scaled_mm.py
MNK_FACTORS = [
    (1, 4096, 4096),      # decode: single token, LLM hidden dim
    (16, 2048, 1024),
    (128, 1024, 512),
]

# Output dtypes to test
OUT_DTYPES = [torch.bfloat16, torch.float16]

# Realistic scale factors for FP8 quantization
# FP8 E4M3 has max value ~448, so scale factors typically range from 0.001-1.0
# These represent typical activation/weight scales observed in LLM inference
SCALE_FACTORS = [
    (0.0125, 0.0156),
    (0.0312, 0.0234),
]

# Emulation configuration
F_BITS = 25
CHUNK_SIZE = 32  # 4, 8, 16, or 32


# ============================================================================
# Helper Functions
# ============================================================================

def generate_realistic_tensor(shape: tuple, device: str = "cuda") -> torch.Tensor:
    """Generate tensor with realistic LLM activation/weight distribution.

    LLM tensors typically have:
    - Values concentrated around 0 but with significant spread
    - Some outliers (important for quantization accuracy)
    - Range roughly [-10, 10] for activations, [-2, 2] for weights

    This uses a mixture of uniform and normal distributions to create
    values that better exercise the FP8 dynamic range.
    """  # noqa: E501
    # Mix of distributions for realistic values:
    # 1. Uniform in [-1, 1] for base (50%)
    # 2. Normal with std=2.0 for spread (40%)
    # 3. Uniform in [-8, 8] for outliers (10%)
    base = torch.empty(shape, device=device).uniform_(-1.0, 1.0)
    spread = torch.randn(shape, device=device) * 2.0
    outliers = torch.empty(shape, device=device).uniform_(-8.0, 8.0)

    # Combine with weights
    mask_spread = torch.rand(shape, device=device) < 0.4
    mask_outlier = torch.rand(shape, device=device) < 0.1

    result = base.clone()
    result[mask_spread] = spread[mask_spread]
    result[mask_outlier] = outliers[mask_outlier]

    return result


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
# CoFDA emulation vs CUTLASS (bit-exact)
# ============================================================================

@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="FP8 is not supported on this GPU type."
)
@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("scale_a_val,scale_b_val", SCALE_FACTORS)
def test_fp8_cofda_bitwise_vs_cutlass(
    m: int, n: int, k: int,
    out_dtype: torch.dtype,
    scale_a_val: float, scale_b_val: float
):
    """CoFDA at the native Blackwell config must match CUTLASS bit for bit."""
    device = "cuda"
    dtype_name = get_dtype_name(out_dtype)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Generate realistic tensors with wider value distribution
    a = to_fp8(generate_realistic_tensor((m, k), device=device))
    b = to_fp8(generate_realistic_tensor((n, k), device=device).t())

    # Use realistic scale factors
    scale_a = torch.tensor([scale_a_val], device=device, dtype=torch.float32)
    scale_b = torch.tensor([scale_b_val], device=device, dtype=torch.float32)

    # CUTLASS output (reference)
    cutlass_out = ops.cutlass_scaled_mm(
        a, b, scale_a, scale_b, out_dtype, None
    )

    # MMA-Emu CoFDA emulation output
    emu_out = ops.mma_emu_scaled_fp8_mm(
        a, b, scale_a, scale_b, out_dtype, None,
        algorithm="cofda",
        f_bits=F_BITS,
        chunk_size=CHUNK_SIZE
    )

    exact = torch.equal(emu_out, cutlass_out)

    if exact:
        print(f"[BITWISE EXACT] ({m}, {n}, {k}) {dtype_name} "
              f"s=({scale_a_val},{scale_b_val}): Emu == CUTLASS")
    else:
        stats = compute_mismatch_stats(emu_out, cutlass_out)
        print(f"[MISMATCH] ({m}, {n}, {k}) {dtype_name} "
              f"s=({scale_a_val},{scale_b_val}): "
              f"{stats['num_mismatches']} / {stats['total_elements']} elements, "
              f"max_diff={stats['max_diff']:.6g}")

    assert exact, (
        f"({m}, {n}, {k}) {dtype_name} s=({scale_a_val},{scale_b_val}): "
        f"CoFDA F={F_BITS} CS={CHUNK_SIZE} is not bit-exact against CUTLASS"
    )


# ============================================================================
# GDFS Tests
# ============================================================================

GDFS_SIZES = [
    (1, 4096, 4096),
    (128, 1024, 512),
]


@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="FP8 is not supported on this GPU type."
)
@pytest.mark.parametrize("m,n,k", GDFS_SIZES)
@pytest.mark.parametrize("g_bits", [3, 4, 5, 6, 8, 13, 32])
@pytest.mark.parametrize("group_size", [8, 16])
def test_fp8_gdfs_vs_cutlass(
    m: int, n: int, k: int,
    g_bits: int, group_size: int
):
    """GDFS vs the CUTLASS baseline.

    GDFS with G=32 (lossless) and F=25 should match CUTLASS exactly.
    Lower G values introduce truncation, so we use tolerance-based comparison.
    """
    device = "cuda"
    out_dtype = torch.bfloat16

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    a = to_fp8(generate_realistic_tensor((m, k), device=device))
    b = to_fp8(generate_realistic_tensor((n, k), device=device).t())

    scale_a = torch.tensor([0.0125], device=device, dtype=torch.float32)
    scale_b = torch.tensor([0.0156], device=device, dtype=torch.float32)

    # CUTLASS reference
    cutlass_out = ops.cutlass_scaled_mm(
        a, b, scale_a, scale_b, out_dtype, None
    )

    # GDFS output
    gdfs_out = ops.mma_emu_scaled_fp8_mm(
        a, b, scale_a, scale_b, out_dtype, None,
        algorithm="gdfs",
        f_bits=F_BITS,
        g_bits=g_bits,
        group_size=group_size
    )

    # Verify no NaN/Inf
    assert not torch.isnan(gdfs_out).any(), \
        f"GDFS output contains NaN (g={g_bits}, gs={group_size})"
    assert not torch.isinf(gdfs_out).any(), \
        f"GDFS output contains Inf (g={g_bits}, gs={group_size})"

    # CUTLASS uses FDA internally, so GDFS is a different algorithm —
    # compare with tolerance even at G=32 (lossless group accum)
    gdfs_float = gdfs_out.float()
    cutlass_float = cutlass_out.float()
    atol = 0.05
    rtol = 0.05
    close = torch.allclose(gdfs_float, cutlass_float, atol=atol, rtol=rtol)

    if close:
        print(f"[CLOSE] ({m}, {n}, {k}) GDFS g={g_bits} "
              f"gs={group_size}: PASSED (atol={atol}, rtol={rtol})")
    else:
        stats = compute_mismatch_stats(gdfs_out, cutlass_out)
        print(f"[DEVIATION] ({m}, {n}, {k}) GDFS g={g_bits} "
              f"gs={group_size}: max_diff={stats['max_diff']:.6f}, "
              f"mean_diff={stats['mean_diff']:.6f}")

    assert close, (
        f"({m}, {n}, {k}) GDFS g={g_bits} gs={group_size}: "
        f"Output deviates too much from CUTLASS"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
