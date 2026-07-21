# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GEMM-level tests for C-decoupled CoFDA (FP8).

In C-decoupled CoFDA the accumulator is excluded from the F-bit datapath: the
chunk's products are summed at F bits, then merged into the accumulator at
wider precision. It is therefore never bit-identical to C-fused CoFDA, but must
stay close to it and to CUTLASS.

C-decoupled is built for FP8 only; NVFP4 and MXFP4 expose gdfs and cofda.

Run: pytest tests/kernels/quantization/test_mma_emu_cofda_decoupled.py -v -s
"""

import pytest
import torch

from tests.kernels.utils import to_fp8
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

# ============================================================================
# Small GEMM sizes for quick verification
# ============================================================================

FP8_SIZES = [
    (1, 4096, 4096),
    (16, 2048, 1024),
    (128, 1024, 512),
]



def assert_close_or_cosine(actual, expected, label, atol=0.1, rtol=0.05,
                           cos_threshold=0.999):
    """Assert closeness using allclose first, fall back to cosine similarity.

    C-decoupled CoFDA intentionally changes the accumulation order vs CoFDA, so
    element-wise differences grow with K. Cosine similarity is a better
    metric for verifying that the outputs are "the same computation with
    different rounding" rather than "completely wrong".
    """
    a_f = actual.float().flatten()
    e_f = expected.float().flatten()

    if torch.allclose(a_f, e_f, atol=atol, rtol=rtol):
        return

    # Fall back to cosine similarity
    cos = torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), e_f.unsqueeze(0),
    ).item()
    diff = (a_f - e_f).abs()
    print(f"  [{label}] allclose failed: max_diff={diff.max():.4f}, "
          f"mean_diff={diff.mean():.6f}, cosine={cos:.6f}")
    assert cos >= cos_threshold, (
        f"{label}: cosine similarity {cos:.6f} < {cos_threshold} "
        f"(max_diff={diff.max():.4f})"
    )


# ============================================================================
# FP8 C-decoupled CoFDA Tests
# ============================================================================

@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="FP8 is not supported on this GPU type."
)
@pytest.mark.parametrize("m,n,k", FP8_SIZES)
@pytest.mark.parametrize("f_bits", [13, 25])
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_fp8_cofda_decoupled_vs_cofda(
    m: int, n: int, k: int,
    f_bits: int, chunk_size: int,
):
    """C-decoupled CoFDA should produce results close to CoFDA at same F/CS."""
    device = "cuda"
    out_dtype = torch.bfloat16

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    a = to_fp8(torch.randn(m, k, device=device))
    b = to_fp8(torch.randn(n, k, device=device).t())
    scale_a = torch.tensor([0.0125], device=device, dtype=torch.float32)
    scale_b = torch.tensor([0.0156], device=device, dtype=torch.float32)

    cofda_out = ops.mma_emu_scaled_fp8_mm(
        a, b, scale_a, scale_b, out_dtype, None,
        algorithm="cofda", f_bits=f_bits, chunk_size=chunk_size,
    )
    decoupled_out = ops.mma_emu_scaled_fp8_mm(
        a, b, scale_a, scale_b, out_dtype, None,
        algorithm="cofda_decoupled", f_bits=f_bits, chunk_size=chunk_size,
    )

    assert not torch.isnan(decoupled_out).any(), "C-decoupled CoFDA FP8 output has NaN"
    assert not torch.isinf(decoupled_out).any(), "C-decoupled CoFDA FP8 output has Inf"

    assert_close_or_cosine(
        decoupled_out, cofda_out,
        f"FP8 ({m},{n},{k}) F={f_bits} CS={chunk_size}",
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="FP8 is not supported on this GPU type."
)
@pytest.mark.parametrize("m,n,k", FP8_SIZES)
def test_fp8_cofda_decoupled_vs_cutlass(m: int, n: int, k: int):
    """C-decoupled CoFDA F=25,CS=32 should be close to CUTLASS."""
    device = "cuda"
    out_dtype = torch.bfloat16

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    a = to_fp8(torch.randn(m, k, device=device))
    b = to_fp8(torch.randn(n, k, device=device).t())
    scale_a = torch.tensor([0.0125], device=device, dtype=torch.float32)
    scale_b = torch.tensor([0.0156], device=device, dtype=torch.float32)

    cutlass_out = ops.cutlass_scaled_mm(
        a, b, scale_a, scale_b, out_dtype, None,
    )
    decoupled_out = ops.mma_emu_scaled_fp8_mm(
        a, b, scale_a, scale_b, out_dtype, None,
        algorithm="cofda_decoupled", f_bits=25, chunk_size=32,
    )

    assert not torch.isnan(decoupled_out).any()
    assert not torch.isinf(decoupled_out).any()

    close = torch.allclose(
        decoupled_out.float(), cutlass_out.float(), atol=0.05, rtol=0.05,
    )
    assert close, f"({m},{n},{k}): C-decoupled CoFDA F=25 CS=32 vs CUTLASS mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
