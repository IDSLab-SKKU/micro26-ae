# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance test for the SM90 FP8 cutlass scaled_mm path.

A given output row must be bit-identical regardless of how many tokens (M) are
processed together. This holds only when the dispatcher serves every M from a
single kernel config, since the tile shape fixes the K-reduction order.
"""
import pytest
import torch

from tests.kernels.utils import baseline_scaled_mm, to_fp8
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

capability = current_platform.get_device_capability()
capability = capability[0] * 10 + capability[1]

# The dispatch under test is SM90 (Hopper) specific.
pytestmark = pytest.mark.skipif(
    capability != 90,
    reason="SM90 FP8 batch-invariant dispatch only applies to Hopper (sm90)")

# Straddle the M thresholds a throughput-tuned dispatcher would switch config
# at, since those are where batch-variance would show up.
M_VALUES = [1, 16, 17, 64, 65, 128, 129, 256]
MMAX = max(M_VALUES)


@pytest.mark.parametrize("n,k", [(4096, 4096), (6144, 4096), (28672, 4096)])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_fp8_scaled_mm_batch_invariant(n, k, use_bias, out_dtype):
    device = "cuda"
    torch.manual_seed(0)

    a = to_fp8(torch.randn((MMAX, k), device=device))
    b = to_fp8(torch.randn((n, k), device=device).t())  # [k, n]

    scale_a = torch.randn((MMAX, 1), device=device, dtype=torch.float32).abs()
    scale_b = torch.randn((1, n), device=device, dtype=torch.float32).abs()
    bias = (torch.rand((n, ), device=device, dtype=out_dtype) * 10
            if use_bias else None)

    # Reference: full batch at M = MMAX.
    ref = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    # Sanity: kernel output is numerically correct (not merely self-consistent).
    baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    torch.testing.assert_close(ref, baseline, rtol=5e-1, atol=1.5e-1)

    # Batch-invariance: row i must be bit-identical at any M.
    for m in M_VALUES:
        out_m = ops.cutlass_scaled_mm(a[:m], b, scale_a[:m], scale_b, out_dtype,
                                      bias)
        assert torch.equal(out_m, ref[:m]), (
            f"batch-variance detected at M={m} "
            f"(n={n}, k={k}, bias={use_bias}, dtype={out_dtype})")
