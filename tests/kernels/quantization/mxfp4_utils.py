# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import struct

import torch

from vllm._custom_ops import scaled_mxfp4_quant
from vllm.scalar_type import scalar_types

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()  # 6.0
MXFP4_BLOCK_SIZE = 32

# E2M1 lookup table (same as NVFP4 — data format is identical)
kE2M1ToFloat = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.],
                            dtype=torch.float32)


def ue8m0_to_float32(ue8m0_val: int) -> float:
    """Convert UE8M0 (unsigned 8-bit exponent) to float32.
    UE8M0 encodes the biased exponent of IEEE float32: value = 2^(e - 127)."""
    if ue8m0_val == 0:
        return 0.0  # subnormal -> treat as zero
    # Reconstruct float32: sign=0, exponent=ue8m0_val, mantissa=0
    bits = ue8m0_val << 23
    return struct.unpack('f', struct.pack('I', bits))[0]


def convert_swizzled_to_linear_mxfp4(sf_swizzled: torch.Tensor, m, k):
    """Unswizzle E8M0 block scales from CUTLASS layout to linear layout.
    Same reshape/permute as NVFP4 but with block_size=32."""
    block_size = MXFP4_BLOCK_SIZE
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4  # = 128 for MXFP4 (vs 64 for NVFP4)
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k // block_size]


def break_fp4_bytes(a, dtype):
    """Unpack two FP4 values from each uint8 byte (same as NVFP4)."""
    assert a.dtype == torch.uint8
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(dtype=dtype)


def dequantize_mxfp4_to_dtype(tensor_fp4, tensor_sf, dtype, device,
                                block_size=32):
    """Dequantize MXFP4 tensor back to high precision.
    Key difference from NVFP4: no global scale, E8M0 scales (uint8)."""
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)

    # Unswizzle E8M0 scales (uint8)
    tensor_sf = tensor_sf.view(torch.uint8)
    tensor_sf = convert_swizzled_to_linear_mxfp4(tensor_sf, m, k)

    # Convert E8M0 to float32: 2^(exponent - 127)
    sf_float = torch.zeros_like(tensor_sf, dtype=torch.float32)
    for i in range(tensor_sf.shape[0]):
        for j in range(tensor_sf.shape[1]):
            sf_float[i, j] = ue8m0_to_float32(tensor_sf[i, j].item())

    # Scale the tensor (no global scale division for MXFP4)
    out = (tensor_f32 * sf_float.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def quant_mxfp4_tensor(a: torch.Tensor):
    """Quantize tensor to MXFP4 using the CUDA kernel."""
    a_quant, a_block_scale = scaled_mxfp4_quant(a)
    return a_quant, a_block_scale
