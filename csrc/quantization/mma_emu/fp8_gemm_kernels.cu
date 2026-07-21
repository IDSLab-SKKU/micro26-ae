/*
 * MMA-Emu FP8 GEMM — torch operator entry point
 *
 * Emulates the internal MMA accumulation arithmetic (FDA / CoFDA / GDFS) for
 * FP8 E4M3 on CUDA cores. Accepted parameter values: core/design_space.cuh.
 */

#include <torch/all.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include "core/design_space.cuh"
#include "gemm/scaled_fp8_mm.cuh"

namespace ds = vllm::mma_emu::design_space;

// ============================================================================
// MMA-Emu Scaled FP8 MM Entry Point (matches cutlass_scaled_mm interface)
// Global namespace for torch binding compatibility
// ============================================================================

void mma_emu_scaled_fp8_mm(
    torch::Tensor& c,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    std::optional<torch::Tensor> const& bias,
    int64_t algorithm,        // ds::Algorithm
    int64_t f_bits,           // fractional bits F
    int64_t g_bits,           // GDFS intra-group bits G
    int64_t group_size,       // GDFS group size GS
    int64_t chunk_size) {     // CoFDA chunk size CS

    // Validate inputs
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
                "MMA-Emu scaled_mm: All tensors must be 2D");
    TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
                b.size(1) == c.size(1),
                "MMA-Emu scaled_mm: Matrix dimensions must match");

    // Check for strides and alignment (matching cutlass_scaled_mm)
    TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1,
                "MMA-Emu scaled_mm: A and C must be row-major");
    TORCH_CHECK(b.stride(0) == 1,
                "MMA-Emu scaled_mm: B must be column-major");

    if (bias) {
        TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                    bias->dim() == 1,
                    "MMA-Emu scaled_mm: bias must be 1D with N elements");
    }

    // Validate algorithm. The native tensor-core path is provided by
    // cutlass_scaled_mm, not by this kernel.
    TORCH_CHECK(algorithm >= 1 && algorithm <= 3,
        "algorithm must be 1 (GDFS), 2 (CoFDA, C-fused), or 3 (CoFDA, C-decoupled), got ", algorithm);

    at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

    if (algorithm == ds::kGDFS) {
        // GDFS emulation path
        TORCH_CHECK(ds::contains(ds::GDFS_F, f_bits),
            "FP8 GDFS: f_bits must be one of {7, 9, 10, 11, 13, 15, 25, 35}, "
            "got ", f_bits);
        TORCH_CHECK(ds::contains(ds::FP8_G, g_bits),
            "FP8 GDFS: g_bits must be one of {3, 4, 5, 6, 8, 13, 32}, "
            "got ", g_bits);
        TORCH_CHECK(ds::contains(ds::FP8_GS, group_size),
            "FP8 GDFS: group_size must be 8 or 16, got ", group_size);

        vllm::mma_emu::mma_emu_scaled_fp8_mm_emu(
            c, a, b, a_scales, b_scales, bias,
            static_cast<int>(algorithm),
            static_cast<int>(f_bits),
            static_cast<int>(g_bits),
            static_cast<int>(group_size),
            static_cast<int>(chunk_size));
    } else if (algorithm == ds::kCoFDA) {
        // CoFDA emulation path (C-fused)
        TORCH_CHECK(ds::contains(ds::COFDA_F, f_bits),
            "FP8 CoFDA: f_bits must be one of "
            "{3, 5, 7, 9, 10, 11, 12, 13, 17, 21, 25}, got ", f_bits);
        TORCH_CHECK(ds::contains(ds::FP8_CS, chunk_size),
            "FP8 CoFDA: chunk_size must be 16 or 32, got ",
            chunk_size);

        vllm::mma_emu::mma_emu_scaled_fp8_mm_emu(
            c, a, b, a_scales, b_scales, bias,
            static_cast<int>(algorithm),
            static_cast<int>(f_bits),
            static_cast<int>(g_bits),
            static_cast<int>(group_size),
            static_cast<int>(chunk_size));
    } else {
        // C-decoupled CoFDA emulation path
        TORCH_CHECK(ds::contains(ds::COFDA_F, f_bits),
            "FP8 CoFDA (C-decoupled): f_bits must be one of "
            "{3, 5, 7, 9, 10, 11, 12, 13, 17, 21, 25}, got ", f_bits);
        TORCH_CHECK(ds::contains(ds::FP8_CS, chunk_size),
            "FP8 CoFDA (C-decoupled): chunk_size must be 16 or 32, got ", chunk_size);

        vllm::mma_emu::mma_emu_scaled_fp8_mm_emu(
            c, a, b, a_scales, b_scales, bias,
            static_cast<int>(algorithm),
            static_cast<int>(f_bits),
            static_cast<int>(g_bits),
            static_cast<int>(group_size),
            static_cast<int>(chunk_size));
    }
}

