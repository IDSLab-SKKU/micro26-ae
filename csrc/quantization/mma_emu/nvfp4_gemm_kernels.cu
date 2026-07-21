/*
 * MMA-Emu NVFP4 GEMM — torch operator entry point
 *
 * This file implements NVFP4 GEMM operations with configurable accumulation
 * algorithms to study the effects of intermediate rounding on low-precision
 * LLM inference with 4-bit quantization.
 *
 * Supports:
 * - NVFP4 (E2M1) format with UE4M3 block scales (scale_vec::4X)
 * - Block size of 16 elements
 * - Alpha epilogue for global scale application
 * - Output dtype: BF16 or FP16
 * - Three algorithms: GDFS, CoFDA (C-fused), CoFDA (C-decoupled)
 */

#include <torch/all.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "core/design_space.cuh"
#include "gemm/scaled_nvfp4_mm.cuh"

namespace ds = vllm::mma_emu::design_space;

// ============================================================================
// Input validation macros
// ============================================================================
#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, m, ": Inconsistent tensor type, expected ", #st)
#define CHECK_TH_CUDA(x, m) \
  TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

// FP4 packed as uint8 (2 x E2M1 per byte)
constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
// Scale factor dtype: Float8_e4m3fn (UE4M3)
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

// ============================================================================
// MMA-Emu Scaled NVFP4 MM Entry Point (matches cutlass_scaled_fp4_mm interface)
// Global namespace for torch binding compatibility
// ============================================================================

void mma_emu_scaled_nvfp4_mm(
    torch::Tensor& D,                           // Output [M, N]
    torch::Tensor const& A,                     // Activation FP4 packed [M, K/2] uint8
    torch::Tensor const& B,                     // Weight FP4 packed [N, K/2] uint8
    torch::Tensor const& A_sf,                  // Activation scales [M_rounded, K/16_rounded]
    torch::Tensor const& B_sf,                  // Weight scales [N_rounded, K/16_rounded]
    torch::Tensor const& alpha,                 // Global scale [1] float32
    int64_t algorithm,                          // ds::Algorithm
    int64_t f_bits,                             // fractional bits F
    int64_t g_bits) {                           // GDFS intra-group bits G

    // ========================================================================
    // Input Validation (consistent with cutlass_scaled_fp4_mm_sm120a)
    // ========================================================================

    // Validate tensor types: CUDA, contiguous, and correct dtype
    CHECK_INPUT(A, FLOAT4_E2M1X2, "A");
    CHECK_INPUT(B, FLOAT4_E2M1X2, "B");
    CHECK_INPUT(A_sf, SF_DTYPE, "A_sf");
    CHECK_INPUT(B_sf, SF_DTYPE, "B_sf");
    CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

    // Validate output tensor
    CHECK_TH_CUDA(D, "D");
    CHECK_CONTIGUOUS(D, "D");

    // Validate dimensions: all must be 2D matrices
    TORCH_CHECK(A.dim() == 2, "A must be a 2D matrix");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D matrix");
    TORCH_CHECK(D.dim() == 2, "D must be a 2D matrix");
    TORCH_CHECK(A_sf.dim() == 2, "A_sf must be a 2D matrix");
    TORCH_CHECK(B_sf.dim() == 2, "B_sf must be a 2D matrix");

    // Extract dimensions
    // A is [M, K/2] (packed), B is [N, K/2] (packed), D is [M, N]
    auto const m = A.sizes()[0];
    auto const n = B.sizes()[0];
    auto const k = A.sizes()[1] * 2;  // Actual K (unpacked from packed FP4)

    // Validate shape compatibility for matrix multiplication
    // A[M, K/2] x B^T[N, K/2] = D[M, N]
    TORCH_CHECK(A.sizes()[1] == B.sizes()[1],
                "A and B shapes cannot be multiplied (", A.sizes()[0], "x",
                A.sizes()[1], " and ", B.sizes()[0], "x", B.sizes()[1], ")");

    // Validate output shape
    TORCH_CHECK(D.sizes()[0] == m && D.sizes()[1] == n,
                "Output D must be [", m, ", ", n, "], got [",
                D.sizes()[0], ", ", D.sizes()[1], "]");

    // Alignment requirements for NVFP4 GEMM
    // K must be divisible by 32 (16 elements per block * 2 for packed format alignment)
    // N must be divisible by 32 for efficient memory access
    constexpr int alignment = 32;
    TORCH_CHECK(k % alignment == 0,
                "Expected K to be divisible by ", alignment,
                ", but got A shape: (", A.sizes()[0], "x", A.sizes()[1],
                "), K: ", k, ".");
    TORCH_CHECK(n % alignment == 0,
                "Expected N to be divisible by ", alignment,
                ", but got B shape: (", B.sizes()[0], "x", B.sizes()[1], ").");

    // Helper lambda for rounding up
    auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };

    // Calculate expected scale factor dimensions
    // Scale factors are computed per 16-element block with padding/swizzling
    int64_t rounded_m = round_up(m, 128);
    int64_t rounded_n = round_up(n, 128);
    // Since K is divisible by 32 (alignment), K/16 is guaranteed to be even
    int64_t rounded_k = round_up(k / 16, 4);

    // Validate scale factor shapes
    TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
                "A_sf and B_sf shapes cannot be multiplied (",
                A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ", B_sf.sizes()[0],
                "x", B_sf.sizes()[1], ")");
    TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
                "A_sf must be padded and swizzled to shape (", rounded_m,
                "x", rounded_k, "), but got shape (", A_sf.sizes()[0], "x",
                A_sf.sizes()[1], ")");
    TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
                "B_sf must be padded and swizzled to shape (", rounded_n,
                "x", rounded_k, "), but got shape (", B_sf.sizes()[0], "x",
                B_sf.sizes()[1], ")");

    // Validate output dtype (must be BF16 or FP16)
    auto out_dtype = D.dtype();
    TORCH_CHECK(out_dtype == at::ScalarType::BFloat16 ||
                out_dtype == at::ScalarType::Half,
                "Unsupported output dtype for MMA-Emu scaled_nvfp4_mm (",
                out_dtype, "). Expected BFloat16 or Half.");

    // ========================================================================
    // Algorithm Validation
    // ========================================================================

    // NVFP4 supports GDFS and C-fused CoFDA. The native tensor-core path is
    // provided by cutlass_scaled_fp4_mm.
    TORCH_CHECK(algorithm == ds::kGDFS ||
                    algorithm == ds::kCoFDA,
                "MMA-Emu scaled_nvfp4_mm: algorithm must be 1 (GDFS) or "
                "2 (CoFDA), got ",
                algorithm);

    // ========================================================================
    // Kernel Dispatch
    // ========================================================================

    at::cuda::OptionalCUDAGuard const device_guard(device_of(A));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

    // NVFP4 is fixed at CS = 16 (CoFDA) and GS = 16 (GDFS), so neither is a
    // kernel parameter.
    if (algorithm == ds::kGDFS) {
        // GDFS path
        TORCH_CHECK(ds::contains(ds::GDFS_F, f_bits),
                    "NVFP4 GDFS: f_bits must be one of "
                    "{7, 9, 10, 11, 13, 15, 25, 35}, got ",
                    f_bits);
        TORCH_CHECK(ds::contains(ds::FP4_G, g_bits),
                    "NVFP4 GDFS: g_bits must be one of {3, 4, 5, 6}, got ",
                    g_bits);
    } else {
        // CoFDA (C-fused, algorithm 2) and C-decoupled CoFDA (algorithm 3)
        TORCH_CHECK(ds::contains(ds::COFDA_F, f_bits),
                    "NVFP4 CoFDA: f_bits must be one of "
                    "{3, 5, 7, 9, 10, 11, 12, 13, 17, 21, 25}, got ",
                    f_bits);
    }

    vllm::mma_emu::mma_emu_scaled_nvfp4_mm_emu(
        D, A, B, A_sf, B_sf, alpha,
        static_cast<int>(algorithm),
        static_cast<int>(f_bits),
        static_cast<int>(g_bits));
}
