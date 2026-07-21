/*
 * MMA-Emu FP8 GEMM — CUDA-core emulation kernels
 *
 * Implements FP8 E4M3 GEMM with three accumulation algorithms:
 *
 * 1. GDFS (Group-Dot-Fused-Sum): Two-level hierarchy with G-bits (group)
 *    then F-bits (fused-sum). No per-block scale factors — FP8 uses
 *    per-tensor scales applied in the epilogue.
 *
 * 2. CoFDA (Chain-of-FDA): Single-level chunked accumulation with F-bits.
 *    FDA = CoFDA with chunk_size matching the K tile dimension.
 *
 * 3. CoFDA, C-decoupled: the accumulator C is excluded from
 *    the F-bit datapath. Products are summed at F bits, then merged with
 *    C in a second alignment stage at wider precision.
 *
 * Emulates the MMA accumulation arithmetic on CUDA cores, where every
 * alignment and truncation step is under software control. The native
 * result is produced by cutlass_scaled_mm.
 *
 * This kernel matches the cutlass_scaled_mm interface for drop-in replacement.
 *
 * Emulation parameters (accepted values: core/design_space.cuh):
 * - f_bits: fractional bits F
 * - g_bits: GDFS intra-group bits G
 * - group_size: GDFS group size GS
 * - chunk_size: CoFDA chunk size CS
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>
#include <type_traits>

#include "../core/design_space.cuh"
#include "../core/types.cuh"
#include "../core/fp32_utils.cuh"
#include "../core/tiling.cuh"
#include "../core/accumulator.cuh"
#include "../core/gdfs_group.cuh"
#include "../formats/fp8_e4m3.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// Tiling Configuration Alias
// ============================================================================

using FP8EmuConfig = TilingConfig<FP8EmulationTag>;

// ============================================================================
// Tiled MMA-Emu FP8 CoFDA GEMM Kernel with Shared Memory Optimization
// ============================================================================

/**
 * @brief Tiled MMA-Emu FP8 CoFDA GEMM kernel with shared memory optimization
 *
 * Block tiling (BM, BN, BK, thread count and per-thread tile) comes from
 * TilingConfig<FP8EmulationTag> in core/tiling.cuh.
 *
 * Memory layout:
 * - A: Row-major [M x K]
 * - B: Column-major [K x N] (transposed to [BN][BK] in shared memory)
 * - C: Row-major [M x N]
 *
 * @tparam F Fractional bits F
 * @tparam CHUNK_SIZE Products per CoFDA chunk (CS)
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 */
template<int F, int CHUNK_SIZE, typename OutDtype>
__launch_bounds__(FP8EmuConfig::NUM_THREADS)
__global__ void mma_emu_scaled_fp8_mm_cofda_kernel(
    const __nv_fp8_storage_t* __restrict__ A,  // [M, K] row-major
    const __nv_fp8_storage_t* __restrict__ B,  // [K, N] column-major
    OutDtype* __restrict__ C,                   // [M, N] row-major
    const float* __restrict__ scale_a_ptr,
    const float* __restrict__ scale_b_ptr,
    const OutDtype* __restrict__ bias,
    const int M, const int N, const int K)
{
    // Import tiling constants
    constexpr int BM = FP8EmuConfig::BM;
    constexpr int BN = FP8EmuConfig::BN;
    constexpr int BK = FP8EmuConfig::BK;
    constexpr int NUM_THREADS = FP8EmuConfig::NUM_THREADS;
    constexpr int THREAD_TILE_M = FP8EmuConfig::THREAD_TILE_M;
    constexpr int THREAD_TILE_N = FP8EmuConfig::THREAD_TILE_N;
    constexpr int BKP = BK + 1;  // padded row stride: removes stride-32 SMEM
                                 // bank conflicts.
    constexpr int A_TILE_SIZE = BM * BK;
    constexpr int B_TILE_SIZE = BN * BK;
    constexpr int A_ELEMS_PER_THREAD = (A_TILE_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int B_ELEMS_PER_THREAD = (B_TILE_SIZE + NUM_THREADS - 1) / NUM_THREADS;

    // Block index determines which output tile to compute
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int tid = threadIdx.x;

    // Compute the starting position of this block's output tile
    const int block_m = block_row * BM;
    const int block_n = block_col * BN;

    // Early exit for out-of-bounds blocks
    if (block_m >= M || block_n >= N) {
        return;
    }

    // Decode-on-load packed tiles: one uint32 per operand (decode_operand +
    // pack_decoded at load). Row stride BKP (= BK+1) pads away stride-32 bank
    // conflicts. Bs is transposed: [BN][BKP].
    __shared__ uint32_t As_dec[BM * BKP];
    __shared__ uint32_t Bs_dec[BN * BKP];

    // Thread tile mapping within block
    int thread_m, thread_n;
    get_thread_tile_position<FP8EmuConfig>(tid, thread_m, thread_n);

    // Register accumulators for this thread's outputs
    float accum[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            accum[i][j] = 0.0f;
        }
    }

    // Outer loop over K dimension in chunks of BK
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // ----------------------------------------
        // Load A tile [BM x BK], decode-on-load, zero-pad OOB
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < A_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < A_TILE_SIZE) {
                int a_row = idx / BK;
                int a_col = idx % BK;
                int global_row = block_m + a_row;
                int global_col = k_tile + a_col;
                const __nv_fp8_storage_t raw =
                    (global_row < M && global_col < K)
                        ? A[global_row * K + global_col]
                        : __nv_fp8_storage_t(0);
                const DecodedOperand d = decode_operand(static_cast<uint8_t>(raw));
                As_dec[a_row * BKP + a_col] = pack_decoded(d);
            }
        }

        // ----------------------------------------
        // Load B tile [BK x BN] transposed, decode-on-load, zero-pad OOB
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < B_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < B_TILE_SIZE) {
                int b_row = idx / BN;  // K dimension
                int b_col = idx % BN;  // N dimension
                int global_row = k_tile + b_row;
                int global_col = block_n + b_col;
                const __nv_fp8_storage_t raw =
                    (global_row < K && global_col < N)
                        ? B[global_row + global_col * K]
                        : __nv_fp8_storage_t(0);
                const DecodedOperand d = decode_operand(static_cast<uint8_t>(raw));
                Bs_dec[b_col * BKP + b_row] = pack_decoded(d);
            }
        }
        __syncthreads();

        // ----------------------------------------
        // Compute: CoFDA chunked accumulation (one cell per thread).
        // SMEM zero-pads OOB bytes, so each chunk always has CHUNK_SIZE valid
        // (possibly zero) elements — no per-element bounds check needed.
        // ----------------------------------------
        #pragma unroll
        for (int tm = 0; tm < THREAD_TILE_M; tm++) {
            #pragma unroll
            for (int tn = 0; tn < THREAD_TILE_N; tn++) {
                const int a_off = (thread_m + tm) * BKP;
                const int b_off = (thread_n + tn) * BKP;
                #pragma unroll
                for (int k_start = 0; k_start < BK; k_start += CHUNK_SIZE) {
                    const DecodedFrag a_frag{&As_dec[a_off + k_start]};
                    const DecodedFrag b_frag{&Bs_dec[b_off + k_start]};
                    accum[tm][tn] =
                        fp8_cofda_mma<F, CHUNK_SIZE>(a_frag, b_frag, accum[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // ----------------------------------------
    // Apply scales and write output
    // ----------------------------------------
    const float scale_a = *scale_a_ptr;
    const float scale_b = *scale_b_ptr;

    #pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn++) {
            int out_row = block_m + thread_m + tm;
            int out_col = block_n + thread_n + tn;

            if (out_row < M && out_col < N) {
                // Apply scale in CUTLASS order: scale_a * (scale_b * acc)
                float val = scale_a * (scale_b * accum[tm][tn]);
                if (bias != nullptr) {
                    val += static_cast<float>(bias[out_col]);
                }
                C[out_row * N + out_col] = float_to_output_rn<OutDtype>(val);
            }
        }
    }
}

// ============================================================================
// Tiled MMA-Emu FP8 C-decoupled CoFDA GEMM Kernel
// ============================================================================

/**
 * @brief Tiled MMA-Emu FP8 C-decoupled CoFDA GEMM kernel.
 *
 * Byte-for-byte the C-fused CoFDA kernel above — same tiling, same
 * decode-on-load shared memory, same multiply — except for how the running
 * accumulator c is integrated:
 *   Step 1: sum the chunk's products among themselves at F bits (c takes no
 *           part in the exponent alignment)
 *   Step 2: merge that partial sum into c at F2 bits
 *
 * Small products therefore keep their precision in Step 1 instead of being
 * flushed by a large c during alignment. Isolating this one difference is what
 * makes the C-fused / C-decoupled comparison meaningful.
 *
 * @tparam F Fractional bits F
 * @tparam CHUNK_SIZE Products per CoFDA chunk (CS)
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 */
template<int F, int CHUNK_SIZE, typename OutDtype>
__launch_bounds__(FP8EmuConfig::NUM_THREADS)
__global__ void mma_emu_scaled_fp8_mm_cofda_decoupled_kernel(
    const __nv_fp8_storage_t* __restrict__ A,
    const __nv_fp8_storage_t* __restrict__ B,
    OutDtype* __restrict__ C,
    const float* __restrict__ scale_a_ptr,
    const float* __restrict__ scale_b_ptr,
    const OutDtype* __restrict__ bias,
    const int M, const int N, const int K)
{
    // F2=23: accumulator merge precision (FP32 mantissa width, effectively lossless)
    constexpr int F2 = 23;

    constexpr int BM = FP8EmuConfig::BM;
    constexpr int BN = FP8EmuConfig::BN;
    constexpr int BK = FP8EmuConfig::BK;
    constexpr int NUM_THREADS = FP8EmuConfig::NUM_THREADS;
    constexpr int THREAD_TILE_M = FP8EmuConfig::THREAD_TILE_M;
    constexpr int THREAD_TILE_N = FP8EmuConfig::THREAD_TILE_N;
    constexpr int BKP = BK + 1;  // padded row stride: removes stride-32 SMEM
                                 // bank conflicts.
    constexpr int A_TILE_SIZE = BM * BK;
    constexpr int B_TILE_SIZE = BN * BK;
    constexpr int A_ELEMS_PER_THREAD = (A_TILE_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int B_ELEMS_PER_THREAD = (B_TILE_SIZE + NUM_THREADS - 1) / NUM_THREADS;

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int tid = threadIdx.x;

    const int block_m = block_row * BM;
    const int block_n = block_col * BN;

    if (block_m >= M || block_n >= N) {
        return;
    }

    // Decode-on-load packed tiles, identical to the C-fused CoFDA kernel: the
    // two modes must differ only in how the accumulator is integrated.
    __shared__ uint32_t As_dec[BM * BKP];
    __shared__ uint32_t Bs_dec[BN * BKP];

    int thread_m, thread_n;
    get_thread_tile_position<FP8EmuConfig>(tid, thread_m, thread_n);

    float accum[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            accum[i][j] = 0.0f;
        }
    }

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Load A tile [BM x BK], decode-on-load, zero-pad OOB
        #pragma unroll
        for (int i = 0; i < A_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < A_TILE_SIZE) {
                int a_row = idx / BK;
                int a_col = idx % BK;
                int global_row = block_m + a_row;
                int global_col = k_tile + a_col;
                const __nv_fp8_storage_t raw =
                    (global_row < M && global_col < K)
                        ? A[global_row * K + global_col]
                        : __nv_fp8_storage_t(0);
                const DecodedOperand d = decode_operand(static_cast<uint8_t>(raw));
                As_dec[a_row * BKP + a_col] = pack_decoded(d);
            }
        }

        // Load B tile [BK x BN] transposed, decode-on-load, zero-pad OOB
        #pragma unroll
        for (int i = 0; i < B_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < B_TILE_SIZE) {
                int b_row = idx / BN;  // K dimension
                int b_col = idx % BN;  // N dimension
                int global_row = k_tile + b_row;
                int global_col = block_n + b_col;
                const __nv_fp8_storage_t raw =
                    (global_row < K && global_col < N)
                        ? B[global_row + global_col * K]
                        : __nv_fp8_storage_t(0);
                const DecodedOperand d = decode_operand(static_cast<uint8_t>(raw));
                Bs_dec[b_col * BKP + b_row] = pack_decoded(d);
            }
        }
        __syncthreads();

        const int actual_bk = min(BK, K - k_tile);

        #pragma unroll
        for (int tm = 0; tm < THREAD_TILE_M; tm++) {
            #pragma unroll
            for (int tn = 0; tn < THREAD_TILE_N; tn++) {
                const int a_off = (thread_m + tm) * BKP;
                const int b_off = (thread_n + tn) * BKP;

                for (int k_start = 0; k_start < actual_bk; k_start += CHUNK_SIZE) {
                    const DecodedFrag a_frag{&As_dec[a_off + k_start]};
                    const DecodedFrag b_frag{&Bs_dec[b_off + k_start]};

                    // SMEM zero-pads OOB, so tail elements decode to zero
                    // operands and yield zero products — no bounds check needed.
                    Product products[CHUNK_SIZE];
                    #pragma unroll
                    for (int i = 0; i < CHUNK_SIZE; i++) {
                        products[i] = fp8_multiply_predecoded<F>(a_frag[i], b_frag[i]);
                    }

                    // Step 1 sums the products at F bits; Step 2 merges them
                    // into the accumulator at F2 bits.
                    accum[tm][tn] = cofda_decoupled_accumulate_products<F, F2, CHUNK_SIZE>(
                        products, accum[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // Apply scales and write output
    const float scale_a = *scale_a_ptr;
    const float scale_b = *scale_b_ptr;

    #pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn++) {
            int out_row = block_m + thread_m + tm;
            int out_col = block_n + thread_n + tn;
            if (out_row < M && out_col < N) {
                float val = scale_a * (scale_b * accum[tm][tn]);
                if (bias != nullptr) {
                    val += static_cast<float>(bias[out_col]);
                }
                C[out_row * N + out_col] = float_to_output_rn<OutDtype>(val);
            }
        }
    }
}

// ============================================================================
// Tiled MMA-Emu FP8 GDFS GEMM Kernel
// ============================================================================

/**
 * @brief Tiled MMA-Emu FP8 GDFS GEMM kernel.
 *
 * Two-level accumulation: group_accumulate<G, GS> → group_to_operand_fp8<F, G>
 * → gdfs_accumulate_tile<F>. No per-block scale factors (FP8 uses per-tensor
 * scales in the epilogue).
 *
 * Same tiling, shared memory, and epilogue as the CoFDA kernel.
 *
 * @tparam F Inter-group fractional bits F
 * @tparam G GDFS intra-group bits G
 * @tparam GS GDFS group size GS
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 */
template<int F, int G, int GS, typename OutDtype>
__launch_bounds__(FP8EmuConfig::NUM_THREADS)
__global__ void mma_emu_scaled_fp8_mm_gdfs_kernel(
    const __nv_fp8_storage_t* __restrict__ A,  // [M, K] row-major
    const __nv_fp8_storage_t* __restrict__ B,  // [K, N] column-major
    OutDtype* __restrict__ C,                   // [M, N] row-major
    const float* __restrict__ scale_a_ptr,
    const float* __restrict__ scale_b_ptr,
    const OutDtype* __restrict__ bias,
    const int M, const int N, const int K)
{
    constexpr int BM = FP8EmuConfig::BM;
    constexpr int BN = FP8EmuConfig::BN;
    constexpr int BK = FP8EmuConfig::BK;
    constexpr int NUM_THREADS = FP8EmuConfig::NUM_THREADS;
    constexpr int THREAD_TILE_M = FP8EmuConfig::THREAD_TILE_M;
    constexpr int THREAD_TILE_N = FP8EmuConfig::THREAD_TILE_N;
    constexpr int GROUPS_PER_TILE = BK / GS;
    constexpr int BKP = BK + 1;  // padded row stride for decoded tiles: removes
                                 // stride bank conflicts.
    constexpr int A_TILE = BM * BK;   // 1 byte/elem -> 1 word/elem (no nibble pack)
    constexpr int B_TILE = BN * BK;
    constexpr int A_ELEMS_PER_THREAD = (A_TILE + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int B_ELEMS_PER_THREAD = (B_TILE + NUM_THREADS - 1) / NUM_THREADS;

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int tid = threadIdx.x;

    const int block_m = block_row * BM;
    const int block_n = block_col * BN;

    if (block_m >= M || block_n >= N) {
        return;
    }

    __shared__ uint32_t As_dec[BM * BKP];  // decode-on-load: one word per FP8 elem
    __shared__ uint32_t Bs_dec[BN * BKP];

    int thread_m, thread_n;
    get_thread_tile_position<FP8EmuConfig>(tid, thread_m, thread_n);

    float accum[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            accum[i][j] = 0.0f;
        }
    }

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // ----------------------------------------
        // Load A tile [BM, BK]: decode-on-load, zero-pad
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < A_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < A_TILE) {
                int a_row = idx / BK;
                int a_col = idx % BK;
                int global_row = block_m + a_row;
                int global_col = k_tile + a_col;
                uint8_t byte = (global_row < M && global_col < K)
                                   ? static_cast<uint8_t>(A[global_row * K + global_col])
                                   : uint8_t(0);
                As_dec[a_row * BKP + a_col] = pack_decoded(decode_operand(byte));
            }
        }

        // ----------------------------------------
        // Load B tile [BN, BK] (transposed): decode-on-load, zero-pad
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < B_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < B_TILE) {
                int b_row = idx / BN;
                int b_col = idx % BN;
                int global_row = k_tile + b_row;
                int global_col = block_n + b_col;
                uint8_t byte = (global_row < K && global_col < N)
                                   ? static_cast<uint8_t>(B[global_row + global_col * K])
                                   : uint8_t(0);
                Bs_dec[b_col * BKP + b_row] = pack_decoded(decode_operand(byte));
            }
        }
        __syncthreads();

        // ----------------------------------------
        // GDFS compute (zero-padded SMEM -> no per-element bounds check)
        // ----------------------------------------
        #pragma unroll
        for (int tm = 0; tm < THREAD_TILE_M; tm++) {
            #pragma unroll
            for (int tn = 0; tn < THREAD_TILE_N; tn++) {
                Operand tile_groups[GROUPS_PER_TILE];
                #pragma unroll
                for (int g = 0; g < GROUPS_PER_TILE; g++) {
                    const DecodedFrag a_frag{&As_dec[(thread_m + tm) * BKP + g * GS]};
                    const DecodedFrag b_frag{&Bs_dec[(thread_n + tn) * BKP + g * GS]};
                    GroupResult gr =
                        fp8_gdfs_group_accumulate_predecoded<G, GS>(a_frag, b_frag);
                    if (gr.all_zero) {
                        tile_groups[g] = make_zero_operand();
                    } else {
                        tile_groups[g] = group_to_operand_fp8<F, G>(
                            gr.mantissa_sum, gr.max_exp);
                    }
                }
                accum[tm][tn] = gdfs_accumulate_tile<F, GROUPS_PER_TILE>(
                    tile_groups, accum[tm][tn]);
            }
        }

        __syncthreads();
    }

    // ----------------------------------------
    // Apply per-tensor scales and write output
    // ----------------------------------------
    const float scale_a = *scale_a_ptr;
    const float scale_b = *scale_b_ptr;

    #pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn++) {
            int out_row = block_m + thread_m + tm;
            int out_col = block_n + thread_n + tn;

            if (out_row < M && out_col < N) {
                float val = scale_a * (scale_b * accum[tm][tn]);
                if (bias != nullptr) {
                    val += static_cast<float>(bias[out_col]);
                }
                C[out_row * N + out_col] = float_to_output_rn<OutDtype>(val);
            }
        }
    }
}

// ============================================================================
// CoFDA Dispatch Macros
// ============================================================================

#define DISPATCH_FP8_COFDA_KERNEL(F_VAL, CHUNK, DTYPE, ...) \
    mma_emu_scaled_fp8_mm_cofda_kernel<F_VAL, CHUNK, DTYPE><<<grid, block>>>(__VA_ARGS__)

#define DISPATCH_BY_F_BITS_FP8_COFDA(CHUNK, DTYPE, ...) \
    switch (f_bits) { \
        case 3:  DISPATCH_FP8_COFDA_KERNEL(3,  CHUNK, DTYPE, __VA_ARGS__); break; \
        case 5:  DISPATCH_FP8_COFDA_KERNEL(5,  CHUNK, DTYPE, __VA_ARGS__); break; \
        case 7:  DISPATCH_FP8_COFDA_KERNEL(7,  CHUNK, DTYPE, __VA_ARGS__); break; \
        case 9:  DISPATCH_FP8_COFDA_KERNEL(9,  CHUNK, DTYPE, __VA_ARGS__); break; \
        case 10: DISPATCH_FP8_COFDA_KERNEL(10, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 11: DISPATCH_FP8_COFDA_KERNEL(11, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 12: DISPATCH_FP8_COFDA_KERNEL(12, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 13: DISPATCH_FP8_COFDA_KERNEL(13, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 17: DISPATCH_FP8_COFDA_KERNEL(17, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 21: DISPATCH_FP8_COFDA_KERNEL(21, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 25: DISPATCH_FP8_COFDA_KERNEL(25, CHUNK, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "CoFDA: f_bits must be one of {3, 5, 7, 9, 10, 11, 12, 13, 17, 21, 25}, got ", f_bits); \
    }

#define DISPATCH_BY_CHUNK_SIZE_FP8(DTYPE, ...) \
    switch (chunk_size) { \
        case 16: DISPATCH_BY_F_BITS_FP8_COFDA(16, DTYPE, __VA_ARGS__); break; \
        case 32: DISPATCH_BY_F_BITS_FP8_COFDA(32, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "CoFDA: chunk_size must be 16 or 32, got ", chunk_size); \
    }

// ============================================================================
// C-decoupled CoFDA Dispatch Macros
// ============================================================================

#define DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(F_VAL, CHUNK, DTYPE, ...) \
    mma_emu_scaled_fp8_mm_cofda_decoupled_kernel<F_VAL, CHUNK, DTYPE><<<grid, block>>>(__VA_ARGS__)

#define DISPATCH_BY_F_BITS_FP8_COFDA_DECOUPLED(CHUNK, DTYPE, ...) \
    switch (f_bits) { \
        case 3:  DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(3,  CHUNK, DTYPE, __VA_ARGS__); break; \
        case 5:  DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(5,  CHUNK, DTYPE, __VA_ARGS__); break; \
        case 7:  DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(7,  CHUNK, DTYPE, __VA_ARGS__); break; \
        case 9:  DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(9,  CHUNK, DTYPE, __VA_ARGS__); break; \
        case 10: DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(10, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 11: DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(11, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 12: DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(12, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 13: DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(13, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 17: DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(17, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 21: DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(21, CHUNK, DTYPE, __VA_ARGS__); break; \
        case 25: DISPATCH_FP8_COFDA_DECOUPLED_KERNEL(25, CHUNK, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "CoFDA (C-decoupled): f_bits must be one of {3, 5, 7, 9, 10, 11, 12, 13, 17, 21, 25}, got ", f_bits); \
    }

#define DISPATCH_BY_CHUNK_SIZE_FP8_COFDA_DECOUPLED(DTYPE, ...) \
    switch (chunk_size) { \
        case 16: DISPATCH_BY_F_BITS_FP8_COFDA_DECOUPLED(16, DTYPE, __VA_ARGS__); break; \
        case 32: DISPATCH_BY_F_BITS_FP8_COFDA_DECOUPLED(32, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "CoFDA (C-decoupled): chunk_size must be 16 or 32, got ", chunk_size); \
    }

// ============================================================================
// GDFS Dispatch Macros
// ============================================================================

#define DISPATCH_FP8_GDFS_KERNEL(F_VAL, G_VAL, GS_VAL, DTYPE, ...) \
    mma_emu_scaled_fp8_mm_gdfs_kernel<F_VAL, G_VAL, GS_VAL, DTYPE><<<grid, block>>>(__VA_ARGS__)

#define DISPATCH_BY_G_BITS_FP8(F_VAL, GS_VAL, DTYPE, ...) \
    switch (g_bits) { \
        case 3:  DISPATCH_FP8_GDFS_KERNEL(F_VAL, 3,  GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 4:  DISPATCH_FP8_GDFS_KERNEL(F_VAL, 4,  GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 5:  DISPATCH_FP8_GDFS_KERNEL(F_VAL, 5,  GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 6:  DISPATCH_FP8_GDFS_KERNEL(F_VAL, 6,  GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 8:  DISPATCH_FP8_GDFS_KERNEL(F_VAL, 8,  GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 13: DISPATCH_FP8_GDFS_KERNEL(F_VAL, 13, GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 32: DISPATCH_FP8_GDFS_KERNEL(F_VAL, 32, GS_VAL, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "GDFS: g_bits must be one of {3, 4, 5, 6, 8, 13, 32}, got ", g_bits); \
    }

#define DISPATCH_BY_F_BITS_FP8_GDFS(GS_VAL, DTYPE, ...) \
    switch (f_bits) { \
        case 7:  DISPATCH_BY_G_BITS_FP8(7,  GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 9:  DISPATCH_BY_G_BITS_FP8(9,  GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 10: DISPATCH_BY_G_BITS_FP8(10, GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 11: DISPATCH_BY_G_BITS_FP8(11, GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 13: DISPATCH_BY_G_BITS_FP8(13, GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 15: DISPATCH_BY_G_BITS_FP8(15, GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 25: DISPATCH_BY_G_BITS_FP8(25, GS_VAL, DTYPE, __VA_ARGS__); break; \
        case 35: DISPATCH_BY_G_BITS_FP8(35, GS_VAL, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "GDFS: f_bits must be one of {7, 9, 10, 11, 13, 15, 25, 35}, got ", f_bits); \
    }

#define DISPATCH_BY_GROUP_SIZE_FP8(DTYPE, ...) \
    switch (group_size) { \
        case 8:  DISPATCH_BY_F_BITS_FP8_GDFS(8,  DTYPE, __VA_ARGS__); break; \
        case 16: DISPATCH_BY_F_BITS_FP8_GDFS(16, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "group_size must be 8 or 16, got ", group_size); \
    }

// ============================================================================
// Host Wrapper Function
// ============================================================================

inline void mma_emu_scaled_fp8_mm_emu(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    std::optional<torch::Tensor> const& bias,
    int algorithm,
    int f_bits,
    int g_bits,
    int group_size,
    int chunk_size)
{
    using Config = FP8EmuConfig;

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    // 2D grid for tiled kernel: each block computes BM x BN output tile
    dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    dim3 block(Config::NUM_THREADS);

    // Get raw pointers
    const __nv_fp8_storage_t* a_ptr = reinterpret_cast<const __nv_fp8_storage_t*>(a.data_ptr());
    const __nv_fp8_storage_t* b_ptr = reinterpret_cast<const __nv_fp8_storage_t*>(b.data_ptr());
    const float* scale_a_ptr = static_cast<const float*>(a_scales.data_ptr());
    const float* scale_b_ptr = static_cast<const float*>(b_scales.data_ptr());

    if (algorithm == design_space::kGDFS) {
        // GDFS path
        if (out.dtype() == torch::kBFloat16) {
            __nv_bfloat16* c_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
            const __nv_bfloat16* bias_ptr = bias.has_value()
                ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr())
                : nullptr;
            DISPATCH_BY_GROUP_SIZE_FP8(__nv_bfloat16,
                a_ptr, b_ptr, c_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, M, N, K);
        } else if (out.dtype() == torch::kFloat16) {
            __half* c_ptr = reinterpret_cast<__half*>(out.data_ptr());
            const __half* bias_ptr = bias.has_value()
                ? reinterpret_cast<const __half*>(bias->data_ptr())
                : nullptr;
            DISPATCH_BY_GROUP_SIZE_FP8(__half,
                a_ptr, b_ptr, c_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, M, N, K);
        } else {
            TORCH_CHECK(false, "MMA-Emu FP8 GDFS emulation: unsupported output dtype. "
                        "Expected BFloat16 or Float16, got ", out.dtype());
        }
    } else if (algorithm == design_space::kCoFDA) {
        // CoFDA path
        if (out.dtype() == torch::kBFloat16) {
            __nv_bfloat16* c_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
            const __nv_bfloat16* bias_ptr = bias.has_value()
                ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr())
                : nullptr;
                DISPATCH_BY_CHUNK_SIZE_FP8(__nv_bfloat16,
                    a_ptr, b_ptr, c_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, M, N, K);
        } else if (out.dtype() == torch::kFloat16) {
            __half* c_ptr = reinterpret_cast<__half*>(out.data_ptr());
            const __half* bias_ptr = bias.has_value()
                ? reinterpret_cast<const __half*>(bias->data_ptr())
                : nullptr;
                DISPATCH_BY_CHUNK_SIZE_FP8(__half,
                    a_ptr, b_ptr, c_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, M, N, K);
        } else {
            TORCH_CHECK(false, "MMA-Emu FP8 CoFDA emulation: unsupported output dtype. "
                        "Expected BFloat16 or Float16, got ", out.dtype());
        }
    } else if (algorithm == design_space::kCoFDADecoupled) {
        // C-decoupled CoFDA path (two-step CoFDA)
        if (out.dtype() == torch::kBFloat16) {
            __nv_bfloat16* c_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
            const __nv_bfloat16* bias_ptr = bias.has_value()
                ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr())
                : nullptr;
            DISPATCH_BY_CHUNK_SIZE_FP8_COFDA_DECOUPLED(__nv_bfloat16,
                a_ptr, b_ptr, c_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, M, N, K);
        } else if (out.dtype() == torch::kFloat16) {
            __half* c_ptr = reinterpret_cast<__half*>(out.data_ptr());
            const __half* bias_ptr = bias.has_value()
                ? reinterpret_cast<const __half*>(bias->data_ptr())
                : nullptr;
            DISPATCH_BY_CHUNK_SIZE_FP8_COFDA_DECOUPLED(__half,
                a_ptr, b_ptr, c_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, M, N, K);
        } else {
            TORCH_CHECK(false, "MMA-Emu FP8 C-decoupled CoFDA emulation: unsupported output dtype. "
                        "Expected BFloat16 or Float16, got ", out.dtype());
        }
    } else {
        TORCH_CHECK(false, "MMA-Emu FP8 emulation: algorithm must be 1 (GDFS), 2 (CoFDA, C-fused), or 3 (CoFDA, C-decoupled), got ", algorithm);
    }
}

#undef DISPATCH_FP8_COFDA_KERNEL
#undef DISPATCH_BY_F_BITS_FP8_COFDA
#undef DISPATCH_BY_CHUNK_SIZE_FP8
#undef DISPATCH_FP8_COFDA_DECOUPLED_KERNEL
#undef DISPATCH_BY_F_BITS_FP8_COFDA_DECOUPLED
#undef DISPATCH_BY_CHUNK_SIZE_FP8_COFDA_DECOUPLED
#undef DISPATCH_FP8_GDFS_KERNEL
#undef DISPATCH_BY_G_BITS_FP8
#undef DISPATCH_BY_F_BITS_FP8_GDFS
#undef DISPATCH_BY_GROUP_SIZE_FP8

}  // namespace mma_emu
}  // namespace vllm
