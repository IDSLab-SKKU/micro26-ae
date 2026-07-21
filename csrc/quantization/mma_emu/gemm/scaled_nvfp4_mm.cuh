/*
 * MMA-Emu NVFP4 GEMM — CUDA-core emulation kernels
 *
 * Implements NVFP4 (E2M1) GEMM with two accumulation algorithms:
 * 1. GDFS (Group-Dot-Fused-Sum): group-level accumulation with configurable
 *    G and F from core/design_space.cuh.
 * 2. CoFDA (Chunked FDA): product-level scale application with chunked
 *    accumulation, CS fixed at 16; F from core/design_space.cuh.
 *
 * Emulates the accumulation arithmetic of the Blackwell OMMA.SF (NVFP4)
 * instruction on CUDA cores. The native result is produced by
 * cutlass_scaled_fp4_mm.
 *
 * Key Features:
 * - Packed FP4 input (2 values per byte)
 * - UE4M3 block scale factors (block size = 16)
 * - Alpha epilogue for global scale application
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>

#include "../core/types.cuh"
#include "../core/fp32_utils.cuh"
#include "../core/tiling.cuh"
#include "../core/accumulator.cuh"
#include "../core/gdfs_group.cuh"
#include "../formats/fp4_e2m1.cuh"
#include "../formats/scale_swizzle.cuh"
#include "../formats/nvfp4_ue4m3.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// Tiling Configuration Alias
// ============================================================================

using NVFP4EmuConfig = TilingConfig<NVFP4EmulationTag>;

// ============================================================================
// Tiled MMA-Emu NVFP4 GDFS GEMM Kernel
// ============================================================================

/**
 * @brief Tiled MMA-Emu NVFP4 GDFS GEMM kernel with shared memory optimization.
 *
 * Block tiling (BM, BN, BK, thread count and per-thread tile) comes from
 * TilingConfig<NVFP4EmulationTag> in core/tiling.cuh. Shared memory
 * for efficient memory access patterns. Each thread computes a 2x1 output tile.
 *
 * Key optimization: Fused GDFS accumulation per K-tile eliminates the need for
 * large local arrays. Each K-tile's groups (BK / GS of them) are fused with
 * the running FP32 accumulator following the FDA pattern.
 *
 * Memory hierarchy:
 * - Global Memory: A[M, K/2], B[N, K/2], A_sf[swizzled], B_sf[swizzled]
 * - Shared Memory: As[BM, BK/2], Bs[BN, BK/2], A_sf_s[BM, 4], B_sf_s[BN, 4]
 * - Registers: accum[THREAD_TILE_M][THREAD_TILE_N], tile_groups[GROUPS_PER_TILE]
 *
 * @tparam F Inter-group fractional bits F
 * @tparam G GDFS intra-group bits G
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 */
template<int F, int G, int GS, typename OutDtype>
__launch_bounds__(NVFP4EmuConfig::NUM_THREADS)
__global__ void mma_emu_scaled_nvfp4_mm_emu_kernel(
    const uint8_t* __restrict__ A,      // [M, K/2] packed FP4
    const uint8_t* __restrict__ B,      // [N, K/2] packed FP4
    OutDtype* __restrict__ C,           // [M, N] output
    const uint8_t* __restrict__ A_sf,   // Swizzled block scales for A
    const uint8_t* __restrict__ B_sf,   // Swizzled block scales for B
    const float* __restrict__ alpha,    // [1] global scale
    const int M, const int N, const int K)
{
    // Import tiling constants
    constexpr int BM = NVFP4EmuConfig::BM;
    constexpr int BN = NVFP4EmuConfig::BN;
    constexpr int BK = NVFP4EmuConfig::BK;
    constexpr int BK_PACKED = NVFP4EmuConfig::BK_PACKED;
    constexpr int NUM_THREADS = NVFP4EmuConfig::NUM_THREADS;
    constexpr int THREAD_TILE_M = NVFP4EmuConfig::THREAD_TILE_M;
    constexpr int THREAD_TILE_N = NVFP4EmuConfig::THREAD_TILE_N;
    constexpr int BLOCK_SIZE = NVFP4EmuConfig::BLOCK_SIZE;
    constexpr int SF_PER_TILE = NVFP4EmuConfig::SF_PER_TILE;

    // Derived from template parameter GS
    constexpr int GROUPS_PER_TILE = BK / GS;
    constexpr int GROUPS_PER_SCALE = BLOCK_SIZE / GS;

    constexpr int BKP = BK + 1;  // padded row stride for decoded tiles: removes
                                 // stride bank conflicts.
    constexpr int A_TILE_PACKED = BM * BK_PACKED;  // packed-byte tile size
    constexpr int B_TILE_PACKED = BN * BK_PACKED;
    constexpr int A_ELEMS_PER_THREAD =
        (A_TILE_PACKED + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int B_ELEMS_PER_THREAD =
        (B_TILE_PACKED + NUM_THREADS - 1) / NUM_THREADS;

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

    // ========================================
    // Shared Memory Declaration
    // ========================================
    // Decode-on-load: one packed uint32 per FP4 elem, row stride BKP (= BK+1)
    // pads away stride bank conflicts. Scale tiles stay uint8 (de-swizzled).
    __shared__ uint32_t As_dec[BM * BKP];
    __shared__ uint32_t Bs_dec[BN * BKP];
    __shared__ uint8_t A_sf_s[NVFP4EmuConfig::SMEM_A_SF_SIZE];
    __shared__ uint8_t B_sf_s[NVFP4EmuConfig::SMEM_B_SF_SIZE];

    // Thread tile mapping within block
    int thread_m, thread_n;
    get_thread_tile_position<NVFP4EmuConfig>(tid, thread_m, thread_n);

    // ========================================
    // Register Accumulators (FP32)
    // ========================================
    float accum[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            accum[i][j] = 0.0f;
        }
    }

    // Precompute constants
    const int K_packed = K / 2;
    const int num_k_tiles = (K + BK - 1) / BK;

    // ========================================
    // OUTER LOOP: K-dimension tiles
    // ========================================
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_tile_start = kt * BK;
        const int k_tile_packed_start = k_tile_start / 2;
        const int k_block_start = kt * SF_PER_TILE;

        // ----------------------------------------
        // Load A tile [BM, BK]: unpack 2 nibbles/byte, decode-on-load, zero-pad
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < A_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < A_TILE_PACKED) {
                int a_row = idx / BK_PACKED;
                int a_col = idx % BK_PACKED;  // packed byte col [0, BK_PACKED)
                int global_row = block_m + a_row;
                int global_col = k_tile_packed_start + a_col;
                uint8_t byte = (global_row < M && global_col < K_packed)
                                   ? A[global_row * K_packed + global_col]
                                   : static_cast<uint8_t>(__nv_fp4_storage_t(0));
                // low nibble -> k=2*a_col (even), high nibble -> k=2*a_col+1 (odd)
                As_dec[a_row * BKP + 2 * a_col] =
                    pack_decoded_fp4(decode_fp4_operand(byte & 0x0F));
                As_dec[a_row * BKP + 2 * a_col + 1] =
                    pack_decoded_fp4(decode_fp4_operand(byte >> 4));
            }
        }

        // ----------------------------------------
        // Load B tile [BN, BK]: unpack 2 nibbles/byte, decode-on-load, zero-pad
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < B_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < B_TILE_PACKED) {
                int b_row = idx / BK_PACKED;
                int b_col = idx % BK_PACKED;
                int global_row = block_n + b_row;
                int global_col = k_tile_packed_start + b_col;
                uint8_t byte = (global_row < N && global_col < K_packed)
                                   ? B[global_row * K_packed + global_col]
                                   : static_cast<uint8_t>(__nv_fp4_storage_t(0));
                Bs_dec[b_row * BKP + 2 * b_col] =
                    pack_decoded_fp4(decode_fp4_operand(byte & 0x0F));
                Bs_dec[b_row * BKP + 2 * b_col + 1] =
                    pack_decoded_fp4(decode_fp4_operand(byte >> 4));
            }
        }

        // ----------------------------------------
        // Load A scale factors (de-swizzle during load)
        // ----------------------------------------
        {
            constexpr int A_sf_tile_size = NVFP4EmuConfig::SMEM_A_SF_SIZE;
            const int num_k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int idx = tid; idx < A_sf_tile_size; idx += NUM_THREADS) {
                int m_local = idx / SF_PER_TILE;
                int sf_local = idx % SF_PER_TILE;
                int m_global = block_m + m_local;
                int k_block_global = k_block_start + sf_local;

                if (m_global < M && k_block_global < num_k_blocks) {
                    A_sf_s[idx] = swizzle::read_scale(A_sf, m_global, k_block_global, M, K);
                } else {
                    A_sf_s[idx] = 0;
                }
            }
        }

        // ----------------------------------------
        // Load B scale factors (de-swizzle during load)
        // ----------------------------------------
        {
            constexpr int B_sf_tile_size = NVFP4EmuConfig::SMEM_B_SF_SIZE;
            const int num_k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int idx = tid; idx < B_sf_tile_size; idx += NUM_THREADS) {
                int n_local = idx / SF_PER_TILE;
                int sf_local = idx % SF_PER_TILE;
                int n_global = block_n + n_local;
                int k_block_global = k_block_start + sf_local;

                if (n_global < N && k_block_global < num_k_blocks) {
                    B_sf_s[idx] = swizzle::read_scale(B_sf, n_global, k_block_global, N, K);
                } else {
                    B_sf_s[idx] = 0;
                }
            }
        }

        __syncthreads();

        // ----------------------------------------
        // Compute: GDFS groups (zero-padded SMEM -> no per-element bounds check)
        // ----------------------------------------
        #pragma unroll
        for (int tm = 0; tm < THREAD_TILE_M; tm++) {
            #pragma unroll
            for (int tn = 0; tn < THREAD_TILE_N; tn++) {
                int local_m = thread_m + tm;
                int local_n = thread_n + tn;
                int out_row = block_m + local_m;
                int out_col = block_n + local_n;

                // Skip if out of bounds
                if (out_row >= M || out_col >= N) {
                    continue;
                }

                // GDFS STP2-4: streaming group accumulate (no Product[GS] array),
                // then UE4M3 scale -> Operand. Zero-padded SMEM makes OOB elems
                // zero products, matching the original actual_groups/actual_k path.
                Operand tile_groups[GROUPS_PER_TILE];

                #pragma unroll
                for (int g = 0; g < GROUPS_PER_TILE; g++) {
                    const DecodedFP4Frag a_frag{&As_dec[local_m * BKP + g * GS]};
                    const DecodedFP4Frag b_frag{&Bs_dec[local_n * BKP + g * GS]};
                    GroupResult gr =
                        fp4_gdfs_group_accumulate_predecoded<G, GS>(a_frag, b_frag);
                    if (gr.all_zero) {
                        tile_groups[g] = make_zero_operand();
                    } else {
                        int scale_idx = g / GROUPS_PER_SCALE;
                        uint8_t sfa = A_sf_s[local_m * SF_PER_TILE + scale_idx];
                        uint8_t sfb = B_sf_s[local_n * SF_PER_TILE + scale_idx];
                        tile_groups[g] = apply_ue4m3_scales<F, G>(
                            gr.mantissa_sum, gr.max_exp, sfa, sfb);
                    }
                }

                // STP5-7: Fused-sum accumulation over groups
                accum[tm][tn] = gdfs_accumulate_tile<F, GROUPS_PER_TILE>(
                    tile_groups, accum[tm][tn]);
            }
        }

        __syncthreads();
    }

    // ========================================
    // EPILOGUE: Apply alpha and write output
    // ========================================
    const float alpha_val = alpha[0];

    #pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn++) {
            int out_row = block_m + thread_m + tm;
            int out_col = block_n + thread_n + tn;

            if (out_row < M && out_col < N) {
                float scaled = accum[tm][tn] * alpha_val;
                C[out_row * N + out_col] = float_to_output_rn<OutDtype>(scaled);
            }
        }
    }
}

// ============================================================================
// MMA-Emu NVFP4 CoFDA GEMM Kernel
// ============================================================================

/**
 * @brief MMA-Emu NVFP4 CoFDA (Chunked FDA) GEMM kernel.
 *
 * Unlike GDFS, which groups products before applying scales, CoFDA applies
 * the UE4M3 block scale at the individual product level and then accumulates
 * in chunks of CS products.
 *
 * @tparam F Fractional bits F
 * @tparam CS CoFDA chunk size CS
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 */
template<int F, int CS, typename OutDtype>
__launch_bounds__(NVFP4EmuConfig::NUM_THREADS)
__global__ void mma_emu_scaled_nvfp4_mm_cofda_kernel(
    const uint8_t* __restrict__ A,      // [M, K/2] packed FP4
    const uint8_t* __restrict__ B,      // [N, K/2] packed FP4
    OutDtype* __restrict__ C,           // [M, N] output
    const uint8_t* __restrict__ A_sf,   // Swizzled block scales for A (UE4M3)
    const uint8_t* __restrict__ B_sf,   // Swizzled block scales for B (UE4M3)
    const float* __restrict__ alpha,    // [1] global scale
    const int M, const int N, const int K)
{
    // Import tiling constants
    constexpr int BM = NVFP4EmuConfig::BM;
    constexpr int BN = NVFP4EmuConfig::BN;
    constexpr int BK = NVFP4EmuConfig::BK;
    constexpr int BK_PACKED = NVFP4EmuConfig::BK_PACKED;
    constexpr int NUM_THREADS = NVFP4EmuConfig::NUM_THREADS;
    constexpr int THREAD_TILE_M = NVFP4EmuConfig::THREAD_TILE_M;
    constexpr int THREAD_TILE_N = NVFP4EmuConfig::THREAD_TILE_N;
    constexpr int BLOCK_SIZE = NVFP4EmuConfig::BLOCK_SIZE;
    constexpr int SF_PER_TILE = NVFP4EmuConfig::SF_PER_TILE;

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

    // ========================================
    // Shared Memory Declaration
    // ========================================
    __shared__ uint8_t As[NVFP4EmuConfig::SMEM_A_SIZE];
    __shared__ uint8_t Bs[NVFP4EmuConfig::SMEM_B_SIZE];
    __shared__ uint8_t A_sf_s[NVFP4EmuConfig::SMEM_A_SF_SIZE];
    __shared__ uint8_t B_sf_s[NVFP4EmuConfig::SMEM_B_SF_SIZE];

    // Thread tile mapping within block
    int thread_m, thread_n;
    get_thread_tile_position<NVFP4EmuConfig>(tid, thread_m, thread_n);

    // ========================================
    // Register Accumulators (FP32)
    // ========================================
    float accum[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            accum[i][j] = 0.0f;
        }
    }

    // Precompute constants
    const int K_packed = K / 2;
    const int num_k_tiles = (K + BK - 1) / BK;

    // ========================================
    // OUTER LOOP: K-dimension tiles
    // ========================================
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_tile_start = kt * BK;
        const int k_tile_packed_start = k_tile_start / 2;
        const int k_block_start = kt * SF_PER_TILE;

        // ----------------------------------------
        // Load A tile: [BM, BK/2] from global to shared
        // ----------------------------------------
        {
            constexpr int A_tile_size = NVFP4EmuConfig::SMEM_A_SIZE;
            constexpr int A_elems_per_thread = NVFP4EmuConfig::A_ELEMS_PER_THREAD;

            #pragma unroll
            for (int i = 0; i < A_elems_per_thread; i++) {
                int idx = tid + i * NUM_THREADS;
                if (idx < A_tile_size) {
                    int a_row = idx / BK_PACKED;
                    int a_col = idx % BK_PACKED;
                    int global_row = block_m + a_row;
                    int global_col = k_tile_packed_start + a_col;

                    if (global_row < M && global_col < K_packed) {
                        As[idx] = A[global_row * K_packed + global_col];
                    } else {
                        As[idx] = static_cast<uint8_t>(__nv_fp4_storage_t(0));
                    }
                }
            }
        }

        // ----------------------------------------
        // Load B tile: [BN, BK/2] from global to shared
        // ----------------------------------------
        {
            constexpr int B_tile_size = NVFP4EmuConfig::SMEM_B_SIZE;
            constexpr int B_elems_per_thread = NVFP4EmuConfig::B_ELEMS_PER_THREAD;

            #pragma unroll
            for (int i = 0; i < B_elems_per_thread; i++) {
                int idx = tid + i * NUM_THREADS;
                if (idx < B_tile_size) {
                    int b_row = idx / BK_PACKED;
                    int b_col = idx % BK_PACKED;
                    int global_row = block_n + b_row;
                    int global_col = k_tile_packed_start + b_col;

                    if (global_row < N && global_col < K_packed) {
                        Bs[idx] = B[global_row * K_packed + global_col];
                    } else {
                        Bs[idx] = static_cast<uint8_t>(__nv_fp4_storage_t(0));
                    }
                }
            }
        }

        // ----------------------------------------
        // Load A scale factors (de-swizzle during load)
        // ----------------------------------------
        {
            constexpr int A_sf_tile_size = NVFP4EmuConfig::SMEM_A_SF_SIZE;
            const int num_k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int idx = tid; idx < A_sf_tile_size; idx += NUM_THREADS) {
                int m_local = idx / SF_PER_TILE;
                int sf_local = idx % SF_PER_TILE;
                int m_global = block_m + m_local;
                int k_block_global = k_block_start + sf_local;

                if (m_global < M && k_block_global < num_k_blocks) {
                    A_sf_s[idx] = swizzle::read_scale(A_sf, m_global, k_block_global, M, K);
                } else {
                    A_sf_s[idx] = 0;
                }
            }
        }

        // ----------------------------------------
        // Load B scale factors (de-swizzle during load)
        // ----------------------------------------
        {
            constexpr int B_sf_tile_size = NVFP4EmuConfig::SMEM_B_SF_SIZE;
            const int num_k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int idx = tid; idx < B_sf_tile_size; idx += NUM_THREADS) {
                int n_local = idx / SF_PER_TILE;
                int sf_local = idx % SF_PER_TILE;
                int n_global = block_n + n_local;
                int k_block_global = k_block_start + sf_local;

                if (n_global < N && k_block_global < num_k_blocks) {
                    B_sf_s[idx] = swizzle::read_scale(B_sf, n_global, k_block_global, N, K);
                } else {
                    B_sf_s[idx] = 0;
                }
            }
        }

        __syncthreads();

        // ----------------------------------------
        // Compute: CoFDA chunked accumulation
        // ----------------------------------------
        const int actual_k = min(BK, K - k_tile_start);

        #pragma unroll
        for (int tm = 0; tm < THREAD_TILE_M; tm++) {
            #pragma unroll
            for (int tn = 0; tn < THREAD_TILE_N; tn++) {
                int local_m = thread_m + tm;
                int local_n = thread_n + tn;
                int out_row = block_m + local_m;
                int out_col = block_n + local_n;

                // Skip if out of bounds
                if (out_row >= M || out_col >= N) {
                    continue;
                }

                // ========================================
                // CoFDA: Chunk-based accumulation with product-level UE4M3 scales
                // ========================================
                for (int chunk_start = 0; chunk_start < actual_k; chunk_start += CS) {
                    Operand chunk_operands[CS];

                    #pragma unroll
                    for (int i = 0; i < CS; i++) {
                        int k_local = chunk_start + i;
                        if (k_local < actual_k) {
                            uint8_t a_val = load_fp4_from_tile(As, local_m, k_local, BK_PACKED);
                            uint8_t b_val = load_fp4_from_tile(Bs, local_n, k_local, BK_PACKED);

                            // Scale mapping: k_local / BLOCK_SIZE gives scale index
                            int scale_idx = k_local / BLOCK_SIZE;
                            uint8_t sfa = A_sf_s[local_m * SF_PER_TILE + scale_idx];
                            uint8_t sfb = B_sf_s[local_n * SF_PER_TILE + scale_idx];

                            chunk_operands[i] = fp4_product_with_ue4m3_scales<F>(
                                decompose_fp4_e2m1(a_val), decompose_fp4_e2m1(b_val),
                                sfa, sfb);
                        } else {
                            chunk_operands[i] = make_zero_operand();
                        }
                    }

                    accum[tm][tn] =
                        chunked_accumulate<F, CS>(chunk_operands, accum[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // ========================================
    // EPILOGUE: Apply alpha and write output
    // ========================================
    const float alpha_val = alpha[0];

    #pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn++) {
            int out_row = block_m + thread_m + tm;
            int out_col = block_n + thread_n + tn;

            if (out_row < M && out_col < N) {
                float scaled = accum[tm][tn] * alpha_val;
                C[out_row * N + out_col] = float_to_output_rn<OutDtype>(scaled);
            }
        }
    }
}

// ============================================================================
// Host Wrapper Function
// ============================================================================

// GDFS Kernel launch macro: instantiate kernel with compile-time F, G, GS=16, and output dtype
#define DISPATCH_NVFP4_KERNEL(F_VAL, G_VAL, DTYPE, ...) \
    mma_emu_scaled_nvfp4_mm_emu_kernel<F_VAL, G_VAL, 16, DTYPE><<<grid, block, 0, stream>>>(__VA_ARGS__)

// CoFDA Kernel launch macro: instantiate kernel with compile-time F, CS=16, and output dtype
#define DISPATCH_NVFP4_COFDA_KERNEL(F_VAL, DTYPE, ...) \
    mma_emu_scaled_nvfp4_mm_cofda_kernel<F_VAL, 16, DTYPE><<<grid, block, 0, stream>>>(__VA_ARGS__)

// Inner dispatch: resolve G at compile time for a given F value. GS is fixed
// at 16; G = 6 is the lossless intra-group width for E2M1 products.
#define DISPATCH_BY_G_BITS_NVFP4(F_VAL, DTYPE, ...) \
    switch (g_bits) { \
        case 3:  DISPATCH_NVFP4_KERNEL(F_VAL, 3, DTYPE, __VA_ARGS__); break; \
        case 4:  DISPATCH_NVFP4_KERNEL(F_VAL, 4, DTYPE, __VA_ARGS__); break; \
        case 5:  DISPATCH_NVFP4_KERNEL(F_VAL, 5, DTYPE, __VA_ARGS__); break; \
        case 6:  DISPATCH_NVFP4_KERNEL(F_VAL, 6, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "GDFS: g_bits must be one of {3, 4, 5, 6}, got ", g_bits); \
    }

// GDFS: resolve F at compile time, then delegate to G. GS is fixed at 16.
#define DISPATCH_BY_F_BITS_NVFP4(DTYPE, ...) \
    switch (f_bits) { \
        case 7:  DISPATCH_BY_G_BITS_NVFP4(7,  DTYPE, __VA_ARGS__); break; \
        case 9:  DISPATCH_BY_G_BITS_NVFP4(9,  DTYPE, __VA_ARGS__); break; \
        case 10: DISPATCH_BY_G_BITS_NVFP4(10, DTYPE, __VA_ARGS__); break; \
        case 11: DISPATCH_BY_G_BITS_NVFP4(11, DTYPE, __VA_ARGS__); break; \
        case 13: DISPATCH_BY_G_BITS_NVFP4(13, DTYPE, __VA_ARGS__); break; \
        case 15: DISPATCH_BY_G_BITS_NVFP4(15, DTYPE, __VA_ARGS__); break; \
        case 25: DISPATCH_BY_G_BITS_NVFP4(25, DTYPE, __VA_ARGS__); break; \
        case 35: DISPATCH_BY_G_BITS_NVFP4(35, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "GDFS: f_bits must be one of {7, 9, 10, 11, 13, 15, 25, 35}, got ", f_bits); \
    }

// CoFDA F-bits dispatch: resolve F at compile time (chunk_size fixed at 16).
#define DISPATCH_BY_F_BITS_NVFP4_COFDA(DTYPE, ...) \
    switch (f_bits) { \
        case 3:  DISPATCH_NVFP4_COFDA_KERNEL(3,  DTYPE, __VA_ARGS__); break; \
        case 5:  DISPATCH_NVFP4_COFDA_KERNEL(5,  DTYPE, __VA_ARGS__); break; \
        case 7:  DISPATCH_NVFP4_COFDA_KERNEL(7,  DTYPE, __VA_ARGS__); break; \
        case 9:  DISPATCH_NVFP4_COFDA_KERNEL(9,  DTYPE, __VA_ARGS__); break; \
        case 10: DISPATCH_NVFP4_COFDA_KERNEL(10, DTYPE, __VA_ARGS__); break; \
        case 11: DISPATCH_NVFP4_COFDA_KERNEL(11, DTYPE, __VA_ARGS__); break; \
        case 12: DISPATCH_NVFP4_COFDA_KERNEL(12, DTYPE, __VA_ARGS__); break; \
        case 13: DISPATCH_NVFP4_COFDA_KERNEL(13, DTYPE, __VA_ARGS__); break; \
        case 17: DISPATCH_NVFP4_COFDA_KERNEL(17, DTYPE, __VA_ARGS__); break; \
        case 21: DISPATCH_NVFP4_COFDA_KERNEL(21, DTYPE, __VA_ARGS__); break; \
        case 25: DISPATCH_NVFP4_COFDA_KERNEL(25, DTYPE, __VA_ARGS__); break; \
        default: TORCH_CHECK(false, "CoFDA: f_bits must be one of {3, 5, 7, 9, 10, 11, 12, 13, 17, 21, 25}, got ", f_bits); \
    }


inline void mma_emu_scaled_nvfp4_mm_emu(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& alpha,
    int algorithm,
    int f_bits,
    int g_bits)
{
    using Config = NVFP4EmuConfig;

    const int M = a.size(0);
    const int K_packed = a.size(1);
    const int K = K_packed * 2;
    const int N = b.size(0);

    dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    dim3 block(Config::NUM_THREADS);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    const uint8_t* a_ptr = static_cast<const uint8_t*>(a.data_ptr());
    const uint8_t* b_ptr = static_cast<const uint8_t*>(b.data_ptr());
    const uint8_t* a_sf_ptr = static_cast<const uint8_t*>(a_scales.data_ptr());
    const uint8_t* b_sf_ptr = static_cast<const uint8_t*>(b_scales.data_ptr());
    const float* alpha_ptr = static_cast<const float*>(alpha.data_ptr());

    if (algorithm == design_space::kCoFDA) {
        // CoFDA path: dispatch by f_bits (chunk_size fixed at 16)
        if (out.dtype() == torch::kBFloat16) {
            __nv_bfloat16* c_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

            DISPATCH_BY_F_BITS_NVFP4_COFDA(__nv_bfloat16,
                a_ptr, b_ptr, c_ptr, a_sf_ptr, b_sf_ptr, alpha_ptr,
                M, N, K);
        } else if (out.dtype() == torch::kFloat16) {
            __half* c_ptr = reinterpret_cast<__half*>(out.data_ptr());

            DISPATCH_BY_F_BITS_NVFP4_COFDA(__half,
                a_ptr, b_ptr, c_ptr, a_sf_ptr, b_sf_ptr, alpha_ptr,
                M, N, K);
        } else {
            TORCH_CHECK(false, "MMA-Emu scaled_nvfp4_mm CoFDA: unsupported output dtype. "
                        "Expected BFloat16 or Float16, got ", out.dtype());
        }
    } else {
        // GDFS path: dispatch by f_bits, g_bits (GS fixed at 16)
        if (out.dtype() == torch::kBFloat16) {
            __nv_bfloat16* c_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

            DISPATCH_BY_F_BITS_NVFP4(__nv_bfloat16,
                a_ptr, b_ptr, c_ptr, a_sf_ptr, b_sf_ptr, alpha_ptr,
                M, N, K);
        } else if (out.dtype() == torch::kFloat16) {
            __half* c_ptr = reinterpret_cast<__half*>(out.data_ptr());

            DISPATCH_BY_F_BITS_NVFP4(__half,
                a_ptr, b_ptr, c_ptr, a_sf_ptr, b_sf_ptr, alpha_ptr,
                M, N, K);
        } else {
            TORCH_CHECK(false, "MMA-Emu scaled_nvfp4_mm GDFS: unsupported output dtype. "
                        "Expected BFloat16 or Float16, got ", out.dtype());
        }
    }
}

#undef DISPATCH_NVFP4_KERNEL
#undef DISPATCH_NVFP4_COFDA_KERNEL
#undef DISPATCH_BY_G_BITS_NVFP4
#undef DISPATCH_BY_F_BITS_NVFP4
#undef DISPATCH_BY_F_BITS_NVFP4_COFDA

}  // namespace mma_emu
}  // namespace vllm
