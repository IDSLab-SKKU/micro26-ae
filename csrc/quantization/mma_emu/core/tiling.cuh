/*
 * MMA-Emu Tiling Configuration
 *
 * Tile shapes for the FP8, NVFP4 and MXFP4 emulation kernels, selected by tag.
 */

#pragma once

#include <cstdint>

namespace vllm {
namespace mma_emu {

// ============================================================================
// Tiling Policy Tags
// ============================================================================

struct FP8EmulationTag;
struct NVFP4EmulationTag;
struct MXFP4EmulationTag;

// ============================================================================
// Generic GEMM Tiling Configuration Template
// ============================================================================

/**
 * @brief Tiling configuration template.
 *
 * Primary template is not defined - must specialize for each kernel type.
 */
template<typename KernelTag>
struct TilingConfig;

// ============================================================================
// FP8 Emulation Tiling Configuration
// ============================================================================

template<>
struct TilingConfig<FP8EmulationTag> {
    // Block tile dimensions
    static constexpr int BM = 32;   ///< M dimension of block tile
    static constexpr int BN = 8;    ///< N dimension of block tile
    static constexpr int BK = 32;   ///< K dimension per iteration

    // Thread configuration
    static constexpr int NUM_THREADS = 128;
    static constexpr int WARP_SIZE = 32;

    // Thread tile dimensions
    static constexpr int THREAD_TILE_M = 2;
    static constexpr int THREAD_TILE_N = 1;
    static constexpr int THREADS_PER_ROW = BN / THREAD_TILE_N;  // 8

    // Shared memory sizes
    static constexpr int SMEM_A_SIZE = BM * BK;
    static constexpr int SMEM_B_SIZE = BN * BK;

    // Helper: elements per thread for tile loading
    static constexpr int A_ELEMS_PER_THREAD = (SMEM_A_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    static constexpr int B_ELEMS_PER_THREAD = (SMEM_B_SIZE + NUM_THREADS - 1) / NUM_THREADS;
};

// ============================================================================
// NVFP4 Emulation Tiling Configuration
// ============================================================================

template<>
struct TilingConfig<NVFP4EmulationTag> {
    // Block tile dimensions
    static constexpr int BM = 32;   ///< M dimension of block tile
    static constexpr int BN = 8;    ///< N dimension of block tile
    static constexpr int BK = 64;   ///< K dimension per iteration (must be multiple of BLOCK_SIZE)

    // Thread configuration
    static constexpr int NUM_THREADS = 128;
    static constexpr int WARP_SIZE = 32;

    // Thread tile dimensions
    static constexpr int THREAD_TILE_M = 2;
    static constexpr int THREAD_TILE_N = 1;
    static constexpr int THREADS_PER_ROW = BN / THREAD_TILE_N;  // 8

    // NVFP4 specific constants
    static constexpr int BLOCK_SIZE = 16;              ///< NVFP4 scale block size
    static constexpr int BK_PACKED = BK / 2;           ///< Packed FP4 bytes per K tile
    static constexpr int SF_PER_TILE = BK / BLOCK_SIZE;  // 4 scale factors per K tile

    // Shared memory sizes
    static constexpr int SMEM_A_SIZE = BM * BK_PACKED;
    static constexpr int SMEM_B_SIZE = BN * BK_PACKED;
    static constexpr int SMEM_A_SF_SIZE = BM * SF_PER_TILE;
    static constexpr int SMEM_B_SF_SIZE = BN * SF_PER_TILE;

    // Helper: elements per thread for tile loading
    static constexpr int A_ELEMS_PER_THREAD = (SMEM_A_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    static constexpr int B_ELEMS_PER_THREAD = (SMEM_B_SIZE + NUM_THREADS - 1) / NUM_THREADS;
};

// ============================================================================
// MXFP4 Emulation Tiling Configuration
// ============================================================================

template<>
struct TilingConfig<MXFP4EmulationTag> {
    // Block tile dimensions
    static constexpr int BM = 32;   ///< M dimension of block tile
    static constexpr int BN = 8;    ///< N dimension of block tile
    static constexpr int BK = 64;   ///< K dimension per iteration

    // Thread configuration
    static constexpr int NUM_THREADS = 128;
    static constexpr int WARP_SIZE = 32;

    // Thread tile dimensions
    static constexpr int THREAD_TILE_M = 2;
    static constexpr int THREAD_TILE_N = 1;
    static constexpr int THREADS_PER_ROW = BN / THREAD_TILE_N;  // 8

    // MXFP4 specific constants
    static constexpr int BLOCK_SIZE = 32;              ///< MXFP4 scale block size
    static constexpr int BK_PACKED = BK / 2;           ///< Packed FP4 bytes per K tile
    static constexpr int SF_PER_TILE = BK / BLOCK_SIZE;      // 2 scale pairs per K tile

    // Shared memory sizes
    static constexpr int SMEM_A_SIZE = BM * BK_PACKED;
    static constexpr int SMEM_B_SIZE = BN * BK_PACKED;
    static constexpr int SMEM_A_SF_SIZE = BM * SF_PER_TILE;   // BM * 2
    static constexpr int SMEM_B_SF_SIZE = BN * SF_PER_TILE;   // BN * 2

    // Helper: elements per thread for tile loading
    static constexpr int A_ELEMS_PER_THREAD = (SMEM_A_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    static constexpr int B_ELEMS_PER_THREAD = (SMEM_B_SIZE + NUM_THREADS - 1) / NUM_THREADS;
};

// ============================================================================
// Thread Index Calculation Helpers
// ============================================================================

/**
 * @brief Calculate thread tile position within a block.
 *
 * @tparam Config Tiling configuration type
 * @param tid Thread ID within block
 * @return pair of (thread_row, thread_col) in terms of thread tiles
 */
template<typename Config>
__device__ __forceinline__
void get_thread_tile_position(int tid, int& thread_m, int& thread_n) {
    int thread_row = tid / Config::THREADS_PER_ROW;
    int thread_col = tid % Config::THREADS_PER_ROW;
    thread_m = thread_row * Config::THREAD_TILE_M;
    thread_n = thread_col * Config::THREAD_TILE_N;
}

// ============================================================================
// Utility Macros
// ============================================================================

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#endif

}  // namespace mma_emu
}  // namespace vllm
