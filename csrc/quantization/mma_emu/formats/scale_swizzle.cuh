/*
 * Swizzled Block-Scale Factor Layout
 *
 * Reads block scale factors from the swizzled layout used by both NVFP4
 * (UE4M3) and MXFP4 (E8M0) tensors. Layout only - no arithmetic.
 */

#pragma once

#include <cstdint>

namespace vllm {
namespace mma_emu {

// ============================================================================
// Swizzled Scale Factor Reading
// ============================================================================

namespace swizzle {

/// Swizzle parameters for vLLM's swizzle_blockscale layout
constexpr int M_TILE_SIZE = 128;      ///< 32 * 4 = 128 rows per m_tile
constexpr int K_BLOCKS_PER_TILE = 4;  ///< 4 k_blocks per k_tile

/**
 * @brief Read scale factor from swizzled layout.
 *
 * vLLM's swizzle_blockscale transforms linear scales for Tensor Core access:
 *
 * Forward swizzle (swizzle_blockscale in quant_utils.py):
 *   1. Linear input: [M, K] where K = num_k_blocks
 *   2. Reshape to: [M/128, 4, 32, K/4, 4]
 *      - m_tile = m / 128
 *      - group_of_4 = (m % 128) / 32   (which group of 32 within 128)
 *      - row_in_32 = (m % 128) % 32    (row within 32-row block)
 *      - k_tile = k_block / 4
 *      - k_in_tile = k_block % 4
 *   3. Permute by (0, 3, 2, 1, 4): [m_tile, k_tile, row_in_32, group_of_4, k_in_tile]
 *   4. Swizzled memory: [num_m_tiles, k_tiles, 32, 4, 4]
 *
 * @tparam BLKSZ Elements covered by one block scale
 * @param sf_swizzled Pointer to swizzled scale factor tensor
 * @param m Row index (0 to M-1)
 * @param k_block K block index (0 to num_k_blocks-1)
 * @param M Total rows
 * @param K Total K dimension
 * @return Scale factor value (UE4M3 or E8M0 depending on format)
 */
template<int BLKSZ = fp4::BLOCK_SIZE>
[[nodiscard]] __device__ __forceinline__
uint8_t read_scale(const uint8_t* sf_swizzled, int m, int k_block, int M, int K) {
    // Compute k_tiles dimension
    const int num_k_blocks = (K + BLKSZ - 1) / BLKSZ;
    const int k_tiles = (num_k_blocks + K_BLOCKS_PER_TILE - 1) / K_BLOCKS_PER_TILE;

    // Decompose m into tile and intra-tile position
    const int m_tile = m / M_TILE_SIZE;
    const int m_in_tile = m % M_TILE_SIZE;
    const int group_of_4 = m_in_tile / 32;  // which of the 4 groups (0-3)
    const int row_in_32 = m_in_tile % 32;   // row within 32-row block (0-31)

    // Decompose k_block into tile and intra-tile position
    const int k_tile = k_block / K_BLOCKS_PER_TILE;
    const int k_in_tile = k_block % K_BLOCKS_PER_TILE;

    // Swizzled memory layout: [m_tile, k_tile, row_in_32, group_of_4, k_in_tile]
    // Linear index: k_in_tile + 4 * (group_of_4 + 4 * (row_in_32 + 32 * (k_tile + k_tiles * m_tile)))
    const int swizzled_idx = k_in_tile +
                             4 * (group_of_4 +
                             4 * (row_in_32 +
                             32 * (k_tile +
                             k_tiles * m_tile)));

    return sf_swizzled[swizzled_idx];
}

}  // namespace swizzle
}  // namespace mma_emu
}  // namespace vllm
