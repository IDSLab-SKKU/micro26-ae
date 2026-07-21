/*
 * MMA Arithmetic Design Space
 *
 * Single source of truth for the parameter values the kernels accept: the
 * fractional bitwidths, group sizes and chunk sizes that the accompanying
 * evaluation sweeps. The kernel dispatch instantiates one template per
 * (F, G, CS, GS, output dtype) combination, so bounding these sets bounds the
 * build.
 *
 * The accumulation configurations of the commercially deployed architectures
 * all fall inside these sets: Ada (CoFDA, CS=16, F=13), Hopper (FDA, CS=32,
 * F=13), Blackwell FP8 (FDA, CS=32, F=25), and Blackwell MXFP4/NVFP4 (GDFS,
 * G=6, F=35, GS=16).
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace vllm {
namespace mma_emu {
namespace design_space {

// Accumulation algorithm, as passed to the torch operators.
enum Algorithm : int64_t {
  kGDFS = 1,            // Group-Dot-Fused-Sum
  kCoFDA = 2,           // Chain-of-FDA, C-fused accumulator
  kCoFDADecoupled = 3,  // Chain-of-FDA, C-decoupled accumulator
};

// Fractional bits F for CoFDA and C-decoupled CoFDA.
inline constexpr int64_t COFDA_F[] = {3, 5, 7, 9, 10, 11, 12, 13, 17, 21, 25};

// Inter-group fractional bits F for GDFS.
inline constexpr int64_t GDFS_F[] = {7, 9, 10, 11, 13, 15, 25, 35};

// Chunk size CS for CoFDA. NVFP4 is fixed at CS = 16 in the kernel.
inline constexpr int64_t FP8_CS[] = {16, 32};
inline constexpr int64_t MXFP4_CS[] = {16, 32};

// Group size GS for GDFS. NVFP4 is fixed at GS = 16 in the kernel.
inline constexpr int64_t FP8_GS[] = {8, 16};
inline constexpr int64_t MXFP4_GS[] = {16, 32};

// Intra-group fractional bits G for GDFS. G = 32 is the theoretical lossless
// width for FP8 products and G = 6 for E2M1 products.
inline constexpr int64_t FP8_G[] = {3, 4, 5, 6, 8, 13, 32};
inline constexpr int64_t FP4_G[] = {3, 4, 5, 6};

template <size_t N>
constexpr bool contains(const int64_t (&set)[N], int64_t value) {
  for (size_t i = 0; i < N; ++i) {
    if (set[i] == value) return true;
  }
  return false;
}

}  // namespace design_space
}  // namespace mma_emu
}  // namespace vllm
