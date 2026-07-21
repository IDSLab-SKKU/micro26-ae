/*
 * GDFS Group Accumulation (STP3)
 *
 * Format-agnostic group accumulation for the GDFS algorithm.
 * Sums GS unnormalized products into a fixed-point group result carrying G
 * fractional bits. Shared by the FP8, NVFP4 and MXFP4 emulation kernels.
 */

#pragma once

#include <cstdint>
#include "types.cuh"
#include "fp32_utils.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// GDFS Group Accumulation with G-bits (GDFS STP3)
// ============================================================================

/**
 * @brief Result of group-level FDA-style accumulation.
 *
 * Represents σ_g = mantissa_sum × 2^(max_exp - G), the output of GDFS STP3.
 */
struct GroupResult {
    int64_t mantissa_sum;   ///< Signed fixed-point sum with G fractional bits
    int     max_exp;        ///< Maximum exponent used for alignment
    bool    all_zero;       ///< True if all products in the group were zero
};

/**
 * @brief Accumulate N products using FDA-style two-pass algorithm with G-bits.
 *
 * This implements GDFS STP3: group dot-product summation with configurable
 * in core/accumulator.cuh, but operates on Product structures without a
 * running accumulator c.
 *
 * Pass 1: Find max exponent among all non-zero products.
 * Pass 2: Align each product to max_exp (right-shift by exp_diff),
 *          truncating shifted-out bits (RZ), then sum.
 *
 * G controls the precision: larger G preserves more bits during alignment.
 * G >= 6 is lossless for E2M1 products (max_exp_diff=4, radix=2).
 *
 * @tparam G GDFS intra-group bits G
 * @tparam N Products per group (GS)
 * @param products Array of N products from fp4_multiply_unnormalized<G>
 * @return GroupResult with (mantissa_sum, max_exp, all_zero)
 */
template<int G, int N>
[[nodiscard]] __device__ __forceinline__
GroupResult group_accumulate(const Product* products) {
    GroupResult gr;
    gr.mantissa_sum = 0;
    gr.max_exp = -9999;
    gr.all_zero = true;

    // Pass 1: Find maximum exponent among non-zero products
    #pragma unroll
    for (int i = 0; i < N; i++) {
        if (!products[i].is_zero) {
            gr.all_zero = false;
            if (products[i].exponent > gr.max_exp) {
                gr.max_exp = products[i].exponent;
            }
        }
    }

    if (gr.all_zero) {
        gr.max_exp = 0;
        return gr;
    }

    // Pass 2: Align to max_exp and accumulate (RZ truncation)
    #pragma unroll
    for (int i = 0; i < N; i++) {
        if (products[i].is_zero) {
            continue;
        }
        int exp_diff = gr.max_exp - products[i].exponent;
        int64_t aligned = (exp_diff >= 64)
                        ? 0
                        : static_cast<int64_t>(products[i].significand) >> exp_diff;
        gr.mantissa_sum += products[i].sign * aligned;
    }

    return gr;
}
}  // namespace mma_emu
}  // namespace vllm
