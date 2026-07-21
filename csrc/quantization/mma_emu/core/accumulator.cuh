/*
 * MMA-Emu Chunked Accumulator
 *
 * Two-pass fixed-point accumulation shared by the FDA, CoFDA and GDFS
 * kernels across all three formats.
 *
 * Algorithm:
 * - Pass 1: Scan for max exponent and check for NaN/Inf
 * - Pass 2: Align all operands to max exponent and sum
 * - Final: Normalize fixed-point sum back to FP32 with truncation
 */

#pragma once

#include <cstdint>
#include "types.cuh"
#include "fp32_utils.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// Two-Pass Accumulation Core
// ============================================================================

/**
 * @brief Fused two-pass accumulation for chunked GEMM algorithms.
 *
 * This function implements the core FDA/GDFS accumulation algorithm:
 * 1. Scan all operands for NaN/Inf and find the maximum exponent
 * 2. Align all operands to the maximum exponent and accumulate
 * 3. Convert the fixed-point sum back to FP32
 *
 * @tparam F Number of fractional bits in the accumulator
 * @tparam N Number of operands to accumulate
 * @param operands Array of operands to accumulate
 * @param c Running accumulator value (FP32)
 * @return New FP32 accumulator value
 */
template<int F, int N>
[[nodiscard]] __device__ __forceinline__
float chunked_accumulate(const Operand* operands, float c) {
    // Convert running accumulator to operand format
    Operand c_operand = fp32_to_operand<F>(c);

    // Early return if accumulator is already NaN
    if (c_operand.is_nan) {
        return bits_to_fp32(fp32::QNAN_BITS);
    }

    // ========================================
    // Pass 1: Fused NaN/Inf check + max exponent
    // ========================================
    bool any_nan = false;
    bool any_inf = false;
    int pos_inf_count = 0;
    int neg_inf_count = 0;
    int max_exp = -9999;
    int non_zero_count = 0;

    // Check accumulator c
    if (c_operand.is_inf) {
        any_inf = true;
        if (c_operand.sign > 0) pos_inf_count++;
        else neg_inf_count++;
    }
    if (!c_operand.is_zero && !c_operand.is_nan && !c_operand.is_inf) {
        max_exp = c_operand.exponent;
        non_zero_count++;
    }

    // Check all operands
    #pragma unroll
    for (int i = 0; i < N; i++) {
        if (operands[i].is_nan) {
            any_nan = true;
        }
        if (operands[i].is_inf) {
            any_inf = true;
            if (operands[i].sign > 0) pos_inf_count++;
            else neg_inf_count++;
        }
        if (!operands[i].is_zero && !operands[i].is_nan && !operands[i].is_inf) {
            max_exp = max(max_exp, operands[i].exponent);
            non_zero_count++;
        }
    }

    // Handle NaN: any NaN input produces NaN output
    if (any_nan) {
        return bits_to_fp32(fp32::QNAN_BITS);
    }

    // Handle Inf: +Inf and -Inf together produce NaN
    if (any_inf) {
        if (pos_inf_count > 0 && neg_inf_count > 0) {
            return bits_to_fp32(fp32::QNAN_BITS);
        }
        return bits_to_fp32(pos_inf_count > 0 ? fp32::POS_INF_BITS : fp32::NEG_INF_BITS);
    }

    // If all operands are zero, return the original accumulator
    if (non_zero_count == 0) {
        return c;
    }

    // ========================================
    // Pass 2: Align and accumulate
    // ========================================
    int64_t mantissa_sum = 0;

    // Add accumulator c (aligned to max exponent)
    if (!c_operand.is_zero && !c_operand.is_inf) {
        int exp_diff = max_exp - c_operand.exponent;
        int64_t aligned = (exp_diff >= 64) ? 0 : (c_operand.significand >> exp_diff);
        mantissa_sum += c_operand.sign * aligned;
    }

    // Add all operands (aligned to max exponent)
    #pragma unroll
    for (int i = 0; i < N; i++) {
        if (operands[i].is_zero || operands[i].is_nan || operands[i].is_inf) {
            continue;
        }
        int exp_diff = max_exp - operands[i].exponent;
        int64_t aligned = (exp_diff >= 64) ? 0 : (operands[i].significand >> exp_diff);
        mantissa_sum += operands[i].sign * aligned;
    }

    return fixed_to_fp32<F>(mantissa_sum, max_exp);
}

// ============================================================================
// FDA-Specific Accumulator (for FP8 Products)
// ============================================================================

/**
 * @brief FDA accumulator for FP8 products with explicit chunk processing.
 *
 * This variant accepts Product structures (from FP8 multiplication) and
 * applies FDA-specific truncation rules.
 *
 * @tparam F Fractional bits F
 * @tparam CHUNK_SIZE Products per CoFDA chunk (CS)
 * @param products Array of FP8 products
 * @param c Running FP32 accumulator
 * @return New FP32 accumulator value
 */
template<int F, int CHUNK_SIZE>
[[nodiscard]] __device__ __forceinline__
float fda_accumulate_chunk(const Product* products, float c = 0.0f) {
    // Convert products to operands
    Operand operands[CHUNK_SIZE];

    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; i++) {
        operands[i].sign = products[i].sign;
        operands[i].exponent = products[i].exponent;
        operands[i].significand = static_cast<int64_t>(products[i].significand);
        operands[i].is_zero = products[i].is_zero;
        operands[i].is_nan = products[i].is_nan;
        operands[i].is_inf = products[i].is_inf;
    }

    return chunked_accumulate<F, CHUNK_SIZE>(operands, c);
}

// ============================================================================
// C-decoupled CoFDA Accumulator (Two-Step CoFDA for FP8)
// ============================================================================

/**
 * @brief C-decoupled CoFDA accumulation over a chunk of FP8 products.
 *
 * Unlike fda_accumulate_chunk which aligns all products AND the running
 * accumulator c together (risking precision loss when c dominates), this
 * the accumulator is excluded from the F-bit datapath:
 *
 *   Step 1: Sum CHUNK_SIZE products among themselves with F bits (c=0).
 *           Products are aligned only to each other's exponent range.
 *
 *   Step 2: Add the partial_sum to the running accumulator c with F2 bits.
 *           Only 2 operands are aligned, with F2 controlling precision.
 *
 * This preserves more precision from small products in Step 1, since they
 * are not dominated by a large accumulated c during alignment.
 *
 * @tparam F  Fractional bits for the product sum (Step 1)
 * @tparam F2 Fractional bits for the accumulator merge (Step 2)
 * @tparam CHUNK_SIZE Products per CoFDA chunk (CS)
 * @param products Array of FP8 products
 * @param c Running FP32 accumulator
 * @return New FP32 accumulator value
 */
template<int F, int F2, int CHUNK_SIZE>
[[nodiscard]] __device__ __forceinline__
float cofda_decoupled_accumulate_products(const Product* products, float c = 0.0f) {
    // Step 1: Sum products only (no c), with F bits
    Operand operands[CHUNK_SIZE];

    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; i++) {
        operands[i].sign = products[i].sign;
        operands[i].exponent = products[i].exponent;
        operands[i].significand = static_cast<int64_t>(products[i].significand);
        operands[i].is_zero = products[i].is_zero;
        operands[i].is_nan = products[i].is_nan;
        operands[i].is_inf = products[i].is_inf;
    }

    // c=0.0f: products are accumulated among themselves only
    float partial_sum = chunked_accumulate<F, CHUNK_SIZE>(operands, 0.0f);

    // Step 2: Add partial_sum to running accumulator c, with F2 bits
    Operand ps_operand = fp32_to_operand<F2>(partial_sum);
    return chunked_accumulate<F2, 1>(&ps_operand, c);
}

// ============================================================================
// GDFS-Specific Accumulator (for FP4 Groups)
// ============================================================================

/**
 * @brief GDFS accumulator for scaled FP4 group sums.
 *
 * This variant is designed for GDFS accumulation where groups have already
 * been scaled and converted to Operand format.
 *
 * @tparam F Inter-group fractional bits F
 * @tparam NUM_GROUPS Groups per K-tile (BK / GS)
 * @param groups Array of scaled group operands
 * @param c Running FP32 accumulator
 * @return New FP32 accumulator value
 */
template<int F, int NUM_GROUPS>
[[nodiscard]] __device__ __forceinline__
float gdfs_accumulate_tile(const Operand* groups, float c) {
    return chunked_accumulate<F, NUM_GROUPS>(groups, c);
}

// ============================================================================
// Zero-Initialize Operand Helper
// ============================================================================

/**
 * @brief Create a zero-valued operand.
 *
 * Useful for padding when the actual data is out of bounds.
 */
[[nodiscard]] __device__ __forceinline__
Operand make_zero_operand() {
    Operand result;
    result.is_zero = true;
    result.is_nan = false;
    result.is_inf = false;
    result.sign = 1;
    result.exponent = 0;
    result.significand = 0;
    return result;
}


}  // namespace mma_emu
}  // namespace vllm
