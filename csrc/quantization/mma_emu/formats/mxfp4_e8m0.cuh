/*
 * MMA-Emu MXFP4 Utilities
 *
 * Provides MXFP4-specific functions for the GDFS and CoFDA emulation kernels:
 * - E8M0 scale factor constants and NaN checking
 * - E8M0 scale factor application (GDFS STP4) — pure exponent addition
 * - FP4 product with E8M0 scales (CoFDA) — product-level scale incorporation
 *
 * MXFP4 shares the E2M1 element format with NVFP4 (formats/fp4_e2m1.cuh) and
 * differs only in the block scale:
 *
 * - Scale format: E8M0 (pure power-of-2) rather than UE4M3 (has a mantissa)
 * - Block size: 32 rather than 16
 * - No per-tensor alpha scale
 *
 * Applying an E8M0 scale is a pure exponent addition — no multiply step.
 *
 */

#pragma once

#include <cstdint>
#include "../core/types.cuh"
#include "../core/fp32_utils.cuh"
#include "../core/gdfs_group.cuh"
#include "fp4_e2m1.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// MXFP4 Format Constants
// ============================================================================

namespace mxfp4 {

/// Block size for MXFP4 (scale_vec::2X)
constexpr int BLOCK_SIZE = 32;

/// Number of scale factors per K=64 MMA tile (64 / BLOCK_SIZE = 2)
constexpr int NUM_SCALES = 2;

}  // namespace mxfp4

// ============================================================================
// E8M0 Scale Factor Constants
// ============================================================================

namespace e8m0 {

/// E8M0 exponent bias
constexpr int EXPONENT_BIAS = 127;

/// NaN value: E8M0 defines 0xFF as NaN
constexpr uint8_t NAN_VALUE = 0xFF;

/**
 * @brief Check if a UE8M0 value is NaN.
 *
 * Per OCP MX specification, E8M0 value 0xFF is NaN.
 */
[[nodiscard]] __device__ __forceinline__
bool is_nan(uint8_t val) {
    return val == NAN_VALUE;
}

/**
 * @brief Get unbiased exponent from E8M0 value.
 *
 * E8M0 represents: value = 2^(val - 127)
 * Returns unbiased exponent: val - 127
 *
 * Note: val=0 represents 2^(-127), not zero.
 * GPU hardware may treat val=0 as zero for practical purposes.
 */
[[nodiscard]] __device__ __forceinline__
int to_exponent(uint8_t val) {
    return static_cast<int>(val) - EXPONENT_BIAS;
}

}  // namespace e8m0

// ============================================================================
// E8M0 Scale Factor Application for GDFS (STP4)
// ============================================================================

/**
 * @brief Apply E8M0 scale factors to a group accumulation result (GDFS STP4).
 *
 * Takes the output of group_accumulate<G, N> (mantissa_sum in G-bit format
 * with associated max_exp) and multiplies by two E8M0 scale factors to
 * produce an Operand<F> for the fused-sum accumulation (STP5-7).
 *
 * E8M0 is a pure power-of-2 format (no mantissa), so the scale application
 * is purely an exponent addition — no significand multiplication needed.
 * This is much simpler than UE4M3's apply_ue4m3_scales which requires
 * a Q1.3 mantissa multiply.
 *
 * Arithmetic:
 *   gamma_g = sigma_g * 2^(sfa_exp - 127) * 2^(sfb_exp - 127)
 *           = mantissa_sum * 2^(max_exp - G + sfa_exp + sfb_exp - 254)
 *
 *   Result significand: mantissa_sum shifted from G radix to F radix
 *     SHIFT_TO_F = F - G  (vs F - (G + 6) for UE4M3)
 *   Result exponent: max_exp + (sfa_exp - 127) + (sfb_exp - 127)
 *
 * @tparam F Fractional bits F
 * @tparam G GDFS intra-group bits G
 * @param mantissa_sum Signed fixed-point group sum from group_accumulate (radix at G)
 * @param max_exp Maximum exponent from group alignment
 * @param sfa E8M0 scale factor for A (uint8)
 * @param sfb E8M0 scale factor for B (uint8)
 * @return Operand with F fractional bits, ready for STP5-7 accumulation
 */
template<int F, int G>
[[nodiscard]] __device__ __forceinline__
Operand apply_e8m0_scales(int64_t mantissa_sum, int max_exp,
                          uint8_t sfa, uint8_t sfb) {
    Operand result;

    // Check for NaN scale factors
    if (e8m0::is_nan(sfa) || e8m0::is_nan(sfb)) {
        result.is_nan = true;
        result.is_zero = false;
        result.is_inf = false;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_nan = false;
    result.is_inf = false;

    // Handle zero mantissa_sum
    if (mantissa_sum == 0) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    // Handle E8M0 value 0 (2^-127) as zero contribution
    // GPU hardware treats E8M0=0 as effectively zero scale factor
    if (sfa == 0 || sfb == 0) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_zero = false;

    // Determine sign and get absolute value
    if (mantissa_sum < 0) {
        result.sign = -1;
        mantissa_sum = -mantissa_sum;
    } else {
        result.sign = 1;
    }

    // E8M0 is pure power-of-2: no mantissa multiplication needed.
    // Just shift the significand from G radix to F radix.
    constexpr int SHIFT_TO_F = F - G;

    if constexpr (SHIFT_TO_F >= 0) {
        result.significand = static_cast<uint64_t>(mantissa_sum) << SHIFT_TO_F;
    } else {
        result.significand = static_cast<uint64_t>(mantissa_sum) >> (-SHIFT_TO_F);
    }

    // Combined exponent: max_exp + unbiased_sfa + unbiased_sfb
    int combined_sf_exp = e8m0::to_exponent(sfa) + e8m0::to_exponent(sfb);
    result.exponent = max_exp + combined_sf_exp;

    return result;
}

// ============================================================================
// FP4 Product with E8M0 Scales for CoFDA
// ============================================================================

/**
 * @brief Compute a single FP4 product with E8M0 scale factors applied.
 *
 * This is the CoFDA analog of GDFS STP2-4 combined into a single operation.
 * Instead of group accumulation + scale application, CoFDA applies scales
 * at the product level and produces an Operand ready for chunked accumulation.
 *
 * Arithmetic:
 *   result = (a * b) * 2^(sfa - 127) * 2^(sfb - 127)
 *          = sig_a * sig_b * 2^(exp_a + exp_b + sfa + sfb - 254)
 *
 *   Significand: raw_product shifted to F-bit radix
 *     SCALE_SHIFT = F - PRODUCT_RADIX_BIT = F - 2
 *   Exponent: exp_a + exp_b + (sfa - 127) + (sfb - 127)
 *
 * @tparam F Fractional bits F
 * @param a First FP4 E2M1 operand (decomposed)
 * @param b Second FP4 E2M1 operand (decomposed)
 * @param sfa E8M0 scale factor for A (uint8)
 * @param sfb E8M0 scale factor for B (uint8)
 * @return Operand with F fractional bits, ready for chunked accumulation
 */
template<int F>
[[nodiscard]] __device__ __forceinline__
Operand fp4_product_with_e8m0_scales(FP4Components a, FP4Components b,
                                      uint8_t sfa, uint8_t sfb) {
    Operand result;
    result.is_nan = false;
    result.is_inf = false;

    // Check for NaN scale factors
    if (e8m0::is_nan(sfa) || e8m0::is_nan(sfb)) {
        result.is_nan = true;
        result.is_zero = false;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    // Handle zero inputs or zero scale factors
    // E8M0 value 0 represents 2^-127, treated as zero for practical purposes
    if (a.is_zero || b.is_zero || sfa == 0 || sfb == 0) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_zero = false;
    result.sign = a.sign * b.sign;

    // Compute significands with implicit bit and unbiased exponents
    // (same logic as fp4_multiply_unnormalized)
    uint32_t sig_a, sig_b;
    int exp_a, exp_b;

    if (a.is_subnormal) {
        sig_a = a.mantissa;
        exp_a = 1 - fp4::EXPONENT_BIAS;
    } else {
        sig_a = fp4::IMPLICIT_ONE + a.mantissa;
        exp_a = a.exponent - fp4::EXPONENT_BIAS;
    }

    if (b.is_subnormal) {
        sig_b = b.mantissa;
        exp_b = 1 - fp4::EXPONENT_BIAS;
    } else {
        sig_b = fp4::IMPLICIT_ONE + b.mantissa;
        exp_b = b.exponent - fp4::EXPONENT_BIAS;
    }

    // Multiply significands: raw product is in Q2.2 format (radix at bit 2)
    uint64_t raw_product = static_cast<uint64_t>(sig_a) * static_cast<uint64_t>(sig_b);

    // Scale to F-bit radix: SCALE_SHIFT = F - PRODUCT_RADIX_BIT = F - 2
    // E8M0 is pure power-of-2 so no significand multiplication needed for scales
    constexpr int SCALE_SHIFT = F - fp4::PRODUCT_RADIX_BIT;

    if constexpr (SCALE_SHIFT >= 0) {
        result.significand = static_cast<int64_t>(raw_product) << SCALE_SHIFT;
    } else {
        result.significand = static_cast<int64_t>(raw_product) >> (-SCALE_SHIFT);
    }

    // Combined exponent: product exponent + E8M0 scale exponents
    int combined_sf_exp = e8m0::to_exponent(sfa) + e8m0::to_exponent(sfb);
    result.exponent = exp_a + exp_b + combined_sf_exp;

    // Product may become zero after right-shift truncation (F < 2)
    if (result.significand == 0) {
        result.is_zero = true;
    }

    return result;
}

}  // namespace mma_emu
}  // namespace vllm
