/*
 * NVFP4 UE4M3 Scale Factor Application
 *
 * NVFP4-specific scale handling: UE4M3 block scales with a
 * mantissa, group size 16. Applied at group level for GDFS and at product
 * level for CoFDA. MXFP4 uses E8M0 instead (mxfp4_e8m0.cuh).
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
// UE4M3 Scale Factor Constants
// ============================================================================

namespace ue4m3 {

constexpr uint8_t NAN_VALUE = 0x7F;
constexpr int EXPONENT_BIAS = 7;
constexpr int Q1_3_SHIFT = 3;  // Significand is in Q1.3 format (scaled by 8)

/**
 * @brief Check if a UE4M3 value is NaN.
 */
[[nodiscard]] __device__ __forceinline__
bool is_nan(uint8_t val) {
    return (val & 0x7F) == NAN_VALUE;
}

}  // namespace ue4m3

// ============================================================================
// UE4M3 Scale Factor Application for GDFS (STP4)
// ============================================================================

/**
 * @brief Apply UE4M3 scale factors to a group accumulation result (GDFS STP4).
 *
 * Takes the output of group_accumulate<G, N> (mantissa_sum in G-bit format
 * with associated max_exp) and multiplies by two UE4M3 scale factors to
 * produce an Operand<F> for the fused-sum accumulation (STP5-7).
 *
 * Arithmetic:
 *   γ_g = σ_g × sfa × sfb
 *       = mantissa_sum × 2^(max_exp - G) × (sig_sfa/8 × 2^exp_sfa) × (sig_sfb/8 × 2^exp_sfb)
 *       = (mantissa_sum × sig_sfa × sig_sfb) × 2^(max_exp + exp_sfa + exp_sfb - G - 6)
 *   COMBINED_RADIX = G + 6
 *   SHIFT_TO_F = F - (G + 6)
 *
 * @tparam F Fractional bits F
 * @tparam G GDFS intra-group bits G
 * @param mantissa_sum Signed fixed-point group sum from group_accumulate (radix at G)
 * @param max_exp Maximum exponent from group alignment
 * @param sfa UE4M3 scale factor for A
 * @param sfb UE4M3 scale factor for B
 * @return Operand with F fractional bits, ready for STP5-7 accumulation
 */
template<int F, int G>
[[nodiscard]] __device__ __forceinline__
Operand apply_ue4m3_scales(int64_t mantissa_sum, int max_exp,
                           uint8_t sfa, uint8_t sfb) {
    Operand result;

    // UE4M3 truncation (Step 0): mask MSB to 0
    uint8_t sfa_masked = sfa & 0x7F;
    uint8_t sfb_masked = sfb & 0x7F;

    // Check for NaN scale factors
    if (ue4m3::is_nan(sfa) || ue4m3::is_nan(sfb)) {
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

    // Handle UE4M3 value 0 as zero scale
    if (sfa_masked == 0 || sfb_masked == 0) {
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

    // Decompose UE4M3 scale factor A
    // Normal (exp != 0):    sig = 8 + mant (Q1.3), unbiased_exp = exp - 7
    // Subnormal (exp == 0): sig = mant (Q0.3),     unbiased_exp = -6
    int exp_a = (sfa_masked >> ue4m3::Q1_3_SHIFT) & 0xF;
    int mant_a = sfa_masked & 0x7;
    int unbiased_exp_a;
    int64_t sig_a;

    if (exp_a == 0) {
        unbiased_exp_a = -6;
        sig_a = mant_a;
    } else {
        unbiased_exp_a = exp_a - ue4m3::EXPONENT_BIAS;
        sig_a = 8 + mant_a;
    }

    // Decompose UE4M3 scale factor B
    int exp_b = (sfb_masked >> ue4m3::Q1_3_SHIFT) & 0xF;
    int mant_b = sfb_masked & 0x7;
    int unbiased_exp_b;
    int64_t sig_b;

    if (exp_b == 0) {
        unbiased_exp_b = -6;
        sig_b = mant_b;
    } else {
        unbiased_exp_b = exp_b - ue4m3::EXPONENT_BIAS;
        sig_b = 8 + mant_b;
    }

    // Scale factor product: sig_a × sig_b in Q2.6 format (×64)
    int64_t sf_product = sig_a * sig_b;
    int combined_sf_exp = unbiased_exp_a + unbiased_exp_b;

    // Multiply mantissa_sum (radix at G) by sf_product (radix at 6)
    // Result radix = G + 6
    int64_t scaled_sum = mantissa_sum * sf_product;

    // Convert to F-bit significand format
    constexpr int COMBINED_RADIX = G + 6;
    constexpr int SHIFT_TO_F = F - COMBINED_RADIX;

    if constexpr (SHIFT_TO_F >= 0) {
        result.significand = scaled_sum << SHIFT_TO_F;
    } else {
        result.significand = scaled_sum >> (-SHIFT_TO_F);
    }
    result.exponent = max_exp + combined_sf_exp;

    return result;
}

// ============================================================================
// UE4M3 Product-Level Scale Application for CoFDA
// ============================================================================

/**
 * @brief Compute FP4 product with UE4M3 scale factors applied at product level.
 *
 * This is the CoFDA analog for NVFP4: instead of grouping products first (GDFS),
 * each FP4 product is immediately scaled by UE4M3 block scales, producing an
 * Operand<F> ready for chunked accumulation.
 *
 * Arithmetic:
 *   result = (a × b) × sfa × sfb
 *          = (sig_a × sig_b) × (sig_sfa/8 × 2^exp_sfa) × (sig_sfb/8 × 2^exp_sfb)
 *          = (sig_a × sig_b × sig_sfa × sig_sfb) × 2^(exp_a + exp_b + exp_sfa + exp_sfb)
 *   Combined radix = PRODUCT_RADIX_BIT + 2 × Q1_3_SHIFT = 2 + 6 = 8
 *   SCALE_SHIFT = F - 8
 *
 * @tparam F Fractional bits F
 * @param a First FP4 E2M1 operand (decomposed)
 * @param b Second FP4 E2M1 operand (decomposed)
 * @param sfa UE4M3 scale factor for A
 * @param sfb UE4M3 scale factor for B
 * @return Operand with F fractional bits, ready for chunked accumulation
 */
template<int F>
[[nodiscard]] __device__ __forceinline__
Operand fp4_product_with_ue4m3_scales(FP4Components a, FP4Components b,
                                       uint8_t sfa, uint8_t sfb) {
    Operand result;
    result.is_inf = false;

    // UE4M3 truncation (Step 0): mask MSB to 0
    uint8_t sfa_masked = sfa & 0x7F;
    uint8_t sfb_masked = sfb & 0x7F;

    // Check for NaN scale factors
    if (ue4m3::is_nan(sfa) || ue4m3::is_nan(sfb)) {
        result.is_nan = true;
        result.is_zero = false;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_nan = false;

    // Handle zero inputs or zero scale factors
    if (a.is_zero || b.is_zero || sfa_masked == 0 || sfb_masked == 0) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_zero = false;
    result.sign = a.sign * b.sign;

    // Compute FP4 significands with implicit bit and unbiased exponents
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

    // Multiply FP4 significands: raw product in Q2.2 format (radix at bit 2)
    uint64_t raw_product = static_cast<uint64_t>(sig_a) * static_cast<uint64_t>(sig_b);

    // Decompose UE4M3 scale factor A
    // Normal (exp != 0):    sig = 8 + mant (Q1.3), unbiased_exp = exp - 7
    // Subnormal (exp == 0): sig = mant (Q0.3),     unbiased_exp = -6
    int exp_sfa = (sfa_masked >> ue4m3::Q1_3_SHIFT) & 0xF;
    int mant_sfa = sfa_masked & 0x7;
    int unbiased_exp_sfa;
    int64_t sig_sfa;

    if (exp_sfa == 0) {
        unbiased_exp_sfa = -6;
        sig_sfa = mant_sfa;
    } else {
        unbiased_exp_sfa = exp_sfa - ue4m3::EXPONENT_BIAS;
        sig_sfa = 8 + mant_sfa;
    }

    // Decompose UE4M3 scale factor B
    int exp_sfb = (sfb_masked >> ue4m3::Q1_3_SHIFT) & 0xF;
    int mant_sfb = sfb_masked & 0x7;
    int unbiased_exp_sfb;
    int64_t sig_sfb;

    if (exp_sfb == 0) {
        unbiased_exp_sfb = -6;
        sig_sfb = mant_sfb;
    } else {
        unbiased_exp_sfb = exp_sfb - ue4m3::EXPONENT_BIAS;
        sig_sfb = 8 + mant_sfb;
    }

    // Multiply: raw_product (radix at 2) × sig_sfa (radix at 3) × sig_sfb (radix at 3)
    // Result radix = 2 + 3 + 3 = 8
    int64_t scaled_product = static_cast<int64_t>(raw_product) * sig_sfa * sig_sfb;

    // Scale to F-bit radix
    constexpr int COMBINED_RADIX = fp4::PRODUCT_RADIX_BIT + 2 * ue4m3::Q1_3_SHIFT;  // 2 + 6 = 8
    constexpr int SCALE_SHIFT = F - COMBINED_RADIX;

    if constexpr (SCALE_SHIFT >= 0) {
        result.significand = scaled_product << SCALE_SHIFT;
    } else {
        result.significand = scaled_product >> (-SCALE_SHIFT);
    }

    // Combined exponent
    int combined_sf_exp = unbiased_exp_sfa + unbiased_exp_sfb;
    result.exponent = exp_a + exp_b + combined_sf_exp;

    // Product may become zero after right-shift truncation
    if (result.significand == 0) {
        result.is_zero = true;
    }

    return result;
}
}  // namespace mma_emu
}  // namespace vllm
