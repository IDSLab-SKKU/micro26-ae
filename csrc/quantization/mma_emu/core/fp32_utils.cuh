/*
 * MMA-Emu FP32 Bit Manipulation
 *
 * IEEE-754 bit extraction, exponent alignment, round-to-zero truncation and
 * fixed-point conversion used by the accumulators.
 */

#pragma once

#include <cstdint>
#include "types.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// FP32 IEEE 754 Constants
// ============================================================================

namespace fp32 {

/// Bit masks for FP32 IEEE 754 format
constexpr unsigned int SIGN_MASK     = 0x80000000U;
constexpr unsigned int ABS_MASK      = 0x7FFFFFFFU;
constexpr unsigned int MANTISSA_MASK = 0x007FFFFFU;
constexpr unsigned int EXPONENT_MASK = 0x7F800000U;
constexpr unsigned int IMPLICIT_BIT  = 0x00800000U;

/// Numeric constants
constexpr int EXPONENT_BIAS  = 127;
constexpr int MANTISSA_BITS  = 23;
constexpr int MIN_NORMAL_EXP = -126;
constexpr int MAX_BIASED_EXP = 255;

/// Special value patterns
constexpr unsigned int POS_INF_BITS = 0x7F800000U;
constexpr unsigned int NEG_INF_BITS = 0xFF800000U;
constexpr unsigned int QNAN_BITS    = 0x7FFFFFFFU;

}  // namespace fp32

// ============================================================================
// Bit Casting Utilities
// ============================================================================

/**
 * @brief Reinterpret FP32 as unsigned int (type punning).
 */
[[nodiscard]] __device__ __forceinline__
unsigned int fp32_to_bits(float val) {
    return __float_as_uint(val);
}

/**
 * @brief Reinterpret unsigned int as FP32 (type punning).
 */
[[nodiscard]] __device__ __forceinline__
float bits_to_fp32(unsigned int bits) {
    return __uint_as_float(bits);
}

// ============================================================================
// FP32 Classification Utilities
// ============================================================================

/**
 * @brief Check if FP32 value is zero (positive or negative).
 */
[[nodiscard]] __device__ __forceinline__
bool fp32_is_zero(float val) {
    return (fp32_to_bits(val) & fp32::ABS_MASK) == 0;
}

/**
 * @brief Check if FP32 value is infinity (positive or negative).
 */
[[nodiscard]] __device__ __forceinline__
bool fp32_is_inf(float val) {
    unsigned int abs_bits = fp32_to_bits(val) & fp32::ABS_MASK;
    return abs_bits == fp32::POS_INF_BITS;
}

/**
 * @brief Check if FP32 value is NaN.
 */
[[nodiscard]] __device__ __forceinline__
bool fp32_is_nan(float val) {
    unsigned int abs_bits = fp32_to_bits(val) & fp32::ABS_MASK;
    return abs_bits > fp32::POS_INF_BITS;
}

// ============================================================================
// FP32 to Operand Conversion
// ============================================================================

/**
 * @brief Convert FP32 value to Operand format with F fractional bits.
 *
 * This function extracts the sign, exponent, and significand from an FP32 value
 * and represents them in a format suitable for FDA/GDFS accumulation.
 *
 * @tparam F Number of fractional bits in the significand
 * @param val FP32 value to convert
 * @return Operand representation
 */
template<int F>
[[nodiscard]] __device__ __forceinline__
Operand fp32_to_operand(float val) {
    Operand result;
    result.is_inf = false;
    result.is_nan = false;

    // Handle zero case
    if (val == 0.0f) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_zero = false;

    // Extract bits
    unsigned int bits = fp32_to_bits(val);
    result.sign = (bits & fp32::SIGN_MASK) ? -1 : 1;

    unsigned int abs_bits = bits & fp32::ABS_MASK;
    int biased_exp = static_cast<int>((abs_bits >> fp32::MANTISSA_BITS) & 0xFF);
    unsigned int mantissa = abs_bits & fp32::MANTISSA_MASK;

    // Handle special cases
    if (biased_exp == fp32::MAX_BIASED_EXP) {
        if (mantissa == 0) {
            result.is_inf = true;
        } else {
            result.is_nan = true;
        }
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    // Handle subnormal
    if (biased_exp == 0) {
        if (mantissa == 0) {
            result.is_zero = true;
            return result;
        }
        // Normalize subnormal
        int leading_zeros = __clz(mantissa);
        int shift = leading_zeros - 8;
        result.exponent = fp32::MIN_NORMAL_EXP - shift;
        mantissa = mantissa << (shift + 1);
        mantissa &= fp32::MANTISSA_MASK;
    } else {
        result.exponent = biased_exp - fp32::EXPONENT_BIAS;
    }

    // Convert to F-bit significand
    uint64_t fp32_sig = fp32::IMPLICIT_BIT | mantissa;
    constexpr int SHIFT_AMOUNT = fp32::MANTISSA_BITS - F;

    if constexpr (SHIFT_AMOUNT >= 0) {
        result.significand = static_cast<int64_t>(fp32_sig >> SHIFT_AMOUNT);
    } else {
        result.significand = static_cast<int64_t>(fp32_sig << (-SHIFT_AMOUNT));
    }

    return result;
}

// ============================================================================
// Fixed-Point to FP32 Conversion
// ============================================================================

/**
 * @brief Convert fixed-point mantissa sum to FP32.
 *
 * This is the final step of FDA/GDFS accumulation, normalizing the accumulated
 * fixed-point result back to FP32.
 *
 * @tparam F Number of fractional bits in the accumulator
 * @param mantissa_sum Signed fixed-point accumulator value
 * @param max_exp Maximum exponent used during accumulation
 * @return Normalized FP32 result
 */
template<int F>
[[nodiscard]] __device__ __forceinline__
float fixed_to_fp32(int64_t mantissa_sum, int max_exp) {
    if (mantissa_sum == 0) {
        return 0.0f;
    }

    // Determine sign and absolute value
    unsigned int result_sign = (mantissa_sum < 0) ? fp32::SIGN_MASK : 0;
    uint64_t abs_mantissa = (mantissa_sum < 0)
                          ? static_cast<uint64_t>(-mantissa_sum)
                          : static_cast<uint64_t>(mantissa_sum);

    // Find leading one position
    int leading_zeros = __clzll(abs_mantissa);
    int leading_one_pos = 63 - leading_zeros;

    // Compute final exponent
    int final_exp = leading_one_pos + max_exp - F;
    int biased_exp = final_exp + fp32::EXPONENT_BIAS;

    // Handle underflow to subnormal or zero
    if (biased_exp <= 0) {
        if (biased_exp < -static_cast<int>(fp32::MANTISSA_BITS)) {
            return 0.0f;
        }
        int subnormal_shift = 1 - biased_exp;
        uint32_t subnormal_mantissa;
        if (leading_one_pos >= static_cast<int>(fp32::MANTISSA_BITS)) {
            subnormal_mantissa = static_cast<uint32_t>(
                abs_mantissa >> (leading_one_pos - fp32::MANTISSA_BITS + subnormal_shift));
        } else {
            subnormal_mantissa = static_cast<uint32_t>(
                abs_mantissa << (fp32::MANTISSA_BITS - leading_one_pos - subnormal_shift));
        }
        subnormal_mantissa &= fp32::MANTISSA_MASK;
        return bits_to_fp32(result_sign | subnormal_mantissa);
    }

    // Handle overflow to infinity
    if (biased_exp >= static_cast<int>(fp32::MAX_BIASED_EXP)) {
        return bits_to_fp32(result_sign | fp32::POS_INF_BITS);
    }

    // Normalize mantissa
    uint32_t normalized_mantissa;
    if (leading_one_pos >= static_cast<int>(fp32::MANTISSA_BITS)) {
        normalized_mantissa = static_cast<uint32_t>(
            abs_mantissa >> (leading_one_pos - fp32::MANTISSA_BITS));
    } else {
        normalized_mantissa = static_cast<uint32_t>(
            abs_mantissa << (fp32::MANTISSA_BITS - leading_one_pos));
    }
    normalized_mantissa &= fp32::MANTISSA_MASK;

    // Final truncation to F fractional bits (round-to-zero)
    // For F < 23: truncate at F-th fractional bit
    // For F >= 23: truncate at 23rd fractional bit (FP32 mantissa width)
    constexpr int EFFECTIVE_TRUNC_POINT = (F < static_cast<int>(fp32::MANTISSA_BITS))
                                           ? F
                                           : static_cast<int>(fp32::MANTISSA_BITS);
    constexpr int FINAL_TRUNC_BITS = fp32::MANTISSA_BITS - EFFECTIVE_TRUNC_POINT;
    if constexpr (FINAL_TRUNC_BITS > 0) {
        constexpr uint32_t FINAL_TRUNC_MASK = fp32::MANTISSA_MASK & (~((1u << FINAL_TRUNC_BITS) - 1));
        normalized_mantissa &= FINAL_TRUNC_MASK;
    }

    // Assemble final result
    unsigned int result_bits = result_sign |
                               (static_cast<unsigned int>(biased_exp) << fp32::MANTISSA_BITS) |
                               normalized_mantissa;
    return bits_to_fp32(result_bits);
}

}  // namespace mma_emu
}  // namespace vllm
