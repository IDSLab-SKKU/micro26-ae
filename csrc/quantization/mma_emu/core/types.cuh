/*
 * MMA-Emu Types
 *
 * Operand, Product and fixed-point representations used by the accumulators.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace vllm {
namespace mma_emu {

// ============================================================================
// Forward Declarations
// ============================================================================

struct FP8E4M3Tag;
struct FP4E2M1Tag;

// ============================================================================
// Floating-Point Operand
// ============================================================================

/**
 * @brief Operand of the fixed-point accumulators.
 *
 * A floating-point value split into the form the FDA, CoFDA and GDFS
 * accumulators consume. The significand is stored
 * with F fractional bits below the radix point.
 */
struct Operand {
    int sign;              ///< Sign: +1 or -1
    int exponent;          ///< Unbiased exponent
    int64_t significand;   ///< Significand with F fractional bits (signed magnitude)
    bool is_zero;          ///< True if value is zero
    bool is_nan;           ///< True if value is NaN
    bool is_inf;           ///< True if value is infinity
};

/**
 * @brief Product of two low-precision floating-point values.
 *
 * Used as intermediate result before FDA/GDFS accumulation.
 * For FP8: product of two E4M3 values
 * For FP4: not directly used (FP4 products are computed as fixed-point)
 */
struct Product {
    int sign;              ///< Sign: +1 or -1
    int exponent;          ///< Combined exponent (exp_a + exp_b)
    uint64_t significand;  ///< Product significand with F fractional bits
    bool is_zero;          ///< True if either operand was zero
    bool is_nan;           ///< True if either operand was NaN
    bool is_inf;           ///< True if product is infinity
    bool is_subnormal;     ///< True if result is subnormal (for FP8)
};

// ============================================================================
// FP8 E4M3 Type Traits
// ============================================================================

/**
 * @brief Compile-time traits for FP8 E4M3 format.
 */
template<typename = FP8E4M3Tag>
struct FP8Traits {
    static constexpr int EXPONENT_BITS = 4;
    static constexpr int MANTISSA_BITS = 3;
    static constexpr int EXPONENT_BIAS = 7;
    static constexpr int IMPLICIT_ONE = 8;       ///< 1 << MANTISSA_BITS
    static constexpr int PRODUCT_RADIX_BIT = 6;  ///< 2 * MANTISSA_BITS
    static constexpr uint8_t NAN_PATTERN = 0x7F;
};

// ============================================================================
// FP4 E2M1 Type Traits
// ============================================================================

/**
 * @brief Compile-time traits for FP4 E2M1 format.
 */
template<typename = FP4E2M1Tag>
struct FP4Traits {
    static constexpr int EXPONENT_BITS = 2;
    static constexpr int MANTISSA_BITS = 1;
    static constexpr int EXPONENT_BIAS = 1;
    static constexpr int BLOCK_SIZE = 16;        ///< NVFP4 scale block size
};

// ============================================================================
// UE4M3 Scale Factor Traits
// ============================================================================

/**
 * @brief Compile-time traits for UE4M3 unsigned scale factor format.
 */
struct UE4M3Traits {
    static constexpr int EXPONENT_BITS = 4;
    static constexpr int MANTISSA_BITS = 3;
    static constexpr int EXPONENT_BIAS = 7;
    static constexpr uint8_t NAN_VALUE = 0x7F;
};

// ============================================================================
// Output Type Conversion Traits
// ============================================================================

/**
 * @brief Type trait for output data type conversion.
 *
 * Provides compile-time information and conversion functions for output types.
 */
template<typename T>
struct OutputTraits;

template<>
struct OutputTraits<__nv_bfloat16> {
    using Type = __nv_bfloat16;

    [[nodiscard]] __device__ __forceinline__
    static Type from_float(float val) {
        return __float2bfloat16_rn(val);
    }
};

template<>
struct OutputTraits<__half> {
    using Type = __half;

    [[nodiscard]] __device__ __forceinline__
    static Type from_float(float val) {
        return __float2half_rn(val);
    }
};

// ============================================================================
// Convenience Type Conversion Function
// ============================================================================

/**
 * @brief Convert FP32 to output type with round-to-nearest.
 *
 * This is the primary interface for output conversion, using tag dispatch
 * to select the appropriate conversion function.
 *
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 * @param val FP32 value to convert
 * @return Converted value in output type
 */
template<typename OutDtype>
[[nodiscard]] __device__ __forceinline__
OutDtype float_to_output_rn(float val) {
    return OutputTraits<OutDtype>::from_float(val);
}

// Overload taking a pointer, for tag dispatch.
template<typename OutDtype>
[[nodiscard]] __device__ __forceinline__
OutDtype float_to_output_rn(float val, OutDtype*) {
    return OutputTraits<OutDtype>::from_float(val);
}

}  // namespace mma_emu
}  // namespace vllm
