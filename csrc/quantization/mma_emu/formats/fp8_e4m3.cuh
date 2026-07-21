/*
 * MMA-Emu FP8 E4M3 Utilities
 *
 * Provides FP8-specific functions for:
 * - FP8 E4M3 decomposition
 * - Unnormalized FP8 multiplication for FDA and GDFS
 * - GDFS group-to-operand conversion (radix shift only, no scale factors)
 * - FP8 special value handling
 */

#pragma once

#include <cuda_fp8.h>
#include <cstdint>
#include "../core/types.cuh"
#include "../core/fp32_utils.cuh"
#include "../core/gdfs_group.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// FP8 E4M3 Component Structure
// ============================================================================

/**
 * @brief Decomposed components of an FP8 E4M3 value.
 */
struct FP8Components {
    int sign;           ///< Sign: +1 or -1
    int exponent;       ///< Biased exponent
    uint32_t mantissa;  ///< Mantissa without implicit bit
    bool is_zero;
    bool is_subnormal;
    bool is_nan;
    bool is_inf;        ///< Note: E4M3 has no infinity representation
};

// ============================================================================
// FP8 E4M3 Traits (Inline Constants)
// ============================================================================

namespace fp8 {

constexpr int EXPONENT_BITS = 4;
constexpr int MANTISSA_BITS = 3;
constexpr int EXPONENT_BIAS = 7;
constexpr int IMPLICIT_ONE = 8;       ///< 1 << MANTISSA_BITS
constexpr int PRODUCT_RADIX_BIT = 6;  ///< 2 * MANTISSA_BITS
constexpr uint8_t NAN_PATTERN = 0x7F;

// GDFS constants for FP8
// E4M3 product exp range: min_exp = 2*(1-7) = -12, max_exp = 2*(14-7) = 14
// Plus subnormal: min_exp with subnormal = (1-7) + (1-7) = -12
constexpr int MAX_PRODUCT_EXP_DIFF = 26;  ///< max_exp(14) - min_exp(-12)
constexpr int G_LOSSLESS = 32;            ///< PRODUCT_RADIX_BIT + MAX_PRODUCT_EXP_DIFF

}  // namespace fp8

// ============================================================================
// Decode-on-load layer (CoFDA fast path)
// ============================================================================
// FP8 byte decoded + subnormal-resolved ONCE at SMEM-load time. `significand`
// folds the implicit one (normal) or equals the raw mantissa (subnormal); `exp`
// is debiased. Consumed directly by fp8_multiply_predecoded — no per-element
// decode/subnormal branch in the inner loop.
struct DecodedOperand {
    int8_t sign;          // +1 / -1
    uint8_t significand;  // normal: 8+mantissa (8..15); subnormal: mantissa (1..7)
    int8_t exp;           // debiased exponent
    uint8_t flags;        // bit0 = is_nan, bit1 = is_zero
};

namespace decoded_flag {
constexpr uint8_t NAN_BIT = 0x1;
constexpr uint8_t ZERO_BIT = 0x2;
}  // namespace decoded_flag

// Pack/unpack DecodedOperand <-> one uint32 word: one LDS.32 instead of four
// 1-byte SoA loads. Signed fields round-trip via uint8 reinterpret.
[[nodiscard]] __device__ __forceinline__ uint32_t pack_decoded(DecodedOperand d) {
    return (static_cast<uint32_t>(static_cast<uint8_t>(d.sign))) |
           (static_cast<uint32_t>(d.significand) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(d.exp)) << 16) |
           (static_cast<uint32_t>(d.flags) << 24);
}

[[nodiscard]] __device__ __forceinline__ DecodedOperand unpack_decoded(uint32_t w) {
    DecodedOperand d;
    d.sign = static_cast<int8_t>(w & 0xFF);
    d.significand = static_cast<uint8_t>((w >> 8) & 0xFF);
    d.exp = static_cast<int8_t>((w >> 16) & 0xFF);
    d.flags = static_cast<uint8_t>((w >> 24) & 0xFF);
    return d;
}

// Accessor over one operand row stored as packed uint32 words.
struct DecodedFrag {
    const uint32_t* __restrict__ words;
    [[nodiscard]] __device__ __forceinline__ DecodedOperand operator[](int i) const {
        return unpack_decoded(words[i]);
    }
};

// Decode + resolve one FP8 E4M3 byte. Called once per byte at SMEM-load time.
[[nodiscard]] __device__ __forceinline__ DecodedOperand decode_operand(uint8_t bits) {
    DecodedOperand d;
    d.sign = (bits & 0x80) ? -1 : 1;
    const int exponent = (bits >> 3) & 0x0F;  // biased field
    const uint32_t mantissa = bits & 0x07;

    if ((bits & 0x7F) == fp8::NAN_PATTERN) {
        d.flags = decoded_flag::NAN_BIT;
        d.significand = 0;
        d.exp = 0;
        return d;
    }
    if (exponent == 0 && mantissa == 0) {  // zero
        d.flags = decoded_flag::ZERO_BIT;
        d.significand = 0;
        d.exp = 0;
        return d;
    }
    d.flags = 0;
    if (exponent == 0) {  // subnormal
        d.significand = static_cast<uint8_t>(mantissa);
        d.exp = static_cast<int8_t>(1 - fp8::EXPONENT_BIAS);
    } else {  // normal
        d.significand = static_cast<uint8_t>(fp8::IMPLICIT_ONE + mantissa);
        d.exp = static_cast<int8_t>(exponent - fp8::EXPONENT_BIAS);
    }
    return d;
}

// Multiply two pre-decoded E4M3 operands into an unnormalized Product carrying
// F fractional bits. The operands are already decoded, so no subnormal branch
// is needed here.
template <int F>
[[nodiscard]] __device__ __forceinline__ Product
fp8_multiply_predecoded(DecodedOperand a, DecodedOperand b) {
    Product result;
    result.is_nan = false;
    result.is_inf = false;

    if ((a.flags & decoded_flag::NAN_BIT) || (b.flags & decoded_flag::NAN_BIT)) {
        result.is_nan = true;
        result.is_zero = false;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }
    if ((a.flags & decoded_flag::ZERO_BIT) || (b.flags & decoded_flag::ZERO_BIT)) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_zero = false;
    result.sign = a.sign * b.sign;

    const uint64_t raw_product = static_cast<uint64_t>(a.significand) *
                                 static_cast<uint64_t>(b.significand);
    constexpr int SCALE_SHIFT = F - fp8::PRODUCT_RADIX_BIT;
    if constexpr (SCALE_SHIFT >= 0) {
        result.significand = raw_product << SCALE_SHIFT;
    } else {
        result.significand = raw_product >> (-SCALE_SHIFT);
    }
    result.exponent = a.exp + b.exp;
    return result;
}

// ============================================================================
// Register-streamed GDFS group accumulate for FP8 (no Product[GS] array)
// ============================================================================
// Returns the SAME GroupResult as group_accumulate<G,GS> over
// fp8_multiply_unnormalized<G> products, so the caller still applies
// group_to_operand_fp8<F,G> afterward. Two passes stream products through
// registers (recomputed in Pass 2 from the pre-decoded operands). Group
// semantics are is_zero-only — exactly matching the shared group_accumulate, so
// NaN/Inf products (significand 0, is_zero false) contribute as in the original.
// Bit-exact to fp8_multiply_unnormalized<G> + group_accumulate<G,GS>.
template <int G, int GS>
[[nodiscard]] __device__ __forceinline__ GroupResult
fp8_gdfs_group_accumulate_predecoded(DecodedFrag a_frag, DecodedFrag b_frag) {
    GroupResult gr;
    gr.mantissa_sum = 0;
    gr.max_exp = -9999;
    gr.all_zero = true;

    // Pass 1: max exponent among non-zero products
    #pragma unroll
    for (int i = 0; i < GS; i++) {
        Product p = fp8_multiply_predecoded<G>(a_frag[i], b_frag[i]);
        if (!p.is_zero) {
            gr.all_zero = false;
            if (p.exponent > gr.max_exp) {
                gr.max_exp = p.exponent;
            }
        }
    }

    if (gr.all_zero) {
        gr.max_exp = 0;
        return gr;
    }

    // Pass 2: align to max_exp (RZ truncation) and accumulate
    #pragma unroll
    for (int i = 0; i < GS; i++) {
        Product p = fp8_multiply_predecoded<G>(a_frag[i], b_frag[i]);
        if (p.is_zero) {
            continue;
        }
        int exp_diff = gr.max_exp - p.exponent;
        int64_t aligned =
            (exp_diff >= 64) ? 0
                             : (static_cast<int64_t>(p.significand) >> exp_diff);
        gr.mantissa_sum += p.sign * aligned;
    }
    return gr;
}

// ============================================================================
// FP8 E4M3 Decomposition
// ============================================================================

/**
 * @brief Decompose FP8 E4M3 value into sign, exponent, mantissa, and flags.
 *
 * E4M3 format:
 * - bit 7: sign (0=positive, 1=negative)
 * - bits 6-3: exponent (biased by 7)
 * - bits 2-0: mantissa (implicit 1 for normalized values)
 *
 * Special cases:
 * - Zero: exponent=0, mantissa=0 (±0)
 * - Subnormal: exponent=0, mantissa≠0
 * - NaN: pattern 0x7F or 0xFF (no infinity in E4M3)
 *
 * @param bits Raw 8-bit FP8 E4M3 representation
 * @return FP8Components with decomposed fields
 */
[[nodiscard]] __device__ __forceinline__
FP8Components decompose_fp8_e4m3(uint8_t bits) {
    FP8Components result;

    result.sign = (bits & 0x80) ? -1 : 1;
    result.exponent = (bits >> 3) & 0x0F;
    result.mantissa = bits & 0x07;
    result.is_inf = false;  // E4M3 has no infinity

    // NaN check: E4M3 NaN is 0x7F or 0xFF (max magnitude)
    if ((bits & 0x7F) == fp8::NAN_PATTERN) {
        result.is_nan = true;
        result.is_zero = false;
        result.is_subnormal = false;
        return result;
    }
    result.is_nan = false;

    // Zero check
    if (result.exponent == 0 && result.mantissa == 0) {
        result.is_zero = true;
        result.is_subnormal = false;
        return result;
    }

    result.is_zero = false;
    result.is_subnormal = (result.exponent == 0);
    return result;
}

// ============================================================================
// FP8 Unnormalized Multiplication
// ============================================================================

/**
 * @brief Multiply two FP8 E4M3 values producing an unnormalized product.
 *
 * The product is returned in a format suitable for FDA accumulation:
 * - significand has F fractional bits (radix at bit F)
 * - exponent is the sum of input exponents (unbiased)
 *
 * This function does NOT normalize the result - that happens during
 * FDA accumulation when converting back to FP32.
 *
 * @tparam F Fractional bits F
 * @param a First FP8 E4M3 operand (decomposed)
 * @param b Second FP8 E4M3 operand (decomposed)
 * @return Product structure ready for FDA accumulation
 */
template<int F>
[[nodiscard]] __device__ __forceinline__
Product fp8_multiply_unnormalized(FP8Components a, FP8Components b) {
    Product result;
    result.is_nan = false;
    result.is_inf = false;

    // Handle NaN inputs
    if (a.is_nan || b.is_nan) {
        result.is_nan = true;
        result.is_zero = false;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    // Handle zero inputs
    if (a.is_zero || b.is_zero) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_zero = false;
    result.sign = a.sign * b.sign;

    // Compute significands with implicit bit
    uint32_t sig_a, sig_b;
    int exp_a, exp_b;

    if (a.is_subnormal) {
        sig_a = a.mantissa;
        exp_a = 1 - fp8::EXPONENT_BIAS;
    } else {
        sig_a = fp8::IMPLICIT_ONE + a.mantissa;
        exp_a = a.exponent - fp8::EXPONENT_BIAS;
    }

    if (b.is_subnormal) {
        sig_b = b.mantissa;
        exp_b = 1 - fp8::EXPONENT_BIAS;
    } else {
        sig_b = fp8::IMPLICIT_ONE + b.mantissa;
        exp_b = b.exponent - fp8::EXPONENT_BIAS;
    }

    // Multiply and scale to put radix at bit F
    uint64_t raw_product = static_cast<uint64_t>(sig_a) * static_cast<uint64_t>(sig_b);
    constexpr int SCALE_SHIFT = F - fp8::PRODUCT_RADIX_BIT;
    if constexpr (SCALE_SHIFT >= 0) {
        result.significand = raw_product << SCALE_SHIFT;
    } else {
        result.significand = raw_product >> (-SCALE_SHIFT);
    }
    result.exponent = exp_a + exp_b;

    return result;
}

/**
 * @brief Compute FP8 product directly from raw bits.
 *
 * Convenience function that combines decomposition and multiplication.
 *
 * @tparam F Number of fractional bits
 * @param a_bits Raw FP8 E4M3 bits for first operand
 * @param b_bits Raw FP8 E4M3 bits for second operand
 * @return Product structure ready for FDA accumulation
 */
template<int F>
[[nodiscard]] __device__ __forceinline__
Product fp8_multiply(uint8_t a_bits, uint8_t b_bits) {
    FP8Components a_comp = decompose_fp8_e4m3(a_bits);
    FP8Components b_comp = decompose_fp8_e4m3(b_bits);
    return fp8_multiply_unnormalized<F>(a_comp, b_comp);
}

// ============================================================================
// FP8 Unnormalized Multiplication with G-bits (GDFS)
// ============================================================================

/**
 * @brief Multiply two FP8 E4M3 values producing an unnormalized product
 *        with G fractional bits in the significand.
 *
 *
 * The raw product significand is in Q2.6 format (radix at PRODUCT_RADIX_BIT=6).
 * It is scaled to G fractional bits:
 *   SCALE_SHIFT = G - PRODUCT_RADIX_BIT = G - 6
 *   if SCALE_SHIFT >= 0: significand = raw_product << SCALE_SHIFT
 *   if SCALE_SHIFT < 0:  significand = raw_product >> (-SCALE_SHIFT)
 *
 *   G=32: lossless (all product bits preserved after group alignment)
 *   G<32: right-shift truncates product bits at multiplication level
 *
 * @tparam G GDFS intra-group bits G
 * @param a First FP8 E4M3 operand (decomposed)
 * @param b Second FP8 E4M3 operand (decomposed)
 * @return Product structure ready for group accumulation
 */
template<int G>
[[nodiscard]] __device__ __forceinline__
Product fp8_multiply_unnormalized_g(FP8Components a, FP8Components b) {
    Product result;
    result.is_inf = false;
    result.is_subnormal = false;

    // Handle NaN inputs
    if (a.is_nan || b.is_nan) {
        result.is_nan = true;
        result.is_zero = false;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }
    result.is_nan = false;

    // Handle zero inputs
    if (a.is_zero || b.is_zero) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_zero = false;
    result.sign = a.sign * b.sign;

    // Compute significands with implicit bit
    uint32_t sig_a, sig_b;
    int exp_a, exp_b;

    if (a.is_subnormal) {
        sig_a = a.mantissa;
        exp_a = 1 - fp8::EXPONENT_BIAS;
    } else {
        sig_a = fp8::IMPLICIT_ONE + a.mantissa;
        exp_a = a.exponent - fp8::EXPONENT_BIAS;
    }

    if (b.is_subnormal) {
        sig_b = b.mantissa;
        exp_b = 1 - fp8::EXPONENT_BIAS;
    } else {
        sig_b = fp8::IMPLICIT_ONE + b.mantissa;
        exp_b = b.exponent - fp8::EXPONENT_BIAS;
    }

    // Multiply and scale to put radix at bit G.
    // Raw product is in Q2.6 format (radix at PRODUCT_RADIX_BIT = 6).
    // SCALE_SHIFT = G - 6
    uint64_t raw_product = static_cast<uint64_t>(sig_a) * static_cast<uint64_t>(sig_b);
    constexpr int SCALE_SHIFT = G - fp8::PRODUCT_RADIX_BIT;

    if constexpr (SCALE_SHIFT >= 0) {
        result.significand = raw_product << SCALE_SHIFT;
    } else {
        result.significand = raw_product >> (-SCALE_SHIFT);
    }
    result.exponent = exp_a + exp_b;

    // Product may become zero after right-shift truncation
    if (result.significand == 0) {
        result.is_zero = true;
    }

    return result;
}

/**
 * @brief Compute FP8 product directly from raw bits (G-bit variant).
 *
 * Convenience function that combines decomposition and multiplication
 * for GDFS group accumulation.
 *
 * @tparam G GDFS intra-group bits G
 * @param a_bits Raw FP8 E4M3 bits for first operand
 * @param b_bits Raw FP8 E4M3 bits for second operand
 * @return Product structure ready for group accumulation
 */
template<int G>
[[nodiscard]] __device__ __forceinline__
Product fp8_multiply_g(uint8_t a_bits, uint8_t b_bits) {
    FP8Components a_comp = decompose_fp8_e4m3(a_bits);
    FP8Components b_comp = decompose_fp8_e4m3(b_bits);
    return fp8_multiply_unnormalized_g<G>(a_comp, b_comp);
}

// ============================================================================
// FP8 GDFS Group-to-Operand Conversion (Radix Shift Only)
// ============================================================================

/**
 * @brief Convert GroupResult to Operand by shifting from G-bit to F-bit radix.
 *
 * Unlike apply_ue4m3_scales<F,G>() (NVFP4) or apply_e8m0_scales<F,G>()
 * (MXFP4), FP8 GDFS does NOT apply per-block scale factors during this step.
 * FP8 uses per-tensor scales applied in the epilogue.
 *
 * This function simply re-radixes the mantissa_sum:
 *   SHIFT = F - G
 *   if SHIFT >= 0: significand = mantissa_sum << SHIFT
 *   if SHIFT < 0:  significand = mantissa_sum >> (-SHIFT)
 *
 * @tparam F Number of fractional bits in the result (fused-sum accumulator)
 * @tparam G GDFS intra-group bits G
 * @param mantissa_sum Signed fixed-point group sum from group_accumulate
 * @param max_exp Maximum exponent from group alignment
 * @return Operand with F fractional bits, ready for GDFS fused-sum accumulation
 */
template<int F, int G>
[[nodiscard]] __device__ __forceinline__
Operand group_to_operand_fp8(int64_t mantissa_sum, int max_exp) {
    Operand result;
    result.is_nan = false;
    result.is_inf = false;

    if (mantissa_sum == 0) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_zero = false;

    // Extract sign and get absolute value
    if (mantissa_sum < 0) {
        result.sign = -1;
        mantissa_sum = -mantissa_sum;
    } else {
        result.sign = 1;
    }

    // Shift from G-bit to F-bit radix
    constexpr int SHIFT = F - G;

    if constexpr (SHIFT >= 0) {
        result.significand = static_cast<int64_t>(mantissa_sum) << SHIFT;
    } else {
        result.significand = static_cast<int64_t>(mantissa_sum) >> (-SHIFT);
    }
    result.exponent = max_exp;

    return result;
}

// ============================================================================
// Fused CoFDA dot-accumulate primitive (register-streamed, no Product[] arrays)
// ============================================================================
// c += dot(A_chunk, B_chunk) over CHUNK_SIZE FP8 (E4M3) values, with F-bit
// CoFDA intermediate rounding. Two passes: Pass 1 scans NaN/Inf + max_exp;
// Pass 2 aligns + sums an int64 mantissa, normalized via fixed_to_fp32<F>.
// Products are streamed through registers and recomputed in Pass 2 from the
// pre-decoded operands. Caller guarantees CHUNK_SIZE valid (zero-padded) elems.
// Bitwise-equivalent to fp8_multiply<F> + fda_accumulate_chunk<F, CHUNK_SIZE>.
template <int F, int CHUNK_SIZE>
[[nodiscard]] __device__ __forceinline__ float
fp8_cofda_mma(DecodedFrag a_frag, DecodedFrag b_frag, float c) {
    Operand c_operand = fp32_to_operand<F>(c);
    if (c_operand.is_nan) {
        return bits_to_fp32(fp32::QNAN_BITS);
    }

    // ---- Pass 1: NaN/Inf scan + max_exp ----
    bool any_nan = false;
    bool any_inf = false;
    int pos_inf_count = 0;
    int neg_inf_count = 0;
    int max_exp = -9999;
    int non_zero_count = 0;

    if (c_operand.is_inf) {
        any_inf = true;
        if (c_operand.sign > 0) pos_inf_count++;
        else neg_inf_count++;
    }
    if (!c_operand.is_zero && !c_operand.is_nan && !c_operand.is_inf) {
        max_exp = c_operand.exponent;
        non_zero_count++;
    }

    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; i++) {
        Product p = fp8_multiply_predecoded<F>(a_frag[i], b_frag[i]);
        if (p.is_nan) any_nan = true;
        if (p.is_inf) {
            any_inf = true;
            if (p.sign > 0) pos_inf_count++;
            else neg_inf_count++;
        }
        if (!p.is_zero && !p.is_nan && !p.is_inf) {
            max_exp = max(max_exp, p.exponent);
            non_zero_count++;
        }
    }

    if (any_nan) {
        return bits_to_fp32(fp32::QNAN_BITS);
    }
    if (any_inf) {
        if (pos_inf_count > 0 && neg_inf_count > 0) {
            return bits_to_fp32(fp32::QNAN_BITS);
        }
        return bits_to_fp32(pos_inf_count > 0 ? fp32::POS_INF_BITS
                                              : fp32::NEG_INF_BITS);
    }
    if (non_zero_count == 0) {
        return c;
    }

    // ---- Pass 2: align + sum (recompute products in-place) ----
    int64_t mantissa_sum = 0;
    if (!c_operand.is_zero && !c_operand.is_inf) {
        int exp_diff = max_exp - c_operand.exponent;
        int64_t aligned = (exp_diff >= 64) ? 0 : (c_operand.significand >> exp_diff);
        mantissa_sum += c_operand.sign * aligned;
    }

    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; i++) {
        Product p = fp8_multiply_predecoded<F>(a_frag[i], b_frag[i]);
        if (p.is_zero || p.is_nan || p.is_inf) {
            continue;
        }
        int exp_diff = max_exp - p.exponent;
        int64_t aligned =
            (exp_diff >= 64) ? 0
                             : (static_cast<int64_t>(p.significand) >> exp_diff);
        mantissa_sum += p.sign * aligned;
    }

    return fixed_to_fp32<F>(mantissa_sum, max_exp);
}

}  // namespace mma_emu
}  // namespace vllm
