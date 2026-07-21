/*
 * FP4 E2M1 Element Arithmetic
 *
 * E2M1 element format shared by NVFP4 and MXFP4: constants,
 * packing, decomposition, decode-on-load, and unnormalized multiplication
 * with G fractional bits. Scale-factor handling lives in nvfp4_ue4m3.cuh
 * (UE4M3) and mxfp4_e8m0.cuh (E8M0).
 */

#pragma once

#include <cstdint>
#include "../core/types.cuh"
#include "../core/fp32_utils.cuh"
#include "../core/gdfs_group.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// FP4 E2M1 Format Constants
// ============================================================================

namespace fp4 {

/// Block size for NVFP4 (scale_vec::4X)
constexpr int BLOCK_SIZE = 16;

// E2M1 format constants
constexpr int EXPONENT_BITS = 2;
constexpr int MANTISSA_BITS = 1;
constexpr int EXPONENT_BIAS = 1;
constexpr int IMPLICIT_ONE = 2;           ///< 1 << MANTISSA_BITS
constexpr int PRODUCT_RADIX_BIT = 2;      ///< 2 * MANTISSA_BITS
constexpr int MAX_PRODUCT_EXP_DIFF = 4;   ///< (max_exp - min_exp) for E2M1 products
constexpr int G_LOSSLESS = 6;             ///< PRODUCT_RADIX_BIT + MAX_PRODUCT_EXP_DIFF

}  // namespace fp4

// ============================================================================
// FP4 Packing/Unpacking
// ============================================================================

/**
 * @brief Load FP4 value from shared memory tile by local indices.
 *
 * Shared memory layout: tile[local_m * bk_packed + k_local/2]
 * Each byte contains 2 FP4 values.
 *
 * @param tile Pointer to shared memory tile
 * @param local_m Local M index within tile
 * @param k_local Local K index within tile
 * @param bk_packed Packed K dimension (BK/2)
 * @return 4-bit FP4 value
 */
[[nodiscard]] __device__ __forceinline__
uint8_t load_fp4_from_tile(const uint8_t* tile, int local_m, int k_local, int bk_packed) {
    int byte_idx = local_m * bk_packed + (k_local / 2);
    int nibble_idx = k_local % 2;
    uint8_t byte = tile[byte_idx];
    return (nibble_idx == 0) ? (byte & 0x0F) : (byte >> 4);
}

// ============================================================================
// FP4 E2M1 Decomposition (GDFS STP2)
// ============================================================================

/**
 * @brief Decomposed components of an FP4 E2M1 value.
 *
 * Stores the biased exponent so that subnormals can be detected during
 * multiplication.
 * FP4 E2M1 has no NaN or infinity representations.
 */
struct FP4Components {
    int sign;              ///< Sign: +1 or -1
    int exponent;          ///< Biased exponent (0-3)
    uint32_t mantissa;     ///< Mantissa without implicit bit (0 or 1)
    bool is_zero;          ///< True if magnitude is zero
    bool is_subnormal;     ///< True if biased exponent is 0 and mantissa != 0
};

/**
 * @brief Decompose a 4-bit FP4 E2M1 value into sign, exponent, and mantissa.
 *
 * E2M1 format: [S][EE][M] — 1 sign, 2 exponent (bias=1), 1 mantissa.
 *
 * Special cases:
 * - Zero: exp=0, mant=0
 * - Subnormal: exp=0, mant=1 (value = 0.5)
 * - No NaN or infinity in E2M1
 *
 * @param val 4-bit E2M1 value (sign at bit 3, exp at bits 2:1, mant at bit 0)
 * @return FP4Components with decomposed fields (biased exponent)
 */
[[nodiscard]] __device__ __forceinline__
FP4Components decompose_fp4_e2m1(uint8_t val) {
    FP4Components result;

    result.sign = (val & 0x8) ? -1 : 1;

    uint8_t mag = val & 0x7;
    result.exponent = (mag >> fp4::MANTISSA_BITS) & 0x3;  // biased, bits[2:1]
    result.mantissa = mag & 0x1;                           // bit[0]

    // Zero: both exponent and mantissa are 0
    if (mag == 0) {
        result.is_zero = true;
        result.is_subnormal = false;
        return result;
    }

    result.is_zero = false;
    result.is_subnormal = (result.exponent == 0);
    return result;
}

// ============================================================================
// Decode-on-load FP4 layer (GDFS fast path)
// ============================================================================
// FP4 nibble decoded + subnormal-resolved ONCE at SMEM-load time. `sig` folds
// the implicit one (normal) or equals the raw mantissa (subnormal); `exp` is
// unbiased. Consumed directly by fp4_multiply_predecoded — no per-element
// decompose/subnormal branch in the inner loop. FP4 E2M1 has no NaN/Inf.
struct DecodedFP4Operand {
    int8_t sign;    // +1 / -1
    uint8_t sig;    // normal: IMPLICIT_ONE+mantissa (2/3); subnormal: mantissa (0/1)
    int8_t exp;     // unbiased exponent
    uint8_t flags;  // bit0 = is_zero
};

namespace decoded_fp4_flag {
constexpr uint8_t ZERO_BIT = 0x1;
}  // namespace decoded_fp4_flag

// Pack/unpack DecodedFP4Operand <-> one uint32 word: one LDS.32 instead of a
// byte load + decompose. Signed fields round-trip via uint8 reinterpret.
[[nodiscard]] __device__ __forceinline__ uint32_t
pack_decoded_fp4(DecodedFP4Operand d) {
    return (static_cast<uint32_t>(static_cast<uint8_t>(d.sign))) |
           (static_cast<uint32_t>(d.sig) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(d.exp)) << 16) |
           (static_cast<uint32_t>(d.flags) << 24);
}

[[nodiscard]] __device__ __forceinline__ DecodedFP4Operand
unpack_decoded_fp4(uint32_t w) {
    DecodedFP4Operand d;
    d.sign = static_cast<int8_t>(w & 0xFF);
    d.sig = static_cast<uint8_t>((w >> 8) & 0xFF);
    d.exp = static_cast<int8_t>((w >> 16) & 0xFF);
    d.flags = static_cast<uint8_t>((w >> 24) & 0xFF);
    return d;
}

// Accessor over one operand row stored as packed uint32 words.
struct DecodedFP4Frag {
    const uint32_t* __restrict__ words;
    [[nodiscard]] __device__ __forceinline__ DecodedFP4Operand operator[](
        int i) const {
        return unpack_decoded_fp4(words[i]);
    }
};

// Decode one E2M1 nibble and resolve the normal/subnormal significand and
// exponent once, at shared-memory load time.
[[nodiscard]] __device__ __forceinline__ DecodedFP4Operand
decode_fp4_operand(uint8_t nibble) {
    DecodedFP4Operand d;
    d.sign = (nibble & 0x8) ? -1 : 1;
    const uint8_t mag = nibble & 0x7;
    const int exp_field = (mag >> fp4::MANTISSA_BITS) & 0x3;  // biased
    const uint8_t mant = mag & 0x1;

    if (mag == 0) {  // zero
        d.flags = decoded_fp4_flag::ZERO_BIT;
        d.sig = 0;
        d.exp = 0;
        return d;
    }
    d.flags = 0;
    if (exp_field == 0) {  // subnormal
        d.sig = mant;
        d.exp = static_cast<int8_t>(1 - fp4::EXPONENT_BIAS);
    } else {  // normal
        d.sig = static_cast<uint8_t>(fp4::IMPLICIT_ONE + mant);
        d.exp = static_cast<int8_t>(exp_field - fp4::EXPONENT_BIAS);
    }
    return d;
}

// Multiply two pre-decoded E2M1 operands into an unnormalized Product carrying
// G fractional bits. The operands are already decoded, so no subnormal branch
// is needed here.
template <int G>
[[nodiscard]] __device__ __forceinline__ Product
fp4_multiply_predecoded(DecodedFP4Operand a, DecodedFP4Operand b) {
    Product result;
    result.is_nan = false;
    result.is_inf = false;
    result.is_subnormal = false;

    if ((a.flags & decoded_fp4_flag::ZERO_BIT) ||
        (b.flags & decoded_fp4_flag::ZERO_BIT)) {
        result.is_zero = true;
        result.sign = 1;
        result.exponent = 0;
        result.significand = 0;
        return result;
    }

    result.is_zero = false;
    result.sign = a.sign * b.sign;

    const uint64_t raw_product =
        static_cast<uint64_t>(a.sig) * static_cast<uint64_t>(b.sig);
    constexpr int SCALE_SHIFT = G - fp4::PRODUCT_RADIX_BIT;
    if constexpr (SCALE_SHIFT >= 0) {
        result.significand = raw_product << SCALE_SHIFT;
    } else {
        result.significand = raw_product >> (-SCALE_SHIFT);
    }
    result.exponent = a.exp + b.exp;

    if (result.significand == 0) {
        result.is_zero = true;
    }
    return result;
}

// ============================================================================
// FP4 Unnormalized Multiplication with G-bits (GDFS STP2)
// ============================================================================

/**
 * @brief Multiply two FP4 E2M1 values producing an unnormalized product
 *        with G fractional bits in the significand.
 *
 *
 * The raw product significand is in Q2.2 format (radix at bit 2).
 * It is scaled to G fractional bits:
 *   SCALE_SHIFT = G - PRODUCT_RADIX_BIT = G - 2
 *   if SCALE_SHIFT >= 0: significand = raw_product << SCALE_SHIFT
 *   if SCALE_SHIFT < 0:  significand = raw_product >> (-SCALE_SHIFT)
 *
 *   G=6: lossless (all product bits preserved after group alignment)
 *   G<2: right-shift truncates product bits at STP2 level
 *
 * @tparam G GDFS intra-group bits G
 * @param a First FP4 E2M1 operand (decomposed)
 * @param b Second FP4 E2M1 operand (decomposed)
 * @return Product structure ready for group accumulation
 */
template<int G>
[[nodiscard]] __device__ __forceinline__
Product fp4_multiply_unnormalized(FP4Components a, FP4Components b) {
    Product result;
    result.is_nan = false;   // E2M1 has no NaN
    result.is_inf = false;   // E2M1 has no infinity
    result.is_subnormal = false;

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

    // Compute significands with implicit bit and unbiased exponents.
    // Subnormal: sig = mantissa (no implicit bit), exp = 1 - BIAS
    // Normal:    sig = IMPLICIT_ONE + mantissa,     exp = biased_exp - BIAS
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

    // Multiply and scale to put radix at bit G.
    // Raw product is in Q2.2 format (radix at PRODUCT_RADIX_BIT = 2).
    // SCALE_SHIFT = G - 2:
    //   G=-1 → >>3,  G=0 → >>2,  G=1 → >>1,  G=2 → nop,  G=6 → <<4
    uint64_t raw_product = static_cast<uint64_t>(sig_a) * static_cast<uint64_t>(sig_b);
    constexpr int SCALE_SHIFT = G - fp4::PRODUCT_RADIX_BIT;

    if constexpr (SCALE_SHIFT >= 0) {
        result.significand = raw_product << SCALE_SHIFT;
    } else {
        result.significand = raw_product >> (-SCALE_SHIFT);
    }
    result.exponent = exp_a + exp_b;

    // Product may become zero after right-shift truncation (G < 2)
    if (result.significand == 0) {
        result.is_zero = true;
    }

    return result;
}


// ============================================================================
// Register-streamed GDFS group accumulate (no Product[GS] array)
// ============================================================================
// Scale-agnostic: returns the same GroupResult as group_accumulate<G,GS>, so
// the caller applies UE4M3 (NVFP4) or E8M0 (MXFP4) scales afterward — the
// primitive is reusable across both formats. Two passes stream products through
// registers (recomputed in Pass 2 from the pre-decoded operands). Bit-exact to
// fp4_multiply_unnormalized<G> + group_accumulate<G,GS>.
template <int G, int GS>
[[nodiscard]] __device__ __forceinline__ GroupResult
fp4_gdfs_group_accumulate_predecoded(DecodedFP4Frag a_frag, DecodedFP4Frag b_frag) {
    GroupResult gr;
    gr.mantissa_sum = 0;
    gr.max_exp = -9999;
    gr.all_zero = true;

    // Pass 1: max exponent among non-zero products
    #pragma unroll
    for (int i = 0; i < GS; i++) {
        Product p = fp4_multiply_predecoded<G>(a_frag[i], b_frag[i]);
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
        Product p = fp4_multiply_predecoded<G>(a_frag[i], b_frag[i]);
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
}  // namespace mma_emu
}  // namespace vllm
