/*
 * karatsuba128.cpp
 *
 * Implementation of 128×128 multiplication using two-level Karatsuba algorithm.
 * Only uses 32×32 bit multiplications for GPU portability.
 */

#include <cassert>
#include "karatsuba128.h"

/*
 * Helper: 64×64 → 128-bit multiplication using Karatsuba with 32-bit base.
 * Uses exactly 3 32×32 multiplications.
 *
 * Algorithm:
 *   X = XH·2^32 + XL
 *   Y = YH·2^32 + YL
 *
 *   Q1 = XL × YL                          (32×32 → 64)
 *   Q2 = XH × YH                          (32×32 → 64)
 *   Q3 = (XL+XH) × (YL+YH)                (33×33, handled below)
 *
 *   Middle = Q3 - Q1 - Q2
 *   Result = Q2·2^64 + Middle·2^32 + Q1
 *
 * The 33×33 case:
 *   (XL+XH) may overflow to 33 bits: cx·2^32 + Sx where cx ∈ {0,1}
 *   (YL+YH) may overflow to 33 bits: cy·2^32 + Sy where cy ∈ {0,1}
 *
 *   Q3 = (cx·2^32 + Sx)(cy·2^32 + Sy)
 *      = Sx·Sy + (cx·Sy + cy·Sx)·2^32 + cx·cy·2^64
 *
 *   The cx·Sy and cy·Sx terms are conditional additions (not multiplies).
 */
static inline void mul64_karatsuba(uint32_t x_lo, uint32_t x_hi,
                                   uint32_t y_lo, uint32_t y_hi,
                                   uint64_t* result_lo, uint64_t* result_hi)
{
    // Q1 = XL × YL (multiply #1)
    uint64_t q1 = static_cast<uint64_t>(x_lo) * y_lo;

    // Q2 = XH × YH (multiply #2)
    uint64_t q2 = static_cast<uint64_t>(x_hi) * y_hi;

    // Compute sums with carry detection (branchless)
    uint32_t sum_x = x_lo + x_hi;
    uint32_t sum_y = y_lo + y_hi;
    uint32_t carry_x = sum_x < x_lo ? 1 : 0;  // 1 if overflow, 0 otherwise
    uint32_t carry_y = sum_y < y_lo ? 1 : 0;

    // Q3 core = Sx × Sy (multiply #3)
    uint64_t q3_core = static_cast<uint64_t>(sum_x) * sum_y;

    // Handle 33×33 overflow terms (no multiplies, just conditional adds)
    // (cx·2^32 + Sx)(cy·2^32 + Sy) = Sx·Sy + (cx·Sy + cy·Sx)·2^32 + cx·cy·2^64
    //
    // The term (cx·Sy + cy·Sx) can be up to 33 bits (when both cx=cy=1 and Sx=Sy=max).
    // We need to track both the low 32 bits (shifted to position 32) and high bits (position 64+).
    // Cast to uint64_t before adding to avoid 32-bit overflow
    uint64_t carry_sum = (carry_x ? static_cast<uint64_t>(sum_y) : 0) +
                         (carry_y ? static_cast<uint64_t>(sum_x) : 0);
    uint64_t q3_carry_contribution_lo = (carry_sum & UINT32_MAX) << 32;
    uint64_t q3_carry_contribution_hi = (carry_sum >> 32) + (carry_x & carry_y);

    // Full Q3 = q3_core + carry_contribution_lo + carry_contribution_hi·2^64
    uint64_t q3_lo = q3_core + q3_carry_contribution_lo;
    uint64_t q3_hi = q3_carry_contribution_hi + (q3_lo < q3_core ? 1 : 0);

    // Middle = Q3 - Q1 - Q2 (subtract both at position 0)
    // Step 1: tmp = Q3 - Q1
    uint64_t tmp_lo = q3_lo - q1;
    uint64_t borrow1 = tmp_lo > q3_lo ? 1 : 0;
    uint64_t tmp_hi = q3_hi - borrow1;

    // Step 2: Middle = tmp - Q2
    uint64_t mid_lo = tmp_lo - q2;
    uint64_t borrow2 = mid_lo > tmp_lo ? 1 : 0;
    uint64_t mid_hi = tmp_hi - borrow2;

    // Result = Q2·2^64 + Middle·2^32 + Q1
    // Position breakdown:
    //   Bits 0-31:   Q1[0:31]
    //   Bits 32-63:  Q1[32:63] + Middle[0:31]
    //   Bits 64-95:  Q2[0:31] + Middle[32:63] + carry
    //   Bits 96-127: Q2[32:63] + Middle[64:95] + carry

    uint64_t pos0_31 = q1 & UINT32_MAX;
    uint64_t pos32_63 = (q1 >> 32) + (mid_lo & UINT32_MAX);
    uint64_t carry_to_64 = pos32_63 >> 32;
    pos32_63 &= UINT32_MAX;

    uint64_t pos64_95 = (q2 & UINT32_MAX) + (mid_lo >> 32) + carry_to_64;
    uint64_t carry_to_96 = pos64_95 >> 32;
    pos64_95 &= UINT32_MAX;

    uint64_t pos96_127 = (q2 >> 32) + mid_hi + carry_to_96;

    *result_lo = pos0_31 | (pos32_63 << 32);
    *result_hi = pos64_95 | (pos96_127 << 32);

#if KARATSUBA_DEBUG
    // Verify against native 128-bit multiplication
    __uint128_t x = (static_cast<__uint128_t>(x_hi) << 32) | x_lo;
    __uint128_t y = (static_cast<__uint128_t>(y_hi) << 32) | y_lo;
    __uint128_t expected = x * y;
    uint64_t exp_lo = static_cast<uint64_t>(expected);
    uint64_t exp_hi = static_cast<uint64_t>(expected >> 64);
    if (*result_lo != exp_lo || *result_hi != exp_hi) {
        printf("mul64_karatsuba MISMATCH!\n");
        printf("  x=0x%08x_%08x, y=0x%08x_%08x\n", x_hi, x_lo, y_hi, y_lo);
        printf("  expected: 0x%016llx_%016llx\n",
               static_cast<unsigned long long>(exp_hi), static_cast<unsigned long long>(exp_lo));
        printf("  got:      0x%016llx_%016llx\n",
               static_cast<unsigned long long>(*result_hi), static_cast<unsigned long long>(*result_lo));
    }
#endif
}

/*
 * 128×128 → 256-bit multiplication using Karatsuba.
 *
 * Algorithm:
 *   A = AH·2^64 + AL
 *   B = BH·2^64 + BL
 *
 *   P1 = AL × BL                          (64×64 → 128, uses 3 muls)
 *   P2 = AH × BH                          (64×64 → 128, uses 3 muls)
 *   P3 = (AL+AH) × (BL+BH)                (65×65, uses 3+1 muls)
 *
 *   Middle = P3 - P1 - P2
 *   Result = P2·2^128 + Middle·2^64 + P1
 *
 * Total: 10 32×32 multiplications
 */
uint256_t mul128_karatsuba(uint64_t a_lo, uint64_t a_hi,
                           uint64_t b_lo, uint64_t b_hi)
{
    uint256_t result;

    // Decompose into 32-bit words
    uint32_t a0 = static_cast<uint32_t>(a_lo);
    uint32_t a1 = static_cast<uint32_t>(a_lo >> 32);
    uint32_t a2 = static_cast<uint32_t>(a_hi);
    uint32_t a3 = static_cast<uint32_t>(a_hi >> 32);

    uint32_t b0 = static_cast<uint32_t>(b_lo);
    uint32_t b1 = static_cast<uint32_t>(b_lo >> 32);
    uint32_t b2 = static_cast<uint32_t>(b_hi);
    uint32_t b3 = static_cast<uint32_t>(b_hi >> 32);

    // P1 = AL × BL where AL = (a1:a0), BL = (b1:b0)
    // Uses 3 multiplies
    uint64_t p1_lo, p1_hi;
    mul64_karatsuba(a0, a1, b0, b1, &p1_lo, &p1_hi);

    // P2 = AH × BH where AH = (a3:a2), BH = (b3:b2)
    // Uses 3 multiplies
    uint64_t p2_lo, p2_hi;
    mul64_karatsuba(a2, a3, b2, b3, &p2_lo, &p2_hi);

    // Compute (AL + AH) and (BL + BH) with carry tracking
    // AL + AH = (a1:a0) + (a3:a2) → 65-bit result (sa1:sa0) + carry_a
    uint64_t sa_lo = a_lo + a_hi;
    uint32_t carry_a = sa_lo < a_lo;  // 65th bit

    uint64_t sb_lo = b_lo + b_hi;
    uint32_t carry_b = sb_lo < b_lo;  // 65th bit

    // Decompose sums into 32-bit for Karatsuba
    uint32_t sa0 = static_cast<uint32_t>(sa_lo);
    uint32_t sa1 = static_cast<uint32_t>(sa_lo >> 32);
    uint32_t sb0 = static_cast<uint32_t>(sb_lo);
    uint32_t sb1 = static_cast<uint32_t>(sb_lo >> 32);

    // P3 core = Sa × Sb (64×64 → 128, uses 3 multiplies)
    uint64_t p3_lo, p3_hi;
    mul64_karatsuba(sa0, sa1, sb0, sb1, &p3_lo, &p3_hi);

    // Handle 65×65 overflow (the carry bits)
    // P3_full = Sa·Sb + (carry_a·Sb + carry_b·Sa)·2^64 + carry_a·carry_b·2^128
    //
    // carry_a·Sb: if carry_a=1, add Sb (64-bit) at position 64
    // carry_b·Sa: if carry_b=1, add Sa (64-bit) at position 64
    // carry_a·carry_b: if both=1, add 1 at position 128
    //
    // Note: (carry_a·Sb + carry_b·Sa) can overflow 64 bits when both carries
    // are 1 and both sums are large. We must track this overflow.

    uint64_t term_a = carry_a ? sb_lo : 0;
    uint64_t term_b = carry_b ? sa_lo : 0;
    uint64_t carry_term = term_a + term_b;
    uint64_t carry_term_overflow = (carry_term < term_a) ? 1 : 0;  // did the addition overflow?

    // Add carry terms to P3
    // P3 = (p3_hi:p3_lo) at positions 0-127
    // Need to add carry_term at position 64, (carry_a&carry_b + carry_term_overflow) at position 128
    uint64_t p3_w1 = p3_hi + carry_term;
    uint64_t p3_w1_carry = (p3_w1 < p3_hi) ? 1 : 0;
    uint64_t p3_w2 = (carry_a & carry_b) + carry_term_overflow + p3_w1_carry;

    // Now P3 is conceptually (p3_w2 : p3_w1 : p3_lo) but we only have 192 bits max
    // Actually P3 can be up to 130 bits (65+65), so we need:
    // p3_lo = bits 0-63
    // p3_w1 = bits 64-127
    // p3_w2 = bits 128-129 (at most 2 bits)

    // Middle = P3 - P1 - P2 (all are 128-bit values, P3 can be 130 bits)
    // This is a multi-precision subtraction at position 0.

    // Step 1: tmp = P3 - P1
    uint64_t tmp_lo = p3_lo - p1_lo;
    uint64_t borrow1 = tmp_lo > p3_lo ? 1 : 0;

    uint64_t tmp_hi = p3_w1 - p1_hi - borrow1;
    uint64_t borrow2 = (p3_w1 < p1_hi) || (p3_w1 == p1_hi && borrow1) ? 1 : 0;

    uint64_t tmp_top = p3_w2 - borrow2;

    // Step 2: Middle = tmp - P2
    uint64_t mid_lo = tmp_lo - p2_lo;
    uint64_t borrow3 = mid_lo > tmp_lo ? 1 : 0;

    uint64_t mid_hi = tmp_hi - p2_hi - borrow3;
    uint64_t borrow4 = (tmp_hi < p2_hi) || (tmp_hi == p2_hi && borrow3) ? 1 : 0;

    uint64_t mid_top = tmp_top - borrow4;

    // Result = P2·2^128 + Middle·2^64 + P1
    //
    // Position mapping:
    //   w[0] = bits 0-63:    P1_lo
    //   w[1] = bits 64-127:  P1_hi + Middle_lo
    //   w[2] = bits 128-191: P2_lo + Middle_hi
    //   w[3] = bits 192-255: P2_hi + Middle_top

    result.w[0] = p1_lo;

    result.w[1] = p1_hi + mid_lo;
    uint64_t carry1 = result.w[1] < p1_hi ? 1 : 0;

    // Three-operand addition: p2_lo + mid_hi + carry1
    // Must track carry properly: first add two, then add third
    uint64_t sum2_partial = p2_lo + mid_hi;
    uint64_t carry2_partial = sum2_partial < p2_lo ? 1 : 0;
    result.w[2] = sum2_partial + carry1;
    uint64_t carry2 = carry2_partial + (result.w[2] < sum2_partial ? 1 : 0);

    // Three-operand addition: p2_hi + mid_top + carry2
    uint64_t sum3_partial = p2_hi + mid_top;
    assert(sum3_partial >= p2_hi);
    uint64_t carry3_partial = sum3_partial < p2_hi ? 1 : 0;
    result.w[3] = sum3_partial + carry2;
    // Note: carry3 would go into w[4] which doesn't exist, but for valid
    // 128x128 products this shouldn't overflow
    assert(carry3_partial == 0);

    return result;
}

/*
 * IEEE 754 optimized version.
 * Returns high 128 bits and sticky bit (OR of low 128 bits).
 */
mul128_ieee754_result mul128_karatsuba_ieee754(uint64_t a_lo, uint64_t a_hi,
                                                uint64_t b_lo, uint64_t b_hi)
{
    uint256_t full = mul128_karatsuba(a_lo, a_hi, b_lo, b_hi);

    mul128_ieee754_result result;
    result.hi_lo = full.w[2];
    result.hi_hi = full.w[3];
    result.sticky = full.w[0] | full.w[1];

    return result;
}

/*
 * Naive 128×128 multiplication for comparison/testing.
 * Uses 16 32×32 multiplications (schoolbook algorithm).
 */
uint256_t mul128_naive(uint64_t a_lo, uint64_t a_hi,
                       uint64_t b_lo, uint64_t b_hi)
{
    // Decompose into 32-bit words
    uint32_t a[4] = {
        static_cast<uint32_t>(a_lo),
        static_cast<uint32_t>(a_lo >> 32),
        static_cast<uint32_t>(a_hi),
        static_cast<uint32_t>(a_hi >> 32)
    };

    uint32_t b[4] = {
        static_cast<uint32_t>(b_lo),
        static_cast<uint32_t>(b_lo >> 32),
        static_cast<uint32_t>(b_hi),
        static_cast<uint32_t>(b_hi >> 32)
    };

    // Accumulate into 8 32-bit words (256 bits)
    uint64_t acc[8] = {0};

    // Schoolbook multiplication: 16 products
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint64_t product = static_cast<uint64_t>(a[i]) * b[j];
            acc[i + j] += product & UINT32_MAX;
            acc[i + j + 1] += product >> 32;
        }
    }

    // Propagate carries
    for (int i = 0; i < 7; i++) {
        acc[i + 1] += acc[i] >> 32;
        acc[i] &= UINT32_MAX;
    }

    // Pack into result
    uint256_t result;
    result.w[0] = acc[0] | (acc[1] << 32);
    result.w[1] = acc[2] | (acc[3] << 32);
    result.w[2] = acc[4] | (acc[5] << 32);
    result.w[3] = acc[6] | (acc[7] << 32);

    return result;
}
