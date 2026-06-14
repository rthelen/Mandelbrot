#include <algorithm>
#include <bit>
#include "debug.h"
#include "ieee754.h"

extern "C" {
    #include "knuth_div.h"
}

using fp = ieee754_64;
using fp_sign_t = ieee754_sign;
using fp_exp_t = fp::exponent_t;
using fp_storage_t = fp::storage_t;

template<>
fp fp::Pi()
{
    // Wolfram says Pi is:
    // 3.243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b547..._16
    return fp(0, 0x400, 0x12'43f6'a888'5a30 >> 1);
}

/*
 * from_fp64()
 *
 * Dummy function: Not called in the real flow.
 */
template <>
fp::storage_t fp::from_fp64([[maybe_unused]] ieee754_sign sign, [[maybe_unused]] int iexp, [[maybe_unused]] uint64_t mantissa)
{
    ASSERT(false, "ieee754::from_fp64 should never be called for ieee754_64");
    return 0;
}

static void debugAdd_64() {}

std::pair<fp, bool> add_internal(const fp& a, const fp& b)
{
    debugAdd_64();

    auto [a_mantissa, a_exp] = a.getMantissaAlignedUnbiased();
    auto [b_mantissa, b_exp] = b.getMantissaAlignedUnbiased();
    fp_sign_t a_neg {a.sign()};
    fp_sign_t b_neg {b.sign()};

    ASSERT(a_exp.exponent() >= b_exp.exponent(), "By this point in the code, A must be >= B");
    ASSERT(a_exp.exponent() == b_exp.exponent() ? a_mantissa >= b_mantissa : true, "By this point in the code, A must be >= B");
    auto exp_diff = a_exp.minus(b_exp);

    // This algorithm performs addition (or subtraction) in-place and thus we need a spot for the carry.
    a_mantissa >>= 1;
    b_mantissa >>= 1;

    b_mantissa = fp::shift_track_sticky_bits(b_mantissa, exp_diff.exponent());

    // If the signs of A and B are equal, then we add.
    // Else we subtract.
    //     2 +  3 =  5   //  2 + 3 -> 5
    //    -2 + -3 = -5   //  2 + 3 -> 5
    //     2 + -3 = -1   //  3 - 2 -> 1
    //    -2 +  3 =  1   //  3 - 2 -> 1
    bool add_op = a_neg == b_neg;

    fp_storage_t ret_add = a_mantissa + b_mantissa;
    fp_storage_t ret_sub = a_mantissa - b_mantissa;
    fp_storage_t ret_mantissa = add_op ? ret_add : ret_sub;
    bool zero_result = ret_mantissa == 0;
    int clz = ret_mantissa > 0 ? std::countl_zero(ret_mantissa) : -1;
    a_exp.increment(clz == 0);
    a_exp.decrease(clz >  1 ? clz -1 : 0);
    ret_mantissa <<= ret_mantissa > 0 ? clz : 0;
    ASSERT(ret_mantissa >> fp::sign_shift || ret_mantissa == 0, "The mantissa must either have the high bit set or be 0");

    fp result = fp::roundMantissa(a_neg, a_exp, ret_mantissa, 0);
    return std::pair<fp, bool> (result, zero_result);
}

static void debugDivide_64() {}

std::pair<fp, fp_sign_t> divide_internal(const fp& lhs, const fp& rhs, bool force_mantissa)
{
    debugDivide_64();

    const int N = 2;
    const int M = 2 * N;
    const int Q = M - N +1;

    unsigned q[Q] {0}, r[N] {0}, u[M] {0}, v[N] {0};
    bzero(q, sizeof(q));
    bzero(r, sizeof(r));
    bzero(u, sizeof(u));
    bzero(v, sizeof(v));

    auto [lhs_mantissa, lhs_exp] = lhs.getMantissaAlignedUnbiased();
    auto [rhs_mantissa, rhs_exp] = rhs.getMantissaAlignedUnbiased();

    // We factor in a dummy high bit set if we already know we're going to
    // return a zero or a nan.
    u[M - 2] = static_cast<unsigned>(lhs_mantissa);
    u[M - 1] = !force_mantissa ? static_cast<unsigned>(lhs_mantissa >> 32) : 0x8000'0000;
    v[N - 2] = static_cast<unsigned>(rhs_mantissa);
    v[N - 1] = !force_mantissa ? static_cast<unsigned>(rhs_mantissa >> 32) : 0x8000'0000;

    ASSERT(u[M -1] >> 31, "The high bit must be set");
    ASSERT(v[N -1] >> 31, "The high bit must be set");
    [[maybe_unused]] int status = divmnu(q, r, u, v, M, N);
    ASSERT(status == 0, "ERROR: Knuth's algorithm returned a non-zero status indicating complete failure");

    fp_storage_t q_hi = q[Q -1];
    fp_storage_t q_mi = q[Q -2];
    fp_storage_t q_lo = q[Q -3];
    ASSERT(((q_hi == 0) && (q_mi >> 31) == 1) || (q_hi == 1), "The result must have 'the' high bit set");

    fp_exp_t ret_exp {lhs_exp.minus(rhs_exp)};
    fp_sign_t ret_sign = lhs.sign() * rhs.sign();

    fp_storage_t ret_hi = (q_mi << 32) | (q_lo & UINT32_MAX);
    bool factor_high_bit = q_hi == 1;
    sticky_bits sticky {factor_high_bit ? (ret_hi & 1) : 0};
    ret_hi = factor_high_bit ? fp::hibit | (ret_hi >> 1) : ret_hi;
    ret_exp = !factor_high_bit ? ret_exp.minus(1) : ret_exp;

    // Factoring in the remainder.  I'm not sure all bits are needed.
    // For example, what bits would a hardware implementation of division
    // have available here?
    sticky |= r[N -1] | r[N -2];

    fp result {fp::roundMantissa(ret_sign, ret_exp, ret_hi, sticky)};

    return std::pair<fp, fp_sign_t> (result, ret_sign);
}

static void debugMultiply_64() {}

fp multiply_internal(const fp& lhs, const fp& rhs)
{
#ifdef DEBUG
    const bool verbose = false;
#endif
    debugMultiply_64();

    auto [lhs_mantissa, lhs_exp] = lhs.getMantissaAlignedUnbiased();
    auto [rhs_mantissa, rhs_exp] = rhs.getMantissaAlignedUnbiased();

    fp_exp_t ret_exponent {lhs_exp.plus(rhs_exp)};

#ifndef FAST
    // Multiply A x B, A = lhs, B = rhs
    // Imagine that A is made up of 32bits AH and 32bits of AL := [AH : AL]
    // Imagine that B is similar.
    //
    //                    AH   AL
    //                    BH   BL
    //                  -----------
    //                    AL x BL       : LL
    //               AL x BH            : LH
    //               AH x BL            : HL
    //          AH x BH                 : HH
    //       -----------------------

    uint64_t A = lhs_mantissa;
    uint64_t B = rhs_mantissa;
    uint64_t AL = A & UINT32_MAX;
    uint64_t AH = A >> 32;
    uint64_t BL = B & UINT32_MAX;
    uint64_t BH = B >> 32;
    uint64_t LL = AL * BL;
    uint64_t LH = AL * BH;
    uint64_t HL = AH * BL;
    uint64_t HH = AH * BH;

    // Add the middle products (LH + HL) with carry tracking
    auto [MM_a, carry1] = add_with_carry(LH, HL);

    // Add the upper 32 bits of LL to MM_a
    auto [MM, carry2] = add_with_carry(MM_a, LL >> 32);

    // Total carry from middle additions (goes to bit 96 of the 128-bit result)
    uint64_t MM_C = carry1 + carry2;

    // Build the low 64 bits: (MM << 32) + (LL & 0xFFFFFFFF)
    auto [ret_mantissa_low, low_carry] = add_with_carry((MM << 32), (LL & UINT32_MAX));

    // Build the high 64 bits: HH + (MM >> 32) + carries
    auto [ret_mantissa_high_partial, high_carry1] = add_with_carry(HH, (MM >> 32));
    auto [ret_mantissa_high, high_carry2] = add_with_carry(ret_mantissa_high_partial,
                                                            (MM_C << 32) + low_carry);
#else
    uint128_t ret128 = static_cast<uint128_t>(lhs_mantissa) * static_cast<uint128_t>(rhs_mantissa);
    uint64_t ret_mantissa_high = static_cast<uint64_t>(ret128 >> 64);
    uint64_t ret_mantissa_low = static_cast<uint64_t>(ret128);
#endif

#ifdef DEBUG
    // Verify against 128-bit reference implementation
    uint128_t ret_optimized = static_cast<uint128_t>(lhs_mantissa) * static_cast<uint128_t>(rhs_mantissa);
    uint64_t ref_high = static_cast<uint64_t>(ret_optimized >> 64);
    uint64_t ref_low = static_cast<uint64_t>(ret_optimized);
    if (ret_mantissa_high != ref_high || ret_mantissa_low != ref_low) {
        printf("  * Reference = %16.16llx.%16.16llx\n", ref_high, ref_low);
        printf("  * Mantissa  = %16.16llx.%16.16llx\n", ret_mantissa_high, ret_mantissa_low);
    }
    ASSERT(ret_mantissa_high == ref_high && ret_mantissa_low == ref_low, "Debug check of a 128 multiply failed to match the logical compuation");
#endif

    VDBG_PRINT("  * Mantissa = %16.16llx.%16.16llx\n", ret_mantissa_high, ret_mantissa_low);

    // Extract sticky bits (lower 64 bits) before shifting
    sticky_bits S {ret_mantissa_low};

    // Right shift by 64 bits
    fp_storage_t ret_mantissa = ret_mantissa_high;

    VDBG_PRINT("  * Mantissa = %16.16llx (after >>42)\n", ret_mantissa);

    // When two numbers are multiplied, they may or may not have
    // a carry into the highest bit.  This captures that carry.
    // And, this is why at the end we only shift right 10 bits
    // instead of 11: The result value is already right shifted
    // one bit. The alternative logic here would be to left shift
    // So that the high bit is always a 1.  I think the latter is what
    // the dividor does.
    bool mult_had_carry = (ret_mantissa >> fp::sign_shift) ? true : false;
    ret_exponent.increment(mult_had_carry);
    ret_mantissa <<= mult_had_carry ? 0 : 1;

    // At this point, either the highest bit must be 1 or the whole value
    // must be 0.  Those are the only options for all possible input numbers.
    // Even with denormals; because we shifted the denormal into the
    // correct position with the higest order bit being set (unless the whole
    // value is 0 ... which leads to the second case.)
    ASSERT((ret_mantissa >> fp::sign_shift) || (ret_mantissa == 0), "The mantissa must have either the high bit set or be 0");

    // Assemble a result, or zero.
    fp_sign_t ret_sign = lhs.sign() * rhs.sign();
    fp mul = fp::roundMantissa(ret_sign, ret_exponent, ret_mantissa, S);

    return mul;
}
