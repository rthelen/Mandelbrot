#include <algorithm>
#include <bit>
#include "debug.h"
#include "ieee754.h"
#include "karatsuba128.h"

extern "C" {
    #include "knuth_div.h"
}

using fp = ieee754_128;
using fp_sign_t = ieee754_sign;
using fp_exp_t = fp::exponent_t;
using fp_storage_t = fp::storage_t;

uint128_t make_uint128(uint64_t hi, uint64_t lo)
{
    return (static_cast<uint128_t>(hi) << 64) | lo;
}

template<>
fp fp::Pi()
{
    // Wolfram says Pi is:
    // 3.243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b547..._16
    return fp(0, 0x4000, make_uint128(0x1'243f'6a88'85a3, 0x08d3'1319'8a2e'0370) >> 1);
}

/*
 * from_fp64()
 *
 * From an ieee754 64 bit (double), construct an fp128.
 * The caller has already handled doubles that were infinities, nans, zeros and denormals.
 * We're left with real numbers that are valid to translate (denormals are 'valid' but tricky).
 * All we need to do is ensure the exponent is within range (for an fp32) and then shift the
 * mantissa if it is.
 */
template <>
fp::storage_t fp::from_fp64(ieee754_sign sign, int iexp, uint64_t mantissa)
{
    // All exponent values from a double are valid with an fp128.
    uint32_t exp = static_cast<uint32_t>(iexp) + fp::exp_bias;
    return make_bits(sign, exp, fp::shift_left(mantissa, (112 - 52)));
}

static void debugAdd_128() {}

std::pair<fp, bool> add_internal(const fp& a, const fp& b)
{
    debugAdd_128();

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

static void debugDivide_128() {}

std::pair<fp, fp_sign_t> divide_internal(const fp& lhs, const fp& rhs, bool force_mantissa)
{
    debugDivide_128();

    const int N = 4;
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
    u[M - 4] = static_cast<unsigned>(lhs_mantissa);
    u[M - 3] = static_cast<unsigned>(lhs_mantissa >> 32);
    u[M - 2] = static_cast<unsigned>(lhs_mantissa >> 64);
    u[M - 1] = !force_mantissa ? static_cast<unsigned>(lhs_mantissa >> 96) : 0x8000'0000;
    v[N - 4] = static_cast<unsigned>(rhs_mantissa);
    v[N - 3] = static_cast<unsigned>(rhs_mantissa >> 32);
    v[N - 2] = static_cast<unsigned>(rhs_mantissa >> 64);
    v[N - 1] = !force_mantissa ? static_cast<unsigned>(rhs_mantissa >> 96) : 0x8000'0000;

    ASSERT(u[M -1] >> 31, "The high bit must be set");
    ASSERT(v[N -1] >> 31, "The high bit must be set");
    [[maybe_unused]] int status = divmnu(q, r, u, v, M, N);
    ASSERT(status == 0, "ERROR: Knuth's algorithm returned a non-zero status indicating complete failure");
    ASSERT(((q[Q -1] == 0) && (q[Q -2] >> 31) == 1) || (q[Q -1] == 1), "The result must have 'the' high bit set");

    fp_exp_t ret_exp {lhs_exp.minus(rhs_exp)};
    fp_sign_t ret_sign = lhs.sign() * rhs.sign();

    fp_storage_t ret_hi = (static_cast<uint128_t>(q[Q -2] & UINT32_MAX) << 96) |
                          (static_cast<uint128_t>(q[Q -3] & UINT32_MAX) << 64) |
                          (static_cast<uint128_t>(q[Q -4] & UINT32_MAX) << 32) |
                          (static_cast<uint128_t>(q[Q -5] & UINT32_MAX) <<  0);
    bool factor_high_bit = q[Q -1] == 1;
    sticky_bits sticky {factor_high_bit ? static_cast<uint64_t>(ret_hi & 1) : 0};
    ret_hi = factor_high_bit ? fp::hibit | (ret_hi >> 1) : ret_hi;
    ret_exp = !factor_high_bit ? ret_exp.minus(1) : ret_exp;

    // Factoring in the remainder.  I'm not sure all bits are needed.
    // For example, what bits would a hardware implementation of division
    // have available here?
    sticky |= r[N -1] | r[N -2] | r[N -3] | r[N -4];

    fp result {fp::roundMantissa(ret_sign, ret_exp, ret_hi, sticky)};

    return std::pair<fp, fp_sign_t> (result, ret_sign);
}

static void debugMultiply_128() {}

#define K128   0
#define F128   1

fp multiply_internal(const fp& lhs, const fp& rhs)
{
    debugMultiply_128();

    auto [lhs_mantissa, lhs_exp] = lhs.getMantissaAlignedUnbiased();
    auto [rhs_mantissa, rhs_exp] = rhs.getMantissaAlignedUnbiased();

    fp_exp_t ret_exponent {lhs_exp.plus(rhs_exp)};

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
    //
    //
    //
    //    Mid1 = LH + HL
    //    Mid2 = Mid1 + LL >> 64
    //    Low  = LL + Mid1 << 64
    //    Hi   = HH + (Mid1 >> 64) + Low.carry
    //    Hi2  = Hi1 + (Mid1.carry ? 1 << 64 : 0)
    //    Hi3  = Hi2 + (Mid2.carry ? 1 << 64 : 0)

    fp_storage_t A = lhs_mantissa;
    fp_storage_t B = rhs_mantissa;
    fp_storage_t AL = A & UINT64_MAX;
    fp_storage_t AH = A >> 64;
    fp_storage_t BL = B & UINT64_MAX;
    fp_storage_t BH = B >> 64;
#if F128
    fp_storage_t LL = AL * BL;
    fp_storage_t LH = AL * BH;
    fp_storage_t HL = AH * BL;
    fp_storage_t HH = AH * BH;

    auto [Mid1, Mid1_carry] = add_with_carry(LH,   HL);
    auto [Mid2, Mid2_carry] = add_with_carry(Mid1, LL >> 64);
    auto [Low,  Low_carry]  = add_with_carry(LL,   Mid1 << 64);
    fp_storage_t carry64 = fp::shift_left(1, 64);
    fp_storage_t Hi1 = HH + (Mid1 >> 64) + Low_carry;
    fp_storage_t Hi2 = Hi1 + (Mid1_carry ? carry64 : 0);
    fp_storage_t Hi3 = Hi2 + (Mid2_carry ? carry64 : 0);

    // Extract sticky bits (lower 64 bits) before shifting
    uint64_t lolo = static_cast<uint64_t>(Low >>  0);
    uint64_t lohi = static_cast<uint64_t>(Low >> 64);
    sticky_bits S {lolo | lohi};

    // Right shift by 64 bits
    fp_storage_t ret_mantissa = Hi3;
#endif // F128

#if K128
    mul128_ieee754_result k128 = mul128_karatsuba_ieee754(static_cast<uint64_t>(AL),
                                                          static_cast<uint64_t>(AH),
                                                          static_cast<uint64_t>(BL),
                                                          static_cast<uint64_t>(BH));
#if F128
    assert(k128.hi_hi == (Hi3 >> 64));
    assert(k128.hi_lo == (Hi3 & UINT64_MAX));
    assert(k128.sticky == (lolo | lohi));
#else // not F128:
    fp_storage_t ret_mantissa = (static_cast<fp_storage_t>(k128.hi_hi) << 64) | (static_cast<fp_storage_t>(k128.hi_lo));
    sticky_bits S {k128.sticky};
#endif // F128
#endif // K128

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
