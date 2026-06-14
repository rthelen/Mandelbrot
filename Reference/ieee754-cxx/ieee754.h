#pragma once

#include "ieee754_util.h"

/*
 *  NOT using ieee754_8   = ieee754<uint8_t,   uint8_t,   3>;
 *  NOT using ieee754_16  = ieee754<uint16_t, _Float16,  10>;
 *      using ieee754_32  = ieee754<uint32_t,    float,  23>;
 *      using ieee754_64  = ieee754<uint64_t,   double,  52>;
 *      using ieee754_128 = ieee754<uint128_t,  double, 113>;
 */

// It seems desirable to change the template to just include the following:
// template<typename _storage_t, int _mantissa_len>
// And then derive everything from there.
// _fp_t could always be replaced with 'double'.
// _sign_shift is always sizeof(storage_t) * 8 -1.
// _exp_max and _exp_bias are side effects of sizeof(storage_t) * 8 -1 - mantissa_len.
// Ultimately, replacing / eliminating storage_t to simply be a number of uint32's / uint64's
//    which itself (the number) could be derived from mantissa_len assuming that an exponent is < 32 bits.
// The use of an array of uint32's or uint64's is good for a couple of things:
//    1. For GPU's which don't support uint128.
//    2. For CPU's when supporting fp256.
template<typename _storage_t, typename _fp_t, int _mantissa_width>
class ieee754 {
public:
    using storage_t = _storage_t;
    using fp_t = _fp_t;

private:
    storage_t _mantissa_bits;

public:
    static const int sign_shift = sizeof(storage_t) * 8 -1;
    static const int exp_shift = _mantissa_width;
    static const int exp_width = sign_shift - _mantissa_width;
    static const int mantissa_width = exp_shift;
    static const uint32_t exp_max = (1 << exp_width) -1;
    static const uint32_t exp_bias = exp_max >> 1;

public:
    using exponent_t = ieee754_exponent<exp_max, exp_bias, storage_t>;

    static_assert(sign_shift == sizeof(storage_t) * 8 -1, "Sign bit must be MSB");
    static_assert(exp_shift + std::popcount(exp_max) == sign_shift, "Exponent sizing / positioning error");
    static_assert((sizeof(storage_t) == 1) || (sizeof(storage_t) > 8) || (sizeof(storage_t) == sizeof(fp_t)), "The unsigned integer representation and the HW Floating Point type must be the same size");

private:
    static constexpr inline storage_t shift_left(storage_t _bits, int shift) {
        return static_cast<storage_t>(_bits << shift);
    }
    static constexpr inline storage_t shift_right(storage_t _bits, int shift) {
        return static_cast<storage_t>(_bits >> shift);
    }
    static constexpr inline storage_t make_mask(int shift) {
        return shift_left(1ULL, shift) -1;
    }

    static const storage_t implied_bit = shift_left(1, mantissa_width);
    static const storage_t hibit = shift_left(1, sign_shift);

    // Hopefully these are true regardless of the width
    static const storage_t sign_mask = hibit;
    static const storage_t exp_mask =  shift_left(exp_max, exp_shift);
    static const int rounding_shift = exp_width -1;
    static const int rounding_width = exp_width -2;
    static const uint32_t rounding_mask = 1 << rounding_width;
    static const uint32_t sticky_mask = rounding_mask -1;

public:
    static const storage_t mantissa_mask = implied_bit -1;

private:
    storage_t to_sign_bit(uint8_t sign) { return shift_left(sign & 1, sign_shift); }
    storage_t to_exp_bits(uint32_t exp) { return shift_left(exp & exp_max, exp_shift); }
    storage_t to_mantissa_bits(storage_t mantissa) { return mantissa & mantissa_mask; }

    /*
    * shift_track_sticky_bits()
    *
    * Shift a number right by some number of bits.
    * But, if non-zero bits are shifted out (off the right),
    * then or in a single one bit into the result.
    *
    * This captures the shifted bits (by naively rounding up).
    * Note that this bit will be in the "sticky bits" position
    * of an add / subtract / multiply.
    */
    static storage_t shift_track_sticky_bits(storage_t n, int shift)
    {
        storage_t ret {0};

        if (shift == 0) {
            ret = n;
        } else if (shift > sign_shift) {
            ret = n ? 1 : 0;
        } else {
            // Shifting some bits out.  Look at those bits and or in a 1
            // if any of those bits are not 0.
            storage_t hi = shift_right(n, shift);
            storage_t lo = n & make_mask(shift);
            ret = hi | (lo ? 1 : 0);
        }

        return ret;
    }

    inline storage_t getFullMantissa() const {
        return exp_bits() != 0 ? implied_bit | mantissa_bits() : shift_left(mantissa_bits(), 1);
    }

    friend inline bool is_unsigned_lessthan(const ieee754& lhs, const ieee754& rhs)
    {
        return (lhs.exp_bits() < rhs.exp_bits()) ||
            ((lhs.exp_bits() == rhs.exp_bits()) && (lhs.getFullMantissa() < rhs.getFullMantissa()));
    }

    // getMantissaAligned() returns mantissa in bits [63..11] of uint64_t, left-justified
    // For denormals, shifts left and adjusts exponent accordingly
    // Returns: {aligned_mantissa, signed_biased_exponent}
    // std::pair<uint64_t, int> getMantissaAligned() const;
    // getMantissaAlignedUnbiased() same as above but returns unbiased exponent
    std::pair<storage_t, exponent_t > getMantissaAlignedUnbiased() const
    {
        storage_t mantissa = shift_left(getFullMantissa(), exp_width);
        exponent_t exponent {exp_bits()};

        // For denormal numbers, left-justify and adjust exponent
        if (exponent.as_bits_is_zero() && mantissa != 0) {
            int leading_zeros = std::countl_zero(mantissa);
            mantissa = shift_left(mantissa, leading_zeros);
            exponent.decrease(leading_zeros);
        }
        if (exponent.as_bits_is_zero() && mantissa == 0) {
            exponent.decrease(mantissa_width);
        }

        return {mantissa, exponent};
    }

    // roundMantissa() will create a valid ieee754 data type from the supplied parameters.
    // sticky could turn around and simply become a bool: true if any sticky bits were found
    // (before passing the arguments to this function); else false.
    // Also note that the mantissa bits are left justified: they consume the bits [63..11],
    // the rest of the bits [10 .. 0] are used (along with sticky) for rounding.
    static ieee754 roundMantissa(ieee754_sign sign, exponent_t exponent, storage_t mantissa, sticky_bits sticky)
    {
        // At this point, mantissa is completely left justified, or zero
        ASSERT((mantissa >> sign_shift) || mantissa == 0, "The mantissa must either have the high bit set or be 0");

        // All numbers come in with the highest order bit set.  That's just how the multiply and
        // divide routines work.

        // But, for rounding and then building an IEEE754 formatted number, that doesn't help.
        // It's time to align the numbers.

        // Aligning the numbers.
        // Step one: Shift the number right at least one bit to create a space for
        // a carry when rounding.
        sticky |= mantissa & 1;
        mantissa >>= 1;
        carry_bit carry_bit{sign_shift};

#ifndef NO_NAN_INF
        // Step two:  If the number is a denormal, shift it right one more bit
        // because the recorded mantissa bits never include the leading one bit
        // (of a normal number).
        sticky |= mantissa & (exponent.is_denormal() ? 1 : 0);
        mantissa >>= exponent.is_denormal() ? 1 : 0;
        carry_bit -= exponent.is_denormal() ? 1 : 0;
        // Now, the mantissa bits that will be recorded are all in the same zone.

        if constexpr(sizeof(mantissa) == 16) {
            uint128_t m = mantissa & exponent.denormal_shift_mask();
            uint64_t mlo = static_cast<uint64_t>(m >>  0);
            uint64_t mhi = static_cast<uint64_t>(m >> 64);
            sticky |= mlo | mhi;
        } else {
            sticky |= mantissa & exponent.denormal_shift_mask();
        }

        mantissa >>= exponent.denormal_shift_count();
        carry_bit -= exponent.denormal_shift_count();
#endif  /* NO_NAN_INF */

        // Now the bits are in the following bit positions for normal numbers:
        //    31 -- Carry
        //    30 -- 1
        //    29 .. 8 -- Mantissa
        //    7 -- Rounding bit
        //    6 .. 0 -- Any remaining sticky bits

        // Now the bits are in the following bit positions for denormal numbers:
        //    31 -- 0
        //    30 -- 0, this could be the carry if bit 22 of the mantissa is non-zero
        //    29 -- 1, if bit 22 of the mantisssa is non-zero, else 0, could be the carry bit if 21 is non-zero
        //    28 -- 1, if bit 21 of the mantisssa is non-zero, else 0, could be the carry bit if 20 is non-zero
        //    ... Etc.
        //    9 -- Second lowest order bit of the mantissa, is the carry if bits 52..1 are all 0's
        //    8 -- Lowest order bit of the mantissa, cannot be the carry bit ever.
        //    7 -- Rounding bit
        //    6 .. 0 -- Any remaining sticky bits

        if constexpr(sizeof(mantissa) == 16) {
            uint128_t m = mantissa & sticky_mask;
            uint64_t mlo = static_cast<uint64_t>(m >>  0);
            uint64_t mhi = static_cast<uint64_t>(m >> 64);
            sticky |= mlo | mhi;
        } else {
            sticky |= mantissa & sticky_mask;
        }
        round_bit rounding {mantissa & rounding_mask ? true : false };
        mantissa >>= rounding_shift;
        carry_bit -= rounding_shift;

        // Now the mantissa is right justified, ready to be consumed by the IEEE 754 format.
        // Unit of Least Precision (ULP)
        ulp_bit ulp {static_cast<uint8_t>(mantissa)};

        // The exponent means that the the exponent has the range -1023 ... 1022.
        // To get the biased exponent, we add 1023.  So, -1023 -> 0, 1022 -> 2046 (0x7FE).
        // Note that in IEEE 754, if the exponent is all bits set (0x7FF) then the "number"
        // is a NAN or an infinity (depening on the mantissa bits).

        // In the general case:
        // The mantissa bits are 63..11, but bits 10..0 contain bits useful in the rounding process.

        // If the number has an exponent >= -1022, then the leading 1 is absorbed (as the implied bit)
        // and thus 53 bits are recorded in the IEEE 754 format.

        // If the exponent is < -1022, then only 52 bits of mantissa are recorded
        // (and the number is a denormal).

        // This determines which bit is used for rounding.

        // There is one corner case in all of this logic.  If the exponent = -1023, and the mantissa is
        // all ones, the rounding bit is set and at least one of the sticky bits is non-zero, then
        // there's an exception:  The number starts as a denormal (the exponent < -1022), but the
        // rounding causes it increment into a regular number.
        // But, that's OK, because the mantissa will go from all bits set to 0.  And in that one case
        // a carry will be produced and that will be the implied one bit in the IEEE representation.
        // Thus, in this one case, the number changes states from denormal to normal.  Here is the case:
        //    (biased) exponent = 0, mantissa = 0x000F'FFFF'FFFF'FFFF
        // Then, the number is rounded up to:
        //    (biased) exponent = 1, mantissa = 0x0010'0000'0000'0000.
        // This algorithm will make that transformation ... keep your eyes out for it.

        // Rounded down is just the value
        storage_t rounded_down_mantissa = mantissa;
        auto rounded_down_exp {exponent};

        // The rounded value.
        storage_t rounded_up_mantissa = mantissa + 1;

        // Has the rounded up value created a carry
        int rounded_up_carry = (rounded_up_mantissa >> carry_bit.bit()) & 1;
        // This exponent increase handles the denormal -> normal transition, as well
        // as lots of other cases.  By adjusting the exponent; we know how many bits
        // to shift the mantissa (if any bits at all; which is only >= 0 for
        // denormal numbers).
        auto rounded_up_exp {exponent.plus(rounded_up_carry)};

        // The description of the "should round up" logic is described at the implementation
        bool round_down = ieee754_should_round_down(ulp, rounding, sticky);

        storage_t ret_mantissa = round_down ? rounded_down_mantissa : rounded_up_mantissa;
        auto rounded_exponent = round_down ? rounded_down_exp : rounded_up_exp;

        uint32_t ret_exponent {rounded_exponent.as_bits()};
        ieee754 result {sign, ret_exponent, ret_mantissa};

        return ret_exponent == exp_max ? inf(result) : result;
    }

    friend std::pair<ieee754, bool> add_internal(const ieee754& lhs, const ieee754& rhs);
    friend ieee754 add(const ieee754& lhs, const ieee754& rhs)
    {
        /*
        *   Operations that produce Infinity:
        *
        *  1. x / 0 where x ≠ 0 - Division by zero
        *  2. ∞ + x where x is no NaN (see below, operations that produce NaN)
        *  3. ∞ × x where x > 0
        *  4. ∞ / x where x > 0
        *  5. Overflow - Result exceeds representable range
        *
        *   Operations that produce NaN:
        *
        *  1. 0/0 - Indeterminate form
        *  2. ∞/∞ - Indeterminate form
        *  3. ∞ - ∞ - Indeterminate form
        *  4. 0 × ∞ - Indeterminate form
        *  5. √(-x) - Square root of negative number (in real arithmetic)
        *  6. log(-x) - Logarithm of negative number
        *  7. asin(x) where |x| > 1 - Invalid domain
        *  8. NaN op anything - Any operation with NaN
        */

#ifndef NO_NAN_INF
        bool return_nan = lhs.is_nan() || rhs.is_nan() ||
                ((lhs.is_infinite() && rhs.is_infinite()) && (lhs.sign() != rhs.sign()));
        bool return_infinity = lhs.is_infinite() || rhs.is_infinite();
        bool return_neg_zero = lhs.is_zero() && rhs.is_zero() && (rhs.sign() == lhs.sign()) && (lhs.sign() == 1);
        bool return_special = return_nan || return_infinity | return_neg_zero;
#endif

        bool swap = is_unsigned_lessthan(lhs, rhs);
        ieee754 a = swap ? rhs : lhs;
        ieee754 b = swap ? lhs : rhs;

        auto [result, return_zero] = add_internal(a, b);

#ifndef NO_NAN_INF
        ieee754 pos_zero = zero();
        ieee754 neg_zero = zero(1);
        ieee754 nan = ieee754::nan(a);
        ieee754 inf = ieee754::inf(a);

        return_special |= return_zero;
        return return_special ? (return_nan ? nan : return_neg_zero ? neg_zero : return_zero ? pos_zero : inf) : result;
#else
        return return_zero ? zero() : result;
#endif
    }

    friend ieee754 multiply_internal(const ieee754& lhs, const ieee754& rhs);
    friend ieee754 multiply(const ieee754& lhs, const ieee754& rhs)
    {
        // In this algorithm, the 53 bits of mantissa for both the
        // LHS and RHS are in the bit positions: [63..11] some of
        // time and sometimes they're in the bit positions [62..10].
        // The difference is:
        // + When performing the muliplication, the numbers are left justified
        //   (bits [63..11]).  This is how the multiplication algorithm wants the
        //   data and leaves the most lower-order bits in place for rounding.
        // + When performing addition, the mantissa data is in bits
        //   [62..10] so that carry can easily be detected by test bit 63.

        /*
        *   Operations that produce Infinity:
        *
        *  1. x/0 where x ≠ 0 - Division by zero
        *  2. Overflow - Result exceeds representable range
        *  3. ∞ + x where x is finite
        *  4. ∞ × x where x > 0
        *
        *   Operations that produce NaN:
        *
        *  1. 0/0 - Indeterminate form
        *  2. ∞/∞ - Indeterminate form
        *  3. ∞ - ∞ - Indeterminate form
        *  4. 0 × ∞ - Indeterminate form
        *  5. √(-x) - Square root of negative number (in real arithmetic)
        *  6. log(-x) - Logarithm of negative number
        *  7. asin(x) where |x| > 1 - Invalid domain
        *  8. NaN op anything - Any operation with NaN
        */
#ifndef NO_NAN_INF
        bool return_nan = (lhs.is_zero()    && rhs.is_infinite()) || (lhs.is_infinite() && rhs.is_zero()) || lhs.is_nan() || rhs.is_nan();
        bool return_inf = (lhs.is_nonzero() && rhs.is_infinite()) || (lhs.is_infinite() && rhs.is_nonzero());
        bool return_zero = lhs.is_zero()    || rhs.is_zero();
        bool return_special = return_nan || return_inf || return_zero;
#endif

        auto result = multiply_internal(lhs, rhs);

#ifndef NO_NAN_INF
        ieee754 zero = ieee754::zero(result.sign());
        ieee754 nan = lhs.is_nan() ? ieee754::nan(lhs) : rhs.is_nan() ? ieee754::nan(rhs) : ieee754::nan(zero);
        ieee754 inf = ieee754::inf(zero);

        return return_special ? (return_nan ? nan : return_inf ? inf : zero) : result;
#else
        return result;
#endif
    }

    friend std::pair<ieee754, ieee754_sign> divide_internal(const ieee754& lhs, const ieee754& rhs, bool force_mantissa);
    friend ieee754 divide(const ieee754& lhs, const ieee754& rhs)
    {
        /*
         *   Operations that produce Infinity:
         *
         *  1. x / 0 where x ≠ 0 - Division by zero
         *  2. ∞ + x where x is finite
         *  3. ∞ × x where x > 0
         *  4. ∞ / x where x > 0
         *  5. Overflow - Result exceeds representable range
         *
         *   Operations that produce NaN:
         *
         *  1. 0/0 - Indeterminate form
         *  2. ∞/∞ - Indeterminate form
         *  3. ∞ - ∞ - Indeterminate form
         *  4. 0 × ∞ - Indeterminate form
         *  5. √(-x) - Square root of negative number (in real arithmetic)
         *  6. log(-x) - Logarithm of negative number
         *  7. asin(x) where |x| > 1 - Invalid domain
         *  8. NaN op anything - Any operation with NaN
         */
#ifndef NO_NAN_INF
        bool return_nan = (lhs.is_zero() && rhs.is_zero()) || (lhs.is_infinite() && rhs.is_infinite()) || lhs.is_nan() || rhs.is_nan();
        bool return_inf = (lhs.is_finite() && rhs.is_zero()) || lhs.is_infinite();
        bool return_zero = lhs.is_zero() || rhs.is_infinite();
        bool return_special = return_nan || return_inf || return_zero;
#else
        bool return_special = false;
#endif
        auto [result, ret_sign] = divide_internal(lhs, rhs, return_special);
#ifndef NO_NAN_INF
        ieee754 zero = ieee754::zero(ret_sign);
        ieee754 nan = ieee754::nan(zero);
        ieee754 inf = ieee754::inf(zero);
        return return_special ? (return_nan ? nan : return_inf ? inf : zero) : result;
#else
        return result;
#endif
    }

    inline bool is_exp_zero() const { return exp_bits() == 0; }
    inline bool is_exp_max()  const { return exp_bits() == exp_max; }
    inline bool is_mantissa_zero() const { return mantissa_bits() == 0; }
    inline uint8_t sign_bit() const { return _mantissa_bits >> sign_shift; }

    inline storage_t make_bits(ieee754_sign _sign, uint32_t _exp, storage_t _mantissa) {
        return to_sign_bit(_sign.bit()) | to_exp_bits(_exp) | to_mantissa_bits(_mantissa);
    }

public:
    static inline ieee754 zero() { return ieee754(); }
    static inline ieee754 zero(ieee754_sign sign) { return ieee754(sign, 0, 0); }
    static inline ieee754 nan(ieee754 n) { return ieee754(n.sign(), exp_max, 1); }
    static inline ieee754 inf(ieee754 n) { return ieee754(n.sign(), exp_max, 0); }
    static ieee754 Pi();

public:
    ieee754() : _mantissa_bits(0) {}
    ieee754(ieee754_sign _sign, uint32_t _exp, storage_t _mantissa) :
        _mantissa_bits(make_bits(_sign, _exp, _mantissa)) {}
    // ieee754(double d) { storage_t real = d; std::memcpy(&n, &real, sizeof(storage_t)); }
    storage_t from_fp64(ieee754_sign sign, int iexp, uint64_t mantissa);
    explicit ieee754(float real) {
        // Don't call through a function for the identity function.
        // Because: We cast away denormals below; here we want to keep
        // all bit patterns.
        if constexpr(sizeof(storage_t) == sizeof(float)) {
            _mantissa_bits = std::bit_cast<uint32_t>(real);
            return;
        }

        uint32_t bits = std::bit_cast<uint32_t>(real);
        ieee754_sign sign {static_cast<uint8_t>((bits >> 31) & 1)};
        uint32_t exp  = (bits >> 23) & 0xFF;
        uint64_t mantissa = bits & ((1ULL << 23) -1);

        if (exp == 0xFF) {
            // Handle infinities and nans in one, fell swoop.
            _mantissa_bits = make_bits(sign, exp_max, mantissa != 0);
        } else if (exp == 0) {
            // Both 0 and denormals result in 0
            _mantissa_bits = make_bits(sign, 0, 0);
        } else {
            // We let each implementation deal with their own conversion
            // from Double to their size.
            int iexp = static_cast<int>(exp) - 0x7F;
            _mantissa_bits = from_fp64(sign, iexp, mantissa << 20);
        }
    }

    explicit ieee754(double real) {
        // Don't call through a function for the identity function.
        // Because: We cast away denormals below; here we want to keep
        // all bit patterns.
        if constexpr(sizeof(storage_t) == sizeof(double)) {
            _mantissa_bits = std::bit_cast<uint64_t>(real);
            return;
        }

        /*
         * Given a double, can we construct a fp32, fp64 (equality) and fp128?
         * If generating an fp32: We need to shrink the exponent space, which
         *    means that we support a smaller set of numbers.
         * If generating an fp64: We do nothing, it's the same thing.
         * If generating an fp128: We need to insert some bits into the exponent
         *    and the rest of the mantissa bits are 0.
         */
        uint64_t bits = std::bit_cast<uint64_t>(real);
        ieee754_sign sign {static_cast<uint8_t>((bits >> 63) & 1)};
        uint32_t exp  = (bits >> 52) & 0x7FF;
        uint64_t mantissa = bits & ((1ULL << 52) -1);

        if (exp == 0x7FF) {
            // Handle infinities and nans in one, fell swoop.
            _mantissa_bits = make_bits(sign, exp_max, mantissa != 0);
        } else if (exp == 0) {
            // Both 0 and denormals result in 0
            _mantissa_bits = make_bits(sign, 0, 0);
        } else {
            // We let each implementation deal with their own conversion
            // from Double to their size.
            int iexp = static_cast<int>(exp) - 0x3FF;
            _mantissa_bits = from_fp64(sign, iexp, mantissa);
        }
    }
    ieee754(int _i) {
        if constexpr(sizeof(storage_t) == 8) {
            _mantissa_bits = std::bit_cast<storage_t, double>(static_cast<double>(_i));
        } else if constexpr(sizeof(storage_t) == 4) {
            _mantissa_bits = std::bit_cast<storage_t, float>(static_cast<float>(_i));
        } else {
            int64_t n = _i;
            if (n == 0) {
                _mantissa_bits = 0;
                return;
            }

            if (n == 1) {
                _mantissa_bits = make_bits(0, exp_bias, 0);
                return;
            }

            if (n == 2) {
                _mantissa_bits = make_bits(0, exp_bias +1, 0);
                return;
            }

            if (n == INT_MIN) {
                // An INT_MIN number can't be inverted.
                _mantissa_bits = make_bits(1 /* we know it's negative */, exp_bias + sizeof(int) * 8, 0);
                return;
            }

            // At this point, we know that we can negate n.
            ieee754_sign s = 0;
            if (n < 0) {
                s = 1;
                n = -n;
            }

            // At this point, we know that n is positive.
            uint64_t m = static_cast<uint64_t>(n);
            int x = std::countl_zero(m);
            int z = 63 - x;
            // If x == 63, then:
            //     m would be 1.
            //     z would be 0.
            //     The exponent should be set to exp_bias + 0.
            //     This would indicate that the value of the implicit bit should be 1
            // If x == 62, then:
            //     m would be either 2 or 3
            //     z would be 1.
            //     The exponent should be set to exp_bias + 1.
            //     This would indicate that the value of the implicit bit should be 2
            // Thus: The exponent for the floating point number should be exp_bias + z.
            // And the mantissa should be n shifted left so the most significant bit is
            // in bit position 52; and then it needs to be chopped off.  (Yikes!)
            if (x < 12) {
                m >>= 12 - x;
            } else if (x > 12) {
                m <<= x - 11;
            }
            m = m & ((1ULL << 52) -1);
            _mantissa_bits = from_fp64(s, z, m);
        }
    }
    // ieee754(size_t i) { ieee754(static_cast<fp_t>(i));  }

    /*
     *  Common IEEE 754 Encodings
     *  Sign:
     *  - unsigned single bit
     *  - 0: Positive
     *  - 1: Negaitve
     *
     *  Exponent:
     *  - Unsigned integer
     *  - exp_max: All 1s: (1 << EXP_WIDTH) -1
     *  - Biased by half the max of the integer: exp_max >> 1
     *
     *  Zero:
     *  - Exponent: 0 (all 0s)
     *  - Mantissa: 0 (all 0s)
     *  - Sign bit determines +0 or -0
     *
     *  Normal:
     *  - Exponent: 0x001 ... 0x7FE (Neither all 1s nor all 0s)
     *  - Mantissa: Anything; there's an implied bit 52 (the 53rd mantissa bit) that's a one
     *  - Sign bit determines +N or -N
     *
     *  Denormal:
     *  - Exponent: 0 (all 0s)
     *  - Mantissa: Anything except all 0s (else, the number would be 0, see above)
     *  - Sign bit determins +N or -N
     *
     *  Infinity:
     *  - Exponent: 0x7FF (all 1s)
     *  - Mantissa: 0 (all 0s)
     *  - Sign bit determines +∞ or -∞
     *
     *  NaN (Not a Number):
     *  - Exponent: 0x7FF (all 1s)
     *  - Mantissa: Any non-zero value (This program uses 1)
     *  - Two types:
     *    - Quiet NaN (qNaN): Most significant mantissa bit = 1 (bit 51) -- Unused in this code
     *    - Signaling NaN (sNaN): Most significant mantissa bit = 0, but mantissa ≠ 0
     */
#ifndef NO_NAN_INF
     // Zero is defined as Exponent zero and a zero mantissa
    constexpr bool is_zero()     const { return  is_exp_zero() &&  is_mantissa_zero(); }

    // Infinity is defined as maximum Exponent and a zero mantissa
    constexpr bool is_infinite() const { return  is_exp_max()  &&  is_mantissa_zero(); }

    // NaN is defined as maximum Exponent and non-zero mantissa
    constexpr bool is_nan()      const { return  is_exp_max()  && !is_mantissa_zero(); }

    // Finite number is any number where the exponent is not maximum, this includes zero.
    constexpr bool is_finite()   const { return !is_exp_max(); }
#else
    constexpr bool is_zero()     const { return false; }
    constexpr bool is_infinite() const { return false; }
    constexpr bool is_nan()      const { return false; }
    constexpr bool is_finite()   const { return false; }
#endif
    // Non-zero is defined as Finite but Not Zero.
    constexpr bool is_nonzero()  const { return  is_finite()   && !is_zero(); }

    constexpr bool is_denormal() const { return  is_exp_zero() && !is_mantissa_zero(); }


    friend ieee754 operator + (const ieee754& lhs, const ieee754& rhs) {
        return add(lhs, rhs);
    }
    friend ieee754 operator - (const ieee754& lhs, const ieee754& rhs) {
        return add(lhs, -rhs);
    }

    friend ieee754 operator * (const ieee754& lhs, const ieee754& rhs) {
        return multiply(lhs, rhs);
    }

    friend ieee754 operator / (const ieee754& lhs, const ieee754& rhs) {
        return divide(lhs, rhs);
    }
    friend ieee754 operator - (const ieee754& rhs) {
        return ieee754(-rhs.sign(), rhs.exp_bits(), rhs.mantissa_bits());
    }

    /*
     * Test code
     */
    [[nodiscard]] bool test_check(fp_t d) const
    {
        return test_check(ieee754{d});
    }

    [[nodiscard]] bool test_check(const ieee754& k) const
    {
#ifdef NO_NAN_INF
        return true;
#endif
        // Two NaNs are always equal, regardless of mantissa specifics
        if (is_nan() && k.is_nan()) {
            return true;
        }

#if 0
        // Infinities have a very specific bit pattern.
        // Two Infinites are only equal if their signs are equal
        if (is_infinite() && k.is_infinite()) {
            return sign() == k.sign();
        }

        // -0 == +0
        if (is_zero() && k.is_zero()) {
            return true;
        }
#endif

        if constexpr (sizeof(ieee754) == 16) {
            uint64_t my_some_bits = static_cast<uint64_t>(as_bits() >> 68);
            uint64_t yu_some_bits = static_cast<uint64_t>(k.as_bits() >> 68);
            if (my_some_bits != yu_some_bits) {
                printf("my_some_bits = %16.16llx\n", my_some_bits);
                printf("yu_some_bits = %16.16llx\n", yu_some_bits);
            }
            return my_some_bits == yu_some_bits;
        }

        // Else, the whole bit patterns had better be equal
        return as_bits() == k.as_bits();
    }

    /**
     * @fn fp fp::test_adj_mantissa(int n)
     * @brief Adjusts the mantissa by incrementing or decrementing it by n ULPs
     *
     * This function creates a new ieee754_32 value with the mantissa adjusted by
     * n units of least precision (ULPs). It handles carry/borrow operations that
     * may affect the exponent.
     *
     * @param n The number of ULPs to adjust the mantissa by:
     *          - Positive values increase the mantissa
     *          - Negative values decrease the mantissa
     *          - Zero returns a copy with no change
     *
     * @return A new ieee754_32 value with the adjusted mantissa
     *
     * @note When incrementing causes mantissa overflow (bit 23 set), the mantissa
     *       wraps around and the exponent is incremented
     * @note When decrementing causes mantissa underflow, bit 23 is set and the
     *       exponent is decremented
     *
     * @warning This function is primarily for testing purposes, particularly for
     *          verifying values that are approximately equal to expected results
     *
     * @see test_approx_one() for example usage
     */
    ieee754 test_adj_mantissa(int adj)
    {
        ieee754 r{as_real()};

        storage_t n = static_cast<storage_t>(std::abs(adj));
        storage_t m = mantissa_bits();
        if (adj >= 0) {
            m += n;
            if (m & implied_bit) {
                assert(r.exp_bits() < exp_max);
                m &= implied_bit -1;
                r.set_exp(r.exp_bits() +1);
            }
        } else {
            if (m < n) {
                assert(r.exp_bits() > 0);
                m |= implied_bit;
                r.set_exp(r.exp_bits() -1);
            }
            m -= n;
        }
        r.set_mantissa(m);

        return r;
    }

    void test_print_parts(const char *short_name, bool verbose = true) const
    {
        if (!verbose) {
            return;
        }

        char sign_char = sign() == 0 ? '+' : '-';
        uint32_t exp = exp_bits();
        storage_t mantissa = mantissa_bits();

        printf("%-5s = %c", short_name, sign_char);
        double value = static_cast<double>(as_real());
        if constexpr (sizeof(storage_t) == 2) {
            printf("%2.2x%6.6x", exp, mantissa);
        } else if constexpr (sizeof(storage_t) == 4) {
            printf("%2.2x%6.6x", exp, mantissa);
        } else if constexpr (sizeof(storage_t) == 8) {
            printf("%3.3x%13.13llx", exp, mantissa);
        } else if constexpr (sizeof(storage_t) == 16) {
            printf("%16.16llx.%16.16llx", static_cast<uint64_t>(as_bits() >> 64), static_cast<uint64_t>(as_bits() & UINT64_MAX));
        }
        printf(", as float = %g\n", value);
    }

    void test_print_raw(const char *short_name, bool verbose = true) const
    {
        if (!verbose) {
            return;
        }

        if constexpr (sizeof(storage_t) == 4) {
            printf("%-15s = %8.8x, as float = ", short_name, as_bits());
            print(*this);
            printf("\n");
        } else if constexpr (sizeof(storage_t) == 8) {
            printf("%-15s = %16.16llx, as double = ", short_name, as_bits());
            print(*this);
            printf("\n");
        } else if constexpr (sizeof(storage_t) == 16) {
            printf("%-15s = %16.16llx.%16.16llx, as float = ", short_name, static_cast<uint64_t>(as_bits() >> 64), static_cast<uint64_t>(as_bits() & UINT64_MAX));
            print(*this);
            printf("\n");
        }
    }

    friend void print(const ieee754& N)
    {
        print_dec(N);
        return;
    }

    inline ieee754_sign sign() const { return ieee754_sign(sign_bit()); }
    inline void set_sign(ieee754_sign sign) { _mantissa_bits = (_mantissa_bits & (exp_mask | mantissa_mask)) | to_sign_bit(sign.bit()); }

    inline uint32_t exp_bits() const { return (_mantissa_bits >> exp_shift) & exp_max; }
    inline void set_exp(uint32_t exp) { _mantissa_bits = (_mantissa_bits & (sign_mask | mantissa_mask)) | to_exp_bits(exp); }

    inline uint32_t mantissa_top_4bits() const { return static_cast<uint32_t>((_mantissa_bits >> (mantissa_width -4)) & 0xF); }
    inline storage_t mantissa_bits() const { return _mantissa_bits & mantissa_mask; }
    inline void set_mantissa(storage_t mantissa) { _mantissa_bits = (_mantissa_bits & (sign_mask | exp_mask)) | (mantissa & mantissa_mask); }

    inline storage_t as_bits() const { return _mantissa_bits; }
    // inline double as_double() const { double d; std::memcpy(&d, &_mantissa_bits, sizeof(double)); return d; }
    inline long as_long() const {
        auto [mantissa, exp] = getMantissaAlignedUnbiased();
        if (exp.exponent() < 0) {
            return 0;
        }

        int shift = sign_shift - exp.exponent();
        return static_cast<long>(shift_right(mantissa, shift));
    }

    inline fp_t as_real() const {
        if constexpr(sizeof(ieee754) == 1) {
            // In this case the bits are not the same size as the Hardware's actual floating point value
            // which represents it.  In fact, sizeof(E4M3) != sizeof(_Float16).
            return 0;
        } else if constexpr(sizeof(ieee754) <= 8) {
            // In this case the bits are the same size as the Hardware's actual floating point value
            // which represents it.  I.e., fp64 <-> double; fp32 <-> float.
            return std::bit_cast<fp_t>(_mantissa_bits);
        } else {
            // Converting an fp128 -> fp64
            if (is_nan()) {
                return sign_bit() ? -0.0 / 0.0 : 0.0 / 0.0;
            }
            if (is_infinite()) {
                return sign_bit() ? -1.0 / 0.0 : 1.0 / 0.0;
            }
            assert(exp_bits() != exp_max);

            if (exp_bits() == 0) {
                return sign_bit() ? -0.0 : +0.0;
            }
            assert(exp_bits() != 0);

            // In this case, the mantissa bits are NOT the same size as a hardware's actual floating point value.
            // I.e., fp128 doesn't have any hardware support that matches it.
            // This algorithm will "narrow" the floating point value (e.g., 128bit or 256bit values) down to a double.
            uint64_t result {0};
            result |= static_cast<uint64_t>(sign_bit()) << 63;
            uint64_t exp = static_cast<uint64_t>(exp_bits());
            uint64_t exp_hi = (exp >> (exp_width - 1)) & 0x01;
            if constexpr (sizeof(ieee754) == 16) {
                // Narrow from a 15bit exponent to an 11.
                // Grab the four bits I'm going to drop (which are bits 13..10)
                uint64_t exp_four = (exp >> (exp_width - 5)) & 0xF;
                if (exp_four == 0xF || exp_four == 0) {
                    result |= exp_hi << 62;
                    result |= (exp & 0x3FF) << 52;
                } else {
                    if (exp_hi) {
                        // Return infinity
                        return sign_bit() ? -1.0 / 0.0 : 1.0 / 0.0;
                    } else {
                        // Return 0
                        return sign_bit() ? -0.0 : +0.0;
                    }
                }
                result |= mantissa_bits() >> (112 - 52);
                return std::bit_cast<double>(result);
            } else {
                static_assert(false, "ERROR: Unhandled size");
            }
        }
    }

    // Spaceship operator for three-way comparison (C++20)
    // Returns std::partial_ordering because NaN comparisons are unordered
    friend std::partial_ordering operator<=>(const ieee754& lhs, const double& rhs) {
        return operator<=>(lhs, ieee754{rhs});
    }
    friend std::partial_ordering operator<=>(const ieee754& lhs, const ieee754& rhs) {
        // Handle NaN cases first - any comparison with NaN is unordered
        if (lhs.is_nan() || rhs.is_nan()) {
            return std::partial_ordering::unordered;
        }

        // Handle zero comparisons (both +0 and -0 are equal)
        if (lhs.is_zero() && rhs.is_zero()) {
            return std::partial_ordering::equivalent;
        }

        // For non-NaN, non-zero values, we can compare the bit patterns
        // but we need to handle the sign bit specially
        // Sign bit: 0 = positive, 1 = negative
        bool lhs_positive = lhs.sign().bit() == 0;
        bool rhs_positive = rhs.sign().bit() == 0;

        // Different signs
        if (lhs_positive != rhs_positive) {
            // I know this is the same as 'greater_than', down below, but I've written this
            // many different ways and it doesn't make sense to put it here.
            return lhs_positive ? std::partial_ordering::greater : std::partial_ordering::less;
        }

        std::partial_ordering less_than = lhs_positive ? std::partial_ordering::less : std::partial_ordering::greater;
        std::partial_ordering greater_than = lhs_positive ? std::partial_ordering::greater : std::partial_ordering::less;

        // Same sign - compare exponents and then magnitudes
        // For negative numbers, larger bit pattern means smaller value

        // Test exponents first, they're the most important bits
        if (lhs.exp_bits() < rhs.exp_bits()) {
            return less_than;
        }
        if (lhs.exp_bits() > rhs.exp_bits()) {
            return greater_than;
        }

        // Exponents are equal, test magnitude
        if (lhs.mantissa_bits() < rhs.mantissa_bits()) {
            return less_than;
        }
        if (lhs.mantissa_bits() > rhs.mantissa_bits()) {
            return greater_than;
        }

        // Even infinities fall through here.  I'm not sure they should.
        return std::partial_ordering::equivalent;
    }

    // Why is a == operator needed?
    friend bool operator==(const ieee754& lhs, const ieee754& rhs) {
        // NaN is not equal to any other number, according to IEEE 754
        if (lhs.is_nan() || rhs.is_nan()) {
            return false;
        }

        // +0 == -0
        if (lhs.is_zero() && rhs.is_zero()) {
            return true;
        }

        int r = memcmp(&lhs, &rhs, sizeof(ieee754));
        return r == 0;
    }

    friend ieee754 sqrt(const ieee754& a) {
        if (a < 0) {
            return ieee754::nan(a);
        }

        if (a == 0) {
                return 0;
        }

        uint32_t x = a.exp_bits();
        if (x > ieee754::exp_bias) {
            x = (x - ieee754::exp_bias) / 2 + ieee754::exp_bias;
        } else if (x < ieee754::exp_bias) {
            x = ieee754::exp_bias - (ieee754::exp_bias - x) / 2;
        }
        ieee754 x0 (0, x, 0);
        ieee754 x1 {0.0};
        for (int i = 0; i < 6; ++i) {
            x1 = (x0 + a / x0) / 2.0;
            x0 = x1;
        }
        return x1;
    }

    friend ieee754 abs(const ieee754& a) {
        ieee754 p{a};
        p.set_sign(0);
        return p;
    }

    friend void print_dec(ieee754 n) {
        double mantissa_bits_printed = 0.;
        if (n < 0) {
            putchar('-');
            n = -n;
        }

        if (n >= 1) {
            ieee754 m = 1;
            while (n >= m) {
                m = m * 10;
            }
            m = m / 10;
            mantissa_bits_printed += m.exp_bits() - ieee754::exp_bias;
            while (m >= 1) {
                int k = static_cast<int>((n / m).as_real());
                putchar("0123456789abcdef"[k]);
                n = n - m * k;
                m = m / 10;
            }
        } else {
            putchar('0');
        }

        if (n > 0) {
            assert(n < 1);
            putchar('.');
            double bits_per_digit = 3.3;
            double Width = static_cast<double>(ieee754::mantissa_width);
            while (n > 0 && mantissa_bits_printed < Width) {
                // printf("\nExpBits = %x; mantissa_bits_printed = %d -- ", n.exp_bits(), mantissa_bits_printed);
                n = n * 10;
                if (mantissa_bits_printed > 0) {
                    mantissa_bits_printed += bits_per_digit;
                } else if (n >= 1) {
                    mantissa_bits_printed = bits_per_digit;
                }

                int k = 0;
                if (n >= 8) {k = k + 8; n = n - 8; }
                if (n >= 4) {k = k + 4; n = n - 4; }
                if (n >= 2) {k = k + 2; n = n - 2; }
                if (n >= 1) {k = k + 1; n = n - 1; }
                assert(k < 10);
                putchar("0123456789"[k]);
            }
        }
        // printf("\n");
    }

    friend void print_hex(ieee754 n) {
        if (n < 0) {
            putchar('-');
            n = -n;
        }

        printf("0x");
        ieee754 m = 1;
        while (n > m) {
            m = m * 10;
        }
        m = m / 10;
        while (m >= 1) {
            int k = static_cast<int>((n / m).as_real());
            assert (k < 10);
            assert (k >= 0);
            putchar("0123456789"[k]);
            n = n - m * k;
            m = m / 10;
        }
        assert(n < 1);
        if (n == 0) {
            return;
        }
        putchar('.');
        int digits = static_cast<int>(9.0 * mantissa_width / 32.0);
        for (int i = digits; i > 0; --i) {
            n = n * 16;
            int k = 0;
            if (n >= 8) {k = k + 8; n = n - 8; }
            if (n >= 4) {k = k + 4; n = n - 4; }
            if (n >= 2) {k = k + 2; n = n - 2; }
            if (n >= 1) {k = k + 1; n = n - 1; }
            putchar("0123456789abcdef"[k]);
        }
        printf("\n");
    }

    // These are debub interfaces, which is why they use a gross interface; they're not meant to be used by regular code.
    ieee754(std::array<storage_t, 1> k) : _mantissa_bits(k[0]) {}
    inline std::array<storage_t, 1> as_array() const { return std::array<storage_t, 1> {_mantissa_bits}; }
};

// using ieee754_8_e4m3 = ieee754<uint8_t,  _Float16,   3>;
// using ieee754_16     = ieee754<uint16_t, _Float16,  10>;
using ieee754_32     = ieee754<uint32_t,    float,  23>;
using ieee754_64     = ieee754<uint64_t,   double,  52>;
using ieee754_128    = ieee754<uint128_t,  double, 112>;
