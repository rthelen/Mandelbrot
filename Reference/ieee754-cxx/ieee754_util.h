/* 
 * IEEE754
 *
 * This file implements a bunch of utility classes that can be used by
 * the 64 and 128 bit implementations.
*/

#pragma once

#include <cstdint>
#include <compare>
#include <concepts>
#include "debug.h"
typedef __uint128_t  uint128_t;

// Add with carry helper
template<typename T>
struct add_with_carry_result {
    T sum;
    bool carry;
};

template<typename T>
    requires std::unsigned_integral<T>
inline add_with_carry_result<T> add_with_carry(T a, T b, bool carry_in = false) {
    T sum = a + b;
    bool carry1 = sum < a;

    if (carry_in) {
        sum += 1;
        bool carry2 = (sum == 0);
        return {sum, carry1 || carry2};
    }

    return {sum, carry1};
}

// Specialization for uint64_t using compiler builtins when available
#ifdef __GNUC__
template<>
inline add_with_carry_result<uint64_t> add_with_carry(uint64_t a, uint64_t b, bool carry_in) {
    uint64_t sum;
    bool carry = __builtin_add_overflow(a, b, &sum);

    if (carry_in) {
        bool carry2 = __builtin_add_overflow(sum, 1ULL, &sum);
        carry = carry || carry2;
    }

    return {sum, carry};
}
#endif

// Specialization for uint128_t using compiler builtins when available
#ifdef __GNUC__
template<>
inline add_with_carry_result<uint128_t> add_with_carry(uint128_t a, uint128_t b, bool carry_in) {
    uint128_t sum;
    bool carry = __builtin_add_overflow(a, b, &sum);

    if (carry_in) {
        bool carry2 = __builtin_add_overflow(sum, 1ULL, &sum);
        carry = carry || carry2;
    }

    return {sum, carry};
}
#endif

class ieee754_sign {
    uint8_t s;
public:
    ieee754_sign() : s(0) {}
    // There seem to be two ways to force s to either 0 or 1:
    //   Use a mask:     _s & 1
    //   Use a compare:  _s != 0
    ieee754_sign(uint8_t _s) : s(_s != 0) {}

    inline uint8_t bit() const { return s; }
    friend ieee754_sign operator * (const ieee754_sign &lhs, const ieee754_sign &rhs) {
        // pos x pos -> pos (0).   neg x neg -> pos (0).
        // pos x neg -> neg (1).   neg x pos -> neg (1).
        // So, if they don't equal each other, set to neg (1).
        // Exclusive-Or performs that computation.
        return ieee754_sign(lhs.bit() ^ rhs.bit());
    }
    friend bool operator == (const ieee754_sign &lhs, const ieee754_sign& rhs) {
        return lhs.bit() == rhs.bit();
    }
    friend bool operator == (const ieee754_sign &lhs, const unsigned& rhs) {
        return lhs.bit() == rhs;
    }
    friend ieee754_sign operator - (const ieee754_sign &rhs) {
        return ieee754_sign(rhs.bit() ^ 1);
    }

    inline int sign() { return s; }
};

/*
 * class ieee754_exponent
 *
 * This class is intended to help with managing an exponent value in its
 * signed form.  The unsigned form is the binary representation that
 * will be stored with the number, but the signed form is good for
 * managing very large changes in the exponent.  Multiply and divide
 * add and subtract exponents.  Which requires at least an extra bit
 * of information.  But more importantly, it could generate a value
 * less than 0 and managing that is easiest for signed integers.
 * 
 * The code needs to specify the Bias for the exponent and the Max value
 * of the exponent.  In most cases, the Max value == NANs and Infinites.
 * FP8 is a bit different and will need to be explored separately.
 */
template<uint32_t _exp_max_bits, uint32_t _exp_bias_bits, typename hw>
class ieee754_exponent {
private:
    int exp;

public:
    static constexpr uint32_t exp_max_bits = _exp_max_bits;
    static constexpr uint32_t exp_bias_bits = _exp_bias_bits;
    static constexpr int exponent_bias = static_cast<int>(exp_bias_bits);
    static constexpr int exponent_min = -exponent_bias;
    static constexpr int exponent_max = static_cast<int>(exp_max_bits) - exponent_bias;

private:
    using exponent_type = ieee754_exponent<exp_max_bits, exp_bias_bits, hw>;

public:
    ieee754_exponent(uint32_t e) : exp (static_cast<int>(e) - exponent_bias) {}
    ieee754_exponent(int e) : exp (e) {}
    inline int exponent() const { return exp; }
#ifndef NO_NAN_INF
    // It's interesting that is_infinity() tests > exponent_max; it seems like it should be >=.
    inline bool is_infinity() const { return exp > exponent_max; }
    inline bool is_denormal() const { return exp <= exponent_min; }
    // inline bool is_normal() const { return exp > exponent_min && exp <= exponent_max; }
#else
    inline bool is_zero() const { return exp <= exponent_min; }
#endif  /* NO_NAN_INF */
    inline bool as_bits_is_zero() const { return exp == -exponent_bias; }
    inline void increase(int n) { exp += n; }
    inline void decrease(int n) { exp -= n; }

    inline exponent_type plus(exponent_type rhs) const {
        return exponent_type {exp + rhs.exponent()};
    }
    inline exponent_type plus(int n) const {
        return exponent_type {exp + n};
    }
    inline exponent_type minus(exponent_type rhs) const {
        return exponent_type {exp - rhs.exponent()};
    }
    inline exponent_type minus(int n) const {
        return exponent_type {exp - n};
    }
    inline void increment(bool flag) {
        exp += flag ? 1 : 0;
    }
    inline uint32_t as_bits() const {
        uint32_t exp_as_uint32 = static_cast<uint32_t>(exp + exponent_bias);
#ifndef NO_NAN_INF
        return is_denormal() ? 0 : is_infinity() ? exp_max_bits : exp_as_uint32;
#else
        return is_zero() ? 0 : exp_as_uint32;
#endif  /* NO_NAN_INF */
    }
    /*
     *  denormal_shift_count() returns the number of bits to shift a mantissa
     *  in order to align it according to a bit position that accounts for the
     *  denormal nature of the mantissa.  This happens in the rounding algorithm.
     *  exponent_min is a negative number, it's defined as -exponent_bias.  So,
     *  let's think about it as -31 (there's no FP format that I know of for which
     *  that's a valid value, but that's not the point).  OK, so, now we know that
     *  exp is less than that value; that's what defines a number as denormal in the first
     *  place.  OK.  So, -31 - -33, for example, is the same as -31 + 33, or 33 - 31,
     *  thus 2.  So, the mantissa needs to be shifted to digits to the right.
     *  That's all there is to it.  However, the algorithm never suggests to shift
     *  by more than the size of the type, -1.
     */
#ifndef NO_NAN_INF
    inline int denormal_shift_count() const {
        return is_denormal() ? std::min(exponent_min - exp, static_cast<int>(8 * sizeof(hw) -1)) : 0;
    }
    inline hw denormal_shift_mask() const {
        return is_denormal() ? static_cast<hw>((static_cast<hw>(1) << denormal_shift_count()) -1) : 0;
    }
#endif  /* NO_NAN_INF */
};

class sticky_bits {
private:
    bool s;

public:
    sticky_bits() : s(false) {}
    sticky_bits(uint64_t _s) : s(_s ? true : false) {}

    sticky_bits& operator |= (uint64_t rhs) {
        s |= rhs ? true : false;
        return *this;
    }
    sticky_bits& operator |= (const sticky_bits& rhs) {
        s |= rhs.s;
        return *this;
    }

    operator bool() const { return s; }
};

class ulp_bit {
private:
    bool ulp;

public:
    explicit ulp_bit(uint64_t _mantissa) : ulp((_mantissa & 1) ? true : false) {}
    operator bool() const { return ulp; }
};

class round_bit {
private:
    bool round;

public:
    explicit round_bit(uint64_t _round) : round(_round ? true : false) {}
    // explicit round_bit(bool _round) : round(_round) {}
    operator bool() const { return round; }
};

/*
 *  R = round_bit, S = sticky bits, ULP = Unit of Least Precision of the mantissa
 *  If R == 0, then we round down (i.e., truncate).
 *  If R == 1 and both ULP and S == 0, then round down.
 *  Else, if R == 1 and ULP is 1, then round up.
 *  Or,   if R == 1 and S is non-zero, then round up.
 *
 *  Some examples, .5 is the rounding bit; .25 is first of the sticky bits:
 *     0.25 -> 0      The rounding bit is 0, round down
 *     0.5  -> 0      The rounding bit is 1 but both the ULP and the Sticky bits are zero, round down
 *     0.75 -> 1      The rounding bit is 1 and the Sticky bits are non-zero, roudn up
 *     1.25 -> 1      The rounding bit is 0, round down
 *     1.5  -> 2      The rounding bit is 1 and the ULP is 1, round up
 *     1.75 -> 2      The rounding bit is 1 and both the ULP and the sticky bits are non-zero, round up
 *
 *  Another way to look at this is via truth table.  Given three variables,
 *  ULP, R and S (sticky bits), we can form this table:
 *      ULP    R    S       Action
 *      ---   ---  ---      --------------------
 *       x     0    x       Round down, e.g., 0.25, 1.25
 *       0     1    0       Round down, e.g., 0.5
 *       0     1    1       Round up, e.g., 0.75
 *       1     1    x       Round up, e.g., 1.5, 1.75
 *
 *  From the above table we conclude that if R == 0, then we round down.
 *  Else (i.e., R != 0), if both ULP == 0 and S == 0, then we round down.
 */

inline bool ieee754_should_round_down(ulp_bit ulp, round_bit round, sticky_bits sticky)
{
    return !round || (!ulp && !sticky);
}

inline bool ieee754_should_round_up(ulp_bit ulp, round_bit round, sticky_bits sticky)
{
    return (ulp && round) || (!ulp && round && sticky);
}

/*
 * carry_bit tracks the bit position where a carry would appear if rounding
 * causes the mantissa to overflow. It starts at sign_shift (e.g., 127 for
 * fp128) and is decremented as the mantissa is shifted right during rounding.
 *
 * Underflow (rhs > bit_num) can occur in extreme denormal/underflow-to-zero
 * scenarios where the combined shifts exceed sign_shift:
 *   - is_denormal adjustment: 1 bit
 *   - denormal_shift_count: up to (8*sizeof(hw)-1) bits (e.g., 127 for fp128)
 *   - rounding_shift: mantissa_bits - sign_shift (e.g., 14 for fp128)
 *
 * When underflow occurs, the mantissa has been shifted to zero (or nearly so),
 * meaning no carry is possible from rounding. Clamping bit_num to 0 is safe
 * because checking bit 0 of a zero mantissa correctly yields no carry.
 */
class carry_bit {
private:
    int bit_num;
public:
    carry_bit(int sign_bitnum) : bit_num(sign_bitnum) {}
    carry_bit& operator -= (int rhs) {
        ASSERT(rhs >= 0, "Cannot decrement the carry bit position index with a negative value");
        // Underflow is expected for extreme denormals approaching zero.
        // See class comment above for explanation.
        bit_num = bit_num > rhs ? bit_num - rhs : 0;
        return *this;
    }
    int bit() const {
        return bit_num;
    }
};
