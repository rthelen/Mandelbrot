/*
 * karatsuba128.h
 *
 * 128×128 bit multiplication using Karatsuba algorithm with only 32×32 multiplies.
 * Designed for GPU portability where 64×64 multiply hardware is unavailable.
 *
 * Multiplication count:
 *   Naive 128×128 with 32-bit base: 16 multiplies
 *   Two-level Karatsuba:            10 multiplies
 *
 * Trade-off: Fewer multiplies, more ALU operations (adds, subtracts, shifts).
 * This is advantageous on GPUs where MUL latency >> ALU latency.
 */

#pragma once
#include <cstdint>

/*
 * 256-bit result of 128×128 multiplication.
 * Stored as four 64-bit words, little-endian (w[0] is lowest).
 */
struct uint256_t {
    uint64_t w[4];  // w[0] = bits 0-63, w[3] = bits 192-255
};

/*
 * Result structure for IEEE 754 integration.
 * For floating-point multiplication, we need:
 *   - High 128 bits of the 256-bit product (for the mantissa)
 *   - Sticky bit: OR of all lower 128 bits (for rounding)
 */
struct mul128_ieee754_result {
    uint64_t hi_lo;   // bits 128-191 of product
    uint64_t hi_hi;   // bits 192-255 of product
    uint64_t sticky;  // non-zero if any of bits 0-127 are set
};

/*
 * Full 128×128 → 256-bit multiplication using Karatsuba.
 * Uses exactly 10 32×32 multiplications.
 *
 * Input:  a = (a_hi << 64) | a_lo
 *         b = (b_hi << 64) | b_lo
 * Output: 256-bit product in uint256_t
 */
uint256_t mul128_karatsuba(uint64_t a_lo, uint64_t a_hi,
                           uint64_t b_lo, uint64_t b_hi);

/*
 * IEEE 754 optimized version that returns only what's needed for
 * floating-point multiplication: high 128 bits and sticky bit.
 */
mul128_ieee754_result mul128_karatsuba_ieee754(uint64_t a_lo, uint64_t a_hi,
                                                uint64_t b_lo, uint64_t b_hi);

/*
 * For testing: Naive 128×128 multiplication using 16 32×32 multiplies.
 * Used to verify Karatsuba produces identical results.
 */
uint256_t mul128_naive(uint64_t a_lo, uint64_t a_hi,
                       uint64_t b_lo, uint64_t b_hi);
