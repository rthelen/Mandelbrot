#include "cmandelbrot.h"

// Software binary128 in C, kept unpacked across the iteration. 1:1 port of the
// Swift Float128 ops (IEEE754_128.swift) — same rounding, same sticky-fold, same
// NO_NAN_INF semantics — so it is bit-exact. __uint128_t lets clang lower the
// 64x64->128 limb products to MUL+UMULH and the carries to ADCS.

typedef unsigned __int128 u128;

typedef struct { int sign; int exp; u128 mant; } parts;   // mant: leading 1 at bit 127

#define BIAS     16383
#define MANTMASK ((((u128)1) << 112) - 1)
#define IMPLIED  (((u128)1) << 112)
#define SIGNMASK (((u128)1) << 127)
#define LEAD127  (((u128)1) << 127)

static inline parts cf_unpack(u128 bits) {
    parts p;
    p.sign = (int)((bits >> 127) & 1);
    uint32_t expb = (uint32_t)((bits >> 112) & 0x7FFF);
    u128 m = bits & MANTMASK;
    if (expb == 0) { p.exp = 0; p.mant = 0; return p; }
    p.exp = (int)expb - BIAS;
    p.mant = (IMPLIED | m) << 15;          // left-justify: leading 1 at bit 127
    return p;
}

static inline u128 cf_pack(parts p) {
    if (p.mant == 0) return p.sign ? SIGNMASK : 0;
    u128 m = p.mant >> 15;                  // implied bit back to position 112
    int biased = p.exp + BIAS;              // magSq is bounded; no under/overflow path needed
    u128 bits = ((u128)(uint32_t)(biased & 0x7FFF) << 112) | (m & MANTMASK);
    if (p.sign) bits |= SIGNMASK;
    return bits;
}

// Round a normalized mantissa (leading 1 at bit 127) + incoming sticky to 113
// bits, staying in parts form. Mirrors bitsFromParts, branchless.
static inline parts cf_round(int sign, int exp, u128 mant, int sticky_in) {
    if (mant == 0) { parts z = {sign, exp, 0}; return z; }
    int sticky = sticky_in || (int)(mant & 1);
    mant >>= 1;                             // bit 127 carry slot, 126 implied
    sticky = sticky || ((mant & ((((u128)1) << 13) - 1)) != 0);
    int roundBit = (int)((mant >> 13) & 1);
    mant >>= 14;                            // leading 1 at bit 112
    int ulp = (int)(mant & 1);
    u128 bumped = (roundBit && (ulp || sticky)) ? mant + 1 : mant;
    int cascade = (int)((bumped >> 113) & 1);
    mant = cascade ? (bumped >> 1) : bumped;
    exp  = cascade ? exp + 1 : exp;
    parts r = { sign, exp, mant << 15 };    // re-justify to bit 127
    return r;
}

static inline parts cf_mul(parts a, parts b) {
    int sign = a.sign ^ b.sign;
    if (a.mant == 0 || b.mant == 0) { parts z = { sign, 0, 0 }; return z; }

    u128 A = a.mant, B = b.mant;
    uint64_t a0 = (uint64_t)A, a1 = (uint64_t)(A >> 64);
    uint64_t b0 = (uint64_t)B, b1 = (uint64_t)(B >> 64);
    u128 p00 = (u128)a0 * b0, p01 = (u128)a0 * b1;
    u128 p10 = (u128)a1 * b0, p11 = (u128)a1 * b1;

    // 256-bit product = p00 + (p01 + p10)<<64 + p11<<128, in {hi, lo}.
    u128 lo = p00, hi = p11;
    uint64_t t_l = (uint64_t)p01, t_h = (uint64_t)(p01 >> 64);
    u128 t = lo + (((u128)t_l) << 64); hi += (u128)t_h + (t < lo); lo = t;
    t_l = (uint64_t)p10; t_h = (uint64_t)(p10 >> 64);
    t = lo + (((u128)t_l) << 64); hi += (u128)t_h + (t < lo); lo = t;

    int exp = a.exp + b.exp;
    int topSet = (int)((hi >> 127) & 1);
    uint64_t loMsb = (uint64_t)((lo >> 127) & 1);
    u128 resMant = topSet ? hi : ((hi << 1) | loMsb);
    exp += topSet ? 1 : 0;
    int sticky = topSet ? (lo != 0) : ((lo & (LEAD127 - 1)) != 0);
    return cf_round(sign, exp, resMant, sticky);
}

static inline int cf_clz128(u128 x) {
    uint64_t hi = (uint64_t)(x >> 64), lo = (uint64_t)x;
    return hi != 0 ? __builtin_clzll(hi) : 64 + __builtin_clzll(lo);
}

static inline parts cf_add(parts a, parts b) {
    if (a.mant == 0) return b;
    if (b.mant == 0) return a;
    int aLarger = (a.exp != b.exp) ? (a.exp > b.exp) : (a.mant >= b.mant);
    parts L = aLarger ? a : b, R = aLarger ? b : a;

    u128 lhsM = L.mant >> 1, rhsM = R.mant >> 1;
    int diff = L.exp - R.exp;
    int alignSticky = 0;
    if (diff > 0) {
        if (diff >= 128) { alignSticky = (rhsM != 0); rhsM = 0; }
        else { alignSticky = ((rhsM & ((((u128)1) << diff) - 1)) != 0); rhsM >>= diff; }
    }
    rhsM |= (u128)(alignSticky ? 1 : 0);

    u128 res = (L.sign == R.sign) ? (lhsM + rhsM) : (lhsM - rhsM);
    if (res == 0) { parts z = { 0, 0, 0 }; return z; }

    int lz = cf_clz128(res);
    int exp = L.exp + (lz == 0 ? 1 : -(lz - 1));
    res <<= lz;
    return cf_round(L.sign, exp, res, 0);
}

static inline parts cf_sub(parts a, parts b) { b.sign ^= 1; return cf_add(a, b); }

uint32_t cf128_mandelbrot_pixel(uint64_t cx_lo, uint64_t cx_hi,
                                uint64_t cy_lo, uint64_t cy_hi,
                                uint32_t max_iter,
                                uint64_t *magsq_lo, uint64_t *magsq_hi) {
    parts cx = cf_unpack(((u128)cx_hi << 64) | cx_lo);
    parts cy = cf_unpack(((u128)cy_hi << 64) | cy_lo);
    parts two = { 0, 1, LEAD127 };                      // 2.0
    parts zx = { 0, 0, 0 }, zy = { 0, 0, 0 }, magSq = { 0, 0, 0 };

    uint32_t n = 0;
    while (n < max_iter) {
        parts zx2 = cf_mul(zx, zx);
        parts zy2 = cf_mul(zy, zy);
        magSq = cf_add(zx2, zy2);
        // magSq > 4.0 ? (both non-negative; 4.0 has exp 2, mant = 1<<127)
        if (magSq.exp > 2 || (magSq.exp == 2 && magSq.mant > LEAD127)) break;
        parts nzx = cf_add(cf_sub(zx2, zy2), cx);
        parts nzy = cf_add(cf_mul(cf_mul(two, zx), zy), cy);   // ((2*zx)*zy)+cy
        zx = nzx; zy = nzy;
        n++;
    }

    if (n < max_iter) {
        u128 ms = cf_pack(magSq);
        *magsq_lo = (uint64_t)ms;
        *magsq_hi = (uint64_t)(ms >> 64);
        return n;
    }
    *magsq_lo = 0; *magsq_hi = 0;
    return 0xFFFFFFFFu;
}
