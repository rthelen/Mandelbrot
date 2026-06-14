import Foundation

/// MSL source for software binary128 (`Float128`) arithmetic — a port of
/// `IEEE754_128.swift`. The 128-bit mantissa is a `(hi, lo)` pair of `ulong`s
/// (no native 128-bit ints on the GPU), and the 128x128->256 mantissa multiply
/// is built from 64x64->128 limb products (`sd_mul64`, defined in the soft-64
/// source this is concatenated after). NO_NAN_INF semantics, identical to the
/// CPU port, so GPU results are bit-identical to the CPU `Float128`.
///
/// This string has NO `#include` — it is appended to `softDouble64MSLSource`
/// and compiled as one translation unit.
let float128MSLSource = #"""

// ---- 128-bit unsigned integer as a (hi, lo) limb pair ----
struct u128 { ulong hi; ulong lo; };

static u128 u128_make(ulong hi, ulong lo) { u128 r; r.hi = hi; r.lo = lo; return r; }
static u128 u128_and(u128 a, u128 b) { return u128_make(a.hi & b.hi, a.lo & b.lo); }
static u128 u128_or(u128 a, u128 b)  { return u128_make(a.hi | b.hi, a.lo | b.lo); }
static bool u128_is_zero(u128 a) { return (a.hi | a.lo) == 0; }
static bool u128_ge(u128 a, u128 b) { return (a.hi != b.hi) ? (a.hi > b.hi) : (a.lo >= b.lo); }
static bool u128_gt(u128 a, u128 b) { return (a.hi != b.hi) ? (a.hi > b.hi) : (a.lo > b.lo); }
static int  u128_clz(u128 a) { return (a.hi != 0) ? (int)clz(a.hi) : 64 + (int)clz(a.lo); }

static u128 u128_add(u128 a, u128 b) {
    ulong lo = a.lo + b.lo;
    ulong carry = (lo < a.lo) ? 1 : 0;
    return u128_make(a.hi + b.hi + carry, lo);
}
static u128 u128_sub(u128 a, u128 b) {   // a - b, assumes a >= b
    ulong borrow = (a.lo < b.lo) ? 1 : 0;
    return u128_make(a.hi - b.hi - borrow, a.lo - b.lo);
}
static u128 u128_shl(u128 a, int n) {    // 0..127
    if (n == 0) return a;
    if (n >= 64) return u128_make(a.lo << (n - 64), 0);
    return u128_make((a.hi << n) | (a.lo >> (64 - n)), a.lo << n);
}
static u128 u128_shr(u128 a, int n) {    // 0..127, logical
    if (n == 0) return a;
    if (n >= 64) return u128_make(0, a.hi >> (n - 64));
    return u128_make(a.hi >> n, (a.lo >> n) | (a.hi << (64 - n)));
}
// Right shift tracking whether any 1 bit was lost (sticky).
static u128 u128_shr_sticky(u128 a, int n, thread bool &sticky) {
    if (n == 0) { sticky = false; return a; }
    if (n >= 128) { sticky = !u128_is_zero(a); return u128_make(0, 0); }
    u128 keep = u128_shr(a, n);
    // lost = a & ((1<<n)-1)
    u128 lost;
    if (n >= 64) {
        ulong m = (n == 64) ? ~(ulong)0 : ((((ulong)1) << (n - 64)) - 1);
        lost = u128_make(a.hi & m, a.lo);
    } else {
        lost = u128_make(0, a.lo & ((((ulong)1) << n) - 1));
    }
    sticky = !u128_is_zero(lost);
    return keep;
}

// 128x128 -> 256, returned as hi (bits 255..128) and lo (bits 127..0).
struct u256 { u128 hi; u128 lo; };
static u256 u128_mul_full(u128 a, u128 b) {
    ulong h, l;
    sd_mul64(a.lo, b.lo, h, l); ulong p00h = h, p00l = l;
    sd_mul64(a.lo, b.hi, h, l); ulong p01h = h, p01l = l;
    sd_mul64(a.hi, b.lo, h, l); ulong p10h = h, p10l = l;
    sd_mul64(a.hi, b.hi, h, l); ulong p11h = h, p11l = l;

    ulong w0 = p00l;

    // column 1: p00h + p01l + p10l
    ulong s = p00h, t;
    t = s + p01l; ulong c1 = (t < s) ? 1 : 0; s = t;
    t = s + p10l; ulong c2 = (t < s) ? 1 : 0; s = t;
    ulong w1 = s; ulong carry1 = c1 + c2;

    // column 2: p01h + p10h + p11l + carry1
    s = p01h;
    t = s + p10h; ulong c3 = (t < s) ? 1 : 0; s = t;
    t = s + p11l; ulong c4 = (t < s) ? 1 : 0; s = t;
    t = s + carry1; ulong c5 = (t < s) ? 1 : 0; s = t;
    ulong w2 = s; ulong carry2 = c3 + c4 + c5;

    // column 3: p11h + carry2
    ulong w3 = p11h + carry2;

    u256 r;
    r.lo = u128_make(w1, w0);
    r.hi = u128_make(w3, w2);
    return r;
}

// ---- binary128 constants & format ----
constant int  F128_BIAS    = 16383;
constant uint F128_EXP_MAX = 32767;
// mantissa mask = (1<<112)-1: low 64 bits all set + hi 48 bits set
constant ulong F128_MM_HI  = ((((ulong)1) << 48) - 1);
// sign bit 127, implied bit 112 (hi bit 48)
// (built inline where needed)

struct Parts128 { bool sign; int exp; u128 mant; };  // mant left-justified, leading 1 at bit 127

static u128 f128_mant_mask() { return u128_make(F128_MM_HI, ~(ulong)0); }
static u128 f128_sign_mask() { return u128_make(((ulong)1) << 63, 0); }
static u128 f128_implied()   { return u128_make(((ulong)1) << 48, 0); }   // 1 << 112

static Parts128 f128_unpack(u128 bits) {
    Parts128 p;
    p.sign = ((bits.hi >> 63) & 1) != 0;
    uint expb = (uint)((bits.hi >> 48) & 0x7FFF);
    u128 mant = u128_and(bits, f128_mant_mask());
    if (expb == 0) { p.exp = 0; p.mant = u128_make(0, 0); return p; }
    u128 full = u128_or(mant, f128_implied());   // (1<<112) | mant, leading 1 at bit 112
    p.exp = (int)expb - F128_BIAS;
    p.mant = u128_shl(full, 15);                 // left-justify: leading 1 at bit 127
    return p;
}

static u128 f128_pack(Parts128 p, bool stickyIn) {
    if (u128_is_zero(p.mant)) return p.sign ? f128_sign_mask() : u128_make(0, 0);

    bool sticky = stickyIn || ((p.mant.lo & 1) != 0);
    u128 mant = u128_shr(p.mant, 1);   // bit 127 carry slot, 126 implied
    int e = p.exp;
    // bits 12..0 = sticky, bit 13 = round, then >>14 -> implied at 112
    ulong stickyMask = (((ulong)1) << 13) - 1;
    sticky = sticky || ((mant.lo & stickyMask) != 0);
    bool roundBit = ((mant.lo >> 13) & 1) != 0;
    mant = u128_shr(mant, 14);

    bool ulp = (mant.lo & 1) != 0;
    if (roundBit && (ulp || sticky)) {
        mant = u128_add(mant, u128_make(0, 1));
        // did the implied bit move 112 -> 113? bit 113 = hi bit 49
        if (((mant.hi >> 49) & 1) != 0) { mant = u128_shr(mant, 1); e += 1; }
    }

    int biased = e + F128_BIAS;
    if (biased <= 0) return p.sign ? f128_sign_mask() : u128_make(0, 0);
    if (biased >= (int)F128_EXP_MAX) {
        u128 b = u128_or(u128_make(((ulong)(F128_EXP_MAX - 1)) << 48, 0), f128_mant_mask());
        return p.sign ? u128_or(b, f128_sign_mask()) : b;
    }
    u128 expPart = u128_make(((ulong)((uint)biased & 0x7FFF)) << 48, 0);
    u128 b = u128_or(expPart, u128_and(mant, f128_mant_mask()));
    return p.sign ? u128_or(b, f128_sign_mask()) : b;
}

static Parts128 f128_add_parts(Parts128 a, Parts128 b, thread bool &stickyOut) {
    stickyOut = false;
    if (u128_is_zero(a.mant)) return b;
    if (u128_is_zero(b.mant)) return a;

    bool lhsLarger = (a.exp != b.exp) ? (a.exp > b.exp) : u128_ge(a.mant, b.mant);
    Parts128 lhs = lhsLarger ? a : b;
    Parts128 rhs = lhsLarger ? b : a;

    u128 lhsM = u128_shr(lhs.mant, 1);
    u128 rhsM = u128_shr(rhs.mant, 1);
    int diff = lhs.exp - rhs.exp;
    bool alignSticky;
    rhsM = u128_shr_sticky(rhsM, diff, alignSticky);
    if (alignSticky) rhsM = u128_or(rhsM, u128_make(0, 1));   // fold sticky before combine

    bool sameSign = (lhs.sign == rhs.sign);
    u128 res = sameSign ? u128_add(lhsM, rhsM) : u128_sub(lhsM, rhsM);
    if (u128_is_zero(res)) { Parts128 z; z.sign = false; z.exp = 0; z.mant = u128_make(0, 0); return z; }

    int e = lhs.exp;
    int lz = u128_clz(res);
    if (lz == 0) e += 1;
    else if (lz > 1) e -= (lz - 1);

    Parts128 r; r.sign = lhs.sign; r.exp = e; r.mant = u128_shl(res, lz);
    return r;
}

static Parts128 f128_mul_parts(Parts128 a, Parts128 b, thread bool &stickyOut) {
    bool sign = (a.sign != b.sign);
    if (u128_is_zero(a.mant) || u128_is_zero(b.mant)) {
        stickyOut = false;
        Parts128 z; z.sign = sign; z.exp = 0; z.mant = u128_make(0, 0); return z;
    }
    u256 prod = u128_mul_full(a.mant, b.mant);
    u128 resMant = prod.hi;
    int e = a.exp + b.exp;
    bool sticky;
    if (((resMant.hi >> 63) & 1) != 0) {     // leading 1 at bit 255
        e += 1;
        sticky = !u128_is_zero(prod.lo);
    } else {                                  // leading 1 at bit 254: shift left 1
        ulong loMsb = (prod.lo.hi >> 63) & 1;
        resMant = u128_or(u128_shl(resMant, 1), u128_make(0, loMsb));
        u128 loCleared = u128_make(prod.lo.hi & ~(((ulong)1) << 63), prod.lo.lo);
        sticky = !u128_is_zero(loCleared);
    }
    Parts128 r; r.sign = sign; r.exp = e; r.mant = resMant; stickyOut = sticky; return r;
}

static u128 f128_add(u128 x, u128 y) {
    bool s; Parts128 r = f128_add_parts(f128_unpack(x), f128_unpack(y), s); return f128_pack(r, s);
}
static u128 f128_sub(u128 x, u128 y) {
    Parts128 py = f128_unpack(y); py.sign = !py.sign;
    bool s; Parts128 r = f128_add_parts(f128_unpack(x), py, s); return f128_pack(r, s);
}
static u128 f128_mul(u128 x, u128 y) {
    bool s; Parts128 r = f128_mul_parts(f128_unpack(x), f128_unpack(y), s); return f128_pack(r, s);
}

// Approximate float of a non-negative binary128 value, for display-only smooth.
static float f128_to_float(u128 bits) {
    uint expb = (uint)((bits.hi >> 48) & 0x7FFF);
    if (expb == 0) return 0.0f;
    float m = 1.0f + (float)((bits.hi >> 25) & 0x7FFFFF) * (1.0f / 8388608.0f);  // top 23 mantissa bits
    return m * exp2((float)((int)expb - F128_BIAS));
}

constant ulong F128_TWO_HI  = 0x4000000000000000;  // 2.0  -> exp 16384
constant ulong F128_FOUR_HI = 0x4001000000000000;  // 4.0  -> exp 16385
constant float F128_LOG2 = 0.69314718055994531f;

// Mandelbrot iteration in software binary128. cx/cy are precomputed on the CPU
// in full precision (2 ulongs each, hi then lo), so this reproduces the CPU
// Float128StripKernel bit-for-bit.
kernel void mandelbrot_f128(device const ulong *cxArr   [[buffer(0)]],
                            device const ulong *cyArr   [[buffer(1)]],
                            constant uint2     &dims     [[buffer(2)]],
                            constant uint      &maxIter  [[buffer(3)]],
                            device uint        *outIter  [[buffer(4)]],
                            device float       *outSmooth [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dims.x || gid.y >= dims.y) return;
    u128 cx = u128_make(cxArr[2 * gid.x], cxArr[2 * gid.x + 1]);
    u128 cy = u128_make(cyArr[2 * gid.y], cyArr[2 * gid.y + 1]);
    u128 four = u128_make(F128_FOUR_HI, 0);
    u128 two  = u128_make(F128_TWO_HI, 0);

    u128 zx = u128_make(0, 0), zy = u128_make(0, 0), magSq = u128_make(0, 0);
    uint n = 0;
    while (n < maxIter) {
        u128 zx2 = f128_mul(zx, zx);
        u128 zy2 = f128_mul(zy, zy);
        magSq = f128_add(zx2, zy2);
        if (u128_gt(magSq, four)) break;
        u128 nzx = f128_add(f128_sub(zx2, zy2), cx);
        u128 nzy = f128_add(f128_mul(two, f128_mul(zx, zy)), cy);
        zx = nzx; zy = nzy;
        n += 1;
    }

    uint idx = gid.y * dims.x + gid.x;
    if (n < maxIter) {
        outIter[idx] = n;
        float mag = f128_to_float(magSq);
        float logZn = 0.5f * log(mag);
        outSmooth[idx] = 1.0f - log(logZn / F128_LOG2) / F128_LOG2;
    } else {
        outIter[idx] = 0xFFFFFFFF;
        outSmooth[idx] = 0.0f;
    }
}

// Arithmetic self-test: a+b and a*b in software binary128 (2 ulongs per value).
kernel void f128_op_test(device const ulong *a      [[buffer(0)]],
                         device const ulong *b      [[buffer(1)]],
                         device ulong       *outAdd [[buffer(2)]],
                         device ulong       *outMul [[buffer(3)]],
                         constant uint      &n      [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    u128 av = u128_make(a[2 * gid], a[2 * gid + 1]);
    u128 bv = u128_make(b[2 * gid], b[2 * gid + 1]);
    u128 s = f128_add(av, bv);
    u128 m = f128_mul(av, bv);
    outAdd[2 * gid] = s.hi; outAdd[2 * gid + 1] = s.lo;
    outMul[2 * gid] = m.hi; outMul[2 * gid + 1] = m.lo;
}
"""#
