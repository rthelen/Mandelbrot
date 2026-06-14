import Foundation

/// Metal Shading Language source for software binary64 (`SoftDouble`) arithmetic,
/// a direct port of `IEEE754_64.swift`. Compiled at runtime via
/// `device.makeLibrary(source:)`. The 64x64->128 mantissa multiply is done with
/// 32-bit schoolbook partial products (the C++ reference's no-128-bit-int path)
/// since GPUs have no native 128-bit integers — and `ulong` has no widening
/// multiply intrinsic in MSL.
///
/// NO_NAN_INF semantics, identical to the CPU port: subnormals flush to zero,
/// over/underflow saturate/flush. The arithmetic is pure integer work, so GPU
/// results are bit-identical to the CPU `SoftDouble` (and hence hardware Double).
let softDouble64MSLSource = #"""
#include <metal_stdlib>
using namespace metal;

constant ulong SD_MANT_MASK = (((ulong)1 << 52) - 1);
constant ulong SD_SIGN_MASK = ((ulong)1 << 63);
constant int   SD_BIAS      = 1023;
constant uint  SD_EXP_MAX   = 2047;

struct Parts { bool sign; int exp; ulong mant; };  // mant left-justified, leading 1 at bit 63

static Parts sd_unpack(ulong bits) {
    Parts p;
    p.sign = ((bits >> 63) & 1) != 0;
    uint expb = (uint)((bits >> 52) & 0x7FF);
    ulong mant = bits & SD_MANT_MASK;
    if (expb == 0) { p.exp = 0; p.mant = 0; return p; }
    ulong full = ((ulong)1 << 52) | mant;
    p.exp = (int)expb - SD_BIAS;
    p.mant = full << 11;
    return p;
}

static ulong sd_pack(Parts p, bool stickyIn) {
    if (p.mant == 0) return p.sign ? SD_SIGN_MASK : 0;
    bool sticky = stickyIn || ((p.mant & 1) != 0);
    ulong mant = p.mant >> 1;
    int e = p.exp;
    ulong stickyMask = (((ulong)1 << 9) - 1);
    sticky = sticky || ((mant & stickyMask) != 0);
    bool roundBit = ((mant >> 9) & 1) != 0;
    mant >>= 10;
    bool ulp = (mant & 1) != 0;
    if (roundBit && (ulp || sticky)) {
        mant += 1;
        if (((mant >> 53) & 1) != 0) { mant >>= 1; e += 1; }
    }
    int biased = e + SD_BIAS;
    if (biased <= 0) return p.sign ? SD_SIGN_MASK : 0;
    if (biased >= (int)SD_EXP_MAX) {
        ulong b = ((ulong)(SD_EXP_MAX - 1) << 52) | SD_MANT_MASK;
        return p.sign ? (b | SD_SIGN_MASK) : b;
    }
    ulong b = ((ulong)((uint)biased & SD_EXP_MAX) << 52) | (mant & SD_MANT_MASK);
    return p.sign ? (b | SD_SIGN_MASK) : b;
}

static ulong sd_shr_sticky(ulong n, int shift, thread bool &sticky) {
    if (shift == 0) { sticky = false; return n; }
    if (shift >= 64) { sticky = (n != 0); return 0; }
    ulong hi = n >> shift;
    ulong lost = n & ((((ulong)1) << shift) - 1);
    sticky = (lost != 0);
    return hi;
}

static Parts sd_add_parts(Parts a, Parts b, thread bool &stickyOut) {
    stickyOut = false;
    if (a.mant == 0) return b;
    if (b.mant == 0) return a;

    bool lhsLarger = (a.exp != b.exp) ? (a.exp > b.exp) : (a.mant >= b.mant);
    Parts lhs = lhsLarger ? a : b;
    Parts rhs = lhsLarger ? b : a;

    ulong lhsM = lhs.mant >> 1;
    ulong rhsM = rhs.mant >> 1;
    int diff = lhs.exp - rhs.exp;
    bool alignSticky;
    rhsM = sd_shr_sticky(rhsM, diff, alignSticky);
    rhsM = rhsM | (alignSticky ? (ulong)1 : (ulong)0);   // fold sticky before combine

    bool sameSign = (lhs.sign == rhs.sign);
    ulong res = sameSign ? (lhsM + rhsM) : (lhsM - rhsM);
    if (res == 0) { Parts z; z.sign = false; z.exp = 0; z.mant = 0; return z; }

    int e = lhs.exp;
    int lz = (int)clz(res);
    if (lz == 0) e += 1;
    else if (lz > 1) e -= (lz - 1);

    Parts r; r.sign = lhs.sign; r.exp = e; r.mant = res << lz;
    return r;
}

// 64x64 -> 128 via 32-bit schoolbook with explicit carries (mirrors the C++
// reference multiply_internal non-FAST path).
static void sd_mul64(ulong A, ulong B, thread ulong &hi, thread ulong &lo) {
    ulong AL = A & 0xFFFFFFFF, AH = A >> 32;
    ulong BL = B & 0xFFFFFFFF, BH = B >> 32;
    ulong LL = AL * BL;
    ulong LH = AL * BH;
    ulong HL = AH * BL;
    ulong HH = AH * BH;

    ulong MM_a = LH + HL;            ulong carry1 = (MM_a < LH) ? 1 : 0;
    ulong MM   = MM_a + (LL >> 32);  ulong carry2 = (MM < MM_a) ? 1 : 0;
    ulong MM_C = carry1 + carry2;

    ulong retLo = (MM << 32) + (LL & 0xFFFFFFFF); ulong lowCarry = (retLo < (MM << 32)) ? 1 : 0;
    ulong hiPartial = HH + (MM >> 32);            ulong hc1 = (hiPartial < HH) ? 1 : 0;
    ulong addend = (MM_C << 32) + lowCarry;
    ulong retHi = hiPartial + addend;             // hc2 unused (fits in 128 bits)
    (void)hc1;

    hi = retHi; lo = retLo;
}

static Parts sd_mul_parts(Parts a, Parts b, thread bool &stickyOut) {
    bool sign = (a.sign != b.sign);
    if (a.mant == 0 || b.mant == 0) {
        stickyOut = false;
        Parts z; z.sign = sign; z.exp = 0; z.mant = 0; return z;
    }
    ulong hi, lo;
    sd_mul64(a.mant, b.mant, hi, lo);

    ulong resMant = hi;
    int e = a.exp + b.exp;
    bool sticky;
    if (((hi >> 63) & 1) != 0) {
        e += 1;
        sticky = (lo != 0);
    } else {
        ulong loMsb = (lo >> 63) & 1;
        resMant = (hi << 1) | loMsb;
        sticky = (lo & ((((ulong)1) << 63) - 1)) != 0;
    }
    Parts r; r.sign = sign; r.exp = e; r.mant = resMant;
    stickyOut = sticky;
    return r;
}

static ulong sd_add(ulong x, ulong y) {
    bool s; Parts r = sd_add_parts(sd_unpack(x), sd_unpack(y), s);
    return sd_pack(r, s);
}
static ulong sd_sub(ulong x, ulong y) {
    Parts py = sd_unpack(y); py.sign = !py.sign;
    bool s; Parts r = sd_add_parts(sd_unpack(x), py, s);
    return sd_pack(r, s);
}
static ulong sd_mul(ulong x, ulong y) {
    bool s; Parts r = sd_mul_parts(sd_unpack(x), sd_unpack(y), s);
    return sd_pack(r, s);
}

// Exact binary64 of a small unsigned integer (used for the per-pixel column
// offset; values are < image width, far below 2^53 so the result is exact).
static ulong sd_from_uint(uint v) {
    if (v == 0) return 0;
    int e = 31 - (int)clz(v);                 // index of MSB, 0..31
    ulong mant = (((ulong)v) << (52 - e)) & SD_MANT_MASK;
    return ((ulong)((uint)(e + SD_BIAS)) << 52) | mant;   // positive
}

// Approximate float of a non-negative software-binary64 value, for the
// display-only smoothing term (kept in float, like the CPU's smoothFraction).
static float sd_to_float(ulong bits) {
    uint expb = (uint)((bits >> 52) & 0x7FF);
    if (expb == 0) return 0.0f;
    ulong mant = bits & SD_MANT_MASK;
    float m = 1.0f + (float)(mant >> 29) * (1.0f / 8388608.0f);  // top 23 mantissa bits / 2^23
    return m * exp2((float)((int)expb - SD_BIAS));
}

constant ulong SD_FOUR = 0x4010000000000000;  // 4.0
constant float SD_LOG2 = 0.69314718055994531f;

// Exact multiply by 2.0: increment the biased exponent (no rounding). Bounded
// Mandelbrot values never overflow. Zero stays zero. Bit-identical to sd_mul(x, 2.0).
static ulong sd_double(ulong x) {
    if (((x >> 52) & 0x7FF) == 0) return x;   // zero (subnormals flushed)
    return x + (((ulong)1) << 52);
}

// Mandelbrot iteration in software binary64. Per-pixel c is reconstructed from
// per-strip x-origins and per-row y-origins (precomputed on the CPU in full
// precision), so it reproduces the CPU SoftDoubleStripKernel bit-for-bit.
kernel void mandelbrot_sd64(device const ulong *stripOriginX [[buffer(0)]],
                            device const ulong *rowOriginY    [[buffer(1)]],
                            constant ulong     &dxBits        [[buffer(2)]],
                            constant uint2     &dims          [[buffer(3)]],
                            constant uint      &maxIter       [[buffer(4)]],
                            device uint        *outIter       [[buffer(5)]],
                            device float       *outSmooth     [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dims.x || gid.y >= dims.y) return;
    uint strip = gid.x >> 5;     // / 32
    uint col   = gid.x & 31;     // % 32
    ulong cx = sd_add(stripOriginX[strip], sd_mul(dxBits, sd_from_uint(col)));
    ulong cy = rowOriginY[gid.y];

    ulong zx = 0, zy = 0, magSq = 0;
    uint n = 0;
    while (n < maxIter) {
        ulong zx2 = sd_mul(zx, zx);
        ulong zy2 = sd_mul(zy, zy);
        magSq = sd_add(zx2, zy2);
        if (magSq > SD_FOUR) break;     // both non-negative: bit order == value order
        ulong nzx = sd_add(sd_sub(zx2, zy2), cx);
        ulong nzy = sd_add(sd_double(sd_mul(zx, zy)), cy);   // 2*zx*zy via exponent bump
        zx = nzx; zy = nzy;
        n += 1;
    }

    uint idx = gid.y * dims.x + gid.x;
    if (n < maxIter) {
        outIter[idx] = n;
        float mag = sd_to_float(magSq);
        float logZn = 0.5f * log(mag);
        outSmooth[idx] = 1.0f - log(logZn / SD_LOG2) / SD_LOG2;
    } else {
        outIter[idx] = 0xFFFFFFFF;      // in-set sentinel (matches PixelResult.inSet)
        outSmooth[idx] = 0.0f;
    }
}

// Arithmetic self-test: compute a+b and a*b in software binary64 on the GPU.
kernel void sd_op_test(device const ulong *a     [[buffer(0)]],
                       device const ulong *b     [[buffer(1)]],
                       device ulong       *outAdd [[buffer(2)]],
                       device ulong       *outMul [[buffer(3)]],
                       constant uint      &n      [[buffer(4)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    outAdd[gid] = sd_add(a[gid], b[gid]);
    outMul[gid] = sd_mul(a[gid], b[gid]);
}
"""#
