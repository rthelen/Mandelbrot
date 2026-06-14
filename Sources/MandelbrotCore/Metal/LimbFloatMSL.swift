import Foundation

/// Generates MSL for a parameterized limb-float: a software floating type whose
/// mantissa is `K` 32-bit limbs (little-endian, m[0] = least significant),
/// left-justified with the leading 1 explicit at the top bit of m[K-1]; the
/// exponent is an **unbiased native int**; the sign is a separate bit.
///
/// One template, instantiated per width by changing `K` — so F96 (K=3), F128
/// (K=4), F160 (K=5) all come from this. With `expBits`/`bias` set to the IEEE
/// values, rounding to `W = 32*K - expBits` bits makes it bit-compatible with
/// binary64 (K=2) / binary128 (K=4), which is how we validate it.
///
/// Design discipline (per the unbiased exponent): a value is zero **iff its
/// mantissa limbs are all zero** — the exponent is never tested for zero-ness.
/// Bias appears only at the IEEE pack/unpack boundary.
func limbFloatMSL(tag: String, K: Int, expBits: Int, bias: Int) -> String {
    let P = expBits          // bits dropped when rounding (W = 32K - expBits)
    return """

// ======== LimbFloat \(tag): K=\(K) limbs, expBits=\(expBits), bias=\(bias) ========
struct LF\(tag) { uint m[\(K)]; int e; uint s; };

static bool lf\(tag)_zero(thread const uint m[\(K)]) {
    uint o = 0;
    for (int i = 0; i < \(K); ++i) o |= m[i];
    return o == 0u;
}
static int lf\(tag)_clz(thread const uint m[\(K)]) {
    int c = 0;
    for (int i = \(K) - 1; i >= 0; --i) {
        if (m[i] == 0u) { c += 32; } else { c += (int)clz(m[i]); break; }
    }
    return c;
}
// magnitude compare: a >= b
static bool lf\(tag)_ge(thread const uint a[\(K)], thread const uint b[\(K)]) {
    for (int i = \(K) - 1; i >= 0; --i) { if (a[i] != b[i]) return a[i] > b[i]; }
    return true;
}
static void lf\(tag)_shl(thread uint m[\(K)], int n) {
    if (n <= 0) return;
    int ls = n >> 5, bs = n & 31;
    uint r[\(K)];
    for (int i = \(K) - 1; i >= 0; --i) {
        uint hi = (i - ls >= 0) ? m[i - ls] : 0u;
        uint lo = (bs && (i - ls - 1 >= 0)) ? m[i - ls - 1] : 0u;
        r[i] = bs ? ((hi << bs) | (lo >> (32 - bs))) : hi;
    }
    for (int i = 0; i < \(K); ++i) m[i] = r[i];
}
static void lf\(tag)_shr_sticky(thread uint m[\(K)], int n, thread bool &sticky) {
    if (n <= 0) { sticky = false; return; }
    int ls = n >> 5, bs = n & 31;
    bool lost = false;
    for (int i = 0; i < ls && i < \(K); ++i) if (m[i] != 0u) lost = true;
    if (ls >= \(K)) lost = lost || false;   // everything shifted out is caught above
    if (bs && ls < \(K)) if ((m[ls] & ((1u << bs) - 1u)) != 0u) lost = true;
    uint r[\(K)];
    for (int i = 0; i < \(K); ++i) {
        uint lo = (i + ls < \(K)) ? m[i + ls] : 0u;
        uint hi = (bs && (i + ls + 1 < \(K))) ? m[i + ls + 1] : 0u;
        r[i] = bs ? ((lo >> bs) | (hi << (32 - bs))) : lo;
    }
    for (int i = 0; i < \(K); ++i) m[i] = r[i];
    sticky = lost;
}
// Round a normalized mantissa (leading 1 at bit 32K-1) to W = 32K-\(P) bits,
// dropping the low \(P) bits with round-to-nearest-even. `incoming` folds in
// sticky from below the limb array (e.g. low product limbs). May carry into the
// leading bit (overflow), in which case shift right 1 and bump the exponent.
static void lf\(tag)_round(thread uint m[\(K)], thread int &e, bool incoming) {
    const int Pp = \(P);
    if (Pp <= 0) return;
    int rbLimb = (Pp - 1) >> 5, rbBit = (Pp - 1) & 31;
    uint roundBit = (m[rbLimb] >> rbBit) & 1u;
    uint sticky = incoming ? 1u : 0u;
    for (int i = 0; i < rbLimb; ++i) sticky |= m[i];
    if (rbBit > 0) sticky |= (m[rbLimb] & ((1u << rbBit) - 1u));
    // clear the low Pp bits
    int ulpLimb = Pp >> 5, ulpBit = Pp & 31;
    for (int i = 0; i < ulpLimb; ++i) m[i] = 0u;
    if (ulpBit > 0) m[ulpLimb] &= ~((1u << ulpBit) - 1u);
    uint ulp = (m[ulpLimb] >> ulpBit) & 1u;
    if (roundBit && (ulp || sticky != 0u)) {
        ulong c = (ulong)1u << ulpBit;
        for (int i = ulpLimb; i < \(K) && c != 0; ++i) {
            ulong t = (ulong)m[i] + c; m[i] = (uint)t; c = t >> 32;
        }
        if (c != 0) {                 // carried out the top: mantissa overflowed -> 1.000..*2^(e+1)
            for (int i = 0; i < \(K); ++i) m[i] = 0u;
            m[\(K) - 1] = 0x80000000u;
            e += 1;
        }
    }
}

static LF\(tag) lf\(tag)_zeroval(uint sign) {
    LF\(tag) z; for (int i = 0; i < \(K); ++i) z.m[i] = 0u; z.e = 0; z.s = sign; return z;
}

static LF\(tag) lf\(tag)_add(LF\(tag) a, LF\(tag) b) {
    if (lf\(tag)_zero(a.m)) return b;
    if (lf\(tag)_zero(b.m)) return a;
    bool aLarger = (a.e != b.e) ? (a.e > b.e) : lf\(tag)_ge(a.m, b.m);
    LF\(tag) L = aLarger ? a : b;
    LF\(tag) R = aLarger ? b : a;

    uint lm[\(K)], rm[\(K)];
    for (int i = 0; i < \(K); ++i) { lm[i] = L.m[i]; rm[i] = R.m[i]; }
    // shift both right 1 for carry room (matches the scalar reference)
    bool dummy;
    lf\(tag)_shr_sticky(lm, 1, dummy);
    lf\(tag)_shr_sticky(rm, 1, dummy);
    bool alignSticky;
    lf\(tag)_shr_sticky(rm, L.e - R.e, alignSticky);
    if (alignSticky) rm[0] |= 1u;

    uint res[\(K)];
    bool sameSign = (L.s == R.s);
    if (sameSign) {
        ulong c = 0;
        for (int i = 0; i < \(K); ++i) { ulong t = (ulong)lm[i] + rm[i] + c; res[i] = (uint)t; c = t >> 32; }
    } else {
        ulong borrow = 0;
        for (int i = 0; i < \(K); ++i) { ulong t = (ulong)lm[i] - rm[i] - borrow; res[i] = (uint)t; borrow = (t >> 32) & 1u; }
    }
    if (lf\(tag)_zero(res)) return lf\(tag)_zeroval(0u);

    int e = L.e;
    int lz = lf\(tag)_clz(res);
    if (lz == 0) e += 1;
    else if (lz > 1) e -= (lz - 1);
    lf\(tag)_shl(res, lz);
    lf\(tag)_round(res, e, false);

    LF\(tag) out; for (int i = 0; i < \(K); ++i) out.m[i] = res[i]; out.e = e; out.s = L.s;
    return out;
}

static LF\(tag) lf\(tag)_mul(LF\(tag) a, LF\(tag) b) {
    uint sign = a.s ^ b.s;
    if (lf\(tag)_zero(a.m) || lf\(tag)_zero(b.m)) return lf\(tag)_zeroval(sign);

    uint p[2 * \(K)];
    for (int i = 0; i < 2 * \(K); ++i) p[i] = 0u;
    for (int i = 0; i < \(K); ++i) {
        ulong carry = 0;
        for (int j = 0; j < \(K); ++j) {
            ulong cur = (ulong)a.m[i] * (ulong)b.m[j] + (ulong)p[i + j] + carry;
            p[i + j] = (uint)cur; carry = cur >> 32;
        }
        int k = i + \(K);
        while (carry != 0 && k < 2 * \(K)) { ulong t = (ulong)p[k] + carry; p[k] = (uint)t; carry = t >> 32; ++k; }
    }

    // top K limbs = high half; leading 1 is at bit 64K-1 or 64K-2
    uint hi[\(K)], lo[\(K)];
    for (int i = 0; i < \(K); ++i) { hi[i] = p[\(K) + i]; lo[i] = p[i]; }

    int e = a.e + b.e;
    bool sticky;
    if ((hi[\(K) - 1] >> 31) & 1u) {           // leading 1 at bit 64K-1
        e += 1;
        sticky = !lf\(tag)_zero(lo);
    } else {                                    // leading 1 at bit 64K-2: shift hi:lo left 1
        uint loMsb = (lo[\(K) - 1] >> 31) & 1u;
        lf\(tag)_shl(hi, 1);
        hi[0] |= loMsb;
        uint loClear[\(K)];
        for (int i = 0; i < \(K); ++i) loClear[i] = lo[i];
        loClear[\(K) - 1] &= 0x7FFFFFFFu;       // drop the bit we shifted in
        sticky = !lf\(tag)_zero(loClear);
    }
    lf\(tag)_round(hi, e, sticky);

    LF\(tag) out; for (int i = 0; i < \(K); ++i) out.m[i] = hi[i]; out.e = e; out.s = sign;
    return out;
}

// ---- IEEE pack/unpack (only at the I/O boundary) ----
static LF\(tag) lf\(tag)_unpack(thread const uint in[\(K)]) {
    LF\(tag) r;
    uint top = in[\(K) - 1];
    r.s = top >> 31;
    uint expField = (top >> (31 - \(expBits))) & ((1u << \(expBits)) - 1u);
    if (expField == 0u) { for (int i = 0; i < \(K); ++i) r.m[i] = 0u; r.e = 0; return r; }
    for (int i = 0; i < \(K); ++i) r.m[i] = in[i];
    r.m[\(K) - 1] &= ((1u << (31 - \(expBits))) - 1u);     // clear sign+exp
    r.m[\(K) - 1] |= (1u << (31 - \(expBits)));            // implied bit
    lf\(tag)_shl(r.m, \(expBits));                          // left-justify: leading 1 -> bit 32K-1
    r.e = (int)expField - \(bias);
    return r;
}
static void lf\(tag)_pack(LF\(tag) a, thread uint out[\(K)]) {
    if (lf\(tag)_zero(a.m)) { for (int i = 0; i < \(K); ++i) out[i] = 0u; out[\(K) - 1] = a.s << 31; return; }
    uint t[\(K)];
    for (int i = 0; i < \(K); ++i) t[i] = a.m[i];
    bool dummy;
    lf\(tag)_shr_sticky(t, \(expBits), dummy);              // implied bit -> bit (31-expBits) of top
    int biased = a.e + \(bias);
    if (biased <= 0) { for (int i = 0; i < \(K); ++i) out[i] = 0u; out[\(K) - 1] = a.s << 31; return; }
    if (biased >= ((1 << \(expBits)) - 1)) {                // overflow: saturate to largest finite
        for (int i = 0; i < \(K); ++i) out[i] = 0xFFFFFFFFu;
        out[\(K) - 1] = (a.s << 31) | ((uint)((1 << \(expBits)) - 2) << (31 - \(expBits))) | ((1u << (31 - \(expBits))) - 1u);
        return;
    }
    for (int i = 0; i < \(K); ++i) out[i] = t[i];
    out[\(K) - 1] &= ((1u << (31 - \(expBits))) - 1u);      // clear implied bit + above
    out[\(K) - 1] |= (a.s << 31) | ((uint)biased << (31 - \(expBits)));
}

kernel void lf\(tag)_op_test(device const uint *a      [[buffer(0)]],
                             device const uint *b      [[buffer(1)]],
                             device uint       *outAdd [[buffer(2)]],
                             device uint       *outMul [[buffer(3)]],
                             constant uint     &n      [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint ab[\(K)], bb[\(K)], oa[\(K)], om[\(K)];
    for (int i = 0; i < \(K); ++i) { ab[i] = a[gid * \(K) + i]; bb[i] = b[gid * \(K) + i]; }
    LF\(tag) av = lf\(tag)_unpack(ab);
    LF\(tag) bv = lf\(tag)_unpack(bb);
    lf\(tag)_pack(lf\(tag)_add(av, bv), oa);
    lf\(tag)_pack(lf\(tag)_mul(av, bv), om);
    for (int i = 0; i < \(K); ++i) { outAdd[gid * \(K) + i] = oa[i]; outMul[gid * \(K) + i] = om[i]; }
}
"""
}

/// The two instantiations we validate against the existing references.
let limbFloat4MSL = limbFloatMSL(tag: "4", K: 4, expBits: 15, bias: 16383)   // binary128
let limbFloat2MSL = limbFloatMSL(tag: "2", K: 2, expBits: 11, bias: 1023)    // binary64

/// Mandelbrot iteration kept entirely in unpacked LF4 form — values are LF4
/// structs across the whole loop, never packed to bits between ops. This is the
/// point of the format: the per-op unpack/repack tax is gone.
let mandelbrotLF4MSL = """

static LF4 lf4_sub(LF4 a, LF4 b) { b.s ^= 1u; return lf4_add(a, b); }
// 2*a: increment the unbiased exponent (exact, no rounding). Zero stays zero.
static LF4 lf4_double(LF4 a) { if (!lf4_zero(a.m)) a.e += 1; return a; }
// a > b for non-negative a, b. Zero handled via the mantissa, not the exponent.
static bool lf4_gt_pos(LF4 a, LF4 b) {
    if (lf4_zero(a.m)) return false;
    if (a.e != b.e) return a.e > b.e;
    for (int i = 3; i >= 0; --i) if (a.m[i] != b.m[i]) return a.m[i] > b.m[i];
    return false;
}
static float lf4_to_float(LF4 a) {
    if (lf4_zero(a.m)) return 0.0f;
    float m = 1.0f + (float)((a.m[3] & 0x7FFFFFFFu) >> 8) * (1.0f / 8388608.0f);
    return m * exp2((float)a.e);
}
constant float LF4_LOG2 = 0.69314718055994531f;

kernel void mandelbrot_lf4(device const uint *cxArr   [[buffer(0)]],
                           device const uint *cyArr   [[buffer(1)]],
                           constant uint2    &dims    [[buffer(2)]],
                           constant uint     &maxIter [[buffer(3)]],
                           device uint       *outIter [[buffer(4)]],
                           device float      *outSmooth [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dims.x || gid.y >= dims.y) return;
    uint cxl[4], cyl[4];
    for (int i = 0; i < 4; ++i) { cxl[i] = cxArr[gid.x * 4 + i]; cyl[i] = cyArr[gid.y * 4 + i]; }
    LF4 cx = lf4_unpack(cxl);
    LF4 cy = lf4_unpack(cyl);
    LF4 four; four.m[0] = 0; four.m[1] = 0; four.m[2] = 0; four.m[3] = 0x80000000u; four.e = 2; four.s = 0;

    LF4 zx = lf4_zeroval(0u), zy = lf4_zeroval(0u), magSq = lf4_zeroval(0u);
    uint n = 0;
    while (n < maxIter) {
        LF4 zx2 = lf4_mul(zx, zx);
        LF4 zy2 = lf4_mul(zy, zy);
        magSq = lf4_add(zx2, zy2);
        if (lf4_gt_pos(magSq, four)) break;
        LF4 nzx = lf4_add(lf4_sub(zx2, zy2), cx);
        LF4 nzy = lf4_add(lf4_double(lf4_mul(zx, zy)), cy);
        zx = nzx; zy = nzy;
        n += 1;
    }
    uint idx = gid.y * dims.x + gid.x;
    if (n < maxIter) {
        outIter[idx] = n;
        float mag = lf4_to_float(magSq);
        float logZn = 0.5f * log(mag);
        outSmooth[idx] = 1.0f - log(logZn / LF4_LOG2) / LF4_LOG2;
    } else {
        outIter[idx] = 0xFFFFFFFF;
        outSmooth[idx] = 0.0f;
    }
}
"""
