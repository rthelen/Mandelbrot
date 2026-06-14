import Foundation

// Swift port of rthelen's C++ ieee754_128 (see Reference/ieee754-cxx/).
// Scalar binary128 add and multiply, using Swift's native UInt128 for the
// 128-bit mantissa. NO_NAN_INF semantics: no NaN, no Inf, no denormals.
// Inputs outside the representable range are clamped to zero (underflow) or
// saturated near the max (overflow). The caller controls the viewport so
// neither case is reached during normal Mandelbrot iteration.

// MARK: - IEEE 754 binary128 constants

@usableFromInline let kF128MantissaWidth: Int = 112
@usableFromInline let kF128ExpWidth: Int = 15
@usableFromInline let kF128ExpMaxBits: UInt32 = (1 << 15) - 1     // 32767
@usableFromInline let kF128ExpBias: Int = 16383
@usableFromInline let kF128MantissaMask: UInt128 = (UInt128(1) << 112) &- 1
@usableFromInline let kF128SignMask: UInt128 = UInt128(1) << 127

// MARK: - Internal computational representation

/// Unpacked representation used during arithmetic. Mantissa is left-justified
/// with the leading 1 at bit 127 (or all zeros for an exact zero).
@usableFromInline
struct Float128Parts {
    @usableFromInline var sign: Bool        // true = negative
    @usableFromInline var exp: Int          // unbiased
    @usableFromInline var mantissa: UInt128

    @inlinable init(sign: Bool, exp: Int, mantissa: UInt128) {
        self.sign = sign; self.exp = exp; self.mantissa = mantissa
    }
    @inlinable static var zero: Float128Parts {
        Float128Parts(sign: false, exp: 0, mantissa: 0)
    }
    @inlinable var isZero: Bool { mantissa == 0 }
}

// MARK: - Unpack

/// Decompose packed binary128 bits into computational parts. Subnormal inputs
/// are flushed to zero (NO_NAN_INF mode).
@inlinable
func partsFromBits(_ bits: UInt128) -> Float128Parts {
    let signBit = (bits >> 127) & 1 != 0
    let expBits = UInt32(truncatingIfNeeded: (bits >> 112) & 0x7FFF)
    let mantissaBits = bits & kF128MantissaMask

    if expBits == 0 {
        return Float128Parts(sign: signBit, exp: 0, mantissa: 0)
    }
    // Normal: insert implied 1 (now 113 bits), then left-justify by exp_width.
    let full = (UInt128(1) << 112) | mantissaBits
    return Float128Parts(sign: signBit,
                         exp: Int(expBits) - kF128ExpBias,
                         mantissa: full << 15)
}

// MARK: - Round and pack

/// Round-to-nearest-even and pack a `Float128Parts` (mantissa left-justified
/// with bit 127 set) into binary128 bits. Handles the carry produced by a
/// round-up cascading through the implied bit.
@inlinable
func bitsFromParts(_ p: Float128Parts, sticky stickyIn: Bool = false) -> UInt128 {
    if p.isZero {
        return p.sign ? kF128SignMask : 0
    }

    // Shift down by 1 to make bit 127 a carry slot.
    var sticky = stickyIn || (p.mantissa & 1) != 0
    var mantissa = p.mantissa >> 1
    var exp = p.exp
    // Layout: bit 127 = carry slot, bit 126 = implied 1,
    //         bits 125..14 = stored mantissa,
    //         bit 13 = round, bits 12..0 = sticky.

    let stickyMask: UInt128 = (UInt128(1) << 13) &- 1
    sticky = sticky || (mantissa & stickyMask) != 0
    let roundBit = ((mantissa >> 13) & 1) != 0
    mantissa >>= 14
    // Now: bit 113 = carry slot, bit 112 = implied 1, bits 111..0 = stored mantissa.

    let ulpBit = (mantissa & 1) != 0
    if roundBit && (ulpBit || sticky) {
        mantissa = mantissa &+ 1
        // Cascade: did the implied bit move from 112 to 113?
        if ((mantissa >> 113) & 1) != 0 {
            mantissa >>= 1
            exp += 1
        }
    }

    let biased = exp + kF128ExpBias
    if biased <= 0 {
        // Underflow flushes to zero (no denormals).
        return p.sign ? kF128SignMask : 0
    }
    if biased >= Int(kF128ExpMaxBits) {
        // Overflow saturates to the largest finite (avoid the NaN/Inf code).
        let satExp = UInt128(kF128ExpMaxBits - 1) << 112
        let satMant = kF128MantissaMask
        var bits = satExp | satMant
        if p.sign { bits |= kF128SignMask }
        return bits
    }

    let biasedU = UInt128(UInt32(biased) & kF128ExpMaxBits)
    let storedMantissa = mantissa & kF128MantissaMask
    var bits: UInt128 = (biasedU << 112) | storedMantissa
    if p.sign { bits |= kF128SignMask }
    return bits
}

// MARK: - Helpers

/// Right-shift, OR-ing any bits shifted out into a sticky flag.
@inlinable
func shiftRightTrackSticky(_ n: UInt128, by shift: Int) -> (value: UInt128, sticky: Bool) {
    if shift == 0 { return (n, false) }
    if shift >= 128 { return (0, n != 0) }
    let hi = n >> shift
    let lostMask: UInt128 = (UInt128(1) << shift) &- 1
    let lost = n & lostMask
    return (hi, lost != 0)
}

// MARK: - Add

/// Algebraic add of two parts (their signs determine add-vs-subtract).
/// Returns unpacked result with mantissa left-justified (leading 1 at bit 127),
/// plus a sticky flag carrying bits lost during alignment.
@inlinable
func addParts(_ a: Float128Parts, _ b: Float128Parts) -> (Float128Parts, sticky: Bool) {
    if a.isZero { return (b, false) }
    if b.isZero { return (a, false) }

    // Order so |lhs| >= |rhs|.
    let lhsLarger: Bool
    if a.exp != b.exp {
        lhsLarger = a.exp > b.exp
    } else {
        lhsLarger = a.mantissa >= b.mantissa
    }
    let lhs = lhsLarger ? a : b
    let rhs = lhsLarger ? b : a

    // Make room for a carry bit.
    let lhsM = lhs.mantissa >> 1
    var rhsM = rhs.mantissa >> 1

    let diff = lhs.exp - rhs.exp
    let (aligned, alignSticky) = shiftRightTrackSticky(rhsM, by: diff)
    // Fold the alignment-lost bits into the operand's LSB *before* combining, so
    // an effective subtraction borrows from them (true result is slightly
    // smaller). Tracking sticky separately and applying it additively at round
    // time would round subtractions the wrong way at the half-ulp point. This
    // is only ever nonzero when diff exceeds the guard region (so the leading
    // bits don't cancel and clz stays ~0 — the folded bit can't shift up into
    // significance). Mirrors the C++ reference's shift_track_sticky_bits.
    rhsM = aligned | (alignSticky ? 1 : 0)

    let sameSign = (lhs.sign == rhs.sign)
    let resultMantissa: UInt128 = sameSign ? (lhsM &+ rhsM) : (lhsM &- rhsM)

    if resultMantissa == 0 {
        return (.zero, false)
    }

    var resultExp = lhs.exp
    let clz = resultMantissa.leadingZeroBitCount
    if clz == 0 {
        resultExp += 1
    } else if clz > 1 {
        resultExp -= clz - 1
    }
    return (Float128Parts(sign: lhs.sign, exp: resultExp, mantissa: resultMantissa &<< clz),
            false)
}

// MARK: - Multiply

/// 128x128 multiply. Both inputs have leading 1 at bit 127. The 256-bit
/// product has its leading 1 at bit 254 or 255; we normalize so the result
/// mantissa has its leading 1 at bit 127.
@inlinable
func multiplyParts(_ a: Float128Parts, _ b: Float128Parts) -> (Float128Parts, sticky: Bool) {
    let resultSign = a.sign != b.sign
    if a.isZero || b.isZero {
        return (Float128Parts(sign: resultSign, exp: 0, mantissa: 0), false)
    }

    let (hi, lo) = a.mantissa.multipliedFullWidth(by: b.mantissa)
    var resultMantissa = hi
    var resultExp = a.exp + b.exp
    var sticky: Bool

    if ((resultMantissa >> 127) & 1) != 0 {
        // Carry into bit 255 of the product: leading 1 already at bit 127 of `hi`.
        resultExp += 1
        sticky = lo != 0
    } else {
        // Leading 1 at bit 126 of `hi`; shift left by 1, pulling in MSB of lo.
        let loMsb = (lo >> 127) & 1
        resultMantissa = (resultMantissa &<< 1) | loMsb
        sticky = (lo & ((UInt128(1) << 127) &- 1)) != 0
    }

    return (Float128Parts(sign: resultSign, exp: resultExp, mantissa: resultMantissa),
            sticky)
}

// MARK: - Conversion to / from Double

/// Convert IEEE 754 binary64 to binary128 bits. Zero and subnormals → zero;
/// NaN/Inf → saturated near max (we don't carry these through Mandelbrot).
@inlinable
func float128FromDouble(_ d: Double) -> UInt128 {
    let bits = d.bitPattern
    let signBit = (bits >> 63) & 1
    let expBits = UInt32((bits >> 52) & 0x7FF)
    let mantissaBits = bits & ((UInt64(1) << 52) - 1)

    if expBits == 0 {
        return signBit != 0 ? kF128SignMask : 0
    }
    if expBits == 0x7FF {
        // Saturate.
        let satExp = UInt128(kF128ExpMaxBits - 1) << 112
        var out = satExp | kF128MantissaMask
        if signBit != 0 { out |= kF128SignMask }
        return out
    }
    // Normal: rebias (binary64 → binary128) is +15360.
    let newBiased = UInt32(expBits) + UInt32(kF128ExpBias - 1023)
    var out = UInt128(newBiased & kF128ExpMaxBits) << 112
    out |= UInt128(mantissaBits) << (112 - 52)
    if signBit != 0 { out |= kF128SignMask }
    return out
}

/// Convert binary128 bits to a binary64 Double. Truncates the low 60 mantissa
/// bits. Under/overflow at the binary64 boundary clamps to ±0 / ±Infinity.
@inlinable
func float128ToDouble(_ bits: UInt128) -> Double {
    let signBit = (bits >> 127) & 1 != 0
    let expBits = UInt32(truncatingIfNeeded: (bits >> 112) & 0x7FFF)
    let mantissaBits = bits & kF128MantissaMask

    if expBits == 0 {
        return signBit ? -0.0 : 0.0
    }
    let newBiased = Int(expBits) - (kF128ExpBias - 1023)
    if newBiased <= 0 { return signBit ? -0.0 : 0.0 }
    if newBiased >= 0x7FF { return signBit ? -.infinity : .infinity }

    let mantissaTop52 = UInt64(truncatingIfNeeded: mantissaBits >> 60)
    var out: UInt64 = UInt64(newBiased) << 52
    out |= mantissaTop52
    if signBit { out |= UInt64(1) << 63 }
    return Double(bitPattern: out)
}
