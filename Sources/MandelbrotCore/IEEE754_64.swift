import Foundation

// Swift port of rthelen's C++ ieee754_64 (see Reference/ieee754-cxx/).
// Scalar *software* binary64 add and multiply — the same IEEE 754 double
// format the hardware implements, computed by hand. Its reason to exist is
// validation: a software binary64 should reproduce hardware `Double` results
// BIT FOR BIT across the Mandelbrot surface (round-to-nearest-even), giving the
// same algorithm-vs-hardware confidence the wider software floats are built on.
//
// NO_NAN_INF semantics: no NaN, no Inf, no denormals. Inputs/results outside the
// normal range flush to zero (underflow) or saturate near the max (overflow).
// The caller controls the viewport so neither path is reached during normal
// Mandelbrot iteration. Because of denormal flush, the bit-exact match with
// hardware holds wherever both operands and the result stay normal (or zero) —
// which the bounded viewport guarantees.
//
// Mirrors IEEE754_128.swift; the only differences are the field widths.

// MARK: - IEEE 754 binary64 constants

@usableFromInline let kF64MantissaWidth: Int = 52
@usableFromInline let kF64ExpWidth: Int = 11
@usableFromInline let kF64ExpMaxBits: UInt32 = (1 << 11) - 1      // 2047
@usableFromInline let kF64ExpBias: Int = 1023
@usableFromInline let kF64MantissaMask: UInt64 = (UInt64(1) << 52) &- 1
@usableFromInline let kF64SignMask: UInt64 = UInt64(1) << 63

// MARK: - Internal computational representation

/// Unpacked representation used during arithmetic. Mantissa is left-justified
/// with the leading 1 at bit 63 (or all zeros for an exact zero).
@usableFromInline
struct SoftDoubleParts {
    @usableFromInline var sign: Bool        // true = negative
    @usableFromInline var exp: Int          // unbiased
    @usableFromInline var mantissa: UInt64

    @inlinable init(sign: Bool, exp: Int, mantissa: UInt64) {
        self.sign = sign; self.exp = exp; self.mantissa = mantissa
    }
    @inlinable static var zero: SoftDoubleParts {
        SoftDoubleParts(sign: false, exp: 0, mantissa: 0)
    }
    @inlinable var isZero: Bool { mantissa == 0 }
}

// MARK: - Unpack

/// Decompose packed binary64 bits into computational parts. Subnormal inputs
/// are flushed to zero (NO_NAN_INF mode).
@inline(__always) @inlinable
func partsFromBits64(_ bits: UInt64) -> SoftDoubleParts {
    let signBit = (bits >> 63) & 1 != 0
    let expBits = UInt32(truncatingIfNeeded: (bits >> 52) & 0x7FF)
    let mantissaBits = bits & kF64MantissaMask

    if expBits == 0 {
        return SoftDoubleParts(sign: signBit, exp: 0, mantissa: 0)
    }
    // Normal: insert implied 1 (now 53 bits), then left-justify by exp_width.
    let full = (UInt64(1) << 52) | mantissaBits
    return SoftDoubleParts(sign: signBit,
                           exp: Int(expBits) - kF64ExpBias,
                           mantissa: full << 11)
}

// MARK: - Round and pack

/// Round-to-nearest-even and pack a `SoftDoubleParts` (mantissa left-justified
/// with bit 63 set) into binary64 bits. Handles the carry produced by a
/// round-up cascading through the implied bit.
@inline(__always) @inlinable
func bitsFromParts64(_ p: SoftDoubleParts, sticky stickyIn: Bool = false) -> UInt64 {
    if p.isZero {
        return p.sign ? kF64SignMask : 0
    }

    // Shift down by 1 to make bit 63 a carry slot.
    var sticky = stickyIn || (p.mantissa & 1) != 0
    var mantissa = p.mantissa >> 1
    var exp = p.exp
    // Layout: bit 63 = carry slot, bit 62 = implied 1,
    //         bits 61..10 = stored mantissa (52),
    //         bit 9 = round, bits 8..0 = sticky.

    let stickyMask: UInt64 = (UInt64(1) << 9) &- 1
    sticky = sticky || (mantissa & stickyMask) != 0
    let roundBit = ((mantissa >> 9) & 1) != 0
    mantissa >>= 10
    // Now: bit 53 = carry slot, bit 52 = implied 1, bits 51..0 = stored mantissa.

    let ulpBit = (mantissa & 1) != 0
    // Round-to-nearest-even, branchless: compute the rounded mantissa and select.
    let roundUp = roundBit && (ulpBit || sticky)
    let bumped = roundUp ? (mantissa &+ 1) : mantissa
    // Cascade only when the +1 overflowed the implied bit (52 -> 53); when no
    // round-up happened bit 53 is 0, so this is a no-op then.
    let cascade = ((bumped >> 53) & 1) != 0
    mantissa = cascade ? (bumped >> 1) : bumped
    exp = cascade ? (exp + 1) : exp

    let biased = exp + kF64ExpBias
    if biased <= 0 {
        // Underflow flushes to zero (no denormals).
        return p.sign ? kF64SignMask : 0
    }
    if biased >= Int(kF64ExpMaxBits) {
        // Overflow saturates to the largest finite (avoid the NaN/Inf code).
        let satExp = UInt64(kF64ExpMaxBits - 1) << 52
        let satMant = kF64MantissaMask
        var bits = satExp | satMant
        if p.sign { bits |= kF64SignMask }
        return bits
    }

    let biasedU = UInt64(UInt32(biased) & kF64ExpMaxBits)
    let storedMantissa = mantissa & kF64MantissaMask
    return (biasedU << 52) | storedMantissa | (p.sign ? kF64SignMask : 0)
}

// MARK: - Helpers

/// Right-shift, OR-ing any bits shifted out into a sticky flag.
@inline(__always) @inlinable
func shiftRightTrackSticky64(_ n: UInt64, by shift: Int) -> (value: UInt64, sticky: Bool) {
    if shift == 0 { return (n, false) }
    if shift >= 64 { return (0, n != 0) }
    let hi = n >> shift
    let lostMask: UInt64 = (UInt64(1) << shift) &- 1
    let lost = n & lostMask
    return (hi, lost != 0)
}

// MARK: - Add

/// Algebraic add of two parts (their signs determine add-vs-subtract).
/// Returns unpacked result with mantissa left-justified (leading 1 at bit 63),
/// plus a sticky flag carrying bits lost during alignment.
@inline(__always) @inlinable
func addParts64(_ a: SoftDoubleParts, _ b: SoftDoubleParts) -> (SoftDoubleParts, sticky: Bool) {
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
    let (aligned, alignSticky) = shiftRightTrackSticky64(rhsM, by: diff)
    // Fold the alignment-lost bits into the operand's LSB *before* combining, so
    // an effective subtraction borrows from them (true result is slightly
    // smaller). Tracking sticky separately and applying it additively at round
    // time would round subtractions the wrong way at the half-ulp point. This
    // is only ever nonzero when diff exceeds the guard region (so the leading
    // bits don't cancel and clz stays ~0 — the folded bit can't shift up into
    // significance). Mirrors the C++ reference's shift_track_sticky_bits.
    rhsM = aligned | (alignSticky ? 1 : 0)

    let sameSign = (lhs.sign == rhs.sign)
    let resultMantissa: UInt64 = sameSign ? (lhsM &+ rhsM) : (lhsM &- rhsM)

    if resultMantissa == 0 {
        return (.zero, false)
    }

    let clz = resultMantissa.leadingZeroBitCount
    // Branchless exponent adjust: +1 if clz==0, else -(clz-1) (which is 0 at clz==1).
    let resultExp = lhs.exp + (clz == 0 ? 1 : -(clz - 1))
    return (SoftDoubleParts(sign: lhs.sign, exp: resultExp, mantissa: resultMantissa &<< clz),
            false)
}

// MARK: - Multiply

/// 64x64 multiply. Both inputs have leading 1 at bit 63. The 128-bit product
/// has its leading 1 at bit 126 or 127; we normalize so the result mantissa
/// has its leading 1 at bit 63.
@inline(__always) @inlinable
func multiplyParts64(_ a: SoftDoubleParts, _ b: SoftDoubleParts) -> (SoftDoubleParts, sticky: Bool) {
    let resultSign = a.sign != b.sign
    if a.isZero || b.isZero {
        return (SoftDoubleParts(sign: resultSign, exp: 0, mantissa: 0), false)
    }

    let (hi, lo) = a.mantissa.multipliedFullWidth(by: b.mantissa)
    // Product leading 1 is at bit 127 (topSet) or 126. Compute both normalizations
    // and select — branchless, no data-dependent branch on the ~50/50 carry.
    let topSet = ((hi >> 63) & 1) != 0
    let loMsb = (lo >> 63) & 1
    let resultMantissa = topSet ? hi : ((hi &<< 1) | loMsb)
    let resultExp = a.exp + b.exp + (topSet ? 1 : 0)
    let sticky = topSet ? (lo != 0) : ((lo & ((UInt64(1) << 63) &- 1)) != 0)

    return (SoftDoubleParts(sign: resultSign, exp: resultExp, mantissa: resultMantissa),
            sticky)
}

// MARK: - Conversion to / from hardware Double

/// Reinterpret a hardware `Double` as software binary64 bits. The packed layout
/// is identical, so this is the bit pattern verbatim for normal values; zero and
/// subnormals flush to zero, NaN/Inf saturate near max (NO_NAN_INF mode).
@inline(__always) @inlinable
func softDoubleFromDouble(_ d: Double) -> UInt64 {
    let bits = d.bitPattern
    let expBits = UInt32(truncatingIfNeeded: (bits >> 52) & 0x7FF)
    let signBit = bits & kF64SignMask

    if expBits == 0 {
        // Zero or subnormal → signed zero.
        return signBit
    }
    if expBits == 0x7FF {
        // NaN/Inf → saturate to largest finite.
        return signBit | (UInt64(kF64ExpMaxBits - 1) << 52) | kF64MantissaMask
    }
    return bits
}

/// Reinterpret software binary64 bits as a hardware `Double`. Identity for
/// normal values; subnormals already flushed to zero upstream.
@inline(__always) @inlinable
func softDoubleToDouble(_ bits: UInt64) -> Double {
    Double(bitPattern: bits)
}
