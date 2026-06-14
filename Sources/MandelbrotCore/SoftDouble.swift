import Foundation

/// Software IEEE 754 binary64 — the same format hardware `Double` implements,
/// computed by hand. Storage is the packed binary64 layout (sign + 11-bit biased
/// exponent + 52-bit mantissa) in a `UInt64`; arithmetic lives in
/// `IEEE754_64.swift`, a Swift port of rthelen's C++ ieee754 library
/// (`Reference/ieee754-cxx/`).
///
/// This type exists to *validate*: running the Mandelbrot iteration in
/// `SoftDouble` should reproduce the hardware-`Double` image bit for bit,
/// cross-checking the hand-written FP algorithm against silicon on a known
/// surface — the same confidence path the wider software floats rely on.
///
/// NO_NAN_INF semantics: no NaN, Inf, or denormals. Out-of-range values flush
/// to zero (underflow) or saturate near max (overflow). The Mandelbrot viewport
/// is bounded so these paths aren't reached during normal iteration.
///
/// Division is intentionally not provided — Mandelbrot doesn't need it.
public struct SoftDouble: Sendable {
    @usableFromInline var bits: UInt64

    @inlinable init(rawBits: UInt64) { self.bits = rawBits }

    @inlinable public init(_ d: Double) { self.bits = softDoubleFromDouble(d) }
    @inlinable public init(_ i: Int) { self.bits = softDoubleFromDouble(Double(i)) }

    /// Reinterpret as a hardware `Double` (identity for normal values).
    @inlinable public var asDouble: Double { softDoubleToDouble(bits) }
}

extension SoftDouble: ExpressibleByFloatLiteral {
    @inlinable public init(floatLiteral value: Double) { self.init(value) }
}

extension SoftDouble: ExpressibleByIntegerLiteral {
    @inlinable public init(integerLiteral value: Int) { self.init(value) }
}

extension SoftDouble {
    @inlinable public static var zero: SoftDouble { SoftDouble(rawBits: 0) }

    @inlinable public static func + (l: SoftDouble, r: SoftDouble) -> SoftDouble {
        let lp = partsFromBits64(l.bits)
        let rp = partsFromBits64(r.bits)
        let (sum, sticky) = addParts64(lp, rp)
        return SoftDouble(rawBits: bitsFromParts64(sum, sticky: sticky))
    }

    @inlinable public static func - (l: SoftDouble, r: SoftDouble) -> SoftDouble {
        let lp = partsFromBits64(l.bits)
        var rp = partsFromBits64(r.bits)
        rp.sign.toggle()
        let (diff, sticky) = addParts64(lp, rp)
        return SoftDouble(rawBits: bitsFromParts64(diff, sticky: sticky))
    }

    @inlinable public static func * (l: SoftDouble, r: SoftDouble) -> SoftDouble {
        let lp = partsFromBits64(l.bits)
        let rp = partsFromBits64(r.bits)
        let (prod, sticky) = multiplyParts64(lp, rp)
        return SoftDouble(rawBits: bitsFromParts64(prod, sticky: sticky))
    }

    @inlinable public static prefix func - (v: SoftDouble) -> SoftDouble {
        // Toggle the sign bit; zero stays zero (its bits are 0 or kF64SignMask).
        SoftDouble(rawBits: v.bits ^ kF64SignMask)
    }

    @inlinable public static func += (l: inout SoftDouble, r: SoftDouble) { l = l + r }
    @inlinable public static func -= (l: inout SoftDouble, r: SoftDouble) { l = l - r }
    @inlinable public static func *= (l: inout SoftDouble, r: SoftDouble) { l = l * r }
}

extension SoftDouble: Equatable {
    @inlinable public static func == (l: SoftDouble, r: SoftDouble) -> Bool {
        // Treat +0 and -0 as equal.
        let lp = partsFromBits64(l.bits)
        let rp = partsFromBits64(r.bits)
        if lp.isZero && rp.isZero { return true }
        return l.bits == r.bits
    }
}

extension SoftDouble: Hashable {
    @inlinable public func hash(into hasher: inout Hasher) {
        // Normalize -0 to +0 so equal values hash equal.
        let p = partsFromBits64(bits)
        if p.isZero {
            hasher.combine(UInt64(0))
        } else {
            hasher.combine(bits)
        }
    }
}

extension SoftDouble: Comparable {
    @inlinable public static func < (l: SoftDouble, r: SoftDouble) -> Bool {
        let lp = partsFromBits64(l.bits)
        let rp = partsFromBits64(r.bits)
        if lp.isZero && rp.isZero { return false }
        if lp.isZero { return !rp.sign }
        if rp.isZero { return lp.sign }
        if lp.sign != rp.sign { return lp.sign }
        // Same sign, both non-zero.
        let magLess: Bool
        if lp.exp != rp.exp {
            magLess = lp.exp < rp.exp
        } else {
            magLess = lp.mantissa < rp.mantissa
        }
        return lp.sign ? !magLess : magLess
    }
}

extension SoftDouble: CustomStringConvertible {
    public var description: String { String(asDouble) }
}
