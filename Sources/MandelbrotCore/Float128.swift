import Foundation

/// 128-bit floating-point coordinate type for Mandelbrot computation.
///
/// Storage is the packed IEEE 754 binary128 layout (sign + 15-bit biased
/// exponent + 112-bit mantissa) in a `UInt128`. Arithmetic is implemented by
/// the routines in `IEEE754_128.swift`, a Swift port of rthelen's C++ ieee754
/// library (`Reference/ieee754-cxx/`).
///
/// NO_NAN_INF semantics: no handling of NaN, Inf, or denormals. Inputs that
/// would land outside the normal range are flushed to zero (underflow) or
/// saturated near the max (overflow). The Mandelbrot viewport is bounded so
/// these paths aren't reached during normal iteration.
///
/// Division is intentionally not provided — Mandelbrot doesn't need it.
public struct Float128: Sendable {
    @usableFromInline var bits: UInt128

    @inlinable init(rawBits: UInt128) { self.bits = rawBits }

    @inlinable public init(_ d: Double) { self.bits = float128FromDouble(d) }
    @inlinable public init(_ i: Int) { self.bits = float128FromDouble(Double(i)) }

    /// Truncate to `Double`. Loses the low 60 mantissa bits.
    @inlinable public var asDouble: Double { float128ToDouble(bits) }
}

extension Float128: ExpressibleByFloatLiteral {
    @inlinable public init(floatLiteral value: Double) { self.init(value) }
}

extension Float128: ExpressibleByIntegerLiteral {
    @inlinable public init(integerLiteral value: Int) { self.init(value) }
}

extension Float128 {
    @inlinable public static var zero: Float128 { Float128(rawBits: 0) }

    @inlinable public static func + (l: Float128, r: Float128) -> Float128 {
        let lp = partsFromBits(l.bits)
        let rp = partsFromBits(r.bits)
        let (sum, sticky) = addParts(lp, rp)
        return Float128(rawBits: bitsFromParts(sum, sticky: sticky))
    }

    @inlinable public static func - (l: Float128, r: Float128) -> Float128 {
        let lp = partsFromBits(l.bits)
        var rp = partsFromBits(r.bits)
        rp.sign.toggle()
        let (diff, sticky) = addParts(lp, rp)
        return Float128(rawBits: bitsFromParts(diff, sticky: sticky))
    }

    @inlinable public static func * (l: Float128, r: Float128) -> Float128 {
        let lp = partsFromBits(l.bits)
        let rp = partsFromBits(r.bits)
        let (prod, sticky) = multiplyParts(lp, rp)
        return Float128(rawBits: bitsFromParts(prod, sticky: sticky))
    }

    @inlinable public static prefix func - (v: Float128) -> Float128 {
        // Toggle the sign bit; zero stays zero (its bits are 0 or kF128SignMask).
        Float128(rawBits: v.bits ^ kF128SignMask)
    }

    @inlinable public static func += (l: inout Float128, r: Float128) { l = l + r }
    @inlinable public static func -= (l: inout Float128, r: Float128) { l = l - r }
    @inlinable public static func *= (l: inout Float128, r: Float128) { l = l * r }
}

extension Float128: Equatable {
    @inlinable public static func == (l: Float128, r: Float128) -> Bool {
        // Treat +0 and -0 as equal.
        let lp = partsFromBits(l.bits)
        let rp = partsFromBits(r.bits)
        if lp.isZero && rp.isZero { return true }
        return l.bits == r.bits
    }
}

extension Float128: Hashable {
    @inlinable public func hash(into hasher: inout Hasher) {
        // Normalize -0 to +0 so equal values hash equal.
        let p = partsFromBits(bits)
        if p.isZero {
            hasher.combine(UInt128(0))
        } else {
            hasher.combine(bits)
        }
    }
}

extension Float128: Comparable {
    @inlinable public static func < (l: Float128, r: Float128) -> Bool {
        let lp = partsFromBits(l.bits)
        let rp = partsFromBits(r.bits)
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

extension Float128: CustomStringConvertible {
    public var description: String { String(asDouble) }
}
