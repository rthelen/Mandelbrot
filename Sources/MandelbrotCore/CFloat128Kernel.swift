import Foundation
import CMandelbrot

/// Float128 strip kernel that runs the whole per-pixel iteration in C
/// (`cf128_mandelbrot_pixel`, __uint128_t). One call per pixel amortizes the
/// C-call overhead over ~1000 iterations; the multiply/add/round are a single
/// tight block clang lowers to MUL+UMULH+ADCS. Bit-exact to `Float128StripKernel`.
public struct CFloat128StripKernel: StripKernel {
    public static let stripWidth: Int = 32

    public init() {}

    @inlinable
    public func computeStrip(
        originX: Float128,
        originY: Float128,
        deltaX: Float128,
        maxIterations: UInt32,
        output: UnsafeMutablePointer<PixelResult>
    ) {
        let cyBits = originY.bits
        let cyLo = UInt64(truncatingIfNeeded: cyBits)
        let cyHi = UInt64(truncatingIfNeeded: cyBits >> 64)

        for i in 0..<Self.stripWidth {
            let cx = originX + deltaX * Float128(i)
            let cxBits = cx.bits
            var msLo: UInt64 = 0, msHi: UInt64 = 0
            let n = cf128_mandelbrot_pixel(
                UInt64(truncatingIfNeeded: cxBits),
                UInt64(truncatingIfNeeded: cxBits >> 64),
                cyLo, cyHi, maxIterations, &msLo, &msHi)
            if n != 0xFFFFFFFF {
                let magSq = Float128(rawBits: (UInt128(msHi) << 64) | UInt128(msLo))
                output[i] = PixelResult(iterations: n, smooth: smoothFraction(magSq: magSq.asDouble))
            } else {
                output[i] = .inSet
            }
        }
    }
}
