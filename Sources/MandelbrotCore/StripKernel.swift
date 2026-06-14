import Foundation

/// Computes a fixed-width horizontal strip of Mandelbrot iterations.
///
/// Strip width is 32 to match natural SIMD/warp lane counts (Apple GPU simdgroup,
/// AVX-512 lanes). Each call is data-parallel-safe: outputs at `[0..<stripWidth]`
/// are written, no shared state with other strip calls.
///
/// The API is in `Float128` so the viewport stays at full precision across
/// kernel choices. CPU kernels are free to downconvert at entry.
public protocol StripKernel: Sendable {
    static var stripWidth: Int { get }

    /// Compute `stripWidth` Mandelbrot points starting at `(originX, originY)`,
    /// stepping by `deltaX` per pixel along the row, writing results to `output`.
    func computeStrip(
        originX: Float128,
        originY: Float128,
        deltaX: Float128,
        maxIterations: UInt32,
        output: UnsafeMutablePointer<PixelResult>
    )
}

/// Smoothing constant `log(2)`, hoisted for the smooth-iteration formula.
@usableFromInline let kLogTwo: Double = 0.6931471805599453

/// `1 - log( log|z| / log 2 ) / log 2`, the fractional smoothing offset.
/// `magSq` is `|z|²` at escape. The integer iteration count carries the
/// precision-sensitive information; the smoothing factor is fine to compute
/// in `Double` even when the iteration ran in higher precision.
@inlinable
func smoothFraction(magSq: Double) -> Float32 {
    let logZn = 0.5 * log(magSq)
    let nu = log(logZn / kLogTwo) / kLogTwo
    return Float32(1.0 - nu)
}

// MARK: - CPU kernel: Double precision

public struct DoubleStripKernel: StripKernel {
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
        let ox = originX.asDouble
        let oy = originY.asDouble
        let dx = deltaX.asDouble
        let bailoutSq: Double = 4.0

        for i in 0..<Self.stripWidth {
            let cx = ox + dx * Double(i)
            let cy = oy

            var zx: Double = 0
            var zy: Double = 0
            var n: UInt32 = 0
            var magSq: Double = 0

            while n < maxIterations {
                let zx2 = zx * zx
                let zy2 = zy * zy
                magSq = zx2 + zy2
                if magSq > bailoutSq { break }
                let newZx = zx2 - zy2 + cx
                let newZy = 2.0 * zx * zy + cy
                zx = newZx
                zy = newZy
                n &+= 1
            }

            if n < maxIterations {
                output[i] = PixelResult(iterations: n, smooth: smoothFraction(magSq: magSq))
            } else {
                output[i] = .inSet
            }
        }
    }
}

// MARK: - CPU kernel: software binary64 (SoftDouble)

/// Iterates in `SoftDouble` — the hand-written software binary64. Numerically
/// it should be identical to `DoubleStripKernel` (same format, same rounding);
/// it exists to cross-check the software FP algorithm against hardware `Double`
/// across the Mandelbrot surface. Much slower than the hardware kernel since
/// every op is a software add/multiply.
public struct SoftDoubleStripKernel: StripKernel {
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
        let ox = SoftDouble(originX.asDouble)
        let oy = SoftDouble(originY.asDouble)
        let dx = SoftDouble(deltaX.asDouble)
        let bailoutSq: SoftDouble = 4.0
        let two: SoftDouble = 2.0

        for i in 0..<Self.stripWidth {
            let cx = ox + dx * SoftDouble(i)
            let cy = oy

            var zx: SoftDouble = .zero
            var zy: SoftDouble = .zero
            var n: UInt32 = 0
            var magSq: SoftDouble = .zero

            while n < maxIterations {
                let zx2 = zx * zx
                let zy2 = zy * zy
                magSq = zx2 + zy2
                if magSq > bailoutSq { break }
                let newZx = zx2 - zy2 + cx
                let newZy = two * zx * zy + cy
                zx = newZx
                zy = newZy
                n &+= 1
            }

            if n < maxIterations {
                output[i] = PixelResult(iterations: n, smooth: smoothFraction(magSq: magSq.asDouble))
            } else {
                output[i] = .inSet
            }
        }
    }
}

// MARK: - CPU kernel: Float128 precision

/// Iterates in `Float128` throughout. Substantially slower than the `Double`
/// kernel (every op is a software binary128 add/multiply) but gains the
/// precision needed for deep zooms where Double's ~10^-15 resolution falls
/// apart.
public struct Float128StripKernel: StripKernel {
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
        let bailoutSq: Float128 = 4.0
        let two: Float128 = 2.0
        let cy = originY

        for i in 0..<Self.stripWidth {
            let cx = originX + deltaX * Float128(i)

            var zx: Float128 = .zero
            var zy: Float128 = .zero
            var n: UInt32 = 0
            var magSq: Float128 = .zero

            while n < maxIterations {
                let zx2 = zx * zx
                let zy2 = zy * zy
                magSq = zx2 + zy2
                if magSq > bailoutSq { break }
                let newZx = zx2 - zy2 + cx
                let newZy = two * zx * zy + cy
                zx = newZx
                zy = newZy
                n &+= 1
            }

            if n < maxIterations {
                output[i] = PixelResult(iterations: n, smooth: smoothFraction(magSq: magSq.asDouble))
            } else {
                output[i] = .inSet
            }
        }
    }
}
