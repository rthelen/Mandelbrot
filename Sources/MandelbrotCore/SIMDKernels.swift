import Foundation

/// Double-precision strip kernel structured for instruction-level parallelism:
/// the iteration is the OUTER loop and 8 pixels are processed together as
/// `SIMD8<Double>` lanes (the inner, data-parallel dimension). Unlike the scalar
/// kernel — which runs each pixel's dependent iteration chain to completion
/// before the next — this keeps 8 independent chains in flight per group, so the
/// compiler emits NEON across the FP pipes and the multiplier latency of one lane
/// is hidden by the others.
///
/// `SIMD8<Double>` (4 NEON registers per vector) is chosen so the working set
/// fits the 32 NEON registers; the 32-wide strip is 4 such groups. Arithmetic
/// matches `DoubleStripKernel` op-for-op, so iteration counts are identical.
public struct SIMDDoubleStripKernel: StripKernel {
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

        let laneIndex = SIMD8<Double>(0, 1, 2, 3, 4, 5, 6, 7)
        let cy = SIMD8<Double>(repeating: oy)
        let bail = SIMD8<Double>(repeating: 4.0)
        let one = SIMD8<Int64>(repeating: 1)   // Int64 so its mask matches SIMD8<Double>'s

        for g in 0..<4 {
            let cx = (laneIndex + Double(g * 8)) * dx + ox

            var zx = SIMD8<Double>(repeating: 0)
            var zy = SIMD8<Double>(repeating: 0)
            var count = SIMD8<Int64>(repeating: 0)
            var escapeMag = SIMD8<Double>(repeating: 0)
            var active = (bail .== bail)        // all true
            var escaped = (bail .!= bail)       // all false

            var iter: UInt32 = 0
            while iter < maxIterations {
                let zx2 = zx * zx
                let zy2 = zy * zy
                let mag = zx2 + zy2

                let nowEscaped = active .& (mag .> bail)
                escapeMag.replace(with: mag, where: nowEscaped)
                escaped = escaped .| nowEscaped
                active = active .& (mag .<= bail)
                if !any(active) { break }

                count.replace(with: count &+ one, where: active)

                let newZx = zx2 - zy2 + cx
                let newZy = 2.0 * zx * zy + cy
                zx = newZx
                zy = newZy
                iter &+= 1
            }

            for l in 0..<8 {
                let idx = g * 8 + l
                if escaped[l] {
                    output[idx] = PixelResult(iterations: UInt32(count[l]),
                                              smooth: smoothFraction(magSq: escapeMag[l]))
                } else {
                    output[idx] = .inSet
                }
            }
        }
    }
}
