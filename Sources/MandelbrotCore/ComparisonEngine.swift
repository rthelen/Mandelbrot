import Foundation
import CoreGraphics

/// Result of a dual-precision comparison render: the hardware-`Double` field
/// (used for display) plus a per-pixel flag marking where the software
/// `SoftDouble` result differed bit-for-bit, and the total mismatch count.
///
/// Because `SoftDouble` reproduces hardware `Double` exactly, a correct build
/// yields `mismatchCount == 0` across the whole surface — so this doubles as a
/// live self-check: any flagged pixel is a real divergence.
/// A captured first point of divergence between the hardware and software
/// iteration, for diagnostics. `label` names the operation that first differed.
public struct DivergenceSample: Sendable, Equatable {
    public let pixelX: Int
    public let pixelY: Int
    public let iteration: UInt32
    public let label: String       // e.g. "zx²", "|z|²", "zx'"
    public let hwBits: UInt64
    public let swBits: UInt64

    public init(pixelX: Int, pixelY: Int, iteration: UInt32, label: String,
                hwBits: UInt64, swBits: UInt64) {
        self.pixelX = pixelX; self.pixelY = pixelY; self.iteration = iteration
        self.label = label; self.hwBits = hwBits; self.swBits = swBits
    }
}

public final class ComparisonField: @unchecked Sendable {
    public let field: IterationField
    public let width: Int
    public let height: Int
    @usableFromInline let flags: UnsafeMutablePointer<Bool>
    public let mismatchCount: Int
    /// For per-step (lockstep) comparison: the first divergence found, if any.
    public let firstDivergence: DivergenceSample?

    init(field: IterationField, flags: UnsafeMutablePointer<Bool>, mismatchCount: Int,
         firstDivergence: DivergenceSample? = nil) {
        self.field = field
        self.width = field.width
        self.height = field.height
        self.flags = flags
        self.mismatchCount = mismatchCount
        self.firstDivergence = firstDivergence
    }

    deinit {
        flags.deinitialize(count: width * height)
        flags.deallocate()
    }

    /// Read-only view of the per-pixel mismatch flags (row-major, `width*height`).
    public func withFlags<R>(_ body: (UnsafeBufferPointer<Bool>) -> R) -> R {
        body(UnsafeBufferPointer(start: flags, count: width * height))
    }
}

/// Wraps a raw pointer so it can be captured in the `@Sendable` strip closure.
/// Safe because each strip writes a disjoint range of indices.
@usableFromInline
struct UnsafeSendablePtr<T>: @unchecked Sendable {
    @usableFromInline let p: UnsafeMutablePointer<T>
    @inlinable init(_ p: UnsafeMutablePointer<T>) { self.p = p }
}

/// Bitwise equality of two pixel results: iteration count and the raw bits of
/// the smoothing value must both match. (`==` on `Float32` would treat -0/+0 as
/// equal and NaN as unequal; we want exact bit identity.)
@inlinable
func pixelBitsEqual(_ a: PixelResult, _ b: PixelResult) -> Bool {
    a.iterations == b.iterations && a.smooth.bitPattern == b.smooth.bitPattern
}

/// Renders each pixel twice — once in hardware `Double`, once in software
/// `SoftDouble` — and compares the two results bit-for-bit. Dispatches the same
/// 32-wide strips as `CPUEngine`; each strip runs both kernels into scratch
/// buffers, writes the hardware result to the field, and records mismatches.
public struct DoubleComparisonEngine: Sendable {
    public init() {}

    public func renderComparison(
        viewport: Viewport,
        width: Int,
        height: Int,
        maxIterations: UInt32
    ) -> ComparisonField {
        let field = IterationField(width: width, height: height)
        let flags = UnsafeMutablePointer<Bool>.allocate(capacity: width * height)
        flags.initialize(repeating: false, count: width * height)
        let flagsBox = UnsafeSendablePtr(flags)

        let stripW = DoubleStripKernel.stripWidth   // == SoftDoubleStripKernel.stripWidth (32)
        let stripsPerRow = (width + stripW - 1) / stripW
        let totalStrips = height * stripsPerRow

        let originX = viewport.originX(forWidth: width)
        let originY = viewport.originY(forHeight: height)
        let pxSize = viewport.pixelSize

        let hwKernel = DoubleStripKernel()
        let swKernel = SoftDoubleStripKernel()

        DispatchQueue.concurrentPerform(iterations: totalStrips) { idx in
            let row = idx / stripsPerRow
            let strip = idx % stripsPerRow
            let pixelX = strip * stripW

            let sox = originX + pxSize * Float128(pixelX)
            let soy = originY - pxSize * Float128(row)

            var hw = [PixelResult](repeating: .inSet, count: stripW)
            var sw = [PixelResult](repeating: .inSet, count: stripW)
            hw.withUnsafeMutableBufferPointer { hwBuf in
                sw.withUnsafeMutableBufferPointer { swBuf in
                    hwKernel.computeStrip(originX: sox, originY: soy, deltaX: pxSize,
                                          maxIterations: maxIterations, output: hwBuf.baseAddress!)
                    swKernel.computeStrip(originX: sox, originY: soy, deltaX: pxSize,
                                          maxIterations: maxIterations, output: swBuf.baseAddress!)
                }
            }

            let remaining = min(stripW, width - pixelX)
            let base = row * width + pixelX
            for i in 0..<remaining {
                field.storage[base + i] = hw[i]          // display the hardware result
                flagsBox.p[base + i] = !pixelBitsEqual(hw[i], sw[i])
            }
        }

        var count = 0
        for i in 0..<(width * height) where flags[i] { count += 1 }
        return ComparisonField(field: field, flags: flags, mismatchCount: count)
    }
}

/// Colorize a comparison field: matching pixels use the chosen colorizer; pixels
/// where hardware and software disagreed are painted with `highlight` so any
/// divergence is impossible to miss while diving.
public func renderComparisonImage(
    _ comp: ComparisonField,
    colorizer: any Colorizer,
    maxIterations: UInt32,
    highlight: RGB = RGB(255, 0, 255)
) -> CGImage {
    let width = comp.width
    let height = comp.height
    let bytesPerRow = width * 4
    var data = Data(count: bytesPerRow * height)

    data.withUnsafeMutableBytes { rawBuf in
        let dst = rawBuf.bindMemory(to: UInt8.self).baseAddress!
        comp.field.withBufferPointer { src in
            comp.withFlags { flag in
                for i in 0..<(width * height) {
                    let rgb: RGB
                    if flag[i] {
                        rgb = highlight
                    } else {
                        let result = src[i]
                        if result.iterations == .max {
                            rgb = colorizer.inSetColor
                        } else {
                            let t = Double(result.iterations) + Double(result.smooth)
                            rgb = colorizer.colorForEscape(t: t, maxIterations: maxIterations)
                        }
                    }
                    let p = dst.advanced(by: i * 4)
                    p[0] = rgb.r; p[1] = rgb.g; p[2] = rgb.b; p[3] = 255
                }
            }
        }
    }

    let provider = CGDataProvider(data: data as CFData)!
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
    return CGImage(
        width: width, height: height,
        bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: bytesPerRow,
        space: colorSpace, bitmapInfo: bitmapInfo,
        provider: provider, decode: nil,
        shouldInterpolate: false, intent: .defaultIntent
    )!
}
