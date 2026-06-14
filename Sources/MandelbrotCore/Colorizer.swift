import Foundation
import CoreGraphics

public struct RGB: Sendable, Equatable {
    public var r: UInt8
    public var g: UInt8
    public var b: UInt8
    @inlinable public init(_ r: UInt8, _ g: UInt8, _ b: UInt8) {
        self.r = r; self.g = g; self.b = b
    }
    public static let black = RGB(0, 0, 0)
    public static let white = RGB(255, 255, 255)
}

/// Maps an `IterationField` to a `CGImage`. The protocol exposes the per-pixel
/// palette decision so each colorizer is just a small `colorForEscape` body.
public protocol Colorizer: Sendable {
    var name: String { get }
    var inSetColor: RGB { get }
    /// Color for a point that escaped. `t = Double(iterations) + Double(smooth)`.
    func colorForEscape(t: Double, maxIterations: UInt32) -> RGB
}

extension Colorizer {
    public var inSetColor: RGB { .black }

    public func render(field: IterationField, maxIterations: UInt32) -> CGImage {
        let width = field.width
        let height = field.height
        let bytesPerRow = width * 4
        let totalBytes = bytesPerRow * height

        var data = Data(count: totalBytes)
        data.withUnsafeMutableBytes { rawBuf in
            let dst = rawBuf.bindMemory(to: UInt8.self).baseAddress!
            field.withBufferPointer { src in
                for i in 0..<(width * height) {
                    let result = src[i]
                    let rgb: RGB
                    if result.iterations == .max {
                        rgb = inSetColor
                    } else {
                        let t = Double(result.iterations) + Double(result.smooth)
                        rgb = colorForEscape(t: t, maxIterations: maxIterations)
                    }
                    let p = dst.advanced(by: i * 4)
                    p[0] = rgb.r; p[1] = rgb.g; p[2] = rgb.b; p[3] = 255
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
}

// MARK: - Palette helpers

@inlinable
func clamp01(_ x: Double) -> Double { max(0, min(1, x)) }

@inlinable
func byte(_ x: Double) -> UInt8 { UInt8(clamp01(x) * 255 + 0.5) }

@inlinable
func hsvToRGB(h: Double, s: Double, v: Double) -> RGB {
    let hh = h - floor(h)
    let i = Int(hh * 6.0) % 6
    let f = hh * 6.0 - floor(hh * 6.0)
    let p = v * (1 - s)
    let q = v * (1 - f * s)
    let t = v * (1 - (1 - f) * s)
    let r: Double; let g: Double; let bl: Double
    switch i {
    case 0: r = v; g = t; bl = p
    case 1: r = q; g = v; bl = p
    case 2: r = p; g = v; bl = t
    case 3: r = p; g = q; bl = v
    case 4: r = t; g = p; bl = v
    default: r = v; g = p; bl = q
    }
    return RGB(byte(r), byte(g), byte(bl))
}

@inlinable
func gradient(_ stops: [(Double, RGB)], at p: Double) -> RGB {
    let pp = clamp01(p)
    for i in 1..<stops.count {
        let (ta, ca) = stops[i - 1]
        let (tb, cb) = stops[i]
        if pp <= tb {
            let f = (pp - ta) / max(tb - ta, 1e-12)
            return RGB(
                byte(Double(ca.r) / 255 + (Double(cb.r) - Double(ca.r)) / 255 * f),
                byte(Double(ca.g) / 255 + (Double(cb.g) - Double(ca.g)) / 255 * f),
                byte(Double(ca.b) / 255 + (Double(cb.b) - Double(ca.b)) / 255 * f)
            )
        }
    }
    return stops.last!.1
}
