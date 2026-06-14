import Foundation

/// Cycling rainbow. Classic look; the smoothing makes hue continuous.
public struct ClassicHSVColorizer: Colorizer {
    public let name = "Classic HSV"
    public var cycleLength: Double
    public init(cycleLength: Double = 64.0) { self.cycleLength = cycleLength }
    public func colorForEscape(t: Double, maxIterations: UInt32) -> RGB {
        hsvToRGB(h: t / cycleLength, s: 0.85, v: 1.0)
    }
}

/// Black → red → orange → yellow → white. Warm.
public struct FireColorizer: Colorizer {
    public let name = "Fire"
    public var cycleLength: Double
    public init(cycleLength: Double = 80.0) { self.cycleLength = cycleLength }
    public func colorForEscape(t: Double, maxIterations: UInt32) -> RGB {
        let p = (t / cycleLength).truncatingRemainder(dividingBy: 1.0)
        return gradient([
            (0.00, RGB(0, 0, 0)),
            (0.25, RGB(160, 0, 0)),
            (0.50, RGB(255, 100, 0)),
            (0.75, RGB(255, 220, 60)),
            (1.00, RGB(255, 255, 230)),
        ], at: p)
    }
}

/// Black → deep blue → cyan → white. Cool, high contrast.
public struct ElectricBlueColorizer: Colorizer {
    public let name = "Electric Blue"
    public var cycleLength: Double
    public init(cycleLength: Double = 64.0) { self.cycleLength = cycleLength }
    public func colorForEscape(t: Double, maxIterations: UInt32) -> RGB {
        let p = (t / cycleLength).truncatingRemainder(dividingBy: 1.0)
        return gradient([
            (0.00, RGB(0, 0, 0)),
            (0.20, RGB(0, 16, 96)),
            (0.50, RGB(0, 120, 220)),
            (0.80, RGB(120, 220, 255)),
            (1.00, RGB(255, 255, 255)),
        ], at: p)
    }
}

/// Discrete grayscale bands. Ignores smoothing — the bands are the point.
public struct MonochromeBandsColorizer: Colorizer {
    public let name = "Monochrome Bands"
    public var bandWidth: Int
    public init(bandWidth: Int = 4) { self.bandWidth = bandWidth }
    public func colorForEscape(t: Double, maxIterations: UInt32) -> RGB {
        let n = Int(t)
        let bandIndex = (n / bandWidth) % 16
        let gray = UInt8(bandIndex * 16 + 8)
        return RGB(gray, gray, gray)
    }
}

/// Convenience list for UI pickers.
public enum AvailableColorizers {
    public static func all() -> [Colorizer] {
        [
            ClassicHSVColorizer(),
            FireColorizer(),
            ElectricBlueColorizer(),
            MonochromeBandsColorizer(),
        ]
    }
}
