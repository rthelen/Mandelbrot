import Foundation

/// A reproducible dive into a fixed Mandelbrot location. Each "frame" zooms
/// `zoomRate`× toward `(centerX, centerY)`. Stops when `pixelSize <= minPixelSize`.
///
/// Coordinates are stored as `Double` (these are well-known points known to ~15
/// digits, plenty for naming a target). The dive *math* runs in whatever
/// precision the chosen kernel provides.
public struct PlaybackTarget: Sendable, Hashable, Codable {
    public let name: String
    public let centerX: Double
    public let centerY: Double
    public let zoomRate: Double       // < 1 zooms in
    public let minPixelSize: Double   // stop condition

    public init(name: String, centerX: Double, centerY: Double,
                zoomRate: Double, minPixelSize: Double) {
        self.name = name
        self.centerX = centerX
        self.centerY = centerY
        self.zoomRate = zoomRate
        self.minPixelSize = minPixelSize
    }
}

extension PlaybackTarget {
    public static let seahorseValley = PlaybackTarget(
        name: "Seahorse Valley",
        centerX: -0.7453,
        centerY:  0.1127,
        zoomRate: 0.97,
        minPixelSize: 1e-32
    )
    public static let miniMandelbrot = PlaybackTarget(
        name: "Mini Mandelbrot",
        centerX: -1.749999,
        centerY:  0.0,
        zoomRate: 0.97,
        minPixelSize: 1e-32
    )
    public static let misiurewicz = PlaybackTarget(
        name: "Misiurewicz",
        centerX: -0.77568377,
        centerY:  0.13646737,
        zoomRate: 0.97,
        minPixelSize: 1e-32
    )

    public static let allPresets: [PlaybackTarget] = [
        .seahorseValley, .miniMandelbrot, .misiurewicz
    ]
}
