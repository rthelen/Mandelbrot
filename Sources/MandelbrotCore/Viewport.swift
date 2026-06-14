import Foundation

/// The Mandelbrot-space region being rendered, stored at full Float128 precision
/// so switching between numeric kernels never loses where you are.
///
/// Convention: math-Y is up, screen-Y is down. Row 0 is the top of the image.
public struct Viewport: Sendable, Equatable {
    public var centerX: Float128
    public var centerY: Float128

    /// World units per screen pixel. Square pixels.
    public var pixelSize: Float128

    public init(centerX: Float128, centerY: Float128, pixelSize: Float128) {
        self.centerX = centerX
        self.centerY = centerY
        self.pixelSize = pixelSize
    }

    /// Default view showing the full Mandelbrot set centered on the canvas.
    public static func defaultView(width: Int, height: Int) -> Viewport {
        let targetWidth: Double = 3.5
        let pxSize = targetWidth / Double(max(width, 1))
        return Viewport(
            centerX: Float128(-0.75),
            centerY: Float128(0.0),
            pixelSize: Float128(pxSize)
        )
    }

    /// World X of pixel column 0 (left edge).
    public func originX(forWidth width: Int) -> Float128 {
        centerX - pixelSize * Float128(width / 2)
    }

    /// World Y of pixel row 0 (top edge).
    public func originY(forHeight height: Int) -> Float128 {
        centerY + pixelSize * Float128(height / 2)
    }

    /// World coordinate at a given screen pixel.
    public func coordinate(atPixelX x: Int, pixelY y: Int, width w: Int, height h: Int) -> (x: Float128, y: Float128) {
        let wx = originX(forWidth: w) + pixelSize * Float128(x)
        let wy = originY(forHeight: h) - pixelSize * Float128(y)
        return (wx, wy)
    }

    /// Shift the view by a screen-pixel offset.
    public func panned(byPixelsX dx: Int, pixelsY dy: Int) -> Viewport {
        Viewport(
            centerX: centerX - pixelSize * Float128(dx),
            centerY: centerY + pixelSize * Float128(dy),
            pixelSize: pixelSize
        )
    }

    /// Zoom around a world-space anchor. `factor < 1` zooms in.
    public func zoomed(by factor: Float128, aroundX ax: Float128, aroundY ay: Float128) -> Viewport {
        let newPx = pixelSize * factor
        let newCX = ax + (centerX - ax) * factor
        let newCY = ay + (centerY - ay) * factor
        return Viewport(centerX: newCX, centerY: newCY, pixelSize: newPx)
    }

    /// Fit a screen-space rectangle so its world bounds become the new viewport.
    /// `rect` is in pixel coordinates of the current viewport's image (width × height).
    public func zoomedToRect(originX rx: Int, originY ry: Int, width rw: Int, height rh: Int,
                             imageWidth iw: Int, imageHeight ih: Int) -> Viewport {
        let centerPxX = rx + rw / 2
        let centerPxY = ry + rh / 2
        let (newCX, newCY) = coordinate(atPixelX: centerPxX, pixelY: centerPxY, width: iw, height: ih)
        // Scale to fit the longest side of the selection to the corresponding image dimension.
        let scaleX = Double(rw) / Double(iw)
        let scaleY = Double(rh) / Double(ih)
        let factor = max(scaleX, scaleY)
        return Viewport(
            centerX: newCX,
            centerY: newCY,
            pixelSize: pixelSize * Float128(factor)
        )
    }
}
