import Foundation

/// Renders a viewport into an `IterationField`. The engine is the boundary
/// between viewport math and the choice of numeric precision / hardware.
public protocol MandelbrotEngine: Sendable {
    func render(
        viewport: Viewport,
        width: Int,
        height: Int,
        maxIterations: UInt32
    ) -> IterationField
}

/// CPU engine that dispatches fixed-width horizontal strips across cores.
public struct CPUEngine<Kernel: StripKernel>: MandelbrotEngine {
    public let kernel: Kernel

    public init(kernel: Kernel) { self.kernel = kernel }

    public func render(
        viewport: Viewport,
        width: Int,
        height: Int,
        maxIterations: UInt32
    ) -> IterationField {
        let field = IterationField(width: width, height: height)
        let stripW = Kernel.stripWidth
        let stripsPerRow = (width + stripW - 1) / stripW
        let totalStrips = height * stripsPerRow

        let originX = viewport.originX(forWidth: width)
        let originY = viewport.originY(forHeight: height)
        let pxSize = viewport.pixelSize
        let kernel = self.kernel

        DispatchQueue.concurrentPerform(iterations: totalStrips) { idx in
            let row = idx / stripsPerRow
            let strip = idx % stripsPerRow
            let pixelX = strip * stripW

            let sox = originX + pxSize * Float128(pixelX)
            let soy = originY - pxSize * Float128(row)

            let remaining = width - pixelX
            if remaining >= stripW {
                let out = field.pointer(row: row, column: pixelX)
                kernel.computeStrip(
                    originX: sox, originY: soy, deltaX: pxSize,
                    maxIterations: maxIterations, output: out
                )
            } else {
                // Right edge: compute into a scratch strip and copy the valid columns.
                var scratch = [PixelResult](repeating: .inSet, count: stripW)
                scratch.withUnsafeMutableBufferPointer { buf in
                    kernel.computeStrip(
                        originX: sox, originY: soy, deltaX: pxSize,
                        maxIterations: maxIterations, output: buf.baseAddress!
                    )
                }
                for i in 0..<remaining {
                    field[row: row, column: pixelX + i] = scratch[i]
                }
            }
        }
        return field
    }
}
