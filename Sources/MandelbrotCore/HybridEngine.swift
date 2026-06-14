import Foundation

/// Renders Float128 by splitting each frame between the GPU (top rows) and the
/// CPU (bottom rows) running concurrently, then **adapts** the split each frame
/// to equalize their finish times — so combined throughput approaches the sum of
/// both engines. The two kernels are bit-exact to `Float128`, so the merged
/// field is identical to either engine alone.
///
/// This is the heterogeneous work-split pattern (measure the GPU:CPU rate, divide
/// the work, run concurrently, re-balance) — the same shape a Pi-engine FFT would
/// use to partition a digit range across both engines.
public final class HybridFloat128Engine: MandelbrotEngine, @unchecked Sendable {
    private let lock = NSLock()
    private var gpuFraction: Double

    /// Last frame's measured timings, for inspection (e.g. a HUD).
    public private(set) var lastGPUms: Double = 0
    public private(set) var lastCPUms: Double = 0
    public private(set) var lastSplitRow: Int = 0

    public init(initialGPUFraction: Double = 0.9) { self.gpuFraction = initialGPUFraction }

    public var currentGPUFraction: Double { lock.lock(); defer { lock.unlock() }; return gpuFraction }

    public func render(viewport: Viewport, width: Int, height: Int, maxIterations: UInt32) -> IterationField {
        let frac: Double = { lock.lock(); defer { lock.unlock() }; return gpuFraction }()
        var splitRow = Int((Double(height) * frac).rounded())
        splitRow = max(0, min(height, splitRow))

        let field = IterationField(width: width, height: height)
        var tGPU = 0.0, tCPU = 0.0
        let group = DispatchGroup()

        if splitRow > 0 {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                let t0 = CFAbsoluteTimeGetCurrent()
                gpuFloat128RenderBand(viewport: viewport, width: width, totalHeight: height,
                                      rowStart: 0, rowCount: splitRow,
                                      maxIterations: maxIterations, into: field)
                tGPU = CFAbsoluteTimeGetCurrent() - t0
                group.leave()
            }
        }
        if splitRow < height {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                let t0 = CFAbsoluteTimeGetCurrent()
                cpuFloat128RenderBand(viewport: viewport, width: width, totalHeight: height,
                                      rowStart: splitRow, rowCount: height - splitRow,
                                      maxIterations: maxIterations, into: field)
                tCPU = CFAbsoluteTimeGetCurrent() - t0
                group.leave()
            }
        }
        group.wait()

        // Re-balance: aim for the split where both bands finish at the same time,
        // i.e. gpuFraction = rateGPU / (rateGPU + rateCPU). Damped to avoid oscillation.
        if splitRow > 0 && splitRow < height && tGPU > 0 && tCPU > 0 {
            let rateGPU = Double(splitRow) / tGPU
            let rateCPU = Double(height - splitRow) / tCPU
            let target = rateGPU / (rateGPU + rateCPU)
            lock.lock()
            gpuFraction = max(0.05, min(0.97, 0.6 * gpuFraction + 0.4 * target))
            lock.unlock()
        }
        lock.lock(); lastGPUms = tGPU * 1000; lastCPUms = tCPU * 1000; lastSplitRow = splitRow; lock.unlock()
        return field
    }
}

/// Renders global rows [rowStart, rowStart+rowCount) of `viewport` on the CPU in
/// Float128, writing into `field` at those rows (global coordinates throughout).
func cpuFloat128RenderBand(viewport: Viewport, width: Int, totalHeight: Int,
                           rowStart: Int, rowCount: Int, maxIterations: UInt32,
                           into field: IterationField) {
    let kernel = CFloat128StripKernel()   // C kernel: 1.46x faster, bit-exact
    let stripW = CFloat128StripKernel.stripWidth
    let stripsPerRow = (width + stripW - 1) / stripW
    let originX = viewport.originX(forWidth: width)
    let originY = viewport.originY(forHeight: totalHeight)
    let pxSize = viewport.pixelSize
    let totalStrips = rowCount * stripsPerRow

    DispatchQueue.concurrentPerform(iterations: totalStrips) { idx in
        let row = rowStart + idx / stripsPerRow
        let pixelX = (idx % stripsPerRow) * stripW
        let sox = originX + pxSize * Float128(pixelX)
        let soy = originY - pxSize * Float128(row)
        let remaining = width - pixelX
        if remaining >= stripW {
            kernel.computeStrip(originX: sox, originY: soy, deltaX: pxSize,
                                maxIterations: maxIterations, output: field.pointer(row: row, column: pixelX))
        } else {
            var scratch = [PixelResult](repeating: .inSet, count: stripW)
            scratch.withUnsafeMutableBufferPointer { buf in
                kernel.computeStrip(originX: sox, originY: soy, deltaX: pxSize,
                                    maxIterations: maxIterations, output: buf.baseAddress!)
            }
            for i in 0..<remaining { field[row: row, column: pixelX + i] = scratch[i] }
        }
    }
}
