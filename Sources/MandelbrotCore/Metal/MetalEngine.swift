import Foundation
import Metal

/// GPU Mandelbrot engine using the software-binary64 (`SoftDouble`) kernel in
/// MSL. Per-pixel `c` is reconstructed on the GPU from per-strip x-origins and
/// per-row y-origins precomputed here in full precision, so the result matches
/// the CPU `SoftDoubleStripKernel` bit-for-bit.
///
/// Falls back to the CPU software-64 kernel when no Metal device is present, so
/// the engine always produces a correct field.
public struct MetalSoftDouble64Engine: MandelbrotEngine {
    public init() {}

    private static let stripW = 32

    public func render(
        viewport: Viewport,
        width: Int,
        height: Int,
        maxIterations: UInt32
    ) -> IterationField {
        guard let ctx = MetalContext.shared,
              let result = try? renderOnGPU(ctx: ctx, viewport: viewport,
                                            width: width, height: height,
                                            maxIterations: maxIterations)
        else {
            // No GPU (or a transient failure): fall back to the CPU soft-64 kernel.
            return CPUEngine(kernel: SoftDoubleStripKernel())
                .render(viewport: viewport, width: width, height: height,
                        maxIterations: maxIterations)
        }
        return result
    }

    private func renderOnGPU(
        ctx: MetalContext, viewport: Viewport,
        width: Int, height: Int, maxIterations: UInt32
    ) throws -> IterationField {
        let stripsPerRow = (width + Self.stripW - 1) / Self.stripW
        let originX = viewport.originX(forWidth: width)
        let originY = viewport.originY(forHeight: height)
        let pxSize = viewport.pixelSize

        // Per-strip x-origin and per-row y-origin as binary64 bit patterns —
        // computed in Float128 exactly as CPUEngine slices strips.
        var stripOriginX = [UInt64](repeating: 0, count: stripsPerRow)
        for s in 0..<stripsPerRow {
            let sox = originX + pxSize * Float128(s * Self.stripW)
            stripOriginX[s] = sox.asDouble.bitPattern
        }
        var rowOriginY = [UInt64](repeating: 0, count: height)
        for r in 0..<height {
            let soy = originY - pxSize * Float128(r)
            rowOriginY[r] = soy.asDouble.bitPattern
        }
        var dxBits = pxSize.asDouble.bitPattern
        var dims = SIMD2<UInt32>(UInt32(width), UInt32(height))
        var maxIter = maxIterations

        let dev = ctx.device
        let pipe = try ctx.pipeline("mandelbrot_sd64")
        let pixelCount = width * height

        guard
            let stripBuf = dev.makeBuffer(bytes: &stripOriginX,
                                          length: stripsPerRow * 8, options: .storageModeShared),
            let rowBuf = dev.makeBuffer(bytes: &rowOriginY,
                                        length: height * 8, options: .storageModeShared),
            let iterBuf = dev.makeBuffer(length: pixelCount * MemoryLayout<UInt32>.stride,
                                         options: .storageModeShared),
            let smoothBuf = dev.makeBuffer(length: pixelCount * MemoryLayout<Float32>.stride,
                                           options: .storageModeShared),
            let cmd = ctx.queue.makeCommandBuffer(),
            let enc = cmd.makeComputeCommandEncoder()
        else { throw MetalSetupError.noDevice }

        enc.setComputePipelineState(pipe)
        enc.setBuffer(stripBuf, offset: 0, index: 0)
        enc.setBuffer(rowBuf, offset: 0, index: 1)
        enc.setBytes(&dxBits, length: 8, index: 2)
        enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        enc.setBytes(&maxIter, length: 4, index: 4)
        enc.setBuffer(iterBuf, offset: 0, index: 5)
        enc.setBuffer(smoothBuf, offset: 0, index: 6)

        let tgw = 16, tgh = max(1, min(16, pipe.maxTotalThreadsPerThreadgroup / 16))
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgw, height: tgh, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error { throw MetalSetupError.libraryCompile("\(err)") }

        let field = IterationField(width: width, height: height)
        let iterPtr = iterBuf.contents().assumingMemoryBound(to: UInt32.self)
        let smoothPtr = smoothBuf.contents().assumingMemoryBound(to: Float32.self)
        for i in 0..<pixelCount {
            field.storage[i] = PixelResult(iterations: iterPtr[i], smooth: smoothPtr[i])
        }
        return field
    }
}
