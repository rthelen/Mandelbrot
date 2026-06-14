import Foundation
import Metal

@inlinable
func splitU128(_ v: UInt128) -> (hi: UInt64, lo: UInt64) {
    (UInt64(truncatingIfNeeded: v >> 64), UInt64(truncatingIfNeeded: v))
}

@inlinable
func joinU128(hi: UInt64, lo: UInt64) -> UInt128 {
    (UInt128(hi) << 64) | UInt128(lo)
}

/// GPU software-binary128 self-test: computes `a+b` and `a*b` in `Float128`
/// arithmetic on the GPU. Operands/results are raw binary128 bit patterns.
/// Returns `nil` if no Metal device. Results must equal the CPU `Float128`
/// bit-for-bit.
public func gpuFloat128Ops(a: [UInt128], b: [UInt128]) throws -> (add: [UInt128], mul: [UInt128])? {
    precondition(a.count == b.count)
    guard let ctx = MetalContext.shared else { return nil }
    let n = a.count
    if n == 0 { return (add: [], mul: []) }

    // Flatten to (hi, lo) ulong pairs.
    var aFlat = [UInt64](repeating: 0, count: 2 * n)
    var bFlat = [UInt64](repeating: 0, count: 2 * n)
    for i in 0..<n {
        let (ah, al) = splitU128(a[i]); aFlat[2*i] = ah; aFlat[2*i+1] = al
        let (bh, bl) = splitU128(b[i]); bFlat[2*i] = bh; bFlat[2*i+1] = bl
    }

    let dev = ctx.device
    let pipe = try ctx.pipeline("f128_op_test")
    let flatLen = 2 * n * MemoryLayout<UInt64>.stride

    guard
        let aBuf = dev.makeBuffer(bytes: &aFlat, length: flatLen, options: .storageModeShared),
        let bBuf = dev.makeBuffer(bytes: &bFlat, length: flatLen, options: .storageModeShared),
        let addBuf = dev.makeBuffer(length: flatLen, options: .storageModeShared),
        let mulBuf = dev.makeBuffer(length: flatLen, options: .storageModeShared),
        let cmd = ctx.queue.makeCommandBuffer(),
        let enc = cmd.makeComputeCommandEncoder()
    else { throw MetalSetupError.noDevice }

    var nn = UInt32(n)
    enc.setComputePipelineState(pipe)
    enc.setBuffer(aBuf, offset: 0, index: 0)
    enc.setBuffer(bBuf, offset: 0, index: 1)
    enc.setBuffer(addBuf, offset: 0, index: 2)
    enc.setBuffer(mulBuf, offset: 0, index: 3)
    enc.setBytes(&nn, length: 4, index: 4)
    let tgWidth = min(pipe.maxTotalThreadsPerThreadgroup, 256)
    enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    if let err = cmd.error { throw MetalSetupError.libraryCompile("\(err)") }

    let addPtr = addBuf.contents().assumingMemoryBound(to: UInt64.self)
    let mulPtr = mulBuf.contents().assumingMemoryBound(to: UInt64.self)
    var addR = [UInt128](repeating: 0, count: n)
    var mulR = [UInt128](repeating: 0, count: n)
    for i in 0..<n {
        addR[i] = joinU128(hi: addPtr[2*i], lo: addPtr[2*i+1])
        mulR[i] = joinU128(hi: mulPtr[2*i], lo: mulPtr[2*i+1])
    }
    return (add: addR, mul: mulR)
}

/// GPU Mandelbrot engine in software binary128. Per-pixel `c` is precomputed on
/// the CPU in full precision (matching `Float128StripKernel`'s strip slicing),
/// uploaded as (hi, lo) limb pairs, and iterated on the GPU. Falls back to the
/// CPU Float128 kernel when no Metal device is present.
public struct MetalFloat128Engine: MandelbrotEngine {
    public init() {}

    private static let stripW = 32

    public func render(
        viewport: Viewport, width: Int, height: Int, maxIterations: UInt32
    ) -> IterationField {
        guard let ctx = MetalContext.shared,
              let field = try? renderOnGPU(ctx: ctx, viewport: viewport,
                                           width: width, height: height,
                                           maxIterations: maxIterations)
        else {
            return CPUEngine(kernel: Float128StripKernel())
                .render(viewport: viewport, width: width, height: height,
                        maxIterations: maxIterations)
        }
        return field
    }

    private func renderOnGPU(
        ctx: MetalContext, viewport: Viewport,
        width: Int, height: Int, maxIterations: UInt32
    ) throws -> IterationField {
        let stripsPerRow = (width + Self.stripW - 1) / Self.stripW
        let originX = viewport.originX(forWidth: width)
        let originY = viewport.originY(forHeight: height)
        let pxSize = viewport.pixelSize

        // Per-column cx and per-row cy as binary128 bit patterns, computed in
        // Float128 exactly as CPUEngine + Float128StripKernel do.
        var cxFlat = [UInt64](repeating: 0, count: 2 * width)
        for s in 0..<stripsPerRow {
            let sox = originX + pxSize * Float128(s * Self.stripW)
            let cols = min(Self.stripW, width - s * Self.stripW)
            for c in 0..<cols {
                let gx = s * Self.stripW + c
                let cx = sox + pxSize * Float128(c)
                let (hi, lo) = splitU128(cx.bits)
                cxFlat[2*gx] = hi; cxFlat[2*gx+1] = lo
            }
        }
        var cyFlat = [UInt64](repeating: 0, count: 2 * height)
        for r in 0..<height {
            let soy = originY - pxSize * Float128(r)
            let (hi, lo) = splitU128(soy.bits)
            cyFlat[2*r] = hi; cyFlat[2*r+1] = lo
        }

        var dims = SIMD2<UInt32>(UInt32(width), UInt32(height))
        var maxIter = maxIterations
        let dev = ctx.device
        let pipe = try ctx.pipeline("mandelbrot_f128")
        let pixelCount = width * height

        guard
            let cxBuf = dev.makeBuffer(bytes: &cxFlat, length: cxFlat.count * 8, options: .storageModeShared),
            let cyBuf = dev.makeBuffer(bytes: &cyFlat, length: cyFlat.count * 8, options: .storageModeShared),
            let iterBuf = dev.makeBuffer(length: pixelCount * MemoryLayout<UInt32>.stride, options: .storageModeShared),
            let smoothBuf = dev.makeBuffer(length: pixelCount * MemoryLayout<Float32>.stride, options: .storageModeShared),
            let cmd = ctx.queue.makeCommandBuffer(),
            let enc = cmd.makeComputeCommandEncoder()
        else { throw MetalSetupError.noDevice }

        enc.setComputePipelineState(pipe)
        enc.setBuffer(cxBuf, offset: 0, index: 0)
        enc.setBuffer(cyBuf, offset: 0, index: 1)
        enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
        enc.setBytes(&maxIter, length: 4, index: 3)
        enc.setBuffer(iterBuf, offset: 0, index: 4)
        enc.setBuffer(smoothBuf, offset: 0, index: 5)

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
