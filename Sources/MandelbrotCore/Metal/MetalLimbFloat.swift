import Foundation
import Metal

/// Runs the limb-float op-test for a given tag/limb-count: computes `a+b` and
/// `a*b` in the K-limb format on the GPU. Each value is `K` little-endian uint
/// limbs (the packed IEEE value split into limbs). Returns nil if no device.
public func gpuLimbFloatOps(tag: String, limbs K: Int,
                            a: [UInt32], b: [UInt32]) throws -> (add: [UInt32], mul: [UInt32])? {
    precondition(a.count == b.count && a.count % K == 0)
    guard let ctx = MetalContext.shared else { return nil }
    let n = a.count / K
    if n == 0 { return (add: [], mul: []) }

    let pipe = try ctx.pipeline("lf\(tag)_op_test")
    let dev = ctx.device
    let byteLen = a.count * MemoryLayout<UInt32>.stride

    var aa = a, bb = b
    guard
        let aBuf = dev.makeBuffer(bytes: &aa, length: byteLen, options: .storageModeShared),
        let bBuf = dev.makeBuffer(bytes: &bb, length: byteLen, options: .storageModeShared),
        let addBuf = dev.makeBuffer(length: byteLen, options: .storageModeShared),
        let mulBuf = dev.makeBuffer(length: byteLen, options: .storageModeShared),
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

    let addPtr = addBuf.contents().assumingMemoryBound(to: UInt32.self)
    let mulPtr = mulBuf.contents().assumingMemoryBound(to: UInt32.self)
    return (add: Array(UnsafeBufferPointer(start: addPtr, count: a.count)),
            mul: Array(UnsafeBufferPointer(start: mulPtr, count: a.count)))
}

/// GPU Mandelbrot engine using the unpacked LimbFloat<4> kernel — values stay
/// in LF4 form across the whole iteration (no per-op pack/unpack). Bit-exact to
/// Float128 (LF4 rounds to 113 bits). CPU fallback if no device.
public struct MetalLimbFloat128Engine: MandelbrotEngine {
    public init() {}
    private static let stripW = 32

    public func render(viewport: Viewport, width: Int, height: Int, maxIterations: UInt32) -> IterationField {
        guard let ctx = MetalContext.shared,
              let field = try? renderOnGPU(ctx: ctx, viewport: viewport, width: width,
                                           height: height, maxIterations: maxIterations)
        else {
            return CPUEngine(kernel: Float128StripKernel())
                .render(viewport: viewport, width: width, height: height, maxIterations: maxIterations)
        }
        return field
    }

    private func renderOnGPU(ctx: MetalContext, viewport: Viewport,
                             width: Int, height: Int, maxIterations: UInt32) throws -> IterationField {
        let (cxHL, cyHL) = float128PixelOrigins(viewport: viewport, width: width, height: height, stripW: Self.stripW)
        // Convert (hi, lo) ulong pairs to little-endian uint32 limb order [lo.lo, lo.hi, hi.lo, hi.hi].
        func toLimbs(_ hl: [UInt64]) -> [UInt32] {
            var out = [UInt32](repeating: 0, count: hl.count * 2)
            for v in 0..<(hl.count / 2) {
                let hi = hl[2*v], lo = hl[2*v + 1]
                out[4*v + 0] = UInt32(truncatingIfNeeded: lo)
                out[4*v + 1] = UInt32(truncatingIfNeeded: lo >> 32)
                out[4*v + 2] = UInt32(truncatingIfNeeded: hi)
                out[4*v + 3] = UInt32(truncatingIfNeeded: hi >> 32)
            }
            return out
        }
        var cxL = toLimbs(cxHL), cyL = toLimbs(cyHL)
        var dims = SIMD2<UInt32>(UInt32(width), UInt32(height))
        var maxIter = maxIterations
        let dev = ctx.device
        let pipe = try ctx.pipeline("mandelbrot_lf4")
        let pixelCount = width * height

        guard
            let cxBuf = dev.makeBuffer(bytes: &cxL, length: cxL.count * 4, options: .storageModeShared),
            let cyBuf = dev.makeBuffer(bytes: &cyL, length: cyL.count * 4, options: .storageModeShared),
            let iterBuf = dev.makeBuffer(length: pixelCount * 4, options: .storageModeShared),
            let smoothBuf = dev.makeBuffer(length: pixelCount * 4, options: .storageModeShared),
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
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
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

// Split/join helpers between the reference integer types and uint32 limbs
// (little-endian: limb 0 = least significant).
public func limbs(of v: UInt64) -> [UInt32] {
    [UInt32(truncatingIfNeeded: v), UInt32(truncatingIfNeeded: v >> 32)]
}
public func limbs(of v: UInt128) -> [UInt32] {
    [UInt32(truncatingIfNeeded: v),
     UInt32(truncatingIfNeeded: v >> 32),
     UInt32(truncatingIfNeeded: v >> 64),
     UInt32(truncatingIfNeeded: v >> 96)]
}
public func uint64(fromLimbs l: ArraySlice<UInt32>) -> UInt64 {
    let a = Array(l)
    return UInt64(a[0]) | (UInt64(a[1]) << 32)
}
public func uint128(fromLimbs l: ArraySlice<UInt32>) -> UInt128 {
    let a = Array(l)
    return UInt128(a[0]) | (UInt128(a[1]) << 32) | (UInt128(a[2]) << 64) | (UInt128(a[3]) << 96)
}
