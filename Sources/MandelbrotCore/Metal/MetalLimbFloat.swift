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
