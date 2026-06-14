import Foundation
import Metal

/// Runs the GPU software-binary64 self-test: for each index, computes `a+b` and
/// `a*b` in `SoftDouble` arithmetic on the GPU. Operands and results are raw
/// binary64 bit patterns. Returns `nil` if no Metal device is available.
///
/// This is the bit-exactness proof for the MSL port: the results must equal the
/// CPU `SoftDouble` results exactly (the arithmetic is pure integer work).
public func gpuSoftDouble64Ops(a: [UInt64], b: [UInt64]) throws -> (add: [UInt64], mul: [UInt64])? {
    precondition(a.count == b.count)
    guard let ctx = MetalContext.shared else { return nil }
    let n = a.count
    if n == 0 { return (add: [], mul: []) }

    let pipe = try ctx.pipeline("sd_op_test")
    let dev = ctx.device
    let byteLen = n * MemoryLayout<UInt64>.stride

    guard
        let aBuf = dev.makeBuffer(bytes: a, length: byteLen, options: .storageModeShared),
        let bBuf = dev.makeBuffer(bytes: b, length: byteLen, options: .storageModeShared),
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
    enc.setBytes(&nn, length: MemoryLayout<UInt32>.size, index: 4)

    let tgWidth = min(pipe.maxTotalThreadsPerThreadgroup, 256)
    enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    if let err = cmd.error { throw MetalSetupError.libraryCompile("\(err)") }

    let addR = [UInt64](unsafeUninitializedCapacity: n) { buf, cnt in
        memcpy(buf.baseAddress!, addBuf.contents(), byteLen); cnt = n
    }
    let mulR = [UInt64](unsafeUninitializedCapacity: n) { buf, cnt in
        memcpy(buf.baseAddress!, mulBuf.contents(), byteLen); cnt = n
    }
    return (add: addR, mul: mulR)
}
