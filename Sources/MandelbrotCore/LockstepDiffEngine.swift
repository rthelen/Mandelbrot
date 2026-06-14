import Foundation

/// Per-step (lockstep) HW-vs-SW comparison. Unlike `DoubleComparisonEngine`,
/// which compares only the final per-pixel result, this runs the hardware
/// `Double` and software `SoftDouble` iterations *in lockstep* and compares
/// every intermediate (`zx²`, `zy²`, `|z|²`, `zx'`, `zy'`) bit-for-bit on every
/// iteration. That is far more sensitive: a rounding bug is caught the instant
/// any single operation diverges on any pixel, long before (or whether or not)
/// it would ever change the rendered pixel.
///
/// On the first differing operation for a pixel, the pixel is flagged and its
/// iteration stops (the two have diverged; continuing would just compare
/// unrelated trajectories).
public struct LockstepDiffEngine: Sendable {
    /// When true, deliberately perturbs the *software* `zx'` result by 1 ULP on
    /// a scattered subset of pixels — a self-test that drives the detector and
    /// UI red on demand WITHOUT touching the real `SoftDouble` arithmetic. The
    /// fault lives only in this comparison harness.
    public let injectFault: Bool

    public init(injectFault: Bool = false) {
        self.injectFault = injectFault
    }

    public func renderLockstepDiff(
        viewport: Viewport,
        width: Int,
        height: Int,
        maxIterations: UInt32
    ) -> ComparisonField {
        let field = IterationField(width: width, height: height)
        let flags = UnsafeMutablePointer<Bool>.allocate(capacity: width * height)
        flags.initialize(repeating: false, count: width * height)
        let flagsBox = UnsafeSendablePtr(flags)

        let stripW = 32
        let stripsPerRow = (width + stripW - 1) / stripW
        let totalStrips = height * stripsPerRow

        let originX = viewport.originX(forWidth: width)
        let originY = viewport.originY(forHeight: height)
        let pxSize = viewport.pixelSize
        let dxHW = pxSize.asDouble

        DispatchQueue.concurrentPerform(iterations: totalStrips) { idx in
            let row = idx / stripsPerRow
            let strip = idx % stripsPerRow
            let pixelX = strip * stripW

            let sox = originX + pxSize * Float128(pixelX)
            let soy = originY - pxSize * Float128(row)
            let oxHW = sox.asDouble
            let cyHW = soy.asDouble
            let cySW = SoftDouble(cyHW)

            let remaining = min(stripW, width - pixelX)
            let base = row * width + pixelX
            for col in 0..<remaining {
                let gx = pixelX + col
                let cxHW = oxHW + dxHW * Double(col)
                let cxSW = SoftDouble(oxHW) + SoftDouble(dxHW) * SoftDouble(col)
                let perturb = injectFault && ((gx &+ row) % 5 == 0)
                let (result, divergence) = lockstepPixel(
                    cxHW: cxHW, cyHW: cyHW, cxSW: cxSW, cySW: cySW,
                    maxIterations: maxIterations, perturb: perturb
                )
                field.storage[base + col] = result
                flagsBox.p[base + col] = (divergence != nil)
            }
        }

        var count = 0
        var firstIndex = -1
        for i in 0..<(width * height) where flags[i] {
            count += 1
            if firstIndex < 0 { firstIndex = i }
        }

        // Re-run the first flagged pixel single-threaded to recover the rich
        // divergence detail (which op, which iteration, the differing bits).
        var sample: DivergenceSample? = nil
        if firstIndex >= 0 {
            let row = firstIndex / width
            let gx = firstIndex % width
            let stripStart = (gx / stripW) * stripW
            let col = gx - stripStart
            let sox = originX + pxSize * Float128(stripStart)
            let soy = originY - pxSize * Float128(row)
            let oxHW = sox.asDouble
            let cyHW = soy.asDouble
            let cxHW = oxHW + dxHW * Double(col)
            let cxSW = SoftDouble(oxHW) + SoftDouble(dxHW) * SoftDouble(col)
            let perturb = injectFault && ((gx &+ row) % 5 == 0)
            let (_, divergence) = lockstepPixel(
                cxHW: cxHW, cyHW: cyHW, cxSW: cxSW, cySW: SoftDouble(cyHW),
                maxIterations: maxIterations, perturb: perturb
            )
            if let d = divergence {
                sample = DivergenceSample(pixelX: gx, pixelY: row, iteration: d.iteration,
                                          label: d.label, hwBits: d.hwBits, swBits: d.swBits)
            }
        }

        return ComparisonField(field: field, flags: flags, mismatchCount: count,
                               firstDivergence: sample)
    }
}

/// One pixel's lockstep iteration. Computes each step in hardware `Double` and
/// software `SoftDouble` and compares bit-for-bit. Returns the hardware pixel
/// result (for display) and, if any operation diverged, which one.
@inlinable
func lockstepPixel(
    cxHW: Double, cyHW: Double,
    cxSW: SoftDouble, cySW: SoftDouble,
    maxIterations: UInt32,
    perturb: Bool = false
) -> (result: PixelResult, divergence: (iteration: UInt32, label: String, hwBits: UInt64, swBits: UInt64)?) {
    let bailHW = 4.0
    let twoSW: SoftDouble = 2.0

    var zxHW = 0.0, zyHW = 0.0
    var zxSW: SoftDouble = .zero, zySW: SoftDouble = .zero
    var n: UInt32 = 0
    var magSqHW = 0.0

    while n < maxIterations {
        let zx2HW = zxHW * zxHW, zy2HW = zyHW * zyHW
        let zx2SW = zxSW * zxSW, zy2SW = zySW * zySW
        if zx2HW.bitPattern != zx2SW.bits {
            return (.inSet, (n, "zx²", zx2HW.bitPattern, zx2SW.bits))
        }
        if zy2HW.bitPattern != zy2SW.bits {
            return (.inSet, (n, "zy²", zy2HW.bitPattern, zy2SW.bits))
        }

        let mHW = zx2HW + zy2HW
        let mSW = zx2SW + zy2SW
        if mHW.bitPattern != mSW.bits {
            return (.inSet, (n, "|z|²", mHW.bitPattern, mSW.bits))
        }
        magSqHW = mHW
        if mHW > bailHW { break }   // escaped — HW and SW agree on |z|², so same decision

        let nzxHW = zx2HW - zy2HW + cxHW
        var nzxSW = zx2SW - zy2SW + cxSW
        if perturb && n == 0 {
            nzxSW = SoftDouble(rawBits: nzxSW.bits ^ 1)   // self-test: 1-ULP fault
        }
        if nzxHW.bitPattern != nzxSW.bits {
            return (.inSet, (n, "zx'", nzxHW.bitPattern, nzxSW.bits))
        }
        let nzyHW = 2.0 * zxHW * zyHW + cyHW
        let nzySW = twoSW * zxSW * zySW + cySW
        if nzyHW.bitPattern != nzySW.bits {
            return (.inSet, (n, "zy'", nzyHW.bitPattern, nzySW.bits))
        }

        zxHW = nzxHW; zyHW = nzyHW
        zxSW = nzxSW; zySW = nzySW
        n &+= 1
    }

    let result: PixelResult = (n < maxIterations)
        ? PixelResult(iterations: n, smooth: smoothFraction(magSq: magSqHW))
        : .inSet
    return (result, nil)
}
