import XCTest
@testable import MandelbrotCore

private extension Double {
    /// The hardware result reinterpreted into SoftDouble's stored bits, so a
    /// bit-exact comparison against a computed `SoftDouble` is apples-to-apples.
    var softBits: UInt64 { SoftDouble(self).bits }
}

private func hex(_ x: UInt64) -> String { String(format: "0x%016llx", x) }

final class MandelbrotCoreTests: XCTestCase {
    func testFloat128BasicArithmetic() {
        let a: Float128 = 2.5
        let b: Float128 = 1.5
        XCTAssertEqual((a + b).asDouble, 4.0)
        XCTAssertEqual((a - b).asDouble, 1.0)
        XCTAssertEqual((a * b).asDouble, 3.75)
        XCTAssertEqual((-a).asDouble, -2.5)
        XCTAssertTrue(b < a)
    }

    /// Cross-validate Float128 against hardware Double across a representative
    /// range. The two should agree to within Double's relative precision
    /// (~2^-52 ≈ 2.22e-16) for values comfortably inside Double's range.
    func testFloat128MatchesDoubleAcrossRange() {
        let inputs: [Double] = [
            0.0, -0.0, 1.0, -1.0, 0.5, 0.25, 0.125, 0.1,
            3.14159265358979, -2.71828182845905,
            1e-10, 1e10, 1e-100, 1e100,
            42.0, 1729.0, -0.000123456789,
            1.0 / 3.0, 2.0 / 7.0,
        ]
        let tol = 1e-14

        for x in inputs {
            for y in inputs {
                let fx = Float128(x)
                let fy = Float128(y)

                XCTAssertEqual((fx + fy).asDouble, x + y, accuracy: max(tol, tol * abs(x + y)),
                               "add mismatch: \(x) + \(y)")
                XCTAssertEqual((fx - fy).asDouble, x - y, accuracy: max(tol, tol * abs(x - y)),
                               "sub mismatch: \(x) - \(y)")
                XCTAssertEqual((fx * fy).asDouble, x * y, accuracy: max(tol, tol * abs(x * y)),
                               "mul mismatch: \(x) * \(y)")
            }
        }
    }

    /// Round-trip Double → Float128 → Double should be exact for any Double
    /// that has a representable binary128 equivalent (which is all finite
    /// non-denormal Doubles).
    func testFloat128DoubleRoundTrip() {
        let inputs: [Double] = [
            0.0, 1.0, -1.0, 0.1, 0.5, .pi, .ulpOfOne, 1e-200, 1e200,
            Double.leastNormalMagnitude, Double.greatestFiniteMagnitude,
            -Double.greatestFiniteMagnitude,
        ]
        for x in inputs {
            XCTAssertEqual(Float128(x).asDouble.bitPattern, x.bitPattern,
                           "round-trip mismatch for \(x)")
        }
    }

    /// At Double-scale viewports, the Float128 strip kernel should produce the
    /// same iteration counts as the Double strip kernel.
    func testFloat128KernelMatchesDoubleKernel() {
        let viewport = Viewport.defaultView(width: 64, height: 64)
        let doubleEngine = CPUEngine(kernel: DoubleStripKernel())
        let f128Engine = CPUEngine(kernel: Float128StripKernel())

        let dField = doubleEngine.render(viewport: viewport, width: 64, height: 64, maxIterations: 256)
        let fField = f128Engine.render(viewport: viewport, width: 64, height: 64, maxIterations: 256)

        var mismatches = 0
        dField.withBufferPointer { dBuf in
            fField.withBufferPointer { fBuf in
                for i in 0..<(64 * 64) {
                    if dBuf[i].iterations != fBuf[i].iterations {
                        mismatches += 1
                    }
                }
            }
        }
        // Allow a small handful of boundary pixels to differ by one iteration
        // due to rounding in the smoothing/escape check at the edge of the set.
        XCTAssertLessThan(mismatches, 10, "Too many iteration mismatches between kernels")
    }

    // MARK: - SoftDouble (software binary64)

    func testSoftDoubleBasicArithmetic() {
        let a: SoftDouble = 2.5
        let b: SoftDouble = 1.5
        XCTAssertEqual((a + b).asDouble, 4.0)
        XCTAssertEqual((a - b).asDouble, 1.0)
        XCTAssertEqual((a * b).asDouble, 3.75)
        XCTAssertEqual((-a).asDouble, -2.5)
        XCTAssertTrue(b < a)
    }

    /// SoftDouble implements the same binary64 format as hardware `Double`, so
    /// add/sub/mul must agree with hardware BIT FOR BIT (round-to-nearest-even)
    /// whenever the operands and result stay in the normal range. This is the
    /// core validation: the hand-written FP reproduces silicon exactly.
    ///
    /// One documented exemption: the *sign* of an exact-zero result. Hardware
    /// follows IEEE's "sum of cancelling operands is +0 in round-to-nearest"
    /// rule; the lean kernel doesn't spend branches on it (and Mandelbrot never
    /// depends on a zero's sign — any `0 - 0` is followed by `+ c`). So zero
    /// results are required to be zero, but either sign is accepted.
    func testSoftDoubleMatchesHardwareBitExact() {
        let inputs: [Double] = [
            0.0, -0.0, 1.0, -1.0, 0.5, 0.25, 0.125, 0.1,
            3.14159265358979, -2.71828182845905,
            1e-10, 1e10, 1e-100, 1e100, 1e200, 1e-200,
            42.0, 1729.0, -0.000123456789,
            1.0 / 3.0, 2.0 / 7.0, .pi, -.pi,
            123456.789, 0.9999999999999, 1.0000000000001,
        ]

        // Bit-exact, except an exact-zero result may differ in sign only.
        func assertMatch(_ soft: SoftDouble, _ hard: Double, _ msg: String) {
            // SoftDouble flushes subnormal results to zero (NO_NAN_INF), so only
            // normal-or-zero hardware results are required to match.
            guard hard == 0 || hard.isNormal else { return }
            if hard == 0 {
                XCTAssertEqual(soft.asDouble, 0.0, msg)   // ±0 both fine
            } else {
                XCTAssertEqual(soft.bits, hard.softBits, msg)
            }
        }

        for x in inputs {
            for y in inputs {
                let sx = SoftDouble(x)
                let sy = SoftDouble(y)
                assertMatch(sx + sy, x + y, "add mismatch: \(x) + \(y)")
                assertMatch(sx - sy, x - y, "sub mismatch: \(x) - \(y)")
                assertMatch(sx * sy, x * y, "mul mismatch: \(x) * \(y)")
            }
        }
    }

    /// Brute-force probe: scan many operand pairs (varied magnitudes, so the
    /// alignment shift drops nonzero low bits) and count where SoftDouble add/
    /// sub/mul disagree with hardware bit-for-bit. Prints the first few. This
    /// pins down whether the discrepancy seen in bench checksums is in add/sub
    /// (alignment-sticky rounding) or multiply.
    func testSoftDoubleBruteForceVsHardware() {
        // Deterministic LCG so we don't depend on Math.random.
        var state: UInt64 = 0x9E3779B97F4A7C15
        func nextDouble() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            // Mantissa in [1,2), random small exponent spread.
            let m = Double(state >> 12) / Double(1 << 52)        // [0,1)
            let e = Int((state >> 3) & 0x3F) - 32                // [-32,31]
            let sign = (state & 1) == 0 ? 1.0 : -1.0
            return sign * (1.0 + m) * pow(2.0, Double(e))
        }

        var addBad = 0, subBad = 0, mulBad = 0, total = 0
        var samples: [String] = []
        for _ in 0..<200_000 {
            let x = nextDouble(), y = nextDouble()
            let sx = SoftDouble(x), sy = SoftDouble(y)
            total += 1

            func check(_ soft: SoftDouble, _ hard: Double, _ op: String, _ bad: inout Int) {
                guard hard == 0 || hard.isNormal else { return }
                let want = hard == 0 ? UInt64(0) : hard.bitPattern
                let got = hard == 0 ? (soft.asDouble == 0 ? UInt64(0) : soft.bits) : soft.bits
                if got != want {
                    bad += 1
                    if samples.count < 8 {
                        let d = Int64(bitPattern: got) - Int64(bitPattern: want)
                        samples.append("\(op): \(x) ⊕ \(y) → got \(soft.asDouble) want \(hard) (Δbits=\(d))")
                    }
                }
            }
            check(sx + sy, x + y, "add", &addBad)
            check(sx - sy, x - y, "sub", &subBad)
            check(sx * sy, x * y, "mul", &mulBad)
        }
        if addBad + subBad + mulBad > 0 {
            print("[brute] pairs=\(total) addBad=\(addBad) subBad=\(subBad) mulBad=\(mulBad)")
            for s in samples { print("  \(s)") }
        }
        XCTAssertEqual(addBad, 0, "SoftDouble add must match hardware bit-for-bit")
        XCTAssertEqual(subBad, 0, "SoftDouble sub must match hardware bit-for-bit")
        XCTAssertEqual(mulBad, 0, "SoftDouble mul must match hardware bit-for-bit")
    }

    /// Reinterpret round-trip Double → SoftDouble → Double is exact for every
    /// finite normal Double (same packed layout).
    func testSoftDoubleRoundTrip() {
        let inputs: [Double] = [
            1.0, -1.0, 0.1, 0.5, .pi, 1e-200, 1e200,
            Double.leastNormalMagnitude, Double.greatestFiniteMagnitude,
            -Double.greatestFiniteMagnitude,
        ]
        for x in inputs {
            XCTAssertEqual(SoftDouble(x).asDouble.bitPattern, x.bitPattern,
                           "round-trip mismatch for \(x)")
        }
    }

    /// The SoftDouble kernel and the hardware Double kernel run the identical
    /// algorithm in the identical format, so they must agree on EVERY pixel —
    /// no boundary slack. This is the cross-surface validation.
    func testSoftDoubleKernelMatchesDoubleKernelExactly() {
        let viewport = Viewport.defaultView(width: 64, height: 64)
        let doubleEngine = CPUEngine(kernel: DoubleStripKernel())
        let softEngine = CPUEngine(kernel: SoftDoubleStripKernel())

        let dField = doubleEngine.render(viewport: viewport, width: 64, height: 64, maxIterations: 512)
        let sField = softEngine.render(viewport: viewport, width: 64, height: 64, maxIterations: 512)

        var mismatches = 0
        dField.withBufferPointer { dBuf in
            sField.withBufferPointer { sBuf in
                for i in 0..<(64 * 64) where dBuf[i].iterations != sBuf[i].iterations {
                    mismatches += 1
                }
            }
        }
        XCTAssertEqual(mismatches, 0, "SoftDouble kernel must match Double kernel pixel-for-pixel")
    }

    /// The HW-vs-SW comparison engine must report zero bitwise mismatches across
    /// the surface — the live self-check the diff viewer relies on. Checked at a
    /// shallow and a deeper viewport, and at high iteration count.
    func testDoubleComparisonEngineReportsZeroMismatches() {
        let engine = DoubleComparisonEngine()
        // Render at full size on the seahorse dive (where the now-fixed sticky
        // bug is known to perturb pixels) so this is a sensitive regression
        // guard, not just a happy-path shallow check.
        let viewports = [
            Viewport.defaultView(width: 600, height: 600),
            Viewport(centerX: -0.7471847332221527, centerY: 0.06750650337518614,
                     pixelSize: 0.001754405924876314),    // seahorse frame 30
        ]
        for vp in viewports {
            let comp = engine.renderComparison(viewport: vp, width: 600, height: 600,
                                               maxIterations: 1024)
            XCTAssertEqual(comp.mismatchCount, 0,
                           "HW Double and SW SoftDouble must agree bit-for-bit on every pixel")
        }
    }

    /// Per-step lockstep comparison: HW and SW must agree on every intermediate
    /// of every iteration of every pixel. Far more sensitive than the final-
    /// pixel diff. Zero divergences expected with the correct kernels.
    func testLockstepDiffEngineReportsZeroDivergences() {
        let engine = LockstepDiffEngine()
        let viewports = [
            Viewport.defaultView(width: 400, height: 400),
            Viewport(centerX: -0.7471847332221527, centerY: 0.06750650337518614,
                     pixelSize: 0.001754405924876314),    // seahorse frame 30
        ]
        for vp in viewports {
            let comp = engine.renderLockstepDiff(viewport: vp, width: 400, height: 400,
                                                 maxIterations: 1024)
            XCTAssertEqual(comp.mismatchCount, 0,
                           "HW and SW must agree at every iteration step; first: \(String(describing: comp.firstDivergence))")
            XCTAssertNil(comp.firstDivergence)
        }
    }

    /// The self-test fault injection must actually drive the detector: with
    /// injectFault on, the lockstep engine should report divergences and name
    /// the perturbed op (`zx'`). Guards the "watch it go red" demo path.
    func testLockstepSelfTestInjectionIsDetected() {
        let comp = LockstepDiffEngine(injectFault: true).renderLockstepDiff(
            viewport: Viewport.defaultView(width: 200, height: 200),
            width: 200, height: 200, maxIterations: 256)
        XCTAssertGreaterThan(comp.mismatchCount, 0, "injected fault must be detected")
        XCTAssertEqual(comp.firstDivergence?.label, "zx'")
    }

    /// The MSL software-binary64 port must reproduce the CPU `SoftDouble`
    /// bit-for-bit on the GPU (and hence hardware Double, since CPU SoftDouble
    /// is proven equal to it). Pure integer arithmetic → exact agreement.
    func testGPUSoftDouble64MatchesCPU() throws {
        // Same deterministic LCG as the CPU brute-force test.
        var state: UInt64 = 0x9E3779B97F4A7C15
        func nextDouble() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let m = Double(state >> 12) / Double(1 << 52)
            let e = Int((state >> 3) & 0x3F) - 32
            let sign = (state & 1) == 0 ? 1.0 : -1.0
            return sign * (1.0 + m) * pow(2.0, Double(e))
        }

        let count = 100_000
        var aBits = [UInt64](repeating: 0, count: count)
        var bBits = [UInt64](repeating: 0, count: count)
        for i in 0..<count {
            aBits[i] = nextDouble().bitPattern
            bBits[i] = nextDouble().bitPattern
        }

        guard let (gpuAdd, gpuMul) = try gpuSoftDouble64Ops(a: aBits, b: bBits) else {
            throw XCTSkip("No Metal device available")
        }

        var addBad = 0, mulBad = 0
        var sample = ""
        for i in 0..<count {
            let sa = SoftDouble(rawBits: aBits[i])
            let sb = SoftDouble(rawBits: bBits[i])
            let cpuAdd = (sa + sb).bits
            let cpuMul = (sa * sb).bits
            if gpuAdd[i] != cpuAdd {
                addBad += 1
                if sample.isEmpty {
                    sample = "add[\(i)]: a=\(hex(aBits[i])) b=\(hex(bBits[i])) gpu=\(hex(gpuAdd[i])) cpu=\(hex(cpuAdd))"
                }
            }
            if gpuMul[i] != cpuMul {
                mulBad += 1
                if sample.isEmpty {
                    sample = "mul[\(i)]: a=\(hex(aBits[i])) b=\(hex(bBits[i])) gpu=\(hex(gpuMul[i])) cpu=\(hex(cpuMul))"
                }
            }
        }
        XCTAssertEqual(addBad, 0, "GPU add must match CPU SoftDouble bit-for-bit. \(sample)")
        XCTAssertEqual(mulBad, 0, "GPU mul must match CPU SoftDouble bit-for-bit. \(sample)")
    }

    /// The GPU software-64 Mandelbrot kernel must match the CPU software-64
    /// kernel on every pixel's iteration count (the precision-sensitive result).
    /// Since CPU SoftDouble == hardware Double, this also matches the hardware
    /// Double render. Checked shallow and deep.
    func testGPUMandelbrotMatchesCPU() throws {
        guard MetalContext.shared != nil else { throw XCTSkip("No Metal device available") }
        let gpu = MetalSoftDouble64Engine()
        let cpu = CPUEngine(kernel: SoftDoubleStripKernel())
        let hw = CPUEngine(kernel: DoubleStripKernel())

        let cases: [(Viewport, Int, Int, UInt32)] = [
            (Viewport.defaultView(width: 320, height: 240), 320, 240, 512),
            (Viewport(centerX: -0.7471847332221527, centerY: 0.06750650337518614,
                      pixelSize: 0.001754405924876314), 320, 240, 1024),  // seahorse frame 30
        ]

        for (vp, w, h, iters) in cases {
            let gField = gpu.render(viewport: vp, width: w, height: h, maxIterations: iters)
            let cField = cpu.render(viewport: vp, width: w, height: h, maxIterations: iters)
            let hField = hw.render(viewport: vp, width: w, height: h, maxIterations: iters)

            var gpuVsSoft = 0, gpuVsHW = 0
            gField.withBufferPointer { g in
                cField.withBufferPointer { c in
                    hField.withBufferPointer { hh in
                        for i in 0..<(w * h) {
                            if g[i].iterations != c[i].iterations { gpuVsSoft += 1 }
                            if g[i].iterations != hh[i].iterations { gpuVsHW += 1 }
                        }
                    }
                }
            }
            XCTAssertEqual(gpuVsSoft, 0, "GPU soft-64 must match CPU soft-64 iterations")
            XCTAssertEqual(gpuVsHW, 0, "GPU soft-64 must match hardware Double iterations")
        }
    }

    /// GPU software-binary128 add/mul must match the CPU `Float128` bit-for-bit
    /// over random full-width normal operands. Validates the MSL u128 port.
    func testGPUFloat128MatchesCPU() throws {
        guard MetalContext.shared != nil else { throw XCTSkip("No Metal device available") }

        // Random *normal* binary128 bit patterns via LCG: random sign, exponent
        // in a safe mid-range (no over/underflow), full 112-bit mantissa.
        var state: UInt64 = 0xD1B54A32D192ED03
        func next64() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
        func randomNormal() -> UInt128 {
            let sign = UInt128(next64() & 1) << 127
            let exp = UInt128(8000 + (next64() % 16000)) << 112    // safe exponent range
            let mantHi = UInt128(next64() & ((UInt64(1) << 48) - 1)) << 64
            let mantLo = UInt128(next64())
            return sign | exp | mantHi | mantLo
        }

        let count = 50_000
        var aBits = [UInt128](repeating: 0, count: count)
        var bBits = [UInt128](repeating: 0, count: count)
        for i in 0..<count { aBits[i] = randomNormal(); bBits[i] = randomNormal() }

        guard let (gpuAdd, gpuMul) = try gpuFloat128Ops(a: aBits, b: bBits) else {
            throw XCTSkip("No Metal device available")
        }

        var addBad = 0, mulBad = 0
        for i in 0..<count {
            let fa = Float128(rawBits: aBits[i])
            let fb = Float128(rawBits: bBits[i])
            if gpuAdd[i] != (fa + fb).bits { addBad += 1 }
            if gpuMul[i] != (fa * fb).bits { mulBad += 1 }
        }
        XCTAssertEqual(addBad, 0, "GPU f128 add must match CPU Float128 bit-for-bit")
        XCTAssertEqual(mulBad, 0, "GPU f128 mul must match CPU Float128 bit-for-bit")
    }

    /// GPU software-128 Mandelbrot must match the CPU Float128 kernel on every
    /// pixel's iteration count, shallow and deep (past Double's precision wall).
    func testGPUFloat128MandelbrotMatchesCPU() throws {
        guard MetalContext.shared != nil else { throw XCTSkip("No Metal device available") }
        let gpu = MetalFloat128Engine()
        let cpu = CPUEngine(kernel: Float128StripKernel())

        let cases: [(Viewport, Int, Int, UInt32)] = [
            (Viewport.defaultView(width: 256, height: 192), 256, 192, 512),
            (Viewport(centerX: -0.743643887037151, centerY: 0.131825904205330,
                      pixelSize: 1e-20), 256, 192, 1024),   // deep past Double
        ]
        for (vp, w, h, iters) in cases {
            let g = gpu.render(viewport: vp, width: w, height: h, maxIterations: iters)
            let c = cpu.render(viewport: vp, width: w, height: h, maxIterations: iters)
            var bad = 0
            g.withBufferPointer { gb in
                c.withBufferPointer { cb in
                    for i in 0..<(w * h) where gb[i].iterations != cb[i].iterations { bad += 1 }
                }
            }
            XCTAssertEqual(bad, 0, "GPU Float128 must match CPU Float128 iterations (px=\(vp.pixelSize.asDouble))")
        }
    }

    /// The parameterized limb-float (K=4, binary128 params) must reproduce CPU
    /// Float128 add/mul bit-for-bit — the new format validated against a known
    /// reference. Same generic limb code, just K=4.
    func testGPULimbFloat4MatchesFloat128() throws {
        guard MetalContext.shared != nil else { throw XCTSkip("No Metal device available") }
        var state: UInt64 = 0xABCDEF0123456789
        func next64() -> UInt64 { state = state &* 6364136223846793005 &+ 1442695040888963407; return state }
        func randomNormal128() -> UInt128 {
            let sign = UInt128(next64() & 1) << 127
            let exp = UInt128(8000 + (next64() % 16000)) << 112
            let mant = (UInt128(next64() & ((UInt64(1) << 48) - 1)) << 64) | UInt128(next64())
            return sign | exp | mant
        }
        let count = 20_000
        var aLimbs = [UInt32](), bLimbs = [UInt32]()
        var aVals = [UInt128](), bVals = [UInt128]()
        for _ in 0..<count {
            let a = randomNormal128(), b = randomNormal128()
            aVals.append(a); bVals.append(b)
            aLimbs += limbs(of: a); bLimbs += limbs(of: b)
        }
        guard let (gAdd, gMul) = try gpuLimbFloatOps(tag: "4", limbs: 4, a: aLimbs, b: bLimbs) else {
            throw XCTSkip("No Metal device available")
        }
        var addBad = 0, mulBad = 0
        for i in 0..<count {
            let fa = Float128(rawBits: aVals[i]), fb = Float128(rawBits: bVals[i])
            if uint128(fromLimbs: gAdd[i*4..<i*4+4]) != (fa + fb).bits { addBad += 1 }
            if uint128(fromLimbs: gMul[i*4..<i*4+4]) != (fa * fb).bits { mulBad += 1 }
        }
        XCTAssertEqual(addBad, 0, "limbfloat<4> add must match Float128")
        XCTAssertEqual(mulBad, 0, "limbfloat<4> mul must match Float128")
    }

    /// Same generic code at K=2 (binary64 params) must reproduce CPU SoftDouble.
    func testGPULimbFloat2MatchesSoftDouble() throws {
        guard MetalContext.shared != nil else { throw XCTSkip("No Metal device available") }
        var state: UInt64 = 0x13579BDF02468ACE
        func nextDouble() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let m = Double(state >> 12) / Double(1 << 52)
            let e = Int((state >> 3) & 0x3F) - 32
            return ((state & 1) == 0 ? 1.0 : -1.0) * (1.0 + m) * pow(2.0, Double(e))
        }
        let count = 20_000
        var aLimbs = [UInt32](), bLimbs = [UInt32]()
        var aVals = [UInt64](), bVals = [UInt64]()
        for _ in 0..<count {
            let a = nextDouble().bitPattern, b = nextDouble().bitPattern
            aVals.append(a); bVals.append(b)
            aLimbs += limbs(of: a); bLimbs += limbs(of: b)
        }
        guard let (gAdd, gMul) = try gpuLimbFloatOps(tag: "2", limbs: 2, a: aLimbs, b: bLimbs) else {
            throw XCTSkip("No Metal device available")
        }
        var addBad = 0, mulBad = 0
        for i in 0..<count {
            let sa = SoftDouble(rawBits: aVals[i]), sb = SoftDouble(rawBits: bVals[i])
            if uint64(fromLimbs: gAdd[i*2..<i*2+2]) != (sa + sb).bits { addBad += 1 }
            if uint64(fromLimbs: gMul[i*2..<i*2+2]) != (sa * sb).bits { mulBad += 1 }
        }
        XCTAssertEqual(addBad, 0, "limbfloat<2> add must match SoftDouble")
        XCTAssertEqual(mulBad, 0, "limbfloat<2> mul must match SoftDouble")
    }

    /// The unpacked Parts128 (64-bit limb) kernel must match CPU Float128 on
    /// every pixel — the faster variant must still be bit-exact.
    func testGPUFloat128UnpackedMatchesCPU() throws {
        guard MetalContext.shared != nil else { throw XCTSkip("No Metal device available") }
        let gpu = MetalFloat128UnpackedEngine()
        let cpu = CPUEngine(kernel: Float128StripKernel())
        let cases: [(Viewport, Int, Int, UInt32)] = [
            (Viewport.defaultView(width: 256, height: 192), 256, 192, 512),
            (Viewport(centerX: -0.743643887037151, centerY: 0.131825904205330,
                      pixelSize: 1e-20), 256, 192, 1024),
        ]
        for (vp, w, h, iters) in cases {
            let g = gpu.render(viewport: vp, width: w, height: h, maxIterations: iters)
            let c = cpu.render(viewport: vp, width: w, height: h, maxIterations: iters)
            var bad = 0
            g.withBufferPointer { gb in c.withBufferPointer { cb in
                for i in 0..<(w*h) where gb[i].iterations != cb[i].iterations { bad += 1 }
            }}
            XCTAssertEqual(bad, 0, "unpacked Parts128 kernel must match Float128 (px=\(vp.pixelSize.asDouble))")
        }
    }

    /// The unpacked-LimbFloat Mandelbrot kernel must match CPU Float128 on every
    /// pixel's iteration count — shallow and deep past Double's wall.
    func testGPULimbFloat128MandelbrotMatchesCPU() throws {
        guard MetalContext.shared != nil else { throw XCTSkip("No Metal device available") }
        let gpu = MetalLimbFloat128Engine()
        let cpu = CPUEngine(kernel: Float128StripKernel())
        let cases: [(Viewport, Int, Int, UInt32)] = [
            (Viewport.defaultView(width: 256, height: 192), 256, 192, 512),
            (Viewport(centerX: -0.743643887037151, centerY: 0.131825904205330,
                      pixelSize: 1e-20), 256, 192, 1024),
        ]
        for (vp, w, h, iters) in cases {
            let g = gpu.render(viewport: vp, width: w, height: h, maxIterations: iters)
            let c = cpu.render(viewport: vp, width: w, height: h, maxIterations: iters)
            var bad = 0
            g.withBufferPointer { gb in c.withBufferPointer { cb in
                for i in 0..<(w*h) where gb[i].iterations != cb[i].iterations { bad += 1 }
            }}
            XCTAssertEqual(bad, 0, "LimbFloat kernel must match Float128 (px=\(vp.pixelSize.asDouble))")
        }
    }

    /// The hybrid CPU+GPU engine must produce a field identical to the pure CPU
    /// Float128 render — the split boundary must not introduce seams or gaps.
    func testHybridFloat128MatchesCPU() throws {
        let cpu = CPUEngine(kernel: Float128StripKernel())
        let hybrid = HybridFloat128Engine(initialGPUFraction: 0.5)   // force a real split
        let vp = Viewport(centerX: -0.743643887037151, centerY: 0.131825904205330, pixelSize: 1e-18)
        let (w, h) = (200, 160)
        // Run a few frames so the split adapts; each must still be exact.
        for _ in 0..<3 {
            let hf = hybrid.render(viewport: vp, width: w, height: h, maxIterations: 600)
            let cf = cpu.render(viewport: vp, width: w, height: h, maxIterations: 600)
            var bad = 0
            hf.withBufferPointer { hb in cf.withBufferPointer { cb in
                for i in 0..<(w*h) where hb[i].iterations != cb[i].iterations { bad += 1 }
            }}
            XCTAssertEqual(bad, 0, "hybrid field must match pure CPU Float128 (splitRow=\(hybrid.lastSplitRow))")
        }
    }

    /// The SIMD Double kernel must be bit-identical to the scalar Double kernel
    /// (same arithmetic op-for-op, just data-parallel) — iterations AND smooth.
    func testSIMDDoubleMatchesScalar() {
        let scalar = CPUEngine(kernel: DoubleStripKernel())
        let simd = CPUEngine(kernel: SIMDDoubleStripKernel())
        let viewports = [
            Viewport.defaultView(width: 200, height: 150),
            Viewport(centerX: -0.7471847332221527, centerY: 0.06750650337518614,
                     pixelSize: 0.001754405924876314),
        ]
        for vp in viewports {
            let a = scalar.render(viewport: vp, width: 200, height: 150, maxIterations: 1024)
            let b = simd.render(viewport: vp, width: 200, height: 150, maxIterations: 1024)
            var bad = 0
            a.withBufferPointer { ab in b.withBufferPointer { bb in
                for i in 0..<(200*150) where ab[i] != bb[i] { bad += 1 }
            }}
            XCTAssertEqual(bad, 0, "SIMD Double kernel must match scalar exactly")
        }
    }

    /// The C Float128 kernel must match the Swift Float128 kernel exactly —
    /// iterations AND smooth (it's a 1:1 port) — shallow and deep past Double.
    func testCFloat128MatchesSwift() {
        let swiftK = CPUEngine(kernel: Float128StripKernel())
        let cK = CPUEngine(kernel: CFloat128StripKernel())
        let viewports = [
            Viewport.defaultView(width: 160, height: 120),
            Viewport(centerX: -0.743643887037151, centerY: 0.131825904205330, pixelSize: 1e-20),
        ]
        for vp in viewports {
            let a = swiftK.render(viewport: vp, width: 160, height: 120, maxIterations: 1024)
            let b = cK.render(viewport: vp, width: 160, height: 120, maxIterations: 1024)
            var bad = 0
            a.withBufferPointer { ab in b.withBufferPointer { bb in
                for i in 0..<(160*120) where ab[i] != bb[i] { bad += 1 }
            }}
            XCTAssertEqual(bad, 0, "C Float128 kernel must match Swift exactly (px=\(vp.pixelSize.asDouble))")
        }
    }

    func testViewportRoundTrip() {
        let v = Viewport(centerX: -0.5, centerY: 0.0, pixelSize: 0.01)
        let (wx, wy) = v.coordinate(atPixelX: 100, pixelY: 50, width: 200, height: 100)
        XCTAssertEqual(wx.asDouble, -0.5, accuracy: 1e-9)
        XCTAssertEqual(wy.asDouble, 0.0, accuracy: 1e-9)
    }

    func testDoubleKernelEscapesOutside() {
        let kernel = DoubleStripKernel()
        var output = [PixelResult](repeating: .inSet, count: 32)
        output.withUnsafeMutableBufferPointer { buf in
            kernel.computeStrip(
                originX: 2.0, originY: 0.0, deltaX: 0.1,
                maxIterations: 256, output: buf.baseAddress!
            )
        }
        // Points far outside the set should escape quickly.
        XCTAssertLessThan(output[0].iterations, 256)
    }

    func testDoubleKernelStaysInSet() {
        let kernel = DoubleStripKernel()
        var output = [PixelResult](repeating: PixelResult(iterations: 0, smooth: 0), count: 32)
        output.withUnsafeMutableBufferPointer { buf in
            kernel.computeStrip(
                originX: -0.5, originY: 0.0, deltaX: 0.0,
                maxIterations: 256, output: buf.baseAddress!
            )
        }
        // The origin (-0.5, 0) is in the set; should hit maxIterations.
        XCTAssertEqual(output[0].iterations, .max)
    }

    func testCPUEngineProducesNonEmptyImage() {
        let engine = CPUEngine(kernel: DoubleStripKernel())
        let viewport = Viewport.defaultView(width: 80, height: 60)
        let field = engine.render(viewport: viewport, width: 80, height: 60, maxIterations: 128)
        XCTAssertEqual(field.width, 80)
        XCTAssertEqual(field.height, 60)
        // At least some pixels should be in the set (center) and some escaped (edges).
        var inSet = 0; var escaped = 0
        field.withBufferPointer { buf in
            for r in buf {
                if r.iterations == .max { inSet += 1 } else { escaped += 1 }
            }
        }
        XCTAssertGreaterThan(inSet, 0)
        XCTAssertGreaterThan(escaped, 0)
    }
}
