import XCTest
@testable import MandelbrotCore

private extension Double {
    /// The hardware result reinterpreted into SoftDouble's stored bits, so a
    /// bit-exact comparison against a computed `SoftDouble` is apples-to-apples.
    var softBits: UInt64 { SoftDouble(self).bits }
}

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
