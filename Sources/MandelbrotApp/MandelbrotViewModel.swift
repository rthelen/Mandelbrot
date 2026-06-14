import Foundation
import CoreGraphics
import SwiftUI
import MandelbrotCore

/// How a kernel mode compares hardware vs software, if at all.
enum ComparisonMode: Sendable {
    case none
    case finalPixel       // compare only the rendered per-pixel result
    case lockstep         // compare every intermediate, every iteration
    case lockstepSelfTest // lockstep, but inject a 1-ULP fault to prove detection
}

enum KernelChoice: String, CaseIterable, Identifiable, Sendable {
    case doubleCPU = "Double (CPU)"
    case softDoubleCPU = "SoftDouble (CPU)"
    case softDoubleMetal = "SoftDouble (Metal GPU)"
    case doubleDiffCPU = "Double: HW vs SW (pixel-diff)"
    case doubleLockstepCPU = "Double: HW vs SW (per-step)"
    case selfTestCPU = "⚠ Self-test (inject fault)"
    case float128CPU = "Float128 (CPU)"
    case float128Metal = "Float128 (Metal GPU)"
    var id: String { rawValue }

    /// How this mode compares HW vs SW (drives the render path).
    var comparisonMode: ComparisonMode {
        switch self {
        case .doubleDiffCPU:     return .finalPixel
        case .doubleLockstepCPU: return .lockstep
        case .selfTestCPU:       return .lockstepSelfTest
        default:                 return .none
        }
    }

    /// True for modes that run two kernels and compare rather than rendering
    /// through a single `MandelbrotEngine`.
    var isComparison: Bool { comparisonMode != .none }

    var engine: any MandelbrotEngine {
        switch self {
        case .doubleCPU: return CPUEngine(kernel: DoubleStripKernel())
        case .softDoubleCPU: return CPUEngine(kernel: SoftDoubleStripKernel())
        case .softDoubleMetal: return MetalSoftDouble64Engine()
        // Comparison modes render via the comparison engines, not this path; the
        // hardware kernel is a harmless default so the switch stays total.
        case .doubleDiffCPU, .doubleLockstepCPU, .selfTestCPU: return CPUEngine(kernel: DoubleStripKernel())
        case .float128CPU: return CPUEngine(kernel: Float128StripKernel())
        case .float128Metal: return MetalFloat128UnpackedEngine()   // unpacked: ~15% faster, bit-identical
        }
    }

    /// Smallest `pixelSize` at which the kernel still produces a useful image.
    /// Below this, neighboring pixels collapse to the same `c` value and the
    /// image goes uniform-colored.
    /// - Double: 52 mantissa bits → ~2.2e-16 relative; ~1e-13 with a safety
    ///   margin for the multiplications in the iteration loop.
    /// - Float128: 112 mantissa bits → ~1.9e-34 relative; ~1e-31 with margin.
    var precisionFloor: Double {
        switch self {
        case .doubleCPU:       return 1e-13
        case .softDoubleCPU:   return 1e-13   // same format as Double
        case .softDoubleMetal: return 1e-13   // same format as Double
        case .doubleDiffCPU:   return 1e-13   // both operands are binary64
        case .doubleLockstepCPU: return 1e-13 // both operands are binary64
        case .selfTestCPU:     return 1e-13
        case .float128CPU:     return 1e-31
        case .float128Metal:   return 1e-31
        }
    }
}

struct RenderRequest: Sendable {
    let engine: any MandelbrotEngine
    let colorizer: any Colorizer
    let viewport: Viewport
    let width: Int
    let height: Int
    let maxIterations: UInt32
    let comparisonMode: ComparisonMode
}

struct RenderResult {
    let image: CGImage
    let mismatchCount: Int?          // nil when not comparing
    let divergence: DivergenceSample?  // first per-step divergence, lockstep mode only
}

/// Renders a request to an image. For comparison modes it also returns the
/// bitwise HW-vs-SW mismatch count (and, for lockstep, the first divergence).
func performRender(_ req: RenderRequest) -> RenderResult {
    switch req.comparisonMode {
    case .none:
        let field = req.engine.render(
            viewport: req.viewport, width: req.width, height: req.height,
            maxIterations: req.maxIterations
        )
        let img = req.colorizer.render(field: field, maxIterations: req.maxIterations)
        return RenderResult(image: img, mismatchCount: nil, divergence: nil)

    case .finalPixel:
        let comp = DoubleComparisonEngine().renderComparison(
            viewport: req.viewport, width: req.width, height: req.height,
            maxIterations: req.maxIterations
        )
        let img = renderComparisonImage(comp, colorizer: req.colorizer,
                                        maxIterations: req.maxIterations)
        return RenderResult(image: img, mismatchCount: comp.mismatchCount, divergence: nil)

    case .lockstep, .lockstepSelfTest:
        let inject = (req.comparisonMode == .lockstepSelfTest)
        let comp = LockstepDiffEngine(injectFault: inject).renderLockstepDiff(
            viewport: req.viewport, width: req.width, height: req.height,
            maxIterations: req.maxIterations
        )
        let img = renderComparisonImage(comp, colorizer: req.colorizer,
                                        maxIterations: req.maxIterations)
        return RenderResult(image: img, mismatchCount: comp.mismatchCount,
                            divergence: comp.firstDivergence)
    }
}

@MainActor
final class MandelbrotViewModel: ObservableObject {
    @Published var image: CGImage?
    @Published var viewport: Viewport = Viewport(centerX: -0.75, centerY: 0.0, pixelSize: 0.005)
    @Published var maxIterations: UInt32 = 512 { didSet { renderRequested() } }
    @Published var kernelChoice: KernelChoice = .doubleCPU {
        didSet { resetMismatchLatch(); renderRequested() }
    }
    @Published var colorizerIndex: Int = 0 { didSet { renderRequested() } }

    @Published var isPlaying = false
    @Published var playbackTargetIndex: Int = 0
    @Published var playbackFrameCount: Int = 0
    @Published var currentFPS: Double = 0
    @Published var lastFrameMs: Double = 0
    @Published var playbackStoppedReason: String? = nil
    /// Bitwise HW-vs-SW mismatch count from the last comparison render; nil when
    /// not in a comparison kernel mode.
    @Published var mismatchCount: Int? = nil
    /// Sticky latch: the worst per-frame mismatch count seen since the run/scrub
    /// began, and the frame it first tripped. Stays set (red) even after later
    /// frames come back clean, so a transient divergence during an unattended
    /// dive isn't lost. Reset on play/jump/reset/kernel-change.
    @Published var peakMismatchCount: Int = 0
    @Published var peakMismatchFrame: Int? = nil
    /// Latched human-readable detail of the first per-step divergence (lockstep
    /// mode): which op, which iteration, which pixel, and the differing bits.
    @Published var divergenceDetail: String? = nil

    let colorizers: [any Colorizer] = AvailableColorizers.all()

    private var canvasWidth: Int = 0
    private var canvasHeight: Int = 0
    private var pendingRequest: RenderRequest?
    private var isRendering = false

    private var recentFrameTimes: [CFAbsoluteTime] = []
    private var lastRenderStart: CFAbsoluteTime = 0

    var currentPlaybackTarget: PlaybackTarget {
        PlaybackTarget.allPresets[playbackTargetIndex]
    }

    func setCanvasSize(width: Int, height: Int) {
        let w = max(1, width)
        let h = max(1, height)
        guard w != canvasWidth || h != canvasHeight else { return }
        let firstSize = canvasWidth == 0
        canvasWidth = w
        canvasHeight = h
        if firstSize {
            viewport = Viewport.defaultView(width: w, height: h)
        }
        renderRequested()
    }

    func renderRequested() {
        guard canvasWidth > 0, canvasHeight > 0 else { return }
        let req = RenderRequest(
            engine: kernelChoice.engine,
            colorizer: colorizers[colorizerIndex],
            viewport: viewport,
            width: canvasWidth,
            height: canvasHeight,
            maxIterations: maxIterations,
            comparisonMode: kernelChoice.comparisonMode
        )
        pendingRequest = req
        if !isRendering { startNextRender() }
    }

    private func startNextRender() {
        guard let req = pendingRequest else { return }
        pendingRequest = nil
        isRendering = true
        lastRenderStart = CFAbsoluteTimeGetCurrent()
        Task.detached(priority: .userInitiated) { [weak self] in
            let result = performRender(req)
            await self?.renderFinished(result)
        }
    }

    private func renderFinished(_ result: RenderResult) {
        self.image = result.image
        self.mismatchCount = result.mismatchCount
        if let n = result.mismatchCount {
            if n > peakMismatchCount { peakMismatchCount = n }
            if n > 0 && peakMismatchFrame == nil { peakMismatchFrame = playbackFrameCount }
        }
        // Latch the first per-step divergence detail seen this run.
        if let d = result.divergence, divergenceDetail == nil {
            divergenceDetail = String(
                format: "%@ diverged @ iter %u, px (%d,%d): HW %016llx · SW %016llx",
                d.label, d.iteration, d.pixelX, d.pixelY, d.hwBits, d.swBits)
        }
        isRendering = false

        let now = CFAbsoluteTimeGetCurrent()
        lastFrameMs = (now - lastRenderStart) * 1000.0

        if isPlaying {
            recordFrameTime(now)
            advancePlaybackFrame()
            if isPlaying {
                renderRequested()
                return
            }
        }
        startNextRender()
    }

    // MARK: - Player-piano

    func togglePlayback() {
        if isPlaying { stopPlayback() } else { startPlayback() }
    }

    func startPlayback() {
        // Begin from the default view so each run is reproducible.
        viewport = Viewport.defaultView(width: canvasWidth, height: canvasHeight)
        recentFrameTimes.removeAll()
        playbackFrameCount = 0
        currentFPS = 0
        playbackStoppedReason = nil
        resetMismatchLatch()
        isPlaying = true
        renderRequested()
    }

    /// Clear the sticky HW-vs-SW divergence latch.
    func resetMismatchLatch() {
        peakMismatchCount = 0
        peakMismatchFrame = nil
        divergenceDetail = nil
    }

    func stopPlayback() {
        isPlaying = false
    }

    /// Jump to a specific frame index along the current dive path. Stops any
    /// in-flight playback. Used to scrub through a dive and pick "interesting"
    /// frames for the bench tool.
    func jumpToDiveFrame(_ n: Int) {
        stopPlayback()
        resetMismatchLatch()
        let target = currentPlaybackTarget
        var v = Viewport.defaultView(width: canvasWidth, height: canvasHeight)
        let zoom = Float128(target.zoomRate)
        let ax = Float128(target.centerX)
        let ay = Float128(target.centerY)
        for _ in 0..<max(0, n) {
            v = v.zoomed(by: zoom, aroundX: ax, aroundY: ay)
        }
        viewport = v
        playbackFrameCount = n
        renderRequested()
    }

    private func advancePlaybackFrame() {
        let target = currentPlaybackTarget
        viewport = viewport.zoomed(
            by: Float128(target.zoomRate),
            aroundX: Float128(target.centerX),
            aroundY: Float128(target.centerY)
        )
        playbackFrameCount += 1

        let kernelFloor = kernelChoice.precisionFloor
        let pxSize = viewport.pixelSize.asDouble
        if pxSize < kernelFloor {
            isPlaying = false
            playbackStoppedReason = "precision floor for \(kernelChoice.rawValue) reached — try Float128 for deeper zoom"
        } else if pxSize < target.minPixelSize {
            isPlaying = false
            playbackStoppedReason = "dive complete"
        }
    }

    private func recordFrameTime(_ now: CFAbsoluteTime) {
        recentFrameTimes.append(now)
        // Keep a rolling 30-sample window.
        if recentFrameTimes.count > 30 { recentFrameTimes.removeFirst() }
        if recentFrameTimes.count >= 2 {
            let elapsed = recentFrameTimes.last! - recentFrameTimes.first!
            if elapsed > 0 {
                currentFPS = Double(recentFrameTimes.count - 1) / elapsed
            }
        }
    }

    // MARK: - Interaction (interrupts playback)

    func zoomIn(atPixelX x: Int, pixelY y: Int) {
        stopPlayback()
        zoom(factor: 0.5, atPixelX: x, pixelY: y)
    }
    func zoomOut(atPixelX x: Int, pixelY y: Int) {
        stopPlayback()
        zoom(factor: 2.0, atPixelX: x, pixelY: y)
    }

    private func zoom(factor: Double, atPixelX x: Int, pixelY y: Int) {
        let (wx, wy) = viewport.coordinate(
            atPixelX: x, pixelY: y, width: canvasWidth, height: canvasHeight
        )
        viewport = viewport.zoomed(by: Float128(factor), aroundX: wx, aroundY: wy)
        renderRequested()
    }

    func zoomToRect(originX: Int, originY: Int, width w: Int, height h: Int) {
        guard w > 2, h > 2 else { return }
        stopPlayback()
        viewport = viewport.zoomedToRect(
            originX: originX, originY: originY, width: w, height: h,
            imageWidth: canvasWidth, imageHeight: canvasHeight
        )
        renderRequested()
    }

    func pan(byPixelsX dx: Int, pixelsY dy: Int) {
        stopPlayback()
        viewport = viewport.panned(byPixelsX: dx, pixelsY: dy)
        renderRequested()
    }

    func resetView() {
        stopPlayback()
        resetMismatchLatch()
        viewport = Viewport.defaultView(width: canvasWidth, height: canvasHeight)
        renderRequested()
    }
}
