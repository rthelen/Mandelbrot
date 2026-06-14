import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CryptoKit
import MandelbrotCore

// MARK: - CLI options

struct Options {
    var target: String?           // nil → all (matrix)
    var kernel: KernelName?       // nil → all (matrix)
    var iters: [UInt32] = [1024]
    var frames: Int = 200
    var sample: [Int]? = nil      // if set, render only these dive-frame indices
    var width: Int = 800
    var height: Int = 600
    var outputDir: String = "bench-results"
    var matrix: Bool = false
    var saveImages: Bool = false
    var imageEvery: Int = 1
    var profile: Bool = false

    static func parse(_ args: [String]) -> Options {
        var opts = Options()
        var i = 1
        while i < args.count {
            let arg = args[i]
            let next: () -> String? = {
                guard i + 1 < args.count else { return nil }
                i += 1
                return args[i]
            }
            switch arg {
            case "--target": opts.target = next()
            case "--kernel":
                if let v = next(), let k = KernelName(rawValue: v) { opts.kernel = k }
            case "--iters":
                if let v = next() {
                    opts.iters = v.split(separator: ",").compactMap { UInt32($0) }
                }
            case "--frames": if let v = next(), let n = Int(v) { opts.frames = n }
            case "--sample":
                if let v = next() {
                    opts.sample = v.split(separator: ",").compactMap { Int($0) }
                }
            case "--width":  if let v = next(), let n = Int(v) { opts.width = n }
            case "--height": if let v = next(), let n = Int(v) { opts.height = n }
            case "--output": if let v = next() { opts.outputDir = v }
            case "--matrix": opts.matrix = true
            case "--profile": opts.profile = true
            case "--save-images": opts.saveImages = true
            case "--image-every": if let v = next(), let n = Int(v) { opts.imageEvery = n }
            case "--help", "-h":
                printUsage(); exit(0)
            default:
                FileHandle.standardError.write(Data("unknown arg: \(arg)\n".utf8))
                exit(1)
            }
            i += 1
        }
        return opts
    }
}

func printUsage() {
    print("""
    Usage: MandelbrotBench [options]
      --target <name>      seahorse | mini | misiurewicz (default: all in matrix mode)
      --kernel <name>      double | softdouble | float128 (default: all in matrix mode)
      --iters <list>       Comma-separated max-iterations list, e.g. 256,1024,4096 (default: 1024)
      --frames <N>         Total dive length when not using --sample (default: 200)
      --sample <list>      Render only these dive-frame indices, e.g. 30,60,150.
                           Viewport advances silently between samples.
      --width <N>          Canvas width (default: 800)
      --height <N>         Canvas height (default: 600)
      --output <dir>       Output directory (default: bench-results)
      --matrix             Run full sweep: all targets × all kernels × all iter caps
      --save-images        Write PNG for each frame
      --image-every <N>    With --save-images, save every Nth frame (default: 1)
    """)
}

// MARK: - Run a single bench cell

func runCell(target: PlaybackTarget, kernel: KernelName, iters: UInt32,
             opts: Options, runDir: URL) {
    let engine: any MandelbrotEngine
    switch kernel {
    case .double:          engine = CPUEngine(kernel: DoubleStripKernel())
    case .softDouble:      engine = CPUEngine(kernel: SoftDoubleStripKernel())
    case .softDoubleMetal: engine = MetalSoftDouble64Engine()
    case .float128:        engine = CPUEngine(kernel: Float128StripKernel())
    case .float128Metal:   engine = MetalFloat128Engine()
    case .float128LimbMetal: engine = MetalLimbFloat128Engine()
    case .float128UnpackedMetal: engine = MetalFloat128UnpackedEngine()
    case .float128Hybrid:        engine = HybridFloat128Engine()
    }

    let label = "\(slug(target.name))-\(kernel.rawValue)-\(iters)"

    // Resolve which dive-frame indices to render.
    let sampleIndices: [Int]
    if let s = opts.sample, !s.isEmpty {
        sampleIndices = s.sorted()
    } else {
        sampleIndices = Array(0..<opts.frames)
    }
    let lastFrame = sampleIndices.last ?? 0
    let sampleSet = Set(sampleIndices)
    print("==> \(label) (samples: \(sampleIndices), \(opts.width)x\(opts.height))")

    var viewport = Viewport.defaultView(width: opts.width, height: opts.height)
    var frames: [FrameRecord] = []
    frames.reserveCapacity(sampleIndices.count)

    let imagesDir = runDir.appendingPathComponent("images/\(label)")
    if opts.saveImages {
        try? FileManager.default.createDirectory(at: imagesDir, withIntermediateDirectories: true)
    }
    let colorizer = ElectricBlueColorizer()

    for i in 0...lastFrame {
        if sampleSet.contains(i) {
            let t0 = CFAbsoluteTimeGetCurrent()
            let field = engine.render(viewport: viewport, width: opts.width, height: opts.height,
                                      maxIterations: iters)
            let renderMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            let checksum = sha256(field)

            if opts.saveImages {
                let img = colorizer.render(field: field, maxIterations: iters)
                let url = imagesDir.appendingPathComponent(String(format: "frame-%04d.png", i))
                writePNG(img, to: url)
            }

            frames.append(FrameRecord(
                i: i,
                centerX: viewport.centerX.asDouble,
                centerY: viewport.centerY.asDouble,
                pixelSize: viewport.pixelSize.asDouble,
                renderMs: renderMs,
                checksum: checksum
            ))
        }
        if i < lastFrame {
            viewport = viewport.zoomed(
                by: Float128(target.zoomRate),
                aroundX: Float128(target.centerX),
                aroundY: Float128(target.centerY)
            )
        }
    }

    let summary = Summary.compute(frames)
    let formatter = ISO8601DateFormatter()
    let runConfig = RunConfig(
        target: target.name,
        kernel: kernel,
        maxIterations: iters,
        width: opts.width, height: opts.height,
        frames: sampleIndices.count,
        sampleIndices: sampleIndices,
        zoomRate: target.zoomRate,
        startedAt: formatter.string(from: Date())
    )
    let bench = BenchRun(
        machine: MachineInfo.current,
        build: BuildInfo.current,
        run: runConfig,
        frames: frames,
        summary: summary
    )

    // Write JSON
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let jsonURL = runDir.appendingPathComponent("\(label).json")
    if let data = try? encoder.encode(bench) {
        try? data.write(to: jsonURL)
    }

    // Append CSV row
    appendCSVRow(bench: bench, to: runDir.appendingPathComponent("summary.csv"))

    print(String(format: "    %d frames in %.1f ms · median %.2f ms · p95 %.2f ms · %.1f fps",
                 frames.count, summary.totalMs, summary.medianFrameMs,
                 summary.p95FrameMs, summary.fps))
}

// MARK: - SHA256 of IterationField

func sha256(_ field: IterationField) -> String {
    var hasher = SHA256()
    field.withBufferPointer { buf in
        let raw = UnsafeRawBufferPointer(buf)
        hasher.update(bufferPointer: raw)
    }
    let digest = hasher.finalize()
    return digest.compactMap { String(format: "%02x", $0) }.joined()
}

// MARK: - PNG output

func writePNG(_ image: CGImage, to url: URL) {
    guard let dest = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.png.identifier as CFString, 1, nil
    ) else { return }
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

// MARK: - CSV

let csvHeader = "soc,cores,memoryGB,kernel,target,maxIters,width,height,frames,zoomRate,totalMs,medianMs,p95Ms,p99Ms,minMs,maxMs,fps,firstChecksum,lastChecksum,buildConfig,swiftVersion\n"

func appendCSVRow(bench: BenchRun, to url: URL) {
    let fm = FileManager.default
    if !fm.fileExists(atPath: url.path) {
        try? csvHeader.write(to: url, atomically: true, encoding: .utf8)
    }
    let s = bench.summary
    let r = bench.run
    let m = bench.machine
    let b = bench.build
    let first = bench.frames.first?.checksum ?? ""
    let last = bench.frames.last?.checksum ?? ""
    let row = "\(csv(m.soc)),\(m.cores),\(m.memoryGB),\(r.kernel.rawValue),\(csv(r.target)),\(r.maxIterations),\(r.width),\(r.height),\(r.frames),\(r.zoomRate),\(fmt(s.totalMs)),\(fmt(s.medianFrameMs)),\(fmt(s.p95FrameMs)),\(fmt(s.p99FrameMs)),\(fmt(s.minFrameMs)),\(fmt(s.maxFrameMs)),\(fmt(s.fps)),\(first),\(last),\(b.config),\(b.swiftVersion)\n"
    if let handle = try? FileHandle(forWritingTo: url) {
        handle.seekToEndOfFile()
        handle.write(Data(row.utf8))
        try? handle.close()
    }
}

func csv(_ s: String) -> String {
    if s.contains(",") || s.contains("\"") { return "\"\(s.replacingOccurrences(of: "\"", with: "\"\""))\"" }
    return s
}

func fmt(_ d: Double) -> String { String(format: "%.3f", d) }

func slug(_ name: String) -> String {
    name.lowercased().replacingOccurrences(of: " ", with: "-")
}

// MARK: - Target name resolution

func resolveTarget(_ name: String) -> PlaybackTarget? {
    let lower = name.lowercased()
    if lower.contains("sea") { return .seahorseValley }
    if lower.contains("mini") { return .miniMandelbrot }
    if lower.contains("mis") { return .misiurewicz }
    return nil
}

// MARK: - Main

let opts = Options.parse(CommandLine.arguments)

// Make a timestamped subdirectory under outputDir for this invocation.
let stamp = ISO8601DateFormatter().string(from: Date())
    .replacingOccurrences(of: ":", with: "-")
let runDir = URL(fileURLWithPath: opts.outputDir).appendingPathComponent(stamp)
try? FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)
print("Output: \(runDir.path)")

// Decide what cells to run.
let targets: [PlaybackTarget]
if opts.matrix {
    targets = PlaybackTarget.allPresets
} else if let name = opts.target, let t = resolveTarget(name) {
    targets = [t]
} else {
    targets = [.seahorseValley]
}

let kernels: [KernelName]
if opts.matrix {
    kernels = KernelName.allCases
} else if let k = opts.kernel {
    kernels = [k]
} else {
    kernels = [.double]
}

let itersList = opts.iters

print("Machine: \(MachineInfo.current.soc) · \(MachineInfo.current.cores) cores · \(MachineInfo.current.memoryGB) GB")
print("Build:   Swift \(BuildInfo.current.swiftVersion) · \(BuildInfo.current.config)")

// --profile: sweep threadgroup configs for the Float128 Metal kernel at the
// first --sample frame of the chosen --target, then exit.
if opts.profile {
    let target = (opts.target.flatMap(resolveTarget)) ?? .seahorseValley
    let frame = opts.sample?.first ?? 30
    var vp = Viewport.defaultView(width: opts.width, height: opts.height)
    for _ in 0..<frame {
        vp = vp.zoomed(by: Float128(target.zoomRate),
                       aroundX: Float128(target.centerX), aroundY: Float128(target.centerY))
    }
    print("Profiling \(target.name) frame \(frame), px=\(vp.pixelSize.asDouble)\n")
    print(profileFloat128(viewport: vp, width: opts.width, height: opts.height,
                          maxIterations: opts.iters.first ?? 1024))
    exit(0)
}
print("Matrix:  \(targets.count) targets × \(kernels.count) kernels × \(itersList.count) iter caps = \(targets.count * kernels.count * itersList.count) cells")
print("")

let overallStart = CFAbsoluteTimeGetCurrent()
for target in targets {
    for kernel in kernels {
        for iters in itersList {
            runCell(target: target, kernel: kernel, iters: iters, opts: opts, runDir: runDir)
        }
    }
}
let overallMs = (CFAbsoluteTimeGetCurrent() - overallStart) * 1000.0
print("")
print(String(format: "Done. Total wall time: %.1f s", overallMs / 1000.0))
print("Summary: \(runDir.appendingPathComponent("summary.csv").path)")
