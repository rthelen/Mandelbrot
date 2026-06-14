import Foundation

struct BuildInfo: Codable {
    let swiftVersion: String
    let config: String  // "release" or "debug"

    static var current: BuildInfo {
        #if DEBUG
        let cfg = "debug"
        #else
        let cfg = "release"
        #endif
        // Swift compile-time version detection. Bump when we cross a version.
        #if swift(>=6.2)
        let sv = "6.2+"
        #elseif swift(>=6.0)
        let sv = "6.0"
        #else
        let sv = "<6.0"
        #endif
        return BuildInfo(swiftVersion: sv, config: cfg)
    }
}

enum KernelName: String, Codable, CaseIterable {
    case double = "double"
    case doubleSIMD = "double-simd"
    case softDouble = "softdouble"
    case softDoubleMetal = "softdouble-metal"
    case float128 = "float128"
    case float128Metal = "float128-metal"
    case float128LimbMetal = "float128-lf-metal"
    case float128UnpackedMetal = "float128-u-metal"
    case float128Hybrid = "float128-hybrid"
}

struct RunConfig: Codable {
    let target: String
    let kernel: KernelName
    let maxIterations: UInt32
    let width: Int
    let height: Int
    let frames: Int                  // number actually rendered
    let sampleIndices: [Int]         // dive frame indices that were rendered
    let zoomRate: Double
    let startedAt: String  // ISO 8601
}

struct FrameRecord: Codable {
    let i: Int
    let centerX: Double
    let centerY: Double
    let pixelSize: Double
    let renderMs: Double
    let checksum: String  // SHA-256 hex of IterationField buffer
}

struct Summary: Codable {
    let totalMs: Double
    let medianFrameMs: Double
    let p95FrameMs: Double
    let p99FrameMs: Double
    let minFrameMs: Double
    let maxFrameMs: Double
    let fps: Double

    static func compute(_ frames: [FrameRecord]) -> Summary {
        let times = frames.map(\.renderMs).sorted()
        let total = times.reduce(0, +)
        func percentile(_ p: Double) -> Double {
            guard !times.isEmpty else { return 0 }
            let idx = min(times.count - 1, Int(Double(times.count - 1) * p))
            return times[idx]
        }
        return Summary(
            totalMs: total,
            medianFrameMs: percentile(0.50),
            p95FrameMs: percentile(0.95),
            p99FrameMs: percentile(0.99),
            minFrameMs: times.first ?? 0,
            maxFrameMs: times.last ?? 0,
            fps: total > 0 ? Double(frames.count) * 1000.0 / total : 0
        )
    }
}

struct BenchRun: Codable {
    let machine: MachineInfo
    let build: BuildInfo
    let run: RunConfig
    let frames: [FrameRecord]
    let summary: Summary
}
