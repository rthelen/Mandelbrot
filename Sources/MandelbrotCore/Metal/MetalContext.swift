import Foundation
import Metal

public enum MetalSetupError: Error, CustomStringConvertible {
    case noDevice
    case libraryCompile(String)
    case missingFunction(String)

    public var description: String {
        switch self {
        case .noDevice: return "No Metal device available"
        case .libraryCompile(let m): return "Metal library failed to compile: \(m)"
        case .missingFunction(let n): return "Metal function not found: \(n)"
        }
    }
}

/// Owns the Metal device, command queue, and the runtime-compiled shader
/// library, with a cache of compute pipeline states. Created lazily; `nil` when
/// the machine has no Metal device (so callers can skip rather than crash).
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    /// Shared context, compiled once. `nil` if no device or the shader source
    /// failed to compile (the error is logged on first failure).
    public static let shared: MetalContext? = {
        do { return try MetalContext() }
        catch { FileHandle.standardError.write(Data("MetalContext: \(error)\n".utf8)); return nil }
    }()

    public init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalSetupError.noDevice }
        guard let q = dev.makeCommandQueue() else { throw MetalSetupError.noDevice }
        self.device = dev
        self.queue = q
        do {
            self.library = try dev.makeLibrary(source: softDouble64MSLSource, options: nil)
        } catch {
            throw MetalSetupError.libraryCompile("\(error)")
        }
    }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelineCache[name] { return p }
        guard let fn = library.makeFunction(name: name) else {
            throw MetalSetupError.missingFunction(name)
        }
        let p = try device.makeComputePipelineState(function: fn)
        pipelineCache[name] = p
        return p
    }
}
