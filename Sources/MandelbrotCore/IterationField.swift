import Foundation

/// One pixel's result from the Mandelbrot iteration.
///
/// `iterations == .max` means the point did not escape within the iteration
/// budget (treat as "in the set" for coloring). `smooth` carries the fractional
/// escape time so colorizers can do continuous gradients; it is 0 for in-set
/// points and for kernels that don't emit smoothing.
public struct PixelResult: Sendable, Equatable {
    public var iterations: UInt32
    public var smooth: Float32

    @inlinable public init(iterations: UInt32, smooth: Float32) {
        self.iterations = iterations
        self.smooth = smooth
    }

    public static let inSet = PixelResult(iterations: .max, smooth: 0)
}

/// A 2D buffer of `PixelResult` values. The storage is raw memory so kernels
/// can write to disjoint regions concurrently without ARC contention.
public final class IterationField: @unchecked Sendable {
    public let width: Int
    public let height: Int

    @usableFromInline let storage: UnsafeMutablePointer<PixelResult>

    public init(width: Int, height: Int) {
        precondition(width > 0 && height > 0)
        self.width = width
        self.height = height
        self.storage = .allocate(capacity: width * height)
        self.storage.initialize(repeating: .inSet, count: width * height)
    }

    deinit {
        storage.deinitialize(count: width * height)
        storage.deallocate()
    }

    /// Writable pointer to a row, optionally offset to a column. Used by strip kernels.
    @inlinable
    public func pointer(row: Int, column: Int = 0) -> UnsafeMutablePointer<PixelResult> {
        storage.advanced(by: row * width + column)
    }

    @inlinable
    public subscript(row row: Int, column column: Int) -> PixelResult {
        get { storage[row * width + column] }
        set { storage[row * width + column] = newValue }
    }

    /// Read-only view over the entire buffer. For colorizers.
    public func withBufferPointer<R>(_ body: (UnsafeBufferPointer<PixelResult>) -> R) -> R {
        body(UnsafeBufferPointer(start: storage, count: width * height))
    }
}
