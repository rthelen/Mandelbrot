// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Mandelbrot",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "MandelbrotCore", targets: ["MandelbrotCore"]),
        .executable(name: "MandelbrotApp", targets: ["MandelbrotApp"]),
        .executable(name: "IconGenerator", targets: ["IconGenerator"]),
        .executable(name: "MandelbrotBench", targets: ["MandelbrotBench"]),
    ],
    targets: [
        .target(
            name: "CMandelbrot",
            path: "Sources/CMandelbrot"
        ),
        .target(
            name: "MandelbrotCore",
            dependencies: ["CMandelbrot"],
            path: "Sources/MandelbrotCore",
            linkerSettings: [.linkedFramework("Metal")]
        ),
        .executableTarget(
            name: "MandelbrotApp",
            dependencies: ["MandelbrotCore"],
            path: "Sources/MandelbrotApp"
        ),
        .executableTarget(
            name: "IconGenerator",
            dependencies: ["MandelbrotCore"],
            path: "Sources/IconGenerator"
        ),
        .executableTarget(
            name: "MandelbrotBench",
            dependencies: ["MandelbrotCore"],
            path: "Sources/MandelbrotBench"
        ),
        .testTarget(
            name: "MandelbrotCoreTests",
            dependencies: ["MandelbrotCore"],
            path: "Tests/MandelbrotCoreTests"
        ),
    ]
)
