import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MandelbrotCore

// IconGenerator <output.png>
// Renders a 1024x1024 PNG of the Mandelbrot set for use as the app icon.

guard CommandLine.arguments.count >= 2 else {
    FileHandle.standardError.write(Data("usage: IconGenerator <output.png>\n".utf8))
    exit(1)
}
let outputPath = CommandLine.arguments[1]

let size = 1024
let viewport = Viewport(
    centerX: -0.6,
    centerY: 0.0,
    pixelSize: Float128(2.8 / Double(size))
)
let engine = CPUEngine(kernel: DoubleStripKernel())
let field = engine.render(
    viewport: viewport,
    width: size,
    height: size,
    maxIterations: 2048
)
let colorizer = ElectricBlueColorizer(cycleLength: 96.0)
let image = colorizer.render(field: field, maxIterations: 2048)

let url = URL(fileURLWithPath: outputPath)
guard let dest = CGImageDestinationCreateWithURL(
    url as CFURL, UTType.png.identifier as CFString, 1, nil
) else {
    FileHandle.standardError.write(Data("could not create image destination\n".utf8))
    exit(1)
}
CGImageDestinationAddImage(dest, image, nil)
guard CGImageDestinationFinalize(dest) else {
    FileHandle.standardError.write(Data("could not finalize PNG\n".utf8))
    exit(1)
}

print("wrote \(outputPath)")
