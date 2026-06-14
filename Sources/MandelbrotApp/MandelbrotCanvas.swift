import SwiftUI
import AppKit
import MandelbrotCore

/// NSView that displays the rendered image and routes mouse events back to the view model.
final class MandelbrotNSView: NSView {
    var image: CGImage? { didSet { needsDisplay = true } }
    var rubberBand: NSRect? { didSet { needsDisplay = true } }

    var onClickAtPoint: ((NSPoint, NSEvent.ModifierFlags) -> Void)?
    var onZoomToRect: ((NSRect) -> Void)?
    var onPan: ((CGFloat, CGFloat) -> Void)?
    var onSizeChanged: ((NSSize) -> Void)?

    private var mouseDownPoint: NSPoint?
    private var lastDragPoint: NSPoint?
    private var didDrag = false
    private var isPanning = false

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func layout() {
        super.layout()
        onSizeChanged?(bounds.size)
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        ctx.setFillColor(NSColor.black.cgColor)
        ctx.fill(bounds)
        if let img = image {
            ctx.interpolationQuality = .none
            // In a flipped view, CG draws upside-down without a transform. Flip vertically.
            ctx.saveGState()
            ctx.translateBy(x: 0, y: bounds.height)
            ctx.scaleBy(x: 1, y: -1)
            ctx.draw(img, in: bounds)
            ctx.restoreGState()
        }
        if let rb = rubberBand {
            ctx.setStrokeColor(NSColor.white.cgColor)
            ctx.setLineDash(phase: 0, lengths: [4, 3])
            ctx.setLineWidth(1)
            ctx.stroke(rb)
        }
    }

    override func mouseDown(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        mouseDownPoint = p
        lastDragPoint = p
        didDrag = false
        isPanning = event.modifierFlags.contains(.command)
    }

    override func mouseDragged(with event: NSEvent) {
        guard let start = mouseDownPoint, let last = lastDragPoint else { return }
        let p = convert(event.locationInWindow, from: nil)
        if hypot(p.x - start.x, p.y - start.y) > 3 { didDrag = true }
        if isPanning {
            onPan?(p.x - last.x, p.y - last.y)
            lastDragPoint = p
        } else if didDrag {
            rubberBand = NSRect(
                x: min(start.x, p.x), y: min(start.y, p.y),
                width: abs(p.x - start.x), height: abs(p.y - start.y)
            )
        }
    }

    override func mouseUp(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        defer {
            mouseDownPoint = nil
            lastDragPoint = nil
            didDrag = false
            isPanning = false
            rubberBand = nil
        }
        if isPanning { return }
        if didDrag, let rb = rubberBand {
            onZoomToRect?(rb)
        } else {
            onClickAtPoint?(p, event.modifierFlags)
        }
    }
}

struct MandelbrotCanvas: NSViewRepresentable {
    @ObservedObject var viewModel: MandelbrotViewModel

    func makeNSView(context: Context) -> MandelbrotNSView {
        let v = MandelbrotNSView()
        wire(v)
        return v
    }

    func updateNSView(_ nsView: MandelbrotNSView, context: Context) {
        nsView.image = viewModel.image
        // Rebind callbacks so they always see the current view model.
        wire(nsView)
    }

    private func wire(_ v: MandelbrotNSView) {
        let vm = viewModel
        v.onClickAtPoint = { point, mods in
            let x = Int(point.x); let y = Int(point.y)
            if mods.contains(.shift) || mods.contains(.option) {
                vm.zoomOut(atPixelX: x, pixelY: y)
            } else {
                vm.zoomIn(atPixelX: x, pixelY: y)
            }
        }
        v.onZoomToRect = { rect in
            vm.zoomToRect(
                originX: Int(rect.minX), originY: Int(rect.minY),
                width: Int(rect.width), height: Int(rect.height)
            )
        }
        v.onPan = { dx, dy in
            vm.pan(byPixelsX: Int(dx), pixelsY: Int(dy))
        }
        v.onSizeChanged = { size in
            vm.setCanvasSize(width: Int(size.width), height: Int(size.height))
        }
    }
}
