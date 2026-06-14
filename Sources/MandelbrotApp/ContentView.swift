import SwiftUI
import MandelbrotCore

struct ContentView: View {
    @ObservedObject var viewModel: MandelbrotViewModel
    @State private var jumpFrameText: String = "60"

    var body: some View {
        VStack(spacing: 0) {
            controlBar
            jumpBar
            Divider()
            MandelbrotCanvas(viewModel: viewModel)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            statusBar
        }
    }

    private var jumpBar: some View {
        HStack(spacing: 8) {
            Text("Jump to dive frame:")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach([0, 30, 60, 100, 150, 200, 300, 400], id: \.self) { n in
                Button("\(n)") { viewModel.jumpToDiveFrame(n) }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            TextField("N", text: $jumpFrameText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 60)
                .onSubmit { jumpToTyped() }
            Button("Go") { jumpToTyped() }
                .controlSize(.small)
            Spacer()
            Text("current dive frame: \(viewModel.playbackFrameCount)")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color(NSColor.windowBackgroundColor).opacity(0.5))
    }

    private func jumpToTyped() {
        if let n = Int(jumpFrameText) { viewModel.jumpToDiveFrame(n) }
    }

    private var controlBar: some View {
        HStack(spacing: 12) {
            Picker("Kernel", selection: $viewModel.kernelChoice) {
                ForEach(KernelChoice.allCases) { k in
                    Text(k.rawValue).tag(k)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 200)

            Picker("Color", selection: $viewModel.colorizerIndex) {
                ForEach(viewModel.colorizers.indices, id: \.self) { i in
                    Text(viewModel.colorizers[i].name).tag(i)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 180)

            Picker("Iters", selection: $viewModel.maxIterations) {
                ForEach([UInt32(128), 256, 512, 1024, 2048, 4096, 8192], id: \.self) { n in
                    Text("\(n)").tag(n)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 140)

            Divider().frame(height: 22)

            Picker("Dive", selection: $viewModel.playbackTargetIndex) {
                ForEach(PlaybackTarget.allPresets.indices, id: \.self) { i in
                    Text(PlaybackTarget.allPresets[i].name).tag(i)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 180)
            .disabled(viewModel.isPlaying)

            Button(viewModel.isPlaying ? "■ Stop" : "▶ Play") {
                viewModel.togglePlayback()
            }
            .keyboardShortcut("p", modifiers: .command)

            Spacer()

            Button("Reset View") { viewModel.resetView() }
        }
        .padding(8)
    }

    private var statusBar: some View {
        VStack(spacing: 2) {
            HStack(spacing: 12) {
                Text(viewportLabel)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                if let n = viewModel.mismatchCount {
                    Text(n == 0 ? "✓ HW=SW (0 bit-diffs)" : "✗ \(n) bit-diff\(n == 1 ? "" : "s")")
                        .font(.system(.caption, design: .monospaced).bold())
                        .foregroundStyle(n == 0 ? .green : .red)
                    // Sticky latch: once any frame diverges, keep showing it
                    // (with the frame it first tripped) until the dive restarts.
                    if viewModel.peakMismatchCount > 0 {
                        Text("⚠ tripped: peak \(viewModel.peakMismatchCount) @ frame \(viewModel.peakMismatchFrame ?? 0)")
                            .font(.system(.caption, design: .monospaced).bold())
                            .foregroundStyle(.red)
                    }
                }
                Spacer()
                if viewModel.isPlaying || viewModel.currentFPS > 0 {
                    Text(perfLabel)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(viewModel.isPlaying ? .primary : .secondary)
                } else {
                    Text("Drag: zoom · Click: zoom in · Shift/Opt-click: zoom out · ⌘-drag: pan")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if let split = viewModel.hybridSplit {
                HStack {
                    Text("⇄ hybrid split: \(split)")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.teal)
                    Spacer()
                }
            }
            if let detail = viewModel.divergenceDetail {
                HStack {
                    Text("⚠ \(detail)")
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                    Spacer()
                }
            }
            if let reason = viewModel.playbackStoppedReason {
                HStack {
                    Text("⏹ \(reason)")
                        .font(.caption)
                        .foregroundStyle(.orange)
                    Spacer()
                }
            }
        }
        .padding(6)
        .background(.thinMaterial)
    }

    private var viewportLabel: String {
        let cx = viewModel.viewport.centerX.asDouble
        let cy = viewModel.viewport.centerY.asDouble
        let px = viewModel.viewport.pixelSize.asDouble
        return String(format: "center: (%+.10f, %+.10f)  px: %.3e", cx, cy, px)
    }

    private var perfLabel: String {
        let frame = viewModel.playbackFrameCount
        let fps = viewModel.currentFPS
        let ms = viewModel.lastFrameMs
        if viewModel.isPlaying {
            return String(format: "▶ frame %d · %.1f fps · %.1f ms/frame", frame, fps, ms)
        } else {
            return String(format: "last render: %.1f ms", ms)
        }
    }
}
