import SwiftUI
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // When launched as a bare SwiftPM binary (no .app bundle), macOS treats
        // the process as a background agent: no Dock icon, window never comes to
        // front. Promote to a regular, activated app so the window appears.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@main
struct MandelbrotApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var viewModel = MandelbrotViewModel()

    var body: some Scene {
        WindowGroup("Mandelbrot") {
            ContentView(viewModel: viewModel)
                .frame(minWidth: 600, minHeight: 400)
        }
        .windowResizability(.contentSize)
        .commands {
            // Replace File > New Window — single-window app.
            CommandGroup(replacing: .newItem) { }

            CommandMenu("View") {
                Button("Reset View") { viewModel.resetView() }
                    .keyboardShortcut("r", modifiers: .command)
            }
        }
    }
}
