import Foundation
import Darwin

struct MachineInfo: Codable {
    let soc: String
    let cores: Int
    let perfCores: Int?
    let efficiencyCores: Int?
    let memoryGB: Int
    let osVersion: String

    static var current: MachineInfo {
        MachineInfo(
            soc: sysctlString("machdep.cpu.brand_string") ?? "unknown",
            cores: Int(ProcessInfo.processInfo.activeProcessorCount),
            perfCores: sysctlInt32("hw.perflevel0.physicalcpu").map(Int.init),
            efficiencyCores: sysctlInt32("hw.perflevel1.physicalcpu").map(Int.init),
            memoryGB: Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)),
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString
        )
    }
}

private func sysctlString(_ name: String) -> String? {
    var size = 0
    if sysctlbyname(name, nil, &size, nil, 0) != 0 { return nil }
    if size == 0 { return nil }
    var buffer = [UInt8](repeating: 0, count: size)
    if sysctlbyname(name, &buffer, &size, nil, 0) != 0 { return nil }
    let bytes = buffer.prefix(while: { $0 != 0 })
    return String(bytes: bytes, encoding: .utf8)
}

private func sysctlInt32(_ name: String) -> Int32? {
    var value: Int32 = 0
    var size = MemoryLayout<Int32>.size
    if sysctlbyname(name, &value, &size, nil, 0) != 0 { return nil }
    return value
}
