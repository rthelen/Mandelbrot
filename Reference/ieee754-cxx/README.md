# IEEE 754 Reference Implementation

Verbatim copy of the C++ implementation from `~/Source/ieee754/` (rthelen's
custom IEEE 754 floating-point library). **Not built** by SwiftPM — these files
are the canonical reference for the Swift port in
`Sources/MandelbrotCore/IEEE754_128.swift` and for the future Metal port.

When the C++ source updates, re-copy these files manually and mirror any
algorithmic changes into the Swift port.
