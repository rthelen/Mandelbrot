# NEON multiply & Apple MUL53 experiments

Standalone microbenchmarks exploring fast 128-bit / bignum multiply on Apple
Silicon — the multiply primitive behind the software-FP Mandelbrot kernels and
the true-north Pi engine. Every kernel is validated **bit-exact** against scalar
`__uint128_t` (deterministic LCG, no tolerance) before any timing.

```sh
make           # build everything
./fpipes       # start here — fully portable
```

## Two groups

**Portable (run on any Apple Silicon / arm64):**

| probe | what it measures |
|-------|------------------|
| `fpipes` | **NEON/SIMD pipe width** — FMLA/FADD (FP) and MUL.4s (integer-multiply) ops/cycle, with the **clock auto-measured** so ops/cycle is correct on any core. Run this on M1/M2/M3/M4/M5 to chart the pipe-count trend. |
| `mul128` | 2-wide NEON 32-bit-radix 128×128 multiply vs scalar `__uint128_t`. |
| `ifma52` | IFMA-52 (AVX-512-IFMA-style) bignum multiply, emulated + bit-exact. |
| `ifma_radix` | same, radix-parameterized (R=48..53) — shows the algorithm maps to MUL53's 53-bit boundary. |
| `fpfma` | FP64-FMA bignum multiply (borrow the FP mantissa multiplier), width sweep 128/256/512-bit. |

**Require Apple-private MUL53** (custom opcode `0x00200000`–`0x002007FF`; a 2-wide
NEON 53-bit multiply-extract used by JavaScriptCore):

| probe | what it does |
|-------|--------------|
| `mul53probe` | **Run this first.** SIGILL-guarded — detects whether MUL53 is present/enabled on this core, then validates it bit-exact. Exits 2 if absent. |
| `mul53bench` | MUL53 raw throughput vs the scalar multiplier. |
| `mul53mul` | full 128×128 multiply built on MUL53, end-to-end vs `__uint128_t`. |
| `dualengine` | **NEON MUL53 ∥ scalar integer-multiply** co-issue: proves the two datapaths run concurrently (free extra multiply throughput). |

> ⚠️ **MUL53 caveat:** it's undocumented/private and was found on M1 (Firestorm).
> The `mul53bench` / `mul53mul` / `dualengine` probes emit the raw instruction
> encodings directly and **will fault (SIGILL) on a core that lacks MUL53**.
> Always run `./mul53probe` first; only run the others if it reports `PRESENT`.
> Encoding source: TrungNguyen1909, via asahilinux.org/docs/hw/cpu/apple-instructions.

## Headline findings (M1 Ultra, 3.22 GHz)

- **Pipe width:** 4 NEON pipes; integer-multiply width == FP width (both 4.0/cyc).
- **MUL53 is real & bit-exact from userspace**, sustains ~8 lane-products/cyc
  (~4 instr/cyc × 2 lanes) — ~2.8× the scalar multiplier per product (radix-adj).
- **But the full 128×128 MUL53 multiply loses ~4× to scalar** — 53-bit radix (9
  products vs 4) + destructive/no-MAC accumulation; 128 bits is below the crossover.
- **Dual-engine wins:** MUL53 (NEON pipes) ∥ `__uint128_t` (scalar int-mul pipes)
  co-issue with ~2.05× concurrency → **+30% multiply throughput, free**, from the
  otherwise-idle scalar multiplier.

The FP-multiplier-for-bignum and dual-engine angles are the levers for Pi; all of
it (private-ISA, bit-exact, pipe-saturating, self-checking) is also a CPU
silicon-validation workload.
