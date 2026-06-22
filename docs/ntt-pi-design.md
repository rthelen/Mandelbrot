# NTT-multi-prime bignum multiply for 8-billion-digit Pi — design sketch

Goal: the big-integer multiply that powers a Chudnovsky/binary-splitting Pi run
to **8×10⁹ decimal digits**, on Apple M-series, *exact* (no FP precision wall),
single-level (no Schönhage–Strassen nesting), with the modular-multiply hot loop
served by the kernels we've built (scalar `mul`+`umulh` / MUL53 / SME).

---

## 0. Size budget (decimal!)

- 8×10⁹ decimal digits × log₂10 ≈ **2.66×10¹⁰ bits ≈ 2³⁴·⁶ ≈ 3.32 GB** per number.
- Working precision in the Pi run is a bit more (guard digits); design the multiply
  to handle operands up to ~N = 2.66×10¹⁰ bits, product 2N.
- ⚠️ Units trap: "8B *32-bit* digits" would be 2.56×10¹¹ bits ≈ 77B decimal — 10×
  bigger. Every parameter below flows from the bit count; get it wrong and you
  size transforms 10× too big. **Write N_bits at the top of the implementation.**

## 1. Parameter chain   bits → coeffs → length → primes

Pack the number into `b`-bit coefficients. Then:

- coefficients per operand `M = N_bits / b`
- linear-convolution length `2M-1` → **transform length `L = next_pow2(2M)`**
- pre-carry coefficient bound `c_k = Σ_{i+j=k} aᵢcⱼ < M · 2^{2b}`
- need the CRT prime product `P = ∏ pᵢ > max c_k`

Tradeoff (larger `b` ⇒ shorter transform but bigger `c_k` ⇒ more primes):

| b  | M=N/b   | L     | c_k <  | 64-bit primes | mem/array (L×8B) |
|----|---------|-------|--------|---------------|------------------|
| 16 | 1.66e9  | 2³²   | 2⁶³    | 2             | 32 GB            |
| 32 | 8.3e8   | 2³¹   | 2⁹⁴    | 2             | 16 GB            |
| 52 | 5.1e8   | 2³⁰   | 2¹³³   | 3             | 8 GB             |

Transform work ∝ (#primes × L): b=32 → 2·2³¹ = 2³² (the minimum here).

**Baseline choice: `b = 32`, `L = 2³¹`, 2 primes.** Balanced work, 16 GB/array
(feasible streaming on a 64–192 GB Mac Studio, primes processed sequentially).

## 2. Primes

Need `p` prime, `p < 2⁶⁴` (word-sized residues), and `2³¹ | (p−1)` (so a
length-2³¹ root of unity exists: `ω = g^{(p−1)/L} mod p`).

- **Goldilocks `p = 2⁶⁴ − 2³² + 1`** — `p−1 = 2³²(2³²−1)`, so `2³² | p−1` ✓.
  Special form ⇒ **reduction is a few shifts/adds, no Montgomery** (big win).
- 2nd prime: any `c·2³² + 1` that's prime (quick search). With two ~2⁶⁴ primes,
  `P ≈ 2¹²⁸ ≫ c_k ≈ 2⁹⁴` — **~34 bits of margin**, robustly safe. Adding a 3rd
  prime is the trivial "more precision" dial if working precision grows.
- Alt that maps to MUL53: pick primes `< 2⁵²` so residues fit 53 bits and the
  modmul is the IFMA-52 / MUL53 path (what those instructions exist for in
  crypto). Needs ~3 primes; good if leaning on the NEON/MUL53 engine.

## 3. The transform — four-step (mandatory at L=2³¹)

A flat radix-2 FFT over a 16 GB array thrashes cache/DRAM. Use **four-step**:
factor `L = N₁·N₂` (e.g. `N₁=2¹⁶, N₂=2¹⁵`), view the array as an `N₁×N₂` matrix:

1. NTT each **column** (`N₂` transforms of size `N₁`)
2. multiply by **twiddles** `ω^{row·col}`  ← a modular multiply per element
3. **transpose** the matrix
4. NTT each **row** (`N₁` transforms of size `N₂`)

Sub-transforms (`N₁,N₂ ≈ 2¹⁵–2¹⁶`) fit in cache; the only non-local step is the
transpose (the real bandwidth bottleneck — block it). The batched small NTTs and
the twiddle step are where SME/SIMD pay off.

## 4. Modular multiply = the hot loop = our kernels

Each butterfly: `a' = a+b mod p`, `b' = (a−b)·ω mod p`. The `·ω` is a modmul.

For a generic 64-bit prime, **Montgomery reduction** turns `x·y mod p` into:
```
t = x*y                      // 64×64→128  ← scalar mul+umulh / SME / MUL53
m = (t_lo * p') mod 2^64     // 64×64→low64
u = (t + m*p) >> 64          // 64×64→128 + add
if (u >= p) u -= p
```
≈ **3 multiplies per modmul** → 3 invocations of the integer-multiply kernel.
For **Goldilocks**, the special form replaces Montgomery with a handful of
shifts/adds (no `p'`, no extra multiplies) — strongly preferred.

Kernel mapping:
- scalar `mul`+`umulh` (+ ADCS) — the portable per-butterfly path.
- **MUL53 / IFMA-52** — if primes `< 2⁵²`, 2-wide; pairs with the dual-engine.
- **SME** — the four-step's batched small NTTs and the twiddle/pointwise steps
  are matrix/batched-shaped → ~1 MOPA/cycle on M4 (Pi-scale regime).
- **dual-engine** (NEON ∥ scalar int-mul) — split the butterfly stream across
  both datapaths (+30% measured).
- **GPU** — whole primes / sub-FFT batches are independent → Metal.

## 5. Full multiply  C = A × B

```
for each prime p_i:                      # independent → parallel or streamed
    pack A,B into b-bit coeffs, zero-pad to L, reduce mod p_i
    NTT(A) ; NTT(B)            (mod p_i, four-step)
    pointwise: Ĉ_k = Â_k·B̂_k mod p_i
    INTT → c_k mod p_i         (uses ω⁻¹ and L⁻¹ mod p_i)
CRT-combine residue vectors → exact c_k ∈ [0, P)        # Garner, per coefficient
carry-propagate c_k (base 2^b) → product digits
```

CRT (2 primes, Garner): `c = r₁ + p₁·((r₂−r₁)·p₁⁻¹ mod p₂)`.

## 6. Where it sits in the Pi computation

Chudnovsky + **binary splitting** builds the series as nested integer
P/Q/T merges — each merge is big multiplies; the top merges are ~full precision.
Final `T/Q` and `√10005` use **Newton iteration**, also full-size multiplies.
So this NTT multiply is *the* workhorse, invoked O(log N) times at the big sizes;
total Pi time ≈ (a few ×) one full multiply.

## 7. Validation (bit-exact discipline, as always)

- Multiply correctness: random `A×B` vs **GMP** (or schoolbook for small) —
  bit-for-bit, swept across sizes up to L.
- Per-stage: NTT∘INTT = identity (mod p); single-prime result vs schoolbook
  before adding CRT.
- End-to-end oracle: **BBP formula** gives the n-th *hex* digit of Pi
  independently (no preceding digits) → spot-check the result at many positions.
  Plus compare against published digit strings and a second (b, prime-set) config.

## 8. Honest hard parts

1. **Memory & the transpose** — 16 GB/array; the four-step transpose is the
   bandwidth wall. Block it; consider out-of-core if RAM-bound.
2. **Parameter correctness** — `c_k < P` must hold with margin or you get
   *silent* wrong digits. Assert `M·2^{2b} < P` in code.
3. **Carry + sign** — c_k up to 2⁹⁴ across CRT; careful base-2^b carry.
4. **NTT-friendliness** — every prime needs `L | p−1`.

## 9. Why this over nested SS or FP-FFT

- vs **nested Schönhage–Strassen**: NTT-small-prime is **single-level** — the
  pointwise step is one modmul, not a recursive big multiply, so the nesting
  (and its parameter-per-level fragility + negacyclic sign traps) vanishes.
- vs **FP-FFT + 128-bit soft-float**: NTT is *exact* (no 2⁵³ wall, precision = a
  discrete prime count), and its inner loop is the integer multiply we've already
  tuned — instead of slow soft-float.

The convergence point of both true-norths: the NTT's butterfly modmul *is* the
kernel work; the four-step's matrix shape *is* SME's wheelhouse.
