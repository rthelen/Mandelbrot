#ifndef CMANDELBROT_H
#define CMANDELBROT_H

#include <stdint.h>

// Software binary128 Mandelbrot iteration in C (using __uint128_t so clang emits
// MUL/UMULH + ADCS chains), bit-exact to the Swift Float128 kernel. The point c
// is passed as the (lo, hi) 64-bit halves of its packed binary128 bits.
//
// Returns the iteration count, or 0xFFFFFFFF for in-set. On escape, writes the
// packed binary128 bits of |z|^2 (lo, hi) so the caller can compute smoothing.
uint32_t cf128_mandelbrot_pixel(uint64_t cx_lo, uint64_t cx_hi,
                                uint64_t cy_lo, uint64_t cy_hi,
                                uint32_t max_iter,
                                uint64_t *magsq_lo, uint64_t *magsq_hi);

#endif
