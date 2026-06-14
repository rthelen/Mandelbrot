#include <alloca.h>

#include "knuth_div.h"

// The original source of the text below came from this website:
// https://skanthak.hier-im-netz.de/division.html

// A Swift Version of this algorithm (which is very readable):
// https://github.com/chipjarred/KnuthAlgorithmD/blob/master/Sources/KnuthAlgorithmD/KnuthAlgorithmD.swift

/* This divides an m-word dividend by an n-word divisor, giving an
m-n+1-word quotient and n-word remainder. The bignums are in arrays of
words. Here a "word" is 32 bits. This routine is designed for a 64-bit
machine which has a 64/64 division instruction. */

/* q[0], r[0], u[0], and v[0] contain the LEAST significant words.
(The sequence is in little-endian order).

This is a fairly precise implementation of Knuth's Algorithm D, for a
binary computer with base b = 2**32. The caller supplies:
   1. Space q for the quotient, m - n + 1 words (at least one).
   2. Space r for the remainder (optional), n words.
   3. The dividend u, m words, m >= 1.
   4. The divisor v, n words, n >= 2.
   q = u / v;
The most significant digit of the divisor, v[n-1], must be nonzero.  The
dividend u may have leading zeros; this just makes the algorithm take
longer and makes the quotient contain more leading zeros.  A value of
NULL may be given for the address of the remainder to signify that the
caller does not want the remainder.
   The program does not alter the input parameters u and v.
   The quotient and remainder returned may have leading zeros.  The
function itself returns a value of 0 for success and 1 for invalid
parameters (e.g., division by 0).
   For now, we must have m >= n.  Knuth's Algorithm D also requires
that the dividend be at least as long as the divisor.  (In his terms,
m >= 0 (unstated).  Therefore m+n >= n.) */

int divmnu(uint32_t q[], uint32_t r[],
     const uint32_t u[], const uint32_t v[],
     ssize_t m, ssize_t n) {

   const uint64_t b = 4294967296LL; // Number base (2**32).
   uint32_t *un, *vn;                         // Normalized form of u, v.
   uint64_t qhat;                   // Estimated quotient digit.
   uint64_t rhat;                   // A remainder.
   uint64_t p;                      // Product of two digits.
   int64_t t, k;
   ssize_t s, i, j;

   if (m < n || n <= 1 || v[n-1] == 0)
      return 1;                         // Return if invalid param.

   /* Normalize by shifting v left just enough so that its high-order
   bit is on, and shift u left the same amount. We may have to append a
   high-order digit on the dividend; we do that unconditionally. */

   s = __builtin_clz(v[n-1]);             // 0 <= s <= 31.
   vn = (uint32_t *)alloca(4*n);
   for (i = n - 1; i > 0; i--)
      vn[i] = (v[i] << s) | ((uint64_t)v[i-1] >> (32-s));
   vn[0] = v[0] << s;

   un = (uint32_t *)alloca(4*(m + 1));
   un[m] = (uint64_t)u[m-1] >> (32-s);
   for (i = m - 1; i > 0; i--)
      un[i] = (u[i] << s) | ((uint64_t)u[i-1] >> (32-s));
   un[0] = u[0] << s;

   for (j = m - n; j >= 0; j--) {       // Main loop.
      // Compute estimate qhat of q[j].
      qhat = (un[j+n]*b + un[j+n-1])/vn[n-1];
#ifdef OPTIMIZE
      rhat = (un[j+n]*b + un[j+n-1])%vn[n-1];
#else // ORIGINAL
      rhat = (un[j+n]*b + un[j+n-1]) - qhat*vn[n-1];
#endif
again:
      if (qhat >= b ||
#ifdef OPTIMIZE
          (uint32_t)qhat*(uint64_t)vn[n-2] > b*rhat + un[j+n-2]) {
#else // ORIGINAL
         qhat*vn[n-2] > b*rhat + un[j+n-2]) {
#endif
        qhat = qhat - 1;
        rhat = rhat + vn[n-1];
        if (rhat < b) goto again;
      }

      // Multiply and subtract.
      k = 0;
      for (i = 0; i < n; i++) {
#ifdef OPTIMIZE
         p = (uint32_t)qhat*(uint64_t)vn[i];
#else // ORIGINAL
        p = qhat*vn[i];
#endif
         t = un[i+j] - k - (p & 0xFFFFFFFFLL);
         un[i+j] = t;
         k = (p >> 32) - (t >> 32);
      }
      t = un[j+n] - k;
      un[j+n] = t;

      q[j] = qhat;              // Store quotient digit.
      if (t < 0) {              // If we subtracted too
         q[j] = q[j] - 1;       // much, add back.
         k = 0;
         for (i = 0; i < n; i++) {
            t = (uint64_t)un[i+j] + vn[i] + k;
            un[i+j] = t;
            k = t >> 32;
         }
         un[j+n] = un[j+n] + k;
      }
   } // End j.
   // If the caller wants the remainder, unnormalize
   // it and pass it back.
   if (r != 0) {
      for (i = 0; i < n-1; i++)
         r[i] = (un[i] >> s) | ((uint64_t)un[i+1] << (32-s));
      r[n-1] = un[n-1] >> s;
   }
   return 0;
}
