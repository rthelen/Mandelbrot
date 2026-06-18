// IFMA-52 bignum multiply, emulated.
//
// Apple's private 52-bit multiplier ~= x86 AVX-512 IFMA (VPMADD52LO/HI): the
// FP64 mantissa multiplier (53-bit) exposed as integer multiply-accumulate into
// 64-bit lanes. This emulates the two ops in scalar so we can PROVE a 52-bit-
// radix 128x128->256 schoolbook is bit-exact vs the __uint128_t reference, and
// count the ops the real instruction would execute. (No tolerance; LCG random.)
//
// The point: 52-bit limbs + 12 bits of accumulator headroom let columns hold up
// to ~4096 products with NO intermediate carry. The carry-save/split/ripple that
// drowns the 32-bit NEON kernel disappears -> one cheap normalize at the end.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef unsigned __int128 u128;
typedef struct { uint64_t w[4]; } u256;
#define MASK52 ((1ull<<52)-1)

// ---- the two IFMA ops, emulated (a,b < 2^52) ----
static inline uint64_t madd52lo(uint64_t acc, uint64_t a, uint64_t b){
    return acc + (uint64_t)(((u128)a*b) & MASK52);
}
static inline uint64_t madd52hi(uint64_t acc, uint64_t a, uint64_t b){
    return acc + (uint64_t)(((u128)a*b) >> 52);
}

// ---- scalar reference (the cf_mul product) ----
static inline u256 mul_scalar(u128 A, u128 B){
    uint64_t a0=(uint64_t)A,a1=(uint64_t)(A>>64),b0=(uint64_t)B,b1=(uint64_t)(B>>64);
    u128 p00=(u128)a0*b0,p01=(u128)a0*b1,p10=(u128)a1*b0,p11=(u128)a1*b1;
    u128 lo=p00,hi=p11; uint64_t tl=(uint64_t)p01,th=(uint64_t)(p01>>64);
    u128 t=lo+((u128)tl<<64); hi+=(u128)th+(t<lo); lo=t;
    tl=(uint64_t)p10; th=(uint64_t)(p10>>64);
    t=lo+((u128)tl<<64); hi+=(u128)th+(t<lo); lo=t;
    u256 r={{(uint64_t)lo,(uint64_t)(lo>>64),(uint64_t)hi,(uint64_t)(hi>>64)}};
    return r;
}

// ---- IFMA-52 multiply: 3 limbs x 3 limbs, deferred carry ----
static u256 mul_ifma52(u128 A, u128 B){
    uint64_t a[3]={ (uint64_t)A&MASK52, (uint64_t)(A>>52)&MASK52, (uint64_t)(A>>104)&MASK52 };
    uint64_t b[3]={ (uint64_t)B&MASK52, (uint64_t)(B>>52)&MASK52, (uint64_t)(B>>104)&MASK52 };
    uint64_t col[6]={0,0,0,0,0,0};
    // 9 products, each two fused MACs. No intermediate carry: each column sees
    // <=3 lo + <=2 hi terms < 5*2^52 << 2^64.
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++){
            col[i+j]   = madd52lo(col[i+j],   a[i], b[j]);
            col[i+j+1] = madd52hi(col[i+j+1], a[i], b[j]);
        }
    // one normalize pass: ripple 52-bit digits
    uint64_t carry=0, dig[6];
    for(int c=0;c<6;c++){ uint64_t t=col[c]+carry; dig[c]=t&MASK52; carry=t>>52; }
    // reassemble 256-bit from six 52-bit digits (Horner, high->low)
    u128 hi=0, lo=0;
    for(int c=5;c>=0;c--){ u128 cr=lo>>76; hi=(hi<<52)|cr; lo=(lo<<52)|dig[c]; }
    u256 r={{(uint64_t)lo,(uint64_t)(lo>>64),(uint64_t)hi,(uint64_t)(hi>>64)}};
    return r;
}

static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }
static inline u128 rnd(void){ return ((u128)lcg()<<64)|lcg(); }

int main(void){
    long fails=0, n=5000000;
    for(long t=0;t<n;t++){
        u128 a=rnd(), b=rnd();
        if(memcmp(mul_scalar(a,b).w, mul_ifma52(a,b).w, 32)){
            if(++fails<=3) printf("MISMATCH t=%ld\n",t);
        }
    }
    printf("IFMA-52: %ld products, %ld mismatches -> %s\n", n, fails, fails?"FAIL":"BIT-EXACT");
    printf("op count per 128x128: 9 products = 18 madd52, +6-limb normalize.\n");
    printf("  vs 32-bit NEON: 16 umull + ~30 usra + ~32 add/and.\n");
    return fails?1:0;
}
