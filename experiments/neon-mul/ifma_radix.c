// Radix-parameterized IFMA bignum multiply — confirms the algorithm maps to
// Apple's MUL53 (native 53-bit boundary), not just the 52-bit IFMA emulation.
// MUL53 = scalar multiply-and-extract-at-53-bit-boundary, IN the integer pipes
// (not AMX). lo/hi at radix R are exactly what MUL53lo/MUL53hi would produce.
// Validate bit-exact across R = 48..53 (radix-robust); R=53 is the MUL53 target.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef unsigned __int128 u128;
typedef struct { uint64_t w[4]; } u256;

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

// MUL53-style ops at radix R (a,b < 2^R). On real silicon each is one MUL53.
static u256 mul_ifma_R(u128 A, u128 B, int R){
    const uint64_t MASK=(R<64)?((1ull<<R)-1):~0ull;
    int nd=(128+R-1)/R, nc=2*nd+1;
    uint64_t a[6],b[6],col[12]={0};
    for(int i=0;i<nd;i++){ a[i]=(uint64_t)(A>>(R*i))&MASK; b[i]=(uint64_t)(B>>(R*i))&MASK; }
    for(int i=0;i<nd;i++)for(int j=0;j<nd;j++){
        u128 p=(u128)a[i]*b[j];
        col[i+j]   += (uint64_t)(p & MASK);   // MUL53lo
        col[i+j+1] += (uint64_t)(p >> R);     // MUL53hi
    }
    uint64_t carry=0, dig[12];
    for(int c=0;c<nc;c++){ uint64_t t=col[c]+carry; dig[c]=t&MASK; carry=t>>R; }
    u128 hi=0, lo=0;
    for(int c=nc-1;c>=0;c--){ u128 cr=lo>>(128-R); hi=(hi<<R)|cr; lo=(lo<<R)|dig[c]; }
    u256 r={{(uint64_t)lo,(uint64_t)(lo>>64),(uint64_t)hi,(uint64_t)(hi>>64)}};
    return r;
}

static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }
static inline u128 rnd(void){ return ((u128)lcg()<<64)|lcg(); }

int main(void){
    for(int R=48;R<=53;R++){
        long fails=0, n=2000000;
        for(long t=0;t<n;t++){ u128 a=rnd(),b=rnd();
            if(memcmp(mul_scalar(a,b).w, mul_ifma_R(a,b,R).w,32)) fails++; }
        int nd=(128+R-1)/R;
        printf("radix 2^%d : %d limbs, %d products -> %s%s\n",
               R, nd, nd*nd, fails?"FAIL":"BIT-EXACT", R==53?"   <- MUL53 native":"");
    }
    return 0;
}
