// SME outer-product bignum multiply — ALGORITHM, emulated bit-exact on M1.
//
// SME (M4-only) computes an outer product into a 2D tile: one MOPA does
// ZA[i][j] += a[i]*b[j] for the whole grid. Schoolbook multiply IS that outer
// product (partial products a_i*c_j), so a bignum multiply = MOPA fill + sum the
// anti-diagonals (same i+j = same output digit) + carry. This file does exactly
// that in plain C so the algorithm is validated on M1; mopa_tile() is the step
// that becomes a real SME MOPA on M4.
//
// Limbs are 16-bit (matches SME2's INT16->INT64 widening MOPA; products < 2^32,
// 64-bit tile accumulators have enormous headroom). With SVL=512 a 32x32 grid =
// one full tile = a 512-bit x 512-bit multiply (the natural SME unit). 128-bit
// (8x8) under-fills it — SME is a Pi-scale tool.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef unsigned __int128 u128;
typedef struct { uint64_t w[4]; } u256;
#define MAXN 32

// trusted reference for the 128-bit anchor
static u256 mul_u128(u128 A,u128 B){
    uint64_t a0=(uint64_t)A,a1=(uint64_t)(A>>64),b0=(uint64_t)B,b1=(uint64_t)(B>>64);
    u128 p00=(u128)a0*b0,p01=(u128)a0*b1,p10=(u128)a1*b0,p11=(u128)a1*b1;
    u128 lo=p00,hi=p11; uint64_t tl=(uint64_t)p01,th=(uint64_t)(p01>>64);
    u128 t=lo+((u128)tl<<64); hi+=(u128)th+(t<lo); lo=t;
    tl=(uint64_t)p10; th=(uint64_t)(p10>>64);
    t=lo+((u128)tl<<64); hi+=(u128)th+(t<lo); lo=t;
    u256 r={{(uint64_t)lo,(uint64_t)(lo>>64),(uint64_t)hi,(uint64_t)(hi>>64)}};
    return r;
}

// THE SME STEP: outer product into a tile.  tile[i][j] = a[i]*b[j].  On M4 this
// is a MOPA accumulating into ZA (here a single product; accumulation is += ).
static void mopa_tile(const uint16_t *a,const uint16_t *b,int N,uint64_t tile[MAXN][MAXN]){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            tile[i][j] = (uint64_t)a[i]*(uint64_t)b[j];   // <- becomes SME UMOPA
}
// Reduce: sum each anti-diagonal (i+j=d), then carry-propagate 16-bit digits.
static void reduce_tile(const uint64_t tile[MAXN][MAXN],int N,uint16_t *out /*2N*/){
    uint64_t col[2*MAXN]={0};
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) col[i+j]+=tile[i][j];
    uint64_t c=0;
    for(int d=0;d<2*N;d++){ uint64_t t=col[d]+c; out[d]=(uint16_t)(t&0xFFFF); c=t>>16; }
}
static void sme_mul(const uint16_t *a,const uint16_t *b,int N,uint16_t *out){
    static uint64_t tile[MAXN][MAXN];
    mopa_tile(a,b,N,tile);
    reduce_tile(tile,N,out);
}

// straightforward 16-bit-limb schoolbook reference (independent cross-check)
static void ref_mul(const uint16_t *a,const uint16_t *b,int N,uint16_t *out){
    uint64_t col[2*MAXN]={0};
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) col[i+j]+=(uint64_t)a[i]*b[j];
    uint64_t c=0;
    for(int d=0;d<2*N;d++){ uint64_t t=col[d]+c; out[d]=(uint16_t)(t&0xFFFF); c=t>>16; }
}

static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }

static void rnd_limbs(uint16_t *x,int N){ for(int i=0;i<N;i++) x[i]=(uint16_t)lcg(); }
static u256 recon(const uint16_t *d){ // 16 digits (16-bit) -> 256-bit
    u128 lo=0,hi=0;
    for(int k=15;k>=0;k--){ u128 cr=lo>>112; hi=(hi<<16)|cr; lo=(lo<<16)|d[k]; }
    u256 r={{(uint64_t)lo,(uint64_t)(lo>>64),(uint64_t)hi,(uint64_t)(hi>>64)}}; return r;
}

static int test(int N,long n){
    long fails=0, anchorfails=0;
    for(long t=0;t<n;t++){
        uint16_t a[MAXN],b[MAXN],so[2*MAXN],ro[2*MAXN];
        rnd_limbs(a,N); rnd_limbs(b,N);
        sme_mul(a,b,N,so); ref_mul(a,b,N,ro);
        if(memcmp(so,ro,sizeof(uint16_t)*2*N)) fails++;
        if(N==8){ // anchor to __uint128_t
            u128 A=0,B=0; for(int k=7;k>=0;k--){A=(A<<16)|a[k];B=(B<<16)|b[k];}
            if(memcmp(recon(so).w,mul_u128(A,B).w,32)) anchorfails++;
        }
    }
    int bits=N*16;
    printf("  %4d-bit (%2dx%2d outer product = %s tile): %s",
           bits,N,N, N==32?"FULL 512b":"partial", fails?"FAIL vs schoolbook":"bit-exact");
    if(N==8) printf(", %s vs __uint128_t", anchorfails?"FAIL":"bit-exact");
    printf("\n");
    return (int)(fails+anchorfails);
}

int main(void){
    printf("SME outer-product bignum multiply (emulated, 16-bit limbs):\n");
    int bad=0;
    bad+=test(8, 1000000);    // 128-bit  (anchored to __uint128_t)
    bad+=test(16,500000);     // 256-bit
    bad+=test(32,200000);     // 512-bit  = one full 32x32 SME tile
    printf("%s\n", bad?"FAILED":"ALL BIT-EXACT — algorithm maps to SME MOPA + anti-diagonal reduce");
    return bad?1:0;
}
