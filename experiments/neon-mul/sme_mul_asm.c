// REAL working SME 128x128 bignum multiply — HAND-ASM (bypasses the Apple-clang
// cntd bug). 16-bit limbs, zero-padded into a pure 8x8 za64 outer product:
//   smstart; ld1h z0/z1; zero za; umopa za0.d,p0/m,p0/m,z0.h,z1.h; read 8 ZA
//   slices; smstop.  Then sum anti-diagonals + carry in normal C. Self-checks
//   vs __uint128_t. Safe on pre-SME cores (asm only runs when the gate passes,
//   and there's no clang SME prologue to fault at startup).
//
// Build (any Mac): clang -O3 -march=armv9-a+sme2+sme-i16i64 -o sme_mul_asm sme_mul_asm.c
// Run on M4+:      ./sme_mul_asm
//
// Instruction sequence transcribed from clang's OWN (correct) body codegen for
// the intrinsic version — only its prologue `cntd` was buggy, which raw asm avoids.
// Assumption A2 (confirmed by the self-check): the za64_u16 widening MOPA groups
// source lanes as consecutive 4s, so limb i at lane 4*i (others 0) => ZA[i][j]=a_i*b_j.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>
#include <mach/mach_time.h>

typedef unsigned __int128 u128;
typedef struct { uint64_t w[4]; } u256;

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

// 8x8 outer product a_i*b_j via SME, hand-asm. a32/b32: 32 u16 (limb i at 4*i,
// rest 0). out64: 64 u64 (row-major 8x8), out64[i*8+j] = ZA[i][j] = a_i*b_j.
__attribute__((noinline))
static void sme_outer8_asm(const uint16_t *a32, const uint16_t *b32, uint64_t *out64){
    __asm__ __volatile__(
        "smstart                               \n"
        "ptrue   p0.h                          \n"
        "ld1h    {z0.h}, p0/z, [%[a]]          \n"
        "ld1h    {z1.h}, p0/z, [%[b]]          \n"
        "zero    {za}                          \n"
        "umopa   za0.d, p0/m, p0/m, z0.h, z1.h \n"
        "ptrue   p0.d                          \n"
        "mov w12, #0\n mov z2.d, p0/m, za0h.d[w12, 0]\n str z2, [%[o], #0, mul vl]\n"
        "mov w12, #1\n mov z2.d, p0/m, za0h.d[w12, 0]\n str z2, [%[o], #1, mul vl]\n"
        "mov w12, #2\n mov z2.d, p0/m, za0h.d[w12, 0]\n str z2, [%[o], #2, mul vl]\n"
        "mov w12, #3\n mov z2.d, p0/m, za0h.d[w12, 0]\n str z2, [%[o], #3, mul vl]\n"
        "mov w12, #4\n mov z2.d, p0/m, za0h.d[w12, 0]\n str z2, [%[o], #4, mul vl]\n"
        "mov w12, #5\n mov z2.d, p0/m, za0h.d[w12, 0]\n str z2, [%[o], #5, mul vl]\n"
        "mov w12, #6\n mov z2.d, p0/m, za0h.d[w12, 0]\n str z2, [%[o], #6, mul vl]\n"
        "mov w12, #7\n mov z2.d, p0/m, za0h.d[w12, 0]\n str z2, [%[o], #7, mul vl]\n"
        "smstop                                \n"
        : : [a]"r"(a32), [b]"r"(b32), [o]"r"(out64)
        : "memory","w12","p0","z0","z1","z2",
          "v8","v9","v10","v11","v12","v13","v14","v15");  // smstart corrupts callee-saved NEON
}

static u256 reduce_tile(const uint64_t *tile){  // tile[i*8+j] = a_i*b_j
    uint64_t col[16]={0};
    for (int i=0;i<8;i++) for (int j=0;j<8;j++) col[i+j]+=tile[i*8+j];
    uint16_t d[16]; uint64_t c=0;
    for (int k=0;k<16;k++){ uint64_t t=col[k]+c; d[k]=(uint16_t)(t&0xFFFF); c=t>>16; }
    u128 lo=0,hi=0;
    for (int k=15;k>=0;k--){ u128 cr=lo>>112; hi=(hi<<16)|cr; lo=(lo<<16)|d[k]; }
    u256 r={{(uint64_t)lo,(uint64_t)(lo>>64),(uint64_t)hi,(uint64_t)(hi>>64)}};
    return r;
}
static void pad(u128 v, uint16_t out[32]){
    for (int k=0;k<32;k++) out[k]=0;
    for (int i=0;i<8;i++) out[4*i]=(uint16_t)(v>>(16*i));   // limb i at lane 4*i
}

static int feat(const char *k){ int v=0; size_t s=sizeof v; return sysctlbyname(k,&v,&s,0,0)==0 && v; }
static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }
static double now_ns(void){ static mach_timebase_info_data_t tb; if(!tb.denom)mach_timebase_info(&tb);
                            return (double)mach_absolute_time()*tb.numer/tb.denom; }

int main(void){
    if (!feat("hw.optional.arm.FEAT_SME") || !feat("hw.optional.arm.FEAT_SME_I16I64")){
        printf("SME / FEAT_SME_I16I64 not available — skipping (need M4+).\n");
        return 0;
    }
    printf("Running hand-asm SME za64_u16 outer-product 128x128 multiply...\n");
    long n=200000, fails=0;
    __attribute__((aligned(64))) uint64_t tile[64];
    for (long t=0;t<n;t++){
        u128 A=((u128)lcg()<<64)|lcg(), B=((u128)lcg()<<64)|lcg();
        uint16_t a[32],b[32]; pad(A,a); pad(B,b);
        sme_outer8_asm(a,b,tile);
        if (memcmp(reduce_tile(tile).w, mul_u128(A,B).w, 32)){
            if (++fails<=3){
                printf("  FAIL t=%ld: tile[0]=%llu tile[1]=%llu  expect a0*b0=%llu a0*b1=%llu\n",
                    t,(unsigned long long)tile[0],(unsigned long long)tile[1],
                    (unsigned long long)((uint64_t)(uint16_t)A*(uint16_t)B),
                    (unsigned long long)((uint64_t)(uint16_t)A*(uint16_t)(B>>16)));
            }
        }
    }
    printf("%ld multiplies, %ld mismatches -> %s\n",
           n, fails, fails?"FAIL (layout A2 wrong? dump above)":"BIT-EXACT on SME hardware!");
    if (fails) return 1;

    // rough timing (NOTE: smstart/smstop per call dominates; not optimized — a
    // real kernel keeps one streaming region around many MOPAs)
    int R=50; double best=1e30;
    uint16_t a[32],b[32]; pad(((u128)lcg()<<64)|lcg(),a); pad(((u128)lcg()<<64)|lcg(),b);
    for (int r=0;r<5;r++){ double t0=now_ns();
        for (int i=0;i<R*1000;i++) sme_outer8_asm(a,b,tile);
        double dt=now_ns()-t0; if(dt<best)best=dt; }
    printf("~%.1f ns / multiply (incl. per-call smstart/smstop + ZA readout — unoptimized)\n",
           best/(R*1000));
    return 0;
}
