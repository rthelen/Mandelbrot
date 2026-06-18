// REAL SME kernel: 128x128 bignum multiply via the za64_u16 outer-product MOPA.
// Apple M4-only (SME + FEAT_SME_I16I64). Written on an M1 (no SME) so it is
// COMPILE-checked but NOT runtime-validated here — the built-in self-check below
// makes the first M4 run immediately tell us BIT-EXACT or FAIL.
//
// Build (any Mac): clang -O3 -march=armv9-a+sme2+sme-i16i64 -o sme_mul sme_mul.c
// RUN ON M4+ ONLY.  A binary containing an __arm_new("za") function SIGILLs at
// STARTUP on a pre-SME core (M1/M2/M3) — the ZA-state init is emitted at load,
// before main, so the runtime feature gate below CANNOT save it there. It runs
// fine on SME hardware. (The gate still guards the rare SME-without-I16I64 case.)
//
// HOW: schoolbook multiply == outer product of the limb vectors. SME's MOPA does
// ZA[i][j] += <dot4>(Zn group i, Zm group j) into a 2D tile. With 16-bit limbs
// placed one-per-group and the other 3 lanes zeroed, the dot collapses to a pure
// product:  ZA64[i][j] = a_i * b_j.  Then sum anti-diagonals (i+j = digit) + carry.
//
// ASSUMPTIONS to confirm on M4 (the self-check verifies all of them at once):
//  (A1) SVL = 512 bits  => svcnth() == 32 u16 lanes => 8 groups of 4 => 8x8 tile.
//  (A2) the za64_u16 widening MOPA groups source lanes as CONSECUTIVE 4s, i.e.
//       ZA[i][j] += sum_{k=0..3} Zn[4i+k]*Zm[4j+k]. We put limb i at lane 4*i.
//       If the grouping is strided instead, the self-check FAILS and we switch
//       the limb stride (try LIMB_AT(i) = i, with zeros at i+8,i+16,i+24).
//  (A3) MOPA source predicate granularity is 16-bit (svptrue_b16()).
// If A1/A2/A3 hold, this prints BIT-EXACT. If not, it prints FAIL + a dump so we
// know exactly which assumption to flip.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>

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

#ifdef __ARM_FEATURE_SME
#include <arm_sme.h>

// The SME step: pure 8x8 outer product a_i*b_j into ZA64 tile 0, read out to
// tileout[8*i + j] (row-major). Also returns the streaming u16 lane count so the
// caller can confirm SVL (A1). __arm_new("za") gives us fresh ZA; __arm_streaming
// runs in streaming mode (compiler inserts smstart/smstop at the call site).
__arm_new("za") __attribute__((noinline))
static void sme_outer8(const uint16_t a[8], const uint16_t b[8],
                       uint64_t tileout[64], uint64_t *lanes_h)
        __arm_streaming
{
    *lanes_h = svcnth();                       // SVL/16 (expect 32 on M4)
    uint16_t za[32], zb[32];
    for (int k=0;k<32;k++){ za[k]=0; zb[k]=0; }
    for (int i=0;i<8;i++){ za[4*i]=a[i]; zb[4*i]=b[i]; }   // limb i at lane 4*i (A2)

    svbool_t p16 = svptrue_b16();
    svuint16_t va = svld1_u16(p16, za);
    svuint16_t vb = svld1_u16(p16, zb);

    svzero_za();
    svmopa_za64_u16_m(0, p16, p16, va, vb);    // ZA64[i][j] += dot4 = a_i*b_j

    svbool_t p64 = svptrue_b64();
    for (uint32_t s=0;s<8;s++){
        svuint64_t row = svread_hor_za64_u64_m(svdup_n_u64(0), p64, 0, s);
        svst1_u64(p64, &tileout[s*8], row);    // row s = [a_s*b_0 .. a_s*b_7]
    }
}
#endif // __ARM_FEATURE_SME

// reduce the 8x8 product tile: sum anti-diagonals (i+j) + carry-propagate 16-bit
static u256 reduce_tile(const uint64_t tile[64]){
    uint64_t col[16]={0};
    for (int i=0;i<8;i++) for (int j=0;j<8;j++) col[i+j]+=tile[i*8+j];
    uint16_t d[16]; uint64_t c=0;
    for (int k=0;k<16;k++){ uint64_t t=col[k]+c; d[k]=(uint16_t)(t&0xFFFF); c=t>>16; }
    u128 lo=0,hi=0;
    for (int k=15;k>=0;k--){ u128 cr=lo>>112; hi=(hi<<16)|cr; lo=(lo<<16)|d[k]; }
    u256 r={{(uint64_t)lo,(uint64_t)(lo>>64),(uint64_t)hi,(uint64_t)(hi>>64)}};
    return r;
}

static int feat(const char *k){ int v=0; size_t s=sizeof v; return sysctlbyname(k,&v,&s,0,0)==0 && v; }
static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }

int main(void){
    if (!feat("hw.optional.arm.FEAT_SME") || !feat("hw.optional.arm.FEAT_SME_I16I64")){
        printf("SME / FEAT_SME_I16I64 not available on this core — skipping (need M4+).\n");
        return 0;
    }
#ifndef __ARM_FEATURE_SME
    printf("built without SME support — rebuild with -march=armv9-a+sme2+sme-i16i64\n");
    return 0;
#else
    printf("SME present. Running za64_u16 outer-product 128x128 multiply...\n");
    long n=200000, fails=0; uint64_t lanes=0;
    for (long t=0;t<n;t++){
        uint16_t a[8],b[8]; u128 A=0,B=0;
        for (int i=0;i<8;i++){ a[i]=(uint16_t)lcg(); b[i]=(uint16_t)lcg(); }
        for (int i=7;i>=0;i--){ A=(A<<16)|a[i]; B=(B<<16)|b[i]; }
        uint64_t tile[64];
        sme_outer8(a,b,tile,&lanes);
        if (memcmp(reduce_tile(tile).w, mul_u128(A,B).w, 32)){
            if (++fails<=2){
                printf("  FAIL t=%ld  (streaming u16 lanes=%llu, expected 32)\n",
                       t,(unsigned long long)lanes);
                printf("  tile[0..2]= %llu %llu %llu  expected a0*b0=%u\n",
                    (unsigned long long)tile[0],(unsigned long long)tile[1],
                    (unsigned long long)tile[2], (unsigned)a[0]*b[0]);
            }
        }
    }
    printf("streaming vector: %llu u16 lanes (SVL=%llu bits)\n",
           (unsigned long long)lanes,(unsigned long long)lanes*16);
    printf("%ld multiplies, %ld mismatches -> %s\n",
           n, fails, fails?"FAIL (see assumptions A1-A3 in source)":"BIT-EXACT on SME!");
    return fails?1:0;
#endif
}
