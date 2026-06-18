// FP64-FMA bignum multiply vs scalar __uint128_t schoolbook — WIDTH SWEEP.
//
// Runnable proxy on M1 (today, no private ISA) for Apple's 52-bit multiplier:
// borrow the FP mantissa multiplier for integer bignum via documented FMA.
// Radix 2^24 limbs as doubles -> each product < 2^48, and <=22 of them sum to
// < 2^53, so every vfmaq_f64 accumulate is EXACT (single rounding, no rounding
// actually occurs). Batch 2 independent multiplies, one per float64x2_t lane.
//
// Thesis under test: scalar wins at 128 bits (Apple's 64x64 multiplier is too
// strong, only 4 sub-products), but FP's O(n^2) cheap-FMA schoolbook overtakes
// as width grows -> the regime where Pi lives. Validate BIT-EXACT at each width.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <arm_neon.h>
#include <mach/mach_time.h>

typedef unsigned __int128 u128;
#define MASK24 0xFFFFFFull

// ---------- scalar schoolbook: nl 64-bit limbs -> 2*nl limbs ----------
static void mul_scalar_n(const uint64_t *A, const uint64_t *B, uint64_t *R, int nl){
    for (int k=0;k<2*nl;k++) R[k]=0;
    for (int i=0;i<nl;i++){
        u128 carry=0;
        for (int j=0;j<nl;j++){
            u128 t=(u128)A[i]*B[j] + R[i+j] + carry;
            R[i+j]=(uint64_t)t; carry=t>>64;
        }
        R[i+nl]+=(uint64_t)carry;
    }
}

// ---------- FP64-FMA, 2-wide: lane 0 / lane 1 are two independent multiplies --
// operands carried as nd doubles per lane (24-bit limbs). Returns 2*nd digits
// (24-bit, in float64x2_t) — caller reconstructs/compares.
static void mul_fp_n(const float64x2_t *A, const float64x2_t *B, int nd,
                     float64x2_t *col /*size 2*nd*/){
    int nc=2*nd;
    for (int c=0;c<nc;c++) col[c]=vdupq_n_f64(0.0);
    for (int i=0;i<nd;i++)
        for (int j=0;j<nd;j++)
            col[i+j]=vfmaq_f64(col[i+j], A[i], B[j]);   // exact MAC, both lanes
    // normalize to 24-bit digits, ripple carry (per lane, exact FP)
    const float64x2_t inv=vdupq_n_f64(1.0/16777216.0), rad=vdupq_n_f64(16777216.0);
    float64x2_t carry=vdupq_n_f64(0.0);
    for (int c=0;c<nc;c++){
        float64x2_t v=vaddq_f64(col[c], carry);
        float64x2_t hi=vrndmq_f64(vmulq_f64(v, inv));   // floor(v / 2^24), exact
        col[c]=vfmsq_f64(v, hi, rad);                   // v - hi*2^24 in [0,2^24)
        carry=hi;
    }
}

// ---------- conversions / reconstruction (validation only) ----------
static void to_fp_limbs(const uint64_t *A, int nl, double *out, int nd){
    for (int c=0;c<nd;c++){
        int off=24*c, w=off>>6, b=off&63;
        uint64_t lo=A[w]>>b;
        uint64_t hi=(b>0 && w+1<nl)? A[w+1]<<(64-b):0;
        out[c]=(double)((lo|hi)&MASK24);
    }
}
static void add_shifted(uint64_t *R, int rl, uint64_t d, int off){
    int w=off>>6, b=off&63; u128 v=(u128)d<<b; int k=w;
    while (v && k<rl){ u128 s=(u128)R[k]+(uint64_t)v; R[k]=(uint64_t)s; v>>=64; v+=(s>>64); k++; }
}

static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }
static double now_ns(void){ static mach_timebase_info_data_t tb; if(!tb.denom)mach_timebase_info(&tb);
                            return (double)mach_absolute_time()*tb.numer/tb.denom; }

#define MAXL 8          // max 64-bit limbs (512-bit)
#define MAXD 22         // max 24-bit limbs
#define MAXRL 16        // max result 64-bit limbs

static int run_width(int bits){
    int nl=bits/64;                 // 64-bit limbs
    int nd=(bits+23)/24;            // 24-bit limbs
    printf("=== %d-bit  (scalar %d x %d-limbs=%d prods | fp %d x %d-limbs=%d prods) ===\n",
           bits, nl,nl,nl*nl, nd,nd,nd*nd);

    enum { N=2048 };
    static uint64_t A0[N][MAXL],B0[N][MAXL],A1[N][MAXL],B1[N][MAXL];
    static float64x2_t FA[N][MAXD],FB[N][MAXD];
    for (int i=0;i<N;i++){
        for (int k=0;k<nl;k++){ A0[i][k]=lcg();B0[i][k]=lcg();A1[i][k]=lcg();B1[i][k]=lcg(); }
        double a0[MAXD],a1[MAXD],b0[MAXD],b1[MAXD];
        to_fp_limbs(A0[i],nl,a0,nd); to_fp_limbs(A1[i],nl,a1,nd);
        to_fp_limbs(B0[i],nl,b0,nd); to_fp_limbs(B1[i],nl,b1,nd);
        for (int k=0;k<nd;k++){ double t0[2]={a0[k],a1[k]},t1[2]={b0[k],b1[k]};
            FA[i][k]=vld1q_f64(t0); FB[i][k]=vld1q_f64(t1); }
    }

    // --- bit-exact gate: FP product (both lanes) == scalar product ---
    long fails=0;
    for (int i=0;i<N;i++){
        uint64_t Rs[2*MAXL]; float64x2_t col[2*MAXD];
        mul_fp_n(FA[i],FB[i],nd,col);
        // lane 0 = A0*B0, lane 1 = A1*B1
        for (int lane=0; lane<2; lane++){
            const uint64_t *X = lane? A1[i]:A0[i], *Y = lane? B1[i]:B0[i];
            mul_scalar_n(X,Y,Rs,nl);
            uint64_t Rf[MAXRL]; for (int k=0;k<MAXRL;k++) Rf[k]=0;
            for (int c=0;c<2*nd;c++){
                double pair[2]; vst1q_f64(pair, col[c]);
                add_shifted(Rf,MAXRL,(uint64_t)pair[lane],24*c);
            }
            if (memcmp(Rs,Rf,sizeof(uint64_t)*2*nl)){ if(++fails<=2) printf("  MISMATCH i=%d lane=%d\n",i,lane); }
        }
    }
    printf("  validation: %s (%ld mismatches over %d products)\n", fails?"FAIL":"BIT-EXACT", fails, 2*N);
    if (fails) return 1;

    const int R=4000; const double muls=(double)N*R*2.0;
    // scalar timing (2 muls/item)
    double bs=1e30; volatile uint64_t sink=0;
    for (int rep=0;rep<5;rep++){ double t=now_ns(); uint64_t acc=0;
        for (int r=0;r<R;r++) for (int i=0;i<N;i++){
            uint64_t R0[2*MAXL],R1[2*MAXL];
            mul_scalar_n(A0[i],B0[i],R0,nl); mul_scalar_n(A1[i],B1[i],R1,nl);
            acc+=R0[0]^R1[nl]^(uint64_t)r;
        } double dt=now_ns()-t; sink^=acc; if(dt<bs)bs=dt; }
    // fp timing (2 muls/item, batched)
    double bf=1e30; volatile uint64_t sinkf=0;
    for (int rep=0;rep<5;rep++){ double t=now_ns(); float64x2_t acc=vdupq_n_f64(0);
        for (int r=0;r<R;r++) for (int i=0;i<N;i++){
            float64x2_t col[2*MAXD]; mul_fp_n(FA[i],FB[i],nd,col);
            acc=vaddq_f64(acc, col[0]);
        } double dt=now_ns()-t; sinkf^=(uint64_t)vgetq_lane_f64(acc,0); if(dt<bf)bf=dt; }

    printf("  scalar u128 : %6.3f ns/mul\n", bs/muls);
    printf("  fp64-fma    : %6.3f ns/mul   speedup %.3fx %s\n",
           bf/muls, bs/bf, bf<bs?"(FP wins)":"(scalar wins)");
    printf("  [sink %llu %llu]\n\n",(unsigned long long)sink,(unsigned long long)sinkf);
    return 0;
}

int main(void){
    int widths[]={128,256,512};
    for (int i=0;i<3;i++) if (run_width(widths[i])) return 1;
    return 0;
}
