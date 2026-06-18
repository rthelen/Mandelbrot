// NEON 128x128->256 multiply experiment.
//
// Question: can a NEON UMULL-based multiply beat scalar __uint128_t (MUL+UMULH)
// on Apple Silicon? Single-multiply NEON loses to scalar (carry merge needs
// cross-lane shuffles, and Apple's scalar MUL+UMULH co-issue is ~1 product/cyc).
// The lever is BATCHING: two independent multiplies, one per NEON lane. Lanes
// never need to talk (no cross-lane carry) -> carry resolution is plain vector
// adds. This is also the SVE/SME generalization (more lanes).
//
// Methodology (per project rules): validate BIT-EXACT vs scalar first, with a
// deterministic LCG (replayable), then measure throughput. No tolerance.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <arm_neon.h>
#include <mach/mach_time.h>

typedef unsigned __int128 u128;
typedef struct { uint64_t w[4]; } u256;   // w[0] low .. w[3] high

// ---- scalar reference: the exact product code from cf_mul (cmandelbrot.c) ----
static inline u256 mul_scalar(u128 A, u128 B) {
    uint64_t a0=(uint64_t)A, a1=(uint64_t)(A>>64);
    uint64_t b0=(uint64_t)B, b1=(uint64_t)(B>>64);
    u128 p00=(u128)a0*b0, p01=(u128)a0*b1, p10=(u128)a1*b0, p11=(u128)a1*b1;
    u128 lo=p00, hi=p11;
    uint64_t tl=(uint64_t)p01, th=(uint64_t)(p01>>64);
    u128 t=lo+((u128)tl<<64); hi+=(u128)th+(t<lo); lo=t;
    tl=(uint64_t)p10; th=(uint64_t)(p10>>64);
    t=lo+((u128)tl<<64); hi+=(u128)th+(t<lo); lo=t;
    u256 r;
    r.w[0]=(uint64_t)lo; r.w[1]=(uint64_t)(lo>>64);
    r.w[2]=(uint64_t)hi; r.w[3]=(uint64_t)(hi>>64);
    return r;
}

// ---- NEON 2-wide: lane 0 and lane 1 are two INDEPENDENT 128x128 multiplies ----
// Operand carried as 4 limbs; limb i holds {laneA.limb_i, laneB.limb_i}.
typedef struct { uint32x2_t l[4]; } op2;

static inline op2 pack2(u128 X, u128 Y) {
    op2 o;
    for (int i=0;i<4;i++) {
        uint32_t v[2] = { (uint32_t)(X>>(32*i)), (uint32_t)(Y>>(32*i)) };
        o.l[i] = vld1_u32(v);
    }
    return o;
}

// Core: 16 UMULL (each does both lanes' limb product), then per-lane column
// accumulate with lo/hi 32-bit split (keeps column sums < 2^35, no overflow),
// then ripple carry. All vector ops act on the two lanes independently.
static inline void mul_neon2_cols(op2 A, op2 B, uint64x2_t col[8]) {
    const uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFull);
    for (int c=0;c<8;c++) col[c]=vdupq_n_u64(0);
    for (int i=0;i<4;i++) {
        for (int j=0;j<4;j++) {
            uint64x2_t p  = vmull_u32(A.l[i], B.l[j]);
            uint64x2_t lo = vandq_u64(p, mask32);
            uint64x2_t hi = vshrq_n_u64(p, 32);
            col[i+j]   = vaddq_u64(col[i+j],   lo);
            col[i+j+1] = vaddq_u64(col[i+j+1], hi);
        }
    }
    // ripple the inter-column carries (still both lanes at once)
    uint64x2_t carry = vdupq_n_u64(0);
    for (int c=0;c<8;c++) {
        uint64x2_t t = vaddq_u64(col[c], carry);
        col[c] = vandq_u64(t, mask32);     // 32-bit digit per lane
        carry  = vshrq_n_u64(t, 32);
    }
}

static inline void mul_neon2(op2 A, op2 B, u256 out[2]) {
    uint64x2_t col[8];
    mul_neon2_cols(A, B, col);
    uint32_t d[2][8];
    for (int c=0;c<8;c++) {
        d[0][c] = (uint32_t)vgetq_lane_u64(col[c], 0);
        d[1][c] = (uint32_t)vgetq_lane_u64(col[c], 1);
    }
    for (int L=0;L<2;L++) {
        out[L].w[0]=(uint64_t)d[L][0]|((uint64_t)d[L][1]<<32);
        out[L].w[1]=(uint64_t)d[L][2]|((uint64_t)d[L][3]<<32);
        out[L].w[2]=(uint64_t)d[L][4]|((uint64_t)d[L][5]<<32);
        out[L].w[3]=(uint64_t)d[L][6]|((uint64_t)d[L][7]<<32);
    }
}

// ---- deterministic LCG (replayable; never rand()) ----
static uint64_t lcg_state = 0x123456789abcdef0ull;
static inline uint64_t lcg(void) {
    lcg_state = lcg_state*6364136223846793005ull + 1442695040888963407ull;
    return lcg_state;
}
static inline u128 rnd128(void) { return ((u128)lcg()<<64) | lcg(); }

static int eq256(u256 a, u256 b){ return !memcmp(a.w,b.w,sizeof a.w); }

// ---- timing ----
static double now_ns(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom==0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom;
}

int main(void) {
    // 1) BIT-EXACT validation gate
    long checked=0, fails=0;
    for (long t=0; t<2000000; t++) {
        u128 a=rnd128(), b=rnd128(), c=rnd128(), d=rnd128();
        u256 s0=mul_scalar(a,b), s1=mul_scalar(c,d);
        u256 nn[2]; mul_neon2(pack2(a,c), pack2(b,d), nn);
        checked+=2;
        if (!eq256(s0,nn[0])) { fails++; if(fails<=3) printf("MISMATCH lane0 t=%ld\n",t); }
        if (!eq256(s1,nn[1])) { fails++; if(fails<=3) printf("MISMATCH lane1 t=%ld\n",t); }
    }
    printf("validation: %ld products checked, %ld mismatches -> %s\n",
           checked, fails, fails? "FAIL":"BIT-EXACT");
    if (fails) return 1;

    // 2) THROUGHPUT. Inputs carried in each method's native form (the form a
    //    real kernel would keep z-state in across iterations), so we measure the
    //    multiply core, not the GPR<->vector pack tax.
    enum { N = 4096 };               // fits cache
    static u128 sa[N], sb[N], sc[N], sd[N];   // scalar operands (two muls/item)
    static op2  na[N], nb[N];                 // neon operands (one call/item)
    for (int i=0;i<N;i++){
        sa[i]=rnd128(); sb[i]=rnd128(); sc[i]=rnd128(); sd[i]=rnd128();
        na[i]=pack2(sa[i],sc[i]); nb[i]=pack2(sb[i],sd[i]);
    }
    const int R = 20000;             // outer passes
    const double muls = (double)N * R * 2.0;   // total multiplies (2 per item)

    // scalar
    double best_s=1e30;
    volatile uint64_t sink_s=0;
    for (int rep=0; rep<5; rep++){
        double t0=now_ns();
        uint64_t acc=0;
        for (int r=0;r<R;r++){
            for (int i=0;i<N;i++){
                u256 x=mul_scalar(sa[i], sb[i]);
                u256 y=mul_scalar(sc[i], sd[i]);
                acc += x.w[0]^y.w[2]^(uint64_t)r;   // cheap, perturbs per pass
            }
        }
        double dt=now_ns()-t0; sink_s^=acc;
        if (dt<best_s) best_s=dt;
    }

    // neon 2-wide
    double best_n=1e30;
    volatile uint64_t sink_n=0;
    for (int rep=0; rep<5; rep++){
        double t0=now_ns();
        uint64x2_t acc=vdupq_n_u64(0);
        for (int r=0;r<R;r++){
            uint64x2_t rr=vdupq_n_u64((uint64_t)r);
            for (int i=0;i<N;i++){
                uint64x2_t col[8];
                mul_neon2_cols(na[i], nb[i], col);
                acc = vaddq_u64(acc, veorq_u64(col[0], vaddq_u64(col[4], rr)));
            }
        }
        double dt=now_ns()-t0; sink_n^=vgetq_lane_u64(acc,0)^vgetq_lane_u64(acc,1);
        if (dt<best_n) best_n=dt;
    }

    printf("\nthroughput (best of 5, %g multiplies each):\n", muls);
    printf("  scalar u128 : %8.3f ms  -> %6.3f ns/mul\n", best_s/1e6, best_s/muls);
    printf("  neon 2-wide : %8.3f ms  -> %6.3f ns/mul\n", best_n/1e6, best_n/muls);
    printf("  speedup     : %.3fx %s\n", best_s/best_n, best_n<best_s?"(NEON wins)":"(scalar wins)");
    printf("[sink %llu %llu]\n", (unsigned long long)sink_s,(unsigned long long)sink_n);
    return 0;
}
