// Dual-engine throughput: NEON MUL53 pipes  ||  scalar integer-multiply pipes.
//
// MUL53 is a NEON/vector op, so it shares pipes with FMA — but it does NOT share
// with the scalar integer multiplier (MUL/UMULH). While a NEON-multiply bignum
// stream runs, the 2 scalar int-mul pipes sit idle. Partition the limb-work
// across BOTH and the scalar throughput is ~free on top of the NEON stream.
//
// Proof of co-issue: time each engine alone, then interleaved. If combined ~=
// max(scalar,neon) instead of sum, the OoO core is running them concurrently on
// separate datapaths -> additive multiply throughput within one core.

#include <stdint.h>
#include <stdio.h>
#include <mach/mach_time.h>

static double now_ns(void){ static mach_timebase_info_data_t tb; if(!tb.denom)mach_timebase_info(&tb);
                            return (double)mach_absolute_time()*tb.numer/tb.denom; }

// 16 MUL53 chains/iter (v16+k = v16+k * vk). 16 lane-pairs = 32 lane-products/iter.
#define MUL53_BODY \
    ".long 0x00200010\n .long 0x00200031\n .long 0x00200052\n .long 0x00200073\n" \
    ".long 0x00200094\n .long 0x002000B5\n .long 0x002000D6\n .long 0x002000F7\n" \
    ".long 0x00200118\n .long 0x00200139\n .long 0x0020015A\n .long 0x0020017B\n" \
    ".long 0x0020019C\n .long 0x002001BD\n .long 0x002001DE\n .long 0x002001FF\n"
// 8 scalar mul chains/iter (low 64x64).
#define SCALAR_BODY \
    "mul x1,x1,x9\n mul x2,x2,x10\n mul x3,x3,x11\n mul x4,x4,x12\n" \
    "mul x5,x5,x13\n mul x6,x6,x14\n mul x7,x7,x15\n mul x8,x8,x16\n"

#define LD_VREGS \
    "ld1 {v0.2d-v3.2d},[x0],#64\n ld1 {v4.2d-v7.2d},[x0],#64\n" \
    "ld1 {v8.2d-v11.2d},[x0],#64\n ld1 {v12.2d-v15.2d},[x0],#64\n" \
    "ld1 {v16.2d-v19.2d},[x0],#64\n ld1 {v20.2d-v23.2d},[x0],#64\n" \
    "ld1 {v24.2d-v27.2d},[x0],#64\n ld1 {v28.2d-v31.2d},[x0],#64\n"
#define ALLV "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12",\
    "v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25",\
    "v26","v27","v28","v29","v30","v31"

static void k_mul53(const uint64_t *vs, uint64_t *sink, long R){
    __asm__ __volatile__(
        "mov x0,%0\n" LD_VREGS "mov x1,%2\n"
        "1:\n" MUL53_BODY "subs x1,x1,#1\n b.ne 1b\n"
        "mov x0,%1\n st1 {v16.2d-v19.2d},[x0],#64\n st1 {v20.2d-v23.2d},[x0],#64\n"
        "st1 {v24.2d-v27.2d},[x0],#64\n st1 {v28.2d-v31.2d},[x0]\n"
        : : "r"(vs),"r"(sink),"r"(R) : "x0","x1","memory", ALLV);
}

static void k_scalar(const uint64_t *ss, uint64_t *sink, long R){
    __asm__ __volatile__(
        "mov x0,%0\n"
        "ldp x1,x9,[x0,#0]\n ldp x2,x10,[x0,#16]\n ldp x3,x11,[x0,#32]\n ldp x4,x12,[x0,#48]\n"
        "ldp x5,x13,[x0,#64]\n ldp x6,x14,[x0,#80]\n ldp x7,x15,[x0,#96]\n ldp x8,x16,[x0,#112]\n"
        "mov x17,%2\n"
        "2:\n" SCALAR_BODY "subs x17,x17,#1\n b.ne 2b\n"
        "mov x0,%1\n stp x1,x2,[x0,#0]\n stp x3,x4,[x0,#16]\n stp x5,x6,[x0,#32]\n stp x7,x8,[x0,#48]\n"
        : : "r"(ss),"r"(sink),"r"(R)
        : "x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12",
          "x13","x14","x15","x16","x17","memory");
}

// both engines interleaved (2 mul53 : 1 mul, repeated) so the scheduler sees both
static void k_both(const uint64_t *vs, const uint64_t *ss, uint64_t *sink, long R){
    __asm__ __volatile__(
        "mov x0,%0\n" LD_VREGS
        "mov x20,%1\n"
        "ldp x1,x9,[x20,#0]\n ldp x2,x10,[x20,#16]\n ldp x3,x11,[x20,#32]\n ldp x4,x12,[x20,#48]\n"
        "ldp x5,x13,[x20,#64]\n ldp x6,x14,[x20,#80]\n ldp x7,x15,[x20,#96]\n ldp x8,x16,[x20,#112]\n"
        "mov x19,%3\n"
        "3:\n"
        ".long 0x00200010\n .long 0x00200031\n mul x1,x1,x9\n"
        ".long 0x00200052\n .long 0x00200073\n mul x2,x2,x10\n"
        ".long 0x00200094\n .long 0x002000B5\n mul x3,x3,x11\n"
        ".long 0x002000D6\n .long 0x002000F7\n mul x4,x4,x12\n"
        ".long 0x00200118\n .long 0x00200139\n mul x5,x5,x13\n"
        ".long 0x0020015A\n .long 0x0020017B\n mul x6,x6,x14\n"
        ".long 0x0020019C\n .long 0x002001BD\n mul x7,x7,x15\n"
        ".long 0x002001DE\n .long 0x002001FF\n mul x8,x8,x16\n"
        "subs x19,x19,#1\n b.ne 3b\n"
        "mov x0,%2\n st1 {v16.2d-v19.2d},[x0],#64\n st1 {v20.2d-v23.2d},[x0],#64\n"
        "st1 {v24.2d-v27.2d},[x0],#64\n st1 {v28.2d-v31.2d},[x0],#64\n"
        "stp x1,x2,[x0,#0]\n stp x3,x4,[x0,#16]\n"
        : : "r"(vs),"r"(ss),"r"(sink),"r"(R)
        : "x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12",
          "x13","x14","x15","x16","x19","x20","memory", ALLV);
}

static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }

int main(void){
    const double GHZ=3.2;
    uint64_t vs[64], ss[16], sink[64];
    for(int i=0;i<64;i++) vs[i]=(lcg()|1)&((1ull<<53)-1);
    for(int i=0;i<16;i++) ss[i]=lcg()|1;
    const long R=4000000;
    double bm=1e30,bs=1e30,bb=1e30;
    for(int r=0;r<6;r++){ double t=now_ns(); k_mul53(vs,sink,R);  double d=now_ns()-t; if(d<bm)bm=d; }
    for(int r=0;r<6;r++){ double t=now_ns(); k_scalar(ss,sink,R); double d=now_ns()-t; if(d<bs)bs=d; }
    for(int r=0;r<6;r++){ double t=now_ns(); k_both(vs,ss,sink,R); double d=now_ns()-t; if(d<bb)bb=d; }

    double ns_m=bm/R, ns_s=bs/R, ns_b=bb/R;
    // products/iter: mul53 32 lane-products, scalar 8, both 40
    printf("per-iteration time (R=%ld):\n", R);
    printf("  MUL53 only  : %6.3f ns/iter  (32 lane-products)  %.2f cyc\n", ns_m, ns_m*GHZ);
    printf("  scalar only : %6.3f ns/iter  ( 8 products)       %.2f cyc\n", ns_s, ns_s*GHZ);
    printf("  BOTH        : %6.3f ns/iter  (40 products)       %.2f cyc\n", ns_b, ns_b*GHZ);
    printf("\nco-issue check:  sum=%.3f  max=%.3f  both=%.3f ns\n", ns_m+ns_s, ns_m>ns_s?ns_m:ns_s, ns_b);
    printf("  concurrency factor (sum/both) = %.2fx   (2.0 = perfect overlap, 1.0 = fully serial)\n",
           (ns_m+ns_s)/ns_b);
    printf("  scalar work hidden behind NEON: %.0f%%\n", 100.0*(1.0-(ns_b-ns_m)/ns_s));
    printf("\nmultiply throughput (lane-products/ns):\n");
    printf("  MUL53 only %.2f | scalar only %.2f | BOTH %.2f\n",
           32.0/ns_m, 8.0/ns_s, 40.0/ns_b);
    printf("[sink %llu]\n",(unsigned long long)sink[0]);
    return 0;
}
