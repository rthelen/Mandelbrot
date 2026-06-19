// Pi-scale SME throughput: sustained umopa rate, amortized in ONE streaming
// region, full 32-lane (real 4-way-dot) utilization, 8 independent ZA tiles to
// hide latency. This is the matrix engine's CEILING — the number that decides
// whether SME helps the huge Pi multiply. Grounded by a bit-exact check of the
// (proven) za64_u16 outer product, and compared to the scalar multiplier in the
// same program (clock auto-measured, so ratios are honest on the M4's clock).
//
// Build: clang -O3 -march=armv9-a+sme2+sme-i16i64 -o sme_wide sme_wide.c
// Run on M4+: ./sme_wide
//
// Each umopa za64_u16 with full lanes = 8x8 tile x 4-way dot = 256 u16 MACs.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>
#include <mach/mach_time.h>
#include <pthread.h>     // QoS -> P-core placement

typedef unsigned __int128 u128;
typedef struct { uint64_t w[4]; } u256;
static double now_ns(void){ static mach_timebase_info_data_t tb; if(!tb.denom)mach_timebase_info(&tb);
                            return (double)mach_absolute_time()*tb.numer/tb.denom; }

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
static u256 reduce_tile(const uint64_t *tile){
    uint64_t col[16]={0}; for(int i=0;i<8;i++)for(int j=0;j<8;j++) col[i+j]+=tile[i*8+j];
    uint16_t d[16]; uint64_t c=0; for(int k=0;k<16;k++){uint64_t t=col[k]+c; d[k]=(uint16_t)(t&0xFFFF); c=t>>16;}
    u128 lo=0,hi=0; for(int k=15;k>=0;k--){u128 cr=lo>>112; hi=(hi<<16)|cr; lo=(lo<<16)|d[k];}
    u256 r={{(uint64_t)lo,(uint64_t)(lo>>64),(uint64_t)hi,(uint64_t)(hi>>64)}}; return r;
}
static void pad(u128 v, uint16_t o[32]){ for(int k=0;k<32;k++)o[k]=0; for(int i=0;i<8;i++)o[4*i]=(uint16_t)(v>>(16*i)); }

// proven correctness kernel (one multiply + readout), for the bit-exact gate
__attribute__((noinline))
static void sme_outer8(const uint16_t *a,const uint16_t *b,uint64_t *out){
    __asm__ __volatile__(
        "smstart\n ptrue p0.h\n ld1h {z0.h},p0/z,[%[a]]\n ld1h {z1.h},p0/z,[%[b]]\n"
        "zero {za}\n umopa za0.d,p0/m,p0/m,z0.h,z1.h\n ptrue p0.d\n"
        "mov w12,#0\n mov z2.d,p0/m,za0h.d[w12,0]\n str z2,[%[o],#0,mul vl]\n"
        "mov w12,#1\n mov z2.d,p0/m,za0h.d[w12,0]\n str z2,[%[o],#1,mul vl]\n"
        "mov w12,#2\n mov z2.d,p0/m,za0h.d[w12,0]\n str z2,[%[o],#2,mul vl]\n"
        "mov w12,#3\n mov z2.d,p0/m,za0h.d[w12,0]\n str z2,[%[o],#3,mul vl]\n"
        "mov w12,#4\n mov z2.d,p0/m,za0h.d[w12,0]\n str z2,[%[o],#4,mul vl]\n"
        "mov w12,#5\n mov z2.d,p0/m,za0h.d[w12,0]\n str z2,[%[o],#5,mul vl]\n"
        "mov w12,#6\n mov z2.d,p0/m,za0h.d[w12,0]\n str z2,[%[o],#6,mul vl]\n"
        "mov w12,#7\n mov z2.d,p0/m,za0h.d[w12,0]\n str z2,[%[o],#7,mul vl]\n smstop\n"
        : : [a]"r"(a),[b]"r"(b),[o]"r"(out)
        : "memory","w12","p0","z0","z1","z2","v8","v9","v10","v11","v12","v13","v14","v15");
}

// PEAK: 8*K umopa in one streaming region, 8 independent tiles (no readout).
__attribute__((noinline))
static void sme_peak(const uint16_t *a,const uint16_t *b,long K){
    __asm__ __volatile__(
        "smstart\n ptrue p0.h\n ld1h {z0.h},p0/z,[%[a]]\n ld1h {z1.h},p0/z,[%[b]]\n"
        "mov x9,%[k]\n 1:\n"
        "umopa za0.d,p0/m,p0/m,z0.h,z1.h\n umopa za1.d,p0/m,p0/m,z0.h,z1.h\n"
        "umopa za2.d,p0/m,p0/m,z0.h,z1.h\n umopa za3.d,p0/m,p0/m,z0.h,z1.h\n"
        "umopa za4.d,p0/m,p0/m,z0.h,z1.h\n umopa za5.d,p0/m,p0/m,z0.h,z1.h\n"
        "umopa za6.d,p0/m,p0/m,z0.h,z1.h\n umopa za7.d,p0/m,p0/m,z0.h,z1.h\n"
        "subs x9,x9,#1\n b.ne 1b\n smstop\n"
        : : [a]"r"(a),[b]"r"(b),[k]"r"(K)
        : "memory","x9","p0","z0","z1","v8","v9","v10","v11","v12","v13","v14","v15");
}

// clock: 64 dependent adds/iter at 1/cycle
#define A8 "add x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\n"
static void clk(long R){ __asm__ __volatile__("mov x0,#0\nmov x1,%0\n9:\n" A8 A8 A8 A8 A8 A8 A8 A8 "subs x1,x1,#1\nb.ne 9b\n"::"r"(R):"x0","x1"); }
// scalar: 8 independent 64x64->64 mul chains/iter
static void smul(const uint64_t *in,uint64_t *out,long R){
    __asm__ __volatile__("mov x0,%0\nldp x1,x9,[x0,#0]\nldp x2,x10,[x0,#16]\nldp x3,x11,[x0,#32]\nldp x4,x12,[x0,#48]\n"
        "ldp x5,x13,[x0,#64]\nldp x6,x14,[x0,#80]\nldp x7,x15,[x0,#96]\nldp x8,x16,[x0,#112]\nmov x17,%2\n2:\n"
        "mul x1,x1,x9\nmul x2,x2,x10\nmul x3,x3,x11\nmul x4,x4,x12\nmul x5,x5,x13\nmul x6,x6,x14\nmul x7,x7,x15\nmul x8,x8,x16\n"
        "subs x17,x17,#1\nb.ne 2b\nmov x0,%1\nstp x1,x2,[x0]\nstp x3,x4,[x0,#16]\n"
        : : "r"(in),"r"(out),"r"(R) : "x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","memory");
}

static int feat(const char *k){ int v=0; size_t s=sizeof v; return sysctlbyname(k,&v,&s,0,0)==0 && v; }
static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }

int main(void){
    // bias the scheduler to a P-core (no hard affinity on macOS; QoS is the lever)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    if (!feat("hw.optional.arm.FEAT_SME") || !feat("hw.optional.arm.FEAT_SME_I16I64")){
        printf("SME / I16I64 not available — need M4+.\n"); return 0; }

    // 1) bit-exact gate (grounds the throughput in a validated MOPA)
    u128 A=((u128)lcg()<<64)|lcg(), B=((u128)lcg()<<64)|lcg();
    uint16_t a[32],b[32]; pad(A,a); pad(B,b);
    __attribute__((aligned(64))) uint64_t tile[64];
    sme_outer8(a,b,tile);
    int ok = !memcmp(reduce_tile(tile).w, mul_u128(A,B).w, 32);
    printf("MOPA bit-exact gate: %s\n", ok?"BIT-EXACT":"FAIL");
    if(!ok) return 1;

    // full-lane operands for peak (not zero-padded -> real 256 MAC/umopa)
    for(int k=0;k<32;k++){ a[k]=(uint16_t)lcg(); b[k]=(uint16_t)lcg(); }

    // WARM UP ~700ms of the SME workload so macOS migrates us onto a P-core AND
    // ramps DVFS to max clock before any measurement. Then all timed runs (back
    // to back, no idle gaps) stay in the hot lane; best-of-N takes the fastest.
    printf("warming up (P-core migration + clock ramp)...\n"); fflush(stdout);
    { double t0=now_ns(); while(now_ns()-t0 < 700e6) sme_peak(a,b,200000); }

    // 2) clock — measured HOT (the achieved GHz is our "are we boosted?" check)
    long Rc=6000000; double cb=1e30,cw=0;            // ~80-160ms/run
    for(int r=0;r<6;r++){double t=now_ns(); clk(Rc); double d=now_ns()-t; if(d<cb)cb=d; if(d>cw)cw=d;}
    double ghz=(64.0*Rc)/cb, ghz_lo=(64.0*Rc)/cw;

    // 3) SME peak umopa — long runs (240M umopa each), report trial spread
    long K=30000000; double sb=1e30,sw=0;
    for(int r=0;r<6;r++){double t=now_ns(); sme_peak(a,b,K); double d=now_ns()-t; if(d<sb)sb=d; if(d>sw)sw=d;}
    double umopa=8.0*K, ns_umopa=sb/umopa;
    double mac_per_ns_sme = 256.0/ns_umopa;            // 256 u16-MACs per umopa

    // 4) scalar 64x64 mul peak -> u16-MAC equiv (one 64x64 schoolbook = (64/16)^2 = 16 u16 MACs)
    uint64_t in[16],out[16]; for(int i=0;i<16;i++) in[i]=lcg()|1;
    long Rs=20000000; double xb=1e30;
    for(int r=0;r<6;r++){double t=now_ns(); smul(in,out,Rs); double d=now_ns()-t; if(d<xb)xb=d;}
    double mul64_per_ns = (8.0*Rs)/xb;
    double mac_per_ns_scalar = mul64_per_ns*16.0;

    printf("\nclock (hot): %.2f GHz   (slowest trial %.2f GHz)%s\n", ghz, ghz_lo,
           ghz<3.5 ? "   <-- LOW: likely eCore/unboosted, RE-RUN" : "   (P-core, boosted)");
    printf("SME umopa: %.3f ns each = %.2f umopa/cycle  (peak, 8 tiles, full lanes; trial spread %.0f%%)\n",
           ns_umopa, 1.0/(ns_umopa*ghz), 100.0*(sw-sb)/sb);
    printf("\nu16 multiply-accumulate throughput (wall-clock, clock-independent):\n");
    printf("  SME (umopa peak) : %8.1f u16-MAC/ns\n", mac_per_ns_sme);
    printf("  scalar (mul x16) : %8.1f u16-MAC/ns\n", mac_per_ns_scalar);
    printf("  SME / scalar     : %.1fx  (engine ceiling; real wide-multiply lower by ZA-readout overhead)\n",
           mac_per_ns_sme/mac_per_ns_scalar);
    return 0;
}
