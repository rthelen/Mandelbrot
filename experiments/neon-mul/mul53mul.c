// Full 128x128->256 multiply using Apple's MUL53, end-to-end vs __uint128_t.
//
// 53-bit-limb schoolbook: 3 limbs/operand, 9 partial products, 2-wide (lane0,
// lane1 = two independent multiplies). MUL53 is destructive + no-MAC, so each
// product is: copy a_i -> mul53lo/hi by b_j -> vector-add into a 53-bit column
// accumulator. Then one carry-ripple normalize to 53-bit digits.
//
// mul53lo.2d vD,vM = 0x00200000|(M<<5)|D ; mul53hi.2d = 0x00200400|(M<<5)|D
// Here lo result -> v12, hi result -> v13 (HW rename gives ILP across the 18
// independent mul53). Encodings used (D=12 lo / D=13 hi ; M=3,4,5 = b0,b1,b2).

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <mach/mach_time.h>

typedef unsigned __int128 u128;
typedef struct { uint64_t w[4]; } u256;
#define MASK53 ((1ull<<53)-1)

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

// a,b -> 3 uint64x2 limbs each (mem: limb0.lane0,limb0.lane1,limb1...). out: 6
// uint64x2 digits. mask -> {MASK53,MASK53}.
__attribute__((always_inline)) static inline
void mul53_128(const uint64_t *a,const uint64_t *b,uint64_t *out,const uint64_t *mask){
    __asm__ __volatile__(
        "ld1 {v0.2d-v2.2d}, [%0]\n"
        "ld1 {v3.2d-v5.2d}, [%1]\n"
        "ld1 {v16.2d}, [%3]\n"
        "movi v6.2d,#0\n movi v7.2d,#0\n movi v8.2d,#0\n"
        "movi v9.2d,#0\n movi v10.2d,#0\n movi v11.2d,#0\n"
        "mov v12.16b,v0.16b\n .long 0x0020006C\n add v6.2d,v6.2d,v12.2d\n"   // P00 lo->c0
        "mov v13.16b,v0.16b\n .long 0x0020046D\n add v7.2d,v7.2d,v13.2d\n"   // P00 hi->c1
        "mov v12.16b,v0.16b\n .long 0x0020008C\n add v7.2d,v7.2d,v12.2d\n"   // P01 lo->c1
        "mov v13.16b,v0.16b\n .long 0x0020048D\n add v8.2d,v8.2d,v13.2d\n"   // P01 hi->c2
        "mov v12.16b,v0.16b\n .long 0x002000AC\n add v8.2d,v8.2d,v12.2d\n"   // P02 lo->c2
        "mov v13.16b,v0.16b\n .long 0x002004AD\n add v9.2d,v9.2d,v13.2d\n"   // P02 hi->c3
        "mov v12.16b,v1.16b\n .long 0x0020006C\n add v7.2d,v7.2d,v12.2d\n"   // P10 lo->c1
        "mov v13.16b,v1.16b\n .long 0x0020046D\n add v8.2d,v8.2d,v13.2d\n"   // P10 hi->c2
        "mov v12.16b,v1.16b\n .long 0x0020008C\n add v8.2d,v8.2d,v12.2d\n"   // P11 lo->c2
        "mov v13.16b,v1.16b\n .long 0x0020048D\n add v9.2d,v9.2d,v13.2d\n"   // P11 hi->c3
        "mov v12.16b,v1.16b\n .long 0x002000AC\n add v9.2d,v9.2d,v12.2d\n"   // P12 lo->c3
        "mov v13.16b,v1.16b\n .long 0x002004AD\n add v10.2d,v10.2d,v13.2d\n" // P12 hi->c4
        "mov v12.16b,v2.16b\n .long 0x0020006C\n add v8.2d,v8.2d,v12.2d\n"   // P20 lo->c2
        "mov v13.16b,v2.16b\n .long 0x0020046D\n add v9.2d,v9.2d,v13.2d\n"   // P20 hi->c3
        "mov v12.16b,v2.16b\n .long 0x0020008C\n add v9.2d,v9.2d,v12.2d\n"   // P21 lo->c3
        "mov v13.16b,v2.16b\n .long 0x0020048D\n add v10.2d,v10.2d,v13.2d\n" // P21 hi->c4
        "mov v12.16b,v2.16b\n .long 0x002000AC\n add v10.2d,v10.2d,v12.2d\n" // P22 lo->c4
        "mov v13.16b,v2.16b\n .long 0x002004AD\n add v11.2d,v11.2d,v13.2d\n" // P22 hi->c5
        "movi v17.2d,#0\n"
        "add v18.2d,v6.2d,v17.2d\n  and v6.16b,v18.16b,v16.16b\n  ushr v17.2d,v18.2d,#53\n"
        "add v18.2d,v7.2d,v17.2d\n  and v7.16b,v18.16b,v16.16b\n  ushr v17.2d,v18.2d,#53\n"
        "add v18.2d,v8.2d,v17.2d\n  and v8.16b,v18.16b,v16.16b\n  ushr v17.2d,v18.2d,#53\n"
        "add v18.2d,v9.2d,v17.2d\n  and v9.16b,v18.16b,v16.16b\n  ushr v17.2d,v18.2d,#53\n"
        "add v18.2d,v10.2d,v17.2d\n and v10.16b,v18.16b,v16.16b\n ushr v17.2d,v18.2d,#53\n"
        "add v18.2d,v11.2d,v17.2d\n and v11.16b,v18.16b,v16.16b\n ushr v17.2d,v18.2d,#53\n"
        "mov x0,%2\n"
        "st1 {v6.2d-v9.2d}, [x0], #64\n"
        "st1 {v10.2d-v11.2d}, [x0]\n"
        : : "r"(a),"r"(b),"r"(out),"r"(mask)
        : "memory","x0","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10",
          "v11","v12","v13","v16","v17","v18");
}

static u256 recon53(const uint64_t *out,int lane){
    u128 hi=0,lo=0;
    for(int c=5;c>=0;c--){ uint64_t d=out[2*c+lane]; u128 cr=lo>>75; hi=(hi<<53)|cr; lo=(lo<<53)|d; }
    u256 r={{(uint64_t)lo,(uint64_t)(lo>>64),(uint64_t)hi,(uint64_t)(hi>>64)}};
    return r;
}
static void split53(u128 v,uint64_t *l0,uint64_t *l1,uint64_t *l2){
    *l0=(uint64_t)v&MASK53; *l1=(uint64_t)(v>>53)&MASK53; *l2=(uint64_t)(v>>106)&MASK53;
}

static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }
static inline u128 rnd(void){ return ((u128)lcg()<<64)|lcg(); }
static double now_ns(void){ static mach_timebase_info_data_t tb; if(!tb.denom)mach_timebase_info(&tb);
                            return (double)mach_absolute_time()*tb.numer/tb.denom; }

int main(void){
    uint64_t mask[2]={MASK53,MASK53};

    // validation
    long fails=0,n=1000000;
    for(long t=0;t<n;t++){
        u128 a0=rnd(),a1=rnd(),b0=rnd(),b1=rnd();
        uint64_t A[6],B[6],O[12];
        split53(a0,&A[0],&A[2],&A[4]); split53(a1,&A[1],&A[3],&A[5]);
        split53(b0,&B[0],&B[2],&B[4]); split53(b1,&B[1],&B[3],&B[5]);
        mul53_128(A,B,O,mask);
        if(memcmp(recon53(O,0).w,mul_scalar(a0,b0).w,32)){ if(++fails<=3)printf("  MISMATCH lane0 t=%ld\n",t); }
        if(memcmp(recon53(O,1).w,mul_scalar(a1,b1).w,32)){ if(++fails<=3)printf("  MISMATCH lane1 t=%ld\n",t); }
    }
    printf("MUL53 128x128 multiply: %ld products, %ld mismatches -> %s\n",
           2*n,fails,fails?"FAIL":"BIT-EXACT");
    if(fails) return 1;

    // timing: N resident pairs
    enum { N=2048 };
    static uint64_t A[N][6],B[N][6]; static u128 av0[N],av1[N],bv0[N],bv1[N];
    for(int i=0;i<N;i++){
        av0[i]=rnd();av1[i]=rnd();bv0[i]=rnd();bv1[i]=rnd();
        split53(av0[i],&A[i][0],&A[i][2],&A[i][4]); split53(av1[i],&A[i][1],&A[i][3],&A[i][5]);
        split53(bv0[i],&B[i][0],&B[i][2],&B[i][4]); split53(bv1[i],&B[i][1],&B[i][3],&B[i][5]);
    }
    const int R=8000; const double muls=(double)N*R*2.0;

    double bm=1e30; volatile uint64_t sink=0;
    for(int r=0;r<5;r++){ double t=now_ns(); uint64_t acc=0; uint64_t O[12];
        for(int rr=0;rr<R;rr++) for(int i=0;i<N;i++){ mul53_128(A[i],B[i],O,mask); acc+=O[0]^O[7]; }
        double dt=now_ns()-t; sink^=acc; if(dt<bm)bm=dt; }

    double bs=1e30; volatile uint64_t sinks=0;
    for(int r=0;r<5;r++){ double t=now_ns(); uint64_t acc=0;
        for(int rr=0;rr<R;rr++) for(int i=0;i<N;i++){
            u256 x=mul_scalar(av0[i],bv0[i]),y=mul_scalar(av1[i],bv1[i]); acc+=x.w[0]^y.w[2]; }
        double dt=now_ns()-t; sinks^=acc; if(dt<bs)bs=dt; }

    printf("\nfull 128x128 multiply, end-to-end (best of 5, %g multiplies):\n", muls);
    printf("  scalar u128 : %6.3f ns/mul\n", bs/muls);
    printf("  MUL53       : %6.3f ns/mul   speedup %.3fx %s\n",
           bm/muls, bs/bm, bm<bs?"(MUL53 wins)":"(scalar wins)");
    printf("  [sink %llu %llu]\n",(unsigned long long)sink,(unsigned long long)sinks);
    return 0;
}
