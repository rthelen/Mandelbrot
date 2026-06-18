// Probe + validate Apple's private MUL53 instructions on THIS machine.
// Encodings (TrungNguyen1909, via asahilinux.org):
//   mul53lo.2d Vd, Vm = 0x00200000 | (m<<5) | d   ; Vd = (Vd*Vm) & (2^53-1), per lane
//   mul53hi.2d Vd, Vm = 0x00200400 | (m<<5) | d   ; Vd = (Vd*Vm) >> 53,      per lane
// .2d = two 53-bit lanes in a 128-bit V register. Destructive 2-operand.
//
// SIGILL-guarded: if this core lacks MUL53 (or it's disabled for EL0), report
// cleanly instead of crashing. If present, validate BIT-EXACT vs __uint128_t.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>

typedef unsigned __int128 u128;
#define MASK53 ((1ull<<53)-1)

static sigjmp_buf jb;
static volatile sig_atomic_t trapped=0;
static void onsigill(int s){ (void)s; trapped=1; siglongjmp(jb,1); }

// mul53lo.2d v1,v0 and mul53hi.2d v1,v0 on (a,b) lane pairs -> lo[2], hi[2]
static void mul53(const uint64_t a[2], const uint64_t b[2], uint64_t lo[2], uint64_t hi[2]){
    __asm__ __volatile__(
        "ld1 {v0.2d}, [%2]\n"
        "ld1 {v1.2d}, [%3]\n"
        ".long 0x00200001\n"     // mul53lo.2d v1, v0
        "st1 {v1.2d}, [%0]\n"
        "ld1 {v0.2d}, [%2]\n"
        "ld1 {v1.2d}, [%3]\n"
        ".long 0x00200401\n"     // mul53hi.2d v1, v0
        "st1 {v1.2d}, [%1]\n"
        : : "r"(lo),"r"(hi),"r"(a),"r"(b) : "v0","v1","memory");
}

static int probe(void){
    struct sigaction sa={0}, old;
    sa.sa_handler=onsigill; sigemptyset(&sa.sa_mask);
    sigaction(SIGILL,&sa,&old);
    int present=0;
    if(sigsetjmp(jb,1)==0){
        __asm__ __volatile__("movi v0.16b, #0\n movi v1.16b, #0\n .long 0x00200001\n":::"v0","v1");
        present=1;
    }
    sigaction(SIGILL,&old,NULL);
    return present;
}

static uint64_t S=0x123456789abcdef0ull;
static inline uint64_t lcg(void){ S=S*6364136223846793005ull+1442695040888963407ull; return S; }

int main(void){
    if(!probe()){
        printf("MUL53: NOT available on this core (SIGILL). Likely not present/enabled on this CPU.\n");
        return 2;
    }
    printf("MUL53: PRESENT — instruction executed without fault. Validating...\n");
    long n=2000000, fails=0;
    for(long t=0;t<n;t++){
        uint64_t a[2]={lcg()&MASK53,lcg()&MASK53}, b[2]={lcg()&MASK53,lcg()&MASK53};
        uint64_t lo[2],hi[2]; mul53(a,b,lo,hi);
        for(int k=0;k<2;k++){
            u128 p=(u128)a[k]*b[k];
            if(lo[k]!=(uint64_t)(p&MASK53) || hi[k]!=(uint64_t)(p>>53)){
                if(++fails<=3) printf("  MISMATCH t=%ld lane=%d a=%llu b=%llu\n",
                    t,k,(unsigned long long)a[k],(unsigned long long)b[k]);
            }
        }
    }
    printf("MUL53 validation: %ld lane-products, %ld mismatches -> %s\n",
           2*n, fails, fails?"FAIL":"BIT-EXACT (matches __uint128_t at the 53-bit boundary)");
    return fails?1:0;
}
