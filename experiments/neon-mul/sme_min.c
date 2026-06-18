// Minimal hand-asm SME test: can userspace enter SME mode on this M4 when we
// BYPASS clang's (buggy) SME codegen?  The intrinsic path SIGILLs not because of
// macOS but because Apple clang 21 emits non-streaming `cntd` (SVE; Apple has
// SME but no non-streaming SVE) in the ZA-frame prologue. Raw inline asm has no
// compiler-generated SME prologue, so it isolates the real question: does the OS
// allow smstart / ZA / streaming from a plain process?
//
// No SME C attributes => no clang SME prologue => no stray cntd. Runtime-gated by
// sysctl so it's safe on pre-SME cores (asm is only executed when FEAT_SME=1).
//
// Build (any Mac): clang -O3 -march=armv9-a+sme2 -o sme_min sme_min.c
// Run:             ./sme_min

#include <stdio.h>
#include <stdint.h>
#include <sys/sysctl.h>

static int feat(const char *k){ int v=0; size_t s=sizeof v; return sysctlbyname(k,&v,&s,0,0)==0 && v; }

// Raw SME: enter streaming+ZA, read SVL, touch ZA, exit. If the OS doesn't allow
// userspace SME, the very first instruction (smstart) SIGILLs here.
__attribute__((noinline)) static uint64_t sme_min(void){
    uint64_t svl_bytes = 0;
    __asm__ __volatile__(
        "smstart            \n\t"   // PSTATE.SM=1, PSTATE.ZA=1
        "rdsvl  %0, #1      \n\t"   // streaming vector length, in bytes
        "zero   {za}        \n\t"   // write the ZA tile (proves ZA usable)
        "smstop             \n\t"   // back to normal mode
        : "=r"(svl_bytes) : : "memory"
    );
    return svl_bytes;
}

int main(void){
    if (!feat("hw.optional.arm.FEAT_SME")){
        printf("FEAT_SME=0 — no SME here (expected on M1/M2/M3). Skipping.\n");
        return 0;
    }
    printf("FEAT_SME=1. Executing raw asm: smstart / rdsvl / zero {za} / smstop ...\n");
    fflush(stdout);
    uint64_t b = sme_min();
    printf("SUCCESS — userspace SME runs! SVL = %llu bytes = %llu bits (%llu u16 lanes)\n",
           (unsigned long long)b, (unsigned long long)b*8, (unsigned long long)b/2);
    printf("=> direct hand-asm SME is viable; the intrinsic crash was the clang cntd bug.\n");
    return 0;
}
