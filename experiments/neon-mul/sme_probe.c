// SME ground-truth probe — PURE sysctl dump, ZERO SME code in the binary, so it
// cannot SIGILL anywhere (M1..M5). Tells us what this machine ADVERTISES to
// userspace. The key question: does the M4 expose FEAT_SME / FEAT_SME_I16I64 at
// all? (sme_mul died at *startup* with no output on the M4 — same as a pre-SME
// core — which means the SME/ZA state init faulted, i.e. userspace SME is not
// enabled for our process. This dump confirms whether the features are even
// advertised.)
//
// Build (any Mac): clang -O3 -o sme_probe sme_probe.c     (NO -march needed)
// Run:             ./sme_probe

#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>

static void show(const char *k){
    int v=0; size_t s=sizeof v;
    if (sysctlbyname(k,&v,&s,0,0)==0) printf("  %-40s = %d\n", k, v);
    else                              printf("  %-40s = (absent)\n", k);
}

int main(void){
    char brand[256]; size_t bs=sizeof brand;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &bs, 0, 0)!=0) strcpy(brand,"?");
    printf("CPU: %s\n\n=== SME / SVE feature sysctls ===\n", brand);
    show("hw.optional.arm.FEAT_SME");
    show("hw.optional.arm.FEAT_SME2");
    show("hw.optional.arm.FEAT_SME_I16I64");
    show("hw.optional.arm.FEAT_SME_F64F64");
    show("hw.optional.arm.FEAT_SME_F16F16");
    show("hw.optional.arm.FEAT_SME_B16F32");
    show("hw.optional.arm.FEAT_SVE");
    show("hw.optional.arm.FEAT_SVE2");
    show("hw.optional.AdvSIMD");
    printf("\nInterpretation:\n");
    printf("  FEAT_SME = 1        -> SME advertised; sme_mul startup crash = userspace SME/ZA not enabled by the OS\n");
    printf("  FEAT_SME = 0/absent -> no userspace SME on this core (matrix unit, if any, is Accelerate-only)\n");
    return 0;
}
