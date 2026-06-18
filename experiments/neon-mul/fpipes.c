// NEON/SIMD pipe-width characterization — portable across M1/M2/M3/M4/M5.
//
// Measures effective issue width (ops/cycle) for several SIMD ops using 28
// independent dependency chains (>> latency x pipes, so throughput-bound).
// ops/cycle ~= number of pipes that can issue that op.
//
// Clock is MEASURED per-machine (a dependent scalar-add chain runs at exactly
// 1 add/cycle = the core clock), so ops/cycle is correct regardless of the very
// different P-core frequencies across M1..M5. ns/op is clock-independent too.
//
// Build:  clang -O3 -o fpipes fpipes.c   (Apple Silicon / macOS)
// Run on a P-core; results are for the core it lands on.

#include <stdint.h>
#include <stdio.h>
#include <mach/mach_time.h>
static double now_ns(void){ static mach_timebase_info_data_t tb; if(!tb.denom)mach_timebase_info(&tb);
                            return (double)mach_absolute_time()*tb.numer/tb.denom; }

#define A8  "add x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\nadd x0,x0,#1\n"
#define A64 A8 A8 A8 A8 A8 A8 A8 A8
// 64 dependent adds/iter at 1/cycle -> measures the core clock.
static void clk(long R){ __asm__ __volatile__("mov x0,#0\nmov x1,%0\n9:\n" A64 "subs x1,x1,#1\nb.ne 9b\n":: "r"(R):"x0","x1"); }

#define ZERO28 \
    "movi v4.2d,#0\nmovi v5.2d,#0\nmovi v6.2d,#0\nmovi v7.2d,#0\nmovi v8.2d,#0\nmovi v9.2d,#0\nmovi v10.2d,#0\n" \
    "movi v11.2d,#0\nmovi v12.2d,#0\nmovi v13.2d,#0\nmovi v14.2d,#0\nmovi v15.2d,#0\nmovi v16.2d,#0\nmovi v17.2d,#0\n" \
    "movi v18.2d,#0\nmovi v19.2d,#0\nmovi v20.2d,#0\nmovi v21.2d,#0\nmovi v22.2d,#0\nmovi v23.2d,#0\nmovi v24.2d,#0\n" \
    "movi v25.2d,#0\nmovi v26.2d,#0\nmovi v27.2d,#0\nmovi v28.2d,#0\nmovi v29.2d,#0\nmovi v30.2d,#0\nmovi v31.2d,#0\n"
#define ALLV "v0","v1","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17",\
    "v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31"
// emit OP for vD = v4..v31 ; ARG is the trailing operand list (uses the dest too)
#define R28(OP,SUF) \
    OP" v4."SUF"\n"OP" v5."SUF"\n"OP" v6."SUF"\n"OP" v7."SUF"\n"OP" v8."SUF"\n"OP" v9."SUF"\n"OP" v10."SUF"\n" \
    OP" v11."SUF"\n"OP" v12."SUF"\n"OP" v13."SUF"\n"OP" v14."SUF"\n"OP" v15."SUF"\n"OP" v16."SUF"\n"OP" v17."SUF"\n" \
    OP" v18."SUF"\n"OP" v19."SUF"\n"OP" v20."SUF"\n"OP" v21."SUF"\n"OP" v22."SUF"\n"OP" v23."SUF"\n"OP" v24."SUF"\n" \
    OP" v25."SUF"\n"OP" v26."SUF"\n"OP" v27."SUF"\n"OP" v28."SUF"\n"OP" v29."SUF"\n"OP" v30."SUF"\n"OP" v31."SUF"\n"

// fmla vD.2d, v0.2d, v1.2d  (FP64 fused multiply-add)
static void b_fmla(long R){ __asm__ __volatile__("fmov v0.2d,#1.0\nfmov v1.2d,#1.0\n" ZERO28 "mov x1,%0\n1:\n"
    R28("fmla","2d, v0.2d, v1.2d") "subs x1,x1,#1\nb.ne 1b\n":: "r"(R):"x1",ALLV); }
// fadd vD.2d, vD.2d, v0.2d  (FP64 add)
static void b_fadd(long R){ __asm__ __volatile__("fmov v0.2d,#1.0\n" ZERO28 "mov x1,%0\n2:\n"
    R28("fadd","2d, v0.2d, v0.2d") "subs x1,x1,#1\nb.ne 2b\n":: "r"(R):"x1",ALLV); }
// mul vD.4s, vD.4s, v0.4s  (int32 vector multiply) -> the INTEGER-multiply pipe width
static void b_mul(long R){ __asm__ __volatile__("movi v0.4s,#1\n" ZERO28 "mov x1,%0\n3:\n"
    R28("mul","4s, v0.4s, v0.4s") "subs x1,x1,#1\nb.ne 3b\n":: "r"(R):"x1",ALLV); }
// add vD.2d, vD.2d, v0.2d  (int64 vector add)
static void b_add(long R){ __asm__ __volatile__("movi v0.2d,#0\n" ZERO28 "mov x1,%0\n4:\n"
    R28("add","2d, v0.2d, v0.2d") "subs x1,x1,#1\nb.ne 4b\n":: "r"(R):"x1",ALLV); }

static double best(void(*f)(long), long R, double per){
    double b=1e30; for(int r=0;r<6;r++){ double t=now_ns(); f(R); double d=now_ns()-t; if(d<b)b=d; }
    return b/(per*R);   // ns per op
}
int main(void){
    long R=3000000;
    // clock: 64 dependent adds/iter at 1/cycle
    double cb=1e30; for(int r=0;r<6;r++){ double t=now_ns(); clk(R); double d=now_ns()-t; if(d<cb)cb=d; }
    double ghz=(64.0*R)/cb;
    double f=best(b_fmla,R,28), a=best(b_fadd,R,28), m=best(b_mul,R,28), ia=best(b_add,R,28);
    printf("=== NEON/SIMD pipe characterization ===\n");
    printf("measured P-core clock: %.2f GHz\n\n", ghz);
    printf("  %-16s %9s  %10s\n","op","ns/op","ops/cycle");
    printf("  %-16s %9.3f  %10.2f\n","FMLA .2d (FP64)",   f, 1.0/(f*ghz));
    printf("  %-16s %9.3f  %10.2f\n","FADD .2d (FP64)",   a, 1.0/(a*ghz));
    printf("  %-16s %9.3f  %10.2f\n","MUL  .4s (int32)",  m, 1.0/(m*ghz));
    printf("  %-16s %9.3f  %10.2f\n","ADD  .2d (int64)",  ia,1.0/(ia*ghz));
    printf("\nops/cycle ~= number of pipes that can issue that op.\n");
    printf("FMLA/FADD = FP pipe width; MUL.4s = integer-multiply pipe width (governs MUL53/bignum).\n");
    return 0;
}
