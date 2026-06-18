# NEON / MUL53 microbenchmarks.  Apple Silicon (arm64), macOS, clang.
#   make            build all probes
#   make fpipes     build one
#   make clean
CC      = clang
CFLAGS  = -O3
PORTABLE = mul128 ifma52 ifma_radix fpfma fpipes sme_emu sme_probe  # run anywhere
MUL53    = mul53probe mul53bench mul53mul dualengine       # need Apple-private MUL53
SME      = sme_mul sme_min sme_mul_asm                      # RUN on M4+ only
BINS     = $(PORTABLE) $(MUL53) $(SME)

all: $(BINS)

# SME targets need feature flags (compile anywhere; only RUN on M4+)
sme_mul: sme_mul.c
	$(CC) -O3 -march=armv9-a+sme2+sme-i16i64 -o $@ $<
sme_min: sme_min.c
	$(CC) -O3 -march=armv9-a+sme2 -o $@ $<
sme_mul_asm: sme_mul_asm.c
	$(CC) -O3 -march=armv9-a+sme2+sme-i16i64 -o $@ $<

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINS) *.s

.PHONY: all clean
