# NEON / MUL53 microbenchmarks.  Apple Silicon (arm64), macOS, clang.
#   make            build all probes
#   make fpipes     build one
#   make clean
CC      = clang
CFLAGS  = -O3
PORTABLE = mul128 ifma52 ifma_radix fpfma fpipes
MUL53    = mul53probe mul53bench mul53mul dualengine   # need Apple-private MUL53
BINS     = $(PORTABLE) $(MUL53)

all: $(BINS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINS) *.s

.PHONY: all clean
