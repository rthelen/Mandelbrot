#pragma once

#include <stdint.h>
#include <sys/types.h>

int divmnu(uint32_t q[], uint32_t r[],
     const uint32_t u[], const uint32_t v[],
     ssize_t m, ssize_t n);
