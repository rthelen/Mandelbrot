#pragma once

#include <cassert>
#include <iostream>

extern void debugger();

#if defined(DEBUG) || defined(SAFE)
#define ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "Assertion failed: " << message << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::abort(); \
    }
#else
#define ASSERT(condition, message)
#endif

#if defined(DEBUG) && DEBUG == 1
    #define DBG_PRINT(fmt, ...) printf(fmt, __VA_ARGS__)
    #define VDBG_PRINT(fmt, ...) do { if (verbose) { printf(fmt, __VA_ARGS__); } } while (0)
#else
    #define DBG_PRINT(fmt, ...) 
    #define VDBG_PRINT(fmt, ...) 
#endif
