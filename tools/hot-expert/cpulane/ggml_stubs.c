// ggml_stubs.c -- stand-ins for the few ggml.c / ggml-cpu.c symbols that ggml-quants.c and the x86 AVX2
// quant kernels (ggml-cpu/arch/x86/quants.c) reference, so the benchmark links just those two objects and
// not the whole of ggml. None of the three dot kernels measured here (IQ4_XS, IQ3_S, Q6_K) reads these
// tables on this build (F16C: __F16C__ -> _cvtsh_ss), they are filled anyway for safety.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#include "ggml.h"

float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "ggml_abort (stub) %s:%d: ", file, line);
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    abort();
}

// ggml-quants.c needs these for its quantise/validate paths, which this benchmark never calls
size_t ggml_type_size(enum ggml_type t)            { (void) t; ggml_abort(__FILE__, __LINE__, "ggml_type_size stub called"); }
size_t ggml_row_size(enum ggml_type t, int64_t ne) { (void) t; (void) ne; ggml_abort(__FILE__, __LINE__, "ggml_row_size stub called"); }
const char * ggml_type_name(enum ggml_type t)      { (void) t; return "stub"; }

void cpulane_ggml_init(void) {
    for (int i = 0; i < (1 << 16); ++i) ggml_table_f32_f16[i] = _cvtsh_ss((unsigned short) i);
    for (int i = 0; i < 256; ++i) { ggml_table_f32_e8m0_half[i] = ldexpf(0.5f, i - 127); ggml_table_f32_ue4m3[i] = 0.0f; }
}
