// qerr_stubs.c -- the pieces of ggml-cpu.c that the quant kernels reference (lookup tables), so that qerr links
// ggml.c + ggml-quants.c + ggml-cpu/arch/x86/quants.c + gguf.cpp without the rest of the CPU backend.
// The three tables are filled with the same formulas ggml_cpu_init() uses (ggml-cpu.c:3875-3895).
#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"

float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

void qerr_ggml_init(void) {
    for (int i = 0; i < (1 << 16); ++i) ggml_table_f32_f16[i] = _cvtsh_ss((unsigned short) i);
    for (int i = 0; i < 256; ++i) {
        ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
        ggml_table_f32_ue4m3[i]     = ggml_ue4m3_to_fp32(i);
    }
}

// ggml.c / gguf.cpp reference a few ggml-backend entry points (graph reset, tensor memset, writing a GGUF out) that
// qerr never reaches: it only reads a GGUF with no_alloc = true. Stubs so that ggml-backend*.cpp is not linked.
void ggml_backend_tensor_get(const struct ggml_tensor * t, void * d, size_t o, size_t n) { (void) t; (void) d; (void) o; (void) n; ggml_abort(__FILE__, __LINE__, "stub ggml_backend_tensor_get"); }
void ggml_backend_tensor_set(struct ggml_tensor * t, const void * d, size_t o, size_t n) { (void) t; (void) d; (void) o; (void) n; ggml_abort(__FILE__, __LINE__, "stub ggml_backend_tensor_set"); }
void ggml_backend_tensor_memset(struct ggml_tensor * t, uint8_t v, size_t o, size_t n) { (void) t; (void) v; (void) o; (void) n; ggml_abort(__FILE__, __LINE__, "stub ggml_backend_tensor_memset"); }
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t b) { (void) b; ggml_abort(__FILE__, __LINE__, "stub ggml_backend_buffer_get_usage"); return GGML_BACKEND_BUFFER_USAGE_ANY; }
