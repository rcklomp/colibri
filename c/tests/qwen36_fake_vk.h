/* qwen36_fake_vk.h -- fake Vulkan backend for the qwen36 Vulkan tier's tests.
 *
 * Mirrors tests/qwen36_fake_cuda.h's shape: defines every coli_vk_* symbol
 * qwen36_tier_vk.c links against (signatures from backend_vulkan.h) and
 * RECORDS what it receives, so a test can assert on real upload/issue
 * traffic across dev2/dev3 without a GPU, a Vulkan loader or the SPIR-V
 * shaders. A test that only checked "qt_init returns 1" would pass even
 * with the multi-device planner fully broken (exactly what #1331/#1339
 * showed on the CUDA side).
 *
 * Settable hooks beyond plain recording:
 *   fake_vk_dev2_avail / fake_vk_dev3_avail  - coli_vk_init_dev2/3's success
 *   fake_vk_budget2_gb / fake_vk_used2_gb    - coli_vk_mem_budget2's report
 *   fake_vk_budget3_gb / fake_vk_used3_gb    - coli_vk_mem_budget3's report
 *   fake_vk_issue_hook  - called by BOTH coli_vk_expert_group_issue2/3, with
 *                         which suffix issued (2 or 3), the row count and the
 *                         input pointer; its return value is what issue
 *                         returns. NULL (the default) reproduces the old
 *                         always-0 stub.
 *   fake_vk_matmul_ok0  - F3 step 2a (trunk placement): coli_vk_tensor_ensure /
 *                         coli_vk_matmul WITHOUT a device suffix are dev0's own
 *                         API (backend_vulkan.h), used by qt_lmhead_init/matmul
 *                         and qt_dnproj_init/matmul when Q36_VK_TRUNK=1.
 *                         Recorded the same way as
 *                         dev2/dev3's uploads (fake_vk_uploads0 etc); set to 0
 *                         to make coli_vk_matmul fail so a test can check the
 *                         GPU-failure fallback path. */
#ifndef QWEN36_FAKE_VK_H
#define QWEN36_FAKE_VK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../backend_vulkan.h"

struct ColiVkTensor { int fmt, I, O, gs, dev; const void *w; };

static int fake_vk_uploads2, fake_vk_uploads3;
static int fake_vk_last_fmt2 = -1, fake_vk_last_fmt3 = -1;
static size_t fake_vk_last_bytes2, fake_vk_last_bytes3;
static unsigned char fake_vk_captured2[4096], fake_vk_captured3[4096];
static size_t fake_vk_captured2_len, fake_vk_captured3_len;

static int fake_vk_dev2_avail = 1, fake_vk_dev3_avail = 1;
static double fake_vk_budget2_gb = 1.0, fake_vk_used2_gb = 0.0;
static double fake_vk_budget3_gb = 1.0, fake_vk_used3_gb = 0.0;
static int (*fake_vk_issue_hook)(int suffix, int count, const float *x) = NULL;

int coli_vk_init(const char *spv_path) { (void)spv_path; return 1; }
void coli_vk_shutdown(void) {}
int coli_vk_available(void) { return 1; }
void coli_vk_mem_info(size_t *u, size_t *c) { if (u) *u = 0; if (c) *c = 0; }
void coli_vk_alloc_priority(float p) { (void)p; }
int coli_vk_mem_budget(double *u, double *b) { if (u) *u = 0; if (b) *b = 0; return 0; }
double coli_vk_ballast_gb(double gb) { return gb; }

int coli_vk_init_dev2(const char *spv_path, int devidx) {
    (void)spv_path; (void)devidx; return fake_vk_dev2_avail;
}
int coli_vk_dev2_available(void) { return fake_vk_dev2_avail; }
int coli_vk_mem_budget2(double *u, double *b) {
    if (u) *u = fake_vk_used2_gb; if (b) *b = fake_vk_budget2_gb; return 1;
}
static int upload_common(ColiVkTensor **t, const void *w, int fmt, int I, int O, int gs, int dev,
                         int *uploads, int *last_fmt, size_t *last_bytes,
                         unsigned char *captured, size_t *captured_len) {
    ColiVkTensor *n = (ColiVkTensor *)calloc(1, sizeof *n);
    n->fmt = fmt; n->I = I; n->O = O; n->gs = gs; n->dev = dev; n->w = w;
    *t = n;
    (*uploads)++;
    *last_fmt = fmt;
    *last_bytes = (size_t)I * O / (fmt == 1 ? 1 : 2);   /* fmt=1 int8 row-wise, fmt=4 packed int4 */
    if (*uploads == 1) {
        *captured_len = *last_bytes < 4096 ? *last_bytes : 4096;
        memcpy(captured, w, *captured_len);
    }
    return 1;
}
int coli_vk_tensor_ensure2(ColiVkTensor **t, const void *w, const float *sc, int fmt, int I, int O, int gs) {
    (void)sc;
    return upload_common(t, w, fmt, I, O, gs, 2, &fake_vk_uploads2, &fake_vk_last_fmt2,
                         &fake_vk_last_bytes2, fake_vk_captured2, &fake_vk_captured2_len);
}
int coli_vk_expert_group_issue2(ColiVkTensor *const *g, ColiVkTensor *const *u, ColiVkTensor *const *d,
                                const int *rows, int count, const float *x) {
    (void)g; (void)u; (void)d; (void)rows;
    if (fake_vk_issue_hook) return fake_vk_issue_hook(2, count, x);
    return 0;
}
int coli_vk_expert_group_take2(float *y) { (void)y; return 1; }
int coli_vk_expert_group2(ColiVkTensor *const *g, ColiVkTensor *const *u, ColiVkTensor *const *d,
                          const int *rows, int count, float *y, const float *x) {
    (void)g; (void)u; (void)d; (void)rows; (void)count; (void)y; (void)x; return 0;
}

int coli_vk_init_dev3(const char *spv_path, int devidx) {
    (void)spv_path; (void)devidx; return fake_vk_dev3_avail;
}
int coli_vk_dev3_available(void) { return fake_vk_dev3_avail; }
int coli_vk_mem_budget3(double *u, double *b) {
    if (u) *u = fake_vk_used3_gb; if (b) *b = fake_vk_budget3_gb; return 1;
}
int coli_vk_tensor_ensure3(ColiVkTensor **t, const void *w, const float *sc, int fmt, int I, int O, int gs) {
    (void)sc;
    return upload_common(t, w, fmt, I, O, gs, 3, &fake_vk_uploads3, &fake_vk_last_fmt3,
                         &fake_vk_last_bytes3, fake_vk_captured3, &fake_vk_captured3_len);
}
int coli_vk_expert_group_issue3(ColiVkTensor *const *g, ColiVkTensor *const *u, ColiVkTensor *const *d,
                                const int *rows, int count, const float *x) {
    (void)g; (void)u; (void)d; (void)rows;
    if (fake_vk_issue_hook) return fake_vk_issue_hook(3, count, x);
    return 0;
}
int coli_vk_expert_group_take3(float *y) { (void)y; return 1; }
int coli_vk_expert_group3(ColiVkTensor *const *g, ColiVkTensor *const *u, ColiVkTensor *const *d,
                          const int *rows, int count, float *y, const float *x) {
    (void)g; (void)u; (void)d; (void)rows; (void)count; (void)y; (void)x; return 0;
}

void coli_vk_tensor_free(ColiVkTensor *t) { free(t); }
size_t coli_vk_tensor_bytes(const ColiVkTensor *t) { return t ? (size_t)t->I * t->O : 0; }
int coli_vk_tensor_dev(const ColiVkTensor *t) { return t ? t->dev : -1; }

/* dev0 (unsuffixed): the trunk-placement API (F3 step 2a). Recorded the same
 * way as dev2/dev3's coli_vk_tensor_ensure2/3 above; coli_vk_matmul does no
 * real arithmetic (none of these tests check computed numerics -- that is
 * the KL chain's job, on real hardware) but mirrors upload_tensor's own
 * early-return shape check in backend_vulkan.c so a second call with
 * weights=NULL (the cached-tensor path qt_lmhead_matmul/qt_dnproj_matmul
 * use) behaves like the real backend instead of crashing on a NULL deref. */
static int fake_vk_uploads0;
static int fake_vk_last_fmt0 = -1;
static size_t fake_vk_last_bytes0;
static unsigned char fake_vk_captured0[4096];
static size_t fake_vk_captured0_len;
static int fake_vk_matmul_calls0;
static int fake_vk_matmul_ok0 = 1;
static int fake_vk_last_S0, fake_vk_last_I0, fake_vk_last_O0;

int coli_vk_tensor_ensure(ColiVkTensor **t, const void *w, const float *sc, int fmt, int I, int O, int gs) {
    (void)sc;
    return upload_common(t, w, fmt, I, O, gs, 0, &fake_vk_uploads0, &fake_vk_last_fmt0,
                         &fake_vk_last_bytes0, fake_vk_captured0, &fake_vk_captured0_len);
}
int coli_vk_matmul(ColiVkTensor **t, float *y, const float *x,
                   const void *w, const float *sc, int fmt, int S, int I, int O, int gs) {
    (void)x;
    if (*t) {
        ColiVkTensor *e = *t;
        if (e->fmt != fmt || e->I != I || e->O != O) return 0;
    } else if (!coli_vk_tensor_ensure(t, w, sc, fmt, I, O, gs)) {
        return 0;
    }
    fake_vk_matmul_calls0++;
    fake_vk_last_S0 = S; fake_vk_last_I0 = I; fake_vk_last_O0 = O;
    if (!fake_vk_matmul_ok0) return 0;
    if (y) for (int i = 0; i < S * O; i++) y[i] = 0.0f;
    return 1;
}

#endif /* QWEN36_FAKE_VK_H */
