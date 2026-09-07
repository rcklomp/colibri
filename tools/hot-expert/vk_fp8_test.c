/* Standalone Vulkan FP8 matmul validation: uploads a real expert-shaped e4m3
 * matrix, dispatches through the newly-added fmt=8 shader path, and compares
 * against the already-validated CPU bit-trick decode (fp8bits2.c) for the
 * SAME weights/scales/input. Also times both, cold, for a real GPU-vs-CPU
 * number at the actual expert shapes before any engine integration happens. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "backend_vulkan.h"

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static const float E4M3_LUT[256] = {
0x0.0p+0f,0x1.0p-9f,0x1.0p-8f,0x1.8p-8f,0x1.0p-7f,0x1.4p-7f,0x1.8p-7f,0x1.cp-7f,
0x1.0p-6f,0x1.2p-6f,0x1.4p-6f,0x1.6p-6f,0x1.8p-6f,0x1.ap-6f,0x1.cp-6f,0x1.ep-6f,
0x1.0p-5f,0x1.2p-5f,0x1.4p-5f,0x1.6p-5f,0x1.8p-5f,0x1.ap-5f,0x1.cp-5f,0x1.ep-5f,
0x1.0p-4f,0x1.2p-4f,0x1.4p-4f,0x1.6p-4f,0x1.8p-4f,0x1.ap-4f,0x1.cp-4f,0x1.ep-4f,
0x1.0p-3f,0x1.2p-3f,0x1.4p-3f,0x1.6p-3f,0x1.8p-3f,0x1.ap-3f,0x1.cp-3f,0x1.ep-3f,
0x1.0p-2f,0x1.2p-2f,0x1.4p-2f,0x1.6p-2f,0x1.8p-2f,0x1.ap-2f,0x1.cp-2f,0x1.ep-2f,
0x1.0p-1f,0x1.2p-1f,0x1.4p-1f,0x1.6p-1f,0x1.8p-1f,0x1.ap-1f,0x1.cp-1f,0x1.ep-1f,
0x1.0p+0f,0x1.2p+0f,0x1.4p+0f,0x1.6p+0f,0x1.8p+0f,0x1.ap+0f,0x1.cp+0f,0x1.ep+0f,
0x1.0p+1f,0x1.2p+1f,0x1.4p+1f,0x1.6p+1f,0x1.8p+1f,0x1.ap+1f,0x1.cp+1f,0x1.ep+1f,
0x1.0p+2f,0x1.2p+2f,0x1.4p+2f,0x1.6p+2f,0x1.8p+2f,0x1.ap+2f,0x1.cp+2f,0x1.ep+2f,
0x1.0p+3f,0x1.2p+3f,0x1.4p+3f,0x1.6p+3f,0x1.8p+3f,0x1.ap+3f,0x1.cp+3f,0x1.ep+3f,
0x1.0p+4f,0x1.2p+4f,0x1.4p+4f,0x1.6p+4f,0x1.8p+4f,0x1.ap+4f,0x1.cp+4f,0x1.ep+4f,
0x1.0p+5f,0x1.2p+5f,0x1.4p+5f,0x1.6p+5f,0x1.8p+5f,0x1.ap+5f,0x1.cp+5f,0x1.ep+5f,
0x1.0p+6f,0x1.2p+6f,0x1.4p+6f,0x1.6p+6f,0x1.8p+6f,0x1.ap+6f,0x1.cp+6f,0x1.ep+6f,
0x1.0p+7f,0x1.2p+7f,0x1.4p+7f,0x1.6p+7f,0x1.8p+7f,0x1.ap+7f,0x1.cp+7f,0x1.ep+7f,
0x1.0p+8f,0x1.2p+8f,0x1.4p+8f,0x1.6p+8f,0x1.8p+8f,0x1.ap+8f,0x1.cp+8f,       NAN,
 -0x0.0p+0f,-0x1.0p-9f,-0x1.0p-8f,-0x1.8p-8f,-0x1.0p-7f,-0x1.4p-7f,-0x1.8p-7f,-0x1.cp-7f,
-0x1.0p-6f,-0x1.2p-6f,-0x1.4p-6f,-0x1.6p-6f,-0x1.8p-6f,-0x1.ap-6f,-0x1.cp-6f,-0x1.ep-6f,
-0x1.0p-5f,-0x1.2p-5f,-0x1.4p-5f,-0x1.6p-5f,-0x1.8p-5f,-0x1.ap-5f,-0x1.cp-5f,-0x1.ep-5f,
-0x1.0p-4f,-0x1.2p-4f,-0x1.4p-4f,-0x1.6p-4f,-0x1.8p-4f,-0x1.ap-4f,-0x1.cp-4f,-0x1.ep-4f,
-0x1.0p-3f,-0x1.2p-3f,-0x1.4p-3f,-0x1.6p-3f,-0x1.8p-3f,-0x1.ap-3f,-0x1.cp-3f,-0x1.ep-3f,
-0x1.0p-2f,-0x1.2p-2f,-0x1.4p-2f,-0x1.6p-2f,-0x1.8p-2f,-0x1.ap-2f,-0x1.cp-2f,-0x1.ep-2f,
-0x1.0p-1f,-0x1.2p-1f,-0x1.4p-1f,-0x1.6p-1f,-0x1.8p-1f,-0x1.ap-1f,-0x1.cp-1f,-0x1.ep-1f,
-0x1.0p+0f,-0x1.2p+0f,-0x1.4p+0f,-0x1.6p+0f,-0x1.8p+0f,-0x1.ap+0f,-0x1.cp+0f,-0x1.ep+0f,
-0x1.0p+1f,-0x1.2p+1f,-0x1.4p+1f,-0x1.6p+1f,-0x1.8p+1f,-0x1.ap+1f,-0x1.cp+1f,-0x1.ep+1f,
-0x1.0p+2f,-0x1.2p+2f,-0x1.4p+2f,-0x1.6p+2f,-0x1.8p+2f,-0x1.ap+2f,-0x1.cp+2f,-0x1.ep+2f,
-0x1.0p+3f,-0x1.2p+3f,-0x1.4p+3f,-0x1.6p+3f,-0x1.8p+3f,-0x1.ap+3f,-0x1.cp+3f,-0x1.ep+3f,
-0x1.0p+4f,-0x1.2p+4f,-0x1.4p+4f,-0x1.6p+4f,-0x1.8p+4f,-0x1.ap+4f,-0x1.cp+4f,-0x1.ep+4f,
-0x1.0p+5f,-0x1.2p+5f,-0x1.4p+5f,-0x1.6p+5f,-0x1.8p+5f,-0x1.ap+5f,-0x1.cp+5f,-0x1.ep+5f,
-0x1.0p+6f,-0x1.2p+6f,-0x1.4p+6f,-0x1.6p+6f,-0x1.8p+6f,-0x1.ap+6f,-0x1.cp+6f,-0x1.ep+6f,
-0x1.0p+7f,-0x1.2p+7f,-0x1.4p+7f,-0x1.6p+7f,-0x1.8p+7f,-0x1.ap+7f,-0x1.cp+7f,-0x1.ep+7f,
-0x1.0p+8f,-0x1.2p+8f,-0x1.4p+8f,-0x1.6p+8f,-0x1.8p+8f,-0x1.ap+8f,-0x1.cp+8f,       NAN,
};
static inline float e4m3_decode(uint8_t b) { return E4M3_LUT[b]; }
#define FP8_BLOCK 128
static inline int64_t fp8_nblk(int n) { return ((int64_t)n + FP8_BLOCK - 1) / FP8_BLOCK; }

static void matmul_fp8_ref(float *y, const float *x, const uint8_t *q8, const float *bscale, int I, int O) {
    int64_t nblkI = fp8_nblk(I);
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q8 + (int64_t)o * I;
        int64_t blkO = o / FP8_BLOCK;
        const float *scl = bscale + blkO * nblkI;
        double a = 0;
        for (int64_t bi = 0; bi * FP8_BLOCK < I; bi++) {
            int base = (int)(bi * FP8_BLOCK), blen = FP8_BLOCK;
            if (base + blen > I) blen = I - base;
            float sc = scl[bi], acc = 0;
            for (int i = base; i < base + blen; i++) acc += e4m3_decode(w[i]) * x[i];
            a += (double)acc * sc;
        }
        y[o] = (float)a;
    }
}

static int run_case(int I, int O, int reps) {
    int64_t nblkI = fp8_nblk(I), nblkO = fp8_nblk(O);
    size_t mb = (size_t)I * O;
    uint8_t *W = malloc(mb);
    for (size_t j = 0; j < mb; j++) W[j] = (uint8_t)((j * 2654435761u + 17) & 0xff);
    float *Scl = malloc((size_t)nblkI * nblkO * sizeof(float));
    for (int64_t j = 0; j < nblkI * nblkO; j++) Scl[j] = 0.01f * (1 + (j % 7));
    float *x = malloc((size_t)I * sizeof(float));
    for (int i = 0; i < I; i++) x[i] = ((i % 13) - 6) * 0.3f;
    float *y_cpu = malloc((size_t)O * sizeof(float));
    float *y_gpu = malloc((size_t)O * sizeof(float));

    matmul_fp8_ref(y_cpu, x, W, Scl, I, O);

    ColiVkTensor *t = NULL;
    if (!coli_vk_matmul(&t, y_gpu, x, W, Scl, 8, 1, I, O, 128)) {
        printf("  I=%d O=%d: coli_vk_matmul FAILED\n", I, O);
        return 1;
    }
    double maxabs = 0, maxrel = 0;
    for (int o = 0; o < O; o++) {
        double d = fabs(y_cpu[o] - y_gpu[o]);
        double rel = d / (fabs(y_cpu[o]) + 1e-6);
        if (d > maxabs) maxabs = d;
        if (rel > maxrel) maxrel = rel;
    }
    printf("  I=%-6d O=%-6d  max_abs_diff=%.3e  max_rel_diff=%.3e  %s\n",
           I, O, maxabs, maxrel, maxrel < 1e-3 ? "PASS" : "FAIL (check shader)");

    /* cold-ish timing: repeated calls to the SAME resident tensor (this is
     * what production looks like once an expert is GPU-resident: upload
     * once, dispatch many times) */
    double t0 = now_s();
    for (int r = 0; r < reps; r++) coli_vk_matmul(&t, y_gpu, x, W, Scl, 8, 1, I, O, 128);
    double t1 = now_s();
    printf("  GPU resident-dispatch: %.4f ms/call (%d reps)\n", (t1 - t0) / reps * 1e3, reps);

    free(W); free(Scl); free(x); free(y_cpu); free(y_gpu);
    return maxrel < 1e-3 ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    if (!coli_vk_init(spv)) { fprintf(stderr, "coli_vk_init failed\n"); return 1; }
    printf("Vulkan init OK\n\n");
    int fail = 0;
    fail |= run_case(2560, 640, 500);
    fail |= run_case(640, 2560, 500);
    return fail;
}
