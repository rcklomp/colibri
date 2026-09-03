/* Realistic-pattern test: N DIFFERENT experts (different weight buffers, each
 * its own ColiVkTensor*), dispatched in round-robin -- this forces a genuine
 * descriptor rebind + command re-record on every call, exactly like decode
 * time where each token selects a different set of experts. This is the
 * number that actually decides whether a GPU expert tier is worth building,
 * not the same-tensor-resubmitted number (which skips that cost). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "backend_vulkan.h"

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    if (!coli_vk_init(spv)) { fprintf(stderr, "coli_vk_init failed\n"); return 1; }

    int I = 2560, O = 640, gs = 128;
    int nblkI = (I + 127) / 128, nblkO = (O + 127) / 128;
    int N = 32;  /* 32 distinct "experts" -- more than a token's top-10*3, cycled */

    ColiVkTensor **tensors = calloc(N, sizeof(ColiVkTensor *));
    uint8_t **Ws = malloc(N * sizeof(uint8_t *));
    float **Scls = malloc(N * sizeof(float *));
    size_t mb = (size_t)I * O;
    for (int n = 0; n < N; n++) {
        Ws[n] = malloc(mb);
        for (size_t j = 0; j < mb; j++) Ws[n][j] = (uint8_t)((j * 2654435761u + n * 97) & 0xff);
        Scls[n] = malloc((size_t)nblkI * nblkO * sizeof(float));
        for (int j = 0; j < nblkI * nblkO; j++) Scls[n][j] = 0.01f * (1 + (j % 7));
    }
    float *x = malloc(I * sizeof(float));
    for (int i = 0; i < I; i++) x[i] = ((i % 13) - 6) * 0.3f;
    float *y = malloc(O * sizeof(float));

    /* warm: upload all N once */
    for (int n = 0; n < N; n++)
        if (!coli_vk_matmul(&tensors[n], y, x, Ws[n], Scls[n], 8, 1, I, O, gs)) {
            fprintf(stderr, "warm upload %d failed\n", n); return 1;
        }

    int reps = 2000;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        int n = r % N;
        coli_vk_matmul(&tensors[n], y, x, Ws[n], Scls[n], 8, 1, I, O, gs);
    }
    double t1 = now_s();
    printf("GPU, DIFFERENT expert each call (realistic decode pattern):\n");
    printf("  %.4f ms/call over %d reps, cycling %d distinct resident tensors\n",
           (t1 - t0) / reps * 1e3, reps, N);
    return 0;
}
