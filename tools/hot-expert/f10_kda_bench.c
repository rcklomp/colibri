/* f10_kda_bench.c -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F10, step 0.
 *
 * The question this answers: what bounds the CPU recurrence step
 * (coli_kda_step, c/delta_attention.h) at GLM-5.3's real KDA shapes --
 * arithmetic, OpenMP fork/join, or the state's own memory traffic -- and
 * whether a single layer's state (4 MiB) behaves differently from the
 * 34-layer working set (136 MiB) the record already flagged as sitting
 * just past this box's 128 MiB L3 (ROME-3x7900XTX-2026-09-04.md, "A cliff
 * worth recording").
 *
 * This links delta_attention.h DIRECTLY -- the same header c/glm53.c
 * includes -- so what is timed is the shipped kernel, not a model of it.
 * No engine, no Vulkan, no model weights: state/window/conv/qkv/gate/beta
 * are synthetic (fixed-seed xorshift32), and outputs are meaningless --
 * only their finiteness and a checksum (to defeat dead-code elimination)
 * are checked.
 *
 * What this file deliberately does NOT measure: the eight KDA projections
 * (kq/kk/kv/kfa/kb/kga/kfb/kgb -- glm53.c's kmv_rows). Reading
 * c/backend_vulkan.c (mv_rows_s -> coli_vk_matmul, g_vk_ready && fmt in
 * {1,4}) shows those run on the GPU in the served engine, tiled shader
 * (qmatmul_tile.spv, 8 rows/workgroup) included, whenever Vulkan is up --
 * which it always is in glm53's serving configuration. Benchmarking that
 * path on the CPU would time a kernel the engine never runs there; the
 * record and this item's write-up describe instead what a GPU-side chain
 * (coli_vk_matmul at real KDA shapes, S=64/128/256/512, VK_PROF-style phase
 * split) would need to measure, and leave it to be written and run as such
 * a chain, not as a CPU probe.
 *
 * Real GLM-5.3 shapes (rig config.json, linear_attn_config + this record):
 * kda_heads=64, kda_hd=128 (k_dim=v_dim), short_conv_kernel_size=4,
 * hidden=4096 (unused here -- that is the projection side). Per layer:
 * state 64*128*128*4B = 4 MiB, window 3*8192*4*4B = 384 KiB. 34 KDA layers
 * per the G12 spec (kda_proj 8192 = 64*128) -> 136 MiB total state.
 *
 * Modes:
 *   single   -- ONE layer's state, reused across N tokens (state should
 *               stay resident past L2, inside a private core's L3 slice
 *               once warm -- the good case).
 *   multilayer -- NLAYER distinct layers' state (136 MiB total), walked in
 *               layer order for every token the way the real forward pass
 *               does it, so a given layer's state is 33 other layers'
 *               traffic away from its last touch -- the cliff case.
 *
 * Thread scaling 1/2/4/8 via omp_set_num_threads(); scratch is resized for
 * each T per coli_kda_scratch_floats()'s own contract (sized under the
 * thread count the call will see).
 *
 *   f10_kda_bench [n_tokens_single] [n_tokens_multi] [nlayers] [seed]
 *
 * Output: one `ROW mode=... threads=... ms_per_token=... gbytes_s=...` line
 * per (mode, thread count); `INFO` lines for machine facts and checksums.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../c/delta_attention.h"

#define HEADS   64
#define KDIM    128
#define VDIM    128
#define KERNEL  4
#define NLAYERS_DEFAULT 34

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static uint32_t S_RND = 424242u;
static uint32_t rndu(void) { S_RND ^= S_RND << 13; S_RND ^= S_RND >> 17; S_RND ^= S_RND << 5; return S_RND; }
static float rnd01(void) { return (float)(rndu() >> 8) / 16777216.0f; }
static float rndsm(void) { return (rnd01() - 0.5f) * 0.2f; }   /* small, plausible activation magnitude */

static const int WIDTH = HEADS * KDIM;          /* 8192 */
static const long STATE_FLOATS  = (long)HEADS * KDIM * VDIM;         /* 1,048,576 */
static const long WINDOW_FLOATS = 3L * HEADS * KDIM * KERNEL;        /* 98,304 */

/* Bytes of state traffic coli_kda_step touches per call: two passes, each a
 * read-modify-write of every [k_dim x v_dim] row per head (state read +
 * state write), so 2 passes x 2 (read+write) x state size. Documented here
 * rather than measured directly (no hardware counters used) so the GB/s
 * figure is falsifiable against this formula, not a black box. */
static double bytes_per_call(void) {
    return 4.0 * (double)STATE_FLOATS * sizeof(float);
}

typedef struct {
    float *state, *window, *conv_w, *alog_unused;
} Layer;

static void layer_init(Layer *l) {
    l->state  = calloc((size_t)STATE_FLOATS, sizeof(float));   /* recurrence starts at 0, as glm53.c does */
    l->window = calloc((size_t)WINDOW_FLOATS, sizeof(float));
    l->conv_w = malloc((size_t)WINDOW_FLOATS * sizeof(float));
    if (!l->state || !l->window || !l->conv_w) { fprintf(stderr, "OOM layer_init\n"); exit(1); }
    for (long i = 0; i < WINDOW_FLOATS; i++) l->conv_w[i] = rndsm();
}
static void layer_free(Layer *l) { free(l->state); free(l->window); free(l->conv_w); }

/* ---------------- single-layer mode: one state reused for N tokens ------- */
static void run_single(int T, int n_tokens, double *out_ms_per_tok, double *out_gbs, double *checksum) {
#ifdef _OPENMP
    omp_set_num_threads(T);
#endif
    Layer l; layer_init(&l);
    int scratch_n = coli_kda_scratch_floats(HEADS, KDIM, VDIM);
    float *scratch = malloc((size_t)scratch_n * sizeof(float));
    float *qkv = malloc((size_t)3 * WIDTH * sizeof(float));
    float *gate = malloc((size_t)WIDTH * sizeof(float));
    float *beta = malloc((size_t)HEADS * sizeof(float));
    float *out = malloc((size_t)HEADS * VDIM * sizeof(float));
    if (!scratch || !qkv || !gate || !beta || !out) { fprintf(stderr, "OOM run_single\n"); exit(1); }

    double sum = 0.0;
    /* warm-up: one call outside the timer so the first-touch page faults
     * for state/window/conv_w/scratch do not land inside the measurement. */
    for (int i = 0; i < 3 * WIDTH; i++) qkv[i] = rndsm();
    for (int i = 0; i < WIDTH; i++) gate[i] = rndsm() - 2.0f;
    for (int h = 0; h < HEADS; h++) beta[h] = rnd01();
    coli_kda_step(out, l.state, l.window, qkv, l.conv_w, gate, beta, HEADS, KDIM, VDIM, KERNEL, 1e-6f, scratch);

    double t0 = now_ms();
    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < 3 * WIDTH; i++) qkv[i] = rndsm();
        for (int i = 0; i < WIDTH; i++) gate[i] = rndsm() - 2.0f;
        for (int h = 0; h < HEADS; h++) beta[h] = rnd01();
        coli_kda_step(out, l.state, l.window, qkv, l.conv_w, gate, beta, HEADS, KDIM, VDIM, KERNEL, 1e-6f, scratch);
        for (int i = 0; i < HEADS * VDIM; i += 997) sum += out[i];   /* defeat DCE, cheap */
    }
    double elapsed = now_ms() - t0;
    *out_ms_per_tok = elapsed / n_tokens;
    *out_gbs = (bytes_per_call() * n_tokens) / (elapsed / 1000.0) / 1e9;
    *checksum = sum;

    for (int i = 0; i < HEADS * VDIM; i++) if (!isfinite(out[i])) { fprintf(stderr, "FATAL: non-finite output, single T=%d\n", T); exit(1); }

    free(scratch); free(qkv); free(gate); free(beta); free(out);
    layer_free(&l);
}

/* ---------------- multilayer mode: NLAYER states, layer order per token -- */
static void run_multilayer(int T, int n_tokens, int nlayers, double *out_ms_per_tok, double *out_gbs, double *checksum) {
#ifdef _OPENMP
    omp_set_num_threads(T);
#endif
    Layer *layers = malloc((size_t)nlayers * sizeof(Layer));
    if (!layers) { fprintf(stderr, "OOM run_multilayer layers\n"); exit(1); }
    for (int i = 0; i < nlayers; i++) layer_init(&layers[i]);

    int scratch_n = coli_kda_scratch_floats(HEADS, KDIM, VDIM);
    float *scratch = malloc((size_t)scratch_n * sizeof(float));
    float *qkv = malloc((size_t)3 * WIDTH * sizeof(float));
    float *gate = malloc((size_t)WIDTH * sizeof(float));
    float *beta = malloc((size_t)HEADS * sizeof(float));
    float *out = malloc((size_t)HEADS * VDIM * sizeof(float));
    if (!scratch || !qkv || !gate || !beta || !out) { fprintf(stderr, "OOM run_multilayer\n"); exit(1); }

    double sum = 0.0;
    for (int i = 0; i < 3 * WIDTH; i++) qkv[i] = rndsm();
    for (int i = 0; i < WIDTH; i++) gate[i] = rndsm() - 2.0f;
    for (int h = 0; h < HEADS; h++) beta[h] = rnd01();
    /* warm-up: touch every layer once outside the timer. */
    for (int i = 0; i < nlayers; i++)
        coli_kda_step(out, layers[i].state, layers[i].window, qkv, layers[i].conv_w,
                      gate, beta, HEADS, KDIM, VDIM, KERNEL, 1e-6f, scratch);

    double t0 = now_ms();
    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < nlayers; i++) {
            for (int j = 0; j < 3 * WIDTH; j++) qkv[j] = rndsm();
            for (int j = 0; j < WIDTH; j++) gate[j] = rndsm() - 2.0f;
            for (int h = 0; h < HEADS; h++) beta[h] = rnd01();
            coli_kda_step(out, layers[i].state, layers[i].window, qkv, layers[i].conv_w,
                          gate, beta, HEADS, KDIM, VDIM, KERNEL, 1e-6f, scratch);
            sum += out[0];
        }
    }
    double elapsed = now_ms() - t0;
    long calls = (long)n_tokens * nlayers;
    *out_ms_per_tok = elapsed / n_tokens;   /* per TOKEN, i.e. per nlayers calls -- matches the record's "ms/token" bucket */
    *out_gbs = (bytes_per_call() * calls) / (elapsed / 1000.0) / 1e9;
    *checksum = sum;

    for (int i = 0; i < HEADS * VDIM; i++) if (!isfinite(out[i])) { fprintf(stderr, "FATAL: non-finite output, multilayer T=%d\n", T); exit(1); }

    free(scratch); free(qkv); free(gate); free(beta); free(out);
    for (int i = 0; i < nlayers; i++) layer_free(&layers[i]);
}

int main(int argc, char **argv) {
    int n_single = argc > 1 ? atoi(argv[1]) : 2048;
    int n_multi  = argc > 2 ? atoi(argv[2]) : 64;
    int nlayers  = argc > 3 ? atoi(argv[3]) : NLAYERS_DEFAULT;
    if (argc > 4) S_RND = (uint32_t)strtoul(argv[4], NULL, 10);

    fprintf(stderr, "INFO shapes heads=%d k_dim=%d v_dim=%d kernel=%d\n", HEADS, KDIM, VDIM, KERNEL);
    fprintf(stderr, "INFO state/layer=%.3f MiB window/layer=%.3f KiB nlayers=%d total_state=%.1f MiB\n",
            STATE_FLOATS * 4.0 / (1024 * 1024), WINDOW_FLOATS * 4.0 / 1024.0, nlayers,
            (double)nlayers * STATE_FLOATS * 4.0 / (1024 * 1024));
    fprintf(stderr, "INFO n_single=%d n_multi=%d bytes_per_call_formula=%.0f (4 * state_bytes)\n",
            n_single, n_multi, bytes_per_call());
#ifdef _OPENMP
    fprintf(stderr, "INFO built with OpenMP, omp_get_max_threads()=%d before any set\n", omp_get_max_threads());
#else
    fprintf(stderr, "INFO built WITHOUT OpenMP -- thread scaling rows will all be T=1\n");
#endif

    int threads[] = {1, 2, 4, 8};
    for (int i = 0; i < 4; i++) {
        int T = threads[i];
        double ms, gbs, chk;
        run_single(T, n_single, &ms, &gbs, &chk);
        printf("ROW mode=single     threads=%d ms_per_token=%.5f gbytes_s=%.2f checksum=%.6f\n", T, ms, gbs, chk);
        fflush(stdout);
    }
    for (int i = 0; i < 4; i++) {
        int T = threads[i];
        double ms, gbs, chk;
        run_multilayer(T, n_multi, nlayers, &ms, &gbs, &chk);
        printf("ROW mode=multilayer threads=%d ms_per_token=%.5f gbytes_s=%.2f checksum=%.6f nlayers=%d\n",
               T, ms, gbs, chk, nlayers);
        fflush(stdout);
    }
    fprintf(stderr, "INFO done\n");
    return 0;
}
