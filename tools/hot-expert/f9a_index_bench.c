/* F9a microbenchmark: head-lane SIMD vs the scalar dot for the DSA
 * indexer's score pass (c/sparse_index.h). CPU only, standalone, no model,
 * no engine, no GPU -- NOT a gate (the gate is f9a_gate_chain.sh's decode
 * tok/s and TTFT ladder against the real engine).
 *
 * Real GLM-5.3 dims throughout: heads=32 (index_n_heads), dim=128
 * (index_head_dim), pool=4 (index_kpool), topk=2048 (index_topk, so
 * wanted=512). Sequence depth 2 048 / 9 216 / 18 432 / 65 536 tokens ->
 * pools = 512 / 2 304 / 4 608 / 16 384 -- at 18 432 that is within a token
 * of the record's own "~4 600 pools" figure for the deep ladder turn. The
 * pooled-key working set is exactly `pools * dim` floats (2.36 MB at
 * 18 432); this bench does not inflate it, and prints it so the number is
 * checked, not assumed.
 *
 * Two axes, matching the engine's own dispatch in
 * coli_sparse_index_score_select: rows=512 exercises the rows-parallel
 * (prefill) branch, rows=1 the "one row, parallel over pools" (decode)
 * branch -- both take the last `rows` positions of the sequence, so the
 * deepest row in each call sees (up to) every pool, same as a real prefill
 * chunk or decode step at that depth.
 *
 * Build:  gcc -O3 -march=native -fopenmp -pthread -I ../../c \
 *             f9a_index_bench.c -o f9a_index_bench -lm
 * Run:    OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
 *             ./f9a_index_bench
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "sparse_index.h"

static unsigned long g_rng = 20260920ULL;
static unsigned long xorshift(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return g_rng;
}
static float randf(void) { return (float)((xorshift() % 1000000) / 1000000.0 * 4.0 - 2.0); }

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* One (sequence, rows) cell: lanes off vs on, ms/call, GMAC/s, memcmp. */
static void bench_one(int sequence, int rows, int reps) {
    const int heads = 32, dim = 128, pool = 4, topk = 2048;
    const int pools = (sequence + pool - 1) / pool;
    const int wanted = topk / pool;
    const int with_tail = 1;
    const int width = coli_sparse_index_width(topk, pool, with_tail);
    const int q_from = sequence - rows > 0 ? sequence - rows : 0;
    const int q_to = sequence;
    const int actual_rows = q_to - q_from;

    float *pool_base = malloc((size_t)pools * dim * sizeof(float));
    float *queries = malloc((size_t)actual_rows * heads * dim * sizeof(float));
    float *head_w = malloc((size_t)actual_rows * heads * sizeof(float));
    unsigned char *complete = malloc((size_t)pools);
    unsigned char *valid = malloc((size_t)sequence);
    int *out_off = malloc((size_t)actual_rows * width * sizeof(int));
    int *out_on = malloc((size_t)actual_rows * width * sizeof(int));
    for (int i = 0; i < pools * dim; i++) pool_base[i] = randf();
    for (int i = 0; i < actual_rows * heads * dim; i++) queries[i] = randf();
    for (int i = 0; i < actual_rows * heads; i++) head_w[i] = randf();
    for (int p = 0; p < pools; p++) complete[p] = ((p + 1) * pool <= sequence) ? 1 : 0;
    for (int i = 0; i < sequence; i++) valid[i] = 1;

    const double total_macs = (double)actual_rows * pools * heads * dim;
    const size_t working_set = (size_t)pools * dim * sizeof(float);

    coli_sparse_index_lanes_set(0);
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        coli_sparse_index_score_select(out_off, pool_base, queries, head_w, complete, valid,
                                       heads, dim, pool, wanted, with_tail, topk, width,
                                       q_from, q_to, 0, pools);
    double t_off = (now_s() - t0) / reps;

    coli_sparse_index_lanes_set(1);
    double t1 = now_s();
    for (int r = 0; r < reps; r++)
        coli_sparse_index_score_select(out_on, pool_base, queries, head_w, complete, valid,
                                       heads, dim, pool, wanted, with_tail, topk, width,
                                       q_from, q_to, 0, pools);
    double t_on = (now_s() - t1) / reps;
    coli_sparse_index_lanes_set(-1);

    int match = memcmp(out_off, out_on, (size_t)actual_rows * width * sizeof(int)) == 0;

    printf("seq=%-6d rows=%-4d pools=%-6d working_set=%.2fMB  "
           "off=%.3fms (%.2f GMAC/s)  on=%.3fms (%.2f GMAC/s)  speedup=%.2fx  memcmp=%s\n",
           sequence, actual_rows, pools, working_set / 1e6,
           t_off * 1e3, total_macs / t_off / 1e9,
           t_on * 1e3, total_macs / t_on / 1e9,
           t_off / t_on, match ? "IDENTICAL" : "DIFFERS");

    free(pool_base); free(queries); free(head_w); free(complete); free(valid);
    free(out_off); free(out_on);
}

int main(void) {
#ifdef _OPENMP
    printf("OMP_NUM_THREADS effective max threads = %d\n", omp_get_max_threads());
#else
    printf("(no OpenMP in this build -- single-threaded)\n");
#endif
#ifdef COLI_INDEX_HEADVEC
    printf("COLI_INDEX_HEADVEC defined -- AVX2/FMA lane kernel active\n");
#else
    printf("COLI_INDEX_HEADVEC NOT defined -- lanes fall back to the portable scalar tail only\n");
#endif
    const int seqs[] = { 2048, 9216, 18432, 65536 };
    const int reps_for[] = { 20, 8, 4, 2 };   /* fewer reps at deeper/slower cells */
    for (size_t i = 0; i < sizeof(seqs) / sizeof(seqs[0]); i++) {
        bench_one(seqs[i], 512, reps_for[i]);
        bench_one(seqs[i], 1, reps_for[i] * 4);
    }
    return 0;
}
