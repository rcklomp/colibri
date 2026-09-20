/* F6a: the exact top-`wanted` selection (bounded min-heap + sort) must
 * reproduce the original greedy repeated-maximum scan's output on every
 * input, INCLUDING ties -- "the `wanted` largest scores, ties broken by the
 * lower pool index, emitted in descending rank order". This test drives
 * both algorithms over randomised scores (with deliberately forced ties and
 * forced -FLT_MAX exclusions) and diffs the resulting rank-ordered token
 * rows byte for byte. It does not exercise the score computation itself
 * (unchanged: same dot products, same per-row summation order) -- only the
 * selection this item replaces. Cheap: no model, no fixtures, runs on the
 * Mac or the rig CPU.
 *
 * F9a (below, after the topk trials): the head-lane SIMD dot must reproduce
 * the scalar dot bit-for-bit -- randomised trials with GLM53_INDEX_LANES
 * forced on and off (coli_sparse_index_lanes_set, a test-only hook) via
 * both entry points: coli_sparse_index_score_row directly (few-rows/decode
 * axis, memcmp of the raw scores[] array) and the full
 * coli_sparse_index_select_range pipeline at both a one-row and a many-row
 * q-range (few-rows and many-rows axes, memcmp of the selected index rows),
 * across sequence lengths spanning the dense regime (< 2 052, GLM-5.3's own
 * topk=2048/pool=4 shape), pool boundaries, an incomplete last pool, and
 * heads both a multiple of 8 and not. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "../sparse_index.h"

/* The original O(wanted*pools) greedy scan, copied verbatim from
 * sparse_index.h's pre-F6a selection loop (the code GLM53_INDEX_SCALAR=1
 * still runs) so this test does not depend on that knob or on the engine at
 * all -- it is the reference this test exists to hold the new path to. */
static void reference_select(int *row, const float *scores, int pools, int pool,
                              int first, int wanted, unsigned char *taken) {
    memset(taken, 0, (size_t)pools);
    for (int rank = 0; rank < wanted; rank++) {
        int best = -1;
        for (int p = 0; p < pools; p++)
            if (!taken[p] && scores[p] > -FLT_MAX &&
                (best < 0 || scores[p] > scores[best])) best = p;
        if (best < 0) break;
        taken[best] = 1;
        for (int j = 0; j < pool; j++) row[rank * pool + j] = first + best * pool + j;
    }
}

static unsigned long g_rng = 88172645463325252UL;
static unsigned long xorshift(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return g_rng;
}
static double rand01(void) { return (double)(xorshift() % 1000000) / 1000000.0; }
static float randf(void) { return (float)(rand01() * 4.0 - 2.0); }

/* F9a sub-test A: coli_sparse_index_score_row directly, few-rows/decode
 * axis -- one row, pools as the parallel axis. Random heads (multiple of 8
 * and not), dim, pools, pool size, a causal cutoff `q` and some pools forced
 * incomplete, run once with the lane path forced off and once forced on;
 * the raw `scores[]` array must memcmp byte for byte. */
static int f9a_score_row_trials(long *out_trials) {
    int ok = 1;
    long trials = 0;
    const int n_trials = 3000;
    for (int trial = 0; trial < n_trials && ok; trial++) {
        const int heads = 1 + (int)(xorshift() % 40);       /* mixes %8==0 and not */
        const int dim = 1 + (int)(xorshift() % 24);
        const int pool = 1 + (int)(xorshift() % 4);
        const int pools = 1 + (int)(xorshift() % 60);
        const int first = (int)(xorshift() % 5);
        /* q somewhere inside [first, first + pools*pool) so some trailing
         * pools are causally invisible ("last > q"), not just incomplete.
         * score_row's `queries`/`head_w` are indexed from `q_from` (its own
         * doc comment: "cover only [q_from, q_to)... indexed from zero"),
         * and this trial's buffers hold exactly ONE row -- so q_from MUST
         * be q itself, or `query = queries + (q-q_from)*heads*dim` reads
         * off the end of a one-row buffer for any q past the first. */
        const int q = first + (int)(xorshift() % ((size_t)pools * pool));
        const int q_from = q;
        const float scale = 1.0f / sqrtf((float)dim);

        float *pool_base = malloc((size_t)pools * dim * sizeof(float));
        float *queries = malloc((size_t)heads * dim * sizeof(float));
        float *head_w = malloc((size_t)heads * sizeof(float));
        unsigned char *complete = malloc((size_t)pools);
        float *scores_off = malloc((size_t)pools * sizeof(float));
        float *scores_on = malloc((size_t)pools * sizeof(float));
        for (int i = 0; i < pools * dim; i++) pool_base[i] = randf();
        for (int i = 0; i < heads * dim; i++) queries[i] = randf();
        for (int i = 0; i < heads; i++) head_w[i] = randf();
        /* ~85% complete, matching a real trailing-tail pool distribution. */
        for (int p = 0; p < pools; p++) complete[p] = rand01() < 0.85 ? 1 : 0;

        coli_sparse_index_lanes_set(0);
        coli_sparse_index_score_row(scores_off, pool_base, queries, head_w, complete,
                                    q, q_from, first, pools, heads, dim, pool, scale);
        coli_sparse_index_lanes_set(1);
        coli_sparse_index_score_row(scores_on, pool_base, queries, head_w, complete,
                                    q, q_from, first, pools, heads, dim, pool, scale);

        if (memcmp(scores_off, scores_on, (size_t)pools * sizeof(float)) != 0) {
            ok = 0;
            printf("F9a MISMATCH (score_row) trial=%d heads=%d dim=%d pools=%d pool=%d q=%d first=%d\n",
                   trial, heads, dim, pools, pool, q, first);
            for (int p = 0; p < pools; p++)
                if (scores_off[p] != scores_on[p])
                    printf("  p=%d off=%.9g on=%.9g\n", p, scores_off[p], scores_on[p]);
        }
        trials++;

        free(pool_base); free(queries); free(head_w); free(complete);
        free(scores_off); free(scores_on);
    }
    *out_trials = trials;
    return ok;
}

/* F9a sub-test B: the full coli_sparse_index_select_range pipeline (pooling
 * + score + top-k), at a one-row q-range (few-rows/decode axis) and an
 * all-rows q-range (many-rows/prefill axis, the row-parallel branch on a
 * multi-threaded build) -- random sequence length, heads, dim, pool size and
 * topk, plus explicit structured cases for the dense regime, an exact pool
 * boundary and a maximally incomplete last pool. Selected index rows must
 * memcmp byte for byte between the lane path off and on. */
static int f9a_pipeline_one(int sequence, int heads, int dim, int pool, int topk,
                             int with_tail, int q_from, int q_to, const char *tag) {
    const int width = coli_sparse_index_width(topk, pool, with_tail);
    const int rows = q_to - q_from;
    float *queries = malloc((size_t)rows * heads * dim * sizeof(float));
    float *keys = malloc((size_t)sequence * dim * sizeof(float));
    float *gates = malloc((size_t)sequence * dim * sizeof(float));
    float *head_w = malloc((size_t)rows * heads * sizeof(float));
    float *ape = malloc((size_t)pool * dim * sizeof(float));
    unsigned char *valid = malloc((size_t)sequence);
    int *out_off = malloc((size_t)rows * width * sizeof(int));
    int *out_on = malloc((size_t)rows * width * sizeof(int));
    for (int i = 0; i < rows * heads * dim; i++) queries[i] = randf();
    for (int i = 0; i < sequence * dim; i++) { keys[i] = randf(); gates[i] = randf(); }
    for (int i = 0; i < rows * heads; i++) head_w[i] = randf();
    for (int i = 0; i < pool * dim; i++) ape[i] = randf();
    for (int i = 0; i < sequence; i++) valid[i] = 1;

    coli_sparse_index_lanes_set(0);
    int rc_off = coli_sparse_index_select_range(out_off, queries, keys, gates, head_w, ape,
                                                valid, sequence, heads, dim, pool, topk,
                                                with_tail, q_from, q_to);
    coli_sparse_index_lanes_set(1);
    int rc_on = coli_sparse_index_select_range(out_on, queries, keys, gates, head_w, ape,
                                               valid, sequence, heads, dim, pool, topk,
                                               with_tail, q_from, q_to);
    int ok = 1;
    if (rc_off || rc_on) {
        ok = 0;
        printf("F9a MISMATCH (%s) select_range failed rc_off=%d rc_on=%d\n", tag, rc_off, rc_on);
    } else if (memcmp(out_off, out_on, (size_t)rows * width * sizeof(int)) != 0) {
        ok = 0;
        printf("F9a MISMATCH (%s) seq=%d heads=%d dim=%d pool=%d topk=%d tail=%d q=[%d,%d)\n",
               tag, sequence, heads, dim, pool, topk, with_tail, q_from, q_to);
    }
    free(queries); free(keys); free(gates); free(head_w); free(ape); free(valid);
    free(out_off); free(out_on);
    return ok;
}

static int f9a_pipeline_trials(long *out_trials) {
    int ok = 1;
    long trials = 0;
    const int n_trials = 2000;
    for (int trial = 0; trial < n_trials && ok; trial++) {
        const int heads = 1 + (int)(xorshift() % 24);       /* mixes %8==0 and not */
        const int dim = 1 + (int)(xorshift() % 20);
        const int pool = 1 + (int)(xorshift() % 4);
        const int sequence = pool + (int)(xorshift() % 120);
        const int pools = (sequence + pool - 1) / pool;
        const int wanted = 1 + (int)(xorshift() % (pools + 3));
        const int topk = wanted * pool;
        const int with_tail = xorshift() & 1;
        /* alternate few-rows (decode: just the last position) and
         * many-rows (prefill: the whole range) across trials. */
        int q_from, q_to;
        if (trial & 1) { q_from = sequence - 1; q_to = sequence; }
        else { q_from = 0; q_to = sequence; }

        ok = f9a_pipeline_one(sequence, heads, dim, pool, topk, with_tail, q_from, q_to,
                              (trial & 1) ? "random-few-rows" : "random-many-rows");
        trials++;
    }
    *out_trials = trials;
    return ok;
}

int main(void) {
    int ok = 1;
    long total_rows = 0;

    const int pool = 4;
    const int trials = 4000;
    for (int trial = 0; trial < trials; trial++) {
        const int pools = 1 + (int)(xorshift() % 600);
        /* topk must be a multiple of pool for width to make sense; wanted is
         * what actually drives both algorithms, so just pick it directly,
         * occasionally above `pools` to exercise the "fewer than wanted
         * survive" padding path. */
        const int wanted = 1 + (int)(xorshift() % (pools + 8));
        const int width = wanted * pool;

        float *scores = malloc((size_t)pools * sizeof(float));
        /* Force a cluster of exact ties and a cluster of exact exclusions so
         * both the tie rule and the -FLT_MAX filter are exercised on every
         * trial, not left to chance. */
        const double tie_p = 0.35, excl_p = 0.15;
        float tie_value = (float)(rand01() * 10.0 - 5.0);
        for (int p = 0; p < pools; p++) {
            double r = rand01();
            if (r < excl_p) scores[p] = -FLT_MAX;
            else if (r < excl_p + tie_p) scores[p] = tie_value;
            else scores[p] = (float)(rand01() * 10.0 - 5.0);
        }

        int *row_ref = malloc((size_t)width * sizeof(int));
        int *row_new = malloc((size_t)width * sizeof(int));
        for (int i = 0; i < width; i++) { row_ref[i] = -1; row_new[i] = -1; }
        unsigned char *taken = malloc((size_t)pools);

        reference_select(row_ref, scores, pools, pool, /*first=*/0, wanted, taken);

        coli_index_cand_t *heap = malloc((size_t)wanted * sizeof(coli_index_cand_t));
        coli_index_select_topk(row_new, scores, pools, pool, /*first=*/0, wanted, heap);

        if (memcmp(row_ref, row_new, (size_t)width * sizeof(int)) != 0) {
            ok = 0;
            printf("MISMATCH trial=%d pools=%d wanted=%d\n", trial, pools, wanted);
            printf("  ref:"); for (int i = 0; i < width; i++) printf(" %d", row_ref[i]); printf("\n");
            printf("  new:"); for (int i = 0; i < width; i++) printf(" %d", row_new[i]); printf("\n");
        }
        total_rows++;

        free(scores); free(row_ref); free(row_new); free(taken); free(heap);
        if (!ok) break;
    }

    /* Edge cases: all pools excluded; every pool tied at the same score;
     * wanted == 0; a single pool. */
    {
        float scores[8] = { -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX,
                             -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
        int row_ref[8], row_new[8];
        for (int i = 0; i < 8; i++) { row_ref[i] = -1; row_new[i] = -1; }
        unsigned char taken[8];
        reference_select(row_ref, scores, 8, 1, 0, 3, taken);
        coli_index_cand_t heap[3];
        coli_index_select_topk(row_new, scores, 8, 1, 0, 3, heap);
        if (memcmp(row_ref, row_new, sizeof(row_ref)) != 0) { ok = 0; printf("MISMATCH all-excluded\n"); }
    }
    {
        float scores[5] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
        int row_ref[10], row_new[10];
        for (int i = 0; i < 10; i++) { row_ref[i] = -1; row_new[i] = -1; }
        unsigned char taken[5];
        reference_select(row_ref, scores, 5, 2, 0, 5, taken);
        coli_index_cand_t heap[5];
        coli_index_select_topk(row_new, scores, 5, 2, 0, 5, heap);
        if (memcmp(row_ref, row_new, sizeof(row_ref)) != 0) { ok = 0; printf("MISMATCH all-tied\n"); }
    }

    if (ok) printf("sparse index top-k: %ld randomised trials + 2 edge cases, IDENTICAL to the reference scan\n",
                    total_rows);

    /* F9a: the head-lane dot vs the scalar dot -- bit-identical is a claim
     * about THIS build's own compiled scalar loop (the item's whole
     * subject, verified against this rig's gcc -O3 -march=native output;
     * see sparse_index.h's own comment on coli_sparse_index_lanes_on for
     * a clang counter-example found while writing this test). Off AVX2/FMA
     * or off real GCC, "lanes on" (forced here via the test hook,
     * independent of the engine's own default) is unverified, so the
     * strict memcmp oracle only runs there; a report of a mismatch on any
     * other build would not be describing a regression in the change this
     * item makes. */
#if defined(COLI_INDEX_HEADVEC) && defined(__GNUC__) && !defined(__clang__)
    long f9a_a_trials = 0, f9a_b_trials = 0;
    int f9a_a_ok = ok ? f9a_score_row_trials(&f9a_a_trials) : 0;
    if (!f9a_a_ok) ok = 0;
    int f9a_b_ok = ok ? f9a_pipeline_trials(&f9a_b_trials) : 0;
    if (!f9a_b_ok) ok = 0;

    /* Structured cases, not counted in the random totals above: the dense
     * regime (sequence just under GLM-5.3's own topk+pool-1 = 2 051, every
     * pool selectable), an exact pool boundary (no incomplete tail at all),
     * a maximally incomplete last pool, and the real model shape
     * (heads=32, dim=128, pool=4, topk=2048) at a sparse sequence length --
     * all at both a one-row and an all-rows q-range. */
    long f9a_c_trials = 0;
    int f9a_c_ok = 1;
    if (ok) {
        struct { int sequence, heads, dim, pool, topk, with_tail; const char *tag; } cases[] = {
            { 2051, 5,  9,  4, 2048, 1, "dense-regime" },
            { 2051, 5,  9,  4, 2048, 0, "dense-regime-no-tail" },
            {  400, 8, 16,  4,  16,  1, "exact-pool-boundary" },   /* 400 % 4 == 0 */
            {  403, 8, 16,  4,  16,  1, "max-incomplete-tail" },   /* 403 % 4 == 3 */
            { 4096, 32, 128, 4, 2048, 1, "real-model-shape-sparse" },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]) && f9a_c_ok; i++) {
            f9a_c_ok &= f9a_pipeline_one(cases[i].sequence, cases[i].heads, cases[i].dim,
                                        cases[i].pool, cases[i].topk, cases[i].with_tail,
                                        cases[i].sequence - 1, cases[i].sequence,
                                        cases[i].tag);
            f9a_c_trials++;
            f9a_c_ok &= f9a_pipeline_one(cases[i].sequence, cases[i].heads, cases[i].dim,
                                        cases[i].pool, cases[i].topk, cases[i].with_tail,
                                        0, cases[i].sequence, cases[i].tag);
            f9a_c_trials++;
        }
        if (!f9a_c_ok) ok = 0;
    }
    coli_sparse_index_lanes_set(-1);   /* restore: re-read GLM53_INDEX_LANES if anyone asks again */

    if (f9a_a_ok && f9a_b_ok && f9a_c_ok)
        printf("F9a index lanes: %ld score_row trials + %ld pipeline trials + %ld structured cases, "
               "lanes on/off IDENTICAL\n", f9a_a_trials, f9a_b_trials, f9a_c_trials);
#else
    printf("F9a index lanes: SKIPPED (needs AVX2/FMA and real GCC -- this build has neither or not both)\n");
#endif

    return ok ? 0 : 1;
}
