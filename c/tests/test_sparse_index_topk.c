/* F6a: the exact top-`wanted` selection (bounded min-heap + sort) must
 * reproduce the original greedy repeated-maximum scan's output on every
 * input, INCLUDING ties -- "the `wanted` largest scores, ties broken by the
 * lower pool index, emitted in descending rank order". This test drives
 * both algorithms over randomised scores (with deliberately forced ties and
 * forced -FLT_MAX exclusions) and diffs the resulting rank-ordered token
 * rows byte for byte. It does not exercise the score computation itself
 * (unchanged: same dot products, same per-row summation order) -- only the
 * selection this item replaces. Cheap: no model, no fixtures, runs on the
 * Mac or the rig CPU. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
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
    return ok ? 0 : 1;
}
