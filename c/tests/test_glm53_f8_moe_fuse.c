/* F8 (record §F8-STEP0): the decode-window CPU-expert fusion.
 *
 * Two independent claims are made about ffn_moe_run_deferred_cpu at
 * tokens==1:
 *
 *  (a) GLM53_MOE_ONE_TEAM=1 (one OpenMP team for the whole window instead
 *      of one per expert) must be BIT-IDENTICAL to the knob-off path: same
 *      per-row kernel (coli_i4_row via i4_row_tail_sel), same z/i/o order
 *      per expert, same serial order over experts, same
 *      `dst[d] += scale * src[d]` accumulate -- only team creation moves.
 *  (b) GLM53_I4_FAST=2 (coli_i4_row_gv, reassociating) must be CLOSE to but
 *      not identical to the default kernel -- proving the knob is live and
 *      within the same reassociation-noise budget as GLM53_I4_FAST=1
 *      (record §F8-STEP0: relL2 ~3.3e-07 isolated; this is a tiny random
 *      fixture, not real weights, so the bound here is generous).
 *
 * Random int4 experts, no model, no Vulkan, no GPU -- everything this file
 * needs is a static in glm53.c, reached the way every other glm53 C test
 * reaches it. */
#define _GNU_SOURCE
#define GLM53_NO_MAIN
#include "../glm53.c"
#include <math.h>

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x); return 1; \
} } while(0)

#define T_HIDDEN 64
#define T_INTER  32
#define T_GS     16
#define T_NEXP   3

static uint32_t t_rng_state = 987654321u;
static uint32_t t_rndu(void) {
    t_rng_state ^= t_rng_state << 13; t_rng_state ^= t_rng_state >> 17; t_rng_state ^= t_rng_state << 5;
    return t_rng_state;
}
static float t_rnd01(void) { return (float)(t_rndu() >> 8) / 16777216.0f; }

static void fill_rand_mat(Mat *m, int O, int I, int gs) {
    m->fmt = 4; m->rows = O; m->columns = I; m->gs = gs;
    m->f = NULL; m->q8 = NULL; m->vk = NULL;
    const int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    uint8_t *q4 = malloc((size_t)O * rb);
    float *s = malloc((size_t)O * ng * sizeof(float));
    for (size_t i = 0; i < (size_t)O * rb; i++) q4[i] = (uint8_t)(t_rndu() & 0xFF);
    for (int i = 0; i < O * ng; i++) s[i] = t_rnd01() * 0.06f;
    m->q4 = q4; m->s = s;
}

static void free_mat(Mat *m) { free((void *)m->q4); free((void *)m->s); }

/* Runs the window once with the given env knobs (reset directly on the
 * cached statics -- getenv is only re-read when the cache is -1, exactly
 * as CLAUDE.md's "How a change is measured" note about this file's own
 * pattern documents for i4_fast_on()/moe_one_team_on()). */
static void run_window(const Mat *gate, const Mat *up, const Mat *down, const int *eid,
                       const int *chosen, const float *weight, int topk,
                       const float *x, const float *baseline, float *out,
                       int one_team, int i4_fast) {
    memcpy(out, baseline, (size_t)T_HIDDEN * sizeof(float));
    g_moe_one_team = one_team;
    g_i4_fast = i4_fast;
    float sg[T_INTER], su[T_INTER], tmp[T_HIDDEN];
    CpuRows cr;
    float xr[T_HIDDEN], tr[T_HIDDEN], sgr[T_INTER], sur[T_INTER];
    cr.xr = xr; cr.tr = tr; cr.sgr = sgr; cr.sur = sur; cr.cap = 1;
    ffn_moe_run_deferred_cpu(gate, up, down, eid, T_NEXP, chosen, weight, /*tokens=*/1, topk,
                             x, T_HIDDEN, /*limit=*/10.0f, sg, su, tmp, out, &cr, /*skip=*/NULL);
}

int main(void) {
    Mat gate[T_NEXP], up[T_NEXP], down[T_NEXP];
    for (int e = 0; e < T_NEXP; e++) {
        fill_rand_mat(&gate[e], T_INTER, T_HIDDEN, T_GS);
        fill_rand_mat(&up[e],   T_INTER, T_HIDDEN, T_GS);
        fill_rand_mat(&down[e], T_HIDDEN, T_INTER, T_GS);
    }
    const int eid[T_NEXP] = { 10, 11, 12 };
    const int topk = 4;
    const int chosen[4] = { 10, 11, 12, 999 };       /* 999: not in eid, weight ignored */
    const float weight[4] = { 0.30f, 0.50f, 0.20f, 0.77f };
    float x[T_HIDDEN], baseline[T_HIDDEN];
    for (int i = 0; i < T_HIDDEN; i++) { x[i] = t_rnd01() * 2.0f - 1.0f; baseline[i] = t_rnd01() * 2.0f - 1.0f; }

    float out_default[T_HIDDEN], out_team[T_HIDDEN], out_gv[T_HIDDEN];
    run_window(gate, up, down, eid, chosen, weight, topk, x, baseline, out_default, /*one_team=*/0, /*i4_fast=*/0);
    run_window(gate, up, down, eid, chosen, weight, topk, x, baseline, out_team,    /*one_team=*/1, /*i4_fast=*/0);
    run_window(gate, up, down, eid, chosen, weight, topk, x, baseline, out_gv,      /*one_team=*/1, /*i4_fast=*/2);

    /* the knobs must not have leaked into a shared read-only default for the
     * next test process, but this process only runs this test, so nothing
     * else to reset here. */

    CHECK(memcmp(out_default, out_team, T_HIDDEN * sizeof(float)) == 0);
    printf("A ok: GLM53_MOE_ONE_TEAM=1 bit-identical to knob-off (%d experts, tokens=1)\n", T_NEXP);

    double num = 0.0, den = 0.0;
    int any_diff = 0;
    for (int i = 0; i < T_HIDDEN; i++) {
        double d = (double)out_gv[i] - (double)out_default[i];
        num += d * d; den += (double)out_default[i] * (double)out_default[i];
        if (out_gv[i] != out_default[i]) any_diff = 1;
    }
    double rel_l2 = den > 0.0 ? sqrt(num / den) : sqrt(num);
    CHECK(any_diff);            /* proves the knob is live, not silently ignored */
    CHECK(rel_l2 <= 1e-5);      /* record §F8-STEP0: isolated relL2 ~3.3e-07 for the real kernel */
    printf("B ok: GLM53_I4_FAST=2 differs from default (relL2=%.3e <= 1e-5) and is live\n", rel_l2);

    for (int e = 0; e < T_NEXP; e++) { free_mat(&gate[e]); free_mat(&up[e]); free_mat(&down[e]); }
    printf("test_glm53_f8_moe_fuse: all cases passed\n");
    return 0;
}
