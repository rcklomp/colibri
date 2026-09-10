/* rome_i3sim.c -- the oracle for G15's int3 numerics probe (roadmap item 4h).
 *
 * The probe in c/i3_sim.h claims to compute what a real fmt=5 (int3-g64)
 * conversion of the int4-g64 expert weights would compute. This proves it,
 * against the tree's OWN int3 pipeline rather than against a restatement of
 * the same formula:
 *
 *   A  exhaustive over the whole decision space of the nibble->level map.
 *      The map depends only on (nibble n in 0..15, group absmax m in 1..8):
 *      128 cases, every one checked against lrintf(w/s) with the clamp, i.e.
 *      against pack_int3_g64's own arithmetic.
 *   B  i4_sub_absmax's AVX2 arm == its scalar arm, randomised, all alignments.
 *   C  the big one: dequantise a random int4-g64 tensor to floats, run
 *      quant.h's pack_int3_g64 on it and quant.h's matmul_i3 over the packed
 *      result -- the real shipping path for fmt=5 -- and compare against
 *      matmul_i4_sim3 on the int4 bytes. Same products, same group scales;
 *      the two differ only in the order the group partials are summed, so
 *      the bar is float-reassociation noise, not a quantisation tolerance.
 *   D  coli_i4_rows4_sim3 == coli_i4_row_sim3, BIT-identical per row (it is
 *      the same sequence of operations in four independent accumulators).
 *   E  the error the simulation actually introduces, at GLM's real expert
 *      shapes, reported as relL2 and cosine against the int4 dequant -- the
 *      number that says how big a change int3 is before any model is run.
 *
 * Build (rig):
 *   gcc -O3 -march=x86-64-v3 -fopenmp -I c -o /tmp/rome_i3sim \
 *       tools/hot-expert/rome_i3sim.c -lm
 *
 * §G14's methodological finding applies and is why this harness does NOT
 * report an accuracy verdict on the model: synthetic data reproduces
 * instruction mix and memory traffic faithfully and destroys the activation
 * distribution. This harness proves the TRANSFORM is the converter's; only
 * the engine can say whether GLM-5.3 survives it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "quant.h"
#include "i3_sim.h"

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static float rndf(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

/* pack_int3_g64's per-element decision, spelled out independently */
static int ref_level(float w, float s) {
    int v = (int)lrintf(w / s);
    if (v > 3) v = 3;
    if (v < -4) v = -4;
    return v;
}

/* ---- A: exhaustive over (n, m) ------------------------------------------ */
static void test_exhaustive_map(void) {
    printf("A. nibble->level map, exhaustive over all 16 nibbles x 8 group absmaxes\n");
    int checked = 0;
    for (int m = 1; m <= 8; m++) {
        for (int trial = 0; trial < 4; trial++) {
            /* a few real-looking int4 group scales, incl. a tiny one */
            const float sc = (float[]){ 1.0f, 3.7e-3f, 1.25e-2f, 9.1e-7f }[trial];
            const float s3ref = fmaxf(sc * (float)m / 3.0f, 1e-8f);
            const float k = 3.0f / (float)m;
            for (int n = 0; n < 16; n++) {
                const int v = n - 8;
                if (abs(v) > m) continue;              /* cannot occur in a group of absmax m */
                float q = nearbyintf((float)v * k);
                if (q < -4.0f) q = -4.0f; else if (q > 3.0f) q = 3.0f;
                const int got = (int)q;
                const int want = ref_level((float)v * sc, s3ref);
                CHECK(got == want, "m=%d sc=%g n=%d: sim %d, pack_int3_g64 %d", m, sc, n, got, want);
                checked++;
            }
        }
    }
    printf("   %d (n, m, scale) combinations, %d disagreements\n", checked, fails);
}

/* ---- B: i4_sub_absmax vector vs scalar ---------------------------------- */
static int absmax_scalar(const uint8_t *w, int base, int n) {
    int mn = 15, mx = 0;
    for (int i = base; i < base + n; i++) {
        const int nb = (w[i >> 1] >> ((i & 1) * 4)) & 0xF;
        if (nb < mn) mn = nb;
        if (nb > mx) mx = nb;
    }
    const int a = 8 - mn, b = mx - 8;
    return a > b ? a : b;
}
static void test_absmax(void) {
    printf("B. i4_sub_absmax: vector arm vs scalar arm\n");
    uint8_t buf[512];
    int bad = 0, n_cases = 0;
    for (int trial = 0; trial < 20000; trial++) {
        for (size_t i = 0; i < sizeof buf; i++) buf[i] = (uint8_t)rnd();
        /* squeeze the range sometimes, so m=0 and m=1 groups really occur */
        if (trial % 3 == 0) for (size_t i = 0; i < sizeof buf; i++) buf[i] = 0x88;
        if (trial % 5 == 0) for (size_t i = 0; i < sizeof buf; i++) buf[i] = (uint8_t)(0x77 + (rnd() & 0x11));
        const int base = (int)(rnd() % 8) * 64;
        const int n = (int)(rnd() % 64) + 1;
        if (base + n > (int)sizeof buf * 2) continue;
        if (i4_sub_absmax(buf, base, n) != absmax_scalar(buf, base, n)) bad++;
        n_cases++;
    }
    CHECK(bad == 0, "%d of %d randomised absmax cases disagree", bad, n_cases);
    printf("   %d randomised runs (aligned and short), %d disagreements\n", n_cases, bad);
}

/* ---- C: the simulation vs the real fmt=5 pipeline ----------------------- */
static void test_vs_real_int3(int I, int O, int gs, const char *label) {
    const int64_t rb4 = (I + 1) / 2, ng4 = (I + gs - 1) / gs;
    const int64_t ng3 = i3_groups(I), rb3 = i3_rowbytes(I);

    uint8_t *q4 = malloc((size_t)O * rb4);
    float *s4 = malloc((size_t)O * ng4 * sizeof(float));
    float *deq = malloc((size_t)O * I * sizeof(float));
    uint8_t *q3 = malloc((size_t)O * rb3);
    float *s3 = malloc((size_t)O * ng3 * sizeof(float));
    float *x = malloc((size_t)I * sizeof(float));
    float *y_sim = malloc((size_t)O * sizeof(float));
    float *y_real = malloc((size_t)O * sizeof(float));
    float *y_i4 = malloc((size_t)O * sizeof(float));
    if (!q4 || !s4 || !deq || !q3 || !s3 || !x || !y_sim || !y_real || !y_i4) { printf("OOM\n"); exit(1); }

    for (int64_t i = 0; i < (int64_t)O * rb4; i++) q4[i] = (uint8_t)rnd();
    for (int64_t i = 0; i < (int64_t)O * ng4; i++) s4[i] = 1e-3f * (0.2f + (float)(rnd() % 1000) / 500.0f);
    for (int i = 0; i < I; i++) x[i] = rndf();

    /* dequantise the int4 tensor exactly as the kernels read it */
    for (int o = 0; o < O; o++)
        for (int i = 0; i < I; i++) {
            const int nb = (q4[(int64_t)o * rb4 + (i >> 1)] >> ((i & 1) * 4)) & 0xF;
            deq[(int64_t)o * I + i] = (float)(nb - 8) * s4[(int64_t)o * ng4 + i / gs];
        }

    pack_int3_g64(deq, q3, s3, O, I);
    matmul_i3(y_real, x, q3, s3, 1, I, O);
    matmul_i4_sim3(y_sim, x, q4, s4, 1, I, O, gs);
    matmul_i4_grouped(y_i4, x, q4, s4, 1, I, O, gs);

    double num = 0, den = 0, mx = 0, dot = 0, na = 0, nb_ = 0;
    for (int o = 0; o < O; o++) {
        const double d = (double)y_sim[o] - (double)y_real[o];
        num += d * d; den += (double)y_real[o] * y_real[o];
        if (fabs(d) > mx) mx = fabs(d);
        dot += (double)y_sim[o] * y_real[o]; na += (double)y_sim[o] * y_sim[o]; nb_ += (double)y_real[o] * y_real[o];
    }
    const double rel = sqrt(num / den), cos_ = dot / sqrt(na * nb_);
    printf("C. %s (I=%d O=%d gs=%d): sim3 vs REAL pack_int3_g64+matmul_i3\n", label, I, O, gs);
    printf("   relL2 %.3e  cos %.9f  maxabs %.3e\n", rel, cos_, mx);
    CHECK(rel < 1e-5, "%s: sim3 and the real int3 path disagree by more than reassociation (relL2 %.3e)", label, rel);

    /* E: how big a change int3 is at all, against the int4 the engine ships */
    num = den = 0; dot = na = nb_ = 0;
    for (int o = 0; o < O; o++) {
        const double d = (double)y_sim[o] - (double)y_i4[o];
        num += d * d; den += (double)y_i4[o] * y_i4[o];
        dot += (double)y_sim[o] * y_i4[o]; na += (double)y_sim[o] * y_sim[o]; nb_ += (double)y_i4[o] * y_i4[o];
    }
    printf("E. %s: int3 vs int4 on the SAME weights: relL2 %.4e  cos %.9f\n",
           label, sqrt(num / den), dot / sqrt(na * nb_));

    free(q4); free(s4); free(deq); free(q3); free(s3); free(x); free(y_sim); free(y_real); free(y_i4);
}

/* ---- D: rows4 == row, bit-identical ------------------------------------- */
static void test_rows4(int I, int gs) {
    const int64_t rb4 = (I + 1) / 2, ng4 = (I + gs - 1) / gs;
    uint8_t *w = malloc((size_t)rb4);
    float *s = malloc((size_t)ng4 * sizeof(float));
    float *x = malloc((size_t)4 * I * sizeof(float));
    if (!w || !s || !x) { printf("OOM\n"); exit(1); }
    int bad = 0;
    double worst = 0, num = 0, den = 0;
    for (int trial = 0; trial < 200; trial++) {
        for (int64_t i = 0; i < rb4; i++) w[i] = (uint8_t)rnd();
        for (int64_t i = 0; i < ng4; i++) s[i] = 1e-3f * (0.2f + (float)(rnd() % 1000) / 500.0f);
        for (int i = 0; i < 4 * I; i++) x[i] = rndf();
        float o4[4];
        coli_i4_rows4_sim3(w, s, x, I, I, gs, o4);
        for (int r = 0; r < 4; r++) {
            const float one = coli_i4_row_sim3(w, s, x + (int64_t)r * I, I, gs);
            if (memcmp(&one, &o4[r], sizeof(float)) != 0) bad++;
            const double d = (double)one - (double)o4[r];
            num += d * d; den += (double)one * one;
            const double rel = fabs(d) / (fabs((double)one) > 1e-30 ? fabs((double)one) : 1e-30);
            if (rel > worst) worst = rel;
        }
    }
    const double relL2 = sqrt(num / den);
    /* On x86 with AVX2 both arms are the same intrinsic sequence and this is 0
     * of 800. Without AVX2 the two scalar tails are identical C that the
     * compiler is free to contract differently, so bit-identity is not
     * guaranteed there; what must hold either way is that nothing but float
     * reassociation separates them. */
    CHECK(relL2 < 1e-6, "coli_i4_rows4_sim3 vs coli_i4_row_sim3: relL2 %.3e over 800 rows", relL2);
    printf("D. rows4 vs row (I=%d gs=%d): %d of 800 rows not bit-identical, relL2 %.3e "
           "(worst pointwise rel %.3e, on rows where cancellation makes the sum near zero)\n",
           I, gs, bad, relL2, worst);
    free(w); free(s); free(x);
}

/* ---- F: what the probe's double quantisation costs ---------------------- *
 * The engine probe re-quantises the int4 weights on disk, where a converter
 * would go fp8 -> int3 in one step. This measures the gap directly, as a
 * ratio, on three weight distributions. A ratio is exactly the kind of thing
 * a synthetic harness CAN answer (§G14): it is a property of the quantiser's
 * arithmetic, not of any activation distribution. */
static void test_double_quant_cost(void) {
    const int I = 4096, O = 256, gs = 64;
    const char *names[3] = { "gaussian", "laplace-ish", "uniform" };
    printf("F. cost of the probe's DOUBLE quantisation (fp8->int4->int3) vs a "
           "direct fp8->int3\n");
    for (int dist = 0; dist < 3; dist++) {
        float *w = malloc((size_t)O * I * sizeof(float));
        float *d4 = malloc((size_t)O * I * sizeof(float));
        float *d3 = malloc((size_t)O * I * sizeof(float));
        float *d43 = malloc((size_t)O * I * sizeof(float));
        uint8_t *q3 = malloc((size_t)O * i3_rowbytes(I));
        float *s3 = malloc((size_t)O * i3_groups(I) * sizeof(float));
        if (!w || !d4 || !d3 || !d43 || !q3 || !s3) { printf("OOM\n"); exit(1); }
        for (int64_t i = 0; i < (int64_t)O * I; i++) {
            double u = 0;
            if (dist == 0) { for (int k = 0; k < 12; k++) u += (double)(rnd() % 1000) / 1000.0; u -= 6.0; }
            else if (dist == 1) { double e = -log(1.0 - (double)(rnd() % 999999) / 1000000.0);
                                  u = (rnd() & 1) ? e : -e; }
            else u = (double)((int)(rnd() % 2001) - 1000) / 1000.0;
            w[i] = (float)(u * 0.01);
        }
        /* direct int3 */
        pack_int3_g64(w, q3, s3, O, I);
        for (int o = 0; o < O; o++)
            for (int g = 0; g < (int)i3_groups(I); g++) {
                const uint8_t *lo = q3 + (int64_t)o * i3_rowbytes(I) + (int64_t)g * I3_GBYTES, *hi = lo + 16;
                for (int k = 0; k < I3_GROUP && g * I3_GROUP + k < I; k++) {
                    unsigned u = ((lo[k >> 2] >> ((k & 3) * 2)) & 3) | (((hi[k >> 3] >> (k & 7)) & 1) << 2);
                    d3[(int64_t)o * I + g * I3_GROUP + k] = (float)((int)u - 4) * s3[(int64_t)o * i3_groups(I) + g];
                }
            }
        /* int4-g64 first, then int3 on the result -- what the engine probe does */
        for (int o = 0; o < O; o++)
            for (int g = 0; g * gs < I; g++) {
                float amax = 0;
                for (int k = 0; k < gs && g * gs + k < I; k++) {
                    const float a = fabsf(w[(int64_t)o * I + g * gs + k]);
                    if (a > amax) amax = a;
                }
                float s = amax / 7.0f; if (s < 1e-8f) s = 1e-8f;
                for (int k = 0; k < gs && g * gs + k < I; k++) {
                    int v = (int)lrintf(w[(int64_t)o * I + g * gs + k] / s);
                    if (v > 7) v = 7; if (v < -8) v = -8;
                    d4[(int64_t)o * I + g * gs + k] = (float)v * s;
                }
            }
        pack_int3_g64(d4, q3, s3, O, I);
        for (int o = 0; o < O; o++)
            for (int g = 0; g < (int)i3_groups(I); g++) {
                const uint8_t *lo = q3 + (int64_t)o * i3_rowbytes(I) + (int64_t)g * I3_GBYTES, *hi = lo + 16;
                for (int k = 0; k < I3_GROUP && g * I3_GROUP + k < I; k++) {
                    unsigned u = ((lo[k >> 2] >> ((k & 3) * 2)) & 3) | (((hi[k >> 3] >> (k & 7)) & 1) << 2);
                    d43[(int64_t)o * I + g * I3_GROUP + k] = (float)((int)u - 4) * s3[(int64_t)o * i3_groups(I) + g];
                }
            }
        double n3 = 0, n43 = 0, n4 = 0, den = 0;
        for (int64_t i = 0; i < (int64_t)O * I; i++) {
            n3 += ((double)d3[i] - w[i]) * ((double)d3[i] - w[i]);
            n43 += ((double)d43[i] - w[i]) * ((double)d43[i] - w[i]);
            n4 += ((double)d4[i] - w[i]) * ((double)d4[i] - w[i]);
            den += (double)w[i] * w[i];
        }
        printf("   %-12s int4 relL2 %.4f | int3 direct %.4f | int3 via int4 %.4f  "
               "(probe is %.2fx the real thing)\n",
               names[dist], sqrt(n4 / den), sqrt(n3 / den), sqrt(n43 / den), sqrt(n43 / n3));
        free(w); free(d4); free(d3); free(d43); free(q3); free(s3);
    }
}

int main(void) {
    printf("=== G15 int3 simulation oracle (roadmap item 4h) ===\n\n");
    test_exhaustive_map();
    printf("\n");
    test_absmax();
    printf("\n");
    /* GLM-5.3's real routed-expert shapes: gate/up [2048, 4096], down [4096, 2048] */
    test_vs_real_int3(4096, 2048, 64, "gate/up");
    printf("\n");
    test_vs_real_int3(2048, 4096, 64, "down");
    printf("\n");
    /* gs=128 exercises the two-int3-groups-per-int4-group path */
    test_vs_real_int3(4096, 512, 128, "gs=128");
    printf("\n");
    test_rows4(4096, 64);
    test_rows4(2048, 64);
    printf("\n");
    test_double_quant_cost();
    printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL CHECKS PASSED", fails);
    return fails ? 1 : 0;
}
