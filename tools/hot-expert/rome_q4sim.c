/* rome_q4sim.c -- the TRANSFORM ORACLE for roadmap item QP's three format
 * simulations (int4-g64 experts, int8 dense, int8 LM head).
 *
 * §G15's precedent is explicit and QP repeats it: an unverified simulator's
 * verdict is worthless whichever way it comes out. rome_i3sim.c proved that
 * GLM's int3 probe was bit-for-bit the tree's own pack_int3_g64 + matmul_i3
 * before anyone believed its "int3 dies". This does the same job for the three
 * transforms in c/q38_sim.h, and it does it against the tree's OWN shipping
 * code and the converter's OWN python, never against a restatement of the same
 * formula in this file:
 *
 *   A  q38_bf16_f32 == st.h's bf16_to_f32, EXHAUSTIVE over all 65 536 bf16 bit
 *      patterns (q38_sim.h restates it to stay includable standalone).
 *   B  the int4 level map, exhaustive over the reachable decision space: all
 *      256 fp8 bytes against all 256 possible group-absmax bytes, at four block
 *      scales, versus the per-element rule spelled out from
 *      quantize_i4_grouped's own arithmetic.
 *   C  THE BIG ONE for (a): dequantise a random fp8 tensor, pack it with the
 *      tree's int4-g64 packer, run quant.h's OWN matmul_i4_grouped over the
 *      packed result -- the shipping fmt=4 path -- and compare against
 *      q38_matmul_fp8_sim_i4 on the fp8 bytes. Same products, same group
 *      scales, same summation order, so the bar is float reassociation, not a
 *      quantisation tolerance.
 *   D  THE BIG ONE for (b)/(c): q38_i8sim_pack_bf16 with gs<=0 == quant.h's OWN
 *      quantize_rows(bits=8), BIT-IDENTICAL, on real dense shapes. And
 *      q38_matmul_i8sim versus a scalar reference over the same int8 bytes.
 *   E  how big each format change actually is, at this engine's real shapes:
 *      relL2 and cosine of int4-g64 against the fp8 it replaces, and of int8
 *      against the bf16 it replaces. NOT an accuracy verdict on the model --
 *      §G14's rule is that a synthetic harness cannot size accuracy, only
 *      ratios and implementation equality. The model says the rest.
 *   F  the levels/scales/dequant dumps that tools/hot-expert/qp_convcheck.py
 *      compares against the converter's own quant_int4_grouped / quant_int8 in
 *      c/tools/convert_fp8_to_int4.py. C proves the sim equals the tree's
 *      kernel; F proves the tree's packer equals the CONVERTER, which is what
 *      closes the loop to "this really is the format Q8/Q7 would ship".
 *
 * Build (rig):
 *   gcc -O3 -march=native -fopenmp -I c -o /tmp/rome_q4sim \
 *       tools/hot-expert/rome_q4sim.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "quant.h"
#include "q38_sim.h"

/* st.h's own function, for test A. Pulled in by name rather than by include:
 * st.h drags the whole shard layer in and this harness needs one inline. */
static inline float st_bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static float rndf(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

/* ---- the tree's int4-g64 packer ------------------------------------------ *
 * quantize_i4_grouped() from c/glm53.c, verbatim (it is static there, so it
 * cannot be included; test F checks the arithmetic against the converter's
 * python instead of trusting this copy). */
static void tree_quantize_i4_grouped(const float *w, uint8_t *q4, float *scale,
                                     int rows, int columns, int gs) {
    const int groups = (columns + gs - 1) / gs;
    const int packed = (columns + 1) / 2;
    for (int r = 0; r < rows; r++) {
        const float *row = w + (size_t)r * columns;
        uint8_t *dst = q4 + (size_t)r * packed;
        memset(dst, 0, (size_t)packed);
        for (int g = 0; g < groups; g++) {
            const int start = g * gs;
            const int stop = start + gs < columns ? start + gs : columns;
            float amax = 0.0f;
            for (int c = start; c < stop; c++) {
                const float value = fabsf(row[c]);
                if (value > amax) amax = value;
            }
            float step = amax / 7.0f;
            if (step < 1e-8f) step = 1e-8f;
            scale[(size_t)r * groups + g] = step;
            for (int c = start; c < stop; c++) {
                int level = (int)lrintf(row[c] / step);
                if (level < -8) level = -8;
                if (level > 7) level = 7;
                const uint8_t nibble = (uint8_t)(level + 8);
                if (c & 1) dst[c >> 1] |= (uint8_t)(nibble << 4);
                else       dst[c >> 1] |= nibble;
            }
        }
    }
}

/* ---- A: q38_bf16_f32, exhaustive --------------------------------------- */
static void test_bf16(void) {
    printf("A. q38_bf16_f32 vs st.h's bf16_to_f32, exhaustive over 65536 patterns\n");
    int bad = 0;
    for (int h = 0; h < 65536; h++) {
        float a = q38_bf16_f32((uint16_t)h), b = st_bf16_to_f32((uint16_t)h);
        if (memcmp(&a, &b, sizeof(float)) != 0) bad++;
    }
    CHECK(bad == 0, "%d of 65536 bf16 patterns decode differently", bad);
    printf("   65536 patterns, %d disagreements\n", bad);
}

/* ---- B: the int4 level map, exhaustive over the reachable space ---------- *
 * A group of 64 fp8 bytes shares ONE block scale sc, so the group's dequantised
 * absmax is |e4m3_decode(bmax)|*|sc| for whichever byte bmax has the largest
 * magnitude, and every level in the group is decided by (byte, bmax, sc).
 * Enumerating all 256x256 byte pairs at four scales covers the whole decision
 * space of q38_i4sim_row's inner loop, including every subnormal and the
 * 1e-8 step floor. */
static void test_level_map_exhaustive(void) {
    printf("B. the int4 level map, exhaustive over 256 fp8 bytes x 256 group-absmax bytes\n");
    const float scales[4] = { 1.0f, 3.7e-3f, 1.25e-2f, 9.1e-7f };
    long checked = 0; int bad = 0, skipped_nan = 0;
    for (int t = 0; t < 4; t++) {
        const float sc = scales[t];
        for (int bm = 0; bm < 256; bm++) {
            const float dmax = e4m3_decode((uint8_t)bm) * sc;
            if (dmax != dmax) { skipped_nan++; continue; }
            const float amax = fabsf(dmax);
            float step = amax / 7.0f; if (step < 1e-8f) step = 1e-8f;
            for (int b = 0; b < 256; b++) {
                const float d = e4m3_decode((uint8_t)b) * sc;
                if (d != d) continue;                       /* NaN policy, tested separately */
                if (fabsf(d) > amax) continue;              /* cannot occur in this group */
                /* q38_i4sim_row, through the real function, one element group */
                uint8_t wrow[Q38_I4_GS]; float lev[Q38_I4_GS], gsc[2];
                for (int k = 0; k < Q38_I4_GS; k++) wrow[k] = (uint8_t)b;
                wrow[0] = (uint8_t)bm;                      /* fix the group absmax */
                float sclr[1] = { sc };
                q38_i4sim_row(wrow, sclr, Q38_I4_GS, lev, gsc);
                /* quantize_i4_grouped's own decision, on the dequantised value */
                int want = (int)lrintf(d / step);
                if (want < -8) want = -8; else if (want > 7) want = 7;
                CHECK(gsc[0] == step, "sc=%g bm=%d: group step %g, want %g", sc, bm, gsc[0], step);
                if ((int)lev[1] != want) {
                    if (bad < 6) printf("  FAIL: sc=%g bm=%d b=%d: sim level %d, "
                                        "quantize_i4_grouped %d\n", sc, bm, b, (int)lev[1], want);
                    bad++; fails++;
                }
                checked++;
            }
        }
    }
    printf("   %ld reachable (byte, absmax-byte, scale) combinations, %d disagreements "
           "(%d NaN absmax bytes skipped)\n", checked, bad, skipped_nan);
}

/* ---- C: the sim vs the REAL fmt=4 pipeline, on fp8 bytes ----------------- */
static void test_vs_real_i4(int I, int O, const char *label) {
    const int64_t nblkI = fp8_nblk(I), nblkO = fp8_nblk(O);
    const int64_t rb4 = (I + 1) / 2, ng4 = (I + Q38_I4_GS - 1) / Q38_I4_GS;

    uint8_t *q8 = malloc((size_t)O * I);
    float *bs = malloc((size_t)nblkO * nblkI * sizeof(float));
    float *deq = malloc((size_t)O * I * sizeof(float));
    uint8_t *q4 = malloc((size_t)O * rb4);
    float *s4 = malloc((size_t)O * ng4 * sizeof(float));
    float *x = malloc((size_t)I * sizeof(float));
    float *y_sim = malloc((size_t)O * sizeof(float));
    float *y_real = malloc((size_t)O * sizeof(float));
    float *y_fp8 = malloc((size_t)O * sizeof(float));
    if (!q8 || !bs || !deq || !q4 || !s4 || !x || !y_sim || !y_real || !y_fp8) { printf("OOM\n"); exit(1); }

    for (int64_t i = 0; i < (int64_t)O * I; i++) {
        uint8_t b = (uint8_t)rnd();
        if ((b & 0x7F) == 0x7F) b &= 0x7E;        /* no NaN bytes: tested separately */
        q8[i] = b;
    }
    for (int64_t i = 0; i < nblkO * nblkI; i++) bs[i] = 1e-3f * (0.2f + (float)(rnd() % 1000) / 500.0f);
    for (int i = 0; i < I; i++) x[i] = rndf();

    /* the fmt=8 dequant rule, exactly as matmul_fp8 reads it */
    for (int o = 0; o < O; o++)
        for (int i = 0; i < I; i++)
            deq[(int64_t)o * I + i] = e4m3_decode(q8[(int64_t)o * I + i]) *
                                      bs[(int64_t)(o / FP8_BLOCK) * nblkI + i / FP8_BLOCK];

    tree_quantize_i4_grouped(deq, q4, s4, O, I, Q38_I4_GS);
    matmul_i4_grouped(y_real, x, q4, s4, 1, I, O, Q38_I4_GS);
    q38_matmul_fp8_sim_i4(y_sim, x, q8, bs, 1, I, O);
    matmul_fp8(y_fp8, x, q8, bs, 1, I, O);

    /* levels, bit for bit: stronger than any norm on the output */
    int64_t lev_bad = 0, sc_bad = 0;
    {
        float lev[Q38_SIM_MAXI], gsc[Q38_SIM_MAXI / Q38_I4_GS + 1];
        for (int o = 0; o < O; o++) {
            q38_i4sim_row(q8 + (int64_t)o * I, bs + (int64_t)(o / FP8_BLOCK) * nblkI, I, lev, gsc);
            for (int i = 0; i < I; i++) {
                const uint8_t byte = q4[(int64_t)o * rb4 + (i >> 1)];
                const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
                if ((int)lev[i] != nib - 8) lev_bad++;
            }
            for (int64_t g = 0; g < ng4; g++)
                if (memcmp(&gsc[g], &s4[(int64_t)o * ng4 + g], sizeof(float)) != 0) sc_bad++;
        }
    }

    double num = 0, den = 0, mx = 0, dot = 0, na = 0, nb = 0;
    for (int o = 0; o < O; o++) {
        const double d = (double)y_sim[o] - (double)y_real[o];
        num += d * d; den += (double)y_real[o] * y_real[o];
        if (fabs(d) > mx) mx = fabs(d);
        dot += (double)y_sim[o] * y_real[o]; na += (double)y_sim[o] * y_sim[o]; nb += (double)y_real[o] * y_real[o];
    }
    const double rel = sqrt(num / den);
    printf("C. %s (I=%d O=%d): sim vs REAL quantize_i4_grouped + matmul_i4_grouped\n", label, I, O);
    printf("   levels %lld of %lld differ, group scales %lld of %lld differ\n",
           (long long)lev_bad, (long long)O * I, (long long)sc_bad, (long long)O * ng4);
    printf("   output relL2 %.3e  cos %.9f  maxabs %.3e\n", rel, dot / sqrt(na * nb), mx);
    CHECK(lev_bad == 0, "%s: the simulation's int4 levels are not the packer's", label);
    CHECK(sc_bad == 0, "%s: the simulation's group scales are not the packer's", label);
    CHECK(rel < 1e-5, "%s: sim and the real fmt=4 path differ by more than reassociation (relL2 %.3e)", label, rel);

    /* E: how big a change int4-g64 is against the fp8 it would replace */
    num = den = dot = na = nb = 0;
    for (int o = 0; o < O; o++) {
        const double d = (double)y_sim[o] - (double)y_fp8[o];
        num += d * d; den += (double)y_fp8[o] * y_fp8[o];
        dot += (double)y_sim[o] * y_fp8[o]; na += (double)y_sim[o] * y_sim[o]; nb += (double)y_fp8[o] * y_fp8[o];
    }
    printf("E. %s: int4-g64 vs fp8 on the SAME weights: relL2 %.4e  cos %.9f\n",
           label, sqrt(num / den), dot / sqrt(na * nb));

    free(q8); free(bs); free(deq); free(q4); free(s4); free(x); free(y_sim); free(y_real); free(y_fp8);
}

/* ---- C2: the NaN policy is the one the header documents ------------------ */
static void test_nan_policy(void) {
    printf("C2. a NaN fp8 byte maps to level 0 and is counted (not silently absorbed)\n");
    uint8_t w[Q38_I4_GS]; float lev[Q38_I4_GS], gsc[2], sc[1] = { 1e-3f };
    for (int k = 0; k < Q38_I4_GS; k++) w[k] = 0x40;      /* +1.0 */
    w[3] = 0x7F;                                          /* NaN */
    const uint64_t before = g_q38_sim_nan;
    q38_i4sim_row(w, sc, Q38_I4_GS, lev, gsc);
    CHECK(lev[3] == 0.0f, "a NaN weight produced level %g, not 0", lev[3]);
    CHECK(g_q38_sim_nan == before + 1, "the NaN counter did not move (%llu -> %llu)",
          (unsigned long long)before, (unsigned long long)g_q38_sim_nan);
    printf("   level %g, counter +%llu\n", lev[3], (unsigned long long)(g_q38_sim_nan - before));
}

/* ---- D: the int8 packer == quantize_rows, and its kernel ---------------- */
static void test_i8(int I, int O, const char *label) {
    uint16_t *W = malloc((size_t)O * I * sizeof(uint16_t));
    float *wf = malloc((size_t)O * I * sizeof(float));
    int8_t *qs = malloc((size_t)O * I), *qt = malloc((size_t)O * I);
    float *ss = malloc((size_t)O * sizeof(float)), *sref = malloc((size_t)O * sizeof(float));
    float *x = malloc((size_t)I * sizeof(float));
    float *y_sim = malloc((size_t)O * sizeof(float)), *y_ref = malloc((size_t)O * sizeof(float));
    float *y_bf = malloc((size_t)O * sizeof(float));
    if (!W || !wf || !qs || !qt || !ss || !sref || !x || !y_sim || !y_ref || !y_bf) { printf("OOM\n"); exit(1); }

    for (int64_t i = 0; i < (int64_t)O * I; i++) {
        /* bf16 patterns in the range real weights occupy, both signs */
        float v = (float)((int)(rnd() % 20001) - 10000) / 300000.0f;
        uint32_t u; memcpy(&u, &v, 4);
        W[i] = (uint16_t)(u >> 16);
        wf[i] = q38_bf16_f32(W[i]);
    }
    for (int i = 0; i < I; i++) x[i] = rndf();

    /* per-ROW: the tree's own int8 quantiser is the authority */
    q38_i8sim_pack_bf16(W, O, I, 0, qs, ss);
    quantize_rows(wf, qt, sref, O, I, 8);
    int64_t qbad = 0, sbad = 0;
    for (int64_t i = 0; i < (int64_t)O * I; i++) if (qs[i] != qt[i]) qbad++;
    for (int o = 0; o < O; o++) if (memcmp(&ss[o], &sref[o], sizeof(float)) != 0) sbad++;
    printf("D. %s (I=%d O=%d) int8 per-row vs quant.h's OWN quantize_rows(bits=8)\n", label, I, O);
    printf("   levels %lld of %lld differ, row scales %lld of %d differ\n",
           (long long)qbad, (long long)O * I, (long long)sbad, O);
    CHECK(qbad == 0, "%s: the int8 packer is not quantize_rows", label);
    CHECK(sbad == 0, "%s: the int8 row scales are not quantize_rows'", label);

    /* the kernel, against a scalar reference over the same bytes */
    q38_matmul_i8sim(y_sim, x, qs, ss, 1, I, O, 0);
    for (int o = 0; o < O; o++) {
        double a = 0;
        for (int i = 0; i < I; i++) a += (double)x[i] * (double)qs[(int64_t)o * I + i];
        y_ref[o] = (float)(a * (double)ss[o]);
    }
    double num = 0, den = 0;
    for (int o = 0; o < O; o++) { double d = (double)y_sim[o] - y_ref[o]; num += d * d; den += (double)y_ref[o] * y_ref[o]; }
    printf("   kernel vs a double-precision scalar reference: relL2 %.3e\n", sqrt(num / den));
    CHECK(sqrt(num / den) < 1e-5, "%s: q38_matmul_i8sim is not summing the same products", label);

    /* E: how big int8 is against the bf16 it replaces, per-row and g64 */
    /* q38_matmul_bf16 lives in qwen38_core.h, which needs the whole engine; a
     * double-precision reference over the SAME bf16-widened values is the right
     * yardstick for a format-size question and is not a claim about the kernel. */
    for (int o = 0; o < O; o++) {
        double a = 0;
        for (int i = 0; i < I; i++) a += (double)x[i] * (double)wf[(int64_t)o * I + i];
        y_bf[o] = (float)a;
    }
    for (int pass = 0; pass < 2; pass++) {
        const int gs = pass ? Q38_I4_GS : 0;
        const int ng = q38_i8sim_groups(I, gs);
        int8_t *qq = malloc((size_t)O * I);
        float *sq = malloc((size_t)O * ng * sizeof(float));
        if (!qq || !sq) { printf("OOM\n"); exit(1); }
        q38_i8sim_pack_bf16(W, O, I, gs, qq, sq);
        q38_matmul_i8sim(y_sim, x, qq, sq, 1, I, O, gs);
        double n2 = 0, d2 = 0, dot = 0, na = 0, nb = 0, wn = 0, wd = 0;
        for (int o = 0; o < O; o++) {
            double d = (double)y_sim[o] - y_bf[o];
            n2 += d * d; d2 += (double)y_bf[o] * y_bf[o];
            dot += (double)y_sim[o] * y_bf[o]; na += (double)y_sim[o] * y_sim[o]; nb += (double)y_bf[o] * y_bf[o];
        }
        for (int o = 0; o < O; o++)
            for (int i = 0; i < I; i++) {
                double deqv = (double)qq[(int64_t)o * I + i] * (double)sq[(int64_t)o * ng + (gs ? i / gs : 0)];
                double w0 = wf[(int64_t)o * I + i];
                wn += (deqv - w0) * (deqv - w0); wd += w0 * w0;
            }
        printf("E. %s: int8 %-7s vs bf16 -- weights relL2 %.4e | matvec relL2 %.4e cos %.9f\n",
               label, gs ? "g64" : "per-row", sqrt(wn / wd), sqrt(n2 / d2), dot / sqrt(na * nb));
        free(qq); free(sq);
    }

    free(W); free(wf); free(qs); free(qt); free(ss); free(sref); free(x); free(y_sim); free(y_ref); free(y_bf);
}

/* ---- F: dumps for the converter cross-check ------------------------------ */
static void dump_for_convcheck(const char *dir) {
    const int I = 2560, O = 256;                 /* Qwen's expert input dim, a few rows */
    const int64_t nblkI = fp8_nblk(I), nblkO = fp8_nblk(O);
    const int64_t ng4 = (I + Q38_I4_GS - 1) / Q38_I4_GS;
    uint8_t *q8 = malloc((size_t)O * I);
    float *bs = malloc((size_t)nblkO * nblkI * sizeof(float));
    float *deq = malloc((size_t)O * I * sizeof(float));
    int8_t *lev8 = malloc((size_t)O * I);
    float *gsc = malloc((size_t)O * ng4 * sizeof(float));
    uint16_t *W = malloc((size_t)O * I * sizeof(uint16_t));
    int8_t *i8 = malloc((size_t)O * I);
    float *i8s = malloc((size_t)O * sizeof(float));
    int8_t *i8g = malloc((size_t)O * I);
    float *i8gs = malloc((size_t)O * ng4 * sizeof(float));
    if (!q8 || !bs || !deq || !lev8 || !gsc || !W || !i8 || !i8s || !i8g || !i8gs) { printf("OOM\n"); exit(1); }

    for (int64_t i = 0; i < (int64_t)O * I; i++) {
        uint8_t b = (uint8_t)rnd(); if ((b & 0x7F) == 0x7F) b &= 0x7E; q8[i] = b;
        float v = (float)((int)(rnd() % 20001) - 10000) / 300000.0f;
        uint32_t u; memcpy(&u, &v, 4); W[i] = (uint16_t)(u >> 16);
    }
    for (int64_t i = 0; i < nblkO * nblkI; i++) bs[i] = 1e-3f * (0.2f + (float)(rnd() % 1000) / 500.0f);
    for (int o = 0; o < O; o++)
        for (int i = 0; i < I; i++)
            deq[(int64_t)o * I + i] = e4m3_decode(q8[(int64_t)o * I + i]) *
                                      bs[(int64_t)(o / FP8_BLOCK) * nblkI + i / FP8_BLOCK];
    {
        float lev[Q38_SIM_MAXI], g[Q38_SIM_MAXI / Q38_I4_GS + 1];
        for (int o = 0; o < O; o++) {
            q38_i4sim_row(q8 + (int64_t)o * I, bs + (int64_t)(o / FP8_BLOCK) * nblkI, I, lev, g);
            for (int i = 0; i < I; i++) lev8[(int64_t)o * I + i] = (int8_t)lev[i];
            memcpy(gsc + (int64_t)o * ng4, g, (size_t)ng4 * sizeof(float));
        }
    }
    q38_i8sim_pack_bf16(W, O, I, 0, i8, i8s);
    q38_i8sim_pack_bf16(W, O, I, Q38_I4_GS, i8g, i8gs);

    char p[512];
    #define DUMP(name, ptr, bytes) do { snprintf(p, sizeof p, "%s/%s", dir, name); \
        FILE *f = fopen(p, "wb"); if (!f) { printf("  cannot write %s\n", p); exit(1); } \
        fwrite(ptr, 1, (size_t)(bytes), f); fclose(f); } while (0)
    DUMP("qp_deq_f32.bin", deq, (int64_t)O * I * 4);
    DUMP("qp_i4_levels_i8.bin", lev8, (int64_t)O * I);
    DUMP("qp_i4_scales_f32.bin", gsc, (int64_t)O * ng4 * 4);
    DUMP("qp_bf16_u16.bin", W, (int64_t)O * I * 2);
    DUMP("qp_i8row_levels_i8.bin", i8, (int64_t)O * I);
    DUMP("qp_i8row_scales_f32.bin", i8s, (int64_t)O * 4);
    DUMP("qp_i8g64_levels_i8.bin", i8g, (int64_t)O * I);
    DUMP("qp_i8g64_scales_f32.bin", i8gs, (int64_t)O * ng4 * 4);
    #undef DUMP
    printf("F. dumps for qp_convcheck.py in %s (O=%d I=%d gs=%d)\n", dir, O, I, Q38_I4_GS);

    free(q8); free(bs); free(deq); free(lev8); free(gsc); free(W); free(i8); free(i8s); free(i8g); free(i8gs);
}

int main(int argc, char **argv) {
    printf("=== QP transform oracle (int4-g64 experts, int8 dense, int8 head) ===\n\n");
    test_bf16();
    printf("\n");
    test_level_map_exhaustive();
    printf("\n");
    /* Qwen3.8's real routed-expert shapes: gate/up [640, 2560], down [2560, 640] */
    test_vs_real_i4(2560, 640, "expert gate/up");
    printf("\n");
    test_vs_real_i4(640, 2560, "expert down");
    printf("\n");
    test_nan_policy();
    printf("\n");
    /* real dense shapes: DeltaNet in_proj_qkv [.,2560], o_proj [2560,.], head [.,2560] */
    test_i8(2560, 1024, "dense [1024,2560]");
    printf("\n");
    test_i8(2560, 2048, "head-shaped [2048,2560]");
    printf("\n");
    if (argc > 1) dump_for_convcheck(argv[1]);
    printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL CHECKS PASSED", fails);
    return fails ? 1 : 0;
}
