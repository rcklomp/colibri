"""Collect a config-matched expert-usage histogram.

The hot tier ranks experts from a histogram at COLI_USAGE_PATH, but that map was
collected on CPU. CPU and 1-GPU routing agree on only ~40% of top-8 selections
(int4 numerical drift shifts the near-degenerate top-8 boundary), so a
CPU-collected map under-serves the GPU config.

This collects into its OWN array (never the loaded ranking map) and writes it to
COLI_USAGE_OUT, so a run can consume one map and emit another.
"""
p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


collector = r'''static uint64_t *g_eusage_obs;   /* [NL*E] observed selections, THIS config */
static int g_obs_NL, g_obs_E, g_obs_fd;
typedef struct { uint64_t u; int layer, eid; } ObsCand;
static int obs_cmp(const void *a, const void *b) {
    uint64_t ua = ((const ObsCand *)a)->u, ub = ((const ObsCand *)b)->u;
    return ua < ub ? 1 : ua > ub ? -1 : 0;
}
__attribute__((destructor)) static void dump_observed(void) {
    const char *out = getenv("COLI_USAGE_OUT");
    if (!g_eusage_obs || !out) return;
    const int NL = g_obs_NL, E = g_obs_E, fd = g_obs_fd;
    ObsCand *cand = malloc((size_t)(NL - fd) * E * sizeof(ObsCand));
    if (!cand) return;
    int n = 0; uint64_t total = 0;
    for (int i = fd; i < NL; i++)
        for (int e = 0; e < E; e++) {
            uint64_t u = g_eusage_obs[(size_t)i * E + e];
            if (u) { cand[n++] = (ObsCand){ u, i, e }; total += u; }
        }
    qsort(cand, (size_t)n, sizeof(ObsCand), obs_cmp);
    static const int marks[] = {500, 1000, 1600, 1695, 3295, 5086};
    uint64_t cum = 0; int mi = 0;
    fprintf(stderr, "[OBS] selections=%llu unique=%d\n", (unsigned long long)total, n);
    for (int i = 0; i < n && mi < 6; i++) {
        cum += cand[i].u;
        while (mi < 6 && i + 1 >= marks[mi]) {
            fprintf(stderr, "[OBS] top-%-5d covers %5.1f%%\n", marks[mi],
                    100.0 * (double)cum / (double)total);
            mi++;
        }
    }
    FILE *f = fopen(out, "wb");
    if (f) {
        fwrite(g_eusage_obs, sizeof(uint64_t), (size_t)NL * E, f);
        fclose(f);
        fprintf(stderr, "[OBS] histogram saved (%s)\n", out);
    }
    free(cand);
}
'''

rep("static void ffn_layer(GModel *m, const GLayer *l, int index, const float *x,",
    collector + "static void ffn_layer(GModel *m, const GLayer *l, int index, const float *x,",
    "observed-usage collector")

rep("    const int topk = c->topk;",
    "    const int topk = c->topk;\n"
    "    if (!g_eusage_obs && getenv(\"COLI_USAGE_OUT\")) {\n"
    "        g_obs_NL = c->n_layers; g_obs_E = c->n_experts; g_obs_fd = c->first_dense;\n"
    "        g_eusage_obs = calloc((size_t)g_obs_NL * g_obs_E, sizeof(uint64_t));\n"
    "    }",
    "observed-usage init")

rep("            mine[k] = best;\n            mine_w[k] = score[best];\n            total += mine_w[k];\n        }",
    "            mine[k] = best;\n            mine_w[k] = score[best];\n            total += mine_w[k];\n        }\n"
    "        if (g_eusage_obs)\n"
    "            for (int k = 0; k < topk; k++) g_eusage_obs[(size_t)index * g_obs_E + mine[k]]++;",
    "observed-usage increment")

open(p, "w").write(s)
print("OBSERVED-USAGE COLLECTOR APPLIED")
