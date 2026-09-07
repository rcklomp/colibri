p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()
def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)

rep("static void ffn_layer(GModel *m, const GLayer *l, int index, const float *x,",
    "static FILE *g_tr = NULL;\nstatic void ffn_layer(GModel *m, const GLayer *l, int index, const float *x,", "globals")

rep("            mine[k] = best;\n            mine_w[k] = score[best];\n            total += mine_w[k];\n        }\n        for (int k = 0; k < topk; k++)\n            mine_w[k] = mine_w[k] / (total + 1e-20f) * c->routed_scale;",
    "            mine[k] = best;\n            mine_w[k] = score[best];\n            total += mine_w[k];\n        }\n        if (!g_tr && getenv(\"COLI_TRACE\")) g_tr = fopen(getenv(\"COLI_TRACE\"), \"a\");\n        if (g_tr) {\n            int cand16[16];\n            for (int k = 0; k < 16 && k < c->n_experts; k++) {\n                int best = -1; float bv = -INFINITY;\n                for (int e = 0; e < c->n_experts; e++) {\n                    int used = 0;\n                    for (int j = 0; j < k; j++) if (cand16[j] == e) { used = 1; break; }\n                    float ch = score[e] + (l->rbias ? l->rbias[e] : 0.0f);\n                    if (!used && ch > bv) { bv = ch; best = e; }\n                }\n                cand16[k] = best;\n                fprintf(g_tr, \"S %d %d %d %d %.9f\\n\", index, t, k, best, bv);\n            }\n        }\n        for (int k = 0; k < topk; k++)\n            mine_w[k] = mine_w[k] / (total + 1e-20f) * c->routed_scale;",
    "top16 dump")

open(p, "w").write(s)
print("SCORE-TRACE APPLIED")
