"""Split the attention timer by layer kind: MLA (is_full) vs KDA linear attention.

[OTHER] says attention is now the largest single block, but GLM-5.3 mixes two
very different attention layers, so 'attention' is not yet an actionable target.
"""
p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


rep("static void kda_layer(const Cfg *c, const GLayer *l, const float *x, int tokens,",
    "static double g_t_mla = 0.0, g_t_kda = 0.0; static long g_n_mla = 0, g_n_kda = 0;\n"
    "static double attn_now_s(void) { struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t); "
    "return (double)_t.tv_sec + (double)_t.tv_nsec / 1e9; }\n"
    "__attribute__((destructor)) static void dump_attn(void) {\n"
    "    if (g_n_mla + g_n_kda == 0) return;\n"
    "    fprintf(stderr, \"[ATTN] mla=%.3fs n=%ld (%.2f ms/call) | kda=%.3fs n=%ld (%.2f ms/call)\\n\",\n"
    "            g_t_mla, g_n_mla, g_n_mla ? 1e3 * g_t_mla / g_n_mla : 0.0,\n"
    "            g_t_kda, g_n_kda, g_n_kda ? 1e3 * g_t_kda / g_n_kda : 0.0);\n"
    "}\n"
    "static void kda_layer(const Cfg *c, const GLayer *l, const float *x, int tokens,",
    "attn timers")

rep("                if (c->is_full[i]) mla_layer(c, l, normed, n, branch, st, start);\n"
    "                else kda_layer(c, l, normed, n, branch, st->kda_state, st->kda_window,\n"
    "                               s->kda_scratch);",
    "                if (c->is_full[i]) { const double _ta = attn_now_s();\n"
    "                    mla_layer(c, l, normed, n, branch, st, start);\n"
    "                    g_t_mla += attn_now_s() - _ta; g_n_mla++;\n"
    "                } else { const double _tb = attn_now_s();\n"
    "                    kda_layer(c, l, normed, n, branch, st->kda_state, st->kda_window,\n"
    "                              s->kda_scratch);\n"
    "                    g_t_kda += attn_now_s() - _tb; g_n_kda++;\n"
    "                }",
    "split call sites")

open(p, "w").write(s)
print("ATTN SPLIT APPLIED (note: kda call needs its closing brace patched)")
