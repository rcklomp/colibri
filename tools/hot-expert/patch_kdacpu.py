"""A/B the KDA projections between the Vulkan path and pure CPU.

kda_layer issues 9 mv() per token, and mv() sends fmt 1/4 to coli_vk_matmul as
its own dispatch + fence wait. With 34 KDA layers that is ~306 GPU round trips
per token. Four of the nine are big (8192x4096), the rest are small, so it is
not obvious the GPU is winning once per-dispatch latency is counted.

COLI_KDA_CPU=1 forces the CPU kernels for KDA only, leaving everything else
alone.
"""
p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()

ANCHOR = "static void kda_layer(const Cfg *c, const GLayer *l, const float *x, int tokens,"
assert s.count(ANCHOR) == 1, "kda_layer anchor"
start = s.index(ANCHOR)
end = s.index("\n/* ---------- MLA + indexer con k-pool ---------- */", start)
body = s[start:end]

n = body.count("mv(")
assert n == 9, "expected 9 mv() in kda_layer, found %d" % n
body = body.replace("mv(", "KMV(")
print("ok: routed %d mv() through KMV" % n)

helper = '''static void mv_cpu(float *out, const Mat *w, const float *x) {
    switch (w->fmt) {
    case 4: matmul_i4_grouped(out, x, w->q4, w->s, 1, w->columns, w->rows, w->gs); break;
    case 1: matmul_q(out, x, w->q8, w->s, 1, w->columns, w->rows); break;
    default: matmul(out, x, w->f, 1, w->columns, w->rows); break;
    }
}
static int g_kda_cpu = -1;
#define KMV(o, w, xx) do { \\
    if (g_kda_cpu < 0) g_kda_cpu = getenv("COLI_KDA_CPU") ? atoi(getenv("COLI_KDA_CPU")) : 0; \\
    if (g_kda_cpu) mv_cpu((o), (w), (xx)); else mv((o), (w), (xx)); \\
} while (0)

'''

s = s[:start] + helper + body + s[end:]
open(p, "w").write(s)
print("KDA CPU/VK A/B APPLIED")
