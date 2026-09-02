p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()

def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)

# ===== 1. registry + preload + load + mlp3_cpu + profiling (before ffn_layer) =====
pre = r'''#ifdef COLI_VULKAN
static void **g_vkreg; static uint64_t *g_eusage;
static int g_vk_E, g_vk_NL, g_vk_budget, g_vk_n;
static double g_t_eg, g_t_cpu; static long g_n_eg, g_n_eg_disp, g_n_cpu, g_n_devloss;
static double prof_now_s(void) { struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts); return (double)_ts.tv_sec + (double)_ts.tv_nsec / 1e9; }
__attribute__((destructor)) static void prof_print(void) {
    fprintf(stderr, "[PROF] eg=%.3fs(disp=%ld experts=%ld) cpu=%.3fs(n=%ld) devloss=%ld\n",
            g_t_eg, g_n_eg_disp, g_n_eg, g_t_cpu, g_n_cpu, g_n_devloss);
    const char *p2 = getenv("COLI_USAGE_PATH");
    if (p2 && g_eusage) { FILE *f = fopen(p2, "wb"); if (f) { fwrite(g_eusage, sizeof(uint64_t), (size_t)g_vk_NL * g_vk_E, f); fclose(f); } }
}
static void **vk_reg_at(int layer, int eid) { return (void **)g_vkreg + ((size_t)layer * g_vk_E + eid) * 3; }
typedef struct { uint64_t u; int layer, eid; } VkCand;
static int vk_cand_cmp(const void *a, const void *b) {
    uint64_t ua = ((const VkCand *)a)->u, ub = ((const VkCand *)b)->u;
    return ua < ub ? 1 : ua > ub ? -1 : 0;
}
/* heat-ranked preload from the (fixed) usage histogram loaded at startup */
static void vk_preload_tier(GModel *m) {
    if (!g_vk_ready || !m->streaming || !g_vkreg || !g_eusage) return;
    int nsp = m->c.n_layers - m->c.first_dense, E = m->c.n_experts, fd = m->c.first_dense;
    int64_t nz = 0; for (int64_t i = 0; i < (int64_t)nsp * E; i++) if (g_eusage[(size_t)fd * E + i]) nz++;
    if (!nz) { fprintf(stderr, "[VK] no usage history loaded — tier empty\n"); return; }
    VkCand *cand = malloc((size_t)nz * sizeof(VkCand));
    int64_t n = 0;
    for (int i = fd; i < m->c.n_layers; i++)
        for (int e = 0; e < E; e++)
            if (g_eusage[(size_t)i * E + e]) cand[n++] = (VkCand){ g_eusage[(size_t)i * E + e], i, e };
    qsort(cand, (size_t)n, sizeof(VkCand), vk_cand_cmp);
    Slot tmp; memset(&tmp, 0, sizeof(tmp)); tmp.eid = -1;
    int loaded = 0;
    for (int i = 0; i < (int)n && (g_vk_budget <= 0 || loaded < g_vk_budget); i++) {
        if ((loaded & 7) == 0) { double u, b; if (coli_vk_mem_budget(&u, &b) && (b - u) < 3.0) break; }
        int layer = cand[i].layer, eid = cand[i].eid;
        expert_read(m, layer, eid, &tmp);
        Mat gate, up, down; expert_mats(m, &tmp, &gate, &up, &down);
        void **reg = vk_reg_at(layer, eid);
        ColiVkTensor *t[3] = {0, 0, 0};
        if (coli_vk_tensor_ensure(&t[0], gate.q4, gate.s, gate.fmt, gate.columns, gate.rows, gate.gs) &&
            coli_vk_tensor_ensure(&t[1], up.q4, up.s, up.fmt, up.columns, up.rows, up.gs) &&
            coli_vk_tensor_ensure(&t[2], down.q4, down.s, down.fmt, down.columns, down.rows, down.gs)) {
            reg[0] = t[0]; reg[1] = t[1]; reg[2] = t[2]; loaded++; g_vk_n++;
        } else { for (int q = 0; q < 3; q++) if (t[q]) coli_vk_tensor_free(t[q]); break; }
    }
    free(cand); if (tmp.base) free(tmp.base);
    fprintf(stderr, "[VK] preload: %d heat-ranked experts resident (of %d candidates)\n", loaded, (int)n);
}
/* pure-CPU expert MLP (bypasses mv/Vulkan for non-resident experts) */
static void mlp3_cpu(float *out, const float *x, const Mat *g, const Mat *u, const Mat *d, float limit, float *sg, float *su) {
    matmul_i4_grouped(sg, x, g->q4, g->s, 1, g->columns, g->rows, g->gs);
    matmul_i4_grouped(su, x, u->q4, u->s, 1, u->columns, u->rows, u->gs);
    swiglu_clamped(sg, su, g->rows, limit);
    matmul_i4_grouped(out, sg, d->q4, d->s, 1, d->columns, d->rows, d->gs);
}
#endif

'''
rep("static void ffn_layer(GModel *m, const GLayer *l, int index, const float *x,",
    pre + "static void ffn_layer(GModel *m, const GLayer *l, int index, const float *x,", "insert pre")

# ===== 2. init: registry + usage load + preload =====
rep("        g_vk_ready = coli_vk_init(spv) && coli_vk_available();",
    '''        g_vk_ready = coli_vk_init(spv) && coli_vk_available();
        if (g_vk_ready) {
            g_vk_NL = m->c.n_layers; g_vk_E = m->c.n_experts;
            g_vk_budget = getenv("COLI_VK_EXPERTS") ? atoi(getenv("COLI_VK_EXPERTS")) : 0;
            g_vkreg = calloc((size_t)g_vk_NL * g_vk_E * 3, sizeof(void *));
            g_eusage = calloc((size_t)g_vk_NL * g_vk_E, sizeof(uint64_t));
            g_vk_n = 0;
            const char *up = getenv("COLI_USAGE_PATH");
            if (up) { FILE *f = fopen(up, "rb"); if (f) { size_t got = fread(g_eusage, sizeof(uint64_t), (size_t)g_vk_NL * g_vk_E, f); fclose(f); fprintf(stderr, "[VK] usage histogram loaded (%zu entries)\\n", got); } }
            vk_preload_tier(m);
        }''', "insert init")

# ===== 3. ffn_layer: fused arrays =====
rep("    for (int base = 0; base < n_union; base += block) {",
    '''#ifdef COLI_VULKAN
    const int maxres = tokens * topk;
    ColiVkTensor **vg = NULL, **vu = NULL, **vd = NULL;
    int *vrows = NULL, *voff = NULL, *vtok = NULL;
    float *vw = NULL, *xk = NULL;
    int nvk = 0, vtot = 0;
    if (g_vk_ready) {
        vg = malloc((size_t)maxres * sizeof(void *));
        vu = malloc((size_t)maxres * sizeof(void *));
        vd = malloc((size_t)maxres * sizeof(void *));
        vrows = malloc((size_t)maxres * sizeof(int));
        voff = malloc((size_t)maxres * sizeof(int));
        vtok = malloc((size_t)maxres * sizeof(int));
        vw = malloc((size_t)maxres * sizeof(float));
        xk = malloc((size_t)maxres * c->hidden * sizeof(float));
    }
#endif
    for (int base = 0; base < n_union; base += block) {''', "insert arrays")

# ===== 4. ffn_layer: resident collection + pure CPU fallback =====
rep("            expert_mats(m, slot, &gate, &up, &down);",
    '''            expert_mats(m, slot, &gate, &up, &down);
#ifdef COLI_VULKAN
            if (g_vk_ready && vk_reg_at(index, eid)[0] != NULL) {
                voff[nvk] = vtot;
                int nr = 0;
                for (int t = 0; t < tokens; t++) {
                    float scale = 0.0f;
                    for (int k = 0; k < topk; k++)
                        if (chosen[(size_t)t * topk + k] == eid) { scale = weight[(size_t)t * topk + k]; break; }
                    if (scale == 0.0f) continue;
                    memcpy(xk + (size_t)vtot * c->hidden, x + (size_t)t * c->hidden, (size_t)c->hidden * sizeof(float));
                    vtok[vtot] = t; vw[vtot] = scale; vtot++; nr++;
                }
                void **reg = vk_reg_at(index, eid);
                vg[nvk] = (ColiVkTensor *)reg[0]; vu[nvk] = (ColiVkTensor *)reg[1]; vd[nvk] = (ColiVkTensor *)reg[2];
                vrows[nvk] = nr; nvk++; g_n_eg++;
                continue;
            }
#endif''', "resident collection")

# ===== 5. ffn_layer: fused dispatch =====
rep("    free(to_read); free(slot_of); free(union_ids);",
    '''#ifdef COLI_VULKAN
    if (g_vk_ready && nvk > 0) {
        double _te0 = prof_now_s();
        float *yk = malloc((size_t)vtot * c->hidden * sizeof(float));
        if (yk && coli_vk_expert_group(vg, vu, vd, vrows, nvk, yk, xk)) {
            for (int c2 = 0; c2 < nvk; c2++) {
                const int o = voff[c2];
                for (int r = 0; r < vrows[c2]; r++) {
                    float *os = out + (size_t)vtok[o + r] * c->hidden;
                    const float wgt = vw[o + r];
                    const float *src = yk + (size_t)(o + r) * c->hidden;
                    for (int d = 0; d < c->hidden; d++) os[d] += wgt * src[d];
                }
            }
            g_n_eg_disp++;
        } else { g_n_devloss++; }
        g_t_eg += prof_now_s() - _te0;
        free(yk);
    }
#endif
    free(to_read); free(slot_of); free(union_ids);''', "fused dispatch")

# ===== 6. ffn_layer: pure-CPU fallback (mlp3_cpu) + timing =====
rep("                mlp3(tmp, x + (size_t)t * c->hidden, &gate, &up, &down,\n                     c->swiglu_limit, sg, su);",
    '''#ifdef COLI_VULKAN
                if (g_vk_ready) {
                    double _tc = prof_now_s();
                    mlp3_cpu(tmp, x + (size_t)t * c->hidden, &gate, &up, &down, c->swiglu_limit, sg, su);
                    g_t_cpu += prof_now_s() - _tc; g_n_cpu++;
                } else
#endif
                mlp3(tmp, x + (size_t)t * c->hidden, &gate, &up, &down,
                     c->swiglu_limit, sg, su);''', "cpu fallback")

# ===== 7. frees =====
rep("    free(tmp); free(su); free(sg); free(weight); free(chosen);\n}",
    '''    free(tmp); free(su); free(sg); free(weight); free(chosen);
#ifdef COLI_VULKAN
    free(vg); free(vu); free(vd); free(vrows); free(voff); free(vtok); free(vw); free(xk);
#endif
}''', "frees")

open(p, "w").write(s)
print("HEAT-RANKED FUSED APPLIED")
