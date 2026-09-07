"""Wire the third device tier into the engine: third collection bucket,
third chunked dispatch, third preload (COLI_VK_EXPERTS3 / COLI_VK_DEV3)."""
import ast

# Recover the exact current text of the dev0/dev1 blocks (they were produced by
# patch_dev2fix and are now committed source) instead of transcribing them.
tree = ast.parse(open("/tmp/patch_dev2fix.py").read())
CUR = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "rep" and len(node.args) == 3:
        CUR[node.args[2].value] = node.args[1].value
for t in ("per-device buffers", "per-device collection", "per-device chunked dispatch", "frees"):
    assert t in CUR, "cannot recover anchor: " + t

p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


# ---- 1. globals for the third tier ----
rep("static int g_vk_budget2 = 0, g_vk_reg_n2 = 0;",
    "static int g_vk_budget2 = 0, g_vk_reg_n2 = 0;\nstatic int g_vk_budget3 = 0, g_vk_reg_n3 = 0;",
    "dev3 globals")

# ---- 2. bring up the third device ----
rep('            if (g_vk_budget2 > 0 && getenv("COLI_VK_DEV2")) { const char *dv = getenv("COLI_VK_DEV2"); '
    'int didx = (!strcmp(dv,"auto")||*dv==0) ? -1 : atoi(dv); if (coli_vk_init_dev2(spv, didx)) '
    'fprintf(stderr,"[VK] dev2 ready (expert tier)\\n"); }',
    '            if (g_vk_budget2 > 0 && getenv("COLI_VK_DEV2")) { const char *dv = getenv("COLI_VK_DEV2"); '
    'int didx = (!strcmp(dv,"auto")||*dv==0) ? -1 : atoi(dv); if (coli_vk_init_dev2(spv, didx)) '
    'fprintf(stderr,"[VK] dev2 ready (expert tier)\\n"); }\n'
    '            g_vk_budget3 = getenv("COLI_VK_EXPERTS3") ? atoi(getenv("COLI_VK_EXPERTS3")) : 0;\n'
    '            if (g_vk_budget3 > 0 && getenv("COLI_VK_DEV3")) { const char *dv = getenv("COLI_VK_DEV3"); '
    'int didx = (!strcmp(dv,"auto")||*dv==0) ? -1 : atoi(dv); if (coli_vk_init_dev3(spv, didx)) '
    'fprintf(stderr,"[VK] dev3 ready (expert tier)\\n"); }',
    "dev3 init")

# ---- 3. preload onto the third device ----
rep('        fprintf(stderr, "[VK] preload dev2: %d experts resident\\n", loaded2);\n    }',
    '        fprintf(stderr, "[VK] preload dev2: %d experts resident\\n", loaded2);\n    }\n'
    '    if (g_vk_budget3 > 0 && coli_vk_dev3_available()) {\n'
    '        int loaded3 = 0;\n'
    '        for (int i = 0; i < (int)n && (g_vk_budget3 <= 0 || loaded3 < g_vk_budget3); i++) {\n'
    '            int layer = cand[i].layer, eid = cand[i].eid;\n'
    '            void **reg = vk_reg_at(layer, eid);\n'
    '            if (reg[0]) continue;\n'
    '            expert_read(m, layer, eid, &tmp);\n'
    '            Mat gate, up, down; expert_mats(m, &tmp, &gate, &up, &down);\n'
    '            ColiVkTensor *t[3] = {0,0,0};\n'
    '            if (coli_vk_tensor_ensure3(&t[0], gate.q4, gate.s, gate.fmt, gate.columns, gate.rows, gate.gs) &&\n'
    '                coli_vk_tensor_ensure3(&t[1], up.q4, up.s, up.fmt, up.columns, up.rows, up.gs) &&\n'
    '                coli_vk_tensor_ensure3(&t[2], down.q4, down.s, down.fmt, down.columns, down.rows, down.gs)) {\n'
    '                reg[0]=t[0]; reg[1]=t[1]; reg[2]=t[2]; loaded3++; g_vk_reg_n3++;\n'
    '            } else { for (int q=0;q<3;q++) if (t[q]) coli_vk_tensor_free(t[q]); break; }\n'
    '        }\n'
    '        fprintf(stderr, "[VK] preload dev3: %d experts resident\\n", loaded3);\n'
    '    }',
    "dev3 preload")

# ---- 4. third staging bucket ----
rep(CUR["per-device buffers"], """    const int maxres = tokens * topk;
    ColiVkTensor **vg0 = NULL, **vu0 = NULL, **vd0 = NULL;
    int *vrows0 = NULL, *vtok0 = NULL; float *vw0 = NULL, *xk0 = NULL;
    ColiVkTensor **vg1 = NULL, **vu1 = NULL, **vd1 = NULL;
    int *vrows1 = NULL, *vtok1 = NULL; float *vw1 = NULL, *xk1 = NULL;
    ColiVkTensor **vg2 = NULL, **vu2 = NULL, **vd2 = NULL;
    int *vrows2 = NULL, *vtok2 = NULL; float *vw2 = NULL, *xk2 = NULL;
    int nvk0 = 0, nvk1 = 0, nvk2 = 0, vtot0 = 0, vtot1 = 0, vtot2 = 0;
    if (g_vk_ready) {
        vg0 = malloc((size_t)maxres * sizeof(void *)); vu0 = malloc((size_t)maxres * sizeof(void *)); vd0 = malloc((size_t)maxres * sizeof(void *));
        vrows0 = malloc((size_t)maxres * sizeof(int)); vtok0 = malloc((size_t)maxres * sizeof(int)); vw0 = malloc((size_t)maxres * sizeof(float));
        xk0 = malloc((size_t)maxres * c->hidden * sizeof(float));
        vg1 = malloc((size_t)maxres * sizeof(void *)); vu1 = malloc((size_t)maxres * sizeof(void *)); vd1 = malloc((size_t)maxres * sizeof(void *));
        vrows1 = malloc((size_t)maxres * sizeof(int)); vtok1 = malloc((size_t)maxres * sizeof(int)); vw1 = malloc((size_t)maxres * sizeof(float));
        xk1 = malloc((size_t)maxres * c->hidden * sizeof(float));
        vg2 = malloc((size_t)maxres * sizeof(void *)); vu2 = malloc((size_t)maxres * sizeof(void *)); vd2 = malloc((size_t)maxres * sizeof(void *));
        vrows2 = malloc((size_t)maxres * sizeof(int)); vtok2 = malloc((size_t)maxres * sizeof(int)); vw2 = malloc((size_t)maxres * sizeof(float));
        xk2 = malloc((size_t)maxres * c->hidden * sizeof(float));
    }
#endif""", "three staging buckets")

# ---- 5. collection routes by device index 0/1/2 ----
rep(CUR["per-device collection"], """                if (g_vk_ready && vk_reg_at(index, eid)[0] != NULL) {
                    void **reg = vk_reg_at(index, eid);
                    int dv = coli_vk_tensor_dev((ColiVkTensor *)reg[0]);
                    if (dv < 0 || dv > 2) dv = 0;
                    float *const xkA[3] = { xk0, xk1, xk2 };
                    int *const vtokA[3] = { vtok0, vtok1, vtok2 };
                    float *const vwA[3] = { vw0, vw1, vw2 };
                    const int vtA[3] = { vtot0, vtot1, vtot2 };
                    float *xkd = xkA[dv]; int *vtokd = vtokA[dv]; float *vwd = vwA[dv];
                    int vt = vtA[dv];
                    int nr = 0;
                    for (int t = 0; t < tokens; t++) {
                        float scale = 0.0f;
                        for (int k = 0; k < topk; k++)
                            if (chosen[(size_t)t * topk + k] == eid) { scale = weight[(size_t)t * topk + k]; break; }
                        if (scale == 0.0f) continue;
                        memcpy(xkd + (size_t)vt * c->hidden, x + (size_t)t * c->hidden, (size_t)c->hidden * sizeof(float));
                        vtokd[vt] = t; vwd[vt] = scale; vt++; nr++;
                    }
                    if (nr < 1) continue;
                    if (dv == 2) {
                        vtot2 = vt; vg2[nvk2] = (ColiVkTensor *)reg[0]; vu2[nvk2] = (ColiVkTensor *)reg[1]; vd2[nvk2] = (ColiVkTensor *)reg[2]; vrows2[nvk2] = nr; nvk2++;
                    } else if (dv == 1) {
                        vtot1 = vt; vg1[nvk1] = (ColiVkTensor *)reg[0]; vu1[nvk1] = (ColiVkTensor *)reg[1]; vd1[nvk1] = (ColiVkTensor *)reg[2]; vrows1[nvk1] = nr; nvk1++;
                    } else {
                        vtot0 = vt; vg0[nvk0] = (ColiVkTensor *)reg[0]; vu0[nvk0] = (ColiVkTensor *)reg[1]; vd0[nvk0] = (ColiVkTensor *)reg[2]; vrows0[nvk0] = nr; nvk0++;
                    }
                    g_n_eg++;
                    continue;
                }""", "three-way collection")

# ---- 6. dispatch each device against its own buffer ----
rep(CUR["per-device chunked dispatch"], """    if (g_vk_ready && (nvk0 > 0 || nvk1 > 0 || nvk2 > 0)) {
        double _te0 = prof_now_s();
        float *yk0 = (nvk0 > 0) ? malloc((size_t)vtot0 * c->hidden * sizeof(float)) : NULL;
        float *yk1 = (nvk1 > 0) ? malloc((size_t)vtot1 * c->hidden * sizeof(float)) : NULL;
        float *yk2 = (nvk2 > 0) ? malloc((size_t)vtot2 * c->hidden * sizeof(float)) : NULL;
        int ok = (nvk0 == 0 || yk0 != NULL) && (nvk1 == 0 || yk1 != NULL) && (nvk2 == 0 || yk2 != NULL);
        int base = 0;
        for (int q = 0; ok && q < nvk0; ) {
            int n = nvk0 - q; if (n > 64) n = 64;
            int rs = 0; for (int w = 0; w < n; w++) rs += vrows0[q + w];
            ok = coli_vk_expert_group(vg0 + q, vu0 + q, vd0 + q, vrows0 + q, n,
                                      yk0 + (size_t)base * c->hidden, xk0 + (size_t)base * c->hidden);
            base += rs; q += n;
        }
        base = 0;
        for (int q = 0; ok && q < nvk1; ) {
            int n = nvk1 - q; if (n > 64) n = 64;
            int rs = 0; for (int w = 0; w < n; w++) rs += vrows1[q + w];
            ok = coli_vk_expert_group2(vg1 + q, vu1 + q, vd1 + q, vrows1 + q, n,
                                       yk1 + (size_t)base * c->hidden, xk1 + (size_t)base * c->hidden);
            base += rs; q += n;
        }
        base = 0;
        for (int q = 0; ok && q < nvk2; ) {
            int n = nvk2 - q; if (n > 64) n = 64;
            int rs = 0; for (int w = 0; w < n; w++) rs += vrows2[q + w];
            ok = coli_vk_expert_group3(vg2 + q, vu2 + q, vd2 + q, vrows2 + q, n,
                                       yk2 + (size_t)base * c->hidden, xk2 + (size_t)base * c->hidden);
            base += rs; q += n;
        }
        if (ok) {
            for (int r = 0; r < vtot0; r++) {
                float *os = out + (size_t)vtok0[r] * c->hidden;
                const float wgt = vw0[r];
                const float *src = yk0 + (size_t)r * c->hidden;
                for (int d = 0; d < c->hidden; d++) os[d] += wgt * src[d];
            }
            for (int r = 0; r < vtot1; r++) {
                float *os = out + (size_t)vtok1[r] * c->hidden;
                const float wgt = vw1[r];
                const float *src = yk1 + (size_t)r * c->hidden;
                for (int d = 0; d < c->hidden; d++) os[d] += wgt * src[d];
            }
            for (int r = 0; r < vtot2; r++) {
                float *os = out + (size_t)vtok2[r] * c->hidden;
                const float wgt = vw2[r];
                const float *src = yk2 + (size_t)r * c->hidden;
                for (int d = 0; d < c->hidden; d++) os[d] += wgt * src[d];
            }
            g_n_eg_disp++;
        } else { g_n_devloss++; }
        g_t_eg += prof_now_s() - _te0;
        free(yk0); free(yk1); free(yk2);
    }""", "three-way dispatch")

# ---- 7. frees ----
rep(CUR["frees"], """#ifdef COLI_VULKAN
    free(vg0); free(vu0); free(vd0); free(vrows0); free(vtok0); free(vw0); free(xk0);
    free(vg1); free(vu1); free(vd1); free(vrows1); free(vtok1); free(vw1); free(xk1);
    free(vg2); free(vu2); free(vd2); free(vrows2); free(vtok2); free(vw2); free(xk2);
#endif""", "frees")

open(p, "w").write(s)
print("DEV3 ENGINE WIRING APPLIED")
