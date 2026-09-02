#!/usr/bin/env python3
"""Fix the dev0/dev1 expert-group dispatch.

BUG: patch_dev2 packs BOTH devices rows into one shared xk buffer with
interleaved offsets (voff0/voff1), but eg_prepare_submit / eg2_prepare_submit
each pack CONTIGUOUSLY FROM INDEX 0 (off[c]=total; total+=rows[c];
memcpy(G.eg_x.ptr, x, total*D*4)).  Consequences:
  1. dev0 group reads the wrong input rows (dev1 rows interleaved between).
  2. dev2 group then re-reads from xk[0] AND overwrites yk[0..] , clobbering
     dev0 results.
  3. CPU accumulation indexes yk by voff0/voff1, matching neither layout.
FIX: give each device its own contiguous staging buffer (xk0/xk1, yk0/yk1,
vtok0/vtok1, vw0/vw1).  Also chunk to the backends hard count<=64 limit
instead of silently failing the whole group (devloss).
"""
import ast

# --- recover the exact post-patch_dev2 text as anchors (no hand transcription) ---
tree = ast.parse(open("/tmp/patch_dev2.py").read())
NEW = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "rep" and len(node.args) == 3:
        NEW[node.args[2].value] = node.args[1].value
for t in ("array decls", "split collection", "dispatch both", "frees"):
    assert t in NEW, "cannot recover anchor: " + t

p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()
def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)

# 1. per-device buffers
rep(NEW["array decls"], """    const int maxres = tokens * topk;
    ColiVkTensor **vg0 = NULL, **vu0 = NULL, **vd0 = NULL;
    int *vrows0 = NULL, *voff0 = NULL, *vtok0 = NULL; float *vw0 = NULL, *xk0 = NULL;
    ColiVkTensor **vg1 = NULL, **vu1 = NULL, **vd1 = NULL;
    int *vrows1 = NULL, *voff1 = NULL, *vtok1 = NULL; float *vw1 = NULL, *xk1 = NULL;
    int nvk0 = 0, nvk1 = 0, vtot0 = 0, vtot1 = 0;
    if (g_vk_ready) {
        vg0 = malloc((size_t)maxres * sizeof(void *)); vu0 = malloc((size_t)maxres * sizeof(void *)); vd0 = malloc((size_t)maxres * sizeof(void *));
        vrows0 = malloc((size_t)maxres * sizeof(int)); voff0 = malloc((size_t)maxres * sizeof(int));
        vtok0 = malloc((size_t)maxres * sizeof(int)); vw0 = malloc((size_t)maxres * sizeof(float));
        xk0 = malloc((size_t)maxres * c->hidden * sizeof(float));
        vg1 = malloc((size_t)maxres * sizeof(void *)); vu1 = malloc((size_t)maxres * sizeof(void *)); vd1 = malloc((size_t)maxres * sizeof(void *));
        vrows1 = malloc((size_t)maxres * sizeof(int)); voff1 = malloc((size_t)maxres * sizeof(int));
        vtok1 = malloc((size_t)maxres * sizeof(int)); vw1 = malloc((size_t)maxres * sizeof(float));
        xk1 = malloc((size_t)maxres * c->hidden * sizeof(float));
    }
#endif""", "per-device buffers")

# 2. collection packs into the owning devices own contiguous buffer
rep(NEW["split collection"], """                if (g_vk_ready && vk_reg_at(index, eid)[0] != NULL) {
                    void **reg = vk_reg_at(index, eid);
                    const int dv = (coli_vk_tensor_dev((ColiVkTensor *)reg[0]) == 1) ? 1 : 0;
                    float *xkd = dv ? xk1 : xk0;
                    int *vtokd = dv ? vtok1 : vtok0;
                    float *vwd = dv ? vw1 : vw0;
                    int vt = dv ? vtot1 : vtot0;
                    const int off = vt;
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
                    if (dv) {
                        vtot1 = vt; voff1[nvk1] = off; vg1[nvk1] = (ColiVkTensor *)reg[0]; vu1[nvk1] = (ColiVkTensor *)reg[1]; vd1[nvk1] = (ColiVkTensor *)reg[2]; vrows1[nvk1] = nr; nvk1++;
                    } else {
                        vtot0 = vt; voff0[nvk0] = off; vg0[nvk0] = (ColiVkTensor *)reg[0]; vu0[nvk0] = (ColiVkTensor *)reg[1]; vd0[nvk0] = (ColiVkTensor *)reg[2]; vrows0[nvk0] = nr; nvk0++;
                    }
                    g_n_eg++;
                    continue;
                }""", "per-device collection")

# 3. dispatch each device against its own buffer, chunked to the count<=64 backend limit
rep(NEW["dispatch both"], """    if (g_vk_ready && (nvk0 > 0 || nvk1 > 0)) {
        double _te0 = prof_now_s();
        float *yk0 = (nvk0 > 0) ? malloc((size_t)vtot0 * c->hidden * sizeof(float)) : NULL;
        float *yk1 = (nvk1 > 0) ? malloc((size_t)vtot1 * c->hidden * sizeof(float)) : NULL;
        int ok = (nvk0 == 0 || yk0 != NULL) && (nvk1 == 0 || yk1 != NULL);
        int base = 0;
        for (int c2 = 0; ok && c2 < nvk0; ) {
            int n = nvk0 - c2; if (n > 64) n = 64;
            int rs = 0; for (int q = 0; q < n; q++) rs += vrows0[c2 + q];
            ok = coli_vk_expert_group(vg0 + c2, vu0 + c2, vd0 + c2, vrows0 + c2, n,
                                      yk0 + (size_t)base * c->hidden, xk0 + (size_t)base * c->hidden);
            base += rs; c2 += n;
        }
        base = 0;
        for (int c2 = 0; ok && c2 < nvk1; ) {
            int n = nvk1 - c2; if (n > 64) n = 64;
            int rs = 0; for (int q = 0; q < n; q++) rs += vrows1[c2 + q];
            ok = coli_vk_expert_group2(vg1 + c2, vu1 + c2, vd1 + c2, vrows1 + c2, n,
                                       yk1 + (size_t)base * c->hidden, xk1 + (size_t)base * c->hidden);
            base += rs; c2 += n;
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
            g_n_eg_disp++;
        } else { g_n_devloss++; }
        g_t_eg += prof_now_s() - _te0;
        free(yk0); free(yk1);
    }""", "per-device chunked dispatch")

# 4. frees
rep(NEW["frees"], """#ifdef COLI_VULKAN
    free(vg0); free(vu0); free(vd0); free(vrows0); free(voff0); free(vtok0); free(vw0); free(xk0);
    free(vg1); free(vu1); free(vd1); free(vrows1); free(voff1); free(vtok1); free(vw1); free(xk1);
#endif""", "frees")

open(p, "w").write(s)
print("DEV2 FIX APPLIED")
