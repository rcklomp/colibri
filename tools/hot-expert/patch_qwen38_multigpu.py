"""Extend qwen38's GPU expert cache from one device to all three.

Single-GPU (43a9a4c) holds ~19% of Qwen's 24576 experts on one 24GB card,
and the all-or-nothing per-group fallback (if any of a token's K experts
isn't GPU-resident, the WHOLE group's compute reverts to CPU) dilutes the
gain once that card saturates.

The dev2/dev3 backend infrastructure already exists -- built for GLM earlier
this session and untouched here: coli_vk_init_dev2/3, coli_vk_tensor_ensure2/3,
coli_vk_mem_budget2/3, coli_vk_expert_group2/3 (identical fused
gate+up+silu->down shape as dev0's coli_vk_expert_group), and
coli_vk_tensor_dev() to ask a tensor which device it landed on. Nothing new
to build there -- this just wires qwen38 to use all three instead of one.

Two changes:

1. q38vk_expert_ensure tries dev0, then dev2, then dev3, in that order, and
   returns WHICH device (0/1/2, matching coli_vk_tensor_dev's encoding) the
   expert landed on, or -1 if all three are full (VRAM budget exhausted
   everywhere). Once landed on a device, coli_vk_tensor_dev(reg[0]) is used
   as the source of truth for future calls -- no separate bookkeeping array,
   since the tensor already knows.

2. q38_moe_decode's GPU path stops being all-or-nothing. Each of the K
   selected experts is placed on whichever device has room; experts that
   couldn't be placed anywhere are collected separately. The three
   per-device buckets each dispatch through their OWN coli_vk_expert_group
   call (up to 3 GPU submits per layer instead of 1, still far fewer than K
   CPU calls), and only the genuinely-unplaced experts fall back to the
   existing per-expert CPU loop -- not the whole group.
"""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "/home/ronald/src/colibri/c/qwen38_core.h"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


# ---- 1. init: bring up dev2/dev3 alongside dev0 ----
rep("""        g_q38vk_ready = coli_vk_init(spv);
        if (g_q38vk_ready) {
            g_q38vk_NL = m->c.layers; g_q38vk_E = m->c.experts;
            g_q38vk_reg = (ColiVkTensor **)calloc((size_t)g_q38vk_NL * g_q38vk_E * 3, sizeof(ColiVkTensor *));
            if (!g_q38vk_reg) { g_q38vk_ready = 0; }
            else fprintf(stderr, "[qwen38] Vulkan expert offload ready (fmt=8 e4m3, persistent cache)\\n");
        }""",
    """        g_q38vk_ready = coli_vk_init(spv);
        if (g_q38vk_ready) {
            g_q38vk_NL = m->c.layers; g_q38vk_E = m->c.experts;
            g_q38vk_reg = (ColiVkTensor **)calloc((size_t)g_q38vk_NL * g_q38vk_E * 3, sizeof(ColiVkTensor *));
            if (!g_q38vk_reg) { g_q38vk_ready = 0; }
            else fprintf(stderr, "[qwen38] Vulkan expert offload ready (fmt=8 e4m3, persistent cache)\\n");
            /* Optional 2nd/3rd device, same env convention as glm53: unset
             * COLI_VK_DEV2/3 means "don't try" rather than "auto", so a
             * single-GPU box behaves exactly as before. */
            if (g_q38vk_ready && getenv("COLI_VK_DEV2")) {
                const char *dv = getenv("COLI_VK_DEV2");
                int didx = (!strcmp(dv, "auto") || !*dv) ? -1 : atoi(dv);
                g_q38vk_dev2_ready = coli_vk_init_dev2(spv, didx);
                if (g_q38vk_dev2_ready) fprintf(stderr, "[qwen38] Vulkan dev2 ready (expert cache)\\n");
            }
            if (g_q38vk_ready && getenv("COLI_VK_DEV3")) {
                const char *dv = getenv("COLI_VK_DEV3");
                int didx = (!strcmp(dv, "auto") || !*dv) ? -1 : atoi(dv);
                g_q38vk_dev3_ready = coli_vk_init_dev3(spv, didx);
                if (g_q38vk_dev3_ready) fprintf(stderr, "[qwen38] Vulkan dev3 ready (expert cache)\\n");
            }
        }""",
    "init dev2/dev3")

rep("static int g_q38vk_ready = 0;",
    "static int g_q38vk_ready = 0;\n"
    "static int g_q38vk_dev2_ready = 0, g_q38vk_dev3_ready = 0;",
    "dev2/dev3 ready globals")

# ---- 2. multi-device upload helper ----
rep("""static int q38vk_expert_ensure(Model *m, int layer, int eid) {
    ColiVkTensor **reg = q38vk_reg_at(layer, eid);
    if (reg[0]) return 1;
    double used = 0, budget = 0;
    if (coli_vk_mem_budget(&used, &budget) && budget > 0 && (budget - used) < 1.5) return 0;
    Slot *s = q38_expert_get(m, layer, eid);
    if (!s || s->gate.kind != Q38_WEIGHT_FP8 || s->up.kind != Q38_WEIGHT_FP8 || s->down.kind != Q38_WEIGHT_FP8) return 0;
    Cfg *c = &m->c;
    ColiVkTensor *tg = NULL, *tu = NULL, *td = NULL;
    if (!coli_vk_tensor_ensure(&tg, s->gate.data, s->gate.scales, 8, c->hidden, c->inter, 128) ||
        !coli_vk_tensor_ensure(&tu, s->up.data, s->up.scales, 8, c->hidden, c->inter, 128) ||
        !coli_vk_tensor_ensure(&td, s->down.data, s->down.scales, 8, c->inter, c->hidden, 128))
        return 0;
    reg[0] = tg; reg[1] = tu; reg[2] = td;
    return 1;
}""",
    """/* Places an expert on whichever device has room, trying dev0 then dev2 then
 * dev3; returns the device index (matching coli_vk_tensor_dev's 0/1/2
 * encoding) it landed on, or -1 if all available devices are full. Already
 * resident: returns coli_vk_tensor_dev(reg[0]) immediately, no re-check. */
static int q38vk_expert_ensure(Model *m, int layer, int eid) {
    ColiVkTensor **reg = q38vk_reg_at(layer, eid);
    if (reg[0]) return coli_vk_tensor_dev(reg[0]);
    Slot *s = q38_expert_get(m, layer, eid);
    if (!s || s->gate.kind != Q38_WEIGHT_FP8 || s->up.kind != Q38_WEIGHT_FP8 || s->down.kind != Q38_WEIGHT_FP8) return -1;
    Cfg *c = &m->c;
    double used = 0, budget = 0;
    ColiVkTensor *tg = NULL, *tu = NULL, *td = NULL;
    if (coli_vk_mem_budget(&used, &budget) && (!budget || budget - used >= 1.5)) {
        if (coli_vk_tensor_ensure(&tg, s->gate.data, s->gate.scales, 8, c->hidden, c->inter, 128) &&
            coli_vk_tensor_ensure(&tu, s->up.data, s->up.scales, 8, c->hidden, c->inter, 128) &&
            coli_vk_tensor_ensure(&td, s->down.data, s->down.scales, 8, c->inter, c->hidden, 128)) {
            reg[0] = tg; reg[1] = tu; reg[2] = td;
            return coli_vk_tensor_dev(reg[0]);
        }
    }
    if (g_q38vk_dev2_ready && coli_vk_mem_budget2(&used, &budget) && (!budget || budget - used >= 1.5)) {
        tg = tu = td = NULL;
        if (coli_vk_tensor_ensure2(&tg, s->gate.data, s->gate.scales, 8, c->hidden, c->inter, 128) &&
            coli_vk_tensor_ensure2(&tu, s->up.data, s->up.scales, 8, c->hidden, c->inter, 128) &&
            coli_vk_tensor_ensure2(&td, s->down.data, s->down.scales, 8, c->inter, c->hidden, 128)) {
            reg[0] = tg; reg[1] = tu; reg[2] = td;
            return coli_vk_tensor_dev(reg[0]);
        }
    }
    if (g_q38vk_dev3_ready && coli_vk_mem_budget3(&used, &budget) && (!budget || budget - used >= 1.5)) {
        tg = tu = td = NULL;
        if (coli_vk_tensor_ensure3(&tg, s->gate.data, s->gate.scales, 8, c->hidden, c->inter, 128) &&
            coli_vk_tensor_ensure3(&tu, s->up.data, s->up.scales, 8, c->hidden, c->inter, 128) &&
            coli_vk_tensor_ensure3(&td, s->down.data, s->down.scales, 8, c->inter, c->hidden, 128)) {
            reg[0] = tg; reg[1] = tu; reg[2] = td;
            return coli_vk_tensor_dev(reg[0]);
        }
    }
    return -1;
}""",
    "multi-device upload helper")

open(p, "w").write(s)
print("QWEN38 MULTI-GPU UPLOAD APPLIED")
