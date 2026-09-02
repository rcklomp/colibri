"""Add a THIRD Vulkan device tier (dev3), mirroring dev2.

dev0 -> dev2 was already a self-contained textual mirror against its own state
struct, so dev3 is generated the same way rather than by threading a device
index through every helper. Residency ceiling goes 2 x 24 GB -> 3 x 24 GB, i.e.
~3295 -> ~5086 resident experts. Measured coverage of routing events on this
workload: 3295 residents = 83.3%, 5086 = 94.4%.
"""
p = "/home/ronald/src/colibri/c/backend_vulkan.c"
s = open(p).read()

START = "/* ==================== SECOND DEVICE"
END = "/* ---- MLA absorb attention core"
assert s.count(START) == 1 and s.count(END) == 1
i0, i1 = s.index(START), s.index(END)
block = s[i0:i1]

# coli_vk_tensor_dev is shared, not per-device: must not be duplicated.
SHARED = "int coli_vk_tensor_dev(const ColiVkTensor *t) { return t ? t->dev : 0; }\n"
assert block.count(SHARED) == 1
block = block.replace(SHARED, "")

# Longest / most specific names first so that the generic "dev2" and "G2"
# rewrites below cannot corrupt them.
for old, new in [
    ("coli_vk_expert_group_issue2", "coli_vk_expert_group_issue3"),
    ("coli_vk_expert_group_take2",  "coli_vk_expert_group_take3"),
    ("coli_vk_expert_group2",       "coli_vk_expert_group3"),
    ("coli_vk_tensor_ensure2",      "coli_vk_tensor_ensure3"),
    ("coli_vk_mem_budget2",         "coli_vk_mem_budget3"),
    ("coli_vk_dev2_available",      "coli_vk_dev3_available"),
    ("coli_vk_init_dev2",           "coli_vk_init_dev3"),
    ("eg2_prepare_submit",          "eg3_prepare_submit"),
    ("alloc_hostvis_d2",            "alloc_hostvis_d3"),
    ("scratch_reserve_d2",          "scratch_reserve_d3"),
    ("arena_suballoc_d2",           "arena_suballoc_d3"),
    ("upload_tensor_d2",            "upload_tensor_d3"),
    ("COLI_VK_DEV2",                "COLI_VK_DEV3"),
    ("SECOND DEVICE",               "THIRD DEVICE"),
    ("G2",                          "G3"),
    ('"d2 ',                        '"d3 '),
    ("d2iss",                       "d3iss"),
    ("dev2",                        "dev3"),
    ("t->dev = 1;",                 "t->dev = 2;"),
]:
    block = block.replace(old, new)

# auto-select must skip BOTH device 0 and whatever dev2 took.
OLD_SKIP = "            if (devs[i] == G.phys) continue;"
assert block.count(OLD_SKIP) == 1, "dev3 auto-select anchor"
block = block.replace(OLD_SKIP,
                      "            if (devs[i] == G.phys) continue;\n"
                      "            if (G2.ready && devs[i] == G2.phys) continue;")

OLD_WARN = ('        if (G3.phys == G.phys)\n'
            '            fprintf(stderr, "[VK] dev3: SAME physical device as dev0 '
            '— second logical device (test mode)\\n");')
assert block.count(OLD_WARN) == 1, "dev3 same-device warning anchor"
block = block.replace(OLD_WARN,
                      '        if (G3.phys == G.phys || (G2.ready && G3.phys == G2.phys))\n'
                      '            fprintf(stderr, "[VK] dev3: SAME physical device as another tier '
                      '(test mode)\\n");')

s = s[:i1] + block + s[i1:]

# free tensors that live on dev3
OLD_FREE = "    if (t->dev == 1) {   /* dev2 tensor: destroy on ITS device, count in ITS counters */"
assert s.count(OLD_FREE) == 1
s = s.replace(OLD_FREE,
              """    if (t->dev == 2) {   /* dev3 tensor: destroy on ITS device, count in ITS counters */
        if (G3.ready) {
            if (t->wbuf) { vkDestroyBuffer(G3.dev, t->wbuf, NULL); vkFreeMemory(G3.dev, t->wmem, NULL); }
            if (t->sbuf) { vkDestroyBuffer(G3.dev, t->sbuf, NULL); vkFreeMemory(G3.dev, t->smem, NULL); }
        }
        __atomic_sub_fetch(&G3.tensor_count, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&G3.used_bytes, t->wbytes + scale_floats(t->fmt, t->I, t->O, t->gs) * sizeof(float), __ATOMIC_RELAXED);
        free(t);
        return;
    }
""" + OLD_FREE)

open(p, "w").write(s)
print("DEV3 BACKEND APPLIED (block %d bytes)" % len(block))

# ---- header ----
h = "/home/ronald/src/colibri/c/backend_vulkan.h"
hs = open(h).read()
ANCHOR = """int  coli_vk_expert_group2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                           ColiVkTensor *const *downs, const int *rows, int count,
                           float *y, const float *x);"""
assert hs.count(ANCHOR) == 1, "header anchor"
hs = hs.replace(ANCHOR, ANCHOR + """

/* Third device: same contract as dev2, on a third GPU. auto skips both dev0's
 * and dev2's physical device. */
int  coli_vk_init_dev3(const char *spv_path, int devidx);
int  coli_vk_dev3_available(void);
int  coli_vk_mem_budget3(double *used_gb, double *budget_gb);
int  coli_vk_tensor_ensure3(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp);
int  coli_vk_expert_group_issue3(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                 ColiVkTensor *const *downs, const int *rows, int count,
                                 const float *x);
int  coli_vk_expert_group_take3(float *y);
int  coli_vk_expert_group3(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                           ColiVkTensor *const *downs, const int *rows, int count,
                           float *y, const float *x);""")
open(h, "w").write(hs)
print("DEV3 HEADER APPLIED")
