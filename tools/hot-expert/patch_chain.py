"""Chain dependent projections inside one submit, behind a barrier.

coli_vk_matmul_multi batched N independent matrices against one shared input.
But KDA's kfb and kgb depend only on outputs of that batch (kfa -> low, kga ->
lowg), so they do not need their own round trips: they can be recorded into the
same command buffer after a memory barrier, reading their input straight out of
the y scratch where the first stage wrote it.

An item now carries its own input width and an optional `src`: -1 means the
shared input vector, >= 0 means "the output of item src, still on the device".
Items with out == NULL are pure intermediates and are never copied back.

KDA goes from 4 submits per layer to 2 (the batch+chain, then ko after the CPU
recurrence step). At ~0.4 ms per round trip over 34 layers that is ~27 ms of a
~309 ms token.
"""
b = "/home/ronald/src/colibri/c/backend_vulkan.c"
h = "/home/ronald/src/colibri/c/backend_vulkan.h"
g = "/home/ronald/src/colibri/c/glm53.c"


def rep_file(path, old, new, tag):
    s = open(path).read()
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    open(path, "w").write(s.replace(old, new))
    print("ok:", tag)


# ------------------------------------------------------------------ header
rep_file(h,
         """typedef struct {
    ColiVkTensor **tensor;
    const void *weights; const float *scales;
    int fmt, O, gs;
    float *out;
} ColiVkMM;""",
         """typedef struct {
    ColiVkTensor **tensor;
    const void *weights; const float *scales;
    int fmt, O, gs;
    float *out;    /* NULL = intermediate, stays on the device */
    int I;         /* input width */
    int src;       /* -1 = the shared input vector; >=0 = output of that item */
} ColiVkMM;""",
         "item gains I and src")

# ------------------------------------------------------------------ backend
rep_file(b,
         """int coli_vk_matmul_multi(ColiVkMM *items, int count, const float *x, int I) {
    if (!G.ready || count < 1 || count > VK_MM_MAX || I < 1) return 0;
    for (int c = 0; c < count; c++)
        if (!upload_tensor(items[c].tensor, items[c].weights, items[c].scales,
                           items[c].fmt, I, items[c].O, items[c].gs)) return 0;""",
         """int coli_vk_matmul_multi(ColiVkMM *items, int count, const float *x, int I) {
    if (!G.ready || count < 1 || count > VK_MM_MAX || I < 1) return 0;
    /* Chained items must follow the independent ones: one barrier, two stages. */
    int nchained = 0;
    for (int c = 0; c < count; c++) {
        const int src = items[c].src;
        if (src < 0) { if (nchained) return 0; continue; }
        if (src >= c) return 0;            /* must already have been produced */
        if (items[src].src >= 0) return 0; /* only one level of chaining */
        nchained++;
    }
    for (int c = 0; c < count; c++) {
        const int iw = items[c].src < 0 ? I : items[items[c].src].O;
        if (iw < 1) return 0;
        items[c].I = iw;
        if (!upload_tensor(items[c].tensor, items[c].weights, items[c].scales,
                           items[c].fmt, iw, items[c].O, items[c].gs)) return 0;
    }""",
         "validate and size chained items")

rep_file(b,
         """    for (int c = 0; c < count; c++) {
        ColiVkTensor *t = *items[c].tensor;
        VkDescriptorBufferInfo bi[4] = {
            {.buffer = G.x.buf, .range = VK_WHOLE_SIZE},
            {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
            {.buffer = t->sbuf, .range = VK_WHOLE_SIZE},
            {.buffer = G.y.buf, .offset = yoff[c], .range = (VkDeviceSize)items[c].O * sizeof(float)}};""",
         """    for (int c = 0; c < count; c++) {
        ColiVkTensor *t = *items[c].tensor;
        const int src = items[c].src;
        VkDescriptorBufferInfo bi[4] = {
            src < 0 ? (VkDescriptorBufferInfo){.buffer = G.x.buf, .range = VK_WHOLE_SIZE}
                    : (VkDescriptorBufferInfo){.buffer = G.y.buf, .offset = yoff[src],
                                               .range = (VkDeviceSize)items[src].O * sizeof(float)},
            {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
            {.buffer = t->sbuf, .range = VK_WHOLE_SIZE},
            {.buffer = G.y.buf, .offset = yoff[c], .range = (VkDeviceSize)items[c].O * sizeof(float)}};""",
         "bind chained input from y")

rep_file(b,
         """    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    for (int c = 0; c < count; c++) {
        ColiVkTensor *t = *items[c].tensor;
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.mm_set[c], 0, NULL);
        struct PC pc = {items[c].fmt, 1, I, items[c].O, t->rowWords, t->gs};
        vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G.cmd, (uint32_t)((items[c].O + 7) / 8), 1, 1);
    }""",
         """    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    int barrier_done = 0;
    for (int c = 0; c < count; c++) {
        ColiVkTensor *t = *items[c].tensor;
        if (items[c].src >= 0 && !barrier_done) {
            VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            barrier_done = 1;
        }
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.mm_set[c], 0, NULL);
        struct PC pc = {items[c].fmt, 1, items[c].I, items[c].O, t->rowWords, t->gs};
        vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G.cmd, (uint32_t)((items[c].O + 7) / 8), 1, 1);
    }""",
         "barrier between stages")

rep_file(b,
         """    for (int c = 0; c < count; c++)
        memcpy(items[c].out, (const uint8_t *)G.y.ptr + yoff[c], (size_t)items[c].O * sizeof(float));""",
         """    for (int c = 0; c < count; c++)
        if (items[c].out)
            memcpy(items[c].out, (const uint8_t *)G.y.ptr + yoff[c], (size_t)items[c].O * sizeof(float));""",
         "skip readback for intermediates")

# ------------------------------------------------------------------ engine
rep_file(g,
         """static int vk_batch_mv(const Mat *const *ws, float *const *os, int n,
                       const float *xin, int I) {
#ifdef COLI_VULKAN
    if (!g_vk_ready || n < 1 || n > VK_MM_MAX) return 0;
    ColiVkMM it[VK_MM_MAX];
    for (int q = 0; q < n; q++) {
        Mat *mw = (Mat *)ws[q];
        if (mw->fmt != 1 && mw->fmt != 4) return 0;
        it[q].tensor = (ColiVkTensor **)&mw->vk;
        it[q].weights = (mw->fmt == 4) ? (const void *)mw->q4 : (const void *)mw->q8;
        it[q].scales = mw->s;
        it[q].fmt = mw->fmt;
        it[q].O = mw->rows;
        it[q].gs = mw->gs;
        it[q].out = os[q];
    }
    return coli_vk_matmul_multi(it, n, xin, I);""",
         """static int vk_batch_mv_chain(const Mat *const *ws, float *const *os, const int *srcs,
                             int n, const float *xin, int I) {
#ifdef COLI_VULKAN
    if (!g_vk_ready || n < 1 || n > VK_MM_MAX) return 0;
    ColiVkMM it[VK_MM_MAX];
    for (int q = 0; q < n; q++) {
        Mat *mw = (Mat *)ws[q];
        if (mw->fmt != 1 && mw->fmt != 4) return 0;
        it[q].tensor = (ColiVkTensor **)&mw->vk;
        it[q].weights = (mw->fmt == 4) ? (const void *)mw->q4 : (const void *)mw->q8;
        it[q].scales = mw->s;
        it[q].fmt = mw->fmt;
        it[q].O = mw->rows;
        it[q].gs = mw->gs;
        it[q].out = os[q];
        it[q].I = 0;
        it[q].src = srcs ? srcs[q] : -1;
    }
    return coli_vk_matmul_multi(it, n, xin, I);""",
         "vk_batch_mv_chain")

rep_file(g,
         """#else
    (void)ws; (void)os; (void)n; (void)xin; (void)I;
    return 0;
#endif
}""",
         """#else
    (void)ws; (void)os; (void)srcs; (void)n; (void)xin; (void)I;
    return 0;
#endif
}

static int vk_batch_mv(const Mat *const *ws, float *const *os, int n,
                       const float *xin, int I) {
    return vk_batch_mv_chain(ws, os, NULL, n, xin, I);
}""",
         "vk_batch_mv wrapper")

rep_file(g,
         "static void mv_cpu(float *out, const Mat *w, const float *x) {",
         "static int vk_batch_mv_chain(const Mat *const *ws, float *const *os, const int *srcs,\n"
         "                             int n, const float *xin, int I);\n"
         "static void mv_cpu(float *out, const Mat *w, const float *x) {",
         "forward declare vk_batch_mv_chain")

KDA_OLD = """        int batched = 0;
#ifdef COLI_VULKAN
        if (g_vk_ready && !g_kda_cpu_on()) {
            const Mat *bw[6] = { &l->kq, &l->kk, &l->kv, &l->kfa, &l->kb, &l->kga };
            float *bo[6] = { qkv, qkv + P, qkv + 2 * P, low, beta, lowg };
            ColiVkMM it[6];
            int ok = 1;
            for (int q = 0; q < 6; q++) {
                if (bw[q]->fmt != 1 && bw[q]->fmt != 4) { ok = 0; break; }
                Mat *mw = (Mat *)bw[q];
                it[q].tensor = (ColiVkTensor **)&mw->vk;
                it[q].weights = mw->fmt == 4 ? (const void *)mw->q4 : (const void *)mw->q8;
                it[q].scales = mw->s; it[q].fmt = mw->fmt; it[q].O = mw->rows;
                it[q].gs = mw->gs; it[q].out = bo[q];
            }
            if (ok) batched = coli_vk_matmul_multi(it, 6, row, c->hidden);
        }
#endif
        if (!batched) {
            KMV(qkv, &l->kq, row);
            KMV(qkv + P, &l->kk, row);
            KMV(qkv + 2 * P, &l->kv, row);
            KMV(low, &l->kfa, row);
            KMV(beta, &l->kb, row);
            KMV(lowg, &l->kga, row);
        }"""

KDA_NEW = """        /* kq/kk/kv/kfa/kb/kga leggono la stessa riga; kfb e kgb dipendono solo
         * da kfa e kga, quindi entrano nello STESSO submit dietro una barriera e
         * i due intermedi non tornano mai in RAM: due round trip invece di
         * quattro. */
        int batched = 0;
        if (!g_kda_cpu_on()) {
            const Mat *bw[8] = { &l->kq, &l->kk, &l->kv, &l->kfa, &l->kb, &l->kga,
                                 &l->kfb, &l->kgb };
            float *bo[8] = { qkv, qkv + P, qkv + 2 * P, NULL, beta, NULL, decay, gate };
            const int src[8] = { -1, -1, -1, -1, -1, -1, 3, 5 };
            batched = vk_batch_mv_chain(bw, bo, src, 8, row, c->hidden);
        }
        if (!batched) {
            KMV(qkv, &l->kq, row);
            KMV(qkv + P, &l->kk, row);
            KMV(qkv + 2 * P, &l->kv, row);
            KMV(low, &l->kfa, row);
            KMV(beta, &l->kb, row);
            KMV(lowg, &l->kga, row);
            KMV(decay, &l->kfb, low);
            KMV(gate, &l->kgb, lowg);
        }"""

rep_file(g, KDA_OLD, KDA_NEW, "kda chained batch")

rep_file(g,
         """        /* decadimento: gate_lower_bound * sigmoid(exp(A_log[h]) * (W_fb W_fa x + dt_bias)) */
        KMV(decay, &l->kfb, low);
        for (int h = 0; h < H; h++)""",
         """        /* decadimento: gate_lower_bound * sigmoid(exp(A_log[h]) * (W_fb W_fa x + dt_bias)) */
        for (int h = 0; h < H; h++)""",
         "kfb now in the batch")

rep_file(g,
         """        KMV(gate, &l->kgb, lowg);
        float *normed = qkv;""",
         """        float *normed = qkv;""",
         "kgb now in the batch")

print("CHAIN APPLIED (backend + KDA)")
