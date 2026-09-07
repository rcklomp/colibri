"""Batch KDA's independent projections into ONE submit.

VK_PROF on the dense path:

    memcpy_x 0.016 | desc 0.003 | record 0.005 | issue->take 0.395 | memcpy_y 0.004 ms

Recording and descriptor updates are noise; the cost is submit + fence wait,
~0.4 ms per dispatch no matter how small the matrix. kda_layer issues 9 mv() per
token and there are 34 KDA layers, so that is ~306 round trips per token, ~122 ms
of pure latency in a ~395 ms token.

Six of the nine (kq, kk, kv, kfa, kb, kga) are independent and all consume the
same input row, so they can be recorded into one command buffer and submitted
once: 9 round trips become 4. The expert path already does exactly this with up
to 64 dispatches per submit.
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


# ---------------------------------------------------------------- backend
TAIL = '''            fprintf(stderr, "[VK_PROF dense] n=%ld | memcpy_x %.0f | desc %.0f | record %.0f | submit %.0f | wait %.0f | memcpy_y %.0f ms\\n",
                    p_n, p_x, p_desc, p_rec, p_sub, p_wait, p_y);
    }
    return 1;
}'''

MULTI = TAIL + '''

/* Batched dense matvec: N independent (tensor -> output) pairs that all consume
 * the SAME input vector, recorded into one command buffer and submitted once.
 * The dense path pays ~0.4 ms of submit+fence per call regardless of matrix
 * size, so N separate calls pay N round trips for work the GPU finishes inside
 * one of them. Returns 0 (and touches nothing) if it cannot serve the batch, so
 * the caller can fall back to N plain coli_vk_matmul calls. */
int coli_vk_matmul_multi(ColiVkMM *items, int count, const float *x, int I) {
    if (!G.ready || count < 1 || count > VK_MM_MAX || I < 1) return 0;
    for (int c = 0; c < count; c++)
        if (!upload_tensor(items[c].tensor, items[c].weights, items[c].scales,
                           items[c].fmt, I, items[c].O, items[c].gs)) return 0;

    /* y offsets must satisfy the storage-buffer offset alignment; 256 covers it. */
    size_t yoff[VK_MM_MAX], ytot = 0;
    for (int c = 0; c < count; c++) {
        yoff[c] = ytot;
        ytot += ((size_t)items[c].O * sizeof(float) + 255) & ~(size_t)255;
    }
    const size_t xb = (size_t)I * sizeof(float);
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.y, ytot, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, x, xb);

    if (!G.mm_pool) {
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * VK_MM_MAX};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = VK_MM_MAX, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.mm_pool), "mm descPool");
        VkDescriptorSetLayout ls[VK_MM_MAX];
        for (int c = 0; c < VK_MM_MAX; c++) ls[c] = G.dsl;
        VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.mm_pool, .descriptorSetCount = VK_MM_MAX, .pSetLayouts = ls};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ai, G.mm_set), "mm descSets");
    }

    for (int c = 0; c < count; c++) {
        ColiVkTensor *t = *items[c].tensor;
        VkDescriptorBufferInfo bi[4] = {
            {.buffer = G.x.buf, .range = VK_WHOLE_SIZE},
            {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
            {.buffer = t->sbuf, .range = VK_WHOLE_SIZE},
            {.buffer = G.y.buf, .offset = yoff[c], .range = (VkDeviceSize)items[c].O * sizeof(float)}};
        VkWriteDescriptorSet w[4];
        for (int i = 0; i < 4; i++) w[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.mm_set[c],
            .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
        vkUpdateDescriptorSets(G.dev, 4, w, 0, NULL);
    }

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "mm resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "mm beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    for (int c = 0; c < count; c++) {
        ColiVkTensor *t = *items[c].tensor;
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.mm_set[c], 0, NULL);
        struct PC pc = {items[c].fmt, 1, I, items[c].O, t->rowWords, t->gs};
        vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G.cmd, (uint32_t)((items[c].O + 7) / 8), 1, 1);
    }
    VKCHECK(vkEndCommandBuffer(G.cmd), "mm endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "mm resetFence");
    double _s0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "mm queueSubmit");
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] batched matmul fence wait failed - disabling GPU offload\\n");
        G.ready = 0; G.cmd_ready = 0; G.bound_tensor = NULL;
        return 0;
    }
    if (G.eg_prof) { double _d = vk_now() - _s0; g_vsub_ms += _d; g_vwait_ms += 0; g_vsub_n++; }
    for (int c = 0; c < count; c++)
        memcpy(items[c].out, (const uint8_t *)G.y.ptr + yoff[c], (size_t)items[c].O * sizeof(float));

    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}'''

rep_file(b, TAIL, MULTI, "coli_vk_matmul_multi")

rep_file(b,
         "    ColiVkTensor *bound_tensor; int bound_S, bound_I, bound_O, cmd_ready;",
         "    VkDescriptorPool mm_pool; VkDescriptorSet mm_set[VK_MM_MAX];  /* batched dense matvec */\n"
         "    ColiVkTensor *bound_tensor; int bound_S, bound_I, bound_O, cmd_ready;",
         "mm pool fields")

# ---------------------------------------------------------------- header
rep_file(h,
         "int  coli_vk_init_dev2(const char *spv_path, int devidx);",
         """/* Batched dense matvec: N independent tensors against one shared input vector,
 * one submit instead of N. The dense path costs ~0.4 ms of submit+fence per call
 * regardless of size, so batching is worth it even for tiny matrices. */
#define VK_MM_MAX 8
typedef struct {
    ColiVkTensor **tensor;
    const void *weights; const float *scales;
    int fmt, O, gs;
    float *out;
} ColiVkMM;
int  coli_vk_matmul_multi(ColiVkMM *items, int count, const float *x, int I);

int  coli_vk_init_dev2(const char *spv_path, int devidx);""",
         "header decl")

# ---------------------------------------------------------------- engine
rep_file(g,
         """    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * c->hidden;
        KMV(qkv, &l->kq, row);
        KMV(qkv + P, &l->kk, row);
        KMV(qkv + 2 * P, &l->kv, row);""",
         """    float *lowg = malloc((size_t)D * sizeof(float));
    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * c->hidden;
        /* kq/kk/kv/kfa/kb/kga sono indipendenti e leggono tutte la stessa riga:
         * un submit invece di sei. Il costo del percorso denso e' il round trip,
         * non la matrice. */
        int batched = 0;
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
        }""",
         "batch the six projections")

# the batched path already produced kfa/kb/kga; drop the now-duplicated singles
rep_file(g,
         """        /* decadimento: gate_lower_bound * sigmoid(exp(A_log[h]) * (W_fb W_fa x + dt_bias)) */
        KMV(low, &l->kfa, row);
        KMV(decay, &l->kfb, low);""",
         """        /* decadimento: gate_lower_bound * sigmoid(exp(A_log[h]) * (W_fb W_fa x + dt_bias)) */
        KMV(decay, &l->kfb, low);""",
         "drop duplicate kfa")

rep_file(g,
         """        KMV(beta, &l->kb, row);
        for (int h = 0; h < H; h++) beta[h] = sigmoidf_(beta[h]);""",
         """        for (int h = 0; h < H; h++) beta[h] = sigmoidf_(beta[h]);""",
         "drop duplicate kb")

rep_file(g,
         """        KMV(low, &l->kga, row);
        KMV(gate, &l->kgb, low);""",
         """        KMV(gate, &l->kgb, lowg);""",
         "drop duplicate kga, gate from lowg")

rep_file(g,
         "    free(core); free(low); free(beta); free(decay); free(gate); free(qkv);",
         "    free(lowg); free(core); free(low); free(beta); free(decay); free(gate); free(qkv);",
         "free lowg")

# KMV consults the env once; expose it so the batched path honours COLI_KDA_CPU too
rep_file(g,
         'static int g_kda_cpu = -1;',
         'static int g_kda_cpu = -1;\n'
         'static int g_kda_cpu_on(void) {\n'
         '    if (g_kda_cpu < 0) g_kda_cpu = getenv("COLI_KDA_CPU") ? atoi(getenv("COLI_KDA_CPU")) : 0;\n'
         '    return g_kda_cpu;\n'
         '}',
         "kda cpu predicate")

print("KDA BATCH APPLIED")
