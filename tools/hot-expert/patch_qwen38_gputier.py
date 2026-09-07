"""Wire a persistent GPU expert cache into qwen38's routed-expert forward pass.

Compiled in only when Q38_VK_TIER is defined (the new qwen38-vk build target
adds -DQ38_VK_TIER explicitly) -- deliberately NOT gated on COLI_VULKAN alone,
because NOCUDA_CFLAGS already leaks -DCOLI_VULKAN into the plain `qwen38`
target whenever VK=1 is passed (a pre-existing, unrelated Makefile quirk),
which would silently pull this code into the documented CPU-only build and
break its link. Q38_VK_TIER isolates this addition to its own target.

At runtime, still opt-in via Q38_VULKAN=1 (mirrors glm53's COLI_VULKAN convention exactly,
including COLI_VK_SHADERS for the shader path). When off, or when Vulkan init
fails, or when a given expert can't be GPU-resident (VRAM budget exhausted,
not natively FP8), the code takes the EXISTING, unmodified CPU path -- this
patch only ADDS a fast path in front of it, never replaces it unconditionally.

The GPU path only engages for true single-token decode (rows==1), where every
selected expert has exactly one assignment (top-k routing can't pick the same
expert twice for one row) -- this sidesteps prefill's multi-row-per-expert
case entirely, leaving it on the CPU path unchanged.

Residency: on first use an expert's gate/up/down are uploaded once via
coli_vk_tensor_ensure (source: the SAME CPU slot the existing code already
fetches, which itself reads straight out of the mmap'd shard -- no extra
copy) and cached in a flat (layer,expert)->tensor registry for the rest of
the process. A VRAM budget check (coli_vk_mem_budget, 1.5 GB margin) stops
new uploads once the device is full; those experts simply stay on CPU for
the rest of the run. This was checked before building: a token's experts
change every token, so uploading fresh weights EVERY call would cost more
in memcpy than it saves (measured ~3-4ms/layer) -- persistent residency is
what makes this a net win, not a per-call convenience.

All GPU-selected experts in a load_count group are batched into ONE
coli_vk_expert_group call (gate+up+silu->down, one submit) -- all-or-nothing
per group: if any expert in the group can't be made GPU-resident, the WHOLE
group falls through to the existing CPU loop, avoiding partial-batch
bookkeeping complexity.
"""
p = "/home/ronald/src/colibri/c/qwen38_core.h"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


# ---- 1. globals + Vulkan include, right after the Q38_MAX_* macros ----
rep("""#define Q38_PREFILL_BATCH_ROWS 32
#define Q38_PREFILL_WORKSPACE_BYTES (64u << 20)""",
    """#define Q38_PREFILL_BATCH_ROWS 32
#define Q38_PREFILL_WORKSPACE_BYTES (64u << 20)

/* Persistent GPU expert cache (fmt=8 e4m3), opt-in via Q38_VULKAN=1. Off by
 * default: g_q38vk_ready stays 0 and every check below short-circuits to the
 * unmodified CPU path. See the module comment in patch_qwen38_gputier.py for
 * the residency rationale (why a persistent cache, not per-call upload). */
#ifdef Q38_VK_TIER
#include "backend_vulkan.h"
static int g_q38vk_ready = 0;
static int g_q38vk_NL, g_q38vk_E;
static ColiVkTensor **g_q38vk_reg;   /* [(layer*E+eid)*3] -> {gate,up,down} */
#endif""",
    "globals + include")

# ---- 2. init hook: after model_init_range() in model_init() ----
rep("""static void model_init(Model *m,const char *snap,int cap,int bits) {
    model_init_range(m,snap,cap,bits,0,0,1,1);
}""",
    """static void model_init(Model *m,const char *snap,int cap,int bits) {
    model_init_range(m,snap,cap,bits,0,0,1,1);
#ifdef Q38_VK_TIER
    /* Vulkan is optional here exactly like in glm53.c: absent or disabled,
     * the engine says nothing more than usual and stays on CPU. */
    if (getenv("Q38_VULKAN") && atoi(getenv("Q38_VULKAN"))) {
        char spv[1024];
        const char *given = getenv("COLI_VK_SHADERS");
        if (given && strstr(given, ".spv")) snprintf(spv, sizeof(spv), "%s", given);
        else snprintf(spv, sizeof(spv), "%s/qmatmul.spv", given ? given : "shaders");
        g_q38vk_ready = coli_vk_init(spv);
        if (g_q38vk_ready) {
            g_q38vk_NL = m->c.layers; g_q38vk_E = m->c.experts;
            g_q38vk_reg = (ColiVkTensor **)calloc((size_t)g_q38vk_NL * g_q38vk_E * 3, sizeof(ColiVkTensor *));
            if (!g_q38vk_reg) { g_q38vk_ready = 0; }
            else fprintf(stderr, "[qwen38] Vulkan expert offload ready (fmt=8 e4m3, persistent cache)\\n");
        }
    }
#endif
}""",
    "vulkan init hook")

# ---- 3. upload helper, placed right after q38_expert_get_batch's definition
#         area is fine anywhere before its first use; anchoring on the LRU
#         single-expert getter since it's what the helper calls. ----
rep("""static Slot *q38_expert_get(Model *m,int layer,int eid) {""",
    """static Slot *q38_expert_get(Model *m,int layer,int eid);
#ifdef Q38_VK_TIER
static ColiVkTensor **q38vk_reg_at(int layer, int eid) {
    return &g_q38vk_reg[((size_t)layer * (size_t)g_q38vk_E + (size_t)eid) * 3];
}
/* Uploads an expert once and caches the GPU tensors; returns 0 (no upload,
 * caller falls back to CPU) once the VRAM budget margin is reached, or if
 * this expert isn't natively FP8-resident on the CPU side. The source bytes
 * come from the SAME Slot the CPU path already fetches -- q38_expert_get's
 * mmap-backed data/scales pointers (see the mmap commit) are stable for the
 * process lifetime, so there is nothing to keep alive beyond this call:
 * coli_vk_tensor_ensure copies into VRAM immediately. */
static int q38vk_expert_ensure(Model *m, int layer, int eid) {
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
}
#endif

static Slot *q38_expert_get(Model *m,int layer,int eid) {""",
    "upload helper")

# ---- 4. the routed-expert loop: GPU fast path in front of the existing,
#         unmodified CPU loop ----
OLD_LOOP = """        for(int unique_base=0;unique_base<unique_count;) {
            int load_count=unique_count-unique_base;
            if(load_count>load_limit)load_count=load_limit;
            int loaded_batch=load_count>=2&&q38_expert_get_batch(
                m,layer,unique+unique_base,load_count,batch_slots);
            for(int offset=0;offset<load_count;offset++) {
                int e=unique[unique_base+offset];
                int count=group_counts[e];
                Slot *expert=loaded_batch?batch_slots[offset]:
                                          q38_expert_get(m,layer,e);
                int first=group_offsets[e];
                for(int a=0;a<count;a++) {
                    int assignment=assignments[first+a];
                    int row=assignment/K;
                    memcpy(expert_input+(int64_t)a*H,
                           x+(int64_t)(base+row)*H,(size_t)H*sizeof(float));
                }
                phase_started=now_s();
                q38_weight_matmul(expert_gate,expert_input,&expert->gate,
                                  count,H,I);
                q38_weight_matmul(expert_up,expert_input,&expert->up,count,H,I);
                for(int a=0;a<count;a++)for(int j=0;j<I;j++)
                    expert_gate[(int64_t)a*I+j]=
                        q38_silu(expert_gate[(int64_t)a*I+j])*
                        expert_up[(int64_t)a*I+j];
                q38_weight_matmul(routed_out+(int64_t)first*H,expert_gate,
                                  &expert->down,count,I,H);
                q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
            }
            unique_base+=load_count;
        }"""

NEW_LOOP = """        for(int unique_base=0;unique_base<unique_count;) {
            int load_count=unique_count-unique_base;
            if(load_count>load_limit)load_count=load_limit;
#ifdef Q38_VK_TIER
            int gpu_handled=0;
            if(rows==1&&g_q38vk_ready&&load_count<=64){
                static ColiVkTensor *q38vk_gt[64],*q38vk_ut[64],*q38vk_dt[64];
                static int q38vk_rows[64];
                static float *q38vk_x=NULL,*q38vk_y=NULL;
                if(!q38vk_x){q38vk_x=falloc(64*(int64_t)H);q38vk_y=falloc(64*(int64_t)H);}
                int ok=1;
                for(int offset=0;offset<load_count;offset++){
                    int e=unique[unique_base+offset];
                    if(!q38vk_expert_ensure(m,layer,e)){ok=0;break;}
                    ColiVkTensor **reg=q38vk_reg_at(layer,e);
                    q38vk_gt[offset]=reg[0];q38vk_ut[offset]=reg[1];q38vk_dt[offset]=reg[2];
                    q38vk_rows[offset]=1;
                    int assignment=assignments[group_offsets[e]];
                    int row=assignment/K;
                    memcpy(q38vk_x+(int64_t)offset*H,x+(int64_t)(base+row)*H,(size_t)H*sizeof(float));
                }
                if(ok){
                    phase_started=now_s();
                    if(coli_vk_expert_group(q38vk_gt,q38vk_ut,q38vk_dt,q38vk_rows,load_count,q38vk_y,q38vk_x)){
                        for(int offset=0;offset<load_count;offset++){
                            int e=unique[unique_base+offset];
                            int first=group_offsets[e];
                            memcpy(routed_out+(int64_t)first*H,q38vk_y+(int64_t)offset*H,(size_t)H*sizeof(float));
                        }
                        q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
                        gpu_handled=1;
                    }
                }
            }
            if(!gpu_handled){
#endif
            int loaded_batch=load_count>=2&&q38_expert_get_batch(
                m,layer,unique+unique_base,load_count,batch_slots);
            for(int offset=0;offset<load_count;offset++) {
                int e=unique[unique_base+offset];
                int count=group_counts[e];
                Slot *expert=loaded_batch?batch_slots[offset]:
                                          q38_expert_get(m,layer,e);
                int first=group_offsets[e];
                for(int a=0;a<count;a++) {
                    int assignment=assignments[first+a];
                    int row=assignment/K;
                    memcpy(expert_input+(int64_t)a*H,
                           x+(int64_t)(base+row)*H,(size_t)H*sizeof(float));
                }
                phase_started=now_s();
                q38_weight_matmul(expert_gate,expert_input,&expert->gate,
                                  count,H,I);
                q38_weight_matmul(expert_up,expert_input,&expert->up,count,H,I);
                for(int a=0;a<count;a++)for(int j=0;j<I;j++)
                    expert_gate[(int64_t)a*I+j]=
                        q38_silu(expert_gate[(int64_t)a*I+j])*
                        expert_up[(int64_t)a*I+j];
                q38_weight_matmul(routed_out+(int64_t)first*H,expert_gate,
                                  &expert->down,count,I,H);
                q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
            }
#ifdef Q38_VK_TIER
            }
#endif
            unique_base+=load_count;
        }"""

rep(OLD_LOOP, NEW_LOOP, "routed-expert loop GPU fast path")

open(p, "w").write(s)
print("QWEN38 GPU TIER APPLIED")
