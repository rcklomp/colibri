/* qwen36_tier_vk.c -- Vulkan VRAM expert tier for the qwen36 engine (V1, F3 step 1).
 *
 * The same qt_* contract qwen36_tier.c implements over CUDA, implemented over
 * the Vulkan backend instead, so qwen36.c's call sites do not move. Built only
 * into the separate `qwen36-vk` binary (-DQ36_VK_TIER, VK=1); `make qwen36`
 * stays CPU-only and links no Vulkan.
 *
 * V1 (2026-09-16) shipped this file as ONE device (dev3 only, packed int4
 * with gs64 group scales). F3 step 1 (2026-09-20) is a PORT, not a new
 * design: it generalises the single fixed device to a small, per-device
 * array -- home(eid), per-device budget/used/cap, planned/resident/refused
 * counted per device -- exactly the shape qwen36_tier.c's CUDA tier already
 * has and already tests (tests/test_qwen36_tier_multidev.c), adapted to
 * Vulkan's fixed-name device API (coli_vk_*2/coli_vk_*3, not one function
 * parameterised by device index) via a tiny two-way dispatch. It also stops
 * refusing a row-wise int8 container (fmt=1, gs absent): F3 step 0
 * (tools/hot-expert/F3-STEP0-2026-09-20.md) measured int4-gs64 costing
 * quality against int8 (mean KL 0.0316, top-1 92.96% over 625 positions) and
 * found the int8 tier needs no new shader -- qmatmul*.comp already carries
 * fmt=1 -- only a second expert card, because int8's 32.34 GB does not fit
 * the 24.75 GB budget of one.
 *
 * Differences from the CUDA tier, deliberate and measured against this box:
 *
 *  - AT MOST TWO devices, dev2 and dev3, not the CUDA tier's up to eight.
 *    Qwen3.6-35B-A3B's expert set is at most 32.34 GB (row-wise int8; 18.12
 *    GB packed int4-gs64), which the CUDA tier's own header arithmetic
 *    (tools/hot-expert/F3-STEP0-2026-09-20.md Card 1) shows fits two 24 GB
 *    7900 XTX with room to spare -- a third device buys nothing on capacity
 *    and was not measured for anything else, so it is not wired here. dev0
 *    (the card that also hosts the trunk: attention, DeltaNet, lm_head in a
 *    future step) is DELIBERATELY EXCLUDED from the expert pool by default,
 *    matching the plan's step-1/step-2 split -- but home()/the per-device
 *    arrays below do not assume exactly two, so adding a third slot later is
 *    the same kind of small extension this port itself is, not a redesign.
 *  - dev2 is OPTIONAL and ADDITIVE (only tried when COLI_VK_DEV2 is set, the
 *    same convention glm53 and qwen38-vk use); dev3 stays MANDATORY exactly
 *    as V1 had it, so with only COLI_VK_DEV3 set (or unset -- it defaults to
 *    auto) this file activates exactly one device and its numerics are V1's,
 *    unchanged. This is the "default behaviour unchanged" requirement, not
 *    an incidental property: home(eid) = eid % 1 = 0 for every expert, the
 *    single-device budget/planning/issue code paths below are the same
 *    arithmetic V1 ran, one iteration of a loop that used to be inline code.
 *  - NO uploader thread and NO LFRU eviction, as V1. A Vulkan "upload" is a
 *    memcpy into mapped device memory inside coli_vk_tensor_ensure2/3 -- no
 *    async copy engine to hide behind a background thread -- and with the
 *    whole expert set resident there is nothing to evict. Preload across two
 *    devices runs SEQUENTIALLY, one device's budget filled from the heat
 *    order before the next device is considered (qt_plan_fill below), which
 *    is the pattern glm53's own preload uses for its three devices
 *    (vk_preload_tier, c/glm53.c: dev0's loop runs to completion, then
 *      dev2's, then dev3's -- device-level concurrency is a decode-time-only
 *      property in this codebase, not a preload one).
 *  - Row-wise int8 (fmt=1, expert_gs absent/0) is accepted, not refused: the
 *    format check now branches on expert_is_int4 the way the CUDA tier's
 *    qt_init already does (qwen36_tier.c, `if(G.wfmt==1 && expert_gs>0)`),
 *    instead of hard-refusing anything that is not the gs64 int4 container.
 *    Bytes-per-expert, the staging buffer size and the scale layout all
 *    follow G.wfmt now (see stage() and qt_init below) instead of assuming
 *    int4's D*Ih/2 packing everywhere.
 *  - The int4 nibble XOR in the CUDA tier's stage() is KEPT for int4, exactly
 *    as V1 had it, and is NOT applied to int8 (whole bytes, not nibbles --
 *    XORing them would be corruption, the same distinction the CUDA tier's
 *    own stage() draws between wfmt==1 and its int4 branch).
 *
 * Knobs (all off by default -- with Q36_VULKAN unset this file's qt_init
 * returns 0 and the engine is exactly the CPU engine):
 *   Q36_VULKAN=1               turn the tier on
 *   COLI_VK_DEV3=<idx>|auto    which Vulkan physical device hosts the tier's
 *                              mandatory device (default auto), as V1
 *   COLI_VK_DEV2=<idx>|auto    ADD a second expert-only device (unset = off,
 *                              V1's single-device behaviour); same convention
 *                              as glm53's COLI_VK_DEV2
 *   COLI_VK_EXPERTS3=<n>       cap dev3's resident expert count (0/unset =
 *                              budget only), as V1
 *   COLI_VK_EXPERTS2=<n>       cap dev2's resident expert count, same
 *                              convention as glm53's COLI_VK_EXPERTS2
 *   COLI_VK_TIER_RESERVE_GB=<g> VRAM to leave free on each tier device
 *                              (default 1.0, as glm53's G6 rule)
 *   COLI_VK_SHADERS=<dir|.spv> where qmatmul.spv and friends live
 *   HEAT_FILE=<path>           routing-heat table, read for fill order and rewritten at exit
 *   QT_NO_WARMSTART=1          (read by qwen36.c) skip the fill; the tier then holds nothing
 *   Q36_VK_TRUNK=1             F3 step 2a: also place lm_head and every DeltaNet
 *                              layer's fused qkv++z input projection on dev0 (the
 *                              device coli_vk_init(spv) above already brought up;
 *                              idle otherwise -- experts live on dev2/dev3 only).
 *                              Off by default: qt_place_of, qt_lmhead_init/matmul
 *                              and qt_dnproj_init/matmul then behave exactly as
 *                              before this knob existed (trunk stays on the
 *                              CPU). Row-wise int8 (fmt=1),
 *                              same q/sc bytes qwen36.c dense-i8 quantization
 *                              (qdw_register) already produces for the CPU path --
 *                              a REASSOCIATION of the same products, not a
 *                              different computation (see qwen36_tier.c own
 *                              R4 comment, this is a port of that mechanism).
 */
#ifdef Q36_VK_TIER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "qwen36_tier.h"
#include "backend_vulkan.h"
#include "tier.h"

#define QT_VK_MAX_DEV 2   /* dev2 + dev3; see the file header for why not more */

typedef struct {
    ColiVkTensor *tg, *tu, *td;
    uint32_t heat;
    uint8_t resident, planned;
    uint8_t dev;                    /* index into G.phys[], valid once planned/resident */
} QSlot;

static struct {
    int on;
    int nl, ne, D, Ih, topk;
    int egs;                       /* expert group size (64 on the gs64 container, 0 = per-row) */
    int wfmt;                      /* 4 = packed int4 (gs64), 1 = int8 per-row */
    size_t sc_gu, sc_d;            /* scale counts per matrix */
    size_t exp_bytes;              /* payload bytes one expert occupies in VRAM */
    int ndev;                      /* 1 or 2 active devices */
    int phys[QT_VK_MAX_DEV];       /* physical suffix, 2 or 3, in activation order */
    size_t budget[QT_VK_MAX_DEV], used[QT_VK_MAX_DEV];
    size_t dev_planned[QT_VK_MAX_DEV];
    int cap_count[QT_VK_MAX_DEV];  /* COLI_VK_EXPERTS2/3, 0 = no count cap on that device */
    int budget_stop[QT_VK_MAX_DEV];
    uint64_t uploads[QT_VK_MAX_DEV], upload_fail[QT_VK_MAX_DEV];
    uint64_t hits[QT_VK_MAX_DEV];
    double reserve_gb;
    QSlot *slot;                   /* [nl*ne] */
    int *fill_order; int fill_cur;
    uint32_t *heat0;
    pthread_mutex_t up_mx;         /* serialises coli_vk_tensor_ensure2/3, both devices */
    /* issue state of the single decode thread, per device */
    int is_cnt[QT_VK_MAX_DEV], is_k[QT_VK_MAX_DEV][32];
    float *is_x, *is_y;            /* [ndev][32][D] each */
    uint64_t miss;
} G;

static QSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
static int home(int eid){ return eid % G.ndev; }

/* ---- two-way dispatch: Vulkan's per-device API is three fixed function
 * names, not one function taking a device index (see backend_vulkan.h). The
 * CUDA tier's home()+per-device arrays generalise to any device count for
 * free because coli_cuda_* already takes a device argument; here the same
 * generalisation needs this small table instead. Two cases, not eight --
 * dev0 is not offered a slot (file header) -- so a switch is clearer than a
 * function-pointer table for two entries. */
static const char *vk_tag(int phys){ return phys == 2 ? "dev2" : "dev3"; }
static int vk_mem_budget(int phys, double *u, double *b){
    return phys == 2 ? coli_vk_mem_budget2(u,b) : coli_vk_mem_budget3(u,b);
}
static int vk_tensor_ensure(int phys, ColiVkTensor **t, const void *w, const float *sc,
                            int fmt, int I, int O, int grp){
    return phys == 2 ? coli_vk_tensor_ensure2(t,w,sc,fmt,I,O,grp)
                     : coli_vk_tensor_ensure3(t,w,sc,fmt,I,O,grp);
}
static int vk_group_issue(int phys, ColiVkTensor *const *g, ColiVkTensor *const *u,
                          ColiVkTensor *const *d, const int *rows, int n, const float *x){
    return phys == 2 ? coli_vk_expert_group_issue2(g,u,d,rows,n,x)
                     : coli_vk_expert_group_issue3(g,u,d,rows,n,x);
}
static int vk_group_take(int phys, float *y){
    return phys == 2 ? coli_vk_expert_group_take2(y) : coli_vk_expert_group_take3(y);
}

/* Staging: qwen36 keeps its packed int4 experts as TWO'S-COMPLEMENT nibbles
 * (qwen36.c's unpack_int4_to_int8 sign-extends: `(int8_t)(byte<<4)>>4`, and
 * c/tools/convert_qwen36.py packs them that way). The Vulkan upload API takes
 * OFFSET-BINARY nibbles (qmatmul.comp:47 and friends decode
 * i4(w,l) = (nibble & 0xf) - 8), so int4 experts get the XOR 0x88 conversion
 * here, exactly as V1 had it (see the file header and
 * tests/test_qwen36_vk_nibble.c). Row-wise int8 (wfmt==1) is already the
 * upload's byte layout -- a straight copy, no XOR, because XORing whole
 * int8 bytes the way the int4 nibble trick does would be corruption, not a
 * convention change (the CUDA tier's own stage() draws the same line between
 * its wfmt==1 and int4 branches, qwen36_tier.c). */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
                  const float *gs, const float *us, const float *ds){
    size_t mb = (G.wfmt == 4) ? (size_t)G.D*G.Ih/2 : (size_t)G.D*G.Ih;
    if(G.wfmt == 1){
        memcpy(dw,       g4, mb);
        memcpy(dw+mb,    u4, mb);
        memcpy(dw+2*mb,  d4, mb);
    } else {
        const uint64_t X = 0x8888888888888888ull;
        const uint64_t *sg=(const uint64_t*)g4, *su=(const uint64_t*)u4, *sd=(const uint64_t*)d4;
        uint64_t *w0=(uint64_t*)dw, *w1=(uint64_t*)(dw+mb), *w2=(uint64_t*)(dw+2*mb);
        for(size_t i=0;i<mb/8;i++){ w0[i]=sg[i]^X; w1[i]=su[i]^X; w2[i]=sd[i]^X; }
    }
    memcpy(dsc,             gs, G.sc_gu*sizeof(float));
    memcpy(dsc+G.sc_gu,     us, G.sc_gu*sizeof(float));
    memcpy(dsc+2*G.sc_gu,   ds, G.sc_d *sizeof(float));
}

/* ---- R4 trunk placement over dev0 (F3 step 2a) ---------------------------
 * A PORT of qwen36_tier.c's own qt_lmhead_init/matmul and qt_dnproj_init/
 * matmul onto this backend's dev0 entry points (coli_vk_matmul,
 * coli_vk_tensor_ensure -- the unsuffixed, single-device API glm53 already
 * drives its own dense matrices and its lm_head through, `mv()` in
 * c/glm53.c). dev0 is not part of the expert pool (dev2/dev3 only, file
 * header); its Vulkan context is already live by the time these run --
 * brought up by qt_init's own coli_vk_init(spv) call above, not a second
 * device bring-up. Off by default (Q36_VK_TRUNK unset): qt_place_of returns
 * QT_PLACE_CPU exactly as before this knob existed and the two _init
 * functions are no-ops, so qwen36.c's unconditional call sites leave the
 * trunk on the CPU, unchanged.
 *
 * Numerics: this REORDERS the same row-wise int8 dot products qwen36.c's
 * dense-i8 quantization already produces for the CPU path (qdw_register:
 * per-row scale, y[o] = acc*sc[o], the exact semantics fmt=1 applies) --
 * AVX2-lane summation on the CPU vs. the GPU's subgroup reduction, not a
 * different quantization or a different weight. Ships behind the knob per
 * CLAUDE.md's rule for any change that reorders a float sum. */
/* Read fresh every call, not cached: it is only consulted at load time
 * (qt_place_of, once per dnproj layer; the two _init entry points, once
 * each), never in the per-token matmul path, so there is no per-token cost
 * to weigh against a cache -- and a test process that runs qt_init more than
 * once with a different Q36_VK_TRUNK (this file's own fake-backend test)
 * gets the value it just set, not a first-call snapshot. */
static int qt_vk_trunk_on(void){
    const char *e = getenv("Q36_VK_TRUNK");
    return (e && atoi(e)) ? 1 : 0;
}
static struct { ColiVkTensor *t; int on; } G_lmh;
static struct { ColiVkTensor *t; int on; } *G_dnp;   /* [G.nl], calloc'd in qt_init */

int  qt_place_of(const char *component, int layer){
    (void)layer;
    if (!qt_vk_trunk_on() || !G.on) return QT_PLACE_CPU;
    if (!strcmp(component, "lmhead") || !strcmp(component, "dnproj")) return 0; /* dev0 */
    return QT_PLACE_CPU;
}
void qt_trunk_offer(const char *component, int layer, size_t bytes){ (void)component; (void)layer; (void)bytes; }

int  qt_lmhead_init(const int8_t *q, const float *sc, int I, int O){
    if (!qt_vk_trunk_on() || !G.on || !q || !sc) return 0;
    if (!coli_vk_tensor_ensure(&G_lmh.t, q, sc, 1, I, O, 0)) {
        fprintf(stderr, "[trunk-vk] lm_head upload to dev0 failed -> stays on CPU\n");
        return 0;
    }
    G_lmh.on = 1;
    fprintf(stderr, "[trunk-vk] lm_head [%d x %d] int8 resident on dev0 (%.2f GB)\n",
            O, I, (double)O * I / 1073741824.0);
    return 1;
}
int  qt_lmhead_matmul(float *y, const float *x, int I, int O){
    if (!G_lmh.on) return 0;
    if (coli_vk_matmul(&G_lmh.t, y, x, NULL, NULL, 1, 1, I, O, 0)) return 1;
    fprintf(stderr, "[trunk-vk] lm_head GPU matmul failed; falling back to CPU from here on\n");
    G_lmh.on = 0;
    return 0;
}
int  qt_dnproj_init(int layer, const int8_t *q, const float *sc, int I, int O, int device){
    (void)device;   /* one target (dev0); the CUDA tier's ordinal has nothing to select here */
    if (!qt_vk_trunk_on() || !G.on || !G_dnp || layer < 0 || layer >= G.nl || !q || !sc) return 0;
    if (!coli_vk_tensor_ensure(&G_dnp[layer].t, q, sc, 1, I, O, 0)) {
        fprintf(stderr, "[trunk-vk] dnproj layer %d upload to dev0 failed -> stays on CPU\n", layer);
        return 0;
    }
    G_dnp[layer].on = 1;
    return 1;
}
int  qt_dnproj_matmul(int layer, float *y, const float *x, int I, int O){
    if (!G_dnp || layer < 0 || layer >= G.nl || !G_dnp[layer].on) return 0;
    if (coli_vk_matmul(&G_dnp[layer].t, y, x, NULL, NULL, 1, 1, I, O, 0)) return 1;
    fprintf(stderr, "[trunk-vk] dnproj layer %d GPU matmul failed; CPU from here on\n", layer);
    G_dnp[layer].on = 0;
    return 0;
}
/* fp8 streaming is Qwen3.8's mode; this engine's experts are int4/int8 and resident. */
int  qt_init_fp8(int nl,int ne,int D,int Ih,int cap,int topk,const float *lut){
    (void)nl;(void)ne;(void)D;(void)Ih;(void)cap;(void)topk;(void)lut; return 0; }

int qt_init(int nl, int ne, int D, int Ih, int cap, int topk, int expert_gs,
            int expert_is_int4){
    const char *e = getenv("Q36_VULKAN");
    if(!(e && atoi(e))) return 0;                 /* default OFF: the CPU engine */
    if(cap != ne){
        fprintf(stderr,"[qtier-vk] cache/layer=%d != n_experts=%d -> tier disabled "
                       "(V1 needs every expert in RAM behind the resident set)\n", cap, ne);
        return 0;
    }
    if(topk > 32){ fprintf(stderr,"[qtier-vk] topk>32 unsupported\n"); return 0; }
    int wfmt;
    if(expert_is_int4){
        if(expert_gs <= 0){
            fprintf(stderr,"[qtier-vk] int4 container without group scales (expert_gs<=0) "
                           "-> tier disabled\n");
            return 0;
        }
        if((expert_gs & 7) || expert_gs < 8){
            fprintf(stderr,"[qtier-vk] expert_gs=%d is not a multiple of 8; the fmt=4 "
                           "shader path requires word-aligned groups -> tier disabled\n", expert_gs);
            return 0;
        }
        wfmt = 4;
    } else {
        if(expert_gs > 0){
            fprintf(stderr,"[qtier-vk] int8 experts with grouped scales (gs=%d) cannot be "
                           "expressed on the GPU (fmt=1 is per-row only) -> tier disabled\n",
                    expert_gs);
            return 0;
        }
        wfmt = 1;
    }
    if(D > 6144){
        fprintf(stderr,"[qtier-vk] hidden=%d exceeds the gate_up shader's xsh[6144] "
                       "staging array -> tier disabled\n", D);
        return 0;
    }
    memset(&G,0,sizeof G);
    G.nl=nl; G.ne=ne; G.D=D; G.Ih=Ih; G.topk=topk; G.egs=expert_gs; G.wfmt=wfmt;
    /* G_lmh/G_dnp are file-scope, not part of G -- qt_init can in principle run
     * more than once in a test process, so reset them here too. */
    G_lmh.t = NULL; G_lmh.on = 0;
    free(G_dnp); G_dnp = NULL;

    char spv[1024];
    const char *given = getenv("COLI_VK_SHADERS");
    if(given && strstr(given,".spv")) snprintf(spv,sizeof spv,"%s",given);
    else snprintf(spv,sizeof spv,"%s/qmatmul.spv", given ? given : "shaders");
    if(!coli_vk_init(spv)){
        fprintf(stderr,"[qtier-vk] coli_vk_init(%s) failed -> CPU path\n", spv);
        return 0;
    }

    /* dev2 is OPTIONAL and ADDITIVE (only tried when COLI_VK_DEV2 is set);
     * dev3 stays MANDATORY, exactly as V1. This ordering is what keeps "only
     * COLI_VK_DEV3 set" the unchanged default: G.ndev==1, home(eid)==0 for
     * every expert, one budget line, one device in every loop below. */
    G.ndev = 0;
    const char *d2 = getenv("COLI_VK_DEV2");
    if(d2 && *d2){
        int didx2 = !strcmp(d2,"auto") ? -1 : atoi(d2);
        if(coli_vk_init_dev2(spv, didx2) && coli_vk_dev2_available()){
            G.phys[G.ndev++] = 2;
        } else {
            fprintf(stderr,"[qtier-vk] dev2 (COLI_VK_DEV2=%s) not available -- continuing "
                           "without it\n", d2);
        }
    }
    const char *d3 = getenv("COLI_VK_DEV3");
    int didx3 = (!d3 || !*d3 || !strcmp(d3,"auto")) ? -1 : atoi(d3);
    if(!coli_vk_init_dev3(spv, didx3) || !coli_vk_dev3_available()){
        fprintf(stderr,"[qtier-vk] dev3 (COLI_VK_DEV3=%s) not available -> CPU path\n",
                d3 && *d3 ? d3 : "auto");
        return 0;
    }
    G.phys[G.ndev++] = 3;

    G.sc_gu = G.egs ? (size_t)Ih * (size_t)((D  + G.egs - 1)/G.egs) : (size_t)Ih;
    G.sc_d  = G.egs ? (size_t)D  * (size_t)((Ih + G.egs - 1)/G.egs) : (size_t)D;
    /* Charged by PAYLOAD, not by an allocator-granularity curve -- see V1's
     * measurement (record V1: predicted 18.12 GB, driver reported 18.26,
     * 0.8% gap, RADV's suballocating arena does not need CUDA's curve). Bytes
     * per expert follow wfmt: int8 (wfmt=1) is one byte/element, int4
     * (wfmt=4) is packed two nibbles/byte. */
    size_t mat_bytes = (G.wfmt == 4) ? (size_t)D*Ih/2 : (size_t)D*Ih;
    G.exp_bytes = 3*mat_bytes + (2*G.sc_gu + G.sc_d)*sizeof(float);

    G.reserve_gb = 1.0;
    { const char *r=getenv("COLI_VK_TIER_RESERVE_GB");
      if(r){ double v=atof(r); if(v>=0.0) G.reserve_gb=v; } }

    for(int i=0;i<G.ndev;i++){
        const char *envname = G.phys[i]==2 ? "COLI_VK_EXPERTS2" : "COLI_VK_EXPERTS3";
        const char *c = getenv(envname);
        G.cap_count[i] = c ? atoi(c) : 0;

        double used_gb=0, budget_gb=0;
        int have = vk_mem_budget(G.phys[i], &used_gb, &budget_gb);
        if(have && budget_gb > G.reserve_gb)
            G.budget[i] = (size_t)((budget_gb - used_gb - G.reserve_gb) * 1e9);
        else
            G.budget[i] = 0;
        fprintf(stderr,"[qtier-vk] %s: %.1f of %.1f GB used, reserve %.1f -> budget %.2f GB "
                       "(~%zu experts at %.2f MB each)\n",
                vk_tag(G.phys[i]), used_gb, budget_gb, G.reserve_gb, G.budget[i]/1e9,
                G.exp_bytes ? G.budget[i]/G.exp_bytes : 0, G.exp_bytes/1048576.0);
    }

    G.slot = calloc((size_t)nl*ne, sizeof(QSlot));
    if(!G.slot) return 0;
    G_dnp = calloc((size_t)nl, sizeof *G_dnp);   /* trunk dnproj slots, one per model layer */
    if(!G_dnp){ free(G.slot); G.slot=NULL; return 0; }
    const char *hf = getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"rb");
        if(f){
            uint32_t hdr[3]={0,0,0};
            if(fread(hdr,4,3,f)==3 && hdr[0]==0x51544831u && hdr[1]==(uint32_t)nl && hdr[2]==(uint32_t)ne){
                G.heat0=malloc((size_t)nl*ne*4);
                if(G.heat0 && fread(G.heat0,4,(size_t)nl*ne,f)==(size_t)nl*ne){
                    for(size_t i=0;i<(size_t)nl*ne;i++) G.slot[i].heat=G.heat0[i]>>1;   /* decay */
                    fprintf(stderr,"[qtier-vk] HEAT_FILE loaded: %s\n",hf);
                } else { free(G.heat0); G.heat0=NULL; }
            }
            fclose(f);
        }
        if(!G.heat0) fprintf(stderr,"[qtier-vk] no usable HEAT_FILE at %s -- filling in natural order\n",hf);
    } else {
        fprintf(stderr,"[qtier-vk] HEAT_FILE unset -- filling in natural order\n");
    }

    G.is_x = malloc((size_t)G.ndev*32*D*sizeof(float));
    G.is_y = malloc((size_t)G.ndev*32*D*sizeof(float));
    if(!G.is_x || !G.is_y){ free(G.is_x); free(G.is_y); free(G.slot); G.slot=NULL; return 0; }
    pthread_mutex_init(&G.up_mx,NULL);
    G.on = 1;
    { char devlist[32]; devlist[0]=0;
      for(int i=0;i<G.ndev;i++){
          char tmp[16]; snprintf(tmp,sizeof tmp,"%s%s", i?"+":"", vk_tag(G.phys[i]));
          strncat(devlist,tmp,sizeof(devlist)-strlen(devlist)-1);
      }
      fprintf(stderr,"[qtier-vk] Vulkan VRAM expert tier active on %s: fmt=%d gs=%d, "
                     "%d x %d experts, %.2f MB/expert\n",
              devlist, G.wfmt, G.egs, nl, ne, G.exp_bytes/1048576.0);
    }
    return 1;
}

int qt_ready(void){ return G.on; }

int qt_is_resident(int layer,int eid){
    if(!G.on) return 0;
    return qs(layer,eid)->resident;
}

/* Heat only. The CUDA tier promotes lazily from here through a background
 * uploader; this one has no such thread (see the file header), so a miss stays
 * a miss for the run and the counted hit rate is what the warmstart achieved. */
void qt_note(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    (void)g4;(void)u4;(void)d4;(void)gs;(void)us;(void)ds;
    if(!G.on || layer<0 || eid<0 || layer>=G.nl || eid>=G.ne) return;
    QSlot *s=qs(layer,eid);
    if(s->heat<0xFFFFFFFFu) s->heat++;
}
/* Declared by the header and not called by qwen36.c (which uses qt_plan_fill +
 * qt_note_planned exclusively). Kept as the heat-only form rather than deleted,
 * so the header contract stays satisfied and nothing acquires a second, untested
 * path into the uploader. */
void qt_note_block(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    qt_note(layer,eid,g4,u4,d4,gs,us,ds);
}

/* Fill order: heat descending when a HEAT_FILE was loaded, natural order
 * otherwise, with (layer,eid) as a tie-break so the planned set is the same on
 * every run -- an unstable sort here would make the resident set, and with it
 * the numerics of a budget-limited run, differ between the two arms of an A/B. */
static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    int ia=*(const int*)a, ib=*(const int*)b;
    uint32_t ha=g_sort_heat[ia], hb=g_sort_heat[ib];
    if(ha!=hb) return ha<hb ? 1 : -1;
    return ia<ib ? -1 : ia>ib ? 1 : 0;
}

/* Is device index di full, either on its live budget or its count cap
 * (COLI_VK_EXPERTS2/3)? Shared by the fill loop and its "give up entirely"
 * check below. */
static int dev_full(int di){
    if(G.used[di] + G.exp_bytes > G.budget[di]) return 1;
    if(G.cap_count[di] > 0 && G.dev_planned[di] >= (size_t)G.cap_count[di]) return 1;
    return 0;
}

int qt_plan_fill(int *layers,int *eids,int max){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    int cnt=0;
    if(!G.fill_order){
        G.fill_order=malloc(n*sizeof(int));
        if(!G.fill_order) return 0;
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        if(G.heat0){ g_sort_heat=G.heat0; qsort(G.fill_order,n,sizeof(int),cmp_heat_desc); }
        G.fill_cur=0;
    }
    while((size_t)G.fill_cur<n && cnt<max){
        int gi=G.fill_order[G.fill_cur];
        int l=gi/G.ne, e=gi%G.ne;
        int di=home(e);
        QSlot *s=qs(l,e);
        if(s->resident||s->planned){ G.fill_cur++; continue; }
        if(dev_full(di)){
            /* This device is done; another might not be -- keep scanning the
             * fill order (an eid homed elsewhere may still fit) instead of
             * stopping the whole plan, which single-device V1 could do
             * because there was nowhere else for an expert to go. Only stop
             * once EVERY device is full. */
            int all_full=1;
            for(int d=0; d<G.ndev; d++) if(!dev_full(d)){ all_full=0; break; }
            if(all_full) break;
            G.fill_cur++;
            continue;
        }
        G.fill_cur++;
        G.used[di] += G.exp_bytes;
        G.dev_planned[di]++;
        s->planned=1; s->dev=(uint8_t)di;
        layers[cnt]=l; eids[cnt]=e; cnt++;
    }
    if(max > 1){
        size_t tot_used=0, tot_budget=0;
        for(int d=0; d<G.ndev; d++){ tot_used+=G.used[d]; tot_budget+=G.budget[d]; }
        fprintf(stderr,"[qtier-vk] planned %d experts (%.2f GB of a %.2f GB budget total across "
                       "%d device%s)\n",
                cnt, tot_used/1e9, tot_budget/1e9, G.ndev, G.ndev>1?"s":"");
        for(int d=0; d<G.ndev; d++)
            fprintf(stderr,"[qtier-vk] %s: planned %zu experts, %.2f of %.2f GB%s\n",
                    vk_tag(G.phys[d]), G.dev_planned[d], G.used[d]/1e9, G.budget[d]/1e9,
                    G.cap_count[d]>0 ? " (count-capped)" : "");
    }
    return cnt;
}

int qt_fill_next(int *layer,int *eid){
    int l[1], e[1];
    if(qt_plan_fill(l,e,1) != 1) return 0;
    *layer=l[0]; *eid=e[0];
    return 1;
}

/* Stage + upload one planned expert. Called from qwen36.c's parallel warmstart
 * loop: the container read and int4/int8 unpack run on many threads, the
 * upload itself is serialised (one mutex across BOTH devices, exactly as V1's
 * single mutex serialised the one device it had) because backend_vulkan's
 * suballocating arena is not re-entrant per device and glm53's own multi-
 * device preload (vk_preload_tier, c/glm53.c) never uploads to two devices
 * concurrently either -- device-level overlap in this codebase is a
 * decode-time (issue/take) property, not a preload one. Which device an
 * expert lands on does not depend on thread order -- qt_plan_fill decided
 * that, deterministically, before any thread ran. */
void qt_note_planned(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || layer<0 || eid<0 || layer>=G.nl || eid>=G.ne) return;
    QSlot *s=qs(layer,eid);
    if(!s->planned) return;
    int di = s->dev;
    if(!g4 || !u4 || !d4 || !gs || !us || !ds){
        /* Nothing to upload: hand the reservation back, exactly as the CUDA
         * tier does, or the bytes stay out of the budget for the whole run
         * (#1331's shape). */
        pthread_mutex_lock(&G.up_mx);
        if(s->planned){ if(G.used[di]>=G.exp_bytes) G.used[di]-=G.exp_bytes; s->planned=0; }
        pthread_mutex_unlock(&G.up_mx);
        return;
    }
    size_t mb=(G.wfmt==4) ? (size_t)G.D*G.Ih/2 : (size_t)G.D*G.Ih;
    uint8_t *w=malloc(3*mb);
    float *sc=malloc((2*G.sc_gu+G.sc_d)*sizeof(float));
    if(!w||!sc){ free(w); free(sc); return; }
    stage(w,sc,g4,u4,d4,gs,us,ds);

    pthread_mutex_lock(&G.up_mx);
    if(G.budget_stop[di]){ pthread_mutex_unlock(&G.up_mx); free(w); free(sc); return; }
    /* G6's rule, ported: stop on the LIVE VRAM budget as well as on the
     * planned byte count, per device. These are device-local allocations
     * that can spill to host RAM over ReBAR rather than fail, so without
     * this a mis-estimated exp_bytes would quietly move the page cache out
     * of RAM instead of stopping -- which is the 91 GB bug G6 fixed for
     * glm53. */
    if((G.uploads[di] & 7) == 0){
        double u=0,b=0;
        if(vk_mem_budget(G.phys[di],&u,&b) && (b-u) < G.reserve_gb){
            fprintf(stderr,"[qtier-vk] preload %s: stopping on VRAM budget "
                           "(%.1f of %.1f GB used, %.1f reserve) after %llu experts\n",
                    vk_tag(G.phys[di]),u,b,G.reserve_gb,(unsigned long long)G.uploads[di]);
            G.budget_stop[di]=1;
            pthread_mutex_unlock(&G.up_mx); free(w); free(sc); return;
        }
    }
    ColiVkTensor *tg=NULL,*tu=NULL,*td=NULL;
    int ok = vk_tensor_ensure(G.phys[di], &tg, w,        sc,             G.wfmt, G.D,  G.Ih, G.egs)
          && vk_tensor_ensure(G.phys[di], &tu, w+mb,     sc+G.sc_gu,     G.wfmt, G.D,  G.Ih, G.egs)
          && vk_tensor_ensure(G.phys[di], &td, w+2*mb,   sc+2*G.sc_gu,   G.wfmt, G.Ih, G.D,  G.egs);
    if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads[di]++; }
    else {
        if(tg) coli_vk_tensor_free(tg);
        if(tu) coli_vk_tensor_free(tu);
        if(td) coli_vk_tensor_free(td);
        if(!G.upload_fail[di])
            fprintf(stderr,"[qtier-vk] upload refused at expert %d/%d on %s -- the rest stay "
                           "on the CPU\n", layer, eid, vk_tag(G.phys[di]));
        G.upload_fail[di]++;
        G.budget_stop[di]=1;                     /* this card is full: stop trying */
    }
    s->planned=0;
    pthread_mutex_unlock(&G.up_mx);
    free(w); free(sc);
}

void qt_fill_wait(void){ /* uploads are synchronous: nothing in flight */ }

/* Launch the GPU groups for the resident subset of K, on EVERY involved
 * device before any of them is taken (glm53, c/glm53.c: "issue every
 * device's chunk first, join afterward" -- so the (at most two) devices run
 * concurrently with each other instead of one after the other; qwen36_tier.c
 * ports the same shape for CUDA). A layer's up-to-32 chosen experts sort
 * into at most G.ndev buckets by home(eid); this issues one group call per
 * non-empty bucket. */
uint32_t qt_issue(int layer,const int *eids,int K,const float *x){
    if(!G.on || K>32 || K<1) return 0;
    ColiVkTensor *tg[QT_VK_MAX_DEV][32], *tu[QT_VK_MAX_DEV][32], *td[QT_VK_MAX_DEV][32];
    static int rows[32];
    if(!rows[0]) for(int i=0;i<32;i++) rows[i]=1;
    uint32_t mask=0;
    for(int d=0; d<G.ndev; d++) G.is_cnt[d]=0;
    for(int k=0;k<K;k++){
        int e=eids[k];
        if(e<0||e>=G.ne){ G.miss++; continue; }
        QSlot *s=qs(layer,e);
        if(!s->resident){ G.miss++; continue; }
        int d=s->dev;
        int c=G.is_cnt[d];
        tg[d][c]=s->tg; tu[d][c]=s->tu; td[d][c]=s->td;
        G.is_k[d][c]=k; G.is_cnt[d]=c+1;
        mask |= 1u<<k; G.hits[d]++;
    }
    if(!mask) return 0;
    for(int d=0; d<G.ndev; d++){
        int c=G.is_cnt[d];
        if(!c) continue;
        float *xr = G.is_x + (size_t)d*32*G.D;
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        if(!vk_group_issue(G.phys[d], tg[d], tu[d], td[d], rows, c, xr)){
            /* hand these k back to the CPU for this token */
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[d][j]);
            G.hits[d] -= (uint64_t)c; G.miss += (uint64_t)c;
            G.is_cnt[d]=0;
        }
    }
    return mask;
}

void qt_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    if(!G.on || !mask) return;
    for(int d=0; d<G.ndev; d++){
        int c=G.is_cnt[d];
        if(!c) continue;
        float *yb = G.is_y + (size_t)d*32*G.D;
        if(!vk_group_take(G.phys[d], yb)){
            /* qt_issue already told the engine these k were handled, so the
             * caller did NOT compute them on the CPU: their contribution is
             * now missing from this token. Say so once and loudly rather
             * than returning a quietly wrong hidden state -- the fence only
             * fails when the device is lost, and a silent drop would
             * survive every text oracle that happens to sample the same
             * argmax. */
            static int said=0;
            if(!said){ said=1;
                fprintf(stderr,"[qtier-vk] expert-group take FAILED on %s -- %d expert(s) "
                               "dropped from this token; output from here on is NOT "
                               "trustworthy\n", vk_tag(G.phys[d]), c); }
            G.hits[d] -= (uint64_t)c; G.miss += (uint64_t)c;
            G.is_cnt[d]=0;
            continue;
        }
        for(int j=0;j<c;j++){
            float w=val[G.is_k[d][j]];
            const float *row=yb + (size_t)j*G.D;
            for(int dd=0; dd<G.D; dd++) out[dd]+=w*row[dd];
        }
        G.is_cnt[d]=0;
    }
}

void qt_stats(void){
    if(!G.on) return;
    size_t res_dev[QT_VK_MAX_DEV] = {0};
    size_t total_res=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++){
        if(!G.slot[i].resident) continue;
        total_res++;
        res_dev[G.slot[i].dev]++;
    }
    uint64_t tot_uploads=0, tot_fail=0, tot_hits=0; int any_stop=0;
    for(int d=0; d<G.ndev; d++){
        double u=0,b=0; int have=vk_mem_budget(G.phys[d],&u,&b);
        fprintf(stderr,"[qtier-vk] %s: resident %zu | uploads %llu | refused %llu | "
                       "budget-stop %d\n",
                vk_tag(G.phys[d]), res_dev[d], (unsigned long long)G.uploads[d],
                (unsigned long long)G.upload_fail[d], G.budget_stop[d]);
        fprintf(stderr,"[qtier-vk] %s planned %.2f GB, driver reports %.2f of %.2f GB used%s\n",
                vk_tag(G.phys[d]), G.used[d]/1e9, u, b,
                have ? "" : " (VK_EXT_memory_budget absent: figures unavailable)");
        tot_uploads += G.uploads[d]; tot_fail += G.upload_fail[d]; tot_hits += G.hits[d];
        any_stop |= G.budget_stop[d];
    }
    fprintf(stderr,"[qtier-vk] resident %zu/%d experts | uploads %llu | refused %llu | "
                   "budget-stop %d\n",
            total_res, G.nl*G.ne, (unsigned long long)tot_uploads,
            (unsigned long long)tot_fail, any_stop);
    double tot=(double)(tot_hits+G.miss);
    fprintf(stderr,"[qtier-vk] VRAM hit rate: %.1f %% (hits %llu, CPU misses %llu)\n",
            tot>0 ? 100.0*(double)tot_hits/tot : 0.0,
            (unsigned long long)tot_hits, (unsigned long long)G.miss);
}

void qt_shutdown(void){
    if(!G.on) return;
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"wb");
        if(f){
            uint32_t hdr[3]={0x51544831u,(uint32_t)G.nl,(uint32_t)G.ne};
            fwrite(hdr,4,3,f);
            for(size_t i=0;i<(size_t)G.nl*G.ne;i++) fwrite(&G.slot[i].heat,4,1,f);
            fclose(f);
            fprintf(stderr,"[qtier-vk] HEAT_FILE saved: %s\n",hf);
        }
    }
    G.on=0;
    /* The device tensors are not freed one by one: 30 720 of them would cost
     * more at exit than the kernel's own address-space teardown, and the VRAM
     * is released when the process dies either way (the chain asserts
     * mem_info_vram_used < 1 GB on every card after the engine exits). */
}

#endif /* Q36_VK_TIER */
