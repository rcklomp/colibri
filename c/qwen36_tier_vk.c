/* qwen36_tier_vk.c -- Vulkan VRAM expert tier for the qwen36 engine (V1).
 *
 * The same qt_* contract qwen36_tier.c implements over CUDA, implemented over
 * the Vulkan backend instead, so qwen36.c's call sites do not move. Built only
 * into the separate `qwen36-vk` binary (-DQ36_VK_TIER, VK=1); `make qwen36`
 * stays CPU-only and links no Vulkan.
 *
 * Differences from the CUDA tier, deliberate and measured against this box:
 *
 *  - ONE device. Qwen3.6-35B-A3B's 40 x 256 int4-gs64 experts are ~18.6 GB
 *    packed, which fits on one 24 GB 7900 XTX with the 1.0 GB reserve glm53's
 *    G6 uses on its expert-only cards. So there is no device loop and no
 *    home(eid) hash: everything lands on the card COLI_VK_DEV3 names, which on
 *    this rig is Vulkan enumeration index 2 = PCI 0000:86:00.0 (record SQ13).
 *    The Vulkan expert-group API is three fixed functions (_issue/_issue2/
 *    _issue3), not one parameterised by device, so a device loop would have
 *    needed a dispatch table for no benefit here.
 *
 *  - NO uploader thread and NO LFRU eviction. A Vulkan "upload" is a memcpy
 *    into mapped device memory inside coli_vk_tensor_ensure3 -- there is no
 *    async copy engine to hide behind a background thread, and with the whole
 *    expert set resident there is nothing to evict. Dropping both removes the
 *    CUDA tier's staging queue, its victim-in-flight interlock and its
 *    heat-swap tick; what stays is the heat table (HEAT_FILE, same on-disk
 *    format) deciding the FILL ORDER, which is all it can decide when the
 *    budget holds everything. If the budget ever stops short, the experts past
 *    the stop run on the CPU for the life of the process and qt_stats says so.
 *
 *  - The int4 nibble XOR in the CUDA tier's stage() is KEPT, not dropped.
 *    See stage() below: both backends take OFFSET-BINARY nibbles at their
 *    upload API, and qwen36's RAM copy is two's-complement.
 *
 * Knobs (all off by default -- with Q36_VULKAN unset this file's qt_init
 * returns 0 and the engine is exactly the CPU engine):
 *   Q36_VULKAN=1               turn the tier on
 *   COLI_VK_DEV3=<idx>|auto    which Vulkan physical device hosts it (default auto)
 *   COLI_VK_EXPERTS3=<n>       cap the resident expert count (0/unset = budget only)
 *   COLI_VK_TIER_RESERVE_GB=<g> VRAM to leave free on that card (default 1.0, as glm53)
 *   COLI_VK_SHADERS=<dir|.spv> where qmatmul.spv and friends live
 *   HEAT_FILE=<path>           routing-heat table, read for fill order and rewritten at exit
 *   QT_NO_WARMSTART=1          (read by qwen36.c) skip the fill; the tier then holds nothing
 */
#ifdef Q36_VK_TIER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "qwen36_tier.h"
#include "backend_vulkan.h"
#include "tier.h"

typedef struct {
    ColiVkTensor *tg, *tu, *td;
    uint32_t heat;
    uint8_t resident, planned;
} QSlot;

static struct {
    int on;
    int nl, ne, D, Ih, topk;
    int egs;                       /* expert group size (64 on the gs64 container) */
    size_t sc_gu, sc_d;            /* scale counts per matrix */
    size_t exp_bytes;              /* payload bytes one expert occupies in VRAM */
    size_t budget, used;           /* byte budget for the planner */
    int cap_count;                 /* COLI_VK_EXPERTS3, 0 = no count cap */
    double reserve_gb;
    QSlot *slot;                   /* [nl*ne] */
    int *fill_order; int fill_cur;
    uint32_t *heat0;
    pthread_mutex_t up_mx;         /* serialises coli_vk_tensor_ensure3 */
    /* issue state of the single decode thread */
    int is_cnt, is_k[32];
    float *is_x, *is_y;            /* 32*D each */
    uint64_t hits, miss, uploads, upload_fail;
    int budget_stop;               /* the live VRAM read stopped the fill */
} G;

static QSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }

/* Staging: qwen36 keeps its packed int4 experts as TWO'S-COMPLEMENT nibbles
 * (qwen36.c's unpack_int4_to_int8 sign-extends: `(int8_t)(byte<<4)>>4`, and
 * c/tools/convert_qwen36.py packs them that way). Both GPU backends take
 * OFFSET-BINARY nibbles at their upload API:
 *
 *   CUDA  coli_cuda_tensor_upload(fmt=2|4) runs offset_to_signed_s4 -- an
 *         XOR 0x88 kernel -- over the bytes right after the H2D copy
 *         (backend_cuda.cu:1428), and weight_at() then sign-extends.
 *   VULKAN upload_tensor copies the nibbles verbatim (backend_vulkan.c:912)
 *         and every shader decodes them with i4(w,l) = (nibble & 0xf) - 8
 *         (qmatmul.comp:47, qmatmul_tile.comp:48, qmatmul_gate_up.comp:33,
 *         qmatmul_gate_up_tile.comp:27) -- i.e. offset binary, which is what
 *         that file's own header comment calls "int4 (nibble-8)". glm53 hands
 *         its q4 straight to coli_vk_tensor_ensure because ITS container is
 *         already offset binary (quant.h decodes `(b&0xF)-8`).
 *
 * So the XOR 0x88 the CUDA tier does here is NOT a CUDA-specific step and must
 * NOT be dropped for Vulkan. tools/hot-expert/V1-STEP0-2026-09-16.md sections
 * (c), (d) and (e) say the opposite; that reading took the shader's `- 8` for a
 * two's-complement decode and looked only at backend_cuda.cu's weight_at(),
 * not at the XOR its upload path performs on the device. Dropping the XOR
 * would offset every expert weight by 8 quanta and is exactly the class of bug
 * V1-STEP0 section (f) warned would "pass a coarse runs-and-looks-plausible
 * check and fail teacher_forcing". tests/test_qwen36_vk_nibble.c proves the
 * round trip over all 256 byte values. */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
                  const float *gs, const float *us, const float *ds){
    size_t mb = (size_t)G.D*G.Ih/2;
    const uint64_t X = 0x8888888888888888ull;
    const uint64_t *sg=(const uint64_t*)g4, *su=(const uint64_t*)u4, *sd=(const uint64_t*)d4;
    uint64_t *w0=(uint64_t*)dw, *w1=(uint64_t*)(dw+mb), *w2=(uint64_t*)(dw+2*mb);
    for(size_t i=0;i<mb/8;i++){ w0[i]=sg[i]^X; w1[i]=su[i]^X; w2[i]=sd[i]^X; }
    memcpy(dsc,             gs, G.sc_gu*sizeof(float));
    memcpy(dsc+G.sc_gu,     us, G.sc_gu*sizeof(float));
    memcpy(dsc+2*G.sc_gu,   ds, G.sc_d *sizeof(float));
}

/* ---- the parts of the qt_* contract V1 does not implement ----------------
 * The trunk placement (R4: lm_head and the DeltaNet projections on their own
 * device) is a second, independent mechanism in qwen36_tier.c. V1's scope is
 * the EXPERT tier; these stubs keep every one of qwen36.c's call sites valid
 * and keep the trunk on the CPU, which is where it is on this binary today. */
int  qt_place_of(const char *component, int layer){ (void)component; (void)layer; return QT_PLACE_CPU; }
void qt_trunk_offer(const char *component, int layer, size_t bytes){ (void)component; (void)layer; (void)bytes; }
int  qt_lmhead_init(const int8_t *q, const float *sc, int I, int O){ (void)q;(void)sc;(void)I;(void)O; return 0; }
int  qt_lmhead_matmul(float *y, const float *x, int I, int O){ (void)y;(void)x;(void)I;(void)O; return 0; }
int  qt_dnproj_init(int layer, const int8_t *q, const float *sc, int I, int O, int device){
    (void)layer;(void)q;(void)sc;(void)I;(void)O;(void)device; return 0; }
int  qt_dnproj_matmul(int layer, float *y, const float *x, int I, int O){
    (void)layer;(void)y;(void)x;(void)I;(void)O; return 0; }
/* fp8 streaming is Qwen3.8's mode; this engine's experts are int4 and resident. */
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
    if(!expert_is_int4 || expert_gs <= 0){
        /* Vulkan fmt=1 (int8, per-row scales) exists, but V1 is specified and
         * measured on the int4-gs64 container and nothing else has been proved
         * here; refusing loudly beats a silent second numeric path. */
        fprintf(stderr,"[qtier-vk] container is %s -> tier disabled (V1 serves packed "
                       "int4 with group scales, fmt=4)\n",
                expert_is_int4 ? "int4 with per-row scales" : "int8");
        return 0;
    }
    if((expert_gs & 7) || expert_gs < 8){
        fprintf(stderr,"[qtier-vk] expert_gs=%d is not a multiple of 8; the fmt=4 "
                       "shader path requires word-aligned groups -> tier disabled\n", expert_gs);
        return 0;
    }
    if(D > 6144){
        fprintf(stderr,"[qtier-vk] hidden=%d exceeds the gate_up shader's xsh[6144] "
                       "staging array -> tier disabled\n", D);
        return 0;
    }
    memset(&G,0,sizeof G);
    G.nl=nl; G.ne=ne; G.D=D; G.Ih=Ih; G.topk=topk; G.egs=expert_gs;

    char spv[1024];
    const char *given = getenv("COLI_VK_SHADERS");
    if(given && strstr(given,".spv")) snprintf(spv,sizeof spv,"%s",given);
    else snprintf(spv,sizeof spv,"%s/qmatmul.spv", given ? given : "shaders");
    if(!coli_vk_init(spv)){
        fprintf(stderr,"[qtier-vk] coli_vk_init(%s) failed -> CPU path\n", spv);
        return 0;
    }
    /* The tier device. coli_vk_init_dev3 needs device 0 up first (it shares the
     * instance), which is what the call above did; device 0 holds nothing of
     * ours. Default auto = the best real GPU that is not device 0; this rig
     * passes COLI_VK_DEV3=2 to name PCI 0000:86:00.0 explicitly. */
    const char *dv = getenv("COLI_VK_DEV3");
    int didx = (!dv || !*dv || !strcmp(dv,"auto")) ? -1 : atoi(dv);
    if(!coli_vk_init_dev3(spv, didx) || !coli_vk_dev3_available()){
        fprintf(stderr,"[qtier-vk] dev3 (COLI_VK_DEV3=%s) not available -> CPU path\n",
                dv && *dv ? dv : "auto");
        return 0;
    }

    G.sc_gu = (size_t)Ih * (size_t)((D  + expert_gs - 1)/expert_gs);
    G.sc_d  = (size_t)D  * (size_t)((Ih + expert_gs - 1)/expert_gs);
    /* Charged by PAYLOAD, not by an allocator-granularity curve. The CUDA tier
     * carries dev_alloc_footprint because cudaMalloc rounds to 2 MiB above
     * 1 MiB; RADV's rounding under this backend's suballocating arena is
     * unmeasured, so rather than port a curve that may not transfer, the
     * planner charges bytes and the LIVE budget read below (G6's rule, every
     * 8 uploads) is the authority that stops the fill. qt_stats() prints both
     * the planned and the driver-reported figure so the gap is visible. */
    G.exp_bytes = 3*((size_t)D*Ih/2) + (2*G.sc_gu + G.sc_d)*sizeof(float);

    G.reserve_gb = 1.0;
    { const char *r=getenv("COLI_VK_TIER_RESERVE_GB");
      if(r){ double v=atof(r); if(v>=0.0) G.reserve_gb=v; } }
    { const char *c3=getenv("COLI_VK_EXPERTS3"); G.cap_count = c3 ? atoi(c3) : 0; }

    double used_gb=0, budget_gb=0;
    if(coli_vk_mem_budget3(&used_gb,&budget_gb) && budget_gb > G.reserve_gb)
        G.budget = (size_t)((budget_gb - used_gb - G.reserve_gb) * 1e9);
    else
        G.budget = 0;
    fprintf(stderr,"[qtier-vk] dev3: %.1f of %.1f GB used, reserve %.1f -> budget %.2f GB "
                   "(~%zu experts at %.2f MB each)\n",
            used_gb, budget_gb, G.reserve_gb, G.budget/1e9,
            G.exp_bytes ? G.budget/G.exp_bytes : 0, G.exp_bytes/1048576.0);

    G.slot = calloc((size_t)nl*ne, sizeof(QSlot));
    if(!G.slot) return 0;
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

    G.is_x = malloc((size_t)32*D*sizeof(float));
    G.is_y = malloc((size_t)32*D*sizeof(float));
    if(!G.is_x || !G.is_y){ free(G.is_x); free(G.is_y); free(G.slot); G.slot=NULL; return 0; }
    pthread_mutex_init(&G.up_mx,NULL);
    G.on = 1;
    fprintf(stderr,"[qtier-vk] Vulkan VRAM expert tier active on dev3: fmt=4 gs=%d, "
                   "%d x %d experts, %.2f MB/expert\n", G.egs, nl, ne, G.exp_bytes/1048576.0);
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
        if(G.used + G.exp_bytes > G.budget) break;
        if(G.cap_count > 0 && cnt >= G.cap_count) break;
        int gi=G.fill_order[G.fill_cur++];
        int l=gi/G.ne, e=gi%G.ne;
        QSlot *s=qs(l,e);
        if(s->resident||s->planned) continue;
        G.used += G.exp_bytes;
        s->planned=1;
        layers[cnt]=l; eids[cnt]=e; cnt++;
    }
    if(max > 1)
        fprintf(stderr,"[qtier-vk] planned %d experts (%.2f GB of a %.2f GB budget%s)\n",
                cnt, G.used/1e9, G.budget/1e9,
                G.cap_count>0 ? ", count-capped by COLI_VK_EXPERTS3" : "");
    return cnt;
}

int qt_fill_next(int *layer,int *eid){
    int l[1], e[1];
    if(qt_plan_fill(l,e,1) != 1) return 0;
    *layer=l[0]; *eid=e[0];
    return 1;
}

/* Stage + upload one planned expert. Called from qwen36.c's parallel warmstart
 * loop: the container read and int4 unpack run on many threads, the upload
 * itself is serialised because backend_vulkan's suballocating arena is not
 * re-entrant. Which experts are resident does not depend on the thread order
 * -- qt_plan_fill decided that, deterministically, before any thread ran. */
void qt_note_planned(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || layer<0 || eid<0 || layer>=G.nl || eid>=G.ne) return;
    QSlot *s=qs(layer,eid);
    if(!s->planned) return;
    if(!g4 || !u4 || !d4 || !gs || !us || !ds){
        /* Nothing to upload: hand the reservation back, exactly as the CUDA
         * tier does, or the bytes stay out of the budget for the whole run
         * (#1331's shape). */
        pthread_mutex_lock(&G.up_mx);
        if(s->planned){ if(G.used>=G.exp_bytes) G.used-=G.exp_bytes; s->planned=0; }
        pthread_mutex_unlock(&G.up_mx);
        return;
    }
    size_t mb=(size_t)G.D*G.Ih/2;
    uint8_t *w=malloc(3*mb);
    float *sc=malloc((2*G.sc_gu+G.sc_d)*sizeof(float));
    if(!w||!sc){ free(w); free(sc); return; }
    stage(w,sc,g4,u4,d4,gs,us,ds);

    pthread_mutex_lock(&G.up_mx);
    if(G.budget_stop){ pthread_mutex_unlock(&G.up_mx); free(w); free(sc); return; }
    /* G6's rule, ported: stop on the LIVE VRAM budget as well as on the
     * planned byte count. These are device-local allocations that can spill to
     * host RAM over ReBAR rather than fail, so without this a mis-estimated
     * exp_bytes would quietly move the page cache out of RAM instead of
     * stopping -- which is the 91 GB bug G6 fixed for glm53. */
    if((G.uploads & 7) == 0){
        double u=0,b=0;
        if(coli_vk_mem_budget3(&u,&b) && (b-u) < G.reserve_gb){
            fprintf(stderr,"[qtier-vk] preload dev3: stopping on VRAM budget "
                           "(%.1f of %.1f GB used, %.1f reserve) after %llu experts\n",
                    u,b,G.reserve_gb,(unsigned long long)G.uploads);
            G.budget_stop=1;
            pthread_mutex_unlock(&G.up_mx); free(w); free(sc); return;
        }
    }
    ColiVkTensor *tg=NULL,*tu=NULL,*td=NULL;
    int ok = coli_vk_tensor_ensure3(&tg, w,        sc,             4, G.D,  G.Ih, G.egs)
          && coli_vk_tensor_ensure3(&tu, w+mb,     sc+G.sc_gu,     4, G.D,  G.Ih, G.egs)
          && coli_vk_tensor_ensure3(&td, w+2*mb,   sc+2*G.sc_gu,   4, G.Ih, G.D,  G.egs);
    if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
    else {
        if(tg) coli_vk_tensor_free(tg);
        if(tu) coli_vk_tensor_free(tu);
        if(td) coli_vk_tensor_free(td);
        if(!G.upload_fail)
            fprintf(stderr,"[qtier-vk] upload refused at expert %d/%d -- the rest stay on the CPU\n",
                    layer, eid);
        G.upload_fail++;
        G.budget_stop=1;                     /* the card is full: stop trying */
    }
    s->planned=0;
    pthread_mutex_unlock(&G.up_mx);
    free(w); free(sc);
}

void qt_fill_wait(void){ /* uploads are synchronous: nothing in flight */ }

uint32_t qt_issue(int layer,const int *eids,int K,const float *x){
    if(!G.on || K>32 || K<1) return 0;
    ColiVkTensor *tg[32],*tu[32],*td[32];
    static int rows[32];
    if(!rows[0]) for(int i=0;i<32;i++) rows[i]=1;
    uint32_t mask=0;
    G.is_cnt=0;
    for(int k=0;k<K;k++){
        int e=eids[k];
        if(e<0||e>=G.ne){ G.miss++; continue; }
        QSlot *s=qs(layer,e);
        if(!s->resident){ G.miss++; continue; }
        int c=G.is_cnt;
        tg[c]=s->tg; tu[c]=s->tu; td[c]=s->td;
        G.is_k[c]=k; G.is_cnt=c+1;
        mask |= 1u<<k; G.hits++;
    }
    if(!G.is_cnt) return 0;
    for(int j=0;j<G.is_cnt;j++)
        memcpy(G.is_x + (size_t)j*G.D, x, (size_t)G.D*sizeof(float));
    if(!coli_vk_expert_group_issue3(tg,tu,td,rows,G.is_cnt,G.is_x)){
        /* hand these k back to the CPU for this token */
        G.hits -= (uint64_t)G.is_cnt; G.miss += (uint64_t)G.is_cnt;
        G.is_cnt=0;
        return 0;
    }
    return mask;
}

void qt_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    if(!G.on || !mask || !G.is_cnt) return;
    int c=G.is_cnt; G.is_cnt=0;
    if(!coli_vk_expert_group_take3(G.is_y)){
        /* qt_issue already told the engine these k were handled, so the caller
         * did NOT compute them on the CPU: their contribution is now missing
         * from this token. Say so once and loudly rather than returning a
         * quietly wrong hidden state -- the fence only fails when the device
         * is lost, and a silent drop would survive every text oracle that
         * happens to sample the same argmax. */
        static int said=0;
        if(!said){ said=1;
            fprintf(stderr,"[qtier-vk] expert-group take FAILED -- %d expert(s) dropped from "
                           "this token; output from here on is NOT trustworthy\n", c); }
        G.hits -= (uint64_t)c; G.miss += (uint64_t)c;
        return;
    }
    for(int j=0;j<c;j++){
        float w=val[G.is_k[j]];
        const float *row=G.is_y + (size_t)j*G.D;
        for(int d=0;d<G.D;d++) out[d]+=w*row[d];
    }
}

void qt_stats(void){
    if(!G.on) return;
    size_t res=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++) res += G.slot[i].resident;
    double u=0,b=0; int have=coli_vk_mem_budget3(&u,&b);
    double tot=(double)(G.hits+G.miss);
    fprintf(stderr,"[qtier-vk] resident %zu/%d experts | uploads %llu | refused %llu | "
                   "budget-stop %d\n",
            res, G.nl*G.ne, (unsigned long long)G.uploads,
            (unsigned long long)G.upload_fail, G.budget_stop);
    fprintf(stderr,"[qtier-vk] dev3 planned %.2f GB, driver reports %.2f of %.2f GB used%s\n",
            G.used/1e9, u, b, have ? "" : " (VK_EXT_memory_budget absent: figures unavailable)");
    fprintf(stderr,"[qtier-vk] VRAM hit rate: %.1f %% (hits %llu, CPU misses %llu)\n",
            tot>0 ? 100.0*(double)G.hits/tot : 0.0,
            (unsigned long long)G.hits, (unsigned long long)G.miss);
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
