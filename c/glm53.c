/* GLM-5.3-Flash inference engine in pure C — sibling of kimi_k3.c / colibri.c /
 * deepseek_v4.c / inkling.c / qwen36.c / olmoe.c, sharing st.h / json.h / tok.h /
 * quant.h / compat.h.
 *
 * The name says GLM, but the skeleton is Kimi K3's, not GLM-5.2's: a hybrid
 * stack of linear-attention layers punctuated by full attention, over a
 * streamed MoE. That is why this file starts from kimi_k3.c and borrows the
 * indexer from colibri.c and the hyper-connections from deepseek_v4.c, instead
 * of extending the GLM engine.
 *
 * Architecture, read off the released checkpoint (321.34 B parameters measured
 * from the shard headers; the card says 320 B total / 18 B active):
 *
 *   - 45 text layers plus one MTP layer (index 45), hidden 4096, vocab 154880.
 *   - Hybrid attention, pattern (KDA KDA KDA FULL) repeating: 34 KDA linear
 *     layers and 11 DeepSeek-sparse-attention layers at 3, 7, ... 43, plus the
 *     MTP layer's own full-attention block. config.layer_types is explicit and
 *     is what this engine follows; linear_attn_config.kda_layers agrees.
 *   - MLA with kv_lora 512, q_lora 1536, qk_nope 256 and **qk_rope 0**: the
 *     full-attention layers are NoPE, exactly like K3. Position lives in the
 *     KDA decay and short convolutions, not in a rotation.
 *   - DSA lightning indexer on every full-attention layer, with k-pooling:
 *     keys are grouped into pools of index_kpool=4, the pools are scored
 *     instead of the tokens, the top index_topk/kpool pools are expanded back
 *     into token indices and the incomplete tail pool is always appended
 *     (index_kpool_always_select_tail). Two tensors carry it that GLM-5.2's
 *     indexer does not have: index_kpool_compress_ape and _compress_gate.
 *   - Manifold-Constrained Hyper-Connections (mHC, hc_mult 4, 20 Sinkhorn
 *     iterations) replace the plain residual at both sites of every layer —
 *     the same mHC DeepSeek V4 uses, down to the config keys, so the split
 *     into pre/post/comb and the Sinkhorn projection are shared code.
 *   - MoE: 288 routed experts (top-8) plus 1 shared expert per layer from
 *     layer 3 on, moe_intermediate 2048, sigmoid scoring with noaux_tc and
 *     e_score_correction_bias, routed_scaling_factor 2.5. The first three
 *     layers are dense (intermediate 12288).
 *   - Natively multimodal: a 24-block ViT (hidden 1024, patch 14, 448 px,
 *     spatial merge 2) whose patches are projected to 4096 and substituted at
 *     the image-token positions of the text stream. Text-only prompts never
 *     touch it.
 *
 * KDA recurrence — the same Kimi Delta Attention kimi_k3.c already reproduces
 * token-exact against the vendor, verified line by line against
 * transformers' recurrent_kimi_delta_attention:
 *     q,k,v = SiLU(ShortConv4(W{q,k,v} x));  q,k L2-normalized, q *= d^-0.5
 *     z  = W_fb(W_fa x) + dt_bias
 *     gk = gmin * sigmoid(exp(A_log[h]) * z),  gmin = gate_lower_bound = -5
 *     S  = (I - beta k k^T) Diag(exp(gk)) S + beta k v^T,  beta = sigmoid(W_b x)
 *     o  = S^T q;  out = W_o [ sigmoid(W_gb(W_ga x)) * RMSNorm_head(o) ]
 * One difference from K3: the output gate is LOW-RANK here (g_a_proj into
 * head_dim, then g_b_proj back out), where K3 has a single full g_proj.
 *
 * Container: tools/convert_glm53.py writes routed experts as int4 group-scaled
 * gs64 (`name` U8 + `name.qs` F32, fmt=4 — the same container GLM-5.2 uses,
 * measured cosine 0.994 against the fp8 source on real weights) and everything
 * else as BF16, quantized at LOAD TIME here. That split is deliberate: the
 * non-expert weights are 3% of the bytes, so keeping them exact on disk costs
 * ~14 GB and means retuning dense precision never requires re-downloading the
 * checkpoint.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>

#include "cli_args.h"
#include "json.h"
/* Expert weights are served from the shard mapping by default in this engine
 * (see expert_map_init); GLM53_NO_MMAP / COLI_MAP_EXPERTS=0 turn it off. */
#define COLI_MAP_EXPERTS_DEFAULT 1
#include "st.h"
#include "quant.h"
#include "tok.h"
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
static int g_vk_ready = 0;
#endif
#include "compat.h"
#include "serve_poll.h"          /* CANCEL a meta' turno (#1332) */
#include <time.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include "hyper_connections.h"   /* mHC, condiviso con deepseek_v4.c */

/* ---------- config ----------
 * Nested like Kimi K3's: the root carries the vision wrapper and `text_config`
 * carries the language model. A text-only export therefore has the text keys at
 * the root, and both shapes are accepted. */
typedef struct {
    /* text */
    int hidden, n_layers, vocab, first_dense, dense_inter;
    int n_heads, q_lora, kv_lora, qk_nope, qk_rope, v_head, qk_head;
    int n_experts, topk, moe_inter, n_shared;
    float routed_scale, eps, swiglu_limit;
    int n_mtp;                       /* num_nextn_predict_layers (1) */
    /* KDA */
    int kda_heads, kda_hd, kda_proj, conv_k;
    float gate_lb;
    /* DSA indexer with k-pooling */
    int index_topk, index_nh, index_hd, index_kpool, index_kpool_tail;
    /* mHC */
    int hc_mult, hc_iters;
    float hc_eps;
    /* per-layer kind: 1 = full attention (MLA + indexer), 0 = KDA */
    unsigned char is_full[128];
    /* vision (0 = text-only checkpoint) */
    int vis_layers, vis_hidden, vis_heads, vis_inter, vis_patch, vis_temporal;
    int vis_merge, vis_out_hidden, vis_proj_inter, vis_image_size, vis_in_ch;
    float vis_swiglu_limit, vis_eps;
    int image_token, image_start_token, image_end_token;
    int video_token, video_start_token, video_end_token;
} Cfg;

static double req_num(jval *object, const char *key) {
    jval *value = json_get(object, key);
    if (!value || value->t != J_NUM) {
        fprintf(stderr, "config.json: missing or non-numeric \"%s\"\n", key);
        exit(1);
    }
    return value->num;
}

static double opt_num(jval *object, const char *key, double fallback) {
    jval *value = object ? json_get(object, key) : NULL;
    return (value && value->t == J_NUM) ? value->num : fallback;
}

static int opt_bool(jval *object, const char *key, int fallback) {
    jval *value = object ? json_get(object, key) : NULL;
    if (!value) return fallback;
    if (value->t == J_BOOL) return value->boolean;
    if (value->t == J_NUM) return value->num != 0.0;
    return fallback;
}

/* layer_types is the authority on which layers are full attention. The
 * linear_attn_config.{kda_layers,full_attn_layers} lists say the same thing;
 * disagreeing checkpoints are refused rather than guessed at, because picking
 * the wrong kind for one layer produces plausible-looking garbage. */
static void load_layer_kinds(Cfg *c, jval *text) {
    memset(c->is_full, 0, sizeof(c->is_full));
    jval *types = json_get(text, "layer_types");
    if (!types || types->t != J_ARR || types->len != c->n_layers) {
        fprintf(stderr, "config.json: layer_types must list %d entries\n", c->n_layers);
        exit(1);
    }
    int full = 0;
    for (int i = 0; i < types->len; i++) {
        jval *entry = types->kids[i];
        if (!entry || entry->t != J_STR) {
            fprintf(stderr, "config.json: layer_types[%d] is not a string\n", i);
            exit(1);
        }
        if (strstr(entry->str, "linear")) {
            c->is_full[i] = 0;
        } else if (strstr(entry->str, "attention") || strstr(entry->str, "full")) {
            c->is_full[i] = 1;
            full++;
        } else {
            fprintf(stderr, "config.json: unknown layer type \"%s\" at %d\n", entry->str, i);
            exit(1);
        }
    }
    jval *linear = json_get(text, "linear_attn_config");
    jval *full_list = linear ? json_get(linear, "full_attn_layers") : NULL;
    if (full_list && full_list->t == J_ARR) {
        if (full_list->len != full) {
            fprintf(stderr, "config.json: full_attn_layers lists %d layers, "
                            "layer_types marks %d\n", full_list->len, full);
            exit(1);
        }
        for (int i = 0; i < full_list->len; i++) {
            int index = (int)full_list->kids[i]->num;
            if (index < 0 || index >= c->n_layers || !c->is_full[index]) {
                fprintf(stderr, "config.json: full_attn_layers disagrees with "
                                "layer_types at %d\n", index);
                exit(1);
            }
        }
    }
    /* The MTP block is a full-attention layer that lives past num_hidden_layers
     * and is not described by layer_types. */
    if (c->n_layers < (int)sizeof(c->is_full)) c->is_full[c->n_layers] = 1;
}

static void load_vision(Cfg *c, jval *root) {
    jval *vision = json_get(root, "vision_config");
    if (!vision || vision->t != J_OBJ) { c->vis_layers = 0; return; }
    c->vis_layers      = (int)req_num(vision, "depth");
    c->vis_hidden      = (int)req_num(vision, "hidden_size");
    c->vis_heads       = (int)req_num(vision, "num_heads");
    c->vis_inter       = (int)req_num(vision, "intermediate_size");
    c->vis_patch       = (int)req_num(vision, "patch_size");
    c->vis_temporal    = (int)opt_num(vision, "temporal_patch_size", 2);
    c->vis_merge       = (int)opt_num(vision, "spatial_merge_size", 2);
    c->vis_out_hidden  = (int)opt_num(vision, "out_hidden_size", c->hidden);
    c->vis_proj_inter  = (int)opt_num(vision, "projection_intermediate_size", 0);
    c->vis_image_size  = (int)opt_num(vision, "image_size", 448);
    c->vis_in_ch       = (int)opt_num(vision, "in_channels", 3);
    c->vis_swiglu_limit= (float)opt_num(vision, "swiglu_limit", 10.0);
    c->vis_eps         = (float)opt_num(vision, "rms_norm_eps", 1e-5);
    c->image_token       = (int)opt_num(root, "image_token_id", -1);
    c->image_start_token = (int)opt_num(root, "image_start_token_id", -1);
    c->image_end_token   = (int)opt_num(root, "image_end_token_id", -1);
    c->video_token       = (int)opt_num(root, "video_token_id", -1);
    c->video_start_token = (int)opt_num(root, "video_start_token_id", -1);
    c->video_end_token   = (int)opt_num(root, "video_end_token_id", -1);
    if (c->vis_layers < 1 || c->vis_layers > 128 || c->vis_hidden < 1 ||
        c->vis_heads < 1 || c->vis_hidden % c->vis_heads ||
        c->vis_patch < 1 || c->vis_merge < 1 || c->vis_out_hidden != c->hidden) {
        fprintf(stderr, "config.json: vision_config out of range "
                        "(out_hidden_size must equal the text hidden size)\n");
        exit(1);
    }
}

static void load_cfg(Cfg *c, const char *snap) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    char *buffer = NULL;
    {
        FILE *stream = fopen(path, "rb");
        if (!stream) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
        if (fseek(stream, 0, SEEK_END)) { fprintf(stderr, "%s: not seekable\n", path); exit(1); }
        long length = ftell(stream);
        if (length < 2 || fseek(stream, 0, SEEK_SET)) {
            fprintf(stderr, "%s: unusable\n", path); exit(1); }
        buffer = malloc((size_t)length + 1);
        if (!buffer) { fprintf(stderr, "OOM reading config\n"); exit(1); }
        if (fread(buffer, 1, (size_t)length, stream) != (size_t)length) {
            fprintf(stderr, "%s: short read\n", path); exit(1); }
        buffer[length] = 0;
        fclose(stream);
    }
    char *arena = NULL;
    jval *root = json_parse(buffer, &arena);
    if (!root || root->t != J_OBJ) { fprintf(stderr, "%s: not a JSON object\n", path); exit(1); }
    memset(c, 0, sizeof(*c));

    jval *tc = json_get(root, "text_config");
    if (!tc || tc->t != J_OBJ) tc = root;      /* text-only export */

    c->hidden      = (int)req_num(tc, "hidden_size");
    c->n_layers    = (int)req_num(tc, "num_hidden_layers");
    c->vocab       = (int)req_num(tc, "vocab_size");
    /* Il checkpoint reale dichiara first_k_dense_replace; la fixture tiny usa
     * mlp_layer_types. Accettiamo entrambi invece di imporne uno: sono la
     * stessa informazione detta in due modi, e rifiutare la fixture
     * significherebbe non poter provare il motore senza 194 GB. */
    {
        jval *fkd = json_get(tc, "first_k_dense_replace");
        if (fkd && fkd->t == J_NUM) {
            c->first_dense = (int)fkd->num;
        } else {
            jval *kinds = json_get(tc, "mlp_layer_types");
            if (!kinds || kinds->t != J_ARR) {
                fprintf(stderr, "config.json: serve first_k_dense_replace "
                                "oppure mlp_layer_types\n");
                exit(1);
            }
            c->first_dense = kinds->len;
            for (int i = 0; i < kinds->len; i++)
                if (kinds->kids[i]->t == J_STR && strstr(kinds->kids[i]->str, "sparse")) {
                    c->first_dense = i;
                    break;
                }
        }
    }
    c->dense_inter = (int)req_num(tc, "intermediate_size");
    c->n_heads     = (int)req_num(tc, "num_attention_heads");
    c->q_lora      = (int)req_num(tc, "q_lora_rank");
    c->kv_lora     = (int)req_num(tc, "kv_lora_rank");
    c->qk_nope     = (int)req_num(tc, "qk_nope_head_dim");
    c->qk_rope     = (int)opt_num(tc, "qk_rope_head_dim", 0);
    c->v_head      = (int)req_num(tc, "v_head_dim");
    c->n_experts   = (int)req_num(tc, "n_routed_experts");
    c->topk        = (int)req_num(tc, "num_experts_per_tok");
    c->moe_inter   = (int)req_num(tc, "moe_intermediate_size");
    c->n_shared    = (int)opt_num(tc, "n_shared_experts", 1);
    c->routed_scale= (float)opt_num(tc, "routed_scaling_factor", 1.0);
    c->eps         = (float)opt_num(tc, "rms_norm_eps", 1e-6);
    c->swiglu_limit= (float)opt_num(tc, "swiglu_limit", 0.0);
    c->n_mtp       = (int)opt_num(tc, "num_nextn_predict_layers", 0);
    c->qk_head     = c->qk_nope + c->qk_rope;

    jval *linear = json_get(tc, "linear_attn_config");
    if (!linear || linear->t != J_OBJ) {
        fprintf(stderr, "config.json: missing linear_attn_config\n"); exit(1);
    }
    c->kda_heads = (int)req_num(linear, "num_heads");
    c->kda_hd    = (int)req_num(linear, "head_dim");
    c->conv_k    = (int)req_num(linear, "short_conv_kernel_size");
    c->gate_lb   = (float)opt_num(linear, "gate_lower_bound", -5.0);
    c->kda_proj  = c->kda_heads * c->kda_hd;

    c->index_topk       = (int)opt_num(tc, "index_topk", 0);
    c->index_nh         = (int)opt_num(tc, "index_n_heads", 0);
    c->index_hd         = (int)opt_num(tc, "index_head_dim", 0);
    c->index_kpool      = (int)opt_num(tc, "index_kpool", 1);
    c->index_kpool_tail = opt_bool(tc, "index_kpool_always_select_tail", 0);

    c->hc_mult  = (int)opt_num(tc, "hc_mult", 1);
    c->hc_iters = (int)opt_num(tc, "hc_sinkhorn_iters", 0);
    c->hc_eps   = (float)opt_num(tc, "hc_eps", 1e-6);

    load_layer_kinds(c, tc);
    load_vision(c, root);

    if (c->hidden < 1 || c->hidden > 65536 ||
        c->n_layers < 1 || c->n_layers > 120 ||
        c->vocab < 1 || c->vocab > (1 << 22) ||
        c->n_experts < 1 || c->n_experts > 4096 ||
        c->topk < 1 || c->topk > 64 || c->topk > c->n_experts ||
        c->kda_proj < 1 || c->kda_proj > (1 << 20) ||
        c->conv_k < 1 || c->conv_k > 8 ||
        c->moe_inter % 32 || c->kda_hd > 512 || c->kv_lora > 4096 ||
        c->first_dense < 0 || c->first_dense > c->n_layers ||
        c->index_kpool < 1 || c->index_kpool > 64 ||
        c->hc_mult < 1 || c->hc_mult > 8) {
        fprintf(stderr, "config.json: dimension out of range\n"); exit(1);
    }
    /* qk_rope must be zero: a rotary GLM-5.3 would need position handling this
     * engine deliberately does not have, and silently ignoring the rotation
     * would produce a model that answers fluently and wrongly. */
    if (c->qk_rope != 0) {
        fprintf(stderr, "config.json: qk_rope_head_dim=%d, but this engine "
                        "implements the NoPE full-attention of GLM-5.3\n", c->qk_rope);
        exit(1);
    }
    free(arena);
    free(buffer);
}

static void cfg_report(const Cfg *c) {
    int full = 0, kda = 0;
    for (int i = 0; i < c->n_layers; i++) { if (c->is_full[i]) full++; else kda++; }
    double expert_params = (double)c->n_experts * 3.0 * c->hidden * c->moe_inter *
                           (c->n_layers - c->first_dense + (c->n_mtp ? 1 : 0));
    fprintf(stderr,
        "GLM-5.3-Flash: %d layers (%d KDA + %d full) + %d MTP, hidden %d, vocab %d\n"
        "  MoE      : %d routed (top-%d) + %d shared, inter %d, from layer %d, scale %.2f\n"
        "  KDA      : %d heads x %d, conv %d, gate floor %.1f\n"
        "  MLA      : q_lora %d, kv_lora %d, qk %d (nope, NoPE), v %d, %d heads\n"
        "  indexer  : top-%d, %d heads x %d, kpool %d%s\n"
        "  mHC      : mult %d, %d Sinkhorn iterations\n"
        "  vision   : %s\n"
        "  routed experts: %.1f B parameters (%.0f GB at int4-g64)\n",
        c->n_layers, kda, full, c->n_mtp, c->hidden, c->vocab,
        c->n_experts, c->topk, c->n_shared, c->moe_inter, c->first_dense, c->routed_scale,
        c->kda_heads, c->kda_hd, c->conv_k, (double)c->gate_lb,
        c->q_lora, c->kv_lora, c->qk_nope, c->v_head, c->n_heads,
        c->index_topk, c->index_nh, c->index_hd, c->index_kpool,
        c->index_kpool_tail ? " (+tail)" : "",
        c->hc_mult, c->hc_iters,
        c->vis_layers ? "yes" : "text-only checkpoint",
        expert_params / 1e9, expert_params * 0.5625 / 1e9);
    if (c->vis_layers)
        fprintf(stderr,
        "             %d blocks x %d, %d heads, patch %d, %dpx, merge %d -> %d\n",
        c->vis_layers, c->vis_hidden, c->vis_heads, c->vis_patch,
        c->vis_image_size, c->vis_merge, c->vis_out_hidden);
}

#ifdef GLM53_CFG_MAIN_UNUSED
/* Config-only entry point: `glm53_cfg <model_dir>` parses and reports, so the
 * parser can be checked against a real checkpoint before any weight exists. */
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    Cfg c;
    load_cfg(&c, argv[1]);
    cfg_report(&c);
    return 0;
}
#endif

/* ---------- RAM-resident weight ----------
 * Same shape as kimi_k3.c's: either f32, int8 per-row, or int4 group-scaled.
 * The container written by tools/convert_glm53.py is U8 + `.qs` F32, which is
 * exactly what K3's loader already reads, so the two engines share a format
 * rather than each inventing one. */
typedef struct {
    int fmt;                              /* 0 = f32, 8 = int8 per-row, 4 = int4-g64 */
    float *f;
    int8_t *q8;
    uint8_t *q4;
    float *s;                             /* scales: [O] per-row, or [O*ngroups] */
    int O, I, gs;
} W;

/* ---------- layer structures ----------
 * Mirrors kimi_k3.c's shapes, with the GLM-5.3 differences called out where
 * they bite: the KDA output gate is low-rank here, the full-attention layers
 * are NOT gated (K3's are), and every layer carries two mHC sites instead of a
 * plain residual. */
typedef struct {                          /* KDA (linear attention) layer */
    W q, k, v, o;                         /* [proj x hidden] x3, [hidden x proj] */
    W ga, gb;                             /* low-rank output gate: hidden->hd->proj */
    float *conv_q, *conv_k, *conv_v;      /* [proj*conv_k] depthwise taps */
    float *fa, *fb;                       /* decay low-rank: [hd,hidden], [proj,hd] */
    float *bp;                            /* beta projection [heads,hidden] */
    float *dt, *A, *onw;                  /* dt_bias[proj], exp(A_log)[heads], o_norm[hd] */
} Kda;

typedef struct {                          /* MLA + DSA indexer (full attention) */
    W qa, qb, kva, kvb, o;
    float *qa_ln, *kva_ln;
    W wq, wk, wp;                         /* indexer: wq_b, wk, weights_proj */
    float *knw, *knb;                     /* indexer key LayerNorm (weight + bias) */
    float *kpool_ape;                     /* [kpool, index_hd] pool position bias */
    W kpool_gate;                         /* [index_hd, hidden] compression gate */
} Mla;

typedef struct {                          /* MoE (routed streamed + shared resident) */
    float *router, *rbias;                /* [E,hidden] f32, [E] correction bias */
    W sh_gate, sh_up, sh_down;
} Moe;

typedef struct {
    int full;                             /* 1 = MLA + indexer, 0 = KDA */
    int dense;                            /* 1 = plain MLP (layers < first_dense) */
    Kda a;
    Mla m;
    Moe moe;
    W d_gate, d_up, d_down;               /* dense layers only */
    float *in_ln, *post_ln;
    /* mHC, two sites per layer. fn is [(2+H)*H, H*hidden], base [(2+H)*H],
     * scale [3]; the split into pre/post/comb and the Sinkhorn projection are
     * the same as DeepSeek V4's. Absent on the MTP layer. */
    float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    float *hc_ffn_fn,  *hc_ffn_base,  *hc_ffn_scale;
    /* MTP layer only */
    W mtp_eh;
    float *mtp_enorm, *mtp_hnorm, *mtp_head_norm;
} Layer;

/* ---------- tensor names ----------
 * One place builds every name the engine asks for. The checkpoint prefixes the
 * language model with `model.language_model.` because the root is the vision
 * wrapper; a text-only export drops it. Both are probed, and the choice is made
 * once from a tensor that must exist either way. */
typedef struct {
    const char *prefix;                   /* "model.language_model." or "model." */
    const char *visual;                   /* "model.visual." */
} Names;

#define GLM53_NAME(dst, fmt, ...) snprintf((dst), sizeof(dst), (fmt), __VA_ARGS__)

/* Emits, in load order, every tensor this engine will look for. `sink` is
 * called with (name, required); a NULL model just prints them, which is how
 * the mapping gets checked against a real checkpoint index before any weight
 * has been downloaded. */
static void glm53_walk_tensors(const Cfg *c, const Names *n,
                               void (*sink)(void *, const char *, int),
                               void *user) {
    char name[512];
#define EMIT(required, fmt, ...) do { \
        GLM53_NAME(name, fmt, __VA_ARGS__); sink(user, name, (required)); } while (0)
    EMIT(1, "%sembed_tokens.weight", n->prefix);
    EMIT(1, "%snorm.weight", n->prefix);
    sink(user, "lm_head.weight", 1);

    int last = c->n_layers + (c->n_mtp ? 1 : 0);
    for (int i = 0; i < last; i++) {
        int mtp = i >= c->n_layers;
        int full = c->is_full[i];
        EMIT(1, "%slayers.%d.input_layernorm.weight", n->prefix, i);
        EMIT(1, "%slayers.%d.post_attention_layernorm.weight", n->prefix, i);
        if (!mtp) {                        /* mHC lives on the 45 real layers */
            EMIT(1, "%slayers.%d.hc_attn_fn", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_attn_base", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_attn_scale", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_ffn_fn", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_ffn_base", n->prefix, i);
            EMIT(1, "%slayers.%d.hc_ffn_scale", n->prefix, i);
        } else {
            EMIT(1, "%slayers.%d.eh_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.enorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.hnorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.shared_head.norm.weight", n->prefix, i);
        }
        if (full) {
            EMIT(1, "%slayers.%d.self_attn.q_a_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.q_a_layernorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.q_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.kv_a_proj_with_mqa.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.kv_a_layernorm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.kv_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.o_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.wq_b.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.wk.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.weights_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.k_norm.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.indexer.k_norm.bias", n->prefix, i);
            if (c->index_kpool > 1) {
                EMIT(1, "%slayers.%d.self_attn.indexer.index_kpool_compress_ape", n->prefix, i);
                EMIT(1, "%slayers.%d.self_attn.indexer.index_kpool_compress_gate", n->prefix, i);
            }
        } else {
            EMIT(1, "%slayers.%d.self_attn.q_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.k_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.v_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.o_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.g_a_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.g_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.q_conv1d.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.k_conv1d.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.v_conv1d.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.f_a_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.f_b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.b_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.dt_bias", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.A_log", n->prefix, i);
            EMIT(1, "%slayers.%d.self_attn.o_norm.weight", n->prefix, i);
        }
        if (i < c->first_dense) {
            EMIT(1, "%slayers.%d.mlp.gate_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.up_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.down_proj.weight", n->prefix, i);
        } else {
            EMIT(1, "%slayers.%d.mlp.gate.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.gate.e_score_correction_bias", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.shared_experts.gate_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.shared_experts.up_proj.weight", n->prefix, i);
            EMIT(1, "%slayers.%d.mlp.shared_experts.down_proj.weight", n->prefix, i);
            /* Routed experts are streamed, not resident: named here so the
             * inventory can prove they are all present and correctly shaped. */
            for (int e = 0; e < c->n_experts; e++) {
                EMIT(1, "%slayers.%d.mlp.experts.%d.gate_proj.weight", n->prefix, i, e);
                EMIT(1, "%slayers.%d.mlp.experts.%d.up_proj.weight", n->prefix, i, e);
                EMIT(1, "%slayers.%d.mlp.experts.%d.down_proj.weight", n->prefix, i, e);
            }
        }
    }
    if (c->vis_layers) {
        EMIT(1, "%spatch_embed.proj.weight", n->visual);
        EMIT(1, "%spatch_embed.proj.bias", n->visual);
        EMIT(1, "%spost_layernorm.weight", n->visual);
        EMIT(1, "%sdownsample.weight", n->visual);
        EMIT(1, "%sdownsample.bias", n->visual);
        EMIT(1, "%smerger.proj.weight", n->visual);
        EMIT(1, "%smerger.post_projection_norm.weight", n->visual);
        EMIT(1, "%smerger.post_projection_norm.bias", n->visual);
        EMIT(1, "%smerger.gate_proj.weight", n->visual);
        EMIT(1, "%smerger.up_proj.weight", n->visual);
        EMIT(1, "%smerger.down_proj.weight", n->visual);
        for (int b = 0; b < c->vis_layers; b++) {
            EMIT(1, "%sblocks.%d.norm1.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.norm2.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.qkv.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.qkv.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.proj.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.q_norm.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.attn.k_norm.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.gate_proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.gate_proj.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.up_proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.up_proj.bias", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.down_proj.weight", n->visual, b);
            EMIT(1, "%sblocks.%d.mlp.down_proj.bias", n->visual, b);
        }
    }
#undef EMIT
}

#ifdef GLM53_INVENTORY_MAIN_UNUSED
/* Inventory entry point: prints every tensor the engine would load, in load
 * order. Checked against a real checkpoint's index offline, so a naming
 * mistake surfaces before 180 GB have been converted rather than after. */
static void print_name(void *user, const char *name, int required) {
    (void)user; printf("%s\t%d\n", name, required);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    Cfg c;
    load_cfg(&c, argv[1]);
    Names n = { "model.language_model.", "model.visual." };
    glm53_walk_tensors(&c, &n, print_name, NULL);
    return 0;
}
#endif

/* ================= forward =================
 * Il modello caricato in RAM come f32. Lo streaming degli esperti e la
 * quantizzazione a load time arrivano col modello vero; qui la priorita' e'
 * che i token siano quelli giusti, provati contro l'oracolo. */
#include "delta_attention.h"
#include "sparse_index.h"
#include "vision_tower.h"

/* Una matrice residente, nel formato in cui conviene tenerla.
 *
 * Il checkpoint porta i densi in BF16 e gli esperti gia' in int4 gs64. Tenere
 * i densi in f32 vuol dire 39 GB di RAM per il 3% dei parametri, quindi qui si
 * quantizzano al volo secondo GLM53_BITS. Il formato 4 e' lo stesso
 * contenitore che scrive tools/convert_glm53.py e che GLM-5.2 usa gia':
 * `nome` U8 con due valori per byte, `nome.qs` F32 con una scala ogni 64
 * colonne. */
typedef struct {
    int fmt;                              /* 0 = f32, 1 = int8 per riga, 4 = int4 gs64 */
    const float *f;
    const int8_t *q8;
    const uint8_t *q4;
    const float *s;
    int rows, columns, gs;
    void *vk;                             /* ColiVkTensor*, caricata alla prima uso */
} Mat;

typedef struct {
    /* comune */
    const float *in_ln, *post_ln;
    const float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    const float *hc_ffn_fn,  *hc_ffn_base,  *hc_ffn_scale;
    /* KDA */
    Mat kq, kk, kv, ko, kga, kgb, kfa, kfb, kb;
    const float *conv, *dt, *alog, *onorm;
    /* MLA + indexer */
    Mat qa, qb, kva, kvb_kt, kvb_v, o, iwq, iwk, iwp, ikpg;
    const float *qa_ln, *kva_ln, *ik_nw, *ik_nb, *ikpa;
    /* FFN */
    Mat dg, du, dd;                       /* denso */
    Mat rg, ru, rd;                       /* router / shared: gate,up,down */
    const float *router, *rbias;
    Mat *eg, *eu, *ed;                    /* esperti routed */
} GLayer;

/* Quello che una conversazione si porta dietro fra un token e il successivo.
 *
 * Senza, generare il token n costa un passaggio su tutti gli n precedenti, e
 * siccome ogni token attraversa 8 esperti per layer, il conto e' n volte il
 * lavoro che serve. Con la cache il passo costa un token.
 *
 * I layer KDA hanno una ricorrenza: lo stato e la finestra della convoluzione
 * sono gia' tutto quello che serve, e non crescono col contesto. I layer DSA
 * invece devono ricordare chiavi e valori di ogni posizione, piu' le chiavi e i
 * gate dell'indexer, perche' una query nuova puo' guardare ovunque dietro di
 * se'. */
typedef struct {
    float *kda_state;                     /* [teste * k * v] */
    float *kda_window;                    /* [3 * proiezione * kernel] */
    int kda_gpu;                          /* G12: recurrence resident on dev0; the
                                           * two host buffers above go STALE and are
                                           * refreshed only by the migration sync */
    float *latent;                        /* [cap][kv_lora]: MLA assorbita */
    float *ikeys, *igates;                /* [cap][dim indexer] */
} GLayerState;

typedef struct {
    GLayerState *layer;
    float *kda_scratch;
    int filled;                           /* posizioni gia' in cache */
    int cap;
    /* P6b: quale insieme di stato sul dispositivo appartiene a questa sessione.
     * Lo stato ricorrente e' della CONVERSAZIONE, quindi dello slot: con piu'
     * slot vivi contemporaneamente due sessioni non possono condividerlo. Il
     * percorso CLI e il segment adapter tengono una sessione sola: slot 0. */
    int slot;
} GSession;

typedef struct {
    Cfg c;
    shards S;
    const float *embed, *final_norm;
    Mat head;
    GLayer *layer;
    char prefix[64];
    /* Quali layer questo motore possiede davvero. Un segment ne carica un
     * pezzo, e caricare il resto vorrebbe dire tenere in RAM i pesi che sta
     * macinando un'altra macchina. */
    int layer_begin, layer_end;
    int has_io;                           /* embedding e testa: solo agli estremi */
    /* esperti: o residenti (checkpoint f32) o in streaming (container int4) */
    int streaming;
    struct ERef *eref;
    struct LCache *ecache;
    int64_t e_len[6], e_at[6], e_slot;
    uint64_t clock, ebytes;
    long hits, miss;
    /* Telemetria per la dashboard (#1376 follow-up: Brain e Profile erano
     * vuoti su Flash perche' il motore non emetteva nulla). Tempi di fase
     * cumulativi dall'avvio; il turno ne prende la differenza. */
    double t_attn, t_ffn, t_disk, t_head;
    uint64_t forwards;
    uint8_t **ehit;                       /* [layer][expert] toccato in questo turno */
    /* torre vision: presente solo se il checkpoint la porta */
    int has_vision;
    ColiVisionTower vision;
    ColiVisionBlock *vblocks;
} GModel;

static const float *load_f32(GModel *m, const char *fmt, ...) {
    char name[512];
    va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "manca il tensore %s\n", name); exit(1); }
    float *buffer = malloc((size_t)t->numel * sizeof(float));
    if (!buffer) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
    st_read_f32_cap(&m->S, name, buffer, t->numel, 0);
    return buffer;
}

/* Bit dei pesi densi residenti: 4, 8 o 32. Il default e' 4 perche' e' quello
 * che fa stare il modello su una macchina normale; chi vuole la precisione
 * piena la chiede. Gli esperti non passano di qui: arrivano dal disco gia'
 * quantizzati e restano com'erano. */
static int glm53_dense_bits(void) {
    static int cached = 0;
    if (cached) return cached;
    const char *setting = getenv("GLM53_BITS");
    cached = setting ? atoi(setting) : 4;
    if (cached != 4 && cached != 8 && cached != 32) {
        fprintf(stderr, "GLM53_BITS=%s: valori ammessi 4, 8, 32\n", setting);
        exit(1);
    }
    return cached;
}

/* Quantizza [rows, columns] f32 in int4 con una scala ogni `gs` colonne.
 * Stessa aritmetica di quant_int4_grouped() in tools/convert_glm53.py, cosi'
 * un peso quantizzato qui e uno quantizzato dal converter sono lo stesso peso. */
static void quantize_i4_grouped(const float *w, uint8_t *q4, float *scale,
                                int rows, int columns, int gs) {
    const int groups = (columns + gs - 1) / gs;
    const int packed = (columns + 1) / 2;
    for (int r = 0; r < rows; r++) {
        const float *row = w + (size_t)r * columns;
        uint8_t *dst = q4 + (size_t)r * packed;
        memset(dst, 0, (size_t)packed);
        for (int g = 0; g < groups; g++) {
            const int start = g * gs;
            const int stop = start + gs < columns ? start + gs : columns;
            float amax = 0.0f;
            for (int c = start; c < stop; c++) {
                const float value = fabsf(row[c]);
                if (value > amax) amax = value;
            }
            float step = amax / 7.0f;
            if (step < 1e-8f) step = 1e-8f;
            scale[(size_t)r * groups + g] = step;
            for (int c = start; c < stop; c++) {
                int level = (int)lrintf(row[c] / step);
                if (level < -8) level = -8;
                if (level > 7) level = 7;
                const uint8_t nibble = (uint8_t)(level + 8);
                if (c & 1) dst[c >> 1] |= (uint8_t)(nibble << 4);
                else       dst[c >> 1] |= nibble;
            }
        }
    }
}

/* Un blocco f32 gia' in memoria, portato alla precisione chiesta. Possiede il
 * buffer: o lo tiene com'e' o lo libera dopo averlo quantizzato. */
static Mat quantize_loaded(float *buffer, int rows, int columns) {
    Mat mat; memset(&mat, 0, sizeof(mat));
    mat.rows = rows; mat.columns = columns;
    const int bits = glm53_dense_bits();
    if (bits == 32) { mat.fmt = 0; mat.f = buffer; return mat; }
    if (bits == 4 && columns % 64 == 0) {
        const int groups = columns / 64;
        uint8_t *packed = malloc((size_t)rows * ((columns + 1) / 2));
        float *step = malloc((size_t)rows * groups * sizeof(float));
        if (!packed || !step) { fprintf(stderr, "OOM quantizzando %dx%d\n", rows, columns); exit(1); }
        quantize_i4_grouped(buffer, packed, step, rows, columns, 64);
        free(buffer);
        mat.fmt = 4; mat.q4 = packed; mat.s = step; mat.gs = 64;
        return mat;
    }
    /* int8 per riga: e' anche il ripiego quando le colonne non sono multiple
     * di 64, che capita sulle proiezioni piccole dell'indexer. */
    int8_t *level = malloc((size_t)rows * columns);
    float *step = malloc((size_t)rows * sizeof(float));
    if (!level || !step) { fprintf(stderr, "OOM quantizzando %dx%d\n", rows, columns); exit(1); }
    quantize_rows(buffer, level, step, rows, columns, 8);
    free(buffer);
    mat.fmt = 1; mat.q8 = level; mat.s = step;
    return mat;
}

/* Le due forme assorbite di kv_b_proj.
 *
 * MLA comprime chiavi e valori in un latente da kv_lora (512 qui) e li
 * riespande con kv_b_proj. Tenere in cache le chiavi espanse costa
 * teste*qk_nope*2 float per posizione, cioe' 1,39 MB a token su questo
 * modello, undici GB a ottomila token: piu' della macchina.
 *
 * Il latente invece costa 512 float, quarantatre volte meno, e non serve
 * riespanderlo se si piegano le proiezioni nei due estremi. Per una testa h,
 * con W_k e W_v le due meta' del blocco di kv_b_proj:
 *
 *     punteggio_j = q . (W_k c_j) = (W_k^T q) . c_j
 *     uscita      = somma_j a_j (W_v c_j) = W_v (somma_j a_j c_j)
 *
 * A sinistra si moltiplica per ogni posizione in cache, a destra una volta per
 * query. E' un'uguaglianza, non un'approssimazione: cambia solo l'ordine dei
 * prodotti, e con esso quanta memoria serve.
 *
 * Le due matrici si costruiscono qui, da kv_b_proj in f32, e poi passano dallo
 * stesso quantizzatore di tutto il resto. */
static void absorb_kvb(GModel *m, GLayer *l, const char *name) {
    const Cfg *c = &m->c;
    const int H = c->n_heads, QK = c->qk_nope, V = c->v_head, L = c->kv_lora;
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "manca %s\n", name); exit(1); }
    if (t->numel != (int64_t)H * (QK + V) * L) {
        fprintf(stderr, "%s: %lld valori, attesi %lld per %d teste\n", name,
                (long long)t->numel, (long long)H * (QK + V) * L, H);
        exit(1);
    }
    float *whole = malloc((size_t)t->numel * sizeof(float));
    float *kt = malloc((size_t)H * L * QK * sizeof(float));
    float *vv = malloc((size_t)H * V * L * sizeof(float));
    if (!whole || !kt || !vv) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
    st_read_f32_cap(&m->S, name, whole, t->numel, 1);

    for (int h = 0; h < H; h++) {
        const float *block = whole + (size_t)h * (QK + V) * L;
        /* W_k^T: [L, QK], da W_k che e' [QK, L] */
        for (int d = 0; d < L; d++)
            for (int i = 0; i < QK; i++)
                kt[((size_t)h * L + d) * QK + i] = block[(size_t)i * L + d];
        /* W_v: [V, L], gia' nel verso giusto, e' una copia di righe */
        memcpy(vv + (size_t)h * V * L, block + (size_t)QK * L,
               (size_t)V * L * sizeof(float));
    }
    free(whole);
    l->kvb_kt = quantize_loaded(kt, H * L, QK);
    l->kvb_v = quantize_loaded(vv, H * V, L);
}

static Mat load_mat(GModel *m, const char *fmt, ...) {
    char name[512];
    va_list args; va_start(args, fmt); vsnprintf(name, sizeof(name), fmt, args); va_end(args);
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "manca la matrice %s\n", name); exit(1); }
    Mat mat; memset(&mat, 0, sizeof(mat));

    /* Gia' quantizzato nel checkpoint: si prende com'e', senza passare per
     * f32. Un giro in f32 costerebbe il picco di RAM che stiamo evitando, e
     * riquantizzare quello che e' gia' quantizzato perde bit per niente. */
    if (t->dtype == 3) {                  /* U8/I8 in st.h: il container int4 */
        char scales[544];
        snprintf(scales, sizeof(scales), "%s.qs", name);
        st_tensor *qs = st_find(&m->S, scales);
        if (!qs) {
            fprintf(stderr, "%s e' int4 ma manca %s\n", name, scales);
            exit(1);
        }
        /* Il contenitore e' piatto: 4.194.304 byte di nibble e 131.072 scale,
         * senza righe ne' colonne scritte da nessuna parte. Va benissimo per
         * gli esperti, che passano dallo streaming e prendono la forma dalla
         * config (moe_inter x hidden, e il down al contrario). Qui invece la
         * forma servirebbe e non c'e': tirarla a indovinare da un solo numero
         * vorrebbe dire calcolare su una matrice trasposta senza accorgersene.
         *
         * Con il converter di oggi questo caso non si presenta, perche' tutto
         * cio' che non e' esperto resta BF16 e la precisione la sceglie
         * GLM53_BITS a load time. Se un domani si quantizzasse anche il resto,
         * la strada e' quella di kimi_k3: la forma la passa il chiamante. */
        const int64_t values = qs->numel * 64;
        if (t->nbytes * 2 != values) {
            fprintf(stderr, "%s: %lld byte e %lld scale non sono un int4 gs64\n",
                    name, (long long)t->nbytes, (long long)qs->numel);
            exit(1);
        }
        if (t->rank != 2) {
            fprintf(stderr, "%s: contenitore int4 piatto fuori dagli esperti; "
                            "la forma non e' nel file e non si indovina\n", name);
            exit(1);
        }
        mat.rows = (int)t->shape[0];
        mat.columns = (int)(values / t->shape[0]);
        if (mat.columns % 64) {
            fprintf(stderr, "%s: %d colonne non sono multiple di 64\n", name, mat.columns);
            exit(1);
        }
        uint8_t *packed = malloc((size_t)t->nbytes);
        float *step = malloc((size_t)qs->numel * sizeof(float));
        if (!packed || !step) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
        st_read_raw(&m->S, name, packed, 1);
        st_read_f32_cap(&m->S, scales, step, qs->numel, 1);
        mat.fmt = 4; mat.q4 = packed; mat.s = step; mat.gs = 64;
        return mat;
    }

    if (t->rank != 2) { fprintf(stderr, "%s: rank %d, attesa 2\n", name, t->rank); exit(1); }
    float *buffer = malloc((size_t)t->numel * sizeof(float));
    if (!buffer) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
    st_read_f32_cap(&m->S, name, buffer, t->numel, 1);
    mat.rows = (int)t->shape[0];
    mat.columns = (int)t->shape[1];

    mat = quantize_loaded(buffer, mat.rows, mat.columns);
    return mat;
}

/* Come mv ma su un blocco di righe contigue: serve alle matrici che tengono
 * una testa dopo l'altra in un unico tensore. */
static void mv_rows(float *out, const Mat *w, const float *x, int row0, int rows) {
    switch (w->fmt) {
    case 4: {
        const int packed = (w->columns + 1) / 2, groups = w->columns / w->gs;
        matmul_i4_grouped(out, x, w->q4 + (size_t)row0 * packed,
                          w->s + (size_t)row0 * groups, 1, w->columns, rows, w->gs);
        break;
    }
    case 1:
        matmul_q(out, x, w->q8 + (size_t)row0 * w->columns, w->s + row0,
                 1, w->columns, rows);
        break;
    default:
        matmul(out, x, w->f + (size_t)row0 * w->columns, 1, w->columns, rows);
        break;
    }
}

/* out[rows] = W x, con W in layout [rows, columns] come transformers.
 *
 * Con COLI_VK=1 le matrici RESIDENTI passano dal backend Vulkan: si caricano
 * sul device alla prima chiamata e restano li'. Gli esperti no, e non e' una
 * dimenticanza: arrivano dal disco a ogni uso, quindi caricarli costerebbe
 * quanto leggerli e il device non ci guadagnerebbe niente. Quelli vogliono un
 * livello residente in VRAM, che e' un'altra cosa.
 *
 * Il campo `vk` e' una cache dentro a una matrice che il resto del codice
 * tratta come sola lettura: da qui il cast, che riguarda solo lui. */
static void mv(float *out, const Mat *w, const float *x) {
#ifdef COLI_VULKAN
    if (g_vk_ready && (w->fmt == 1 || w->fmt == 4)) {
        Mat *mutable_w = (Mat *)w;
        if (coli_vk_matmul((ColiVkTensor **)&mutable_w->vk, out, x,
                           w->fmt == 4 ? (const void *)w->q4 : (const void *)w->q8,
                           w->s, w->fmt, 1, w->columns, w->rows, w->gs))
            return;
    }
#endif
    switch (w->fmt) {
    case 4: matmul_i4_grouped(out, x, w->q4, w->s, 1, w->columns, w->rows, w->gs); break;
    case 1: matmul_q(out, x, w->q8, w->s, 1, w->columns, w->rows); break;
    default: matmul(out, x, w->f, 1, w->columns, w->rows); break;
    }
}

/* P2 (PREFILL-ROADMAP P2): mv() for S rows at once. Same kernels, same
 * per-row reduction order -- the GPU shader is per-row independent
 * (s = WorkGroupID.y) and the CPU kernels loop rows outside the dot -- so
 * every row is bit-identical to S separate mv() calls. The point is one submit
 * (or one OpenMP team) per matrix per chunk instead of one per token. */
static void mv_rows_s(float *out, const Mat *w, const float *x, int S) {
    if (S == 1) { mv(out, w, x); return; }
#ifdef COLI_VULKAN
    if (g_vk_ready && (w->fmt == 1 || w->fmt == 4)) {
        Mat *mutable_w = (Mat *)w;
        if (coli_vk_matmul((ColiVkTensor **)&mutable_w->vk, out, x,
                           w->fmt == 4 ? (const void *)w->q4 : (const void *)w->q8,
                           w->s, w->fmt, S, w->columns, w->rows, w->gs))
            return;
    }
#endif
    switch (w->fmt) {
    case 4: matmul_i4_grouped(out, x, w->q4, w->s, S, w->columns, w->rows, w->gs); break;
    case 1: matmul_q(out, x, w->q8, w->s, S, w->columns, w->rows); break;
    default: matmul(out, x, w->f, S, w->columns, w->rows); break;
    }
}

static void rms(float *out, const float *x, const float *w, int n, float eps) {
    float square = 0.0f;
    for (int i = 0; i < n; i++) square += x[i] * x[i];
    float inverse = 1.0f / sqrtf(square / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inverse * w[i];
}

static void layer_norm(float *out, const float *x, const float *w, const float *b,
                       int n, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float variance = 0.0f;
    for (int i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
    variance /= n;
    float inverse = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inverse * w[i] + (b ? b[i] : 0.0f);
}

static float sigmoidf_(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}
static float siluf_(float x) { return x / (1.0f + expf(-x)); }

/* G3/G4 per-op timer state and helpers (tools/hot-expert/ROADMAP-2026-09.md).
 * Declared here, above kda_layer, because every timed site from kda_layer
 * down to forward_span reads them; the report and the reset live next to
 * run_layers. Gated on COLI_TIMERS=1 (the name rome_bench.sh already
 * exports) so the default path pays nothing -- not even the clock reads. */
static int g_optime = -1;
static double g_ot_kda, g_ot_mla, g_ot_ffn_dense, g_ot_ffn_moe, g_ot_hc, g_ot_head, g_ot_layers;
static long   g_on_kda, g_on_mla, g_on_ffn_dense, g_on_ffn_moe, g_on_hc, g_on_head, g_on_layers;
static double g_ot_router, g_ot_shared; static long g_on_router;   /* MoE sub-split */
/* G4 diagnosis: kda_layer sub-split. Parallelising the 64 heads moved the
 * total 2%, so the 2.2 ms/call is somewhere other than the recurrence. */
static double g_kt_proj, g_kt_decay, g_kt_step, g_kt_norm, g_kt_ko;
static long g_kn_calls, g_kn_batched;
/* G5 evidence: mla_layer sub-split. The DSA indexer re-pools the whole prefix
 * on every call and mla_layer runs once per decode token, so `index` is the
 * O(context)-per-token term G5 would cache away. Reported per call AND per
 * token of context so linear-vs-quadratic is readable off two runs. */
static double g_mt_proj, g_mt_index, g_mt_attn;
static long g_mn_calls; static double g_mn_seen;
static inline int optime_on(void) {
    if (g_optime < 0) g_optime = getenv("COLI_TIMERS") && atoi(getenv("COLI_TIMERS"));
    return g_optime;
}
static inline double optime_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

/* SwiGLU clampata: il gate ha solo il tetto, up e' limitato da entrambi i lati.
 * Vale sia per l'MLP denso che per gli esperti -- il testo di GLM-5.3 NON usa
 * la SiLU semplice, ed e' un errore che darebbe un modello che parla bene e
 * sbaglia. */
static void swiglu_clamped(float *gate, const float *up, int n, float limit) {
    for (int i = 0; i < n; i++) {
        float g = gate[i] > limit ? limit : gate[i];
        float u = up[i] < -limit ? -limit : (up[i] > limit ? limit : up[i]);
        gate[i] = siluf_(g) * u;
    }
}

static void mlp3(float *out, const float *x, const Mat *g, const Mat *u, const Mat *d,
                 float limit, float *sg, float *su) {
    mv(sg, g, x); mv(su, u, x);
    swiglu_clamped(sg, su, g->rows, limit);
    mv(out, d, sg);
}

/* G13: mlp3() on the GPU path is three separate submit+wait round trips
 * (mv(gate), mv(up), mv(down)), one call per token -- the shared expert
 * calls it `tokens` times per layer, always (never cached, never skipped).
 *
 * The routed-expert fused kernel (coli_vk_expert_group / coli_vk_gate_up)
 * looked like the fix -- one submit for gate+up+down together -- but its
 * shader computes silu(gate)*up with NO CLAMP: it has no `limit` push
 * constant at all (c/shaders/qmatmul_gate_up.comp). GLM-5.3's swiglu_limit
 * is 10.0, not the disabled default, and it is regularly exceeded in
 * practice: routing the shared expert through that kernel and diffing
 * against the pristine (clamped) CPU path failed the --logits oracle
 * outright on the very first run. That the routed experts have used this
 * same unclamped kernel since the expert-group path first landed is a
 * pre-existing characteristic of the shipped GPU path, not a bug this item
 * introduces or is in scope to fix -- no oracle had ever caught it because
 * no prior change put a clamped and an unclamped computation of the SAME
 * op side by side. Out of scope for a one-day port; left as a finding.
 *
 * What DOES fuse without touching numerics: coli_vk_matmul_pair, already
 * shipping for exactly this shape in c/kimi_k3.c's vk_expert_apply ("w1/w3
 * in one paired submit, SiTU-GLU on CPU, w2 down"). It runs the SAME plain
 * matmul shader coli_vk_matmul/mv() already use (no fused activation), just
 * two dispatches (gate, up) in one command buffer / one submit / one fence
 * wait instead of two. swiglu_clamped and the down projection are UNCHANGED
 * from mlp3 -- only the gate+up round trip merges. 3 submits/call -> 2.
 *
 * Bit-identical: same shader, same push constants, same per-row math as the
 * two mv() calls it replaces; batching independent dispatches into one
 * command buffer changes nothing about what either one computes. Returns 0
 * on any unmet precondition (mirrors mv()'s own fmt==1||fmt==4 GPU gate,
 * plus gate/up must share fmt+gs, which coli_vk_matmul_pair requires and
 * quantize_loaded already guarantees under one GLM53_BITS setting) and the
 * caller's existing mlp3() runs exactly as before. */
static int shared_gate_up_gpu(const GLayer *l, const float *x, float *sg, float *su, int S) {
#ifndef COLI_VULKAN
    (void)l; (void)x; (void)sg; (void)su; (void)S;
    return 0;                       /* G13 shipped without this guard: the CPU-only build did not compile */
#else
    if (!g_vk_ready) return 0;
    Mat *g = (Mat *)&l->rg, *u = (Mat *)&l->ru;   /* vk cache write, as mv() does */
    if (!(g->fmt == 1 || g->fmt == 4) || !(u->fmt == 1 || u->fmt == 4)) return 0;
    if (g->fmt != u->fmt || g->gs != u->gs || g->columns != u->columns) return 0;
    return coli_vk_matmul_pair((ColiVkTensor **)&g->vk, sg,
                               g->fmt == 4 ? (const void *)g->q4 : (const void *)g->q8, g->s, g->rows,
                               (ColiVkTensor **)&u->vk, su,
                               u->fmt == 4 ? (const void *)u->q4 : (const void *)u->q8, u->s, u->rows,
                               g->fmt, x, S, g->columns, g->gs);
#endif
}

static void mlp3_shared(float *out, const float *x, const GLayer *l,
                        float limit, float *sg, float *su) {
    if (!shared_gate_up_gpu(l, x, sg, su, 1)) { mv(sg, &l->rg, x); mv(su, &l->ru, x); }
    swiglu_clamped(sg, su, l->rg.rows, limit);
    mv(out, &l->rd, sg);
}

/* P2.2: the shared expert for S rows in three calls instead of 3*S. sg/su
 * must hold S * rg.rows floats. Row t of out is bit-identical to
 * mlp3_shared() on row t of x. */
static void mlp3_shared_rows(float *out, const float *x, int S, const GLayer *l,
                             float limit, float *sg, float *su) {
    const int W = l->rg.rows;
    if (!shared_gate_up_gpu(l, x, sg, su, S)) { mv_rows_s(sg, &l->rg, x, S); mv_rows_s(su, &l->ru, x, S); }
    swiglu_clamped(sg, su, S * W, limit);
    mv_rows_s(out, &l->rd, sg, S);
}

/* ---------- KDA: proiezioni, gate, ricorrenza ---------- */
static int vk_batch_mv_chain(const Mat *const *ws, float *const *os, const int *srcs,
                             int n, const float *xin, int I);
static void mv_cpu(float *out, const Mat *w, const float *x) {
    switch (w->fmt) {
    case 4: matmul_i4_grouped(out, x, w->q4, w->s, 1, w->columns, w->rows, w->gs); break;
    case 1: matmul_q(out, x, w->q8, w->s, 1, w->columns, w->rows); break;
    default: matmul(out, x, w->f, 1, w->columns, w->rows); break;
    }
}
/* G12: run the recurrence on dev0 (G12-KDA-GPU-SPEC-2026-09-05.md). Off by
 * default -- it is NOT bit-identical (GLSL exp, tree-reduced norms), so it
 * ships behind a knob per CLAUDE.md. COLI_KDA_CPU takes precedence. */
/* P2: GLM53_PREFILL_UNBATCHED=1 keeps every prefill stage on its per-token
 * loop -- the pre-P2 behaviour, for bisecting a numerics or timing question.
 * Every batched path is bit-identical per row, so this is a diagnostic, not a
 * numerics knob. */
static int g_prefill_unbatched_v = -1;
static int g_prefill_unbatched(void) {
    if (g_prefill_unbatched_v < 0)
        g_prefill_unbatched_v = getenv("GLM53_PREFILL_UNBATCHED") ? atoi(getenv("GLM53_PREFILL_UNBATCHED")) : 0;
    return g_prefill_unbatched_v;
}
#define GLM53_MAX_SLOTS 16

/* P6b: KV_SLOTS letto in due posti che devono concordare -- model_load, che
 * alloca il pool di stato KDA sul dispositivo PRIMA del preload degli esperti,
 * e slots_init, che apre gli slot. Se divergessero, il pool sarebbe corto e il
 * motore ricadrebbe sulla CPU senza motivo. */
static int kv_slots_wanted(void) {
    const char *setting = getenv("KV_SLOTS");
    int n = setting ? atoi(setting) : 1;
    if (n < 1) n = 1;
    if (n > GLM53_MAX_SLOTS) n = GLM53_MAX_SLOTS;
    return n;
}
static int g_kda_gpu = -1;
static int kda_gpu_on(void) {
    if (g_kda_gpu < 0) g_kda_gpu = getenv("COLI_KDA_GPU") ? atoi(getenv("COLI_KDA_GPU")) : 0;
    return g_kda_gpu;
}
/* P5.2: one submit per layer per chunk for the prefill recurrence. On by
 * default; =0 restores the S per-token submits, for the before/after profile
 * in one binary. */
static int g_kda_rows = -1;
static int g_kda_rows_on(void) {
    if (g_kda_rows < 0) g_kda_rows = getenv("COLI_KDA_ROWS") ? atoi(getenv("COLI_KDA_ROWS")) : 1;
    return g_kda_rows;
}
static int g_kda_cpu = -1;
static int g_kda_cpu_on(void) {
    if (g_kda_cpu < 0) g_kda_cpu = getenv("COLI_KDA_CPU") ? atoi(getenv("COLI_KDA_CPU")) : 0;
    return g_kda_cpu;
}
#define KMV(o, w, xx) do { \
    if (g_kda_cpu < 0) g_kda_cpu = getenv("COLI_KDA_CPU") ? atoi(getenv("COLI_KDA_CPU")) : 0; \
    if (g_kda_cpu) mv_cpu((o), (w), (xx)); else mv((o), (w), (xx)); \
} while (0)

/* KMV for S rows: the same CPU/GPU choice as KMV, on mv_rows_s / the S-row CPU
 * kernels. Row-wise bit-identical to S KMV calls. */
static void kmv_rows(float *out, const Mat *w, const float *x, int S) {
    if (g_kda_cpu < 0) g_kda_cpu = getenv("COLI_KDA_CPU") ? atoi(getenv("COLI_KDA_CPU")) : 0;
    if (!g_kda_cpu) { mv_rows_s(out, w, x, S); return; }
    switch (w->fmt) {
    case 4: matmul_i4_grouped(out, x, w->q4, w->s, S, w->columns, w->rows, w->gs); break;
    case 1: matmul_q(out, x, w->q8, w->s, S, w->columns, w->rows); break;
    default: matmul(out, x, w->f, S, w->columns, w->rows); break;
    }
}

/* P2.1 (PREFILL-ROADMAP P2, P2-BATCH-DENSE-SPEC): a prefill chunk of S rows
 * through one KDA layer in five stages instead of S per-token passes.
 *
 *   A  the 8 projections, S rows each (one call per matrix per chunk)
 *   B  decay / beta gating, per row
 *   C  the recurrence, per token, SEQUENTIAL -- on the CPU state
 *   D  head norm x o_norm x sigmoid(gate), per row
 *   E  ko, S rows in one call
 *
 * When the recurrence lives on the device (gpu != 0) the state is synced to
 * the CPU copy before C and uploaded back after it, so decode resumes on the
 * device from the chunk's final state (G12's kda_sync / kda_upload). With
 * COLI_KDA_GPU=0 every stage is the per-token code in the per-token order:
 * bit-identical. With the device recurrence, prefill moves from the GLSL
 * chain to the CPU recurrence -- closer to the pristine CPU numerics; decode
 * is untouched. Timers: A -> proj, B -> decay, C -> step, D -> norm, E -> ko,
 * so the per-op table stays comparable. */
static void kda_layer_rows(const Cfg *c, const GLayer *l, const float *x, int S,
                           float *out, float *state, float *window, float *scratch,
                           int layer, int gpu, int slot) {
    const int P = c->kda_proj, H = c->kda_heads, D = c->kda_hd;
    float *q = malloc((size_t)S * P * sizeof(float));
    float *k = malloc((size_t)S * P * sizeof(float));
    float *v = malloc((size_t)S * P * sizeof(float));
    float *low = malloc((size_t)S * D * sizeof(float));
    float *lowg = malloc((size_t)S * D * sizeof(float));
    float *beta = malloc((size_t)S * H * sizeof(float));
    float *decay = malloc((size_t)S * P * sizeof(float));
    float *gate = malloc((size_t)S * P * sizeof(float));
    float *qkv = malloc((size_t)3 * P * sizeof(float));
    float *core = malloc((size_t)S * P * sizeof(float));
    float *normed = malloc((size_t)S * P * sizeof(float));
    if (!q || !k || !v || !low || !lowg || !beta || !decay || !gate || !qkv || !core || !normed) {
        fprintf(stderr, "OOM nel KDA a righe\n"); exit(1);
    }
    /* A */
    const double _t0 = optime_on() ? optime_now() : 0.0;
    kmv_rows(q, &l->kq, x, S);
    kmv_rows(k, &l->kk, x, S);
    kmv_rows(v, &l->kv, x, S);
    kmv_rows(low, &l->kfa, x, S);
    kmv_rows(beta, &l->kb, x, S);
    kmv_rows(lowg, &l->kga, x, S);
    kmv_rows(decay, &l->kfb, low, S);
    kmv_rows(gate, &l->kgb, lowg, S);
    if (optime_on()) { g_kt_proj += optime_now() - _t0; g_kn_batched += S; }
    /* B */
    const double _t1 = optime_on() ? optime_now() : 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < S; t++) {
        float *dc = decay + (size_t)t * P;
        for (int h = 0; h < H; h++) {
            const float alpha = expf(l->alog[h]);
            for (int d = 0; d < D; d++) {
                int i = h * D + d;
                dc[i] = c->gate_lb * sigmoidf_(alpha * (dc[i] + l->dt[i]));
            }
        }
        float *bt = beta + (size_t)t * H;
        for (int h = 0; h < H; h++) bt[h] = sigmoidf_(bt[h]);
    }
    if (optime_on()) { g_kt_decay += optime_now() - _t1; }
    /* C -- the state stays where it lives. On the device (gpu != 0) each
     * token is one coli_vk_kda_step submit, the same shader and the same
     * round-trip count as the per-token chain; reading the 4 MB state back
     * per layer per chunk instead (kda_sync from the host-visible arena is
     * a memcpy from write-combined memory) cost 11 s on a 21-token prompt
     * (p2c gate 2026-09-06: 3.1 s -> 14.6 s). On the CPU (gpu == 0) it is
     * the per-token code in the per-token order: bit-identical. */
    const double _t2 = optime_on() ? optime_now() : 0.0;
#ifndef COLI_VULKAN
    (void)layer; (void)gpu; (void)slot;
#endif
#ifdef COLI_VULKAN
    /* P5.2: the S dispatches of one chunk in ONE command buffer, one submit,
     * one fence, with a compute barrier between consecutive tokens -- the
     * recurrence stays sequential, only the S-1 extra round trips go. Same
     * shader, same push constants but `tok`, same device state at the end, so
     * P6b's per-slot state and P7's capture are unaffected and the result is
     * bit-identical to the per-token loop below. COLI_KDA_ROWS=0 restores it. */
    if (gpu && S > 1 && g_kda_rows_on() &&
        coli_vk_kda_step_rows(layer, slot, S, q, k, v, decay, beta, 1e-6f, core)) {
        /* all S tokens advanced on the device */
    } else
#endif
    for (int t = 0; t < S; t++) {
        memcpy(qkv,         q + (size_t)t * P, (size_t)P * sizeof(float));
        memcpy(qkv + P,     k + (size_t)t * P, (size_t)P * sizeof(float));
        memcpy(qkv + 2 * P, v + (size_t)t * P, (size_t)P * sizeof(float));
#ifdef COLI_VULKAN
        if (gpu && coli_vk_kda_step(layer, slot, qkv, decay + (size_t)t * P, beta + (size_t)t * H,
                                    1e-6f, core + (size_t)t * P))
            continue;   /* state and window advanced on the device */
#endif
        coli_kda_step(core + (size_t)t * P, state, window, qkv, l->conv,
                      decay + (size_t)t * P, beta + (size_t)t * H,
                      H, D, D, c->conv_k, 1e-6f, scratch);
    }
    if (optime_on()) { g_kt_step += optime_now() - _t2; }
    /* D */
    const double _t3 = optime_on() ? optime_now() : 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < S; t++) {
        const float *gt = gate + (size_t)t * P;
        for (int h = 0; h < H; h++) {
            const float *src = core + (size_t)t * P + (size_t)h * D;
            float *dst = normed + (size_t)t * P + (size_t)h * D;
            float square = 0.0f;
            for (int d = 0; d < D; d++) square += src[d] * src[d];
            float inverse = 1.0f / sqrtf(square / D + c->eps);
            for (int d = 0; d < D; d++)
                dst[d] = src[d] * inverse * l->onorm[d] * sigmoidf_(gt[(size_t)h * D + d]);
        }
    }
    if (optime_on()) { g_kt_norm += optime_now() - _t3; }
    /* E */
    const double _t4 = optime_on() ? optime_now() : 0.0;
    kmv_rows(out, &l->ko, normed, S);
    if (optime_on()) { g_kt_ko += optime_now() - _t4; g_kn_calls += S; }
    free(normed); free(core); free(qkv); free(gate); free(decay); free(beta);
    free(lowg); free(low); free(v); free(k); free(q);
}

static void kda_layer(const Cfg *c, const GLayer *l, const float *x, int tokens,
                      float *out, float *state, float *window, float *scratch,
                      int layer, int gpu, int slot) {
    if (tokens > 1 && !g_prefill_unbatched()) {
        kda_layer_rows(c, l, x, tokens, out, state, window, scratch, layer, gpu, slot);
        return;
    }
    const int P = c->kda_proj, H = c->kda_heads, D = c->kda_hd;
    float *qkv = malloc((size_t)3 * P * sizeof(float));
    float *gate = malloc((size_t)P * sizeof(float));
    float *decay = malloc((size_t)P * sizeof(float));
    float *beta = malloc((size_t)H * sizeof(float));
    float *low = malloc((size_t)D * sizeof(float));
    float *core = malloc((size_t)P * sizeof(float));
    float *normed_gpu = malloc((size_t)P * sizeof(float));   /* G12 gpu==3 bisection */
    float *lowg = malloc((size_t)D * sizeof(float));
    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * c->hidden;
        /* kq/kk/kv/kfa/kb/kga sono indipendenti e leggono tutte la stessa riga:
         * un submit invece di sei. Il costo del percorso denso e' il round trip,
         * non la matrice. */
        /* kq/kk/kv/kfa/kb/kga leggono la stessa riga; kfb e kgb dipendono solo
         * da kfa e kga, quindi entrano nello STESSO submit dietro una barriera e
         * i due intermedi non tornano mai in RAM: due round trip invece di
         * quattro. */
        const double _tk0 = optime_on() ? optime_now() : 0.0;
#ifdef COLI_VULKAN
        /* G12 full chain: projections -> decay -> recurrence -> head norm -> ko
         * in ONE submit. Everything between the first projection and ko stays on
         * the device, so this replaces the whole per-token body below, both round
         * trips and all three CPU stages. Falls through on any refusal. */
        if ((gpu == 2 || gpu == 3) && !g_kda_cpu_on()) {
            Mat *cw[8] = { (Mat *)&l->kq, (Mat *)&l->kk, (Mat *)&l->kv, (Mat *)&l->kfa,
                           (Mat *)&l->kb, (Mat *)&l->kga, (Mat *)&l->kfb, (Mat *)&l->kgb };
            const int csrc[8] = { -1, -1, -1, -1, -1, -1, 3, 5 };
            ColiVkMM it[8];
            int ok = 1;
            for (int q = 0; q < 8 && ok; q++) {
                if (cw[q]->fmt != 1 && cw[q]->fmt != 4) ok = 0;
                it[q].tensor = (ColiVkTensor **)&cw[q]->vk;
                it[q].weights = (cw[q]->fmt == 4) ? (const void *)cw[q]->q4 : (const void *)cw[q]->q8;
                it[q].scales = cw[q]->s; it[q].fmt = cw[q]->fmt; it[q].O = cw[q]->rows;
                it[q].gs = cw[q]->gs; it[q].out = NULL; it[q].I = 0; it[q].src = csrc[q];
            }
            Mat *ko = (Mat *)&l->ko;
            /* gpu==3 bisects: chain through the head norm, ko on the CPU. */
            const int stop_at_norm = (gpu == 3);
            float *chain_dst = stop_at_norm ? normed_gpu : out + (size_t)t * c->hidden;
            if (ok && coli_vk_kda_layer(layer, slot, it, 8, row, c->hidden,
                                        (ColiVkTensor **)&ko->vk,
                                        (ko->fmt == 4) ? (const void *)ko->q4 : (const void *)ko->q8,
                                        ko->s, ko->fmt, ko->gs, ko->rows,
                                        c->gate_lb, 1e-6f, c->eps,
                                        stop_at_norm, chain_dst)) {
                if (stop_at_norm) KMV(out + (size_t)t * c->hidden, &l->ko, normed_gpu);
                if (optime_on()) {
                    /* one fused stage: attribute it to proj so kda's total stays
                     * comparable, and leave the other four buckets at zero rather
                     * than inventing a split the timers cannot see. */
                    g_kt_proj += optime_now() - _tk0;
                    g_kn_batched++; g_kn_calls++;
                }
                continue;
            }
        }
#endif
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
        }
        if (optime_on()) { g_kt_proj += optime_now() - _tk0; g_kn_batched += batched ? 1 : 0; }
        const double _tk1 = optime_on() ? optime_now() : 0.0;
        /* decadimento: gate_lower_bound * sigmoid(exp(A_log[h]) * (W_fb W_fa x + dt_bias))
         *
         * exp(A_log[h]) dipende solo dal peso: fuori dal ciclo su d si calcola
         * 64 volte per token invece di 8.192 (G3: i trascendentali sono ~1 ms
         * dei 2,3 ms per chiamata). Stesso expf sullo stesso ingresso, quindi
         * stessi bit. */
        for (int h = 0; h < H; h++) {
            const float alpha = expf(l->alog[h]);
            for (int d = 0; d < D; d++) {
                int i = h * D + d;
                decay[i] = c->gate_lb * sigmoidf_(alpha * (decay[i] + l->dt[i]));
            }
        }
        for (int h = 0; h < H; h++) beta[h] = sigmoidf_(beta[h]);
        if (optime_on()) { g_kt_decay += optime_now() - _tk1; }
        const double _tk2 = optime_on() ? optime_now() : 0.0;
#ifdef COLI_VULKAN
        /* The recurrence, and only it: decay/beta gating above and the norm+ko
         * below still run where they did. This is the increment that measures
         * the shader's real in-engine dispatch cost before the full chain is
         * built -- it ADDS a round trip, so it is not expected to be a win. */
        if (gpu && coli_vk_kda_step(layer, slot, qkv, decay, beta, 1e-6f, core)) {
            /* state and window advanced on the device */
        } else
#endif
        coli_kda_step(core, state, window, qkv, l->conv, decay, beta,
                      H, D, D, c->conv_k, 1e-6f, scratch);
        if (optime_on()) { g_kt_step += optime_now() - _tk2; }
        const double _tk3 = optime_on() ? optime_now() : 0.0;
        /* uscita: RMSNorm per testa, pesata da o_norm, moltiplicata dal gate
         * low-rank, poi la proiezione di uscita. */
        float *normed = qkv;                         /* riuso: 3P >= P */
        for (int h = 0; h < H; h++) {
            const float *src = core + (size_t)h * D;
            float *dst = normed + (size_t)h * D;
            float square = 0.0f;
            for (int d = 0; d < D; d++) square += src[d] * src[d];
            float inverse = 1.0f / sqrtf(square / D + c->eps);
            for (int d = 0; d < D; d++)
                dst[d] = src[d] * inverse * l->onorm[d] * sigmoidf_(gate[(size_t)h * D + d]);
        }
        if (optime_on()) { g_kt_norm += optime_now() - _tk3; }
        const double _tk4 = optime_on() ? optime_now() : 0.0;
        KMV(out + (size_t)t * c->hidden, &l->ko, normed);
        if (optime_on()) { g_kt_ko += optime_now() - _tk4; g_kn_calls++; }
    }
    free(normed_gpu);
    free(lowg); free(core); free(low); free(beta); free(decay); free(gate); free(qkv);
}

/* N matrici indipendenti contro lo stesso vettore d'ingresso, in un submit
 * solo. Ritorna 0 se il batch non e' servibile, e il chiamante ricade sui mv()
 * singoli. */
static int vk_batch_mv_chain(const Mat *const *ws, float *const *os, const int *srcs,
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
    return coli_vk_matmul_multi(it, n, xin, I);
#else
    (void)ws; (void)os; (void)srcs; (void)n; (void)xin; (void)I;
    return 0;
#endif
}

static int vk_batch_mv(const Mat *const *ws, float *const *os, int n,
                       const float *xin, int I) {
    return vk_batch_mv_chain(ws, os, NULL, n, xin, I);
}

/* ---------- MLA + indexer con k-pool ---------- */

/* P5.1 (PREFILL-ROADMAP P5): the MLA attention core's score pass with the
 * HEADS in the SIMD lanes.
 *
 * Found by RP4: at 3 462 tokens `mla.attn` is 69.6 ms/token, 40 % of the whole
 * prefill token, and `tools/hot-expert/rome_mlaattn.c` shows where it goes.
 * The per-(token, head) score loop
 *
 *     for (d) dot += q[d] * c_j[d];
 *
 * runs at **10.0-10.8 GMAC/s on eight threads** (0.4 MAC/cycle/thread): gcc 15
 * vectorises the PRODUCTS (vmulps over 8 d at a time) but must keep the
 * summation strictly sequential, so the loop is bound by the latency of the
 * scalar add chain, not by the machine's FP throughput. The dot is ~73 % of
 * `attn` at 3 462 tokens.
 *
 * The fix does not change one flop. It transposes the token's 64 absorbed
 * queries once, to qT[d][h], and then computes all 64 heads' scores against
 * one latent row at a time: lane h accumulates over d in ascending order, one
 * rounded product added per step -- exactly the reference's arithmetic, now in
 * 64 independent chains instead of one. Measured 215-230 GMAC/s, 18-22x, with
 * the same numbers at a 7 MB (one layer) and a 77 MB (eleven layers) latent
 * pool, so it is compute-bound and not an L3 artefact.
 *
 * Bit-identical, and that is not an assumption: `rome_mlaattn.c` compares
 * every one of the 131 072 (head, slot) scores at the widest shape and reports
 * 0 mismatches. The one trap is FMA -- the reference rounds each product
 * before accumulating, gcc contracts an intrinsic mul+add straight back into
 * vfmadd231ps, and `#pragma STDC FP_CONTRACT OFF` does not stop it; the empty
 * asm below does, at no instruction cost.
 *
 * COLI_MLA_HEADVEC=0 restores the per-head scalar loop (same numbers, for the
 * before/after profile in one binary). */
#if defined(__AVX2__) && defined(__FMA__) && (defined(__GNUC__) || defined(__clang__))
#define GLM53_MLA_HEADVEC 1
#define MLA_MULADD(acc, x, y) do { __m256 m_ = _mm256_mul_ps((x), (y)); \
                                   __asm__("" : "+x"(m_)); \
                                   (acc) = _mm256_add_ps((acc), m_); } while (0)
#endif

static int g_router_lanes = -1;
static int router_lanes_on(void) {
    if (g_router_lanes < 0) {
#ifdef GLM53_MLA_HEADVEC
        const char *e = getenv("COLI_ROUTER_LANES");
        g_router_lanes = e ? atoi(e) : 1;
#else
        g_router_lanes = 0;
#endif
    }
    return g_router_lanes;
}

static int g_mla_headvec = -1;
static int mla_headvec_on(void) {
    if (g_mla_headvec < 0) {
#ifdef GLM53_MLA_HEADVEC
        const char *e = getenv("COLI_MLA_HEADVEC");
        g_mla_headvec = e ? atoi(e) : 1;
#else
        g_mla_headvec = 0;
#endif
    }
    return g_mla_headvec;
}

/* a[N][K] -> aT[K][N], 8x8 blocked. */
static void glm_transpose(float *aT, const float *a, int N, int K) {
    const int nb = N & ~7, kb = K & ~7;
    for (int n = 0; n < nb; n += 8)
        for (int k = 0; k < kb; k += 8)
            for (int nn = 0; nn < 8; nn++)
                for (int kk = 0; kk < 8; kk++)
                    aT[(size_t)(k + kk) * N + n + nn] = a[(size_t)(n + nn) * K + k + kk];
    for (int n = 0; n < N; n++)
        for (int k = (n < nb) ? kb : 0; k < K; k++)
            aT[(size_t)k * N + n] = a[(size_t)n * K + k];
}

/* dst[j] = scale * sum_k aT[k][j] * b[k], for all N lanes j at once, each lane
 * accumulating over k in ascending order with the product rounded before it is
 * added -- the summation order of the scalar `sum += a[k]*b[k]` this replaces.
 * The 64-lane block is what breaks the add-chain latency (eight independent
 * chains); the 32- and 8-lane blocks and the scalar tail cover any N. */
static void glm_lane_dots(float *dst, const float *aT, const float *b,
                          int N, int K, float scale) {
    const int H = N, L = K;
    const float *qT = aT, *c_j = b;
    int h = 0;
#ifdef GLM53_MLA_HEADVEC
    for (; h + 64 <= H; h += 64) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        const float *p = qT + h;
        for (int d = 0; d < L; d++, p += H) {
            const __m256 b = _mm256_broadcast_ss(c_j + d);
            MLA_MULADD(a0, b, _mm256_loadu_ps(p));
            MLA_MULADD(a1, b, _mm256_loadu_ps(p + 8));
            MLA_MULADD(a2, b, _mm256_loadu_ps(p + 16));
            MLA_MULADD(a3, b, _mm256_loadu_ps(p + 24));
            MLA_MULADD(a4, b, _mm256_loadu_ps(p + 32));
            MLA_MULADD(a5, b, _mm256_loadu_ps(p + 40));
            MLA_MULADD(a6, b, _mm256_loadu_ps(p + 48));
            MLA_MULADD(a7, b, _mm256_loadu_ps(p + 56));
        }
        const __m256 s = _mm256_set1_ps(scale);
        _mm256_storeu_ps(dst + h,      _mm256_mul_ps(a0, s));
        _mm256_storeu_ps(dst + h + 8,  _mm256_mul_ps(a1, s));
        _mm256_storeu_ps(dst + h + 16, _mm256_mul_ps(a2, s));
        _mm256_storeu_ps(dst + h + 24, _mm256_mul_ps(a3, s));
        _mm256_storeu_ps(dst + h + 32, _mm256_mul_ps(a4, s));
        _mm256_storeu_ps(dst + h + 40, _mm256_mul_ps(a5, s));
        _mm256_storeu_ps(dst + h + 48, _mm256_mul_ps(a6, s));
        _mm256_storeu_ps(dst + h + 56, _mm256_mul_ps(a7, s));
    }
    for (; h + 32 <= H; h += 32) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        const float *p = qT + h;
        for (int d = 0; d < L; d++, p += H) {
            const __m256 b = _mm256_broadcast_ss(c_j + d);
            MLA_MULADD(a0, b, _mm256_loadu_ps(p));
            MLA_MULADD(a1, b, _mm256_loadu_ps(p + 8));
            MLA_MULADD(a2, b, _mm256_loadu_ps(p + 16));
            MLA_MULADD(a3, b, _mm256_loadu_ps(p + 24));
        }
        const __m256 s = _mm256_set1_ps(scale);
        _mm256_storeu_ps(dst + h,      _mm256_mul_ps(a0, s));
        _mm256_storeu_ps(dst + h + 8,  _mm256_mul_ps(a1, s));
        _mm256_storeu_ps(dst + h + 16, _mm256_mul_ps(a2, s));
        _mm256_storeu_ps(dst + h + 24, _mm256_mul_ps(a3, s));
    }
    for (; h + 8 <= H; h += 8) {
        __m256 a0 = _mm256_setzero_ps();
        const float *p = qT + h;
        for (int d = 0; d < L; d++, p += H) {
            const __m256 b = _mm256_broadcast_ss(c_j + d);
            MLA_MULADD(a0, b, _mm256_loadu_ps(p));
        }
        _mm256_storeu_ps(dst + h, _mm256_mul_ps(a0, _mm256_set1_ps(scale)));
    }
#endif
    for (; h < H; h++) {
        float dot = 0.0f;
        for (int d = 0; d < L; d++) dot += qT[(size_t)d * H + h] * c_j[d];
        dst[h] = dot * scale;
    }
}

/* P5b: the MLA weighted pool.
 *
 * COLI_MLA_POOL selects one of three shapes, so the profile can attribute the
 * two findings separately in ONE binary:
 *
 *   0  exactly the code P5 left behind: `pooled`/`score` are plain malloc'd
 *      per-thread slices, and the pool walks the layer's selected latent once
 *      PER HEAD.
 *   1  (DEFAULT) the same nest, with the per-thread slices 64-byte aligned and
 *      their stride rounded up to a cache line. `pooled` is L = 512 floats =
 *      2 048 bytes and `score` is width = 2 051 floats = 8 204: neither the
 *      base (glibc malloc gives 16 bytes) nor the stride is a multiple of 64,
 *      so the eight threads share the boundary lines of their neighbours'
 *      slices and write them `used` times per head. `rome_mlaattn.c` measures
 *      that artefact directly: at used = 1 444 the whole core is 43.4-45.4
 *      ms/token with the slices as they are and 14.7-15.7 with them padded,
 *      and the engine agrees -- `mla.attn` 51.6 -> 22.5 ms/token at 3 462
 *      tokens, 23.0 -> 13.7 at 781.
 *   2  the padded slices AND the blocked pool below. NOT the default: in the
 *      engine it is neutral, +0.2 ms/token at 781 tokens and -0.3 at 3 462
 *      (`mla.attn` 22.5 -> 22.3), because once the false sharing is gone the
 *      per-head pool runs at L3 bandwidth and the layer's latent (7.1 MB at
 *      3 462 tokens, 78 MB for all eleven) is L3-resident while it runs. The
 *      roadmap's rule is "default on only if faster at both prompt sizes",
 *      and it is not. The knob stays because §P5b's table is measured through
 *      it, and because the blocked form reads 2.8 MB per token per layer
 *      where the per-head form reads 180 -- which is the shape that wins if
 *      the latent ever stops fitting in L3.
 *
 * The pool is pooled[h][d] = sum_u w[u][h] * latent[slot[u]][d]. Written per
 * head it reads 64 x used x L x 4 = 180 MB per token per layer at used=1444
 * of which 2.8 MB is distinct. Blocked over (d-tile x head group) it reads
 * each latent byte once: the d-tiles are the parallel axis, each tile owns an
 * accumulator plane acc[dt][H] (h contiguous, 2 KB at dt = 8) that stays in
 * L1, and eight slots are folded at a time so one accumulator load+store
 * serves eight FMAs.
 *
 * Bit-identical, and the reason is the opposite of the score pass's: gcc 15
 * -O3 -march=native CONTRACTS `pooled[d] += w * c_j[d]` into vfmadd213ps
 * (-ffp-contract=fast is the GNU default; verified in the objdump), so the
 * reference term is fmaf(w, c, acc) with ONE rounding, and the blocked kernel
 * uses _mm256_fmadd_ps -- no MLA_MULADD barrier here, that one is for the
 * score reduction, which cannot vectorise and therefore rounds its products.
 * Every (h, d) still accumulates u = 0, 1, 2, ... in ascending order.
 * `rome_mlaattn.c` compares all 32 768 pooled values against the reference
 * nest and reports 0 mismatches. */
#define GLM53_POOL_DTILE 8

static int g_mla_pool = -1;
static int mla_pool_mode(void) {
    if (g_mla_pool < 0) {
        const char *e = getenv("COLI_MLA_POOL");
        g_mla_pool = e ? atoi(e) : 1;
        if (g_mla_pool < 0) g_mla_pool = 0;
#ifndef GLM53_MLA_HEADVEC
        if (g_mla_pool > 1) g_mla_pool = 1;   /* no AVX2: the blocked kernel */
#endif
    }
    return g_mla_pool;
}

/* malloc + hand-rounded to a 64-byte boundary: portable, and it avoids the
 * posix_memalign/_aligned_free asymmetry compat.h warns about. */
static float *glm_alloc64(size_t floats, void **raw) {
    char *p = malloc(floats * sizeof(float) + 64);
    *raw = p;
    if (!p) return NULL;
    return (float *)(void *)(((uintptr_t)p + 63) & ~(uintptr_t)63);
}

static void glm_pool_blocked(float *pooled_all, const float *wT,
                             const float *latent, const int *slot_at,
                             int used, int L, int H, float *accbuf, int dtile) {
    const int ntiles = (L + dtile - 1) / dtile;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int tb = 0; tb < ntiles; tb++) {
        const int d0 = tb * dtile;
        const int dt = (L - d0 < dtile) ? (L - d0) : dtile;
        float *acc = accbuf + (size_t)tb * dtile * H;
        memset(acc, 0, (size_t)dt * H * sizeof(float));
        int u = 0;
#ifdef GLM53_MLA_HEADVEC
        for (; u + 8 <= used; u += 8) {
            const float *r[8]; const float *w[8];
            for (int k = 0; k < 8; k++) {
                r[k] = latent + (size_t)slot_at[u + k] * L + d0;
                w[k] = wT + (size_t)(u + k) * H;
            }
            for (int hg = 0; hg + 8 <= H; hg += 8) {
                const __m256 v0 = _mm256_loadu_ps(w[0] + hg), v1 = _mm256_loadu_ps(w[1] + hg);
                const __m256 v2 = _mm256_loadu_ps(w[2] + hg), v3 = _mm256_loadu_ps(w[3] + hg);
                const __m256 v4 = _mm256_loadu_ps(w[4] + hg), v5 = _mm256_loadu_ps(w[5] + hg);
                const __m256 v6 = _mm256_loadu_ps(w[6] + hg), v7 = _mm256_loadu_ps(w[7] + hg);
                float *a = acc + hg;
                for (int dd = 0; dd < dt; dd++, a += H) {
                    __m256 y = _mm256_loadu_ps(a);
                    y = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(r[0] + dd), y);
                    y = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(r[1] + dd), y);
                    y = _mm256_fmadd_ps(v2, _mm256_broadcast_ss(r[2] + dd), y);
                    y = _mm256_fmadd_ps(v3, _mm256_broadcast_ss(r[3] + dd), y);
                    y = _mm256_fmadd_ps(v4, _mm256_broadcast_ss(r[4] + dd), y);
                    y = _mm256_fmadd_ps(v5, _mm256_broadcast_ss(r[5] + dd), y);
                    y = _mm256_fmadd_ps(v6, _mm256_broadcast_ss(r[6] + dd), y);
                    y = _mm256_fmadd_ps(v7, _mm256_broadcast_ss(r[7] + dd), y);
                    _mm256_storeu_ps(a, y);
                }
            }
            const int htail = H & ~7;
            for (int h = htail; h < H; h++)
                for (int k = 0; k < 8; k++)
                    for (int dd = 0; dd < dt; dd++)
                        acc[(size_t)dd * H + h] = fmaf(w[k][h], r[k][dd], acc[(size_t)dd * H + h]);
        }
        for (; u < used; u++) {
            const float *r = latent + (size_t)slot_at[u] * L + d0;
            const float *w = wT + (size_t)u * H;
            int hg = 0;
            for (; hg + 8 <= H; hg += 8) {
                const __m256 v = _mm256_loadu_ps(w + hg);
                float *a = acc + hg;
                for (int dd = 0; dd < dt; dd++, a += H)
                    _mm256_storeu_ps(a, _mm256_fmadd_ps(v, _mm256_broadcast_ss(r + dd),
                                                        _mm256_loadu_ps(a)));
            }
            for (; hg < H; hg++)
                for (int dd = 0; dd < dt; dd++)
                    acc[(size_t)dd * H + hg] = fmaf(w[hg], r[dd], acc[(size_t)dd * H + hg]);
        }
#else
        for (; u < used; u++) {
            const float *r = latent + (size_t)slot_at[u] * L + d0;
            const float *w = wT + (size_t)u * H;
            for (int h = 0; h < H; h++)
                for (int dd = 0; dd < dt; dd++)
                    acc[(size_t)dd * H + h] = fmaf(w[h], r[dd], acc[(size_t)dd * H + h]);
        }
#endif
        for (int h = 0; h < H; h++)
            for (int dd = 0; dd < dt; dd++)
                pooled_all[(size_t)h * L + d0 + dd] = acc[(size_t)dd * H + h];
    }
}

static void mla_layer(const Cfg *c, const GLayer *l, const float *x, int tokens,
                      float *out, GLayerState *st, int base) {
    const int H = c->n_heads, QK = c->qk_nope, V = c->v_head;
    const int IH = c->index_nh, ID = c->index_hd;
    const int seen = base + tokens;
    float *qa = malloc((size_t)tokens * c->q_lora * sizeof(float));
    const int L = c->kv_lora;
    float *queries = malloc((size_t)tokens * H * QK * sizeof(float));
    float *absorbed = malloc((size_t)tokens * H * L * sizeof(float));
    float *latent = st->latent;           /* tutto il prefisso, non solo i nuovi */
    float *iq = malloc((size_t)tokens * IH * ID * sizeof(float));
    float *ik = st->ikeys;
    float *gates = st->igates;
    float *head_w = malloc((size_t)tokens * IH * sizeof(float));
    unsigned char *valid = malloc((size_t)seen);
    memset(valid, 1, (size_t)seen);
    const double _tm0 = optime_on() ? optime_now() : 0.0;

    /* P2.4: with a chunk of rows, each projection runs once on the S-row slab
     * (its outputs for consecutive t are contiguous: qa by t, latent/ik/gates
     * by absolute position base+t, head_w by t), the per-row norms run in
     * parallel, then qb/iwq on the normalised slab. Per row bit-identical to
     * the per-token path below; the absorb stays per token (P5). */
    const int mla_rows = tokens > 1 && !g_prefill_unbatched();
    if (mla_rows) {
        mv_rows_s(qa, &l->qa, x, tokens);
        mv_rows_s(latent + (size_t)base * L, &l->kva, x, tokens);
        mv_rows_s(ik + (size_t)base * ID, &l->iwk, x, tokens);
        mv_rows_s(gates + (size_t)base * ID, &l->ikpg, x, tokens);
        mv_rows_s(head_w, &l->iwp, x, tokens);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int t = 0; t < tokens; t++) {
            const int at = base + t;
            rms(qa + (size_t)t * c->q_lora, qa + (size_t)t * c->q_lora, l->qa_ln, c->q_lora, c->eps);
            rms(latent + (size_t)at * L, latent + (size_t)at * L, l->kva_ln, L, c->eps);
            layer_norm(ik + (size_t)at * ID, ik + (size_t)at * ID, l->ik_nw, l->ik_nb, ID, 1e-5f);
        }
        mv_rows_s(queries, &l->qb, qa, tokens);
        mv_rows_s(iq, &l->iwq, qa, tokens);
    }
    for (int t = 0; t < tokens; t++) {
        const int at = base + t;          /* posizione assoluta nella cache */
        const float *row = x + (size_t)t * c->hidden;
        float *qn = qa + (size_t)t * c->q_lora;
        float *here = latent + (size_t)at * L;
        float *kraw = ik + (size_t)at * ID;
        /* qa/kva/iwk/ikpg/iwp leggono tutte la stessa riga: un submit invece di
         * cinque. Sul percorso denso si paga il round trip, non la matrice. */
        if (!mla_rows) {
            const Mat *bw[5] = { &l->qa, &l->kva, &l->iwk, &l->ikpg, &l->iwp };
            float *bo[5] = { qn, here, kraw,
                             gates + (size_t)at * ID, head_w + (size_t)t * IH };
            if (!vk_batch_mv(bw, bo, 5, row, c->hidden)) {
                mv(qn, &l->qa, row);
                mv(here, &l->kva, row);
                mv(kraw, &l->iwk, row);
                mv(gates + (size_t)at * ID, &l->ikpg, row);
                mv(head_w + (size_t)t * IH, &l->iwp, row);
            }
        }
        if (!mla_rows) {
        rms(qn, qn, l->qa_ln, c->q_lora, c->eps);
        rms(here, here, l->kva_ln, L, c->eps);
        layer_norm(kraw, kraw, l->ik_nw, l->ik_nb, ID, 1e-5f);
        }
        /* qb e iwq leggono entrambe il q_a normalizzato: altro submit unico. */
        if (!mla_rows) {
            const Mat *bw[2] = { &l->qb, &l->iwq };
            float *bo[2] = { queries + (size_t)t * H * QK, iq + (size_t)t * IH * ID };
            if (!vk_batch_mv(bw, bo, 2, qn, c->q_lora)) {
                mv(queries + (size_t)t * H * QK, &l->qb, qn);
                mv(iq + (size_t)t * IH * ID, &l->iwq, qn);
            }
        }
        /* la query entra nello spazio del latente una volta per testa, invece
         * che il latente nello spazio della query una volta per posizione.
         *
         * Le teste sono indipendenti: ognuna scrive la propria fetta di
         * absorbed[] e legge la propria fetta di queries[], nessuno stato
         * condiviso. mv_rows ha gia' un proprio #pragma omp parallel for
         * (quant.h) sulle proprie L righe -- annidarlo qui dentro collassa a
         * un thread per chiamata (default max-active-levels=1 su GCC/libgomp
         * su questa macchina, verificato empiricamente G8 2026-09-05), quindi
         * non e' un rischio di overselling, solo un'occasione oggi sprecata:
         * 64 team OpenMP minuscoli per token invece di uno solo. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int h = 0; h < H; h++)
            mv_rows(absorbed + ((size_t)t * H + h) * L, &l->kvb_kt,
                    queries + ((size_t)t * H + h) * QK, h * L, L);
        for (int h = 0; h < IH; h++) head_w[(size_t)t * IH + h] /= sqrtf((float)IH);
    }

    if (optime_on()) { g_mt_proj += optime_now() - _tm0; }
    const double _tm1 = optime_on() ? optime_now() : 0.0;
    const int width = coli_sparse_index_width(c->index_topk, c->index_kpool, c->index_kpool_tail);
    int *selected = malloc((size_t)tokens * width * sizeof(int));
    if (coli_sparse_index_select_range(selected, iq, ik, gates, head_w, l->ikpa, valid,
                                       seen, IH, ID, c->index_kpool, c->index_topk,
                                       c->index_kpool_tail, base, seen)) {
        fprintf(stderr, "selezione indexer fallita\n"); exit(1);
    }
    /* GLM53_DUMP_INDEX=1 stampa le righe scelte dall'indexer: e' il primo
     * posto da guardare quando il motore diverge solo su certe lunghezze. */
    if (getenv("GLM53_DUMP_INDEX")) {
        for (int t = 0; t < tokens; t++) {
            fprintf(stderr, "index q=%d ->", base + t);
            for (int i = 0; i < width; i++) fprintf(stderr, " %d", selected[(size_t)t * width + i]);
            fprintf(stderr, "\n");
        }
    }
    /* Attenzione nello spazio del latente. La scala resta 1/sqrt(qk_nope):
     * il prodotto e' lo stesso numero di prima, calcolato in un altro ordine. */
    if (optime_on()) { g_mt_index += optime_now() - _tm1; }
    const double _tm2 = optime_on() ? optime_now() : 0.0;
    float *context = malloc((size_t)H * V * sizeof(float));
    /* score/pooled are the one thing 64 independent heads would share if run
     * concurrently -- each gets its own slice of a pool sized per thread,
     * the same fix as KDA's per-thread `memory` (G4): a shared slice would
     * corrupt every head's softmax while still emitting plausible tokens.
     * coli_kda_threads() (delta_attention.h, included above) already has the
     * #ifdef _OPENMP guard this needs. */
    const int nthreads_mla = coli_kda_threads();
    /* P5b.1: at mode 0 these are the plain malloc'd slices P5 left behind --
     * 2 048 and 8 204 bytes per thread from a 16-byte-aligned base, so the
     * neighbours' boundary cache lines are shared and written `used` times
     * per head. At mode >= 1 the base is 64-byte aligned and the stride is
     * rounded up to a cache line, which is the whole of P5b.1. */
    const int pool_mode = mla_pool_mode();
    const int pstride = pool_mode ? ((L + 15) & ~15) : L;
    const int sstride = pool_mode ? ((width + 15) & ~15) : width;
    void *pooled_raw = NULL, *score_raw = NULL;
    float *pooled_pool, *score_pool;
    if (pool_mode) {
        pooled_pool = glm_alloc64((size_t)nthreads_mla * pstride, &pooled_raw);
        score_pool  = glm_alloc64((size_t)nthreads_mla * sstride, &score_raw);
    } else {
        pooled_pool = malloc((size_t)nthreads_mla * pstride * sizeof(float));
        score_pool  = malloc((size_t)nthreads_mla * sstride * sizeof(float));
        pooled_raw = pooled_pool; score_raw = score_pool;
    }
    if (!pooled_pool || !score_pool) { fprintf(stderr, "OOM nell'attenzione MLA\n"); exit(1); }
    const float scale = 1.0f / sqrtf((float)QK);
    /* P5.1: the head-lane score pass wants the token's queries as qT[d][h] and
     * a place to leave the 64 scores of each slot. 128 KB + width*H floats
     * (525 KB at width 2051), allocated once per layer per chunk. */
    const int headvec = mla_headvec_on();
    float *qT = headvec ? malloc((size_t)L * H * sizeof(float)) : NULL;
    float *sc = headvec ? malloc((size_t)width * H * sizeof(float)) : NULL;
    int *slot_at = headvec ? malloc((size_t)width * sizeof(int)) : NULL;
    if (headvec && (!qT || !sc || !slot_at)) { fprintf(stderr, "OOM nell'attenzione MLA\n"); exit(1); }
    /* P5b: the blocked pool's H x L output plane and the per-tile accumulator
     * planes, once per layer per chunk (128 KB each at H = 64, L = 512). */
    const int blocked = headvec && pool_mode >= 2;
    void *pool_all_raw = NULL, *acc_raw = NULL;
    float *pooled_all = NULL, *accbuf = NULL;
    if (blocked) {
        const int ntiles = (L + GLM53_POOL_DTILE - 1) / GLM53_POOL_DTILE;
        pooled_all = glm_alloc64((size_t)H * L, &pool_all_raw);
        accbuf     = glm_alloc64((size_t)ntiles * GLM53_POOL_DTILE * H, &acc_raw);
        if (!pooled_all || !accbuf) { fprintf(stderr, "OOM nell'attenzione MLA\n"); exit(1); }
    }
    for (int t = 0; t < tokens; t++) {
        const int *chosen = selected + (size_t)t * width;
        if (headvec) {
            /* Same slots, same order, same arithmetic -- only the loop nest
             * changes: the 64 heads move into the SIMD lanes and the slots
             * become the parallel axis. `score[u]` below is then read out of
             * sc[u][h] instead of recomputed per head, and `top` is the max
             * over the same set. */
            int used_all = 0;
            for (int i = 0; i < width; i++) {
                const int at = chosen[i];
                if (at < 0 || at >= seen) continue;
                slot_at[used_all++] = at;
            }
            glm_transpose(qT, absorbed + (size_t)t * H * L, H, L);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int u = 0; u < used_all; u++)
                glm_lane_dots(sc + (size_t)u * H, qT, latent + (size_t)slot_at[u] * L,
                              H, L, scale);
            if (blocked) {
                /* P5b: the softmax turns each head's column of sc[u][h] into
                 * that head's weights IN PLACE -- same max, same expf, same
                 * double total, same (float) cast -- and then ONE walk of the
                 * latent serves all H heads. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int h = 0; h < H; h++) {
#ifdef _OPENMP
                    float *score = score_pool + (size_t)omp_get_thread_num() * sstride;
#else
                    float *score = score_pool;
#endif
                    float top = -INFINITY;
                    for (int u = 0; u < used_all; u++) {
                        score[u] = sc[(size_t)u * H + h];
                        if (score[u] > top) top = score[u];
                    }
                    if (!used_all) continue;
                    double total = 0.0;
                    for (int i = 0; i < used_all; i++) { score[i] = expf(score[i] - top); total += score[i]; }
                    for (int u = 0; u < used_all; u++)
                        sc[(size_t)u * H + h] = (float)(score[u] / total);
                }
                if (!used_all) {
                    memset(context, 0, (size_t)H * V * sizeof(float));
                } else {
                    glm_pool_blocked(pooled_all, sc, latent, slot_at, used_all, L, H,
                                     accbuf, GLM53_POOL_DTILE);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                    for (int h = 0; h < H; h++)
                        mv_rows(context + (size_t)h * V, &l->kvb_v,
                                pooled_all + (size_t)h * L, h * V, V);
                }
                mv(out + (size_t)t * c->hidden, &l->o, context);
                continue;
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int h = 0; h < H; h++) {
#ifdef _OPENMP
                float *pooled = pooled_pool + (size_t)omp_get_thread_num() * pstride;
                float *score = score_pool + (size_t)omp_get_thread_num() * sstride;
#else
                float *pooled = pooled_pool;
                float *score = score_pool;
#endif
                float top = -INFINITY;
                for (int u = 0; u < used_all; u++) {
                    score[u] = sc[(size_t)u * H + h];
                    if (score[u] > top) top = score[u];
                }
                float *result = context + (size_t)h * V;
                memset(result, 0, (size_t)V * sizeof(float));
                if (!used_all) continue;
                double total = 0.0;
                for (int i = 0; i < used_all; i++) { score[i] = expf(score[i] - top); total += score[i]; }
                memset(pooled, 0, (size_t)L * sizeof(float));
                for (int u = 0; u < used_all; u++) {
                    const float weight = (float)(score[u] / total);
                    const float *c_j = latent + (size_t)slot_at[u] * L;
                    for (int d = 0; d < L; d++) pooled[d] += weight * c_j[d];
                }
                mv_rows(result, &l->kvb_v, pooled, h * V, V);
            }
            mv(out + (size_t)t * c->hidden, &l->o, context);
            continue;
        }
        /* Heads are independent: each reads its own absorbed[] query and
         * writes its own context[] slice. mv_rows's own #pragma omp parallel
         * for (quant.h) collapses to one thread per call inside this region
         * -- max-active-levels defaults to 1 on this box's GCC/libgomp
         * (checked empirically, G8 2026-09-05) -- so nesting is not a
         * hazard, only a currently-wasted opportunity: two tiny OMP teams
         * spun up per head per token today (this loop's mv_rows call and the
         * absorb loop's above), which is what G3 actually measured. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int h = 0; h < H; h++) {
#ifdef _OPENMP
            float *pooled = pooled_pool + (size_t)omp_get_thread_num() * pstride;
            float *score = score_pool + (size_t)omp_get_thread_num() * sstride;
#else
            float *pooled = pooled_pool;
            float *score = score_pool;
#endif
            const float *q = absorbed + ((size_t)t * H + h) * L;
            float top = -INFINITY;
            int used = 0;
            for (int i = 0; i < width; i++) {
                const int at = chosen[i];
                if (at < 0 || at >= seen) continue;
                const float *c_j = latent + (size_t)at * L;
                float dot = 0.0f;
                for (int d = 0; d < L; d++) dot += q[d] * c_j[d];
                score[used] = dot * scale;
                if (score[used] > top) top = score[used];
                used++;
            }
            float *result = context + (size_t)h * V;
            memset(result, 0, (size_t)V * sizeof(float));
            if (!used) continue;
            double total = 0.0;
            for (int i = 0; i < used; i++) { score[i] = expf(score[i] - top); total += score[i]; }
            memset(pooled, 0, (size_t)L * sizeof(float));
            int seen_slot = 0;
            for (int i = 0; i < width; i++) {
                const int at = chosen[i];
                if (at < 0 || at >= seen) continue;
                const float weight = (float)(score[seen_slot++] / total);
                const float *c_j = latent + (size_t)at * L;
                for (int d = 0; d < L; d++) pooled[d] += weight * c_j[d];
            }
            mv_rows(result, &l->kvb_v, pooled, h * V, V);
        }
        mv(out + (size_t)t * c->hidden, &l->o, context);
    }
    if (optime_on()) { g_mt_attn += optime_now() - _tm2; g_mn_calls++; g_mn_seen += seen; }
    free(slot_at); free(sc); free(qT);
    free(pool_all_raw); free(acc_raw);
    free(score_raw); free(pooled_raw);

    free(context); free(selected); free(valid); free(head_w);
    free(iq); free(absorbed); free(queries); free(qa);
}

/* ---------- FFN: denso oppure MoE ---------- */
/* ================= streaming degli esperti =================
 *
 * 288 esperti per 42 layer sparsi, 14,2 MB l'uno in int4 gs64: 171 GB, che
 * stanno su disco e non in RAM. Per ogni token se ne toccano 8 per layer, e la
 * stessa manciata torna spesso, quindi ogni layer tiene una cache LRU di slot
 * e il resto arriva con una lettura quando serve.
 *
 * Uno slot e' un blocco unico che contiene i sei pezzi dell'esperto nell'ordine
 * in cui li vuole il calcolo. Le Mat che ne escono puntano dentro allo slot,
 * cosi' il codice del MoE non sa da dove arrivi il peso e resta quello di
 * prima. */
/* Slot per layer chiesti dalla riga di comando; 0 = decide il budget misurato. */
static int g_cap_override = 0;

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GLM53_EXPERT_PIECES 6

typedef struct ERef {
    int fd[GLM53_EXPERT_PIECES];
    int64_t off[GLM53_EXPERT_PIECES];
    int contig;                           /* i sei pezzi sono consecutivi in un file */
} ERef;

/* I sei pezzi non sono adiacenti nel file, ma non devono esserlo nemmeno in
 * memoria: expert_mats ci costruisce sopra solo tre viste in sola lettura. */
typedef struct { int eid; uint8_t *piece[GLM53_EXPERT_PIECES]; uint8_t *own; uint64_t used; } Slot;

/* Una mappatura per file, non per esperto: gli offset dei pezzi non sono
 * allineati alla pagina, ma mappando il file intero ci pensa il kernel.
 *
 * #1325 (upstream `dev`, 2026-09-08 merge): the mapping table itself now
 * lives in st.h as st_map_shard_range -- one implementation for every engine
 * instead of a private g_fmap[] here. What stays engine-local is what st.h
 * cannot know: whether EVERY expert of this checkpoint is servable from the
 * mapping (g_map_all sizes the LRU below) and the two counters [PROF] and the
 * streaming test print. */
static long     g_map_serve, g_map_copy;
static int      g_map_active, g_map_all;
typedef struct LCache { Slot *s; int n, cap; } LCache;

/* Lunghezze e posizioni dei sei pezzi dentro allo slot. Gate e up sono
 * [moe_inter, hidden], down e' [hidden, moe_inter]: stessi byte, forme
 * scambiate. */
static void expert_geometry(GModel *m) {
    const int64_t hidden = m->c.hidden, inter = m->c.moe_inter;
    const int64_t packed = inter * hidden / 2;
    const int64_t scales = inter * hidden / 64 * (int64_t)sizeof(float);
    const int64_t length[GLM53_EXPERT_PIECES] = { packed, scales, packed, scales, packed, scales };
    int64_t at = 0;
    for (int p = 0; p < GLM53_EXPERT_PIECES; p++) {
        m->e_len[p] = length[p];
        m->e_at[p] = at;
        at += length[p];
    }
    m->e_slot = at;
}

/* Indirizzi su disco di ogni esperto. Le lunghezze dichiarate dal file devono
 * combaciare con la geometria: un checkpoint che dice altro non e' questo
 * modello, e leggerlo lo stesso vorrebbe dire calcolare su byte a caso. */
static void expert_table_init(GModel *m) {
    const Cfg *c = &m->c;
    static const char *piece[GLM53_EXPERT_PIECES] = {
        "gate_proj.weight", "gate_proj.weight.qs",
        "up_proj.weight",   "up_proj.weight.qs",
        "down_proj.weight", "down_proj.weight.qs",
    };
    m->eref = calloc((size_t)c->n_layers * c->n_experts, sizeof(*m->eref));
    if (!m->eref) { fprintf(stderr, "OOM sulla tabella degli esperti\n"); exit(1); }

    const int from = c->first_dense > m->layer_begin ? c->first_dense : m->layer_begin;
    for (int i = from; i < m->layer_end; i++) {
        for (int e = 0; e < c->n_experts; e++) {
            ERef *ref = &m->eref[(size_t)i * c->n_experts + e];
            for (int p = 0; p < GLM53_EXPERT_PIECES; p++) {
                char name[512];
                snprintf(name, sizeof(name), "%slayers.%d.mlp.experts.%d.%s",
                         m->prefix, i, e, piece[p]);
                st_tensor *t = st_find(&m->S, name);
                if (!t) { fprintf(stderr, "manca %s\n", name); exit(1); }
                if (t->nbytes != m->e_len[p]) {
                    fprintf(stderr, "%s: %lld byte, attesi %lld\n", name,
                            (long long)t->nbytes, (long long)m->e_len[p]);
                    exit(1);
                }
                ref->fd[p] = t->fd;
                ref->off[p] = t->off;
            }
            /* Sei letture per esperto costano sei code invece di una. Vale la
             * pena accorgersi quando il file le ha gia' messe in fila. */
            ref->contig = 1;
            for (int p = 1; p < GLM53_EXPERT_PIECES; p++)
                if (ref->fd[p] != ref->fd[0] ||
                    ref->off[p] != ref->off[p - 1] + m->e_len[p - 1]) { ref->contig = 0; break; }
        }
    }
}

/* Quanti slot per layer: il budget diviso i layer sparsi. Il pavimento e' 1 e
 * non topk, perche' un pavimento a topk impegnerebbe topk*layer slot comunque,
 * cioe' molti GB, a dispetto del budget chiesto. */
/* Quanta RAM il sistema dice di poter dare adesso. MemAvailable e non MemFree:
 * la seconda ignora la page cache riutilizzabile e farebbe stimare molto meno
 * di quello che c'e'. */
static double memory_total_gb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    double gb = 0.0;
    while (fgets(line, sizeof(line), f)) {
        long kb;
        if (sscanf(line, "MemTotal: %ld kB", &kb) == 1) { gb = kb / 1048576.0; break; }
    }
    fclose(f);
    return gb;
}

static double memory_available_gb(void) {
    /* #1375: era una lettura di /proc/meminfo, che su Windows e macOS non
     * esiste: 0 -> budget 1 GB -> uno slot per layer, in silenzio. */
    return compat_mem_available_gb();
}

/* Un pezzo e' servibile dalla mappatura se st.h riesce a mapparne il file,
 * l'intervallo ci sta dentro (lo verifica st_map_shard_range) ed e' allineato
 * a 4 byte: le scale si rileggono come float *, e st.h non lo sa. */
static const uint8_t *piece_mapped(const GModel *m, const ERef *ref, int q) {
    if (ref->fd[q] <= 0 || (ref->off[q] & 3)) return NULL;
    return (const uint8_t *)st_map_shard_range(ref->fd[q], ref->off[q], m->e_len[q]);
}

/* Port of qwen38_core.h's q38_populate_range: same MADV_POPULATE_READ
 * primitive, same call shape (prefault on the thread that binds an expert
 * instead of inside the matmul that first touches it), but the opposite
 * default. Ported and A/B'd on the 2026-09-04 rig against G0
 * (tools/hot-expert/ROME-3x7900XTX-2026-09-04.md): where this helped Qwen's
 * CPU expert path 4-6x (fault stalls were 45% of decode as libgomp barrier
 * spin, fifteen threads idling on the one taking the fault), it measured a
 * REGRESSION on GLM's -- rotating median 1.65-1.66 -> 1.48 tok/s, cold
 * decode 1.23-1.25 -> 0.98-0.99, reproduced twice each way, isolated by
 * toggling only this knob with everything else held fixed. teacher_forcing
 * and last_logits are bit-identical to pristine either way (this changes
 * only when pages fault, never what gets read), so the difference is real
 * runtime cost, not a shortcut. Root cause not profiled here (out of scope
 * for a port) -- candidates are GLM's much higher call frequency (this fires
 * on every expert_read cache miss inside the per-token dispatch loop, not a
 * bounded one-time loader like Qwen's) and/or mmap_lock contention from
 * eight threads calling madvise concurrently on the SAME per-file mapping.
 * Default OFF (opt-in only) until that regression is understood; do not flip
 * this to default-on without a new measurement. */
/* G1b instrumentation: how many advises, how long inside them, how many bytes.
 * Reported by prof_print. Wall time here is what tells apart "the madvise calls
 * themselves cost the regression" from "they cost little but slow everything
 * else down" -- the two hypotheses G1 left open. */
static double g_t_pop; static long g_n_pop; static double g_b_pop;
static double g_t_pop_max;
static long g_n_bind_contig, g_n_bind_split;   /* mmap binds: 1 advise vs 6 */
static long g_n_bind_gpu, g_n_bind_cpu;        /* binds whose expert is/isn't VRAM-resident */

/* Is this expert served from VRAM? Defined with the tier below (it needs
 * vk_reg_at); declared here because expert_read has to ask before it decides
 * whether prefaulting the expert's host pages is worth anything. */
#ifdef COLI_VULKAN
static int glm53_expert_on_gpu(int layer, int eid);
#else
#define glm53_expert_on_gpu(layer, eid) 0
#endif
static void glm53_populate_range(const uint8_t *p, int64_t nbytes) {
#if defined(__linux__) && defined(MADV_POPULATE_READ)
    static int on = -1;
    if (on < 0) on = (getenv("GLM53_MMAP_POPULATE") && atoi(getenv("GLM53_MMAP_POPULATE")));
    if (!on || !p || nbytes <= 0) return;
    static long page = 0;
    if (!page) page = sysconf(_SC_PAGESIZE);
    uintptr_t a = (uintptr_t)p & ~((uintptr_t)page - 1);
    uintptr_t e = ((uintptr_t)p + (uintptr_t)nbytes + (uintptr_t)page - 1) & ~((uintptr_t)page - 1);
    struct timespec _t0, _t1;
    clock_gettime(CLOCK_MONOTONIC, &_t0);
    (void)madvise((void *)a, (size_t)(e - a), MADV_POPULATE_READ);
    clock_gettime(CLOCK_MONOTONIC, &_t1);
    double _dt = (double)(_t1.tv_sec - _t0.tv_sec) + (double)(_t1.tv_nsec - _t0.tv_nsec) / 1e9;
#ifdef _OPENMP
#pragma omp atomic
#endif
    g_t_pop += _dt;
#ifdef _OPENMP
#pragma omp atomic
#endif
    g_n_pop++;
#ifdef _OPENMP
#pragma omp atomic
#endif
    g_b_pop += (double)(e - a);
    if (_dt > g_t_pop_max) g_t_pop_max = _dt;   /* racy, indicative only */
#else
    (void)p; (void)nbytes;
#endif
}

static void expert_map_init(GModel *m) {
    if (getenv("GLM53_NO_MMAP") || !m->eref) return;
    const Cfg *c = &m->c;
    /* Le mappature si forzano QUI, su un thread solo. st.h mappa pigramente al
     * primo uso e la sua tabella per-fd non e' sincronizzata; expert_read gira
     * dentro a una regione OpenMP, quindi il primo tocco non puo' stare li'. */
    static unsigned char seen[ST_MAX_MAPPED_FD];
    int files = 0;
    for (int i = 0; i < c->n_layers; i++)
        for (int e = 0; e < c->n_experts; e++) {
            const ERef *ref = &m->eref[(size_t)i * c->n_experts + e];
            if (ref->fd[0] <= 0) continue;
            for (int q = 0; q < GLM53_EXPERT_PIECES; q++) {
                const int fd = ref->fd[q];
                if (fd <= 0 || fd >= ST_MAX_MAPPED_FD || seen[fd]) continue;
                seen[fd] = 1;
                struct stat st;
                if (fstat(fd, &st) != 0 || st.st_size <= 0) continue;
                const void *base = st_map_shard_range(fd, 0, (int64_t)st.st_size);
                if (!base) continue;
                files++;
                /* GLM53_POPULATE_LOAD: fault the whole shard in right here,
                 * instead of leaving it to the first per-expert bind. Default
                 * OFF, unlike the per-bind prefault above -- this is new and
                 * unmeasured on this box, and the record already has one
                 * incident of an eager, unbounded preload thrashing the page
                 * cache (91 GB "resident" from an uncapped dev2/dev3 loop).
                 * The 182 GiB checkpoint fits the 247 GB box alone, per the
                 * "one model at a time" discipline in CLAUDE.md, but that is
                 * exactly the assumption a bad interaction could violate, so
                 * this stays opt-in until it has its own A/B on record. */
                if (getenv("GLM53_POPULATE_LOAD") && atoi(getenv("GLM53_POPULATE_LOAD")))
                    glm53_populate_range((const uint8_t *)base, (int64_t)st.st_size);
            }
        }
    if (!files) return;
    g_map_active = 1;
    long full = 0, partial = 0;
    for (int i = 0; i < c->n_layers; i++)
        for (int e = 0; e < c->n_experts; e++) {
            const ERef *ref = &m->eref[(size_t)i * c->n_experts + e];
            if (ref->fd[0] <= 0) continue;
            int ok = 1;
            for (int q = 0; q < GLM53_EXPERT_PIECES; q++)
                if (!piece_mapped(m, ref, q)) { ok = 0; break; }
            if (ok) full++; else partial++;
        }
    g_map_all = (partial == 0 && full > 0);
    fprintf(stderr, "[MAP] %d file mappati (st_map_shard_range), esperti mappabili %ld/%ld%s\n",
            files, full, full + partial, g_map_all ? " (tutti: nessuna copia)" : "");
}

static void expert_cache_init(GModel *m) {
    const Cfg *c = &m->c;
    if (!g_map_active) expert_map_init(m);
    const char *setting = getenv("GLM53_EXPERT_GB");
    /* Il default si misura: quello che resta libero dopo i pesi residenti,
     * meno un margine per lo stato della conversazione, i temporanei del
     * prefill e il resto del sistema. Un numero fisso sbaglia in entrambi i
     * versi -- su una macchina piccola va in OOM, su una grande lascia RAM
     * inutilizzata mentre il disco fa tutto il lavoro, che e' esattamente
     * quello che e' successo alla prima esecuzione vera. */
    const int from = c->first_dense > m->layer_begin ? c->first_dense : m->layer_begin;
    int sparse = m->layer_end - from;
    if (sparse < 0) sparse = 0;
    double budget;
    if (setting) budget = atof(setting);
    else {
        /* MemAvailable conta la page cache come "libera". Fidarsi di quel
         * numero significa allocare in anonimo tutta la RAM e sfrattare
         * proprio le pagine del modello che poi si rileggono dal disco: le
         * due cache tengono gli stessi byte. Si lascia quindi al modello lo
         * spazio per restare in page cache e si prende solo l'avanzo. */
        const double free_now = memory_available_gb();
        const double total = memory_total_gb();
        /* gli esperti instradati dominano; il resto dei pesi e' un ~7% */
        const double model_gb = (double)sparse * c->n_experts * (double)m->e_slot / 1e9 * 1.07;
        double margin = total * 0.08;
        if (margin < 4.0) margin = 4.0;
        if (total <= 0.0 || model_gb >= total - margin) {
            /* il modello non ci sta comunque: la page cache non puo' aiutare
             * e tanto vale tenere piu' slot possibile, come prima. */
            budget = free_now - 3.0;
        } else {
            budget = total - model_gb - margin;
            if (budget > free_now - 3.0) budget = free_now - 3.0;
        }
        if (budget < 1.0) budget = 1.0;
        if (getenv("GLM53_VERBOSE"))
            fprintf(stderr, "budget esperti: %.1f GB (%.1f totali, %.1f modello, "
                            "%.1f margine, %.1f disponibili)\n",
                    budget, total, model_gb, margin, free_now);
    }
    int cap = (int)((budget * 1e9) / ((double)m->e_slot * (sparse > 0 ? sparse : 1)));
    /* Se ogni esperto arriva dalla mappatura, uno slot non costa memoria
     * nostra: tenerne uno per esperto toglie di mezzo lo sfratto. */
    if (g_map_all) cap = c->n_experts;
    if (g_cap_override > 0) cap = g_cap_override;      /* scelta esplicita: vince */
    if (cap < 1) cap = 1;
    if (cap > c->n_experts) cap = c->n_experts;

    m->ecache = calloc((size_t)c->n_layers, sizeof(*m->ecache));
    if (!m->ecache) { fprintf(stderr, "OOM sulla cache degli esperti\n"); exit(1); }
    for (int i = from; i < m->layer_end; i++) {
        LCache *cache = &m->ecache[i];
        cache->cap = cap;
        cache->s = calloc((size_t)cap, sizeof(*cache->s));
        if (!cache->s) { fprintf(stderr, "OOM sugli slot del layer %d\n", i); exit(1); }
        for (int j = 0; j < cap; j++) cache->s[j].eid = -1;
    }
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "esperti: slot da %.1f MB, %d per layer su %d layer sparsi "
                        "(%.1f GB residenti)\n",
                m->e_slot / 1e6, cap, sparse, (double)cap * sparse * m->e_slot / 1e9);
}

static Slot *slot_find(GModel *m, int layer, int eid) {
    LCache *cache = &m->ecache[layer];
    for (int j = 0; j < cache->n; j++)
        if (cache->s[j].eid == eid) {
            cache->s[j].used = ++m->clock;
            m->hits++;
            return &cache->s[j];
        }
    return NULL;
}

static void expert_read(GModel *m, int layer, int eid, Slot *slot) {
    const ERef *ref = &m->eref[(size_t)layer * m->c.n_experts + eid];
    if (g_map_active) {
        const uint8_t *pc[GLM53_EXPERT_PIECES];
        int ok = 1;
        for (int p = 0; p < GLM53_EXPERT_PIECES; p++)
            if (!(pc[p] = piece_mapped(m, ref, p))) { ok = 0; break; }
        if (ok) {
            for (int p = 0; p < GLM53_EXPERT_PIECES; p++) slot->piece[p] = (uint8_t *)pc[p];
            /* Prefault on the thread that binds this expert, not inside the
             * matmul that first touches it -- see glm53_populate_range above.
             *
             * But ONLY if the CPU is the one that will read it. An expert that
             * is VRAM-resident is still bound here (expert_mats builds its
             * views unconditionally) and then served from its GPU tensors, so
             * its host pages are never touched at all: faulting the whole
             * ~14 MB slot in for it is pure waste. That waste is what G1
             * measured as a regression -- see the record.
             *
             * ref->contig decides one advice for the whole slot vs six small
             * ones; measured contig=0 on this checkpoint, so in practice it is
             * always the six. */
            const int on_gpu = glm53_expert_on_gpu(layer, eid);
#ifdef _OPENMP
#pragma omp atomic
#endif
            g_n_bind_gpu += on_gpu ? 1 : 0;
#ifdef _OPENMP
#pragma omp atomic
#endif
            g_n_bind_cpu += on_gpu ? 0 : 1;
            if (!on_gpu) {
                if (ref->contig) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                    g_n_bind_contig++;
                    glm53_populate_range(slot->piece[0], m->e_slot);
                } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                    g_n_bind_split++;
                    for (int p = 0; p < GLM53_EXPERT_PIECES; p++)
                        glm53_populate_range(slot->piece[p], m->e_len[p]);
                }
            }
            slot->eid = eid;
#ifdef _OPENMP
#pragma omp atomic
#endif
            g_map_serve++;
            return;
        }
    }
    /* Fallback: si torna a scrivere in memoria NOSTRA, non nella mappatura di
     * sola lettura che questo slot poteva star usando prima. */
    if (!slot->own) {
        slot->own = malloc((size_t)m->e_slot);
        if (!slot->own) {
            fprintf(stderr, "OOM su uno slot esperto (%.1f MB): la cache esperti non ci sta "
                            "in memoria; riduci con --ram N o GLM53_EXPERT_GB=N (#1375)\n",
                    m->e_slot / 1e6);
            exit(1);
        }
    }
    for (int p = 0; p < GLM53_EXPERT_PIECES; p++) slot->piece[p] = slot->own + m->e_at[p];
#ifdef _OPENMP
#pragma omp atomic
#endif
    g_map_copy++;
    if (ref->contig) {
        st_pread_full(ref->fd[0], slot->own, m->e_slot, ref->off[0], "expert");
    } else {
        for (int p = 0; p < GLM53_EXPERT_PIECES; p++)
            st_pread_full(ref->fd[p], slot->own + m->e_at[p], m->e_len[p],
                          ref->off[p], "expert piece");
    }
    slot->eid = eid;
    /* expert_read gira dentro a un ciclo parallelo: i contatori sono condivisi
     * e senza questo sarebbero una corsa, cioe' numeri sbagliati proprio nel
     * posto in cui si va a guardare per capire se il riuso funziona. */
#ifdef _OPENMP
#pragma omp atomic
#endif
    m->miss++;
#ifdef _OPENMP
#pragma omp atomic
#endif
    m->ebytes += (uint64_t)m->e_slot;
}

/* Lo slot dell'esperto chiesto, letto se non c'e'. La vittima e' quella usata
 * meno di recente. */
static double now_s(void);
static void ehit_mark(GModel *m, int layer, int eid);
static void hits_emit(GModel *m);
static void emap_emit(GModel *m);
static Slot *expert_slot(GModel *m, int layer, int eid) {
    ehit_mark(m, layer, eid);
    Slot *slot = slot_find(m, layer, eid);
    if (slot) return slot;
    LCache *cache = &m->ecache[layer];
    if (cache->n < cache->cap) slot = &cache->s[cache->n++];
    else {
        int lru = 0;
        for (int j = 1; j < cache->n; j++)
            if (cache->s[j].used < cache->s[lru].used) lru = j;
        slot = &cache->s[lru];
    }
    double t_read0 = now_s();
    expert_read(m, layer, eid, slot);
    m->t_disk += now_s() - t_read0;
    slot->used = ++m->clock;
    return slot;
}

/* Le tre matrici di un esperto, che puntano dentro al suo slot. */
static void expert_mats(const GModel *m, const Slot *slot, Mat *gate, Mat *up, Mat *down) {
    const int hidden = m->c.hidden, inter = m->c.moe_inter;
    const Mat shape[3] = {
        { 4, NULL, NULL, slot->piece[0], (const float *)slot->piece[1],
          inter, hidden, 64 },
        { 4, NULL, NULL, slot->piece[2], (const float *)slot->piece[3],
          inter, hidden, 64 },
        { 4, NULL, NULL, slot->piece[4], (const float *)slot->piece[5],
          hidden, inter, 64 },
    };
    *gate = shape[0]; *up = shape[1]; *down = shape[2];
}

/* Il MoE, in due tempi.
 *
 * Prima si decide: per ogni token del blocco quali esperti servono e con che
 * peso. Poi si legge: l'UNIONE di quegli esperti, in parallelo, perche' una
 * lettura da 14,2 MB e' tempo in cui il disco lavora e la CPU no, e farne una
 * per volta e' stato il collo della prima esecuzione vera (2976 letture in
 * fila). Poi si calcola.
 *
 * Fare l'unione paga due volte: le letture vanno insieme, e un esperto che
 * serve a piu' token del blocco si legge una volta sola. */
#ifdef COLI_VULKAN
static void **g_vkreg; static uint64_t *g_eusage;
static int g_vk_E, g_vk_NL, g_vk_budget, g_vk_n;
static double g_t_eg, g_t_cpu; static long g_n_eg, g_n_eg_disp, g_n_cpu, g_n_devloss;
static int g_vk_budget2 = 0, g_vk_reg_n2 = 0;
static int g_vk_budget3 = 0, g_vk_reg_n3 = 0;
static double prof_now_s(void) { struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts); return (double)_ts.tv_sec + (double)_ts.tv_nsec / 1e9; }
__attribute__((destructor)) static void prof_print(void) {
    fprintf(stderr, "[PROF] eg=%.3fs(disp=%ld experts=%ld) cpu=%.3fs(n=%ld) devloss=%ld\n",
            g_t_eg, g_n_eg_disp, g_n_eg, g_t_cpu, g_n_cpu, g_n_devloss);
    fprintf(stderr, "[PROF] mmap serve=%ld copy=%ld | binds gpu=%ld cpu=%ld (contig=%ld split=%ld)\n",
            g_map_serve, g_map_copy, g_n_bind_gpu, g_n_bind_cpu,
            g_n_bind_contig, g_n_bind_split);
    if (g_n_pop)
        fprintf(stderr, "[PROF] populate n=%ld t=%.3fs (%.1f us/call, max %.1f us) bytes=%.2f GB\n",
                g_n_pop, g_t_pop, g_t_pop / (double)g_n_pop * 1e6,
                g_t_pop_max * 1e6, g_b_pop / 1e9);
    const char *p2 = getenv("COLI_USAGE_PATH");
    if (p2 && g_eusage) { FILE *f = fopen(p2, "wb"); if (f) { fwrite(g_eusage, sizeof(uint64_t), (size_t)g_vk_NL * g_vk_E, f); fclose(f); } }
}
static void **vk_reg_at(int layer, int eid) { return (void **)g_vkreg + ((size_t)layer * g_vk_E + eid) * 3; }
static int glm53_expert_on_gpu(int layer, int eid) {
    return g_vk_ready && g_vkreg && vk_reg_at(layer, eid)[0] != NULL;
}
typedef struct { uint64_t u; int layer, eid; } VkCand;
static int vk_cand_cmp(const void *a, const void *b) {
    uint64_t ua = ((const VkCand *)a)->u, ub = ((const VkCand *)b)->u;
    return ua < ub ? 1 : ua > ub ? -1 : 0;
}
/* heat-ranked preload from the (fixed) usage histogram loaded at startup */
/* Il denso prima del tier. Attenzione e shared expert servono a OGNI token,
 * un esperto instradato a circa un token su quaranta: lasciare che il tier si
 * prenda la VRAM per primo manda le matrici dense a rimbalzare sulla CPU e
 * costa molto piu' di quanto rendano gli esperti residenti in piu'. */
static void vk_warm_dense(GModel *m) {
    if (!m->layer) return;
    long warmed = 0, skipped = 0;
    double bytes = 0.0;
    for (int i = m->layer_begin; i < m->layer_end; i++) {
        GLayer *l = &m->layer[i];
        Mat *set[] = {
            &l->kq, &l->kk, &l->kv, &l->ko, &l->kga, &l->kgb, &l->kfa, &l->kfb, &l->kb,
            &l->qa, &l->qb, &l->kva, &l->kvb_kt, &l->kvb_v, &l->o,
            &l->iwq, &l->iwk, &l->iwp, &l->ikpg,
            &l->dg, &l->du, &l->dd, &l->rg, &l->ru, &l->rd,
        };
        for (size_t q = 0; q < sizeof(set) / sizeof(set[0]); q++) {
            Mat *w = set[q];
            if (w->fmt != 1 && w->fmt != 4) continue;
            const void *src = (w->fmt == 4) ? (const void *)w->q4 : (const void *)w->q8;
            if (!src || !w->s || w->rows < 1 || w->columns < 1) continue;
            if (coli_vk_tensor_ensure((ColiVkTensor **)&w->vk, src, w->s, w->fmt,
                                      w->columns, w->rows, w->gs)) {
                warmed++;
                bytes += (double)w->rows * (double)w->columns * 0.5;
            } else {
                skipped++;
            }
        }
    }
    fprintf(stderr, "[VK] denso residente: %ld matrici (~%.2f GB)%s\n",
            warmed, bytes / 1e9, skipped ? " (alcune non entrate)" : "");
}

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
    if (!getenv("COLI_VK_NO_WARM_DENSE")) vk_warm_dense(m);
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
    /* G6: how much VRAM to leave unused on the EXPERT-ONLY devices. dev0 keeps
     * 3.0 GB because it also holds the dense set (~3.78 GB), the KV mirror and
     * every scratch buffer; dev2/dev3 hold experts and nothing else, so their
     * only other consumers are command buffers, descriptor pools and a ~1 MB
     * expert-group scratch. 3.0 GB there is not caution, it is a re-tuning:
     * measured on this rig it stops the tier at 1600 of the 1695 every recorded
     * number was taken with, which would silently change routing and invalidate
     * the baseline. This guard exists to stop a RUNAWAY -- an unset cap
     * spilling into host RAM over ReBAR -- not to resize the tier. */
    double tier_reserve = 1.0;
    { const char *r = getenv("COLI_VK_TIER_RESERVE_GB");
      if (r) { double v = atof(r); if (v >= 0.0) tier_reserve = v; } }
    if (g_vk_budget2 > 0 && coli_vk_dev2_available()) {
        int loaded2 = 0;
        for (int i = 0; i < (int)n && (g_vk_budget2 <= 0 || loaded2 < g_vk_budget2); i++) {
            /* G6: stop on the VRAM budget, not only on the count cap. dev0's
             * loop has always done this; dev2/dev3 did not, and that is how an
             * unset cap put 91 GB "in VRAM" and evicted the page cache. These
             * are HOST_VISIBLE allocations that spill to host RAM over ReBAR,
             * so they do NOT fail at the VRAM limit -- without this check the
             * loop happily keeps succeeding into system memory. Same 3 GB
             * reserve and same every-8 cadence as dev0. */
            if ((loaded2 & 7) == 0) { double u, b;
                if (coli_vk_mem_budget2(&u, &b) && (b - u) < tier_reserve) {
                    fprintf(stderr, "[VK] preload dev2: stopping on VRAM budget "
                            "(%.1f of %.1f GB used, %.1f reserve)\n", u, b, tier_reserve);
                    break; } }
            int layer = cand[i].layer, eid = cand[i].eid;
            void **reg = vk_reg_at(layer, eid);
            if (reg[0]) continue;
            expert_read(m, layer, eid, &tmp);
            Mat gate, up, down; expert_mats(m, &tmp, &gate, &up, &down);
            ColiVkTensor *t[3] = {0,0,0};
            if (coli_vk_tensor_ensure2(&t[0], gate.q4, gate.s, gate.fmt, gate.columns, gate.rows, gate.gs) &&
                coli_vk_tensor_ensure2(&t[1], up.q4, up.s, up.fmt, up.columns, up.rows, up.gs) &&
                coli_vk_tensor_ensure2(&t[2], down.q4, down.s, down.fmt, down.columns, down.rows, down.gs)) {
                reg[0]=t[0]; reg[1]=t[1]; reg[2]=t[2]; loaded2++; g_vk_reg_n2++;
            } else { for (int q=0;q<3;q++) if (t[q]) coli_vk_tensor_free(t[q]); break; }
        }
        fprintf(stderr, "[VK] preload dev2: %d experts resident\n", loaded2);
    }
    if (g_vk_budget3 > 0 && coli_vk_dev3_available()) {
        int loaded3 = 0;
        for (int i = 0; i < (int)n && (g_vk_budget3 <= 0 || loaded3 < g_vk_budget3); i++) {
            if ((loaded3 & 7) == 0) { double u, b;            /* G6, as dev2 above */
                if (coli_vk_mem_budget3(&u, &b) && (b - u) < tier_reserve) {
                    fprintf(stderr, "[VK] preload dev3: stopping on VRAM budget "
                            "(%.1f of %.1f GB used, %.1f reserve)\n", u, b, tier_reserve);
                    break; } }
            int layer = cand[i].layer, eid = cand[i].eid;
            void **reg = vk_reg_at(layer, eid);
            if (reg[0]) continue;
            expert_read(m, layer, eid, &tmp);
            Mat gate, up, down; expert_mats(m, &tmp, &gate, &up, &down);
            ColiVkTensor *t[3] = {0,0,0};
            if (coli_vk_tensor_ensure3(&t[0], gate.q4, gate.s, gate.fmt, gate.columns, gate.rows, gate.gs) &&
                coli_vk_tensor_ensure3(&t[1], up.q4, up.s, up.fmt, up.columns, up.rows, up.gs) &&
                coli_vk_tensor_ensure3(&t[2], down.q4, down.s, down.fmt, down.columns, down.rows, down.gs)) {
                reg[0]=t[0]; reg[1]=t[1]; reg[2]=t[2]; loaded3++; g_vk_reg_n3++;
            } else { for (int q=0;q<3;q++) if (t[q]) coli_vk_tensor_free(t[q]); break; }
        }
        fprintf(stderr, "[VK] preload dev3: %d experts resident\n", loaded3);
    }
    free(cand); if (tmp.own) free(tmp.own);
    fprintf(stderr, "[VK] preload: %d heat-ranked experts resident (of %d candidates)\n", loaded, (int)n);
}
/* pure-CPU expert MLP (bypasses mv/Vulkan for non-resident experts)
 *
 * G11: the three matmuls used to be three separate `#pragma omp parallel for`
 * regions with the swiglu running SERIAL between the second and the third —
 * three fork/joins and one single-threaded stretch per expert, ~70 experts per
 * decode token. They are now one parallel region with three worksharing
 * constructs: gate and up share one (they are independent and read the same
 * x), then the swiglu is shared across the same team, then down. Two implicit
 * barriers replace two fork/joins, and the swiglu stops being serial.
 *
 * Bit-identical by construction: every output element is still one
 * `coli_i4_row` over the same groups in the same order — only which thread
 * runs it, and how the team is entered, changed. Verified against the pristine
 * binary on both prompts, not assumed.
 *
 * GLM53_EXPERT_SPLIT=1 restores the old three-region path for A/B measurement;
 * it is not a numerics knob, both paths produce the same bits. */
static int g_expert_split = -1;
static int expert_split_on(void) {
    if (g_expert_split < 0)
        g_expert_split = getenv("GLM53_EXPERT_SPLIT") ? atoi(getenv("GLM53_EXPERT_SPLIT")) : 0;
    return g_expert_split;
}
/* G14: the one alternative int4 expert kernel that preserved the output.
 *
 * Its only numeric change is summation order -- four accumulators instead of
 * one -- because the bit-trick nibble decode is exactly (float)(n-8). Measured
 * in the engine against a pristine binary: [PROF] cpu 11.777 -> 11.215 s with
 * the expert count IDENTICAL at 19706 (identical text means identical routing,
 * so it is a properly controlled pair), last_logits relL2 3.0e-6 on the short
 * prompt, teacher_forcing exact over 1232 positions, greedy text identical over
 * 128 tokens. Worth ~1.3% of the token -- real, safe, and below the serving
 * gate's floor.
 *
 * Off by default because it is not BIT-identical, and §G10 set this track's bar
 * for shipping on: that item reverted a working optimisation rather than accept
 * a 1.9e-4 drift.
 *
 * Three quantised-activation variants (int8 maddubs at 1.32x, int16 madd_epi16
 * at 1.075x, and int8 gate/up with a float down) were built, measured and
 * REMOVED: all three changed the greedy text, because one outlier in a group of
 * 64 -- the thing swiglu_limit=10.0 exists to clamp -- crushes the other 63.
 * The numbers are in record §G14 and in commits a0af56e / be95eb5; the code is
 * not kept, because the record is where this project keeps evidence.
 *
 * The finding that mattered was not the kernel: isolated-to-in-engine
 * attenuation is 0.55-0.65x consistently, because the engine streams 985 MB of
 * COLD expert weights per token where the microbenchmark cycles 406 MB. In situ
 * this path is closer to memory-bound than §G3's isolated "27% of bandwidth"
 * suggests, so the lever is FEWER BYTES (int3 experts, fmt=5) not faster
 * arithmetic. */
static int g_i4_fast = -1;
static int i4_fast_on(void) {
    if (g_i4_fast < 0) g_i4_fast = getenv("GLM53_I4_FAST") ? atoi(getenv("GLM53_I4_FAST")) : 0;
    return g_i4_fast;
}

static void mlp3_cpu(float *out, const float *x, const Mat *g, const Mat *u, const Mat *d, float limit, float *sg, float *su) {
    if (expert_split_on()) {
        matmul_i4_grouped(sg, x, g->q4, g->s, 1, g->columns, g->rows, g->gs);
        matmul_i4_grouped(su, x, u->q4, u->s, 1, u->columns, u->rows, u->gs);
        swiglu_clamped(sg, su, g->rows, limit);
        matmul_i4_grouped(out, sg, d->q4, d->s, 1, d->columns, d->rows, d->gs);
        return;
    }
    /* fmt is checked by the caller's classify loop (int4 experts only), but be
     * explicit: anything else falls back rather than silently misreading. */
    if (g->fmt != 4 || u->fmt != 4 || d->fmt != 4) {
        matmul_i4_grouped(sg, x, g->q4, g->s, 1, g->columns, g->rows, g->gs);
        matmul_i4_grouped(su, x, u->q4, u->s, 1, u->columns, u->rows, u->gs);
        swiglu_clamped(sg, su, g->rows, limit);
        matmul_i4_grouped(out, sg, d->q4, d->s, 1, d->columns, d->rows, d->gs);
        return;
    }
    const int Ig = g->columns, Og = g->rows, Id = d->columns, Od = d->rows;
    /* G14: the fused float path below, with the row kernel swapped for the
     * four-accumulator bit-trick decode. Structure, swiglu and down projection
     * are otherwise identical -- the only change is summation order inside a
     * row. No scratch buffers: it reads x and sg directly, like coli_i4_row. */
    if (i4_fast_on() && g->gs >= 8 && d->gs >= 8 &&
        u->gs == g->gs && u->columns == Ig && u->rows == Og) {
        const int grb_ = (Ig + 1) / 2, gng_ = (Ig + g->gs - 1) / g->gs;
        const int drb_ = (Id + 1) / 2, dng_ = (Id + d->gs - 1) / d->gs;
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int z = 0; z < 2 * Og; z++) {
                if (z < Og)
                    sg[z] = coli_i4_row_f4(g->q4 + (int64_t)z * grb_,
                                           g->s + (int64_t)z * gng_, x, Ig, g->gs);
                else {
                    const int o = z - Og;
                    su[o] = coli_i4_row_f4(u->q4 + (int64_t)o * grb_,
                                           u->s + (int64_t)o * gng_, x, Ig, u->gs);
                }
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < Og; i++) {
                float gv = sg[i] > limit ? limit : sg[i];
                float uv = su[i] < -limit ? -limit : (su[i] > limit ? limit : su[i]);
                sg[i] = siluf_(gv) * uv;
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int o = 0; o < Od; o++)
                out[o] = coli_i4_row_f4(d->q4 + (int64_t)o * drb_,
                                        d->s + (int64_t)o * dng_, sg, Id, d->gs);
        }
        return;
    }
    const int grb = (Ig + 1) / 2, gng = (Ig + g->gs - 1) / g->gs;
    const int urb = (u->columns + 1) / 2, ung = (u->columns + u->gs - 1) / u->gs;
    const int drb = (Id + 1) / 2, dng = (Id + d->gs - 1) / d->gs;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int z = 0; z < 2 * Og; z++) {
            if (z < Og)
                sg[z] = coli_i4_row(g->q4 + (int64_t)z * grb, g->s + (int64_t)z * gng, x, Ig, g->gs);
            else {
                const int o = z - Og;
                su[o] = coli_i4_row(u->q4 + (int64_t)o * urb, u->s + (int64_t)o * ung, x, u->columns, u->gs);
            }
        }
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int i = 0; i < Og; i++) {
            float gv = sg[i] > limit ? limit : sg[i];
            float uv = su[i] < -limit ? -limit : (su[i] > limit ? limit : su[i]);
            sg[i] = siluf_(gv) * uv;
        }
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int o = 0; o < Od; o++)
            out[o] = coli_i4_row(d->q4 + (int64_t)o * drb, d->s + (int64_t)o * dng, sg, Id, d->gs);
    }
}

/* P3 (PREFILL-ROADMAP P3): one CPU expert for S rows at once. Same three
 * stages and the same OpenMP shape as mlp3_cpu's default path; each output
 * row is computed for all S activation rows, four at a time through
 * coli_i4_rows4 (weights decoded once per four rows), the remainder through
 * coli_i4_row. Row r of out is bit-identical to mlp3_cpu on row r of x.
 * Only the default path (fmt 4, no split, no I4_FAST) has a rows version;
 * the caller falls back to the per-row loop otherwise. sg/su: S * g->rows. */
static int mlp3_cpu_rows_ok(const Mat *g, const Mat *u, const Mat *d) {
    return !expert_split_on() && !i4_fast_on() &&
           g->fmt == 4 && u->fmt == 4 && d->fmt == 4 &&
           u->columns == g->columns && u->rows == g->rows && u->gs == g->gs;
}
static void mlp3_cpu_rows(float *out, const float *x, int S, const Mat *g, const Mat *u,
                          const Mat *d, float limit, float *sg, float *su) {
    const int Ig = g->columns, Og = g->rows, Id = d->columns, Od = d->rows;
    const int grb = (Ig + 1) / 2, gng = (Ig + g->gs - 1) / g->gs;
    const int drb = (Id + 1) / 2, dng = (Id + d->gs - 1) / d->gs;
    const int S4 = S & ~3;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int z = 0; z < 2 * Og; z++) {
            const int o = z < Og ? z : z - Og;
            const Mat *w = z < Og ? g : u;
            float *dst = z < Og ? sg : su;
            const uint8_t *wr = w->q4 + (int64_t)o * grb;
            const float *sr = w->s + (int64_t)o * gng;
            float o4[4];
            int r = 0;
            for (; r < S4; r += 4) {
                coli_i4_rows4(wr, sr, x + (int64_t)r * Ig, Ig, Ig, w->gs, o4);
                dst[(size_t)r * Og + o] = o4[0]; dst[(size_t)(r + 1) * Og + o] = o4[1];
                dst[(size_t)(r + 2) * Og + o] = o4[2]; dst[(size_t)(r + 3) * Og + o] = o4[3];
            }
            for (; r < S; r++)
                dst[(size_t)r * Og + o] = coli_i4_row(wr, sr, x + (int64_t)r * Ig, Ig, w->gs);
        }
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int i = 0; i < S * Og; i++) {
            float gv = sg[i] > limit ? limit : sg[i];
            float uv = su[i] < -limit ? -limit : (su[i] > limit ? limit : su[i]);
            sg[i] = siluf_(gv) * uv;
        }
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int o = 0; o < Od; o++) {
            const uint8_t *wr = d->q4 + (int64_t)o * drb;
            const float *sr = d->s + (int64_t)o * dng;
            float o4[4];
            int r = 0;
            for (; r < S4; r += 4) {
                coli_i4_rows4(wr, sr, sg + (int64_t)r * Id, Id, Id, d->gs, o4);
                out[(size_t)r * Od + o] = o4[0]; out[(size_t)(r + 1) * Od + o] = o4[1];
                out[(size_t)(r + 2) * Od + o] = o4[2]; out[(size_t)(r + 3) * Od + o] = o4[3];
            }
            for (; r < S; r++)
                out[(size_t)r * Od + o] = coli_i4_row(wr, sr, sg + (int64_t)r * Id, Id, d->gs);
        }
    }
}

/* Scratch for the rows path, allocated once per ffn_layer call when the
 * chunk has more than one row: gathered activations, per-row outputs and
 * the S x inter gate/up slabs. NULL members = rows path off. */
typedef struct { float *xr, *tr, *sgr, *sur; int cap; } CpuRows;

/* Run one CPU expert on every row of the chunk that chose it: gather those
 * rows, one mlp3_cpu_rows, scatter with the routing weights. Accumulation
 * into out is the same per-row `dst[d] += scale * tmp[d]` as the per-token
 * loop, in the same t order. Returns the number of rows served. */
static int cpu_expert_rows(const CpuRows *cr, int eid, const int *chosen, const float *weight,
                           int tokens, int topk, const float *x, int hidden,
                           const Mat *gate, const Mat *up, const Mat *down, float limit,
                           float *out) {
    int nr = 0;
    static _Thread_local int   *tidx = NULL; static _Thread_local float *tsc = NULL;
    static _Thread_local int tcap = 0;
    if (tcap < tokens) {
        tidx = realloc(tidx, (size_t)tokens * sizeof(int));
        tsc = realloc(tsc, (size_t)tokens * sizeof(float));
        if (!tidx || !tsc) { fprintf(stderr, "OOM nel MoE (righe)\n"); exit(1); }
        tcap = tokens;
    }
    for (int t = 0; t < tokens; t++) {
        float scale = 0.0f;
        for (int k = 0; k < topk; k++)
            if (chosen[(size_t)t * topk + k] == eid) { scale = weight[(size_t)t * topk + k]; break; }
        if (scale == 0.0f) continue;
        memcpy(cr->xr + (size_t)nr * hidden, x + (size_t)t * hidden, (size_t)hidden * sizeof(float));
        tidx[nr] = t; tsc[nr] = scale; nr++;
    }
    if (nr < 1) return 0;
    double _tc = prof_now_s();
    mlp3_cpu_rows(cr->tr, cr->xr, nr, gate, up, down, limit, cr->sgr, cr->sur);
    g_t_cpu += prof_now_s() - _tc; g_n_cpu += nr;
    for (int r = 0; r < nr; r++) {
        float *dst = out + (size_t)tidx[r] * hidden;
        const float *src = cr->tr + (size_t)r * hidden;
        const float scale = tsc[r];
        for (int d = 0; d < hidden; d++) dst[d] += scale * src[d];
    }
    return nr;
}

/* G9: run CPU-only experts that were deferred past the GPU issue point (see
 * ffn_layer's dispatch block), so their compute overlaps the fence wait
 * instead of finishing entirely before it -- G2 issued the three devices
 * concurrently with each other; this is the "CPU share in between" half of
 * the same idea, deliberately left for later there. Same per-token
 * scale/accumulate as the immediate path, just replayed against saved Mat
 * triples instead of the single (gate,up,down,eid) the classify loop had at
 * hand. Only ever called with experts collected while can_defer held (single
 * block, no eviction risk between classifying and running them). */
static void ffn_moe_run_deferred_cpu(const Mat *cpu_gate, const Mat *cpu_up, const Mat *cpu_down,
                                     const int *cpu_eid, int n_cpu_deferred,
                                     const int *chosen, const float *weight, int tokens, int topk,
                                     const float *x, int hidden, float limit,
                                     float *sg, float *su, float *tmp, float *out,
                                     const CpuRows *cr) {
    for (int j = 0; j < n_cpu_deferred; j++) {
        const int eid = cpu_eid[j];
        if (cr && cr->xr && mlp3_cpu_rows_ok(&cpu_gate[j], &cpu_up[j], &cpu_down[j])) {
            cpu_expert_rows(cr, eid, chosen, weight, tokens, topk, x, hidden,
                            &cpu_gate[j], &cpu_up[j], &cpu_down[j], limit, out);
            continue;
        }
        for (int t = 0; t < tokens; t++) {
            float scale = 0.0f;
            for (int k = 0; k < topk; k++)
                if (chosen[(size_t)t * topk + k] == eid) { scale = weight[(size_t)t * topk + k]; break; }
            if (scale == 0.0f) continue;
            double _tc = prof_now_s();
            mlp3_cpu(tmp, x + (size_t)t * hidden, &cpu_gate[j], &cpu_up[j], &cpu_down[j], limit, sg, su);
            g_t_cpu += prof_now_s() - _tc; g_n_cpu++;
            float *dst = out + (size_t)t * hidden;
            for (int d = 0; d < hidden; d++) dst[d] += scale * tmp[d];
        }
    }
}
#endif

static void ffn_layer(GModel *m, const GLayer *l, int index, const float *x,
                      int tokens, float *out) {
    const Cfg *c = &m->c;
    const int wide = c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter;

    if (index < c->first_dense) {                 /* layer denso: nessun router */
        float *sg = malloc((size_t)wide * sizeof(float));
        float *su = malloc((size_t)wide * sizeof(float));
        for (int t = 0; t < tokens; t++)
            mlp3(out + (size_t)t * c->hidden, x + (size_t)t * c->hidden,
                 &l->dg, &l->du, &l->dd, c->swiglu_limit, sg, su);
        free(su); free(sg);
        return;
    }

    const int topk = c->topk;
    int *chosen = malloc((size_t)tokens * topk * sizeof(int));
    float *weight = malloc((size_t)tokens * topk * sizeof(float));
    float *score = malloc((size_t)c->n_experts * sizeof(float));
    if (!chosen || !weight || !score) { fprintf(stderr, "OOM nel router\n"); exit(1); }

    const double t_router0 = optime_on() ? optime_now() : 0.0;
    /* --- primo tempo: il router, per ogni token --- */
    /* P2.3: with a chunk of rows, one team over t (each thread its own score
     * slice) instead of one team per token; the dot over d keeps its scalar
     * order per (t, e), so every score is bit-identical. Decode (tokens == 1)
     * keeps the e-parallel loop that G3 measured. */
    const int rt_par = tokens >= 8 && !g_prefill_unbatched();
    float *score_all = rt_par ? malloc((size_t)tokens * c->n_experts * sizeof(float)) : NULL;
    if (rt_par && !score_all) { fprintf(stderr, "OOM nel router\n"); exit(1); }
    /* P5.3: the router dot is the same latency-bound scalar reduction P5.1
     * found in the MLA attention core -- 288 x 4 096 per token per MoE layer at
     * ~10 GMAC/s on eight threads. Here the free lane axis is the CHUNK's
     * TOKENS: transposing x to xT[d][t] once per layer (2 MB at S=128) lets one
     * router row be dotted against 64 tokens at a time, each lane summing over
     * d in the same ascending order with the product rounded first. So, unlike
     * what the roadmap assumed, this needs no reordering and is bit-identical;
     * COLI_ROUTER_LANES=0 restores the per-token scalar dot. */
    float *xT = NULL, *rdot = NULL;
    const int rt_lanes = rt_par && router_lanes_on();
    int rt_thr = 1;
    if (rt_lanes) {
#ifdef _OPENMP
        rt_thr = coli_kda_threads();
#endif
        xT = malloc((size_t)c->hidden * tokens * sizeof(float));
        rdot = malloc((size_t)rt_thr * tokens * sizeof(float));
        if (!xT || !rdot) { fprintf(stderr, "OOM nel router\n"); exit(1); }
        glm_transpose(xT, x, tokens, c->hidden);
    }
    if (rt_lanes) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int e = 0; e < c->n_experts; e++) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *dots = rdot + (size_t)tid * tokens;
            glm_lane_dots(dots, xT, l->router + (size_t)e * c->hidden,
                          tokens, c->hidden, 1.0f);
            for (int t = 0; t < tokens; t++)
                score_all[(size_t)t * c->n_experts + e] = sigmoidf_(dots[t]);
        }
    } else if (rt_par) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int t = 0; t < tokens; t++) {
            const float *row = x + (size_t)t * c->hidden;
            float *sc = score_all + (size_t)t * c->n_experts;
            for (int e = 0; e < c->n_experts; e++) {
                const float *w = l->router + (size_t)e * c->hidden;
                float sum = 0.0f;
                for (int d = 0; d < c->hidden; d++) sum += w[d] * row[d];
                sc[e] = sigmoidf_(sum);
            }
        }
    }
    for (int t = 0; t < tokens; t++) {
        const float *row = x + (size_t)t * c->hidden;
        if (rt_par) {
            memcpy(score, score_all + (size_t)t * c->n_experts, (size_t)c->n_experts * sizeof(float));
        } else {
        /* Le 288 righe sono indipendenti: ognuna legge la propria riga di
         * l->router e scrive il proprio score[e], nessun accumulatore
         * condiviso. Parallelizzare su e non tocca l'ordine della somma su d
         * per nessuna riga -- cambia solo in che ordine le righe vengono
         * calcolate, non il valore di nessuna di esse. `score` e' un unico
         * buffer riusato a ogni t (non ha una dimensione per token), quindi
         * qui dentro e non sul ciclo esterno: parallelizzare su t farebbe
         * scrivere thread diversi nello stesso buffer per t diversi. G3
         * (2026-09-04): 0.98 ms/call, un thread solo, riduzione scalare che
         * GCC non vettorizza da sé -- 1.2 GMAC/s contro gli 8 core
         * disponibili. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int e = 0; e < c->n_experts; e++) {
            const float *w = l->router + (size_t)e * c->hidden;
            float sum = 0.0f;
            for (int d = 0; d < c->hidden; d++) sum += w[d] * row[d];
            score[e] = sigmoidf_(sum);
        }
        }
        /* la selezione usa score+bias, il PESO usa lo score puro: la
         * distinzione e' sottile e sbagliarla cambia quali esperti contano
         * quanto. */
        int *mine = chosen + (size_t)t * topk;
        float *mine_w = weight + (size_t)t * topk;
        float total = 0.0f;
        for (int k = 0; k < topk; k++) {
            int best = -1; float value = -INFINITY;
            for (int e = 0; e < c->n_experts; e++) {
                int used = 0;
                for (int j = 0; j < k; j++) if (mine[j] == e) { used = 1; break; }
                float choice = score[e] + (l->rbias ? l->rbias[e] : 0.0f);
                if (!used && choice > value) { value = choice; best = e; }
            }
            mine[k] = best;
            mine_w[k] = score[best];
            total += mine_w[k];
        }
        for (int k = 0; k < topk; k++)
            mine_w[k] = mine_w[k] / (total + 1e-20f) * c->routed_scale;
    }
    free(score); free(score_all); free(xT); free(rdot);
    if (optime_on()) { g_ot_router += optime_now() - t_router0; g_on_router++; }

    /* --- secondo e terzo tempo, a blocchi che stanno in cache ---
     *
     * L'unione di un blocco di prefill puo' superare gli slot che il layer
     * possiede: con 13 slot e sette token si arriva a 56 esperti distinti. Se
     * si assegnassero comunque, due esperti finirebbero sullo stesso slot e il
     * secondo sovrascriverebbe il primo mentre il primo e' ancora in uso, il
     * che non da' errore, da' numeri sbagliati. Quindi si lavora a blocchi
     * grandi al piu' quanto la cache: si legge il blocco in parallelo, si
     * applica a tutti i token, si passa al prossimo. */
    /* P2.2: sg/su sized for the whole chunk so the shared expert runs once per
     * chunk; `wide` >= rg.rows, and the routed experts below still use only
     * the first `wide` floats. */
    const size_t shared_w = (size_t)l->rg.rows * (size_t)(tokens > 1 ? tokens : 1);
    float *sg = malloc((shared_w > (size_t)wide ? shared_w : (size_t)wide) * sizeof(float));
    float *su = malloc((shared_w > (size_t)wide ? shared_w : (size_t)wide) * sizeof(float));
    float *tmp = malloc((size_t)c->hidden * sizeof(float));
    if (!sg || !su || !tmp) { fprintf(stderr, "OOM nel MoE\n"); exit(1); }
    /* P3: rows scratch for the CPU experts of a chunk (tokens x hidden twice,
     * tokens x moe_inter twice); off for a single row or GLM53_PREFILL_UNBATCHED. */
#ifdef COLI_VULKAN
    CpuRows cpu_rows = { NULL, NULL, NULL, NULL, 0 };
    if (tokens > 1 && !g_prefill_unbatched()) {
        cpu_rows.xr = malloc((size_t)tokens * c->hidden * sizeof(float));
        cpu_rows.tr = malloc((size_t)tokens * c->hidden * sizeof(float));
        cpu_rows.sgr = malloc((size_t)tokens * c->moe_inter * sizeof(float));
        cpu_rows.sur = malloc((size_t)tokens * c->moe_inter * sizeof(float));
        cpu_rows.cap = tokens;
        if (!cpu_rows.xr || !cpu_rows.tr || !cpu_rows.sgr || !cpu_rows.sur) {
            fprintf(stderr, "OOM nel MoE (righe)\n"); exit(1);
        }
    }
#endif

    /* l'esperto condiviso e' sempre attivo e non passa dalla cache */
    const double t_shared0 = optime_on() ? optime_now() : 0.0;
    if (tokens > 1 && !g_prefill_unbatched())
        mlp3_shared_rows(out, x, tokens, l, c->swiglu_limit, sg, su);
    else
        for (int t = 0; t < tokens; t++)
            mlp3_shared(out + (size_t)t * c->hidden, x + (size_t)t * c->hidden, l,
                       c->swiglu_limit, sg, su);
    if (optime_on()) g_ot_shared += optime_now() - t_shared0;

    if (!m->streaming) {
        for (int t = 0; t < tokens; t++)
            for (int k = 0; k < topk; k++) {
                const int eid = chosen[(size_t)t * topk + k];
                mlp3(tmp, x + (size_t)t * c->hidden, &l->eg[eid], &l->eu[eid], &l->ed[eid],
                     c->swiglu_limit, sg, su);
                const float scale = weight[(size_t)t * topk + k];
                float *dst = out + (size_t)t * c->hidden;
                for (int d = 0; d < c->hidden; d++) dst[d] += scale * tmp[d];
            }
        #ifdef COLI_VULKAN
    free(cpu_rows.xr); free(cpu_rows.tr); free(cpu_rows.sgr); free(cpu_rows.sur);
#endif
    free(tmp); free(su); free(sg); free(weight); free(chosen);
        return;
    }

    /* unione dei distinti, nell'ordine in cui compaiono */
    int *union_ids = malloc((size_t)tokens * topk * sizeof(int));
    int n_union = 0;
    if (!union_ids) { fprintf(stderr, "OOM sull'unione\n"); exit(1); }
    for (int i = 0; i < tokens * topk; i++) {
        int seen = 0;
        for (int j = 0; j < n_union; j++) if (union_ids[j] == chosen[i]) { seen = 1; break; }
        if (!seen) union_ids[n_union++] = chosen[i];
    }

    LCache *cache = &m->ecache[index];
    const int block = cache->cap;
    int *slot_of = malloc((size_t)block * sizeof(int));
    int *to_read = malloc((size_t)block * sizeof(int));
    if (!slot_of || !to_read) { fprintf(stderr, "OOM sugli slot\n"); exit(1); }

#ifdef COLI_VULKAN
    const int maxres = tokens * topk;
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
    /* G9: CPU-only experts get deferred here instead of computed immediately,
     * so they can run after the GPU groups are issued (overlapping the fence
     * wait) instead of entirely before it. Safe only when the whole routing
     * fits in one block: n_union <= block means expert_read for a LATER
     * block can never evict a slot this block's deferred compute still
     * needs. cache->cap is 512 in every measured config on this rig and
     * n_union <= topk*tokens, so this always holds for decode; a smaller cap
     * or a larger prefill batch falls back to today's immediate-compute path
     * untouched, exactly as before this change. */
    const int can_defer = g_vk_ready && (n_union <= block);
    Mat *cpu_gate = NULL, *cpu_up = NULL, *cpu_down = NULL;
    int *cpu_eid = NULL, n_cpu_deferred = 0, cpu_deferred_done = 0;
    if (can_defer) {
        cpu_gate = malloc((size_t)maxres * sizeof(Mat));
        cpu_up   = malloc((size_t)maxres * sizeof(Mat));
        cpu_down = malloc((size_t)maxres * sizeof(Mat));
        cpu_eid  = malloc((size_t)maxres * sizeof(int));
    }
#endif
    for (int base = 0; base < n_union; base += block) {
        const int here = base + block <= n_union ? block : n_union - base;
        int reads = 0;
        for (int i = 0; i < here; i++) {
            const int eid = union_ids[base + i];
            ehit_mark(m, index, eid);
            Slot *hit = slot_find(m, index, eid);
            if (hit) { slot_of[i] = (int)(hit - cache->s); continue; }
            Slot *victim;
            if (cache->n < cache->cap) victim = &cache->s[cache->n++];
            else {
                int lru = 0;
                for (int j = 1; j < cache->n; j++)
                    if (cache->s[j].used < cache->s[lru].used) lru = j;
                victim = &cache->s[lru];
            }
            /* prenotato subito: cosi' la scelta successiva non lo ripesca */
            victim->used = ++m->clock;
            victim->eid = -1;
            slot_of[i] = (int)(victim - cache->s);
            to_read[reads++] = i;
        }
        double t_batch0;
        t_batch0 = now_s();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int r = 0; r < reads; r++) {
            const int i = to_read[r];
            expert_read(m, index, union_ids[base + i], &cache->s[slot_of[i]]);
        }
        m->t_disk += now_s() - t_batch0;  /* fuori dalla regione omp: e' il muro del batch */

        /* Un esperto per volta, e per ognuno tutti i token che lo hanno
         * scelto. Nell'ordine opposto i suoi 12,6 MB di pesi verrebbero
         * ripercorsi da capo per ogni token, e a questa taglia la banda di
         * memoria e' quanto costa davvero il calcolo. */
        for (int i = 0; i < here; i++) {
            const int eid = union_ids[base + i];
            Slot *slot = &cache->s[slot_of[i]];
            slot->used = ++m->clock;
            Mat gate, up, down;
            expert_mats(m, slot, &gate, &up, &down);
#ifdef COLI_VULKAN
                if (g_vk_ready && vk_reg_at(index, eid)[0] != NULL) {
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
                }
                if (can_defer) {
                    cpu_gate[n_cpu_deferred] = gate;
                    cpu_up[n_cpu_deferred] = up;
                    cpu_down[n_cpu_deferred] = down;
                    cpu_eid[n_cpu_deferred] = eid;
                    n_cpu_deferred++;
                    continue;
                }
#endif
#ifdef COLI_VULKAN
            if (g_vk_ready && cpu_rows.xr && mlp3_cpu_rows_ok(&gate, &up, &down)) {
                cpu_expert_rows(&cpu_rows, eid, chosen, weight, tokens, topk, x, c->hidden,
                                &gate, &up, &down, c->swiglu_limit, out);
                continue;
            }
#endif
            for (int t = 0; t < tokens; t++) {
                float scale = 0.0f;
                for (int k = 0; k < topk; k++)
                    if (chosen[(size_t)t * topk + k] == eid) {
                        scale = weight[(size_t)t * topk + k];
                        break;
                    }
                if (scale == 0.0f) continue;          /* non lo ha scelto */
#ifdef COLI_VULKAN
                if (g_vk_ready) {
                    double _tc = prof_now_s();
                    mlp3_cpu(tmp, x + (size_t)t * c->hidden, &gate, &up, &down, c->swiglu_limit, sg, su);
                    g_t_cpu += prof_now_s() - _tc; g_n_cpu++;
                } else
#endif
                mlp3(tmp, x + (size_t)t * c->hidden, &gate, &up, &down,
                     c->swiglu_limit, sg, su);
                float *dst = out + (size_t)t * c->hidden;
                for (int d = 0; d < c->hidden; d++) dst[d] += scale * tmp[d];
            }
        }
    }
#ifdef COLI_VULKAN
    if (g_vk_ready && (nvk0 > 0 || nvk1 > 0 || nvk2 > 0)) {
        double _te0 = prof_now_s();
        float *yk0 = (nvk0 > 0) ? malloc((size_t)vtot0 * c->hidden * sizeof(float)) : NULL;
        float *yk1 = (nvk1 > 0) ? malloc((size_t)vtot1 * c->hidden * sizeof(float)) : NULL;
        float *yk2 = (nvk2 > 0) ? malloc((size_t)vtot2 * c->hidden * sizeof(float)) : NULL;
        int ok = (nvk0 == 0 || yk0 != NULL) && (nvk1 == 0 || yk1 != NULL) && (nvk2 == 0 || yk2 != NULL);
        /* Issue every device's chunk first, join afterward: the three GPUs
         * then run at the same time as each other instead of one after the
         * other. Mirrors qwen38_core.h's q38_moe_decode -- measured there at
         * 2.37 ms/layer sequential vs 0.63 ms/layer concurrent issue/take on
         * this rig's 3x RX 7900 XTX (2026-09-04). Each device is still capped
         * at 64 items per call (the descriptor-pool size eg_prepare_submit
         * allocates once, in backend_vulkan.c), so a device whose group needs
         * more than one chunk still serializes against ITSELF across rounds;
         * only the three devices' matching round overlaps. A group that was
         * successfully issued is always taken before this loop can exit or
         * advance to the next round, win or lose, so a mid-round failure on
         * one device never leaves another device's single in-flight slot
         * (coli_vk_expert_group_take's precondition) stuck for the engine's
         * next call. */
        int q0 = 0, q1 = 0, q2 = 0, base0 = 0, base1 = 0, base2 = 0, fail = !ok;
        int first_round = 1;
        while (!fail && (q0 < nvk0 || q1 < nvk1 || q2 < nvk2)) {
            int n0 = 0, n1 = 0, n2 = 0, issued0 = 0, issued1 = 0, issued2 = 0;
            if (q0 < nvk0) {
                n0 = nvk0 - q0; if (n0 > 64) n0 = 64;
                issued0 = coli_vk_expert_group_issue(vg0 + q0, vu0 + q0, vd0 + q0, vrows0 + q0, n0,
                                                     xk0 + (size_t)base0 * c->hidden);
                if (!issued0) fail = 1;
            }
            if (q1 < nvk1) {
                n1 = nvk1 - q1; if (n1 > 64) n1 = 64;
                issued1 = coli_vk_expert_group_issue2(vg1 + q1, vu1 + q1, vd1 + q1, vrows1 + q1, n1,
                                                      xk1 + (size_t)base1 * c->hidden);
                if (!issued1) fail = 1;
            }
            if (q2 < nvk2) {
                n2 = nvk2 - q2; if (n2 > 64) n2 = 64;
                issued2 = coli_vk_expert_group_issue3(vg2 + q2, vu2 + q2, vd2 + q2, vrows2 + q2, n2,
                                                      xk2 + (size_t)base2 * c->hidden);
                if (!issued2) fail = 1;
            }
            /* G9: the three devices' fences are now signalled but not yet
             * waited on -- run the deferred CPU-only experts in this gap,
             * once, on whichever round first issues anything, so their
             * compute overlaps the take/fence-wait below instead of having
             * finished entirely beforehand. Unconditional on this round's
             * own success: an issue failure elsewhere doesn't change that
             * these still have to run, and a device that DID issue is still
             * in flight regardless. */
            if (first_round && can_defer && n_cpu_deferred > 0 && !cpu_deferred_done) {
                ffn_moe_run_deferred_cpu(cpu_gate, cpu_up, cpu_down, cpu_eid, n_cpu_deferred,
                                         chosen, weight, tokens, topk, x, c->hidden,
                                         c->swiglu_limit, sg, su, tmp, out, &cpu_rows);
                cpu_deferred_done = 1;
            }
            first_round = 0;
            if (issued0) {
                if (!coli_vk_expert_group_take(yk0 + (size_t)base0 * c->hidden)) fail = 1;
                int rs = 0; for (int w = 0; w < n0; w++) rs += vrows0[q0 + w];
                base0 += rs; q0 += n0;
            } else if (n0 > 0) q0 = nvk0;
            if (issued1) {
                if (!coli_vk_expert_group_take2(yk1 + (size_t)base1 * c->hidden)) fail = 1;
                int rs = 0; for (int w = 0; w < n1; w++) rs += vrows1[q1 + w];
                base1 += rs; q1 += n1;
            } else if (n1 > 0) q1 = nvk1;
            if (issued2) {
                if (!coli_vk_expert_group_take3(yk2 + (size_t)base2 * c->hidden)) fail = 1;
                int rs = 0; for (int w = 0; w < n2; w++) rs += vrows2[q2 + w];
                base2 += rs; q2 += n2;
            } else if (n2 > 0) q2 = nvk2;
        }
        ok = !fail;
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
    }
    if (can_defer && n_cpu_deferred > 0 && !cpu_deferred_done) {
        /* Nothing was GPU-resident this layer (nvk0=nvk1=nvk2=0, so the
         * dispatch block above was never entered and the while loop's
         * overlap hook never ran) -- no GPU work to overlap with, but these
         * still have to run. */
        ffn_moe_run_deferred_cpu(cpu_gate, cpu_up, cpu_down, cpu_eid, n_cpu_deferred,
                                 chosen, weight, tokens, topk, x, c->hidden,
                                 c->swiglu_limit, sg, su, tmp, out, &cpu_rows);
    }
    free(cpu_gate); free(cpu_up); free(cpu_down); free(cpu_eid);
#endif
    free(to_read); free(slot_of); free(union_ids);
    #ifdef COLI_VULKAN
    free(cpu_rows.xr); free(cpu_rows.tr); free(cpu_rows.sgr); free(cpu_rows.sur);
#endif
    free(tmp); free(su); free(sg); free(weight); free(chosen);
#ifdef COLI_VULKAN
    free(vg0); free(vu0); free(vd0); free(vrows0); free(vtok0); free(vw0); free(xk0);
    free(vg1); free(vu1); free(vd1); free(vrows1); free(vtok1); free(vw1); free(xk1);
    free(vg2); free(vu2); free(vd2); free(vrows2); free(vtok2); free(vw2); free(xk2);
#endif
}

/* ---------- caricamento ---------- */
static void vision_load(GModel *m);
static void model_load_range(GModel *m, const char *dir, int b, int e, int io);
static void expert_geometry(GModel *m);
static void expert_table_init(GModel *m);
static void expert_cache_init(GModel *m);

static void model_load_range(GModel *m, const char *dir, int layer_begin,
                             int layer_end, int load_io) {
    load_cfg(&m->c, dir);
    st_init(&m->S, dir);
    /* Il checkpoint reale annida il modello testuale sotto il wrapper vision;
     * un export solo-testo no. Si sceglie una volta, da un tensore che deve
     * esistere in entrambe le forme. */
    snprintf(m->prefix, sizeof(m->prefix), "model.language_model.");
    char probe[256];
    snprintf(probe, sizeof(probe), "%sembed_tokens.weight", m->prefix);
    if (!st_find(&m->S, probe)) snprintf(m->prefix, sizeof(m->prefix), "model.");
    const char *P = m->prefix;

    if (layer_end < 0 || layer_end > m->c.n_layers) layer_end = m->c.n_layers;
    if (layer_begin < 0) layer_begin = 0;
    if (layer_begin > layer_end) layer_begin = layer_end;
    m->layer_begin = layer_begin;
    m->layer_end = layer_end;
    m->has_io = load_io;

    if (load_io) {
        m->embed = load_f32(m, "%sembed_tokens.weight", P);
        m->final_norm = load_f32(m, "%snorm.weight", P);
    }
    /* La testa e' l'unica matrice grande fuori dagli esperti: a vocab 154880
     * per hidden 4096 sono 2,5 GB in f32, quindi passa dallo stesso
     * quantizzatore dei densi. Se il checkpoint la lega all'embedding, si
     * riusa quella tabella cosi' com'e'. Un segment che non tiene l'ultimo
     * layer non la carica proprio: non ha logit da produrre. */
    if (load_io) {
        if (st_find(&m->S, "lm_head.weight")) {
            m->head = load_mat(m, "lm_head.weight");
        } else {
            memset(&m->head, 0, sizeof(m->head));
            m->head.f = m->embed;
            m->head.rows = m->c.vocab;
            m->head.columns = m->c.hidden;
        }
    }
    m->layer = calloc((size_t)m->c.n_layers, sizeof(*m->layer));

    /* Streaming o residenti: lo decide il contenitore, non una variabile
     * d'ambiente. Si guarda il primo esperto del primo layer sparso. */
    int probe_layer = m->c.first_dense > layer_begin ? m->c.first_dense : layer_begin;
    if (probe_layer < layer_end) {
        char first[512];
        snprintf(first, sizeof(first), "%slayers.%d.mlp.experts.0.gate_proj.weight",
                 P, probe_layer);
        st_tensor *probe_expert = st_find(&m->S, first);
        if (!probe_expert) { fprintf(stderr, "manca %s\n", first); exit(1); }
        m->streaming = probe_expert->dtype == 3;
    }
    if (m->streaming) {
        expert_geometry(m);
        expert_table_init(m);
        expert_map_init(m);
    }

    for (int i = layer_begin; i < layer_end; i++) {
        GLayer *l = &m->layer[i];
        l->in_ln = load_f32(m, "%slayers.%d.input_layernorm.weight", P, i);
        l->post_ln = load_f32(m, "%slayers.%d.post_attention_layernorm.weight", P, i);
        l->hc_attn_fn = load_f32(m, "%slayers.%d.hc_attn_fn", P, i);
        l->hc_attn_base = load_f32(m, "%slayers.%d.hc_attn_base", P, i);
        l->hc_attn_scale = load_f32(m, "%slayers.%d.hc_attn_scale", P, i);
        l->hc_ffn_fn = load_f32(m, "%slayers.%d.hc_ffn_fn", P, i);
        l->hc_ffn_base = load_f32(m, "%slayers.%d.hc_ffn_base", P, i);
        l->hc_ffn_scale = load_f32(m, "%slayers.%d.hc_ffn_scale", P, i);
        if (m->c.is_full[i]) {
            l->qa = load_mat(m, "%slayers.%d.self_attn.q_a_proj.weight", P, i);
            l->qa_ln = load_f32(m, "%slayers.%d.self_attn.q_a_layernorm.weight", P, i);
            l->qb = load_mat(m, "%slayers.%d.self_attn.q_b_proj.weight", P, i);
            l->kva = load_mat(m, "%slayers.%d.self_attn.kv_a_proj_with_mqa.weight", P, i);
            l->kva_ln = load_f32(m, "%slayers.%d.self_attn.kv_a_layernorm.weight", P, i);
            {
                char kvb_name[512];
                snprintf(kvb_name, sizeof(kvb_name),
                         "%slayers.%d.self_attn.kv_b_proj.weight", P, i);
                absorb_kvb(m, l, kvb_name);
            }
            l->o = load_mat(m, "%slayers.%d.self_attn.o_proj.weight", P, i);
            l->iwq = load_mat(m, "%slayers.%d.self_attn.indexer.wq_b.weight", P, i);
            l->iwk = load_mat(m, "%slayers.%d.self_attn.indexer.wk.weight", P, i);
            l->iwp = load_mat(m, "%slayers.%d.self_attn.indexer.weights_proj.weight", P, i);
            l->ik_nw = load_f32(m, "%slayers.%d.self_attn.indexer.k_norm.weight", P, i);
            l->ik_nb = load_f32(m, "%slayers.%d.self_attn.indexer.k_norm.bias", P, i);
            if (m->c.index_kpool > 1) {
                l->ikpa = load_f32(m, "%slayers.%d.self_attn.indexer.index_kpool_compress_ape", P, i);
                l->ikpg = load_mat(m, "%slayers.%d.self_attn.indexer.index_kpool_compress_gate", P, i);
            }
        } else {
            l->kq = load_mat(m, "%slayers.%d.self_attn.q_proj.weight", P, i);
            l->kk = load_mat(m, "%slayers.%d.self_attn.k_proj.weight", P, i);
            l->kv = load_mat(m, "%slayers.%d.self_attn.v_proj.weight", P, i);
            l->ko = load_mat(m, "%slayers.%d.self_attn.o_proj.weight", P, i);
            l->kga = load_mat(m, "%slayers.%d.self_attn.g_a_proj.weight", P, i);
            l->kgb = load_mat(m, "%slayers.%d.self_attn.g_b_proj.weight", P, i);
            l->kfa = load_mat(m, "%slayers.%d.self_attn.f_a_proj.weight", P, i);
            l->kfb = load_mat(m, "%slayers.%d.self_attn.f_b_proj.weight", P, i);
            l->kb = load_mat(m, "%slayers.%d.self_attn.b_proj.weight", P, i);
            l->dt = load_f32(m, "%slayers.%d.self_attn.dt_bias", P, i);
            l->alog = load_f32(m, "%slayers.%d.self_attn.A_log", P, i);
            l->onorm = load_f32(m, "%slayers.%d.self_attn.o_norm.weight", P, i);
            /* Il checkpoint tiene q/k/v conv separate; la ricorrenza le vuole
             * concatenate nello stesso ordine di qkv. */
            {
                int width = m->c.kda_proj * m->c.conv_k;
                float *conv = malloc((size_t)3 * width * sizeof(float));
                const char *parts[3] = { "q_conv1d", "k_conv1d", "v_conv1d" };
                for (int p = 0; p < 3; p++) {
                    const float *piece = load_f32(m, "%slayers.%d.self_attn.%s.weight",
                                                  P, i, parts[p]);
                    memcpy(conv + (size_t)p * width, piece, (size_t)width * sizeof(float));
                    free((void *)piece);
                }
                l->conv = conv;
            }
        }
        if (i < m->c.first_dense) {
            l->dg = load_mat(m, "%slayers.%d.mlp.gate_proj.weight", P, i);
            l->du = load_mat(m, "%slayers.%d.mlp.up_proj.weight", P, i);
            l->dd = load_mat(m, "%slayers.%d.mlp.down_proj.weight", P, i);
        } else {
            l->router = load_f32(m, "%slayers.%d.mlp.gate.weight", P, i);
            l->rbias = st_find(&m->S, (snprintf(probe, sizeof(probe),
                        "%slayers.%d.mlp.gate.e_score_correction_bias", P, i), probe))
                       ? load_f32(m, "%s", probe) : NULL;
            l->rg = load_mat(m, "%slayers.%d.mlp.shared_experts.gate_proj.weight", P, i);
            l->ru = load_mat(m, "%slayers.%d.mlp.shared_experts.up_proj.weight", P, i);
            l->rd = load_mat(m, "%slayers.%d.mlp.shared_experts.down_proj.weight", P, i);
            /* Gli esperti quantizzati restano su disco: sono il 97% dei byte
             * e nessuna macchina li tiene in RAM. Un checkpoint f32, come le
             * fixture degli oracoli, e' piccolo e si carica tutto. */
            if (!m->streaming) {
                l->eg = malloc((size_t)m->c.n_experts * sizeof(Mat));
                l->eu = malloc((size_t)m->c.n_experts * sizeof(Mat));
                l->ed = malloc((size_t)m->c.n_experts * sizeof(Mat));
                for (int e = 0; e < m->c.n_experts; e++) {
                    l->eg[e] = load_mat(m, "%slayers.%d.mlp.experts.%d.gate_proj.weight", P, i, e);
                    l->eu[e] = load_mat(m, "%slayers.%d.mlp.experts.%d.up_proj.weight", P, i, e);
                    l->ed[e] = load_mat(m, "%slayers.%d.mlp.experts.%d.down_proj.weight", P, i, e);
                }
            }
        }
    }
    vision_load(m);
#ifdef COLI_VULKAN
    /* Il device si apre dopo i pesi: se non c'e', il motore continua sulla CPU
     * senza dire niente di piu' di una riga, perche' Vulkan qui e' un'opzione
     * e non un requisito. */
    if (getenv("COLI_VULKAN") && atoi(getenv("COLI_VULKAN"))) {
        /* Il backend vuole il file qmatmul.spv e da li' ricava i fratelli.
         * COLI_VK_SHADERS puo' essere il file o la cartella che lo contiene,
         * come nel resto del progetto; senza, si guarda accanto al binario. */
        char spv[1024];
        const char *given = getenv("COLI_VK_SHADERS");
        if (given && strstr(given, ".spv")) snprintf(spv, sizeof(spv), "%s", given);
        else snprintf(spv, sizeof(spv), "%s/qmatmul.spv", given ? given : "shaders");
        g_vk_ready = coli_vk_init(spv) && coli_vk_available();
        if (g_vk_ready) {
            g_vk_NL = m->c.n_layers; g_vk_E = m->c.n_experts;
            g_vk_budget = getenv("COLI_VK_EXPERTS") ? atoi(getenv("COLI_VK_EXPERTS")) : 0;
            g_vk_budget2 = getenv("COLI_VK_EXPERTS2") ? atoi(getenv("COLI_VK_EXPERTS2")) : 0;
            if (g_vk_budget2 > 0 && getenv("COLI_VK_DEV2")) { const char *dv = getenv("COLI_VK_DEV2"); int didx = (!strcmp(dv,"auto")||*dv==0) ? -1 : atoi(dv); if (coli_vk_init_dev2(spv, didx)) fprintf(stderr,"[VK] dev2 ready (expert tier)\n"); }
            g_vk_budget3 = getenv("COLI_VK_EXPERTS3") ? atoi(getenv("COLI_VK_EXPERTS3")) : 0;
            if (g_vk_budget3 > 0 && getenv("COLI_VK_DEV3")) { const char *dv = getenv("COLI_VK_DEV3"); int didx = (!strcmp(dv,"auto")||*dv==0) ? -1 : atoi(dv); if (coli_vk_init_dev3(spv, didx)) fprintf(stderr,"[VK] dev3 ready (expert tier)\n"); }
            g_vkreg = calloc((size_t)g_vk_NL * g_vk_E * 3, sizeof(void *));
            g_eusage = calloc((size_t)g_vk_NL * g_vk_E, sizeof(uint64_t));
            g_vk_n = 0;
            const char *up = getenv("COLI_USAGE_PATH");
            if (up) { FILE *f = fopen(up, "rb"); if (f) { size_t got = fread(g_eusage, sizeof(uint64_t), (size_t)g_vk_NL * g_vk_E, f); fclose(f); fprintf(stderr, "[VK] usage histogram loaded (%zu entries)\n", got); } }
            /* P6b: the per-slot KDA state pool goes up BEFORE the expert
             * preload. dev0 sits at 23.2 of 24 GB in service and the preload
             * stops when free VRAM drops under its 3.0 GB reserve; a pool
             * allocated afterwards would eat that reserve, which the KV mirror
             * and every scratch buffer need. Allocated here it costs experts
             * instead -- 625 MB = 44 of dev0's 1 296 at 4 slots -- and the
             * "preload: N heat-ranked experts" line below reports the price. */
            if (m->c.kda_proj && kda_gpu_on() && !g_kda_cpu_on()) {
                unsigned char *is_kda = calloc((size_t)m->c.n_layers, 1);
                int nk = 0;
                if (is_kda) {
                    for (int i = 0; i < m->c.n_layers; i++)
                        if (!m->c.is_full[i]) { is_kda[i] = 1; nk++; }
                    if (nk) coli_vk_kda_pool_init(kv_slots_wanted(), m->c.n_layers, is_kda,
                                                  m->c.kda_heads, m->c.kda_hd, m->c.kda_hd,
                                                  m->c.conv_k);
                    free(is_kda);
                }
            }
            vk_preload_tier(m);
        }
        fprintf(stderr, g_vk_ready
                ? "Vulkan: attivo sulle matrici residenti\n"
                : "Vulkan: nessun device utilizzabile (%s), resto su CPU\n", spv);
    }
#endif
    /* La cache si dimensiona qui, non prima: quanto si puo' spendere dipende
     * da quanto hanno gia' preso i pesi, e prima del ciclo sui layer non
     * l'avevano ancora preso. */
    if (m->streaming) expert_cache_init(m);
}

/* ---------- vision ----------
 * La torre e' in vision_tower.h e non sa nulla di GLM: qui si riempiono solo i
 * puntatori ai pesi e si traduce la config. Il checkpoint solo-testo non porta
 * `model.visual.*` e allora has_vision resta 0: l'engine funziona identico. */
static void vision_load(GModel *m) {
    const Cfg *c = &m->c;
    m->has_vision = 0;
    if (c->vis_layers <= 0) return;
    if (!st_find(&m->S, "model.visual.patch_embed.proj.weight")) return;
    const char *V = "model.visual.";

    m->vision.config = (ColiVisionConfig){
        .depth = c->vis_layers, .hidden = c->vis_hidden, .heads = c->vis_heads,
        .head_dim = c->vis_hidden / c->vis_heads, .intermediate = c->vis_inter,
        .patch = c->vis_patch, .temporal = c->vis_temporal, .merge = c->vis_merge,
        .in_channels = c->vis_in_ch, .out_hidden = c->vis_out_hidden,
        .proj_intermediate = c->vis_proj_inter ? c->vis_proj_inter : c->vis_inter,
        .eps = c->vis_eps, .swiglu_limit = c->vis_swiglu_limit,
        .rope_theta = 10000.0f,
    };
    m->vision.patch_w = load_f32(m, "%spatch_embed.proj.weight", V);
    m->vision.patch_b = load_f32(m, "%spatch_embed.proj.bias", V);
    m->vision.post_norm = load_f32(m, "%spost_layernorm.weight", V);
    m->vision.down_w = load_f32(m, "%sdownsample.weight", V);
    m->vision.down_b = load_f32(m, "%sdownsample.bias", V);
    m->vision.merger_proj = load_f32(m, "%smerger.proj.weight", V);
    m->vision.merger_norm_w = load_f32(m, "%smerger.post_projection_norm.weight", V);
    m->vision.merger_norm_b = load_f32(m, "%smerger.post_projection_norm.bias", V);
    m->vision.merger_gate = load_f32(m, "%smerger.gate_proj.weight", V);
    m->vision.merger_up = load_f32(m, "%smerger.up_proj.weight", V);
    m->vision.merger_down = load_f32(m, "%smerger.down_proj.weight", V);

    m->vblocks = calloc((size_t)c->vis_layers, sizeof(*m->vblocks));
    if (!m->vblocks) { fprintf(stderr, "OOM sui blocchi vision\n"); exit(1); }
    for (int b = 0; b < c->vis_layers; b++) {
        ColiVisionBlock *vb = &m->vblocks[b];
        vb->norm1 = load_f32(m, "%sblocks.%d.norm1.weight", V, b);
        vb->norm2 = load_f32(m, "%sblocks.%d.norm2.weight", V, b);
        vb->qkv_w = load_f32(m, "%sblocks.%d.attn.qkv.weight", V, b);
        vb->qkv_b = load_f32(m, "%sblocks.%d.attn.qkv.bias", V, b);
        vb->q_norm = load_f32(m, "%sblocks.%d.attn.q_norm.weight", V, b);
        vb->k_norm = load_f32(m, "%sblocks.%d.attn.k_norm.weight", V, b);
        vb->proj_w = load_f32(m, "%sblocks.%d.attn.proj.weight", V, b);
        vb->proj_b = load_f32(m, "%sblocks.%d.attn.proj.bias", V, b);
        vb->gate_w = load_f32(m, "%sblocks.%d.mlp.gate_proj.weight", V, b);
        vb->gate_b = load_f32(m, "%sblocks.%d.mlp.gate_proj.bias", V, b);
        vb->up_w = load_f32(m, "%sblocks.%d.mlp.up_proj.weight", V, b);
        vb->up_b = load_f32(m, "%sblocks.%d.mlp.up_proj.bias", V, b);
        vb->down_w = load_f32(m, "%sblocks.%d.mlp.down_proj.weight", V, b);
        vb->down_b = load_f32(m, "%sblocks.%d.mlp.down_proj.bias", V, b);
    }
    m->vision.blocks = m->vblocks;
    m->has_vision = 1;
}

/* Un'immagine gia' in patch -> embedding pronti per il flusso testuale.
 *
 * `patches` e' [grid_h*grid_w, in_channels*temporal*patch*patch] nell'ordine in
 * cui il processore li produce; la torre restituisce
 * grid_h/merge * grid_w/merge righe da out_hidden. Il chiamante possiede il
 * buffer restituito. */
static float *vision_encode(GModel *m, const float *patches,
                            int grid_h, int grid_w, int *out_tokens) {
    if (!m->has_vision) {
        fprintf(stderr, "questo checkpoint non porta la torre vision\n");
        exit(1);
    }
    const int tokens = coli_vision_output_tokens(&m->vision.config, grid_h, grid_w);
    if (tokens <= 0) {
        fprintf(stderr, "griglia %dx%d non divisibile per merge %d\n",
                grid_h, grid_w, m->vision.config.merge);
        exit(1);
    }
    float *out = malloc((size_t)tokens * m->vision.config.out_hidden * sizeof(float));
    if (!out) { fprintf(stderr, "OOM sugli embedding vision\n"); exit(1); }
    if (coli_vision_forward(out, &m->vision, patches, grid_h, grid_w) != 0) {
        fprintf(stderr, "la torre vision ha rifiutato l'ingresso\n");
        exit(1);
    }
    /* La torre esce a out_hidden; il flusso testuale vuole hidden. Il
     * checkpoint li dichiara uguali (4096) e il merger e' proprio il pezzo che
     * fa combaciare i due, quindi una differenza qui e' una config sbagliata,
     * non qualcosa da rattoppare in silenzio. */
    if (m->vision.config.out_hidden != m->c.hidden) {
        fprintf(stderr, "vision out_hidden %d != hidden %d\n",
                m->vision.config.out_hidden, m->c.hidden);
        exit(1);
    }
    *out_tokens = tokens;
    return out;
}

/* ---------- sessione ---------- */
static GSession *session_open(const GModel *m, int cap, int slot) {
    const Cfg *c = &m->c;
    GSession *s = calloc(1, sizeof(*s));
    if (!s) { fprintf(stderr, "OOM sulla sessione\n"); exit(1); }
    s->cap = cap;
    s->slot = slot;
    s->layer = calloc((size_t)c->n_layers, sizeof(*s->layer));
    if (!s->layer) { fprintf(stderr, "OOM sugli stati di layer\n"); exit(1); }
    if (c->kda_proj)
        s->kda_scratch = malloc((size_t)coli_kda_scratch_floats(c->kda_heads, c->kda_hd,
                                                                c->kda_hd) * sizeof(float));
    for (int i = 0; i < c->n_layers; i++) {
        GLayerState *st = &s->layer[i];
        if (c->is_full[i]) {
            st->latent = malloc((size_t)cap * c->kv_lora * sizeof(float));
            st->ikeys = malloc((size_t)cap * c->index_hd * sizeof(float));
            st->igates = malloc((size_t)cap * c->index_hd * sizeof(float));
            if (!st->latent || !st->ikeys || !st->igates) {
                fprintf(stderr, "OOM sulla cache del layer %d\n", i); exit(1);
            }
        } else if (c->kda_proj) {
            st->kda_state = calloc((size_t)c->kda_heads * c->kda_hd * c->kda_hd,
                                   sizeof(float));
            st->kda_window = calloc((size_t)3 * c->kda_proj * c->conv_k, sizeof(float));
            if (!st->kda_state || !st->kda_window) {
                fprintf(stderr, "OOM sullo stato KDA del layer %d\n", i); exit(1);
            }
#ifdef COLI_VULKAN
            /* G12: hand the (zero) state and the immutable conv taps to dev0 once.
             * Per layer, so a device that runs out of room falls back for that
             * layer alone rather than silently mixing two recurrences. */
            /* COLI_KDA_GPU=1 runs only the recurrence on dev0 (two submits still);
             * =2 runs the whole layer as one submit. kda_gpu carries the mode so a
             * layer that could not take its weights stays on the CPU entirely. */
            if (kda_gpu_on() && !g_kda_cpu_on())
                st->kda_gpu = coli_vk_kda_init(i, slot, c->kda_heads, c->kda_hd, c->kda_hd,
                                               c->conv_k, st->kda_state, st->kda_window,
                                               m->layer[i].conv, m->layer[i].alog,
                                               m->layer[i].dt, m->layer[i].onorm)
                            ? kda_gpu_on() : 0;
#endif
        }
    }
    if (getenv("GLM53_VERBOSE")) {
        int full = 0;
        for (int i = 0; i < c->n_layers; i++) if (c->is_full[i]) full++;
        const double per_token = (double)full * (c->kv_lora + 2 * c->index_hd) * sizeof(float);
        fprintf(stderr, "cache: %.1f KB per token su %d layer DSA "
                        "(%.2f GB a %d posizioni)\n",
                per_token / 1024.0, full, per_token * cap / 1e9, cap);
    }
    return s;
}

static void session_close(const GModel *m, GSession *s) {
    if (!s) return;
    for (int i = 0; i < m->c.n_layers; i++) {
        GLayerState *st = &s->layer[i];
        free(st->latent); free(st->ikeys); free(st->igates);
        free(st->kda_state); free(st->kda_window);
    }
    free(s->kda_scratch);
    free(s->layer);
    free(s);
}

/* I layer [begin, end) su `streams`, che entra e esce come H flussi residui
 * per posizione. E' il pezzo che un segment esegue: prima e dopo ci sono
 * l'embedding e la testa, che stanno agli estremi della catena e non in mezzo.
 *
 * `next` e' il secondo banco, della stessa misura: il passaggio li scambia a
 * ogni sito, quindi alla fine il risultato puo' essere in uno o nell'altro, e
 * la funzione restituisce quale. */
/* G3 per-op report: where a decode token goes, split KDA / MLA / dense FFN /
 * MoE FFN (router / shared expert / [PROF] expert groups and CPU experts) /
 * mHC+norm plumbing / lm_head. Summed over layers; the CLI zeroes everything
 * after prefill so a --greedy run prints decode-only figures, which is the
 * regime the per-op table describes. State and helpers are declared with the
 * [PROF] counters, above ffn_layer. */
static void optime_reset(void) {
    g_ot_kda = g_ot_mla = g_ot_ffn_dense = g_ot_ffn_moe = g_ot_hc = g_ot_head = g_ot_layers = 0.0;
    g_on_kda = g_on_mla = g_on_ffn_dense = g_on_ffn_moe = g_on_hc = g_on_head = g_on_layers = 0;
    g_ot_router = g_ot_shared = 0.0; g_on_router = 0;
    g_kt_proj = g_kt_decay = g_kt_step = g_kt_norm = g_kt_ko = 0.0;
    g_kn_calls = g_kn_batched = 0;
    g_mt_proj = g_mt_index = g_mt_attn = 0.0; g_mn_calls = 0; g_mn_seen = 0.0;
    /* The [PROF] sub-split of the MoE bucket must describe the same window,
     * so it is zeroed here too -- only under COLI_TIMERS=1, so the default
     * [PROF] line keeps its whole-run meaning. */
    g_t_pop = g_b_pop = g_t_pop_max = 0.0; g_n_pop = 0;
    g_n_bind_contig = g_n_bind_split = g_n_bind_gpu = g_n_bind_cpu = 0;
    g_map_serve = g_map_copy = 0;
#ifdef COLI_VULKAN
    g_t_eg = g_t_cpu = 0.0; g_n_eg = g_n_eg_disp = g_n_cpu = g_n_devloss = 0;
#endif
}
__attribute__((destructor)) static void optime_print(void) {
    if (!optime_on() || !g_on_layers) return;
    const double sum = g_ot_kda + g_ot_mla + g_ot_ffn_dense + g_ot_ffn_moe + g_ot_hc;
    fprintf(stderr, "[OPTIME] forwards=%ld layers=%.3fs head=%.3fs (n=%ld)\n",
            g_on_layers, g_ot_layers, g_ot_head, g_on_head);
    fprintf(stderr, "[OPTIME] kda=%.3fs n=%ld (%.3f ms/call) | mla=%.3fs n=%ld (%.3f ms/call)\n",
            g_ot_kda, g_on_kda, g_on_kda ? 1e3 * g_ot_kda / g_on_kda : 0.0,
            g_ot_mla, g_on_mla, g_on_mla ? 1e3 * g_ot_mla / g_on_mla : 0.0);
    fprintf(stderr, "[OPTIME] ffn_dense=%.3fs n=%ld (%.3f ms/call) | ffn_moe=%.3fs n=%ld (%.3f ms/call)\n",
            g_ot_ffn_dense, g_on_ffn_dense, g_on_ffn_dense ? 1e3 * g_ot_ffn_dense / g_on_ffn_dense : 0.0,
            g_ot_ffn_moe, g_on_ffn_moe, g_on_ffn_moe ? 1e3 * g_ot_ffn_moe / g_on_ffn_moe : 0.0);
    fprintf(stderr, "[OPTIME] hc+norm=%.3fs n=%ld (%.3f ms/site) | layers-sum=%.3fs unaccounted=%.3fs\n",
            g_ot_hc, g_on_hc, g_on_hc ? 1e3 * g_ot_hc / g_on_hc : 0.0, sum, g_ot_layers - sum);
    if (g_kn_calls)
        fprintf(stderr, "[OPTIME] kda split (n=%ld, gpu-batched=%ld): proj=%.3fs (%.3f ms) "
                        "decay=%.3fs (%.3f) step=%.3fs (%.3f) norm=%.3fs (%.3f) ko=%.3fs (%.3f)\n",
                g_kn_calls, g_kn_batched,
                g_kt_proj,  1e3 * g_kt_proj  / g_kn_calls,
                g_kt_decay, 1e3 * g_kt_decay / g_kn_calls,
                g_kt_step,  1e3 * g_kt_step  / g_kn_calls,
                g_kt_norm,  1e3 * g_kt_norm  / g_kn_calls,
                g_kt_ko,    1e3 * g_kt_ko    / g_kn_calls);
    if (g_mn_calls)
        fprintf(stderr, "[OPTIME] mla split (n=%ld, mean ctx=%.0f): proj=%.3fs (%.3f ms) "
                        "index=%.3fs (%.3f ms) attn=%.3fs (%.3f ms) | index/ctx=%.4f us\n",
                g_mn_calls, g_mn_seen / (double)g_mn_calls,
                g_mt_proj,  1e3 * g_mt_proj  / g_mn_calls,
                g_mt_index, 1e3 * g_mt_index / g_mn_calls,
                g_mt_attn,  1e3 * g_mt_attn  / g_mn_calls,
                1e6 * g_mt_index / g_mn_seen);
    fprintf(stderr, "[OPTIME] moe split: router=%.3fs shared=%.3fs (n=%ld, %.3f + %.3f ms/call); "
                    "eg and cpu experts are the [PROF] line; the rest of ffn_moe is bind+dispatch+accumulate\n",
            g_ot_router, g_ot_shared, g_on_router,
            g_on_router ? 1e3 * g_ot_router / g_on_router : 0.0,
            g_on_router ? 1e3 * g_ot_shared / g_on_router : 0.0);
}

static float *run_layers(GModel *m, GSession *s, float *streams, float *next,
                         int n, int start, int begin, int end) {
    const Cfg *c = &m->c;
    const int H = c->hc_mult, D = c->hidden;
    float *collapsed = malloc((size_t)n * D * sizeof(float));
    float *normed = malloc((size_t)n * D * sizeof(float));
    float *branch = malloc((size_t)n * D * sizeof(float));
    float *post = malloc((size_t)n * H * sizeof(float));
    float *comb = malloc((size_t)n * H * H * sizeof(float));
    if (!collapsed || !normed || !branch || !post || !comb) {
        fprintf(stderr, "OOM nei temporanei del passaggio\n"); exit(1);
    }

    const int timed = optime_on();
    const double t_layers0 = timed ? optime_now() : 0.0;
    for (int i = begin; i < end; i++) {
        GLayer *l = &m->layer[i];
        for (int site = 0; site < 2; site++) {
            const float *fn = site ? l->hc_ffn_fn : l->hc_attn_fn;
            const float *base = site ? l->hc_ffn_base : l->hc_attn_base;
            const float *scale = site ? l->hc_ffn_scale : l->hc_attn_scale;
            double t0 = timed ? optime_now() : 0.0;
            /* P2.3: every t writes only its own slices of collapsed/post/comb/
             * normed, so the rows run in parallel; per-row math unchanged. */
            /* No OpenMP `if` clause: a 1-row decode step must not enter libgomp
             * at all (181 one-thread regions per token cost ~300 ms/token on
             * rome with OMP_PROC_BIND=close, p2 gate 2026-09-06). */
            const int hc_par = n > 1 && !g_prefill_unbatched();
            if (hc_par) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int t = 0; t < n; t++) {
                    coli_hc_pre(collapsed + (size_t)t * D, post + (size_t)t * H,
                                comb + (size_t)t * H * H, streams + (size_t)t * H * D,
                                fn, scale, base, H, D, c->hc_iters, c->eps, c->hc_eps);
                    rms(normed + (size_t)t * D, collapsed + (size_t)t * D,
                        site ? l->post_ln : l->in_ln, D, c->eps);
                }
            } else {
                for (int t = 0; t < n; t++)
                    coli_hc_pre(collapsed + (size_t)t * D, post + (size_t)t * H,
                                comb + (size_t)t * H * H, streams + (size_t)t * H * D,
                                fn, scale, base, H, D, c->hc_iters, c->eps, c->hc_eps);
                for (int t = 0; t < n; t++)
                    rms(normed + (size_t)t * D, collapsed + (size_t)t * D,
                        site ? l->post_ln : l->in_ln, D, c->eps);
            }
            if (timed) { g_ot_hc += optime_now() - t0; g_on_hc++; t0 = optime_now(); }
            /* dev's per-site wall clock, kept next to the fork's op timers:
             * it feeds the serve PROF frame (attn/ffn split) the dashboard
             * reads, which optime's stderr table does not. */
            const double t_phase = now_s();
            if (!site) {
                GLayerState *st = &s->layer[i];
                /* Lo stato non si azzera a ogni chiamata: e' della
                 * conversazione, e azzerarlo qui vorrebbe dire ricominciare
                 * la ricorrenza a ogni token generato. */
                if (c->is_full[i]) {
                    mla_layer(c, l, normed, n, branch, st, start);
                    if (timed) { g_ot_mla += optime_now() - t0; g_on_mla++; }
                } else {
                    kda_layer(c, l, normed, n, branch, st->kda_state, st->kda_window,
                              s->kda_scratch, i, st->kda_gpu, s->slot);
                    if (timed) { g_ot_kda += optime_now() - t0; g_on_kda++; }
                }
            } else {
                ffn_layer(m, l, i, normed, n, branch);
                if (timed) {
                    if (i < c->first_dense) { g_ot_ffn_dense += optime_now() - t0; g_on_ffn_dense++; }
                    else                    { g_ot_ffn_moe   += optime_now() - t0; g_on_ffn_moe++; }
                }
            }
            /* Un solo paio di letture del clock per sito, il ramo dice a chi
             * va il tempo. */
            *(site ? &m->t_ffn : &m->t_attn) += now_s() - t_phase;
            if (timed) t0 = optime_now();
            if (hc_par) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int t = 0; t < n; t++)
                    coli_hc_post(next + (size_t)t * H * D, branch + (size_t)t * D,
                                 streams + (size_t)t * H * D, post + (size_t)t * H,
                                 comb + (size_t)t * H * H, H, D);
            } else {
                for (int t = 0; t < n; t++)
                    coli_hc_post(next + (size_t)t * H * D, branch + (size_t)t * D,
                                 streams + (size_t)t * H * D, post + (size_t)t * H,
                                 comb + (size_t)t * H * H, H, D);
            }
            if (timed) g_ot_hc += optime_now() - t0;   /* same site: counted once above */
            float *swap = streams; streams = next; next = swap;
        }
    }
    if (timed) { g_ot_layers += optime_now() - t_layers0; g_on_layers++; }
    free(comb); free(post); free(branch); free(normed); free(collapsed);
    return streams;
}

/* Rilascio completo del modello.
 *
 * Una CLI che finisce lascia fare al sistema operativo; un ospite che tiene
 * piu' motori nello stesso processo no, e un motore che non si smonta diventa
 * una perdita per richiesta. Ogni allocazione fatta dal caricamento ha qui il
 * suo rilascio, l'indice dei tensori compreso. */
static void mat_release(Mat *mat) {
    free((void *)mat->f); free((void *)mat->q8);
    free((void *)mat->q4); free((void *)mat->s);
    memset(mat, 0, sizeof(*mat));
}

static void model_release(GModel *m) {
    if (!m) return;
    if (m->layer) {
        for (int i = m->layer_begin; i < m->layer_end; i++) {
            GLayer *l = &m->layer[i];
            Mat *mats[] = { &l->kq, &l->kk, &l->kv, &l->ko, &l->kga, &l->kgb,
                            &l->kfa, &l->kfb, &l->kb, &l->qa, &l->qb, &l->kva,
                            &l->kvb_kt, &l->kvb_v, &l->o, &l->iwq, &l->iwk,
                            &l->iwp, &l->ikpg, &l->dg, &l->du, &l->dd,
                            &l->rg, &l->ru, &l->rd };
            for (size_t k = 0; k < sizeof(mats) / sizeof(*mats); k++) mat_release(mats[k]);
            const float *vectors[] = { l->in_ln, l->post_ln, l->hc_attn_fn,
                                       l->hc_attn_base, l->hc_attn_scale,
                                       l->hc_ffn_fn, l->hc_ffn_base, l->hc_ffn_scale,
                                       l->conv, l->dt, l->alog, l->onorm,
                                       l->qa_ln, l->kva_ln, l->ik_nw, l->ik_nb,
                                       l->ikpa, l->router, l->rbias };
            for (size_t k = 0; k < sizeof(vectors) / sizeof(*vectors); k++)
                free((void *)vectors[k]);
            if (!m->streaming && l->eg) {
                for (int e = 0; e < m->c.n_experts; e++) {
                    mat_release(&l->eg[e]); mat_release(&l->eu[e]); mat_release(&l->ed[e]);
                }
            }
            free(l->eg); free(l->eu); free(l->ed);
        }
        free(m->layer);
    }
    if (m->ecache) {
        for (int i = 0; i < m->c.n_layers; i++) {
            LCache *cache = &m->ecache[i];
            for (int j = 0; j < cache->cap; j++) free(cache->s[j].own);
            free(cache->s);
        }
        free(m->ecache);
    }
    free(m->eref);
    if (m->vblocks) {
        for (int b = 0; b < m->c.vis_layers; b++) {
            ColiVisionBlock *vb = &m->vblocks[b];
            const float *parts[] = { vb->norm1, vb->norm2, vb->qkv_w, vb->qkv_b,
                                     vb->q_norm, vb->k_norm, vb->proj_w, vb->proj_b,
                                     vb->gate_w, vb->gate_b, vb->up_w, vb->up_b,
                                     vb->down_w, vb->down_b };
            for (size_t k = 0; k < sizeof(parts) / sizeof(*parts); k++)
                free((void *)parts[k]);
        }
        free(m->vblocks);
        const float *tower[] = { m->vision.patch_w, m->vision.patch_b,
                                 m->vision.post_norm, m->vision.down_w,
                                 m->vision.down_b, m->vision.merger_proj,
                                 m->vision.merger_norm_w, m->vision.merger_norm_b,
                                 m->vision.merger_gate, m->vision.merger_up,
                                 m->vision.merger_down };
        for (size_t k = 0; k < sizeof(tower) / sizeof(*tower); k++) free((void *)tower[k]);
    }
    /* La testa puo' essere la tabella degli embedding: liberarla due volte
     * sarebbe un doppio free, non un risparmio. */
    if (m->head.f != m->embed) mat_release(&m->head);
    free((void *)m->embed);
    free((void *)m->final_norm);
    st_destroy(&m->S);
    memset(m, 0, sizeof(*m));
}

/* Il caso pieno: tutti i layer, embedding e testa comprese. */
static void model_load(GModel *m, const char *dir) {
    model_load_range(m, dir, 0, -1, 1);
}

/* ---------- il passaggio completo ----------
 *
 * `vision` sono gli embedding usciti dalla torre, `n_vision` quanti sono: uno
 * per ogni token immagine presente in `tokens`, nello stesso ordine. Il
 * processore ha gia' espanso ogni immagine in altrettanti segnaposto
 * `image_token_id`, quindi qui non c'e' nulla da inserire: si sostituisce la
 * riga dell'embedding testuale con quella della torre e le posizioni restano
 * quelle che sono. Prompt di solo testo passano vision=NULL, n_vision=0. */
static int g_span_last_only = 0;   /* set by forward_prefill around its chunks */
static float *forward_span(GModel *m, GSession *s, const int *tokens, int n,
                           const float *vision, int n_vision) {
    const Cfg *c = &m->c;
    const int start = s->filled;   /* NON 'base': nel ciclo dei layer e' gia' preso */
    if (start + n > s->cap) {
        fprintf(stderr, "contesto esaurito: %d posizioni su %d\n", start + n, s->cap);
        exit(1);
    }
    const int H = c->hc_mult, D = c->hidden;
    float *streams = malloc((size_t)n * H * D * sizeof(float));
    float *next = malloc((size_t)n * H * D * sizeof(float));
    /* l'embedding entra replicato in ognuno degli H flussi residui; sui token
     * immagine la riga viene dalla torre invece che dalla tabella */
    int consumed = 0;
    for (int t = 0; t < n; t++) {
        const float *row;
        /* Un id fuori dal vocabolario legge oltre la tabella degli embedding.
         * Da CLI sarebbe un errore di battitura; da server e' input di rete. */
        if (tokens[t] < 0 || tokens[t] >= c->vocab) {
            fprintf(stderr, "token %d fuori dal vocabolario (0..%d)\n",
                    tokens[t], c->vocab - 1);
            exit(1);
        }
        if (vision && c->image_token >= 0 && tokens[t] == c->image_token) {
            if (consumed >= n_vision) {
                fprintf(stderr, "il prompt ha piu' token immagine (%d+) degli "
                                "embedding forniti (%d)\n", consumed + 1, n_vision);
                exit(1);
            }
            row = vision + (size_t)consumed++ * D;
        } else {
            row = m->embed + (size_t)tokens[t] * D;
        }
        for (int h = 0; h < H; h++)
            memcpy(streams + ((size_t)t * H + h) * D, row, (size_t)D * sizeof(float));
    }
    /* Avanzare in silenzio con embedding non consumati vorrebbe dire mandare al
     * modello un'immagine diversa da quella che il chiamante crede di aver
     * passato: e' un errore, non un caso limite. */
    if (consumed != n_vision) {
        fprintf(stderr, "%d embedding vision forniti ma solo %d token immagine "
                        "nel prompt\n", n_vision, consumed);
        exit(1);
    }

    streams = run_layers(m, s, streams, next, n, start,
                         m->layer_begin, m->layer_end);

    float *collapsed = malloc((size_t)n * D * sizeof(float));
    float *normed = malloc((size_t)n * D * sizeof(float));
    if (!collapsed || !normed) { fprintf(stderr, "OOM in chiusura\n"); exit(1); }

    /* i flussi si richiudono con una media NON pesata */
    const int tail_par = n > 1 && !g_prefill_unbatched();
    if (tail_par) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int t = 0; t < n; t++) {
            for (int d = 0; d < D; d++) {
                float sum = 0.0f;
                for (int h = 0; h < H; h++) sum += streams[((size_t)t * H + h) * D + d];
                collapsed[(size_t)t * D + d] = sum / H;
            }
            rms(normed + (size_t)t * D, collapsed + (size_t)t * D, m->final_norm, D, c->eps);
        }
    } else {
        for (int t = 0; t < n; t++)
            for (int d = 0; d < D; d++) {
                float sum = 0.0f;
                for (int h = 0; h < H; h++) sum += streams[((size_t)t * H + h) * D + d];
                collapsed[(size_t)t * D + d] = sum / H;
            }
        for (int t = 0; t < n; t++)
            rms(normed + (size_t)t * D, collapsed + (size_t)t * D, m->final_norm, D, c->eps);
    }

    float *logits = malloc((size_t)n * c->vocab * sizeof(float));
    const double t_head0 = optime_on() ? optime_now() : 0.0;
    const double t_head_wall = now_s();
    /* P2.3: a prefill chunk whose caller keeps only the last row (the serve
     * path, forward_prefill keep_all == 0) gets the head for that row only;
     * the teacher-forcing oracle keeps every row and pays every head. */
    const int t_first = (g_span_last_only && n > 1) ? n - 1 : 0;
    for (int t = t_first; t < n; t++)
        mv(logits + (size_t)t * c->vocab, &m->head, normed + (size_t)t * D);
    if (optime_on()) { g_ot_head += optime_now() - t_head0; g_on_head++; }
    m->t_head += now_s() - t_head_wall;   /* dev's PROF frame */
    m->forwards++;

    free(normed); free(collapsed);
    free(next); free(streams);
    s->filled = start + n;
    return logits;
}

/* Prefill a pezzi.
 *
 * Un prefill in un colpo solo tiene in memoria i temporanei di tutto il
 * prompt: due banchi di flussi residui per hc_mult, piu' le proiezioni di
 * ogni layer. Su GLM-5.3-Flash sono quasi quattro GB per ottomila token, che
 * e' assurdo per un motore che esiste per stare stretto. A pezzi il lavoro e'
 * lo stesso e i temporanei costano quanto il pezzo.
 *
 * Il pezzo resta grande abbastanza da non buttare via il vantaggio del
 * prefill: gli esperti si leggono da disco una volta per pezzo e per layer,
 * quindi pezzi minuscoli li rileggerebbero in continuazione.
 *
 * Con `keep_all` si tengono i logit di ogni posizione, che serve solo al
 * confronto con l'oracolo; altrimenti si tiene l'ultima riga, che e' l'unica
 * che decide il token successivo. */
/* ---------- CANCEL mentre il turno e' in volo ----------
 *
 * Misurato il 2026-09-06: un prompt di 1 230 token, CANCEL a 5 s, il motore ha
 * confermato dopo 251,3 s -- cioe' ha macinato tutto il prefill e tutta la
 * generazione. Il gateway intanto tiene l'ammissione dello scheduler in attesa
 * dell'ack (openai_server.py, generate(): "releasing it before the engine
 * confirms the cancel lets the next request SUBMIT into a pipe the busy engine
 * is not reading"), quindi un client che si disconnette non libera niente: la
 * richiesta dopo aspetta comunque la fine del turno.
 *
 * La primitiva e' serve_poll.h (#1332): "c'e' un byte leggibile su stdin,
 * adesso?", senza mai leggere e senza mai bloccare. Il formato delle righe lo
 * conosce questo file, ed e' lo stesso che parsa serve_read_req.
 *
 * Dove si guarda: FRA un pezzo di prefill e l'altro (forward_prefill) e FRA un
 * token di decode e l'altro (serve_one). Mai dentro un pezzo: il pezzo e'
 * atomico, e fermarsi a meta' lascerebbe la sessione in uno stato che nessuna
 * posizione descrive. Il costo e' una select() con timeout zero ogni 128
 * posizioni di prefill e ogni token generato -- il gate lo misura invece di
 * assumerlo, e GLM53_NO_CANCEL_POLL=1 e' l'A/B.
 *
 * Cosa si fa delle righe lette:
 *   CANCEL <id in volo>  -> si alza la bandiera; il turno finisce il pezzo (o
 *                           il token) in corso e si ferma;
 *   CANCEL <altro id>    -> ERROR <id> NOT_FOUND, la stessa risposta che dava
 *                           serve_loop quando il CANCEL arrivava a turno finito;
 *   STOP e il resto      -> si scartano, com'e' sempre stato: serve_loop le
 *                           ignora gia' fra una richiesta e l'altra.
 *
 * E un SUBMIT a meta' turno? Il compito diceva che non puo' arrivare perche'
 * il gateway serializza. NON E' VERO in questa configurazione, ed e' la cosa
 * piu' importante che questa modifica ha dovuto scoprire da sola:
 * GenerationScheduler nasce con `capacity = kv_slots` (openai_server.py:3258),
 * cioe' QUATTRO in servizio, e generate() scrive il frame appena e' ammessa,
 * sotto il solo write_lock. Con --kv-slots 4 il gateway puo' quindi avere fino
 * a quattro SUBMIT nella stessa pipa: il motore ne serve uno per volta e gli
 * altri aspettano nel tubo, che e' come questo motore fa concorrenza oggi.
 * Un drain che scartasse quell'intestazione (qwen36.c:2616 fa cosi'; i suoi
 * frame pero' passano dal codec) lascerebbe nel flusso il payload a byte
 * contati e disallineerebbe il parser -- cioe' romperebbe il servizio invece
 * di sistemarlo.
 *
 * Quindi: la riga che non e' un comando NON si consuma davvero, si mette da
 * parte (`g_pushback`) e serve_read_req la rilegge come sua intestazione al
 * giro dopo. Il payload resta intatto nella pipa, dove serve_read_req lo
 * aspetta. E il drain si ferma li': i byte che seguono appartengono a QUEL
 * frame, e leggerne un altro vorrebbe dire parsarne il corpo.
 *
 * Il prezzo, detto per intero: se un SUBMIT e' gia' in coda quando arriva il
 * CANCEL del turno in volo, il CANCEL sta DIETRO quel frame e non si vede
 * fino a fine turno -- cioe' esattamente il comportamento di prima. Non e'
 * mai peggio di prima, e nel caso che il gate misura (una richiesta in volo,
 * il browser che si chiude) e' la differenza fra 251 s e pochi secondi.
 * Vederlo anche dietro un frame in coda vorrebbe dire parsare il SUBMIT per
 * intero qui dentro -- la strada di kimi_k3 (k3_serve_poll_cancel, che
 * risponde "engine busy") -- e quindi una fread bloccante su un payload
 * arrivato a meta' dentro il ciclo di decode. Non con questa modifica. */
typedef struct {
    unsigned long long id;   /* la richiesta in volo */
    int cancelled;           /* un CANCEL per lei e' arrivato */
    int done;                /* posizioni macinate dall'ultima forward_prefill */
} ServeCancel;

static int cancel_poll_off(void) {
    static int off = -1;
    if (off < 0) {
        const char *v = getenv("GLM53_NO_CANCEL_POLL");
        off = (v && atoi(v) != 0) ? 1 : 0;
    }
    return off;
}

/* Un'intestazione letta dal drain che non era un comando: serve_read_req la
 * riprende da qui invece che da stdin, e il corpo del frame la aspetta ancora
 * nella pipa. */
static char g_pushback[512];
static int g_pushback_full = 0;

static int serve_cancel_pending(ServeCancel *c) {
    if (!c) return 0;
    if (c->cancelled) return 1;
    if (cancel_poll_off()) return 0;
    while (!g_pushback_full && coli_serve_stdin_ready()) {
        char line[512], cmd[16];
        unsigned long long who = 0;
        if (!fgets(line, sizeof(line), stdin)) {
            /* EOF. Su una pipe chiusa "pronto" resta vero per sempre, quindi si
             * esce dal ciclo -- e si tratta come un annullamento: continuare a
             * generare per un lettore che non c'e' piu' e' l'unico esito
             * peggiore del non fermarsi. */
            c->cancelled = 1;
            break;
        }
        if (sscanf(line, "%15s %llu", cmd, &who) < 1) continue;
        if (!strcmp(cmd, "CANCEL")) {
            if (who == c->id) c->cancelled = 1;
            else { printf("ERROR %llu NOT_FOUND\n", who); fflush(stdout); }
            continue;
        }
        /* STOP non porta corpo e serve_loop lo ignora gia' fra una richiesta e
         * l'altra: scartarlo qui e' lo stesso comportamento. */
        if (!strcmp(cmd, "STOP")) continue;
        /* Tutto il resto (SUBMIT, IMAGE, una riga che non conosciamo) puo'
         * avere un corpo a byte contati dietro: si rimette da parte e il drain
         * finisce qui. */
        snprintf(g_pushback, sizeof(g_pushback), "%s", line);
        g_pushback_full = 1;
        break;
    }
    return c->cancelled;
}

static float *forward_prefill(GModel *m, GSession *s, const int *tokens, int n,
                              const float *vision, int n_vision, int keep_all,
                              ServeCancel *cancel) {
    const Cfg *c = &m->c;
    const char *setting = getenv("GLM53_PREFILL_CHUNK");
    int chunk = setting ? atoi(setting) : 128;
    if (chunk < 1) chunk = 1;
    if (chunk > n) chunk = n;

    float *all = keep_all ? malloc((size_t)n * c->vocab * sizeof(float)) : NULL;
    if (keep_all && !all) { fprintf(stderr, "OOM sui logit del prefill\n"); exit(1); }
    float *last = NULL;
    int used_vision = 0;

    if (cancel) cancel->done = 0;
    for (int at = 0; at < n; at += chunk) {
        const int here = at + chunk <= n ? chunk : n - at;
        /* Gli embedding dell'immagine vanno divisi come i token: a ogni pezzo
         * quelli dei segnaposto che contiene, altrimenti il conto non torna e
         * il motore si ferma -- che e' quello che deve fare. */
        int mine = 0;
        if (vision && c->image_token >= 0)
            for (int i = 0; i < here; i++)
                if (tokens[at + i] == c->image_token) mine++;
        g_span_last_only = !keep_all;
        float *part = forward_span(m, s, tokens + at, here,
                                   vision ? vision + (size_t)used_vision * c->hidden : NULL,
                                   mine);
        g_span_last_only = 0;
        used_vision += mine;
        if (keep_all) {
            memcpy(all + (size_t)at * c->vocab, part,
                   (size_t)here * c->vocab * sizeof(float));
            free(part);
        } else {
            free(last);
            last = part;
            if (here > 1) {
                /* si tiene solo l'ultima riga */
                float *tail = malloc((size_t)c->vocab * sizeof(float));
                if (!tail) { fprintf(stderr, "OOM sui logit\n"); exit(1); }
                memcpy(tail, last + (size_t)(here - 1) * c->vocab,
                       (size_t)c->vocab * sizeof(float));
                free(last);
                last = tail;
            }
        }
        /* Il punto di annullamento: qui la sessione e' esattamente a
         * `start + at + here` e nient'altro e' a meta'. */
        if (cancel) {
            cancel->done = at + here;
            if (serve_cancel_pending(cancel)) break;
        }
    }
    /* Un prefill annullato consuma solo i segnaposto dei pezzi che ha
     * macinato: il conto non torna per costruzione, e non e' un errore. */
    if (vision && used_vision != n_vision && !(cancel && cancel->cancelled)) {
        fprintf(stderr, "%d embedding vision forniti ma %d segnaposto nel prompt\n",
                n_vision, used_vision);
        exit(1);
    }
    return keep_all ? all : last;
}

/* Il passaggio senza sessione: apre, macina tutto, chiude. E' quello che usano
 * gli oracoli, dove il punto e' il risultato e non il tempo. */
static float *forward(GModel *m, const int *tokens, int n,
                      const float *vision, int n_vision) {
    GSession *s = session_open(m, n, 0);
    float *logits = forward_span(m, s, tokens, n, vision, n_vision);
    session_close(m, s);
    return logits;
}

/* ---------- CLI ----------
 * Due modi, entrambi pensati per essere confrontati con ref.json:
 *   --ids a,b,c            teacher forcing: stampa l'argmax a ogni posizione
 *   --ids a,b,c --greedy N genera N token continuando dal prompt
 * Il tokenizzatore non serve: l'oracolo parla in id. */
static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* Gli id di fine generazione. GLM ne dichiara piu' di uno (fine turno, fine
 * testo, fine blocco strumenti) e fermarsi solo sul primo vuol dire vedere il
 * modello continuare a parlare oltre la sua risposta. */
static int load_stops(const char *dir, int *out, int max) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/generation_config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, f) != (size_t)size) {
        free(text); fclose(f); return 0;
    }
    text[size] = 0; fclose(f);
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    int found = 0;
    if (root && root->t == J_OBJ) {
        jval *eos = json_get(root, "eos_token_id");
        if (eos && eos->t == J_NUM && found < max) out[found++] = (int)eos->num;
        else if (eos && eos->t == J_ARR)
            for (int i = 0; i < eos->len && found < max; i++)
                if (eos->kids[i]->t == J_NUM) out[found++] = (int)eos->kids[i]->num;
    }
    free(arena);
    free(text);
    return found;
}

/* ================= protocollo serve =================
 *
 * Due protocolli sulla stessa pipa, come gli altri motori (docs/serve_protocol.md):
 * il mux con SERVE_BATCH=1, che e' quello che parlano openai_server.py e
 * `coli web`, e quello interattivo con SERVE=1 da solo, che usa `coli chat`.
 * Le righe escono una per write con fflush, e su Windows i due capi vanno messi
 * in binario prima di tutto: la traduzione CRLF del CRT rovina i sentinelli e
 * blocca le letture a byte contati (#195).
 *
 * Le richieste portano un indice di slot KV. Qui viene accettato e non usato:
 * questo motore riprefilla ogni volta invece di riprendere la conversazione da
 * dove era. E' piu' lento e non e' sbagliato, e il giorno che ci sara' una
 * cache il protocollo non cambia. */
/* noinline: GCC 16.1 (MSYS2 UCRT64) crashes in its IPA inliner when this
 * clock is inlined into run_layers/forward_span at every phase timer; the
 * bisect on the CI runner points at the inlining, not the timers, and a call
 * per phase costs nothing next to a layer. */
__attribute__((noinline)) static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static double rss_gb(void) {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1e6;             /* ru_maxrss e' in KB anche su Windows */
}

static float g_temp = 0.0f, g_topp = 1.0f;

/* Confronto per qsort. Una funzione annidata sarebbe piu' comoda ma e'
 * un'estensione GCC, e questo file deve compilare anche con MSVC. */
static const float *g_sort_key = NULL;
static int compare_by_probability(const void *a, const void *b) {
    const int x = *(const int *)a, y = *(const int *)b;
    if (g_sort_key[x] < g_sort_key[y]) return 1;
    if (g_sort_key[x] > g_sort_key[y]) return -1;
    return 0;
}

/* Temperatura piu' nucleus. Con temperatura nulla e' argmax e basta. */
static int sample_token(const float *logits, int vocab) {
    if (g_temp <= 1e-4f) return argmax(logits, vocab);
    static float *probability = NULL;
    static int *order = NULL;
    static int room = 0;
    if (room < vocab) {
        probability = realloc(probability, (size_t)vocab * sizeof(float));
        order = realloc(order, (size_t)vocab * sizeof(int));
        room = vocab;
        if (!probability || !order) { fprintf(stderr, "OOM nel campionamento\n"); exit(1); }
    }
    float top = -INFINITY;
    for (int i = 0; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    const float inverse = 1.0f / g_temp;
    for (int i = 0; i < vocab; i++) {
        probability[i] = expf((logits[i] - top) * inverse);
        total += probability[i];
        order[i] = i;
    }
    for (int i = 0; i < vocab; i++) probability[i] = (float)(probability[i] / total);

    /* nucleus: si ordina per probabilita' e si taglia dove la somma supera
     * top_p. Con vocab 154880 l'ordinamento pieno costa, ma succede una volta
     * per token e il token costa incomparabilmente di piu'. */
    g_sort_key = probability;
    qsort(order, (size_t)vocab, sizeof(int), compare_by_probability);
    double cumulative = 0.0;
    int keep = vocab;
    for (int i = 0; i < vocab; i++) {
        cumulative += probability[order[i]];
        if (cumulative >= g_topp) { keep = i + 1; break; }
    }
    double pick = ((double)rand() / ((double)RAND_MAX + 1.0)) * cumulative;
    double walk = 0.0;
    for (int i = 0; i < keep; i++) {
        walk += probability[order[i]];
        if (walk >= pick) return order[i];
    }
    return order[0];
}

/* ---------- slot KV ----------
 *
 * Un turno di chat rende tutta la conversazione da capo, quindi il prompt del
 * secondo turno comincia con quello del primo piu' la risposta che il motore
 * ha appena dato. Tenere la sessione dello slot e ripartire da dove il prefisso
 * smette di combaciare fa pagare solo la parte nuova.
 *
 * Il riuso vale solo in avanti. Lo stato ricorrente dei layer KDA non si
 * riavvolge: la cache DSA si potrebbe troncare, essendo posizionale, ma la
 * ricorrenza no, e fingere il contrario darebbe risposte sbagliate in silenzio.
 * Se il prompt nuovo non estende quello vecchio, la sessione si rifa'. */
/* GLM53_MAX_SLOTS is defined near kv_slots_wanted(): model_load sizes the KDA
 * device pool from KV_SLOTS before the expert preload and needs the same cap. */

typedef struct {
    GSession *session;
    int *tokens;                          /* la sequenza che lo slot tiene */
    int n, cap;
} KVSlot;

static KVSlot g_slots[GLM53_MAX_SLOTS];
static int g_n_slots = 0;
static int g_slot_context = 0;

static void slots_init(const GModel *m) {
    g_n_slots = kv_slots_wanted();
    /* P6 (2026-09-06): the device-side KDA recurrence kept ONE state per layer,
     * so two live slots on the GPU path shared and corrupted it, and any
     * KV_SLOTS > 1 forced the CPU recurrence (~10 % decode, ~5 % prefill).
     *
     * P6b (2026-09-07): model_load allocates one state set per slot before the
     * expert preload, so slots coexist on the device. What is left of the guard
     * is the honest version of the same rule: force the CPU recurrence when the
     * pool does NOT cover every slot -- all or nothing, never some slots on the
     * device and some on the CPU, which would make numerics depend on which
     * slot a conversation hashed to. Still loud: a serving config must not
     * drift into silent corruption, and it must not silently lose 10 % of
     * decode either. */
    if (kda_gpu_on()) {
#ifdef COLI_VULKAN
        const int pool = coli_vk_kda_pool_slots();
#else
        const int pool = 0;
#endif
        if (pool < g_n_slots) {
            fprintf(stderr, "KV_SLOTS=%d: forcing COLI_KDA_GPU=0 "
                            "(KDA device slot pool holds %d; P6b)\n", g_n_slots, pool);
            g_kda_gpu = 0;
        } else if (getenv("GLM53_VERBOSE")) {
            fprintf(stderr, "KV_SLOTS=%d: KDA device slot pool holds %d -- "
                            "COLI_KDA_GPU=%d stays on\n", g_n_slots, pool, g_kda_gpu);
        }
    }
    const char *context = getenv("GLM53_MAXT");
    g_slot_context = context ? atoi(context) : 8192;
    if (g_slot_context < 64) g_slot_context = 64;
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "slot KV: %d da %d posizioni\n", g_n_slots, g_slot_context);
    (void)m;
}

static void slot_reset(const GModel *m, KVSlot *slot) {
    if (slot->session) session_close(m, slot->session);
    slot->session = NULL;
    slot->n = 0;
}

/* Quanti token iniziali lo slot ha gia' in cache e puo' tenere. */
static int slot_shared(const KVSlot *slot, const int *tokens, int n) {
    int shared = 0;
    while (shared < slot->n && shared < n && slot->tokens[shared] == tokens[shared])
        shared++;
    return shared;
}

static void slot_remember(KVSlot *slot, const int *tokens, int n) {
    if (n > slot->cap) {
        slot->tokens = realloc(slot->tokens, (size_t)n * sizeof(int));
        if (!slot->tokens) { fprintf(stderr, "OOM sulla storia dello slot\n"); exit(1); }
        slot->cap = n;
    }
    memcpy(slot->tokens, tokens, (size_t)n * sizeof(int));
    slot->n = n;
}

/* ---------- P7: checkpoint del prefisso ----------
 *
 * Uno slot riusa solo IN AVANTI e solo la propria conversazione. Il prefisso
 * che Open WebUI mette davanti a ogni primo turno -- il blocco degli strumenti
 * piu' il system -- e' identico fra conversazioni diverse, ma nessuno slot lo
 * possiede: la conversazione nuova finisce su un altro slot (o sullo stesso
 * dopo un reset) e ripaga 5 400 token di prefill, misurati 876 s il 2026-09-07.
 *
 * Il rimedio e' quello di deepseek_v4.c (v4_ckpt_*, :12470-12760): una copia
 * dello stato della sessione presa AL confine del prefisso e rimessa in una
 * sessione fresca. Lo stato e' esattamente quello che il segment adapter
 * dichiara (glm53_segment_spans): per i layer DSA le `len` righe di latente,
 * chiavi e gate dell'indexer; per i layer KDA la ricorrenza e la finestra
 * della convoluzione, che non dipendono dalla lunghezza.
 *
 * Il confine non si dichiara, si scopre: o e' il prefisso comune di due
 * prompt freschi successivi (il piano), o e' il byte offset che il gateway
 * manda nell'ottavo campo del SUBMIT (il suggerimento). In entrambi i casi il
 * motore verifica che cada su un confine di token e che i primi `len` id
 * coincidano davvero, perche' un confine sbagliato darebbe risposte plausibili
 * e sbagliate.
 *
 * GLM53_PREFIX_CKPT=0 spegne tutto e riporta esattamente il percorso di prima.
 */
enum { GLM53_CKPT_MAX_SLOTS = 8, GLM53_CKPT_MAX_SPANS = 512 };

typedef struct {
    int *ids;                 /* il prefisso, esattamente i token macinati */
    int len;
    unsigned char *blob;      /* gli span in ordine, uno dietro l'altro */
    size_t bytes;
    unsigned long long used;  /* orologio LRU */
    int kind;                 /* 0 = prefisso (piano/suggerimento), 1 = fine prompt */
} PrefixCkpt;

static PrefixCkpt g_ckpt[GLM53_CKPT_MAX_SLOTS];
static unsigned long long g_ckpt_clock;
static int *g_ckpt_prev_ids;
static int g_ckpt_prev_len;

static int ckpt_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *setting = getenv("GLM53_PREFIX_CKPT");
        cached = setting ? atoi(setting) != 0 : 1;
    }
    return cached;
}

static int ckpt_slot_count(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *setting = getenv("GLM53_PREFIX_CKPT_SLOTS");
        cached = setting ? atoi(setting) : 4;
        if (cached < 1) cached = 1;
        if (cached > GLM53_CKPT_MAX_SLOTS) cached = GLM53_CKPT_MAX_SLOTS;
    }
    return cached;
}

static int ckpt_min_tokens(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *setting = getenv("GLM53_PREFIX_CKPT_MIN");
        cached = setting ? atoi(setting) : 128;
        if (cached < 8) cached = 8;
    }
    return cached;
}

static int ckpt_disk_wanted(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *setting = getenv("GLM53_PREFIX_CKPT_DISK");
        cached = setting ? atoi(setting) : 1;
    }
    return cached;
}

/* Una copia intera per richiesta non e' gratis qui (334 MB a 5 400 token),
 * quindi la cattura a fine prompt -- il kind 1 di DeepSeek V4 -- e' spenta di
 * default e si accende solo dove serve. */
static int ckpt_end_wanted(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *setting = getenv("GLM53_PREFIX_CKPT_END");
        cached = setting ? atoi(setting) != 0 : 0;
    }
    return cached;
}

typedef struct { float *ptr; size_t bytes; } CkptSpan;

/* Gli stessi pezzi, nello stesso ordine, di glm53_segment_spans -- con `rows`
 * righe invece della capienza intera della sessione. */
static size_t ckpt_spans(const GModel *m, GSession *s, int rows,
                         CkptSpan *out, size_t capacity) {
    const Cfg *c = &m->c;
    size_t count = 0;
    for (int i = 0; i < c->n_layers; i++) {
        GLayerState *st = &s->layer[i];
        if (c->is_full[i]) {
            if (count + 3 > capacity) return 0;
            if (!st->latent || !st->ikeys || !st->igates) return 0;
            out[count++] = (CkptSpan){ st->latent, (size_t)rows * (size_t)c->kv_lora * sizeof(float) };
            out[count++] = (CkptSpan){ st->ikeys,  (size_t)rows * (size_t)c->index_hd * sizeof(float) };
            out[count++] = (CkptSpan){ st->igates, (size_t)rows * (size_t)c->index_hd * sizeof(float) };
        } else if (c->kda_proj) {
            if (count + 2 > capacity) return 0;
            if (!st->kda_state || !st->kda_window) return 0;
            out[count++] = (CkptSpan){ st->kda_state,
                (size_t)c->kda_heads * (size_t)c->kda_hd * (size_t)c->kda_hd * sizeof(float) };
            out[count++] = (CkptSpan){ st->kda_window,
                3u * (size_t)c->kda_proj * (size_t)c->conv_k * sizeof(float) };
        }
    }
    return count;
}

static size_t ckpt_span_bytes(const CkptSpan *spans, size_t count) {
    size_t total = 0;
    for (size_t i = 0; i < count; i++) total += spans[i].bytes;
    return total;
}

/* G12: con la ricorrenza sul dispositivo le due copie host sono STALE. Sono i
 * due soli punti in cui il checkpoint le fa combaciare col dispositivo, e sono
 * esattamente le due direzioni di glm53_kda_sync_out / _in (:4433, :4445). */
/* P6b: quanto e' costata l'ULTIMA lettura dello stato dal dispositivo. Il
 * percorso vecchio leggeva memoria write-combined a ~14 MB/s (150 MB in 11 s,
 * p2c 2026-09-06); il gate legge questo numero dalla riga CKPT store, perche'
 * "il sync e' veloce" non e' un'affermazione che si possa fare a occhio. */
static double g_ckpt_sync_ms = 0.0;

static void ckpt_sync_out(const GModel *m, GSession *s) {
    const double t0 = now_s();
#ifdef COLI_VULKAN
    const Cfg *c = &m->c;
    for (int i = 0; i < c->n_layers; i++) {
        GLayerState *st = &s->layer[i];
        if (!c->is_full[i] && c->kda_proj && st->kda_gpu)
            coli_vk_kda_sync(i, s->slot, st->kda_state, st->kda_window);
    }
#else
    (void)m; (void)s;
#endif
    g_ckpt_sync_ms = (now_s() - t0) * 1000.0;
}

static void ckpt_sync_in(const GModel *m, GSession *s) {
#ifdef COLI_VULKAN
    const Cfg *c = &m->c;
    for (int i = 0; i < c->n_layers; i++) {
        GLayerState *st = &s->layer[i];
        if (!c->is_full[i] && c->kda_proj && st->kda_gpu)
            coli_vk_kda_upload(i, s->slot, st->kda_state, st->kda_window);
    }
#else
    (void)m; (void)s;
#endif
}

static void ckpt_free(PrefixCkpt *slot) {
    free(slot->ids);
    free(slot->blob);
    memset(slot, 0, sizeof(*slot));
}

static int ckpt_have(const int *ids, int len) {
    for (int i = 0; i < ckpt_slot_count(); i++)
        if (g_ckpt[i].ids && g_ckpt[i].len == len &&
            !memcmp(g_ckpt[i].ids, ids, (size_t)len * sizeof(int)))
            return 1;
    return 0;
}

/* ---- persistenza su disco ----
 *
 * Gli slot in memoria muoiono col processo, e il gateway si riavvia: senza il
 * file il primo turno dopo ogni riavvio ripaga il prefisso intero. Il
 * fingerprint dice che il file descrive QUESTO modello con QUESTA config; se
 * non combacia il file si cancella invece di essere interpretato. */
static char g_ckpt_dir[1024];
static uint32_t g_ckpt_fp;
static int g_ckpt_disk_loaded;

static uint32_t ckpt_fingerprint(const GModel *m) {
    const Cfg *c = &m->c;
    uint32_t h = 2166136261u;
    const int fields[] = { c->n_layers, c->kv_lora, c->index_hd, c->kda_heads,
                           c->kda_hd, c->conv_k, glm53_dense_bits() };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
        h ^= (uint32_t)fields[i]; h *= 16777619u;
    }
    for (int i = 0; i < c->n_layers; i++) { h ^= (uint32_t)c->is_full[i]; h *= 16777619u; }
    const char *snap = getenv("SNAP");
    for (const char *p = snap ? snap : ""; *p; p++) { h ^= (uint32_t)(unsigned char)*p; h *= 16777619u; }
    return h;
}

static void ckpt_disk_init(const GModel *m) {
    if (g_ckpt_dir[0]) return;
    const char *dir = getenv("COLI_CKPT_DIR");
    const char *snap = getenv("SNAP");
    if (dir && *dir) snprintf(g_ckpt_dir, sizeof(g_ckpt_dir), "%s", dir);
    else if (snap && *snap) snprintf(g_ckpt_dir, sizeof(g_ckpt_dir), "%s/.coli_ckpt", snap);
    else { g_ckpt_dir[0] = '-'; return; }        /* CLI: nessun disco */
    g_ckpt_fp = ckpt_fingerprint(m);
}

static void ckpt_disk_path(char *out, size_t size, int index) {
    snprintf(out, size, "%s/ckpt_%08x_%d.bin", g_ckpt_dir, g_ckpt_fp, index);
}

static void ckpt_disk_write(const GModel *m, int index) {
    if (!ckpt_disk_wanted()) return;
    ckpt_disk_init(m);
    if (!g_ckpt_dir[0] || g_ckpt_dir[0] == '-') return;
    PrefixCkpt *slot = &g_ckpt[index];
    if (!slot->ids || !slot->blob) return;
    char path[1200], tmp[1260];
    ckpt_disk_path(path, sizeof(path), index);
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        mkdir(g_ckpt_dir, 0755);
        f = fopen(tmp, "wb");
        if (!f) return;
    }
    const char magic[8] = { 'G','5','3','C','K','P','T','1' };
    const int32_t head[2] = { slot->len, slot->kind };
    const uint64_t bytes = (uint64_t)slot->bytes;
    int ok = fwrite(magic, 8, 1, f) == 1 &&
             fwrite(&g_ckpt_fp, sizeof(g_ckpt_fp), 1, f) == 1 &&
             fwrite(head, sizeof(head), 1, f) == 1 &&
             fwrite(&bytes, sizeof(bytes), 1, f) == 1 &&
             fwrite(slot->ids, sizeof(int), (size_t)slot->len, f) == (size_t)slot->len &&
             fwrite(slot->blob, 1, slot->bytes, f) == slot->bytes;
    ok = (fclose(f) == 0) && ok;
    if (!ok) { remove(tmp); return; }
    remove(path);
    if (rename(tmp, path)) { remove(tmp); return; }
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "CKPT disk write prefix=%d (%.0f MB) %s\n",
                slot->len, slot->bytes / 1e6, path);
}

static void ckpt_disk_load(const GModel *m) {
    if (g_ckpt_disk_loaded || !ckpt_disk_wanted()) return;
    g_ckpt_disk_loaded = 1;
    ckpt_disk_init(m);
    if (!g_ckpt_dir[0] || g_ckpt_dir[0] == '-') return;
    for (int i = 0; i < ckpt_slot_count(); i++) {
        if (g_ckpt[i].ids) continue;
        char path[1200];
        ckpt_disk_path(path, sizeof(path), i);
        FILE *f = fopen(path, "rb");
        if (!f) continue;
        char magic[8]; uint32_t fp = 0; int32_t head[2] = {0, 0}; uint64_t bytes = 0;
        PrefixCkpt *slot = &g_ckpt[i];
        int ok = fread(magic, 8, 1, f) == 1 && !memcmp(magic, "G53CKPT1", 8) &&
                 fread(&fp, sizeof(fp), 1, f) == 1 && fp == g_ckpt_fp &&
                 fread(head, sizeof(head), 1, f) == 1 &&
                 fread(&bytes, sizeof(bytes), 1, f) == 1 &&
                 head[0] > 0 && head[0] < (1 << 22) && (head[1] == 0 || head[1] == 1) &&
                 bytes > 0 && bytes < (1ull << 36);
        if (ok) {
            slot->len = head[0];
            slot->kind = head[1];
            slot->bytes = (size_t)bytes;
            slot->ids = malloc((size_t)slot->len * sizeof(int));
            slot->blob = malloc(slot->bytes);
            ok = slot->ids && slot->blob &&
                 fread(slot->ids, sizeof(int), (size_t)slot->len, f) == (size_t)slot->len &&
                 fread(slot->blob, 1, slot->bytes, f) == slot->bytes;
        }
        fclose(f);
        if (!ok) { ckpt_free(slot); remove(path); continue; }
        slot->used = ++g_ckpt_clock;
        if (getenv("GLM53_VERBOSE"))
            fprintf(stderr, "CKPT disk load prefix=%d (%.0f MB) kind=%d\n",
                    slot->len, slot->bytes / 1e6, slot->kind);
    }
}

/* La copia. Il vittimario e' quello di DeepSeek V4: prima uno slot vuoto, poi
 * il piu' vecchio di fine prompt, poi il piu' vecchio in assoluto -- il
 * prefisso e' l'ultimo a uscire. */
static void ckpt_store(const GModel *m, GSession *s, const int *ids, int len, int kind) {
    if (!ckpt_on() || len < ckpt_min_tokens() || len > s->filled) return;
    CkptSpan spans[GLM53_CKPT_MAX_SPANS];
    const size_t count = ckpt_spans(m, s, len, spans, GLM53_CKPT_MAX_SPANS);
    if (!count) return;
    const size_t bytes = ckpt_span_bytes(spans, count);
    const int slots = ckpt_slot_count();
    int victim = -1;
    for (int i = 0; i < slots && victim < 0; i++) if (!g_ckpt[i].ids) victim = i;
    for (int pass = 1; victim < 0 && pass >= 0; pass--)
        for (int i = 0; i < slots; i++)
            if (g_ckpt[i].kind >= pass &&
                (victim < 0 || g_ckpt[i].used < g_ckpt[victim].used))
                victim = i;
    if (victim < 0) return;
    PrefixCkpt *slot = &g_ckpt[victim];
    ckpt_free(slot);
    slot->ids = malloc((size_t)len * sizeof(int));
    slot->blob = malloc(bytes);
    if (!slot->ids || !slot->blob) { ckpt_free(slot); return; }
    size_t at = 0;
    const double t_copy = now_s();
    for (size_t i = 0; i < count; i++) {
        memcpy(slot->blob + at, spans[i].ptr, spans[i].bytes);
        at += spans[i].bytes;
    }
    const double copy_ms = (now_s() - t_copy) * 1000.0;
    memcpy(slot->ids, ids, (size_t)len * sizeof(int));
    slot->len = len;
    slot->bytes = bytes;
    slot->kind = kind;
    slot->used = ++g_ckpt_clock;
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "CKPT store prefix=%d %.0f MB kind=%d slot=%d "
                        "sync=%.0f ms copy=%.0f ms\n",
                len, bytes / 1e6, kind, victim, g_ckpt_sync_ms, copy_ms);
    if (kind == 0) ckpt_disk_write(m, victim);
    else if (ckpt_disk_wanted()) {
        /* Il file di questo slot descrive ormai un'altra cattura. */
        ckpt_disk_init(m);
        if (g_ckpt_dir[0] && g_ckpt_dir[0] != '-') {
            char path[1200];
            ckpt_disk_path(path, sizeof(path), victim);
            remove(path);
        }
    }
}

/* Il checkpoint piu' lungo che sia prefisso STRETTO di questa sequenza, rimesso
 * nella sessione (fresca) dello slot. Torna quante posizioni valgono, 0 se
 * nessuno serve. */
static int ckpt_restore(const GModel *m, KVSlot *slot, const int *sequence, int total) {
    if (!ckpt_on()) return 0;
    ckpt_disk_load(m);
    int best = -1, best_len = 0;
    for (int i = 0; i < ckpt_slot_count(); i++) {
        PrefixCkpt *ck = &g_ckpt[i];
        if (!ck->ids || ck->len <= best_len || ck->len >= total ||
            ck->len < ckpt_min_tokens()) continue;
        if (memcmp(ck->ids, sequence, (size_t)ck->len * sizeof(int))) continue;
        best = i; best_len = ck->len;
    }
    if (best < 0) return 0;
    GSession *s = slot->session;
    CkptSpan spans[GLM53_CKPT_MAX_SPANS];
    const size_t count = ckpt_spans(m, s, best_len, spans, GLM53_CKPT_MAX_SPANS);
    if (!count || ckpt_span_bytes(spans, count) != g_ckpt[best].bytes) {
        /* Una sessione con un'altra forma: il checkpoint non la riguarda. */
        fprintf(stderr, "CKPT layout mismatch prefix=%d: ignorato\n", best_len);
        return 0;
    }
    size_t at = 0;
    for (size_t i = 0; i < count; i++) {
        memcpy(spans[i].ptr, g_ckpt[best].blob + at, spans[i].bytes);
        at += spans[i].bytes;
    }
    s->filled = best_len;
    ckpt_sync_in(m, s);
    slot_remember(slot, sequence, best_len);
    g_ckpt[best].used = ++g_ckpt_clock;
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "CKPT hit prefix=%d slot=%d\n", best_len, best);
    return best_len;
}

/* Dove catturare durante QUESTO prefill fresco, o 0. La regola e' quella di
 * v4_ckpt_plan: il prefisso comune con il prompt fresco precedente e' il
 * prefisso di sistema stabile, mai il prompt intero, con almeno otto token di
 * coda da macinare dopo. */
static int ckpt_plan(const int *ids, int count) {
    int plan = 0;
    if (ckpt_on() && g_ckpt_prev_ids) {
        const int minimum = ckpt_min_tokens();
        int limit = g_ckpt_prev_len < count ? g_ckpt_prev_len : count;
        if (limit >= count) limit = count - 1;     /* mai tutto il prompt */
        int lcp = 0;
        while (lcp < limit && g_ckpt_prev_ids[lcp] == ids[lcp]) lcp++;
        if (lcp > count - 8) lcp = 0;              /* una coda minuscola non paga */
        if (lcp >= minimum) {
            plan = lcp;
            if (ckpt_have(ids, plan)) plan = 0;    /* gia' catturato */
        }
    }
    free(g_ckpt_prev_ids);
    g_ckpt_prev_ids = malloc((size_t)count * sizeof(int));
    if (g_ckpt_prev_ids) {
        memcpy(g_ckpt_prev_ids, ids, (size_t)count * sizeof(int));
        g_ckpt_prev_len = count;
    } else {
        g_ckpt_prev_len = 0;
    }
    return plan;
}

/* L'ottavo campo del SUBMIT: il gateway sa dove finisce la parte condivisa e
 * la manda in byte. Il motore ritokenizza quel pezzo e lo accetta SOLO se cade
 * su un confine di token e i suoi id sono davvero il prefisso del prompt --
 * altrimenti lo ignora e il piano trovera' il confine vero al prossimo prompt
 * fresco. */
static int ckpt_hint(Tok *tokenizer, const char *payload, int prefix_bytes,
                     const int *sequence, int total, int shared) {
    if (!ckpt_on() || prefix_bytes <= 0 || prefix_bytes >= total * 64) return 0;
    const int cap = total + 16;
    int *pids = malloc((size_t)cap * sizeof(int));
    if (!pids) return 0;
    const int pn = tok_encode(tokenizer, payload, prefix_bytes, pids, cap);
    int at = 0;
    if (pn >= ckpt_min_tokens() && pn > shared && pn <= total - 8 &&
        !memcmp(pids, sequence, (size_t)pn * sizeof(int)) &&
        !ckpt_have(sequence, pn))
        at = pn;
    free(pids);
    return at;
}

/* L'oracolo del percorso di servizio: i logit della prima posizione generata,
 * uno per richiesta. Il percorso CLI ha il teacher forcing; questo non lo
 * aveva, e una modifica al prefill si giudica dove gira davvero. */
static void ckpt_logit_dump(unsigned long long id, const float *logits, int vocab) {
    const char *dir = getenv("GLM53_LOGIT_DUMP");
    if (!dir || !*dir || !logits) return;
    mkdir(dir, 0755);
    char path[1200];
    snprintf(path, sizeof(path), "%s/req_%llu.f32", dir, id);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(logits, sizeof(float), (size_t)vocab, f);
    fclose(f);
}

/* Un'immagine annunciata prima della richiesta a cui appartiene.
 *
 * Il protocollo porta testo; le patch sono binarie e grosse, quindi arrivano
 * con un frame loro (IMAGE) subito prima del SUBMIT con lo stesso id. Il
 * motore la tiene da parte finche' quella richiesta non arriva, e se arriva
 * un'altra immagine prima la vecchia si butta: tenere quella sbagliata
 * vorrebbe dire rispondere sulla foto precedente senza dirlo. */
typedef struct {
    unsigned long long id;
    float *patches;
    int grid_h, grid_w;
} PendingImage;

static PendingImage g_pending = {0, NULL, 0, 0};

static void pending_clear(void) {
    free(g_pending.patches);
    g_pending.patches = NULL;
    g_pending.id = 0;
}

typedef struct {
    unsigned long long id;
    int slot, max_tokens;
    float temp, top_p;
    char *payload;
    int plen;
    /* P7: ottavo campo del SUBMIT -- il byte offset dove finisce la parte
     * condivisa del prompt. 0 = il gateway non l'ha mandato. */
    int prefix_bytes;
} ServeReq;

static int g_stop[16], g_nstop = 0;

/* Gli stop, con la regola che gli altri motori hanno imparato a caro prezzo.
 *
 * In modalita' batch resta solo il vero fine testo. I marcatori di ruolo sono
 * un confine che possiede il server Python, e tenerli come stop duri taglia la
 * generazione nel momento in cui il modello apre un blocco strumenti, perche'
 * il rumore dell'argmax a int4 preferisce l'id di uno stop al '<' giusto
 * (#401). Con SERVE=1 e basta, invece, `coli chat` non ha nessun filtro a
 * valle e gli stop gli servono tutti. */
static void arm_stops(const char *dir, Tok *tokenizer, int batched) {
    g_nstop = load_stops(dir, g_stop, 16);
    if (!batched && tokenizer)
        for (int id = 0; id < tokenizer->n_ids && g_nstop < 16; id++) {
            if (!tokenizer->id_special[id]) continue;
            int seen = 0;
            for (int i = 0; i < g_nstop; i++) if (g_stop[i] == id) seen = 1;
            if (!seen) g_stop[g_nstop++] = id;
        }
    fprintf(stderr, "[stop] %d token di stop:", g_nstop);
    for (int i = 0; i < g_nstop; i++) fprintf(stderr, " %d", g_stop[i]);
    fprintf(stderr, "%s\n", batched ? " (modalita' batch: solo fine testo)" : "");
}

static int is_stop(int token) {
    for (int i = 0; i < g_nstop; i++) if (token == g_stop[i]) return 1;
    return 0;
}

static void serve_line(const char *format, ...) {
    va_list args; va_start(args, format);
    vprintf(format, args);
    va_end(args);
    fflush(stdout);
}

/* Un token decodificato verso il server, con la sua lunghezza in byte. */
static void serve_data(unsigned long long id, const char *text, int n) {
    printf("DATA %llu %d\n", id, n);
    fwrite(text, 1, (size_t)n, stdout);
    putchar('\n');
    fflush(stdout);
}

/* Una richiesta intera, o 0 su EOF. Il payload si legge a byte contati, non a
 * righe: puo' contenerne. */
static int serve_read_req(ServeReq *q, char *verb, size_t verb_size) {
    char header[512];
    if (g_pushback_full) {
        /* L'intestazione che il drain di un turno precedente ha letto e messo
         * da parte; il suo corpo e' ancora nella pipa, subito qui sotto. */
        memcpy(header, g_pushback, sizeof(header));
        g_pushback_full = 0;
    } else if (!fgets(header, sizeof(header), stdin)) return 0;
    memset(q, 0, sizeof(*q));
    if (sscanf(header, "%15s", verb) != 1) { verb[0] = 0; return 1; }
    if (!strcmp(verb, "STOP") || !strcmp(verb, "CANCEL")) {
        sscanf(header, "%*s %llu", &q->id);
        return 1;
    }
    if (!strcmp(verb, "IMAGE")) {
        unsigned long long id; int bytes, grid_h, grid_w;
        if (sscanf(header, "IMAGE %llu %d %d %d", &id, &bytes, &grid_h, &grid_w) != 4 ||
            bytes < 0 || bytes > (1 << 28) || grid_h < 1 || grid_w < 1) {
            strcpy(verb, "BAD_FRAME");
            return 1;
        }
        pending_clear();
        float *patches = malloc((size_t)bytes);
        if (!patches || fread(patches, 1, (size_t)bytes, stdin) != (size_t)bytes) {
            free(patches);
            strcpy(verb, "BAD_FRAME");
            return 1;
        }
        (void)fgetc(stdin);                       /* il '\n' di chiusura */
        g_pending.id = id;
        g_pending.patches = patches;
        g_pending.grid_h = grid_h;
        g_pending.grid_w = grid_w;
        q->id = id;
        return 1;
    }
    if (strcmp(verb, "SUBMIT")) return 1;
    /* Sei campi come sempre; otto quando il gateway manda anche la lunghezza
     * del payload extra (grammatica/audio) e il suggerimento di prefisso. Il
     * suggerimento vale solo con xlen == 0, perche' a glm53 il gateway non
     * manda mai un frame extra e altrimenti andrebbe consumato prima. Un
     * motore vecchio ne legge sei e ignora il resto: il cambio lato gateway e'
     * sicuro contro il binario in servizio. */
    int xlen = 0, hint_bytes = 0;
    const int fields = sscanf(header, "SUBMIT %llu %d %d %d %f %f %d %d",
                              &q->id, &q->slot, &q->plen, &q->max_tokens,
                              &q->temp, &q->top_p, &xlen, &hint_bytes);
    if (fields < 6) {
        strcpy(verb, "BAD_FRAME");
        return 1;
    }
    if (fields == 8 && xlen == 0 && hint_bytes > 0) q->prefix_bytes = hint_bytes;
    if (q->plen < 0 || q->plen > (1 << 24)) { strcpy(verb, "BAD_FRAME"); return 1; }
    q->payload = malloc((size_t)q->plen + 1);
    if (!q->payload) { strcpy(verb, "BAD_FRAME"); return 1; }
    if (q->plen && fread(q->payload, 1, (size_t)q->plen, stdin) != (size_t)q->plen) {
        free(q->payload); q->payload = NULL;
        strcpy(verb, "BAD_FRAME");
        return 1;
    }
    q->payload[q->plen] = 0;
    int trailing = fgetc(stdin);
    (void)trailing;                        /* il '\n' di chiusura del frame */
    return 1;
}

/* Genera per una richiesta e chiude col suo DONE. */
static void serve_one(GModel *m, Tok *tokenizer, ServeReq *q) {
    if (!q->plen) { serve_line("ERROR %llu EMPTY_PROMPT\n", q->id); return; }
    g_temp = q->temp; g_topp = q->top_p > 0.0f ? q->top_p : 1.0f;

    if (q->slot < 0 || q->slot >= g_n_slots) {
        serve_line("ERROR %llu BAD_REQUEST\n", q->id);
        return;
    }
    KVSlot *slot = &g_slots[q->slot];
    const int room = g_slot_context;
    int *sequence = malloc((size_t)room * sizeof(int));
    if (!sequence) { serve_line("ERROR %llu BAD_REQUEST\n", q->id); return; }
    int total = tok_encode(tokenizer, q->payload, q->plen, sequence, room);
    const int prompt_tokens = total;
    if (!total) {
        free(sequence);
        serve_line("ERROR %llu EMPTY_PROMPT\n", q->id);
        return;
    }
    if (total >= room) {
        free(sequence);
        serve_line("ERROR %llu BAD_REQUEST\n", q->id);
        return;
    }

    const double started = now_s();
    const double s_attn = m->t_attn, s_ffn = m->t_ffn, s_disk = m->t_disk, s_head = m->t_head;
    const uint64_t s_fw = m->forwards;
    int emitted = 0, limited = 0;
    ServeCancel cancel = { q->id, 0, 0 };
    int cancel_pos = -1;                  /* >= 0: annullato DURANTE il prefill */
    const int budget = q->max_tokens > 0 ? q->max_tokens : 256;

    /* Quanto di questo prompt e' gia' nella sessione.
     *
     * Il numero che conta e' `filled`, non quanti token lo slot ricorda: sono
     * diversi, perche' l'ultimo token generato viene emesso al client ma non
     * rimacinato, quindi resta nella storia e non nella cache. Riprendere da un
     * punto che la sessione non ha davvero raggiunto darebbe al modello un
     * contesto sfasato di un token, che e' esattamente il tipo di errore che
     * non fa rumore.
     *
     * Si riusa solo se il prompt nuovo concorda su TUTTE le posizioni in cache
     * e ne ha almeno una in piu': lo stato ricorrente dei layer KDA non si
     * riavvolge, quindi una divergenza a meta' cache obbliga a rifare. */
    const int cached = slot->session ? slot->session->filled : 0;
    int shared = 0;
    if (cached > 0 && cached < total && slot_shared(slot, sequence, total) >= cached)
        shared = cached;
    if (shared <= 0) {
        slot_reset(m, slot);
        slot->session = session_open(m, room, (int)(slot - g_slots));
        shared = 0;
    }
    /* L'immagine annunciata per QUESTA richiesta, se c'e'. Il prefisso in
     * cache non la riguarda: la torre ha gia' dato i suoi embedding quando
     * quei token sono stati macinati, e i segnaposto stanno nel prompt. Se il
     * riuso salta i token immagine, gli embedding da consumare sono quelli
     * delle posizioni nuove, quindi il riuso si annulla quando c'e' un'immagine
     * -- costa un prefill in piu' ed e' l'unica cosa che non puo' sbagliare. */
    float *vision = NULL;
    int n_vision = 0;
    if (g_pending.patches && g_pending.id == q->id) {
        if (!m->has_vision) {
            pending_clear();
            free(sequence);
            serve_line("ERROR %llu BAD_REQUEST\n", q->id);
            return;
        }
        if (shared) {                             /* niente riuso con un'immagine */
            slot_reset(m, slot);
            slot->session = session_open(m, room, (int)(slot - g_slots));
            shared = 0;
        }
        vision = vision_encode(m, g_pending.patches, g_pending.grid_h,
                               g_pending.grid_w, &n_vision);
        pending_clear();
    }

    /* P7: il checkpoint del prefisso.
     *
     * Nell'ordine, e mai al posto del riuso di slot -- quello copre sempre
     * almeno quanto un checkpoint potrebbe:
     *   1. riuso di slot (sopra, invariato);
     *   2. se non ha coperto niente e non c'e' un'immagine, si rimette il
     *      checkpoint piu' lungo che sia prefisso stretto di questa sequenza;
     *   3. se non si e' riusato niente, si PIANIFICA una cattura, o dal
     *      prefisso comune col prompt fresco precedente o dal suggerimento del
     *      gateway;
     *   4. il prefill si spezza in due al confine e la copia si prende in
     *      mezzo.
     * Un'immagine annunciata annulla sia il ripristino sia la cattura, per la
     * stessa ragione per cui annulla il riuso: gli embedding della torre
     * appartengono a posizioni precise di questo prompt. */
    int ckpt_at = 0;
    const int has_image = (vision != NULL);
    if (ckpt_on() && !has_image) {
        if (shared == 0) shared = ckpt_restore(m, slot, sequence, total);
        if (shared == 0) {
            const char *why = "lcp";
            ckpt_at = ckpt_plan(sequence, total);
            if (!ckpt_at && q->prefix_bytes > 0 && q->prefix_bytes < q->plen) {
                ckpt_at = ckpt_hint(tokenizer, q->payload, q->prefix_bytes,
                                    sequence, total, shared);
                why = "hint";
            }
            if (ckpt_at && getenv("GLM53_VERBOSE"))
                fprintf(stderr, "CKPT plan prefix=%d src=%s\n", ckpt_at, why);
        }
    }

    const int reused = shared;
    float *logits;
    if (ckpt_at > shared) {
        /* Due chiamate invece di una. Il confine diventa un bordo di pezzo:
         * P2/P3/P4 sono indipendenti per riga dentro il pezzo, quindi il
         * risultato e' atteso identico -- e l'oracolo del gate dice se lo e'. */
        float *upto = forward_prefill(m, slot->session, sequence + shared,
                                      ckpt_at - shared, NULL, 0, 0, &cancel);
        free(upto);
        /* Un CANCEL a meta' della prima meta' NON lascia un checkpoint: la
         * copia descriverebbe `ckpt_at` posizioni che la sessione non ha. Si
         * cattura solo se il confine e' stato raggiunto esatto -- e allora si
         * cattura anche annullando, perche' quel lavoro e' gia' fatto e la
         * copia costa un memcpy. */
        if (cancel.done == ckpt_at - shared) {
            ckpt_sync_out(m, slot->session);
            ckpt_store(m, slot->session, sequence, ckpt_at, 0);
        }
        logits = NULL;
        if (!cancel.cancelled)
            logits = forward_prefill(m, slot->session, sequence + ckpt_at,
                                     total - ckpt_at, vision, n_vision, 0, &cancel);
    } else {
        logits = forward_prefill(m, slot->session, sequence + shared,
                                 total - shared, vision, n_vision, 0, &cancel);
    }
    if (cancel.cancelled) cancel_pos = slot->session->filled;
    /* La cattura a fine prompt (kind 1 di DeepSeek V4) e' spenta di default:
     * qui una copia intera per richiesta non e' gratis. */
    if (!cancel.cancelled && ckpt_on() && ckpt_end_wanted() && !has_image && total > reused &&
        total >= ckpt_min_tokens() && !ckpt_have(sequence, total)) {
        ckpt_sync_out(m, slot->session);
        ckpt_store(m, slot->session, sequence, total, 1);
    }
    /* L'oracolo del percorso di servizio, indipendente dal checkpoint: i logit
     * che scelgono il primo token generato. */
    ckpt_logit_dump(q->id, logits, m->c.vocab);
    GSession *session = slot->session;
    int rows = 1;
    for (int step = 0; !cancel.cancelled && step < budget; step++) {
        if (total >= room) { limited = 1; break; }
        int next = sample_token(logits + (size_t)(rows - 1) * m->c.vocab, m->c.vocab);
        free(logits);
        logits = NULL;
        if (is_stop(next)) break;
        sequence[total++] = next;
        emitted++;
        char piece[512];
        int written = tok_decode(tokenizer, &next, 1, piece, sizeof(piece) - 1);
        serve_data(q->id, piece, written);
        /* Il secondo punto di annullamento: il token appena emesso e' gia' nel
         * client e gia' nella sequenza, quindi lo stato dello slot resta
         * quello di una risposta piu' corta -- niente di speciale da riparare. */
        if (serve_cancel_pending(&cancel)) break;
        if (step + 1 == budget) { limited = 1; break; }
        logits = forward_span(m, session, &next, 1, NULL, 0);
        rows = 1;
    }
    free(logits);
    free(vision);
    /* La sessione resta allo slot per il turno dopo, con la sequenza che ha
     * davvero macinato: prompt piu' quello che ha generato.
     *
     * Annullato dentro il prefill, quello che ha macinato e' `filled` e basta:
     * ricordare l'intero prompt vorrebbe dire promettere al turno dopo delle
     * posizioni che la sessione non ha mai raggiunto, ed e' esattamente
     * l'errore silenzioso contro cui la nota su `filled` qui sopra mette in
     * guardia. Ricordando il pezzo vero, invece, un ritentativo identico
     * riparte da li' -- il prefill riprendibile di DeepSeek V4. */
    if (cancel_pos >= 0) slot_remember(slot, sequence, cancel_pos);
    else slot_remember(slot, sequence, total);
    if (cancel.cancelled && getenv("GLM53_VERBOSE")) {
        if (cancel_pos >= 0)
            fprintf(stderr, "CANCEL %llu at prefill pos=%d/%d\n",
                    q->id, cancel_pos, prompt_tokens);
        else
            fprintf(stderr, "CANCEL %llu at decode tok=%d\n", q->id, emitted);
    }
    const double elapsed = now_s() - started;
    /* Quanto prefisso lo slot ha risparmiato.
     *
     * Su STDERR, non nel protocollo. La specifica dice che un server ignora le
     * righe che non conosce, ma questo progetto ha scelto il contrario apposta
     * e lo mette per iscritto in un test: una riga sconosciuta uccide il
     * dispatcher, cosi' un motore non puo' parlare a un server che non lo
     * capisce. La regola vera e' quella del test, non quella del documento.
     *
     * E dietro GLM53_VERBOSE, perche' `coli chat` eredita lo stderr del server:
     * senza guardia questa riga compare a schermo dopo ogni risposta, sotto gli
     * occhi di chi voleva solo la risposta. */
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "REUSE %llu %d %d\n", q->id, reused, prompt_tokens);
    hits_emit(m);
    {
        const double disk = m->t_disk - s_disk, ffn = m->t_ffn - s_ffn;
        serve_line("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n", elapsed,
                   prompt_tokens, emitted, disk, 0.0, ffn > disk ? ffn - disk : 0.0,
                   m->t_attn - s_attn, m->t_head - s_head,
                   (unsigned long long)(m->forwards - s_fw));
    }
    serve_line("DONE %llu STAT %d %.2f %.1f %.1f %d %d\n", q->id, emitted,
               elapsed > 0 ? emitted / elapsed : 0.0,
               m->miss + m->hits ? 100.0 * m->hits / (double)(m->hits + m->miss) : 0.0,
               rss_gb(), prompt_tokens, limited);
    /* L'ack, DOPO il DONE e in quest'ordine, che e' il risultato di aver letto
     * cosa ne fa il gateway (openai_server.py, Engine._dispatch e generate()):
     *
     *   - la riga ERROR fa `pending.pop(id)`, quindi un DONE che la SEGUISSE
     *     verrebbe buttato e con lui il blocco `[req]` -- la richiesta
     *     annullata sparirebbe da owui_report.sh, che legge proprio quelle
     *     righe. Con il DONE davanti, `[req]` c'e' e porta prompt_tokens,
     *     ttft e i token consegnati prima dell'annullamento;
     *   - il ramo "done" di generate() alza ClientCancelled da solo quando ha
     *     mandato un CANCEL ("honored it at a token boundary and still framed a
     *     DONE"), quindi il DONE E' gia' un ack valido per il gateway;
     *   - la riga ERROR che segue trova `pending` vuoto e viene ignorata senza
     *     rumore. Resta perche' e' l'ack esplicito del protocollo, quello che
     *     scrivono qwen36 e kimi_k3 e quello che legge chi non e' il gateway
     *     (`coli chat` con SERVE=1, che di ClientCancelled non sa niente). */
    if (cancel.cancelled) serve_line("ERROR %llu CANCELLED\n", q->id);
    free(sequence);
}

/* --- Dashboard: Brain e Profile ------------------------------------------
 * Il server legge quattro righe dallo stdout del motore e nient'altro:
 *   EMAP rows cols hex   griglia (layer sparsi x esperti), una volta all'avvio
 *   HITS rows cols hex   bitmap degli esperti toccati nel turno, a fine turno
 *   PROF + 9 numeri      tempi per fase del turno
 * colibri.c le emette da mesi (telemetry.h); deepseek_v4.c anche. Qui non
 * c'erano, e le due schede su GLM-5.3-Flash restavano vuote: non c'era
 * niente da attivare, mancava l'emissione. Stesso formato byte per byte,
 * cosi' la dashboard non distingue i motori.
 *
 * Fasi che questo motore misura: disco (expert_read, muro del batch),
 * matmul esperti (ffn_layer meno il disco), attention (mla/kda), testa
 * (mv su lm_head). L'attesa asincrona non esiste qui: glm53 legge in modo
 * sincrono, quindi expert_wait_s e' 0 per costruzione, non per omissione. */
static void ehit_mark(GModel *m, int layer, int eid) {
    const Cfg *c = &m->c;
    if (!m->ehit) {
        m->ehit = calloc((size_t)c->n_layers, sizeof(uint8_t *));
        for (int i = 0; i < c->n_layers; i++) m->ehit[i] = calloc((size_t)c->n_experts, 1);
    }
    if (layer >= 0 && layer < c->n_layers && eid >= 0 && eid < c->n_experts) m->ehit[layer][eid] = 1;
}
static int dash_rows(const GModel *m) {
    int rows = 0;
    for (int i = m->c.first_dense; i < m->c.n_layers; i++) if (i >= m->layer_begin && i < m->layer_end) rows++;
    return rows;
}
static void emap_emit(GModel *m) {
    const Cfg *c = &m->c;
    const int rows = dash_rows(m), cols = c->n_experts;
    char *hex = malloc((size_t)rows * cols * 2 + 1); int w = 0;
    for (int i = c->first_dense; i < c->n_layers; i++) {
        if (i < m->layer_begin || i >= m->layer_end) continue;
        for (int e = 0; e < cols; e++) {
            int tier = 0;                       /* 0 = su disco, 1 = in RAM (cache) */
            if (m->ecache) { const LCache *cache = &m->ecache[i];
                for (int j = 0; j < cache->n; j++) if (cache->s[j].eid == e) { tier = 1; break; } }
            const int b = tier << 6;            /* nessun contatore di calore qui: heat = 0 */
            hex[w++] = "0123456789abcdef"[b >> 4]; hex[w++] = "0123456789abcdef"[b & 15];
        }
    }
    hex[w] = 0;
    serve_line("EMAP %d %d %s\n", rows, cols, hex); free(hex);
}
static void hits_emit(GModel *m) {
    const Cfg *c = &m->c;
    /* Un turno che non ha toccato esperti (tutto denso, o tutto riuso) emette
     * una bitmap a zero: la dashboard deve vedere QUESTO turno, non l'ultimo
     * che ha avuto hit. */
    if (!m->ehit) ehit_mark(m, -1, -1);
    const int rows = dash_rows(m), cols = c->n_experts, nb = (rows * cols + 7) / 8;
    uint8_t *bm = calloc((size_t)nb, 1); int bit = 0;
    for (int i = c->first_dense; i < c->n_layers; i++) {
        if (i < m->layer_begin || i >= m->layer_end) continue;
        for (int e = 0; e < cols; e++, bit++)
            if (m->ehit[i][e]) { bm[bit >> 3] |= (uint8_t)(1 << (bit & 7)); m->ehit[i][e] = 0; }
    }
    char *hex = malloc((size_t)nb * 2 + 1); int w = 0;
    for (int b = 0; b < nb; b++) { hex[w++] = "0123456789abcdef"[bm[b] >> 4]; hex[w++] = "0123456789abcdef"[bm[b] & 15]; }
    hex[w] = 0;
    serve_line("HITS %d %d %s\n", rows, cols, hex); free(hex); free(bm);
}

static void serve_loop(GModel *m, Tok *tokenizer) {
    coli_serve_binary_mode();
    setvbuf(stdin, NULL, _IONBF, 0);
    slots_init(m);
    serve_line("\x01\x01READY\x01\x01\n");
    serve_line("STAT 0 0.00 0.0 %.1f\n", rss_gb());
    /* La griglia va DOPO READY: il lettore di boot del server scarta tutto
     * fino al sentinel, e colibri.c fa lo stesso (READY, STAT, poi EMAP). */
    emap_emit(m);
    for (;;) {
        ServeReq q; char verb[16];
        if (!serve_read_req(&q, verb, sizeof(verb))) break;   /* EOF: si esce */
        if (!strcmp(verb, "SUBMIT")) {
            serve_one(m, tokenizer, &q);
            free(q.payload);
        } else if (!strcmp(verb, "IMAGE")) {
            /* annunciata: nessuna risposta, la si usa al SUBMIT che segue */
        } else if (!strcmp(verb, "BAD_FRAME")) {
            serve_line("ERROR %llu BAD_FRAME\n", q.id);
        } else if (!strcmp(verb, "CANCEL")) {
            /* Qui ci arriva solo un CANCEL per una richiesta che non e' piu' in
             * volo: quello per la richiesta in corso lo vede serve_cancel_pending
             * dentro il turno, e non arriva mai fin qui. */
            serve_line("ERROR %llu NOT_FOUND\n", q.id);
        }
        /* STOP e le righe che non riconosciamo si ignorano: la regola di
         * compatibilita' del protocollo vale in tutte e due le direzioni. */
    }
}

#ifndef GLM53_NO_MAIN
int main(int argc, char **argv) {
    const char *dir = NULL, *ids = NULL, *patch_file = NULL, *prompt_text = NULL;
    int greedy = 0, show_logits = 0, grid_h = 0, grid_w = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids = argv[++i];
        else if (!strcmp(argv[i], "--greedy") && i + 1 < argc) greedy = coli_arg_int(argv[++i], "--greedy");
        else if (!strcmp(argv[i], "--logits")) show_logits = 1;
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_text = argv[++i];
        else if (!strcmp(argv[i], "--patches") && i + 1 < argc) patch_file = argv[++i];
        else if (!strcmp(argv[i], "--grid") && i + 1 < argc &&
                 sscanf(argv[++i], "%dx%d", &grid_h, &grid_w) == 2) continue;
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            /* Capienza posizionale: e' cosi' che il gateway lancia ogni motore
             * (openai_server.py, Engine.__init__). Zero vuol dire "decidila
             * tu", che qui e' il budget misurato dalla RAM disponibile. */
            const int cap = coli_arg_int(argv[i], "cache/layer");
            if (cap > 0) g_cap_override = cap;
        }
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }
    /* SERVE=1 e SNAP=<dir>: il motore non e' piu' una CLI ma il capo di una
     * pipa, e il modello arriva dall'ambiente perche' e' cosi' che lo lancia
     * openai_server.py. */
    if (getenv("SERVE")) {
        const char *snap = getenv("SNAP");
        if (!snap) { fprintf(stderr, "SERVE senza SNAP: non so quale modello aprire\n"); return 2; }
        GModel served;
        memset(&served, 0, sizeof(served));
        model_load(&served, snap);
        Tok serve_tok;
        char tokenizer_path[1024];
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", snap);
        tok_load(&serve_tok, tokenizer_path);
        const char *batch = getenv("SERVE_BATCH");
        arm_stops(snap, &serve_tok, batch && atoi(batch));
        serve_loop(&served, &serve_tok);
        tok_free(&serve_tok);
        return 0;
    }

    if (!dir || (!ids && !prompt_text)) {
        fprintf(stderr, "uso: %s --model DIR (--prompt TESTO | --ids a,b,c)\n"
                        "         [--greedy N] [--logits] [--patches FILE.f32 --grid HxW]\n",
                argv[0]);
        return 2;
    }
    if (ids && prompt_text) {
        fprintf(stderr, "--prompt e --ids sono due modi di dire la stessa cosa\n");
        return 2;
    }
    if (!!patch_file != (grid_h > 0 && grid_w > 0)) {
        fprintf(stderr, "--patches e --grid vanno insieme\n");
        return 2;
    }
    int capacity = 1024, count = 0;
    int *tokens = malloc((size_t)capacity * sizeof(int));
    Tok tokenizer; int has_tokenizer = 0;

    if (prompt_text) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
        tok_load(&tokenizer, path);
        has_tokenizer = 1;
        const int room = 1 << 16;
        tokens = realloc(tokens, (size_t)room * sizeof(int));
        capacity = room;
        count = tok_encode(&tokenizer, prompt_text, (int)strlen(prompt_text), tokens, room);
    } else {
        for (const char *p = ids; *p; ) {
            char *end;
            long value = strtol(p, &end, 10);
            if (end == p) break;
            if (count == capacity) tokens = realloc(tokens, (size_t)(capacity *= 2) * sizeof(int));
            tokens[count++] = (int)value;
            p = (*end == ',') ? end + 1 : end;
        }
    }
    if (!count) { fprintf(stderr, "nessun token nel prompt\n"); return 2; }

    GModel model;
    memset(&model, 0, sizeof(model));     /* contatori e puntatori opzionali */
    const double load_start = now_s();
    model_load(&model, dir);
    const double load_seconds = now_s() - load_start;
    if (getenv("GLM53_VERBOSE")) cfg_report(&model.c);

    float *vision = NULL;
    int n_vision = 0;
    if (patch_file) {
        const ColiVisionConfig *vc = &model.vision.config;
        if (!model.has_vision) {
            fprintf(stderr, "--patches ma il checkpoint non porta la torre vision\n");
            return 2;
        }
        const size_t per_patch = (size_t)vc->in_channels * vc->temporal * vc->patch * vc->patch;
        const size_t wanted = (size_t)grid_h * grid_w * per_patch;
        FILE *f = fopen(patch_file, "rb");
        if (!f) { fprintf(stderr, "non apro %s\n", patch_file); return 2; }
        float *patches = malloc(wanted * sizeof(float));
        if (!patches) { fprintf(stderr, "OOM sulle patch\n"); return 2; }
        size_t got = fread(patches, sizeof(float), wanted, f);
        /* Una patch corta darebbe comunque un'uscita, con la coda letta da
         * memoria non inizializzata: meglio fermarsi e dire di quanto. */
        if (got != wanted) {
            fprintf(stderr, "%s: %zu float su %zu attesi per una griglia %dx%d\n",
                    patch_file, got, wanted, grid_h, grid_w);
            return 2;
        }
        fclose(f);
        vision = vision_encode(&model, patches, grid_h, grid_w, &n_vision);
        free(patches);
        printf("vision_tokens %d\n", n_vision);
    }

    /* Una sola sessione per tutta la generazione: il prompt si prefilla una
     * volta e ogni token dopo costa un token, non tutto il prefisso. */
    GSession *session = session_open(&model, count + (greedy > 0 ? greedy : 0) + 1, 0);
    const double prefill_start = now_s();
    float *logits = forward_prefill(&model, session, tokens, count, vision, n_vision, 1, NULL);
    if (getenv("GLM53_VERBOSE"))
        fprintf(stderr, "caricamento %.1fs, prefill %d token in %.1fs\n",
                load_seconds, count, now_s() - prefill_start);
    printf("teacher_forcing");
    for (int t = 0; t < count; t++)
        printf(" %d", argmax(logits + (size_t)t * model.c.vocab, model.c.vocab));
    printf("\n");
    if (show_logits) {
        printf("last_logits");
        const float *last = logits + (size_t)(count - 1) * model.c.vocab;
        for (int v = 0; v < model.c.vocab; v++) printf(" %.9g", last[v]);
        printf("\n");
    }
    if (greedy > 0) {
        int stops[8];
        const int n_stops = has_tokenizer ? load_stops(dir, stops, 8) : 0;
        /* Il costo per token misurato qui e non ricavato per sottrazione dal
         * tempo totale: caricamento e prefill costano quanto costano, e
         * confonderli col decode ha gia' fatto sbagliare un confronto. */
        const double decode_start = now_s();
        if (optime_on()) optime_reset();   /* the per-op table describes decode only */
        int produced = 0;
        /* `rows` dice quante righe ha l'ultimo blocco di logit: il prefill ne
         * restituisce una per posizione, un passo incrementale una sola. In
         * entrambi i casi quella che serve e' l'ultima. */
        int rows = count;
        if (!has_tokenizer) printf("greedy");
        for (int step = 0; step < greedy; step++) {
            int next = argmax(logits + (size_t)(rows - 1) * model.c.vocab, model.c.vocab);
            free(logits);
            logits = NULL;
            int done = 0;
            for (int i = 0; i < n_stops; i++) if (next == stops[i]) done = 1;
            if (done) break;
            if (has_tokenizer) {
                char piece[512];
                int written = tok_decode(&tokenizer, &next, 1, piece, sizeof(piece) - 1);
                fwrite(piece, 1, (size_t)written, stdout);
                fflush(stdout);
            } else {
                printf(" %d", next);
            }
            /* Il token nuovo non e' mai un segnaposto immagine: la torre ha
             * gia' dato i suoi embedding durante il prefill. */
            logits = forward_span(&model, session, &next, 1, NULL, 0);
            rows = 1;
            produced++;
        }
        printf("\n");
        const double spent = now_s() - decode_start;
        if (produced)
            printf("decode %d token in %.1fs = %.3f tok/s (%.1f s/token)\n",
                   produced, spent, produced / spent, spent / produced);
    }
    if (has_tokenizer) tok_free(&tokenizer);
    /* Contatori della cache esperti: servono a un test per accorgersi se lo
     * streaming e' stato aggirato invece che esercitato. */
    /* Le graffe mancavano: la riga "experts hits" e' SEMPRE stampata, ed e'
     * quella che il test legge. Comportamento invariato, resa esplicita. */
    if (model.streaming && g_map_active)
        printf("[MAP] serviti da mmap %ld, copiati %ld\n", g_map_serve, g_map_copy);
    printf("experts hits %ld miss %ld bytes %llu\n",
           model.hits, model.miss, (unsigned long long)model.ebytes);
    free(logits);
    session_close(&model, session);
    free(vision);
    free(tokens);
    return 0;
}
#endif /* GLM53_NO_MAIN */

/* ================= segment adapter =================
 *
 * Un segment esegue un INTERVALLO di layer su attivazioni, non un modello
 * intero su token: entrano H flussi residui per posizione, escono gli stessi
 * flussi dopo quei layer. Embedding e testa stanno agli estremi della catena e
 * non in mezzo, quindi qui non si caricano proprio.
 *
 * Lo stato della conversazione appartiene alla sessione, i pesi al motore. Un
 * ospite puo' tenere piu' motori nello stesso processo, quindi qui non ci sono
 * variabili globali e ogni allocazione ha il suo rilascio.
 */
#ifdef COLI_SEGMENT_ADAPTER
#include <pthread.h>
#include "segment_runtime.h"
#include "segment_adapters.h"
#include "segment_adapter_internal.h"

typedef struct {
    GModel model;
    uint32_t layer_begin, layer_end, context_tokens, state_width;
    pthread_mutex_t run_lock;
} Glm53SegmentEngine;

typedef struct {
    Glm53SegmentEngine *engine;
    GSession *session;
    uint32_t context_tokens, position;
} Glm53SegmentSession;

/* G12 sync points. While COLI_KDA_GPU is on the recurrence lives on dev0 and
 * st->kda_state / st->kda_window are STALE -- reading them back every token
 * would cost ~5 ms/token over PCIe, a third of what the chain saves. These two
 * are the only places the host copies are made to agree with the device, and
 * they are exactly the two directions segment migration needs. */
static void glm53_kda_sync_out(Glm53SegmentSession *session) {
#ifdef COLI_VULKAN
    const Cfg *c = &session->engine->model.c;
    for (uint32_t i = session->engine->layer_begin; i < session->engine->layer_end; i++) {
        GLayerState *st = &session->session->layer[i];
        if (!c->is_full[i] && c->kda_proj && st->kda_gpu)
            coli_vk_kda_sync((int)i, session->session->slot, st->kda_state, st->kda_window);
    }
#else
    (void)session;
#endif
}
static void glm53_kda_sync_in(Glm53SegmentSession *session) {
#ifdef COLI_VULKAN
    const Cfg *c = &session->engine->model.c;
    for (uint32_t i = session->engine->layer_begin; i < session->engine->layer_end; i++) {
        GLayerState *st = &session->session->layer[i];
        if (!c->is_full[i] && c->kda_proj && st->kda_gpu)
            coli_vk_kda_upload((int)i, session->session->slot, st->kda_state, st->kda_window);
    }
#else
    (void)session;
#endif
}

/* I pezzi di stato che uno snapshot deve portarsi dietro, nell'ordine in cui
 * si scrivono. Un layer DSA tiene il latente MLA e le due file dell'indexer,
 * un layer KDA la ricorrenza e la finestra della convoluzione. */
static size_t glm53_segment_spans(const Glm53SegmentEngine *engine,
                                  const Glm53SegmentSession *session,
                                  ColiSegmentStateSpan *spans, size_t capacity) {
    const Cfg *c = &engine->model.c;
    const size_t cap = session->context_tokens;
    size_t count = 0;
    for (uint32_t i = engine->layer_begin; i < engine->layer_end; i++) {
        GLayerState *st = &session->session->layer[i];
        if (c->is_full[i]) {
            if (count + 3 > capacity) return 0;
            spans[count++] = (ColiSegmentStateSpan){ st->latent, cap * (size_t)c->kv_lora * sizeof(float) };
            spans[count++] = (ColiSegmentStateSpan){ st->ikeys, cap * (size_t)c->index_hd * sizeof(float) };
            spans[count++] = (ColiSegmentStateSpan){ st->igates, cap * (size_t)c->index_hd * sizeof(float) };
        } else if (c->kda_proj) {
            if (count + 2 > capacity) return 0;
            spans[count++] = (ColiSegmentStateSpan){
                st->kda_state, (size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float) };
            spans[count++] = (ColiSegmentStateSpan){
                st->kda_window, 3u * (size_t)c->kda_proj * c->conv_k * sizeof(float) };
        }
    }
    return count;
}

#define GLM53_SEGMENT_MAX_SPANS 512

static int glm53_segment_engine_open(void **engine_impl,
                                     ColiSegmentCapabilities *capabilities,
                                     const ColiSegmentEngineOptions *options,
                                     char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options || !options->model_dir)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment needs a model directory");
    Glm53SegmentEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_segment_adapter_error(error, error_size,
                                          "out of memory opening GLM-5.3 Segment");

    /* Solo i layer chiesti: e' quello che promette RANGE_NATIVE, e caricare il
     * resto vorrebbe dire tenere in RAM i pesi che macina un'altra macchina. */
    const int begin = (int)options->layer_begin;
    const int end = options->layer_end ? (int)options->layer_end : -1;
    model_load_range(&engine->model, options->model_dir, begin, end, 0);
    engine->layer_begin = (uint32_t)engine->model.layer_begin;
    engine->layer_end = (uint32_t)engine->model.layer_end;
    engine->context_tokens = options->context_tokens ? options->context_tokens : 4096u;
    engine->state_width = (uint32_t)(engine->model.c.hc_mult * engine->model.c.hidden);
    pthread_mutex_init(&engine->run_lock, NULL);

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = (uint32_t)sizeof(*capabilities);
    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_SNAPSHOT | COLI_SEGMENT_CAP_RANGE_NATIVE |
                          COLI_SEGMENT_CAP_MULTI_SESSION | COLI_SEGMENT_CAP_CPU;
    coli_segment_capability_string(capabilities->engine_id,
                                   sizeof(capabilities->engine_id), "glm53");
    coli_segment_capability_string(capabilities->state_schema,
                                   sizeof(capabilities->state_schema),
                                   "glm53/mla-latent-kda-conv-dsa-f32-v1");
    /* Ogni leva che cambia i numeri deve comparire qui: due ospiti costruiti
     * diversamente si scambierebbero snapshot incompatibili in silenzio. */
    snprintf(capabilities->numeric_class, sizeof(capabilities->numeric_class),
             "glm53/q%d-i4g64-experts/f32/cpu-v1", glm53_dense_bits());
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = engine->state_width;
    capabilities->max_batch_rows = engine->context_tokens;
    capabilities->max_context_tokens = engine->context_tokens;
    /* Il totale del modello, non quanti ne possiede questo segment: il
     * runtime ci confronta layer_end, che e' un indice assoluto. */
    capabilities->num_layers = (uint32_t)engine->model.c.n_layers;
    *engine_impl = engine;
    return 0;
}

static void glm53_segment_engine_destroy(void *engine_impl) {
    Glm53SegmentEngine *engine = (Glm53SegmentEngine *)engine_impl;
    if (!engine) return;
    pthread_mutex_destroy(&engine->run_lock);
    model_release(&engine->model);
    free(engine);
}

static int glm53_segment_session_create(void *engine_impl, void **session_impl,
                                        const ColiSegmentSessionOptions *options,
                                        char *error, size_t error_size) {
    Glm53SegmentEngine *engine = (Glm53SegmentEngine *)engine_impl;
    if (!engine || !session_impl)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment session needs an engine");
    Glm53SegmentSession *session = calloc(1, sizeof(*session));
    if (!session)
        return coli_segment_adapter_error(error, error_size,
                                          "out of memory creating a GLM-5.3 session");
    session->engine = engine;
    session->context_tokens = (options && options->context_tokens)
                              ? options->context_tokens : engine->context_tokens;
    if (session->context_tokens > engine->context_tokens)
        session->context_tokens = engine->context_tokens;
    session->session = session_open(&engine->model, (int)session->context_tokens, 0);
    session->position = 0;
    *session_impl = session;
    return 0;
}

static void glm53_segment_session_destroy(void *session_impl) {
    Glm53SegmentSession *session = (Glm53SegmentSession *)session_impl;
    if (!session) return;
    session_close(&session->engine->model, session->session);
    free(session);
}

static int glm53_segment_session_run(void *session_impl,
                                     const ColiSegmentRunRequest *request,
                                     char *error, size_t error_size) {
    Glm53SegmentSession *session = (Glm53SegmentSession *)session_impl;
    if (!session || !request)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment run needs a request");
    if (request->position != session->position)
        return coli_segment_adapter_error(
            error, error_size, "GLM-5.3 Segment requires contiguous positions");
    if (request->should_cancel && request->should_cancel(request->cancel_user_data))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment run cancelled");
    Glm53SegmentEngine *engine = session->engine;
    const size_t width = engine->state_width;
    size_t cells;
    if (coli_segment_size_mul(request->rows, width, &cells))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 boundary state overflows");
    if (request->input_bytes < cells * sizeof(float) ||
        request->output_bytes < cells * sizeof(float))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment buffers are too small");
    if (session->position + request->rows > session->context_tokens)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment context is full");

    float *streams = malloc(cells * sizeof(float));
    float *next = malloc(cells * sizeof(float));
    if (!streams || !next) {
        free(streams); free(next);
        return coli_segment_adapter_error(error, error_size,
                                          "out of memory running GLM-5.3 Segment");
    }
    memcpy(streams, request->input, cells * sizeof(float));

    pthread_mutex_lock(&engine->run_lock);
    float *result = run_layers(&engine->model, session->session, streams, next,
                               (int)request->rows, (int)session->position,
                               (int)engine->layer_begin, (int)engine->layer_end);
    session->session->filled = (int)(session->position + request->rows);
    pthread_mutex_unlock(&engine->run_lock);

    memcpy(request->output, result, cells * sizeof(float));
    free(streams); free(next);
    session->position += request->rows;
    return 0;
}

static int glm53_segment_session_snapshot(void *session_impl,
                                          ColiSegmentWriteFn write_fn,
                                          void *write_user_data,
                                          char *error, size_t error_size) {
    Glm53SegmentSession *session = (Glm53SegmentSession *)session_impl;
    if (!session || !write_fn)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment snapshot needs a sink");
    glm53_kda_sync_out(session);   /* G12: device -> host before the snapshot reads */
    ColiSegmentStateSpan spans[GLM53_SEGMENT_MAX_SPANS];
    const size_t count = glm53_segment_spans(session->engine, session, spans,
                                             GLM53_SEGMENT_MAX_SPANS);
    size_t bytes = 0;
    if (coli_segment_spans_size(spans, count, &bytes))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment state overflows");
    ColiSegmentSnapshotHeader header;
    coli_segment_snapshot_header_init(&header, "glm53", session->engine->layer_begin,
                                      session->engine->layer_end,
                                      session->context_tokens, session->position,
                                      bytes,
                                      coli_segment_spans_hash(spans, count));
    if (coli_segment_stream_write(write_fn, write_user_data, &header, sizeof(header),
                                  error, error_size))
        return -1;
    return coli_segment_spans_write(spans, count, write_fn, write_user_data,
                                    error, error_size);
}

static int glm53_segment_session_restore(void *session_impl,
                                         ColiSegmentReadFn read_fn,
                                         void *read_user_data,
                                         char *error, size_t error_size) {
    Glm53SegmentSession *session = (Glm53SegmentSession *)session_impl;
    if (!session || !read_fn)
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment restore needs a source");
    ColiSegmentStateSpan spans[GLM53_SEGMENT_MAX_SPANS];
    const size_t count = glm53_segment_spans(session->engine, session, spans,
                                             GLM53_SEGMENT_MAX_SPANS);
    size_t bytes = 0;
    if (coli_segment_spans_size(spans, count, &bytes))
        return coli_segment_adapter_error(error, error_size,
                                          "GLM-5.3 Segment state overflows");
    ColiSegmentSnapshotHeader header;
    if (coli_segment_stream_read(read_fn, read_user_data, &header, sizeof(header),
                                 error, error_size))
        return -1;
    if (coli_segment_snapshot_header_valid(&header, "glm53",
                                           session->engine->layer_begin,
                                           session->engine->layer_end,
                                           session->context_tokens, bytes,
                                           error, error_size))
        return -1;
    if (coli_segment_spans_restore(spans, count, header.payload_hash, read_fn,
                                   read_user_data, error, error_size))
        return -1;
    glm53_kda_sync_in(session);    /* G12: host -> device after the restore writes */
    session->position = header.position;
    session->session->filled = (int)header.position;
    return 0;
}

static const ColiSegmentAdapter glm53_segment_adapter = {
    sizeof(ColiSegmentAdapter), COLI_SEGMENT_ABI_VERSION, "glm53",
    glm53_segment_engine_open, glm53_segment_engine_destroy,
    glm53_segment_session_create, glm53_segment_session_destroy,
    glm53_segment_session_run, glm53_segment_session_snapshot,
    glm53_segment_session_restore, {0}
};

int coli_glm53_segment_adapter_register(void) {
    return coli_segment_adapter_register(&glm53_segment_adapter);
}
#endif /* COLI_SEGMENT_ADAPTER */

/* ================= edge adapter =================
 *
 * I due capi della catena: da una parte i token diventano stato, dall'altra lo
 * stato torna token. In mezzo ci sono i segment, che di token non sanno nulla.
 *
 * Qui servono l'embedding, la norma finale, la testa e il tokenizzatore, e NON
 * serve nessun layer: si carica esattamente quello.
 */
#ifdef COLI_EDGE_ADAPTER
#include "edge_runtime.h"
#include "edge_adapters.h"
#include "edge_adapter_internal.h"

typedef struct {
    GModel model;
    Tok tokenizer;
    int has_tokenizer;
    uint32_t state_width;
} Glm53EdgeEngine;

static int glm53_edge_engine_open(void **engine_impl,
                                  ColiEdgeCapabilities *capabilities,
                                  const ColiEdgeEngineOptions *options,
                                  char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options || !options->model_dir)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge needs a model directory");
    Glm53EdgeEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory opening GLM-5.3 Edge");
    /* Nessun layer: i capi della catena non ne eseguono. */
    model_load_range(&engine->model, options->model_dir, 0, 0, 1);
    engine->state_width = (uint32_t)(engine->model.c.hc_mult * engine->model.c.hidden);

    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", options->model_dir);
    FILE *probe = fopen(path, "rb");
    if (probe) {
        fclose(probe);
        tok_load(&engine->tokenizer, path);
        engine->has_tokenizer = 1;
    }

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = (uint32_t)sizeof(*capabilities);
    capabilities->abi_version = COLI_EDGE_ABI_VERSION;
    /* Le bandiere devono corrispondere ai puntatori che l'adapter fornisce:
     * il runtime rifiuta chi dichiara meno o piu' di quello che ha. Il
     * tokenizzatore puo' mancare in un checkpoint di sola matematica. */
    capabilities->flags = COLI_EDGE_CAP_CPU | COLI_EDGE_CAP_GREEDY |
                          COLI_EDGE_CAP_LOGITS;
    if (engine->has_tokenizer)
        capabilities->flags |= COLI_EDGE_CAP_TOKENIZE | COLI_EDGE_CAP_DETOKENIZE;
    coli_edge_capability_string(capabilities->engine_id,
                                sizeof(capabilities->engine_id), "glm53");
    coli_edge_capability_string(capabilities->state_schema,
                                sizeof(capabilities->state_schema),
                                "glm53/mla-latent-kda-conv-dsa-f32-v1");
    snprintf(capabilities->numeric_class, sizeof(capabilities->numeric_class),
             "glm53/q%d-i4g64-experts/f32/cpu-v1", glm53_dense_bits());
    coli_edge_capability_string(capabilities->tokenizer_class,
                                sizeof(capabilities->tokenizer_class),
                                engine->has_tokenizer ? "glm53/bpe" : "none");
    capabilities->state_dtype = COLI_EDGE_DTYPE_F32;
    capabilities->state_width = engine->state_width;
    capabilities->vocab_size = (uint32_t)engine->model.c.vocab;
    capabilities->max_batch_rows = 1024;
    capabilities->max_context_tokens = 0;   /* lo decide il chiamante */
    capabilities->num_layers = (uint32_t)engine->model.c.n_layers;
    capabilities->bos_token_id = -1;
    capabilities->eos_token_id = -1;
    *engine_impl = engine;
    return 0;
}

static void glm53_edge_engine_destroy(void *engine_impl) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine) return;
    if (engine->has_tokenizer) tok_free(&engine->tokenizer);
    model_release(&engine->model);
    free(engine);
}

static int glm53_edge_tokenize(void *engine_impl, const char *text,
                               size_t text_bytes, int32_t *token_ids,
                               size_t token_capacity, size_t *token_count,
                               char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !engine->has_tokenizer)
        return coli_edge_adapter_error(error, error_size,
                                       "this GLM-5.3 checkpoint carries no tokenizer");
    if (!text || !token_count || text_bytes > (size_t)INT_MAX)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge tokenize got bad arguments");
    /* Giro di dimensionamento: il chiamante chiede quanti token servono prima
     * di allocare, e passa un buffer vuoto. Va risposto, non rifiutato.
     * Il tetto e' un token per byte, che con un vocabolario a livello di byte
     * e' il caso peggiore vero e non una stima. */
    const int ceiling = (int)text_bytes + 1;
    const int room = (token_ids && token_capacity) ? (int)token_capacity : ceiling;
    int *scratch = malloc((size_t)(room > 0 ? room : 1) * sizeof(int));
    if (!scratch)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory tokenizing");
    const int produced = tok_encode(&engine->tokenizer, text, (int)text_bytes,
                                    scratch, room);
    if (token_ids && token_capacity)
        for (int i = 0; i < produced && (size_t)i < token_capacity; i++)
            token_ids[i] = (int32_t)scratch[i];
    free(scratch);
    *token_count = (size_t)produced;
    return 0;
}

static int glm53_edge_detokenize(void *engine_impl, const int32_t *token_ids,
                                 size_t token_count, char *text,
                                 size_t text_capacity, size_t *text_bytes,
                                 char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !engine->has_tokenizer)
        return coli_edge_adapter_error(error, error_size,
                                       "this GLM-5.3 checkpoint carries no tokenizer");
    if (!token_ids || !text_bytes || token_count > (size_t)INT_MAX)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge detokenize got bad arguments");
    int *scratch = malloc(token_count ? token_count * sizeof(int) : sizeof(int));
    if (!scratch)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory detokenizing");
    for (size_t i = 0; i < token_count; i++) scratch[i] = (int)token_ids[i];
    /* Anche qui il primo giro puo' essere solo per la misura. Un token del
     * vocabolario piu' lungo limita quanto puo' venire fuori. */
    char *sink = NULL;
    size_t room = text_capacity;
    if (!text || !text_capacity) {
        room = token_count * 512u + 1u;
        sink = malloc(room);
        if (!sink) {
            free(scratch);
            return coli_edge_adapter_error(error, error_size,
                                           "out of memory detokenizing");
        }
    }
    const int written = tok_decode(&engine->tokenizer, scratch, (int)token_count,
                                   sink ? sink : text, (int)room);
    free(scratch);
    free(sink);
    *text_bytes = (size_t)written;
    return 0;
}

/* Token -> stato iniziale: l'embedding entra replicato in ognuno degli
 * hc_mult flussi residui, che e' esattamente come parte il passaggio. */
static int glm53_edge_embed(void *engine_impl, const ColiEdgeEmbedRequest *request,
                            char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !request || !request->token_ids || !request->output)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge embed got bad arguments");
    const Cfg *c = &engine->model.c;
    const size_t width = engine->state_width;
    if (request->output_bytes < (size_t)request->rows * width * sizeof(float))
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge embed output is too small");
    float *output = (float *)request->output;
    for (uint32_t row = 0; row < request->rows; row++) {
        const int32_t token = request->token_ids[row];
        if (token < 0 || token >= c->vocab)
            return coli_edge_adapter_error(error, error_size,
                                           "GLM-5.3 token ID is out of range");
        const float *source = engine->model.embed + (size_t)token * c->hidden;
        float *state = output + (size_t)row * width;
        for (int h = 0; h < c->hc_mult; h++)
            memcpy(state + (size_t)h * c->hidden, source,
                   (size_t)c->hidden * sizeof(float));
    }
    return 0;
}

/* Lo stato finale come logit interi, senza scegliere.
 *
 * E' la stessa trasformazione di glm53_edge_select fino all'ultimo passo: chi
 * vuole campionare a modo suo ha bisogno della distribuzione, non del vincitore,
 * e temperatura e generatore sono politica di chi serve, non matematica del
 * modello. Le due funzioni condividono la chiusura proprio perche' non possano
 * scostarsi: un argmax che guardasse numeri diversi da questi darebbe un token
 * che la distribuzione non spiega. */
static int glm53_edge_final(const Glm53EdgeEngine *engine, const float *streams,
                            float *logits, float *collapsed, float *normed) {
    const Cfg *c = &engine->model.c;
    const int H = c->hc_mult, D = c->hidden;
    for (int d = 0; d < D; d++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++) sum += streams[(size_t)h * D + d];
        collapsed[d] = sum / H;
    }
    rms(normed, collapsed, engine->model.final_norm, D, c->eps);
    mv(logits, &engine->model.head, normed);
    return 0;
}

static int glm53_edge_logits(void *engine_impl, const ColiEdgeLogitsRequest *request,
                             char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !request || !request->input || !request->logits)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge logits got bad arguments");
    const Cfg *c = &engine->model.c;
    const size_t width = engine->state_width;
    if (request->input_bytes < (size_t)request->rows * width * sizeof(float) ||
        request->logits_capacity < (size_t)request->rows * (size_t)c->vocab)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge logits buffers are too small");
    float *collapsed = malloc((size_t)c->hidden * sizeof(float));
    float *normed = malloc((size_t)c->hidden * sizeof(float));
    if (!collapsed || !normed) {
        free(collapsed); free(normed);
        return coli_edge_adapter_error(error, error_size, "out of memory for logits");
    }
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(collapsed); free(normed);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM-5.3 Edge logits cancelled");
        }
        glm53_edge_final(engine, input + (size_t)row * width,
                         request->logits + (size_t)row * c->vocab, collapsed, normed);
    }
    free(collapsed); free(normed);
    return 0;
}

/* Stato finale -> token. I flussi si richiudono con una media NON pesata, poi
 * la norma finale e la testa: le stesse tre righe con cui finisce il
 * passaggio, perche' un capo che chiudesse diversamente darebbe token diversi
 * dallo stesso stato. */
static int glm53_edge_select(void *engine_impl, const ColiEdgeSelectRequest *request,
                             char *error, size_t error_size) {
    Glm53EdgeEngine *engine = (Glm53EdgeEngine *)engine_impl;
    if (!engine || !request || !request->input || !request->token_ids)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge select got bad arguments");
    const Cfg *c = &engine->model.c;
    const int H = c->hc_mult, D = c->hidden;
    const size_t width = engine->state_width;
    if (request->input_bytes < (size_t)request->rows * width * sizeof(float) ||
        request->token_capacity < request->rows)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM-5.3 Edge select buffers are too small");
    float *collapsed = malloc((size_t)D * sizeof(float));
    float *normed = malloc((size_t)D * sizeof(float));
    float *logits = malloc((size_t)c->vocab * sizeof(float));
    if (!collapsed || !normed || !logits) {
        free(collapsed); free(normed); free(logits);
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory selecting");
    }
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(collapsed); free(normed); free(logits);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM-5.3 Edge select cancelled");
        }
        glm53_edge_final(engine, input + (size_t)row * width, logits, collapsed, normed);
        const int best = argmax(logits, c->vocab);
        request->token_ids[row] = (int32_t)best;
        if (request->scores && request->score_capacity > row)
            request->scores[row] = logits[best];
    }
    free(collapsed); free(normed); free(logits);
    return 0;
}

static const ColiEdgeAdapter glm53_edge_adapter = {
    sizeof(ColiEdgeAdapter), COLI_EDGE_ABI_VERSION, "glm53",
    glm53_edge_engine_open, glm53_edge_engine_destroy,
    glm53_edge_tokenize, glm53_edge_detokenize,
    glm53_edge_embed, glm53_edge_select, glm53_edge_logits, {0}
};

int coli_glm53_edge_adapter_register(void) {
    return coli_edge_adapter_register(&glm53_edge_adapter);
}
#endif /* COLI_EDGE_ADAPTER */
