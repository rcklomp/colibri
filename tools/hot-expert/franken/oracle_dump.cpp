// tools/hot-expert/franken/oracle_dump.cpp
//
// L0 step 2, Deliverable A (L0-STEP2-BRIEF-2026-09-22.md).
//
// A standalone tool, linked directly against libllama + ggml (no `common`),
// that loads a GGUF model with n_gpu_layers=0, decodes one batch of tokens,
// and -- via a ggml_backend_sched_eval_callback modelled on
// examples/eval-callback/eval-callback.cpp and common/debug.cpp's
// common_debug_cb_eval -- dumps the LAST token's column of every named
// intermediate tensor that qwen4exp.cpp's cb() calls produce, for the names
// the brief lists. Every op in this run happens on the CPU backend because
// the container it is built and run in is never given a --device flag, so
// ggml-hip finds no ROCm device; n_gpu_layers=0 is belt and suspenders.
//
// cb() (llama_context::graph_get_cb(), src/llama-context.cpp) names a
// tensor "<name>-<il>" when il >= 0 and plain "<name>" when il == -1 --
// several of the names on the brief's list (result_norm, result_output,
// ple_embd, and this file's own extra probes model.input_embed / hc_init)
// are called with il == -1 and so never carry a "-<il>" suffix.
//
// Two extensions on top of the original 1256/1257-tap dump (same rules:
// CPU only, no GPU, docker without --device):
//   --greedy N   decode N further tokens greedily (argmax, no sampler
//                state) after the prompt, print/write the ids and (via
//                llama_token_to_piece) the text.
//   ffn_moe_topk the per-token selected-expert-id tensor build_moe_ffn
//                names (llama-graph.cpp:2058), I32 [n_expert_used,
//                n_tokens]. Captured for EVERY position of EVERY decode
//                call (prompt AND each greedy step), unlike the rest of
//                WANTED which stays last-token-only and prompt-only (see
//                CbCtx::prompt_phase) so the original 1256-tap dump that
//                Deliverable B already validated against is unchanged.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------
// The names the brief asks for (build_hc_mix / build_layer_attn /
// build_layer_attn_linear / build_layer_ffn / the PLE input, all cb()'d
// with il >= 0 unless noted) plus result_norm / result_output / ple_embd
// (il == -1) and two extra "is the post-PLE embedding named" probes
// (model.input_embed, hc_init -- both il == -1, both upstream of the PLE
// add itself, which qwen4exp.cpp never names on its own; see the report).
// ---------------------------------------------------------------------
static const std::set<std::string> WANTED = {
    "hc_norm", "hc_gate", "hc_mixed", "hc_inject", "hc_combine",
    "attn_pregate", "attn_gated", "attn_output", "kqv_out",
    "indexer_q", "indexer_k", "indexer_k_pooled", "indexer_score", "indexer_top_k",
    "linear_attn_qkv_mixed", "z", "q_conv", "k_conv", "v_conv", "conv_output_silu",
    "alpha", "beta", "a_softplus", "beta_sigmoid", "state_predelta", "linear_attn_out",
    "ffn_moe_out", "ffn_shexp", "ffn_shexp_gated", "shared_expert_gate_sigmoid", "ffn_out",
    "l_last", "ple_embd", "ple_gated_value",
    // il == -1 in the graph regardless of layer count:
    "result_norm", "result_output",
    // extra probes for "the model's input embedding after the PLE add, if
    // it is named" -- see the report for why neither is actually that.
    "model.input_embed", "hc_init",
    // per-token selected-expert ids (build_moe_ffn, llama-graph.cpp); see
    // the "special case" comment in cb_eval() -- unlike everything else in
    // this set, it is captured every position of every decode call.
    "ffn_moe_topk",
};

// DeepSeek-V4-Flash (arch deepseek4): the cb() names of
// src/models/deepseek4.cpp and build_moe_ffn / build_ffn that
// decode/ds4_graph.cpp taps under the same names (decode/DEEPSEEK4.md
// section 4). Chosen after the model loads, from general.architecture, so a
// qwen4exp dump is byte-for-byte what it was.
static const std::set<std::string> WANTED_DS4 = {
    "hc_init",
    "hc_mixes", "hc_pre", "hc_post", "hc_comb", "hc_attn_pre", "attn_norm",
    "qr", "qr_norm", "q_norm", "q", "kv_norm", "kv",
    "csa_state_kv", "csa_state_score", "csa_state_score_ape",
    "lid_state_kv", "lid_state_score", "lid_state_score_ape",
    "lid_q", "lid_q_rope", "lid_q_rot", "lid_weights", "lid_score_masked", "lid_top_k",
    "hca_state_kv", "hca_state_score", "hca_state_score_ape",
    "attn_raw", "attn_csa_lid", "attn_hca", "attn_derope", "attn_out",
    "hc_attn_post", "hc_ffn_pre", "ffn_norm",
    "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_probs_biased", "ffn_moe_topk",
    "ffn_moe_weights", "ffn_moe_weights_norm", "ffn_moe_weights_scaled",
    "ffn_moe_up", "ffn_moe_gate", "ffn_moe_swiglu_limited", "ffn_moe_down",
    "ffn_moe_weighted", "ffn_moe_out",
    "ffn_up", "ffn_gate", "ffn_swiglu_limited", "ffn_shexp", "ffn_out", "l_last",
    "hc_head_mixes", "hc_head_pre", "hc_head", "result_norm", "result_output",
};
static const std::set<std::string> * g_wanted = &WANTED;

struct Hit {
    std::string   file_name;  // t->name + ".f32"
    std::string   base;       // name without the "-<il>" suffix
    int           il;         // -1 if the name carried no suffix
    int64_t       ne[3];      // ne0 ne1 ne2 of the FULL tensor (ne3 is always 1 here)
    std::string   type_name;
};

// Parse "<base>-<il>" -> (base, il); returns (name, -1) if there is no
// "-<digits>" suffix, or the suffix is not all digits.
static bool split_name(const std::string & name, std::string & base, int & il) {
    const std::set<std::string> & WANTED = *g_wanted;
    if (WANTED.count(name)) {
        base = name;
        il = -1;
        return true;
    }
    auto pos = name.find_last_of('-');
    if (pos == std::string::npos || pos + 1 >= name.size()) {
        return false;
    }
    std::string b = name.substr(0, pos);
    std::string suf = name.substr(pos + 1);
    if (suf.empty() || !std::all_of(suf.begin(), suf.end(), [](unsigned char c) { return isdigit(c); })) {
        return false;
    }
    if (!WANTED.count(b)) {
        return false;
    }
    base = b;
    il = atoi(suf.c_str());
    return true;
}

static float get_float_value(const uint8_t * data, ggml_type type, const size_t * nb,
                              int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    size_t off = (size_t) i3 * nb[3] + (size_t) i2 * nb[2] + (size_t) i1 * nb[1] + (size_t) i0 * nb[0];
    const uint8_t * p = data + off;
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) p;
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
        case GGML_TYPE_I64:  return (float) *(const int64_t *) p;
        case GGML_TYPE_I32:  return (float) *(const int32_t *) p;
        case GGML_TYPE_I16:  return (float) *(const int16_t *) p;
        case GGML_TYPE_I8:   return (float) *(const int8_t *) p;
        default:
            fprintf(stderr, "oracle_dump: unsupported/quantized type %s, skipping\n", ggml_type_name(type));
            return NAN;
    }
}

struct CbCtx {
    std::string  out_dir;
    std::vector<uint8_t> host_buf;
    std::vector<Hit> hits;
    int n_dumped = 0;
    // build_hc_mix() is called TWICE per layer (once before the attention
    // block, once before the FFN block) with the SAME cb() names -- "hc_norm",
    // "hc_gate", "hc_mixed", "hc_inject" -- on two DIFFERENT tensor objects,
    // so t->name collides across two real graph nodes (this is a property
    // of qwen4exp.cpp's graph, not of this tool). Writing both to
    // "<name>-<il>.f32" would silently keep only the later-executed (FFN)
    // one. Track how many times each t->name has been dumped so a second
    // occurrence gets "<name>-<il>.2.f32" instead of overwriting the first.
    std::map<std::string, int> occurrence;

    // true only while decoding the original prompt batch; set false before
    // the first greedy step so the WANTED single-column taps stay exactly
    // the dump Deliverable B already validated (no extra occurrences pile
    // up across 16 more forward passes). ffn_moe_topk ignores this flag.
    bool prompt_phase = true;

    // absolute position of column 0 of the tensor currently being
    // processed; main() advances it by the ubatch's token count right
    // after each llama_decode() call returns.
    int64_t position_base = 0;

    // per-layer, position-ordered, flat [n_expert_used] blocks of expert
    // ids, accumulated across every decode call (prompt + every greedy step).
    std::map<int, std::vector<int32_t>> moe_ids;
    std::map<int, int64_t> moe_n_expert_used;
    // one "<il> <pos>: <id> <id> ..." entry per (il, position); sorted by
    // (pos, il) before being written so the file reads as a timeline.
    std::vector<std::tuple<int64_t, int, std::string>> moe_lines;

    // --stop-after-layer N: once "l_last-N" has been dumped, the callback
    // returns false and ggml_backend_sched stops computing the graph
    // (ggml-backend.cpp, the `break` after callback_eval(t, false)). The
    // layers past N are never evaluated, so their weights are never paged
    // in -- which is what bounds this tool's reads on a model that does not
    // fit beside the one the box is serving.
    std::string stop_key;
    bool stopped = false;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ctx = (CbCtx *) user_data;

    std::string base;
    int il;
    if (!split_name(t->name, base, il)) {
        return false; // not one of ours: no need to force materialisation
    }

    const bool is_moe_topk = (base == "ffn_moe_topk");

    if (!is_moe_topk && !ctx->prompt_phase) {
        return false; // the rest of WANTED is prompt-only, see CbCtx::prompt_phase
    }

    if (ask) {
        return true; // we want the data for this one
    }

    // copy to host if needed (handles both plain host buffers and views on them)
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        size_t n_bytes = ggml_nbytes(t);
        ctx->host_buf.resize(n_bytes);
        ggml_backend_tensor_get(t, ctx->host_buf.data(), 0, n_bytes);
        data = ctx->host_buf.data();
    }

    if (is_moe_topk) {
        // I32 [n_expert_used, n_tokens] -- every position of this call, not
        // just the last, accumulated across every decode call so far.
        if (t->type != GGML_TYPE_I32) {
            fprintf(stderr, "oracle_dump: %s has unexpected type %s, expected i32, skipping\n",
                    t->name, ggml_type_name(t->type));
            return true;
        }
        const int64_t n_used   = t->ne[0];
        const int64_t n_tokens = t->ne[1];
        ctx->moe_n_expert_used[il] = n_used;
        auto & vec = ctx->moe_ids[il];
        for (int64_t pos_i = 0; pos_i < n_tokens; ++pos_i) {
            std::ostringstream line;
            line << il << " " << (ctx->position_base + pos_i) << ":";
            for (int64_t k = 0; k < n_used; ++k) {
                size_t off = (size_t) pos_i * t->nb[1] + (size_t) k * t->nb[0];
                int32_t id = *(const int32_t *)(data + off);
                vec.push_back(id);
                line << " " << id;
            }
            ctx->moe_lines.emplace_back(ctx->position_base + pos_i, il, line.str());
        }
        fprintf(stderr, "oracle_dump: moe_ids layer %d: %lld position(s) starting at %lld (n_expert_used=%lld)\n",
                il, (long long) n_tokens, (long long) ctx->position_base, (long long) n_used);
        return true;
    }

    if (ggml_is_quantized(t->type)) {
        fprintf(stderr, "oracle_dump: %s is quantized (%s), skipping\n", t->name, ggml_type_name(t->type));
        return true;
    }

    // pick the axis to fix at its LAST index ("the last token's column"):
    // the highest-index dimension with ne[i] > 1. By llama.cpp convention
    // that is always the token/batch axis for a per-token activation,
    // whatever its numeric size happens to be (so this does not rely on
    // n_tokens being distinct from an unrelated dim, e.g. the hc multiplier).
    int axis = 0;
    for (int i = 3; i >= 0; --i) {
        if (t->ne[i] > 1) { axis = i; break; }
    }
    int64_t last_idx = t->ne[axis] - 1;

    int64_t dims[4];
    for (int i = 0; i < 4; ++i) dims[i] = (i == axis) ? 1 : t->ne[i];

    std::vector<float> out;
    out.reserve((size_t)(dims[0] * dims[1] * dims[2] * dims[3]));
    int64_t idx[4];
    for (int64_t i3 = 0; i3 < dims[3]; ++i3) {
        idx[3] = (axis == 3) ? last_idx : i3;
        for (int64_t i2 = 0; i2 < dims[2]; ++i2) {
            idx[2] = (axis == 2) ? last_idx : i2;
            for (int64_t i1 = 0; i1 < dims[1]; ++i1) {
                idx[1] = (axis == 1) ? last_idx : i1;
                for (int64_t i0 = 0; i0 < dims[0]; ++i0) {
                    idx[0] = (axis == 0) ? last_idx : i0;
                    out.push_back(get_float_value(data, t->type, t->nb, idx[0], idx[1], idx[2], idx[3]));
                }
            }
        }
    }

    int occ = ++ctx->occurrence[t->name];
    std::string stem = std::string(t->name) + (occ > 1 ? ("." + std::to_string(occ)) : "");
    std::string file_name = stem + ".f32";
    std::string path = ctx->out_dir + "/" + file_name;
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "oracle_dump: cannot open %s for writing\n", path.c_str());
        return true;
    }
    f.write((const char *) out.data(), (std::streamsize)(out.size() * sizeof(float)));
    f.close();

    Hit h;
    h.file_name = stem; // name field in index.txt; carries ".2" etc. on a collision
    h.base = base;
    h.il = il;
    h.ne[0] = t->ne[0];
    h.ne[1] = t->ne[1];
    h.ne[2] = t->ne[2];
    h.type_name = ggml_type_name(t->type);
    ctx->hits.push_back(h);
    ctx->n_dumped++;

    fprintf(stderr, "oracle_dump: dumped %-28s ne=[%lld,%lld,%lld,%lld] type=%s axis=%d occ=%d -> %s (%zu floats)\n",
            t->name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            ggml_type_name(t->type), axis, occ, file_name.c_str(), out.size());

    if (!ctx->stop_key.empty() && ctx->stop_key == t->name) {
        ctx->stopped = true;
        fprintf(stderr, "oracle_dump: %s dumped -- stopping the graph here (--stop-after-layer)\n", t->name);
        return false;
    }
    return true;
}

static bool mkdir_p(const std::string & dir) {
    std::string cur;
    for (size_t i = 0; i <= dir.size(); ++i) {
        if (i == dir.size() || dir[i] == '/') {
            if (!cur.empty()) {
                mkdir(cur.c_str(), 0775);
            }
            if (i < dir.size()) cur += dir[i];
        } else {
            cur += dir[i];
        }
    }
    struct stat st;
    return stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string out_dir = "./oracle";
    std::vector<llama_token> tokens;
    int n_threads = 8;
    int n_greedy = 0;
    int stop_after = -1;
    bool no_extra_bufts = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (a == "--greedy" && i + 1 < argc) {
            n_greedy = atoi(argv[++i]);
        } else if (a == "--stop-after-layer" && i + 1 < argc) {
            stop_after = atoi(argv[++i]);
        } else if (a == "--no-extra-bufts") {
            // no CPU weight repacking: a repacked tensor is READ IN FULL at
            // load time, which is exactly the read --stop-after-layer avoids
            no_extra_bufts = true;
        } else if (a == "--tokens") {
            while (i + 1 < argc && isdigit((unsigned char) argv[i + 1][0])) {
                tokens.push_back((llama_token) atoi(argv[++i]));
            }
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "usage: %s --model <first-shard.gguf> --tokens <id...> [--out <dir>] [--threads N] [--greedy N] [--stop-after-layer N] [--no-extra-bufts]\n", argv[0]);
        return 1;
    }
    if (tokens.empty()) {
        tokens = { 248044, 785, 10945, 315, 1495, 374 };
        fprintf(stderr, "no --tokens given, using the brief's default 6 ids\n");
    }

    fprintf(stderr, "oracle_dump: model=%s\n", model_path.c_str());
    fprintf(stderr, "oracle_dump: tokens = [");
    for (size_t i = 0; i < tokens.size(); ++i) fprintf(stderr, "%s%d", i ? ", " : "", tokens[i]);
    fprintf(stderr, "]\n");

    if (!mkdir_p(out_dir)) {
        fprintf(stderr, "oracle_dump: could not create out dir %s\n", out_dir.c_str());
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    if (no_extra_bufts) mparams.use_extra_bufts = false;
    // --stop-after-layer bounds the reads only through a LAZY mapping: pin the
    // load mode to mmap rather than let AUTO decide (a non-mmap load reads
    // every tensor into a buffer up front). Run it with no_populate.so
    // preloaded, or libllama's MAP_POPULATE reads the whole file anyway.
    if (stop_after >= 0) mparams.load_mode = LLAMA_LOAD_MODE_MMAP;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "oracle_dump: failed to load model %s\n", model_path.c_str());
        return 1;
    }

    CbCtx cb_ctx;
    cb_ctx.out_dir = out_dir;
    {
        char arch[64] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (std::string(arch) == "deepseek4") {
            g_wanted = &WANTED_DS4;
            fprintf(stderr, "oracle_dump: arch deepseek4, using the DeepSeek-V4 tap names\n");
        }
    }
    if (stop_after >= 0) {
        if (n_greedy > 0) {
            fprintf(stderr, "oracle_dump: --stop-after-layer and --greedy exclude each other\n");
            llama_model_free(model);
            return 1;
        }
        cb_ctx.stop_key = "l_last-" + std::to_string(stop_after);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = std::max<uint32_t>(64, (uint32_t) tokens.size() + (uint32_t) n_greedy + 8);
    cparams.n_batch         = (uint32_t) tokens.size();
    cparams.n_ubatch        = (uint32_t) tokens.size();
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.cb_eval          = cb_eval;
    cparams.cb_eval_user_data = &cb_ctx;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "oracle_dump: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "oracle_dump: decoding %zu tokens as one batch...\n", tokens.size());

    int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()));
    if (rc != 0) {
        fprintf(stderr, "oracle_dump: llama_decode failed, rc=%d\n", rc);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    cb_ctx.position_base += (int64_t) tokens.size();

    fprintf(stderr, "oracle_dump: decode done, %d tensors dumped%s\n", cb_ctx.n_dumped,
            cb_ctx.stopped ? " (graph stopped after the requested layer)" : "");
    if (!cb_ctx.stop_key.empty() && !cb_ctx.stopped) {
        fprintf(stderr, "oracle_dump: --stop-after-layer: %s was never dumped\n", cb_ctx.stop_key.c_str());
    }

    // --- greedy generation (argmax, temperature 0, no sampler state) ---
    std::vector<llama_token> greedy_ids;
    std::string greedy_text;
    if (n_greedy > 0) {
        cb_ctx.prompt_phase = false; // freeze the WANTED single-column taps at the prompt's dump

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);

        llama_token next_tok = -1;
        for (int step = 0; step < n_greedy; ++step) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) {
                fprintf(stderr, "oracle_dump: no logits available at greedy step %d\n", step);
                break;
            }
            int32_t best = 0;
            float best_val = logits[0];
            for (int32_t v = 1; v < n_vocab; ++v) {
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }
            next_tok = (llama_token) best;
            greedy_ids.push_back(next_tok);

            char piece[256];
            int32_t n = llama_token_to_piece(vocab, next_tok, piece, sizeof(piece), 0, true);
            if (n > 0) {
                greedy_text.append(piece, n);
            } else if (n < 0) {
                fprintf(stderr, "oracle_dump: llama_token_to_piece buffer too small for token %d\n", next_tok);
            }

            rc = llama_decode(ctx, llama_batch_get_one(&next_tok, 1));
            if (rc != 0) {
                fprintf(stderr, "oracle_dump: llama_decode failed at greedy step %d, rc=%d\n", step, rc);
                break;
            }
            cb_ctx.position_base += 1;
        }

        fprintf(stderr, "greedy_ids: ");
        for (size_t i = 0; i < greedy_ids.size(); ++i) fprintf(stderr, "%s%d", i ? " " : "", greedy_ids[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "greedy_text: %s\n", greedy_text.c_str());

        std::ofstream gf(out_dir + "/greedy.txt");
        gf << "greedy_ids:";
        for (auto id : greedy_ids) gf << " " << id;
        gf << "\n";
        gf << "greedy_text: " << greedy_text << "\n";
        gf.close();
    }

    // --- per-layer, per-position MoE expert ids, prompt + every greedy step ---
    if (!cb_ctx.moe_ids.empty()) {
        for (auto & kv : cb_ctx.moe_ids) {
            int il = kv.first;
            int64_t n_used = cb_ctx.moe_n_expert_used[il];
            std::string path = out_dir + "/moe_ids-" + std::to_string(il) + ".i32";
            std::ofstream f(path, std::ios::binary);
            f.write((const char *) kv.second.data(), (std::streamsize)(kv.second.size() * sizeof(int32_t)));
            f.close();
            fprintf(stderr, "oracle_dump: wrote %s (%zu positions x %lld experts)\n",
                    path.c_str(), kv.second.size() / (size_t) std::max<int64_t>(1, n_used), (long long) n_used);
        }
        std::sort(cb_ctx.moe_lines.begin(), cb_ctx.moe_lines.end());
        std::ofstream mf(out_dir + "/moe_ids.txt");
        for (auto & line : cb_ctx.moe_lines) {
            mf << std::get<2>(line) << "\n";
        }
        mf.close();
        fprintf(stderr, "oracle_dump: wrote %s/moe_ids.txt (%zu lines)\n", out_dir.c_str(), cb_ctx.moe_lines.size());
    }

    // report which of the WANTED names never showed up at all
    std::set<std::string> seen_bases;
    for (auto & h : cb_ctx.hits) seen_bases.insert(h.base);
    if (!cb_ctx.moe_ids.empty()) seen_bases.insert("ffn_moe_topk"); // handled outside cb_ctx.hits
    for (auto & w : *g_wanted) {
        if (!seen_bases.count(w)) {
            fprintf(stderr, "oracle_dump: NOT FOUND in this build's graph: %s\n", w.c_str());
        }
    }

    // index.txt columns: name il ne0 ne1 ne2 type -- "name" is the file's
    // stem (without ".f32"): equal to "<base>-<il>" (or plain "<base>" when
    // il == -1) on a tensor's first occurrence, and "<base>-<il>.2" etc. on
    // a repeat name (see CbCtx::occurrence above) -- il is always the true
    // parsed layer index either way, so a reader that only wants layer info
    // does not need to parse the ".2" suffix off the name.
    std::string idx_path = out_dir + "/index.txt";
    std::ofstream idx(idx_path);
    for (auto & h : cb_ctx.hits) {
        idx << h.file_name << " " << h.il << " " << h.ne[0] << " " << h.ne[1] << " " << h.ne[2]
            << " " << h.type_name << "\n";
    }
    idx.close();
    fprintf(stderr, "oracle_dump: wrote %s (%zu lines)\n", idx_path.c_str(), cb_ctx.hits.size());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
