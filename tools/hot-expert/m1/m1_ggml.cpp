// tools/hot-expert/m1/m1_ggml.cpp
//
// M1 candidate 1: llama.cpp's served routed-expert path via the public ggml
// API (tools/hot-expert/M1-M5-BRIEF-2026-09-22.md, section M1, candidate 1).
//
// Builds the SAME graph shape `llm_graph_context::build_moe_ffn` builds at
// decode for qwen4exp (src/llama-graph.cpp, src/models/qwen4exp.cpp), read
// on the rig at commit 39931761a of ~/src/llama-glm53:
//   - the served GGUF (~/models/Qwen3.8-Flash-Next/UD-IQ4_XS/...) carries
//     separate `ffn_gate_exps` / `ffn_up_exps` tensors, no fused
//     `ffn_gate_up_exps` (confirmed by `strings` on the gguf header -- the
//     rig has no numpy, so gguf-py's reader can't run here; llama-gguf
//     needed no --device and was not used to avoid loading the file), so
//     build_moe_ffn takes its "separate gate and up" branch, NOT the fused
//     one this file's header comment first assumed from the brief's mention
//     of a merged tensor.
//   - qwen4exp never sets hparams.swiglu_clamp_exp (the GGUF has no
//     qwen4exp.swiglu_clamp_exp key), so it stays at its default fill 0.0f,
//     which is below the `limit > eps` gate in build_moe_ffn -- the served
//     path is the PLAIN swiglu_split branch, not the swiglu_clamp branch
//     (that branch is GLM-5.3-Flash's/DeepSeek4's, not Qwen3.8's).
// So the graph replicated here is exactly:
//   up    = ggml_mul_mat_id(up_exps,   x, ids)   // [n_ff,    n_used, T]
//   gate  = ggml_mul_mat_id(gate_exps, x, ids)   // [n_ff,    n_used, T]
//   h     = ggml_swiglu_split(gate, up)          // silu(gate)*up
//   down  = ggml_mul_mat_id(down_exps, h, ids)   // [n_embd,  n_used, T]
//   down  = ggml_mul(down, weights)              // weights: [1, n_used, T]
//   out   = sum over n_used of down's per-expert 2D views (ggml_add chain),
//           exactly as build_moe_ffn's "order the views before the adds"
//           loop does (llama-graph.cpp:2265-2295).
// The gating/logits/argsort nodes that pick `ids`/`weights` in the real
// model are NOT built here -- M1's brief scopes the measured step to
// "gate+up on 10 experts, SiLU*up, down on 10 experts, weighted sum"; ids
// and weights are synthesised on the host (m1_common.h) and uploaded as
// graph inputs, which is exactly the boundary build_moe_ffn itself draws
// with its optional `selected_experts_in`/`probs_in` parameters.
//
// Shapes (tools/hot-expert/M1-M5-BRIEF-2026-09-22.md, "Shapes" table):
//   routed gate, up: 640 rows x K=2560, 512 experts, IQ3_S
//   routed down:     2560 rows x K=640, 512 experts, IQ4_NL
// (the brief lists IQ4_XS/Q8_0 for a handful of outlier layers; this
// harness measures the modal format used by 47 of 48 gate/up layers and
// 43 of 48 down layers, per the brief's own table).
//
// Backend: `ggml_backend_cuda_init(0)` -- the HIP backend keeps the CUDA
// symbol names in this ggml (see ggml/include/ggml-cuda.h), confirmed by
// reading that header on the rig; libggml-hip.so is what actually answers
// the call. Linked against the already-built
// ~/src/llama-glm53/build-hip/bin/libggml*.so (see the Makefile).
//
// Building this program does NOT touch the GPUs: it links a HIP backend
// but the offline hipcc compile only needs --offload-arch=gfx1100 codegen.
// It is not run here; the orchestrator runs it under the rig lock.
//
// BUG FOUND AND FIXED 2026-09-22 (all-zero oracle on the GPU run: rc 0,
// ggml_check_maxabs=0, ggml_check_cos=nan -- both gpu_out and ref_out were
// zero). Root cause: this harness allocates ONE graph per T and calls
// ggml_backend_graph_compute on it 1100 times (100 warmup + 1000 timed) to
// keep allocation cost out of the timing -- unlike llama.cpp's own usage,
// which builds a fresh ggml_context/cgraph every token. ggml_set_input() on
// a leaf does NOT stop ggml_gallocr from reusing that leaf's buffer for a
// later node's output once nothing else in the graph reads it again; across
// repeated ggml_backend_graph_compute calls on the SAME allocated graph,
// down's/moe_out's output ended up written into x/ids/weights' memory, so
// by warmup iteration 2 `ids` no longer held the uploaded expert indices.
// A `--cpu` diagnostic mode (ggml_backend_cpu_init(), touches no GPU) plus
// a one-shot pre-warmup read isolated it: the FIRST compute produced a
// healthy `up` (diag_oneshot_up_l1 nonzero), but reading `up` again after
// the 100-iteration warmup loop gave 0, and diag_ids_unchanged_after_warmup
// caught `ids` itself corrupted. Standalone ggml_mul_mat (plain, no _id)
// and standalone ggml_mul_mat_id calls (n_used=1 and n_used=10, sequential
// and random ids) on the same weight tensors all worked individually,
// ruling out the quantiser, the fork-local IQ-panel CPU GEMM path
// (ggml-cpu/iqp.cpp, ruled out with ggml_backend_cpu_set_use_ref(true)),
// and ggml_mul_mat_id's row-selection logic itself. Fix: ggml_set_output()
// on x/ids/weights (below) marks them persistent, so gallocr never reuses
// their memory -- confirmed on --cpu (ggml_check_cos=0.999965, exit 0).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"  // ggml_backend_cpu_init -- --cpu diagnostic mode
#include "ggml-cuda.h" // ggml_backend_cuda_init -- HIP backend, CUDA-named API

#include "m1_common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {

// N_EXPERT is mutable (not the brief's fixed 512) ONLY so `--cpu` can shrink
// it to 16: the diagnostic CPU-backend mode this file gained after the GPU
// run of 2026-09-22 reported an all-zero oracle (ggml_check_maxabs=0,
// ggml_check_cos=nan on BOTH gpu_out and ref_out) exists to isolate that bug
// without touching the GPUs, and quantising 512 experts x 3 tensors on a CPU
// backend is slow enough to want a smaller N_EXPERT for that path. The GPU
// path (no --cpu) always uses the real N_EXPERT=512.
int64_t N_EXPERT           = 512;
constexpr int64_t N_EXPERT_USED = 10;
constexpr int64_t D_MODEL       = 2560; // K for gate/up, n_embd for down's output
constexpr int64_t N_FF          = 640;  // rows for gate/up, K for down
constexpr int     WARMUP_ITERS  = 100;
constexpr int     TIMED_ITERS   = 1000;

struct WeightSet {
    ggml_context       *ctx = nullptr;
    ggml_backend_buffer *buf = nullptr;
    ggml_tensor         *gate_exps = nullptr; // [D_MODEL, N_FF,   N_EXPERT] IQ3_S
    ggml_tensor         *up_exps   = nullptr; // [D_MODEL, N_FF,   N_EXPERT] IQ3_S
    ggml_tensor         *down_exps = nullptr; // [N_FF,    D_MODEL,N_EXPERT] IQ4_NL
};

// Quantise `nrows` x `n_per_row` random fp32 weights into `type` and upload
// them into `dst`'s expert slice `e`. Generates and quantises ONE expert at
// a time (a few MB of host fp32) rather than materialising all 512 experts
// of a tensor in host RAM at once (that would be ~3.3 GB per tensor).
//
// `diag_label`, when non-null, prints a diagnostic for THIS call's row 0
// only: the L1 norm of the pre-quant source row and of the same row
// immediately dequantised back (ggml_get_type_traits(type)->to_float) --
// BEFORE any upload/readback round-trip -- so a zero here means the
// quantiser itself is degenerate for this type/shape, independent of any
// backend or ggml_mul_mat_id bug.
void quantize_upload_expert(ggml_tensor *dst, int64_t e, int64_t nrows, int64_t n_per_row,
                             const char *diag_label = nullptr) {
    std::vector<float> src = m1::randf_vec(static_cast<size_t>(nrows) * n_per_row, -1.0f, 1.0f);
    size_t row_size = ggml_row_size(dst->type, n_per_row);
    std::vector<uint8_t> qbuf(row_size * nrows);
    size_t written = ggml_quantize_chunk(dst->type, src.data(), qbuf.data(),
                                          /*start=*/0, nrows, n_per_row, /*imatrix=*/nullptr);
    if (written != qbuf.size()) {
        std::fprintf(stderr, "quantize_chunk wrote %zu, expected %zu\n", written, qbuf.size());
        std::abort();
    }
    if (diag_label) {
        std::vector<float> dequant_row0(n_per_row);
        ggml_get_type_traits(dst->type)->to_float(qbuf.data(), dequant_row0.data(), n_per_row);
        char lbl[128];
        std::snprintf(lbl, sizeof(lbl), "diag_%s_src_row0_l1", diag_label);
        std::printf("%s=%.6g\n", lbl, m1::l1_norm(src.data(), n_per_row));
        m1::print_first_n(lbl, src.data(), n_per_row);
        std::snprintf(lbl, sizeof(lbl), "diag_%s_dequant_row0_l1", diag_label);
        std::printf("%s=%.6g\n", lbl, m1::l1_norm(dequant_row0.data(), n_per_row));
        m1::print_first_n(lbl, dequant_row0.data(), n_per_row);
    }
    size_t offset = dst->nb[2] * static_cast<size_t>(e);
    if (qbuf.size() != dst->nb[2]) {
        std::fprintf(stderr, "expert stride mismatch: qbuf=%zu nb[2]=%zu\n", qbuf.size(), (size_t)dst->nb[2]);
        std::abort();
    }
    ggml_backend_tensor_set(dst, qbuf.data(), offset, qbuf.size());
}

WeightSet build_weights(ggml_backend_t backend) {
    WeightSet w;
    ggml_init_params params = {
        /* .mem_size   = */ 3 * ggml_tensor_overhead() + 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    w.ctx = ggml_init(params);

    w.gate_exps = ggml_new_tensor_3d(w.ctx, GGML_TYPE_IQ3_S, D_MODEL, N_FF, N_EXPERT);
    w.up_exps   = ggml_new_tensor_3d(w.ctx, GGML_TYPE_IQ3_S, D_MODEL, N_FF, N_EXPERT);
    w.down_exps = ggml_new_tensor_3d(w.ctx, GGML_TYPE_IQ4_NL, N_FF, D_MODEL, N_EXPERT);
    ggml_set_name(w.gate_exps, "gate_exps");
    ggml_set_name(w.up_exps,   "up_exps");
    ggml_set_name(w.down_exps, "down_exps");

    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, backend);
    if (!w.buf) {
        std::fprintf(stderr, "failed to allocate weight tensors on backend\n");
        std::abort();
    }

    for (int64_t e = 0; e < N_EXPERT; e++) {
        quantize_upload_expert(w.gate_exps, e, N_FF, D_MODEL, e == 0 ? "gate_e0" : nullptr);
        quantize_upload_expert(w.up_exps,   e, N_FF, D_MODEL);
        quantize_upload_expert(w.down_exps, e, D_MODEL, N_FF);
    }
    return w;
}

// The per-T graph (build_moe_ffn's "separate gate and up" branch,
// llama-graph.cpp:2135-2160; plain-swiglu activation, 2165-2196, has_gate
// branch with no clamp; down projection, 2249-2250; weighting, 2260-2263;
// per-expert view+add aggregation loop, 2268-2295) is built directly in
// main() below, once per T value, since it needs the just-allocated
// backend buffer type and is only built twice (T=1, T=32).

} // namespace

int main(int argc, char **argv) {
    bool cpu_mode = false;
    bool use_ref  = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--cpu") == 0) cpu_mode = true;
        // --use-ref: ggml-cpu's own escape hatch (ggml_backend_cpu_set_use_ref)
        // that disables its fork-local "IQ panel" GEMM path (ggml-cpu/iqp.cpp)
        // for MUL_MAT/MUL_MAT_ID and falls back to the plain vec_dot path.
        // Added here as a diagnostic toggle to test suspect (d): a bug in
        // that fork-local kernel, not in build_moe_ffn's graph shape (which
        // is upstream, heavily used code) or in IQ3_S/IQ4_NL quantisation
        // (already ruled out: diag_gate_e0_dequant_row0_l1 is nonzero).
        if (std::strcmp(argv[i], "--use-ref") == 0) use_ref = true;
    }

    ggml_backend_t backend = nullptr;
    if (cpu_mode) {
        // Diagnostic mode (2026-09-22): runs the identical weight build +
        // graph + oracle on the ggml CPU backend, touching no GPU, to
        // isolate whether the GPU run's all-zero oracle is a quantiser bug
        // (would reproduce here too) or a HIP-backend/upload-readback bug
        // (would NOT reproduce here). N_EXPERT shrinks to 16 -- 512 experts
        // x 3 tensors of CPU-side quantisation is slow and the oracle only
        // ever reads N_EXPERT_USED=10 of them per token anyway.
        N_EXPERT = 16;
        backend = ggml_backend_cpu_init();
        if (!backend) {
            std::fprintf(stderr, "ggml_backend_cpu_init() failed\n");
            return 2;
        }
        if (use_ref) {
            ggml_backend_cpu_set_use_ref(backend, true);
        }
        std::fprintf(stderr, "--cpu: N_EXPERT=%lld, backend=CPU, use_ref=%d\n",
                     (long long)N_EXPERT, (int)use_ref);
    } else {
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            std::fprintf(stderr, "ggml_backend_cuda_init(0) failed -- no HIP device backend\n");
            return 2;
        }
    }

    WeightSet w = build_weights(backend);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Minimal isolation test (--cpu only): plain ggml_mul_mat on a single
    // expert's plane of w.up_exps, entirely bypassing ggml_mul_mat_id's
    // row-selection machinery. diag_gate_e0_dequant_row0_l1 already showed
    // IQ3_S's to_float() dequantiser is fine -- to_float is NOT what the
    // matmul op itself uses (that's vec_dot_iq3_s / a Q8_K-converted src1);
    // this isolates whether THAT is where the zero comes from, independent
    // of _id row selection.
    if (cpu_mode) {
        size_t mm_mem = 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false) + 4096;
        ggml_init_params mm_params = {mm_mem, nullptr, true};
        ggml_context *mm_ctx = ggml_init(mm_params);
        ggml_cgraph *mm_gf = ggml_new_graph_custom(mm_ctx, 8, false);

        ggml_tensor *expert0_up = ggml_view_2d(mm_ctx, w.up_exps, D_MODEL, N_FF,
                                                w.up_exps->nb[1], /*offset=*/0);
        ggml_tensor *xs = ggml_new_tensor_1d(mm_ctx, GGML_TYPE_F32, D_MODEL);
        ggml_set_input(xs);
        ggml_tensor *mm_out = ggml_mul_mat(mm_ctx, expert0_up, xs);
        ggml_set_output(mm_out);
        ggml_build_forward_expand(mm_gf, mm_out);

        ggml_gallocr_t mm_galloc = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(mm_galloc, mm_gf)) {
            std::fprintf(stderr, "diag mul_mat gallocr_alloc_graph failed\n");
        } else {
            std::vector<float> xs_host = m1::randf_vec(D_MODEL, -1.0f, 1.0f);
            ggml_backend_tensor_set(xs, xs_host.data(), 0, xs_host.size() * sizeof(float));
            ggml_status st = ggml_backend_graph_compute(backend, mm_gf);
            std::printf("diag_plain_mulmat_status=%s\n", ggml_status_to_string(st));
            std::vector<float> mm_out_host(N_FF);
            ggml_backend_tensor_get(mm_out, mm_out_host.data(), 0, N_FF * sizeof(float));
            std::printf("diag_plain_mulmat_l1=%.6g\n", m1::l1_norm(mm_out_host.data(), N_FF));
            m1::print_first_n("diag_plain_mulmat", mm_out_host.data(), N_FF);
        }
        ggml_gallocr_free(mm_galloc);
        ggml_free(mm_ctx);

        // Second isolation test: ggml_mul_mat_id itself, trimmed to the
        // simplest possible case -- ONE token, ONE selected expert (id 0),
        // b's ne[1]=1 broadcast still exercised (same as the real graph).
        // Plain mul_mat on the same plane already proved nonzero above, so
        // this isolates the "_id" row-selection/dispatch machinery itself.
        auto minimal_mulmatid_test = [&](const char *label, const std::vector<int32_t> &ids_vec) {
            int64_t n_used = (int64_t)ids_vec.size();
            size_t mid_mem = 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false) + 4096;
            ggml_init_params mid_params = {mid_mem, nullptr, true};
            ggml_context *mid_ctx = ggml_init(mid_params);
            ggml_cgraph *mid_gf = ggml_new_graph_custom(mid_ctx, 8, false);

            ggml_tensor *xs2 = ggml_new_tensor_3d(mid_ctx, GGML_TYPE_F32, D_MODEL, 1, 1);
            ggml_set_input(xs2);
            ggml_tensor *ids1 = ggml_new_tensor_2d(mid_ctx, GGML_TYPE_I32, n_used, 1); // T=1
            ggml_set_input(ids1);
            ggml_tensor *mid_out = ggml_mul_mat_id(mid_ctx, w.up_exps, xs2, ids1);
            ggml_set_output(mid_out);
            ggml_build_forward_expand(mid_gf, mid_out);

            ggml_gallocr_t mid_galloc = ggml_gallocr_new(buft);
            if (!ggml_gallocr_alloc_graph(mid_galloc, mid_gf)) {
                std::fprintf(stderr, "diag mul_mat_id(%s) gallocr_alloc_graph failed\n", label);
            } else {
                std::vector<float> xs2_host = m1::randf_vec(D_MODEL, -1.0f, 1.0f);
                ggml_backend_tensor_set(xs2, xs2_host.data(), 0, xs2_host.size() * sizeof(float));
                ggml_backend_tensor_set(ids1, ids_vec.data(), 0, ids_vec.size() * sizeof(int32_t));
                ggml_status st = ggml_backend_graph_compute(backend, mid_gf);
                std::printf("diag_mulmatid_%s_status=%s\n", label, ggml_status_to_string(st));
                // Read back ALL n_used columns (not just column 0).
                std::vector<float> mid_out_host((size_t)N_FF * n_used);
                ggml_backend_tensor_get(mid_out, mid_out_host.data(), 0, mid_out_host.size() * sizeof(float));
                std::printf("diag_mulmatid_%s_l1=%.6g\n", label, m1::l1_norm(mid_out_host.data(), mid_out_host.size()));
                for (int64_t c = 0; c < n_used; c++) {
                    double col_l1 = m1::l1_norm(mid_out_host.data() + c * N_FF, N_FF);
                    std::printf("diag_mulmatid_%s_col%lld_l1=%.6g (id=%d)\n", label, (long long)c, col_l1, ids_vec[c]);
                }
            }
            ggml_gallocr_free(mid_galloc);
            ggml_free(mid_ctx);
        };
        minimal_mulmatid_test("used1", std::vector<int32_t>{0});
        minimal_mulmatid_test("used10_seq", std::vector<int32_t>{0,1,2,3,4,5,6,7,8,9});
        minimal_mulmatid_test("used10_distinct_hi", m1::random_distinct_ids(N_EXPERT, 10, 1));
    }

    bool any_check_fail = false;

    for (int64_t T : {int64_t(1), int64_t(32)}) {
        size_t mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false) + 4096;
        ggml_init_params gparams = {mem_size, nullptr, true};
        ggml_context *ctx = ggml_init(gparams);
        ggml_cgraph *gf = ggml_new_graph_custom(ctx, 64, false);

        ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_MODEL, 1, T);
        ggml_set_name(x, "x");
        ggml_set_input(x);
        ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_EXPERT_USED, T);
        ggml_set_name(ids, "ids");
        ggml_set_input(ids);
        ggml_tensor *weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_EXPERT_USED, T);
        ggml_set_name(weights, "weights");
        ggml_set_input(weights);
        // FIX (2026-09-22, all-zero oracle): ggml_set_input() alone does not
        // stop ggml_gallocr from reusing an input leaf's memory for another
        // tensor's output once the allocator's liveness analysis decides
        // nothing in THIS graph reads it again -- true here, since nothing
        // downstream re-reads x/ids/weights after the last mul_mat_id/mul
        // that consumes them. That's fine for the llama.cpp pattern (a new
        // ggml_context + a fresh ggml_gallocr_alloc_graph every token), but
        // this harness intentionally allocates ONE graph and calls
        // ggml_backend_graph_compute on it 1100 times (100 warmup + 1000
        // timed) to amortise allocation cost out of the timing -- across
        // those repeats, the SAME buffer got reused for down's/moe_out's
        // output, so by the second call ids/x no longer held what was
        // uploaded (confirmed below: diag_ids_unchanged_after_warmup).
        // ggml_set_output() marks a leaf as persistent -- gallocr never
        // reuses its memory for anything else -- so it also fixes the case
        // where CALLER-owned inputs must survive repeated computes of one
        // allocated graph.
        ggml_set_output(x);
        ggml_set_output(ids);
        ggml_set_output(weights);

        ggml_tensor *up      = ggml_mul_mat_id(ctx, w.up_exps, x, ids);
        ggml_tensor *gate    = ggml_mul_mat_id(ctx, w.gate_exps, x, ids);
        ggml_tensor *h       = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor *down_raw = ggml_mul_mat_id(ctx, w.down_exps, h, ids);
        ggml_tensor *down    = ggml_mul(ctx, down_raw, weights);
        ggml_build_forward_expand(gf, down);

        // T=1 diagnostic: keep every stage's own memory alive (no gallocr
        // buffer reuse across stages) so each can be read back after one
        // compute, to find which stage first turns zero.
        bool extra_diag = (T == 1);
        if (extra_diag) {
            ggml_set_output(up);
            ggml_set_output(gate);
            ggml_set_output(h);
            ggml_set_output(down_raw);
            ggml_set_output(down);
        }

        std::vector<ggml_tensor *> cur_experts(N_EXPERT_USED);
        for (int64_t i = 0; i < N_EXPERT_USED; i++) {
            cur_experts[i] = ggml_view_2d(ctx, down, D_MODEL, T, down->nb[2], i * down->nb[1]);
            ggml_build_forward_expand(gf, cur_experts[i]);
        }
        ggml_tensor *moe_out = cur_experts[0];
        for (int64_t i = 1; i < N_EXPERT_USED; i++) {
            moe_out = ggml_add(ctx, moe_out, cur_experts[i]);
            ggml_build_forward_expand(gf, moe_out);
        }
        // Suspect (b): without this, gallocr is free to consider moe_out's
        // memory reclaimable the moment nothing else in the SAME graph
        // reads it -- true here since it's the last node, so in a single
        // ggml_backend_graph_compute this shouldn't matter, but it costs
        // nothing and rules the allocator out as a cause outright.
        ggml_set_output(moe_out);

        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "gallocr_alloc_graph failed for T=%lld\n", (long long)T);
            return 2;
        }

        // Host-side synthetic inputs, deterministic (m1_common.h).
        std::vector<int32_t> ids_host = m1::random_distinct_ids((int)N_EXPERT, (int)N_EXPERT_USED, (int)T);
        std::vector<float> weights_host = m1::random_norm_weights((int)N_EXPERT_USED, (int)T);
        std::vector<float> x_host = m1::randf_vec((size_t)D_MODEL * T, -1.0f, 1.0f);

        {
            const char *tag0 = (T == 1) ? "T1" : "T32";
            std::printf("diag_x_host_%s_l1=%.6g\n", tag0, m1::l1_norm(x_host.data(), x_host.size()));
        }

        ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * sizeof(float));
        ggml_backend_tensor_set(ids, ids_host.data(), 0, ids_host.size() * sizeof(int32_t));
        ggml_backend_tensor_set(weights, weights_host.data(), 0, weights_host.size() * sizeof(float));

        if (extra_diag) {
            // ONE-SHOT compute + read, before the 100+1000-iteration loop:
            // isolates whether the zero appears from the very first call
            // (a construction/compute bug) or only after repeated reruns of
            // the same allocated graph (an in-place/aliasing corruption
            // that accumulates across iterations).
            ggml_status st1 = ggml_backend_graph_compute(backend, gf);
            std::printf("diag_oneshot_status=%s\n", ggml_status_to_string(st1));
            std::vector<float> up_oneshot(ggml_nelements(up));
            ggml_backend_tensor_get(up, up_oneshot.data(), 0, up_oneshot.size() * sizeof(float));
            std::printf("diag_oneshot_up_l1=%.6g\n", m1::l1_norm(up_oneshot.data(), up_oneshot.size()));
            m1::print_first_n("diag_oneshot_up", up_oneshot.data(), up_oneshot.size());
        }

        ggml_status warmup_status = GGML_STATUS_SUCCESS;
        for (int i = 0; i < WARMUP_ITERS; i++) {
            warmup_status = ggml_backend_graph_compute(backend, gf);
        }
        ggml_backend_synchronize(backend);
        std::printf("diag_graph_compute_status_%s=%s\n", (T == 1) ? "T1" : "T32",
                    ggml_status_to_string(warmup_status));

        if (extra_diag) {
            // Confirm/deny the buffer-reuse hypothesis directly: are the
            // INPUT tensors (ids especially) still what we uploaded, after
            // the graph has been recomputed 100 more times on the same
            // allocation? diag_oneshot_up_l1 above was healthy; if ids has
            // changed here, gallocr let something overwrite an input.
            std::vector<int32_t> ids_readback(ids_host.size());
            ggml_backend_tensor_get(ids, ids_readback.data(), 0, ids_readback.size() * sizeof(int32_t));
            bool ids_match = (ids_readback == ids_host);
            std::printf("diag_ids_unchanged_after_warmup=%d\n", (int)ids_match);
            if (!ids_match) {
                std::printf("diag_ids_readback:");
                for (size_t k = 0; k < ids_readback.size() && k < 10; k++) std::printf(" %d", ids_readback[k]);
                std::printf("\n");
            }
            auto dump_stage = [&](const char *label, ggml_tensor *t) {
                std::vector<float> buf(ggml_nelements(t));
                ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
                char lbl[64];
                std::snprintf(lbl, sizeof(lbl), "diag_stage_%s_l1", label);
                std::printf("%s=%.6g\n", lbl, m1::l1_norm(buf.data(), buf.size()));
                std::snprintf(lbl, sizeof(lbl), "diag_stage_%s", label);
                m1::print_first_n(lbl, buf.data(), buf.size());
            };
            dump_stage("up", up);
            dump_stage("gate", gate);
            dump_stage("h", h);
            dump_stage("down_raw", down_raw);
            dump_stage("down_weighted", down);
        }

        std::vector<double> samples_us;
        samples_us.reserve(TIMED_ITERS);
        for (int i = 0; i < TIMED_ITERS; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            auto t1 = std::chrono::high_resolution_clock::now();
            samples_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        m1::TimingStats ts = m1::stats_from_samples_us(samples_us);

        size_t row_gate = ggml_row_size(GGML_TYPE_IQ3_S, D_MODEL);
        size_t row_up   = ggml_row_size(GGML_TYPE_IQ3_S, D_MODEL);
        size_t row_down = ggml_row_size(GGML_TYPE_IQ4_NL, N_FF);
        // Bytes touched by ONE token's MoE step (10 experts): the brief's
        // own definition. For T=32 we report gbs assuming 32 INDEPENDENT
        // token footprints (no cross-token expert-weight reuse credited),
        // which is the pessimistic/worst-case bound -- flagged explicitly
        // below rather than silently assumed.
        double bytes_per_step = (double)N_EXPERT_USED *
            ((double)row_gate * N_FF + (double)row_up * N_FF + (double)row_down * D_MODEL);

        const char *tag = (T == 1) ? "T1" : "T32";
        std::printf("ggml_moe_%s_us_median=%.3f\n", tag, ts.median_us);
        std::printf("ggml_moe_%s_us_p10=%.3f\n", tag, ts.p10_us);
        std::printf("ggml_moe_%s_us_p90=%.3f\n", tag, ts.p90_us);
        if (T == 1) {
            std::printf("ggml_bytes_per_step=%.0f\n", bytes_per_step);
            double gbs = bytes_per_step / (ts.median_us * 1e-6) / 1e9;
            std::printf("ggml_gbs_T1=%.3f\n", gbs);
        } else {
            double gbs32 = (bytes_per_step * (double)T) / (ts.median_us * 1e-6) / 1e9;
            std::printf("ggml_gbs_T32=%.3f\n", gbs32);
            std::printf("ggml_gbs_T32_assumption=32x_independent_token_footprints_no_reuse_credited\n");
        }
        std::printf("ggml_graph_nodes=%d\n", ggml_graph_n_nodes(gf));

        // ---- numerics check, T=1 only: CPU fp32 reference on DEQUANTISED
        // weights, same ids/weights/x, compared against the GPU output.
        if (T == 1) {
            std::vector<float> gpu_out(D_MODEL);
            ggml_backend_tensor_get(moe_out, gpu_out.data(), 0, D_MODEL * sizeof(float));

            std::vector<float> ref_out(D_MODEL, 0.0f);
            const ggml_type_traits *tt_gate = ggml_get_type_traits(GGML_TYPE_IQ3_S);
            const ggml_type_traits *tt_down = ggml_get_type_traits(GGML_TYPE_IQ4_NL);

            std::vector<uint8_t> row_buf_gate(row_gate), row_buf_up(row_up), row_buf_down(row_down);
            std::vector<float> gate_row_f(D_MODEL), up_row_f(D_MODEL), down_row_f(N_FF);
            std::vector<float> y_gate(N_FF), y_up(N_FF), h_vec(N_FF);

            for (int64_t k = 0; k < N_EXPERT_USED; k++) {
                int32_t e = ids_host[k];
                // gate/up: dequantise this expert's N_FF rows, dot with x.
                for (int64_t j = 0; j < N_FF; j++) {
                    size_t off = w.gate_exps->nb[2] * (size_t)e + w.gate_exps->nb[1] * (size_t)j;
                    ggml_backend_tensor_get(w.gate_exps, row_buf_gate.data(), off, row_gate);
                    tt_gate->to_float(row_buf_gate.data(), gate_row_f.data(), D_MODEL);
                    double acc = 0.0;
                    for (int64_t c = 0; c < D_MODEL; c++) acc += (double)gate_row_f[c] * x_host[c];
                    y_gate[j] = (float)acc;

                    off = w.up_exps->nb[2] * (size_t)e + w.up_exps->nb[1] * (size_t)j;
                    ggml_backend_tensor_get(w.up_exps, row_buf_up.data(), off, row_up);
                    tt_gate->to_float(row_buf_up.data(), up_row_f.data(), D_MODEL);
                    acc = 0.0;
                    for (int64_t c = 0; c < D_MODEL; c++) acc += (double)up_row_f[c] * x_host[c];
                    y_up[j] = (float)acc;

                    float g = y_gate[j];
                    float silu = g / (1.0f + std::exp(-g));
                    h_vec[j] = silu * y_up[j];
                }
                // down: dequantise this expert's D_MODEL rows, dot with h_vec.
                for (int64_t j = 0; j < D_MODEL; j++) {
                    size_t off = w.down_exps->nb[2] * (size_t)e + w.down_exps->nb[1] * (size_t)j;
                    ggml_backend_tensor_get(w.down_exps, row_buf_down.data(), off, row_down);
                    tt_down->to_float(row_buf_down.data(), down_row_f.data(), N_FF);
                    double acc = 0.0;
                    for (int64_t c = 0; c < N_FF; c++) acc += (double)down_row_f[c] * h_vec[c];
                    ref_out[j] += weights_host[k] * (float)acc;
                }
            }

            double gpu_l1 = m1::l1_norm(gpu_out.data(), D_MODEL);
            double ref_l1 = m1::l1_norm(ref_out.data(), D_MODEL);
            std::printf("diag_gpu_out_l1=%.6g\n", gpu_l1);
            m1::print_first_n("diag_gpu_out", gpu_out.data(), D_MODEL);
            std::printf("diag_ref_out_l1=%.6g\n", ref_l1);
            m1::print_first_n("diag_ref_out", ref_out.data(), D_MODEL);

            // Refuse a trivial pass: two all-zero vectors have an undefined
            // (NaN, from 0/0) cosine, which is NOT >= 0.999 and so used to
            // fail correctly by accident -- but a near-zero L1 on just one
            // side (a partially-broken path) could still cosine-pass by
            // chance. Check both norms explicitly and say so.
            if (gpu_l1 == 0.0 || ref_l1 == 0.0) {
                std::fprintf(stderr, "CHECK_FAIL zero output (gpu_out_l1=%.6g ref_out_l1=%.6g)\n",
                             gpu_l1, ref_l1);
                any_check_fail = true;
            } else {
                double cos = m1::cosine_similarity(gpu_out.data(), ref_out.data(), D_MODEL);
                double maxabs = m1::max_abs_diff(gpu_out.data(), ref_out.data(), D_MODEL);
                std::printf("ggml_check_maxabs=%.6g\n", maxabs);
                std::printf("ggml_check_cos=%.6g\n", cos);
                if (cos < 0.999) any_check_fail = true;
            }
        }

        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }

    if (any_check_fail) {
        std::fprintf(stderr, "CHECK_FAIL\n");
        return 1;
    }
    return 0;
}
