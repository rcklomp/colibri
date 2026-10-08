#!/bin/bash
# build_pf0dbg.sh -- build_dbgskip.sh (2026-10-06, unchanged patch set) PLUS `--debug-route FILE` (PF0 (d), plan Rev 102): a per-config option that, for every
# prefill chunk (T > 1) and MoE layer, writes the router's expert ids and the layer's miss flags to FILE.devN (one line `R il= dev= T= miss=<288 bits> ids=<T*8 ints>`),
# so the in-place hit rate / link bytes by layer and chunk, and what a different resident set would save, are computed offline. A stream sync per layer: a config that
# runs --debug-route is for ANALYSIS, its timing is void. Output binary: franken_decode_glm_pf0dbg (mask 0 and no route = the shipped path plus a few not-taken branches).
# Original header of build_dbgskip.sh: DEBUG-ONLY build of franken-engine `main` + `--debug-skip MASK` (a per-config run option; supersedes build_dbgmoe.sh's
# `--debug-skip-moe`). TIMING ONLY: the output is garbage, there is no oracle config. Staging copies, events, norms, hyper-connection,
# router, plan and boundary P2P always run; the MASK removes the heavy kernels, so the staged DMA can be timed against less and less compute:
#     1  routed-expert GEMMs (moe_launch; what --debug-skip-moe 1 did)
#     2  trunk GEMMs/GEMVs (GpuBackend::gemv_batch_impl -- the --gemm-lds path included -- Ds4Ops::gemv's k-quant kernels, head_gemv)
#     4  KDA (kda_conv, kda_gate, kda_step)
#     8  MLA attention (the attn_part / attn_combine pair)
#    16  the indexer (idx_pool, idx_scores, idx_topk); implies 8 -- attention would read a stale `sel`
# 31 is the pure skeleton. A garbage router is safe: k_glm5_router's ids stay in [0, 288) even on NaN logits and k_glm5_chunk_plan only
# compares them. Purpose: record §M7-HOSTSRC left a factor ~2 in what the engine does beside the copies (27 GB/s with the experts
# skipped, 57 for the probe's replay of the same lists); handoff 2026-10-06 §5.1(a).
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-pf0dbg; OUT=$HOME/bench/franken_bin
git -C "$SRC" worktree add --detach "$WT" main || exit 1
cd "$WT/franken/decode" || exit 1
python3 - <<'PY'
def sub(p, pairs):
    s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70], s.count(a))
        s = s.replace(a, b)
    open(p, 'w').write(s)

sub('glm5_graph.cpp', [
 ("    int prefill_stage = 0, stage_mb = 0, adapt_prefill = 1;",
  "    int prefill_stage = 0, stage_mb = 0, adapt_prefill = 1;\n    int dbg_skip = 0;                // --debug-skip MASK (debug: TIMING ONLY, see build_dbgskip.sh)\n    std::string dbg_route;           // --debug-route FILE (analysis only: a stream sync per layer)"),
 ('    else if (a == "--glm-stage-mb")  { if (!(v = val())) return -1; r.stage_mb = std::atoi(v); }',
  '    else if (a == "--glm-stage-mb")  { if (!(v = val())) return -1; r.stage_mb = std::atoi(v); }\n'
  '    else if (a == "--debug-skip")    { if (!(v = val())) return -1; r.dbg_skip = std::atoi(v); if (r.dbg_skip & 16) r.dbg_skip |= 8; }\n'
  '    else if (a == "--debug-route")   { if (!(v = val())) return -1; r.dbg_route = v; }'),
 ("    for (int d = 0; d < n_devices; ++d) gops[d]->set_prefill_stage(stage, r.stage_mb);",
  "    for (int d = 0; d < n_devices; ++d) gops[d]->set_prefill_stage(stage, r.stage_mb);\n"
  "    for (int d = 0; d < n_devices; ++d) gops[d]->set_debug_skip(r.dbg_skip);\n"
  "    if (r.dbg_skip) std::printf(\"debug_skip=%d (TIMING ONLY, garbage output)\\n\", r.dbg_skip);\n"
  "    for (int d = 0; d < n_devices; ++d) gops[d]->set_debug_route(r.dbg_route.empty() ? nullptr : r.dbg_route.c_str(), d);\n"
  "    if (!r.dbg_route.empty()) std::printf(\"debug_route=%s.devN (ANALYSIS ONLY, timing void)\\n\", r.dbg_route.c_str());"),
 ("    g.moe_plan(L.et, s.ids, s.xn, L.dev, cfg_.links, (7 * il) % 11, il & 1, T);",
  "    g.dbg_layer(il);\n    g.moe_plan(L.et, s.ids, s.xn, L.dev, cfg_.links, (7 * il) % 11, il & 1, T);"),
])
sub('glm5_ops.h', [
 ("    virtual void set_prefill_stage(int on, int ring_mb) { (void) on; (void) ring_mb; }",
  "    virtual void set_prefill_stage(int on, int ring_mb) { (void) on; (void) ring_mb; }\n"
  "    virtual void set_debug_skip(int mask) { (void) mask; }   // DEBUG, TIMING ONLY: bits 1 moe, 2 trunk GEMM, 4 KDA, 8 attention, 16 indexer\n"
  "    virtual void set_debug_route(const char * path, int dev) { (void) path; (void) dev; }   // DEBUG: --debug-route (ids + miss flags of every chunk layer)\n"
  "    virtual void dbg_layer(int il) { (void) il; }"),
])
sub('glm5_gpu.inc', [
 ("    int stage_mb_req_ = 0;", "    int stage_mb_req_ = 0;\n    FILE * dbg_rf_ = nullptr;   // --debug-route\n    int dbg_il_ = -1;"),
 ("        if (fetch_ctr_) HIP_IGNORE(hipFree(fetch_ctr_));", "        if (dbg_rf_) { std::fclose(dbg_rf_); dbg_rf_ = nullptr; }\n        if (fetch_ctr_) HIP_IGNORE(hipFree(fetch_ctr_));"),
 ("            cx_[b] = x;                                  // the helpers copy the chunk's x rows from here",
  "            cx_[b] = x;                                  // the helpers copy the chunk's x rows from here\n"
  "            if (dbg_rf_) {   // --debug-route: ANALYSIS ONLY (a stream sync per layer; this config's timing is void)\n"
  "                dev_ensure(g_.dev_);\n"
  "                HIP_CHECK(hipStreamSynchronize(g_.stream_));\n"
  "                std::vector<int> hid((size_t) T * G5::N_EXPERT_USED), hms(G5::N_EXPERT);\n"
  "                HIP_CHECK(hipMemcpy(hid.data(), ids, hid.size() * sizeof(int), hipMemcpyDeviceToHost));\n"
  "                HIP_CHECK(hipMemcpy(hms.data(), t.miss_bytes, hms.size() * sizeof(int), hipMemcpyDeviceToHost));\n"
  "                std::fprintf(dbg_rf_, \"R il=%d dev=%d T=%d miss=\", dbg_il_, g_.dev_, T);\n"
  "                for (int v : hms) std::fputc(v ? '1' : '0', dbg_rf_);\n"
  "                std::fprintf(dbg_rf_, \" ids=\");\n"
  "                for (int v : hid) std::fprintf(dbg_rf_, \"%d,\", v);\n"
  "                std::fputc('\\n', dbg_rf_); std::fflush(dbg_rf_);\n"
  "            }"),
 ("    void set_prefill_stage(int on, int ring_mb) override {",
  "    void set_debug_skip(int mask) override { g_.dbg_skip_ = mask; }\n"
  "    void dbg_layer(int il) override { dbg_il_ = il; }\n"
  "    void set_debug_route(const char * path, int dev) override {\n"
  "        if (dbg_rf_) { std::fclose(dbg_rf_); dbg_rf_ = nullptr; }\n"
  "        if (path && *path) { const std::string p = std::string(path) + \".dev\" + std::to_string(dev); dbg_rf_ = std::fopen(p.c_str(), \"w\"); }\n"
  "    }\n"
  "    void set_prefill_stage(int on, int ring_mb) override {"),
 ("        k_glm5_moe<TY><<<dim3((unsigned) ceil_div(rows, DS4_GEMV_ROWS)",
  "        if (g_.dbg_skip_ & 1) return;   // --debug-skip: TIMING ONLY\n        k_glm5_moe<TY><<<dim3((unsigned) ceil_div(rows, DS4_GEMV_ROWS)"),
 ('        g_.mark(PC_GLM_KDA_CONV); g_.last_op_ = "glm5_kda_conv";',
  '        if (g_.dbg_skip_ & 4) return;\n        g_.mark(PC_GLM_KDA_CONV); g_.last_op_ = "glm5_kda_conv";'),
 ('        g_.mark(PC_GLM_KDA_CONV); g_.last_op_ = "glm5_kda_gate";',
  '        if (g_.dbg_skip_ & 4) return;\n        g_.mark(PC_GLM_KDA_CONV); g_.last_op_ = "glm5_kda_gate";'),
 ('        g_.mark(PC_GLM_KDA); g_.last_op_ = "glm5_kda_step";',
  '        if (g_.dbg_skip_ & 4) return;\n        g_.mark(PC_GLM_KDA); g_.last_op_ = "glm5_kda_step";'),
 ('        g_.mark(PC_GLM_IDX); g_.last_op_ = "glm5_idx_pool";',
  '        if (g_.dbg_skip_ & 16) return;\n        g_.mark(PC_GLM_IDX); g_.last_op_ = "glm5_idx_pool";'),
 ("        const int n = (bpos(pos) + T) / G5::KPOOL;\n        if (n <= 0) return;",
  "        if (g_.dbg_skip_ & 16) return;\n        const int n = (bpos(pos) + T) / G5::KPOOL;\n        if (n <= 0) return;"),
 ('        g_.mark(PC_GLM_IDX_TOPK); g_.last_op_ = "glm5_topk";',
  '        if (g_.dbg_skip_ & 16) return;\n        g_.mark(PC_GLM_IDX_TOPK); g_.last_op_ = "glm5_topk";'),
 ('        if (W.type != FK_Q_Q8_0) die("head_gemv: the batched per-head kernel is Q8_0 only");',
  '        if (g_.dbg_skip_ & 2) return;\n        if (W.type != FK_Q_Q8_0) die("head_gemv: the batched per-head kernel is Q8_0 only");'),
 ('        if (!po_) die("mla_attn before reserve()");',
  '        if (g_.dbg_skip_ & 8) return;\n        if (!po_) die("mla_attn before reserve()");'),
])
sub('ds4_gpu.inc', [
 ("    void gemv(const Mat & W, const float * x, float * y, int T) override {\n        switch (W.type) {",
  "    void gemv(const Mat & W, const float * x, float * y, int T) override {\n        if (g_.dbg_skip_ & 2) return;   // --debug-skip: TIMING ONLY\n        switch (W.type) {"),
])
sub('decode_gpu.hip', [
 ("        GemvBatchD b{};\n        b.n = n < GEMV_BATCH_MAX ? n : GEMV_BATCH_MAX;",
  "        if (dbg_skip_ & 2) return false;   // --debug-skip: TIMING ONLY\n        GemvBatchD b{};\n        b.n = n < GEMV_BATCH_MAX ? n : GEMV_BATCH_MAX;"),
 ("    int        gemm_lds_ = 0;", "    int        gemm_lds_ = 0;\n    int        dbg_skip_ = 0;   // --debug-skip MASK (TIMING ONLY; Glm5GpuOps::set_debug_skip)"),
])
PY
[ $? -eq 0 ] || { echo "PATCH FAILED"; git -C "$SRC" worktree remove --force "$WT"; exit 1; }
nice -n 19 make -j4 gpu GPU_BIN=franken_decode_glm_pf0dbg 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -8
[ -x franken_decode_glm_pf0dbg ] && cp -p franken_decode_glm_pf0dbg "$OUT/" && sha256sum "$OUT/franken_decode_glm_pf0dbg" | cut -c1-16
git -C "$SRC" worktree remove --force "$WT"; echo done
