#!/bin/bash
# build_dbgskip.sh -- DEBUG-ONLY build of franken-engine `main` + `--debug-skip MASK` (a per-config run option; supersedes build_dbgmoe.sh's
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
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-dbgskip; OUT=$HOME/bench/franken_bin
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
  "    int prefill_stage = 0, stage_mb = 0, adapt_prefill = 1;\n    int dbg_skip = 0;                // --debug-skip MASK (debug: TIMING ONLY, see build_dbgskip.sh)"),
 ('    else if (a == "--glm-stage-mb")  { if (!(v = val())) return -1; r.stage_mb = std::atoi(v); }',
  '    else if (a == "--glm-stage-mb")  { if (!(v = val())) return -1; r.stage_mb = std::atoi(v); }\n'
  '    else if (a == "--debug-skip")    { if (!(v = val())) return -1; r.dbg_skip = std::atoi(v); if (r.dbg_skip & 16) r.dbg_skip |= 8; }'),
 ("    for (int d = 0; d < n_devices; ++d) gops[d]->set_prefill_stage(stage, r.stage_mb);",
  "    for (int d = 0; d < n_devices; ++d) gops[d]->set_prefill_stage(stage, r.stage_mb);\n"
  "    for (int d = 0; d < n_devices; ++d) gops[d]->set_debug_skip(r.dbg_skip);\n"
  "    if (r.dbg_skip) std::printf(\"debug_skip=%d (TIMING ONLY, garbage output)\\n\", r.dbg_skip);"),
])
sub('glm5_ops.h', [
 ("    virtual void set_prefill_stage(int on, int ring_mb) { (void) on; (void) ring_mb; }",
  "    virtual void set_prefill_stage(int on, int ring_mb) { (void) on; (void) ring_mb; }\n"
  "    virtual void set_debug_skip(int mask) { (void) mask; }   // DEBUG, TIMING ONLY: bits 1 moe, 2 trunk GEMM, 4 KDA, 8 attention, 16 indexer"),
])
sub('glm5_gpu.inc', [
 ("    int stage_mb_req_ = 0;", "    int stage_mb_req_ = 0;"),   # anchor check only
 ("    void set_prefill_stage(int on, int ring_mb) override {",
  "    void set_debug_skip(int mask) override { g_.dbg_skip_ = mask; }\n    void set_prefill_stage(int on, int ring_mb) override {"),
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
nice -n 19 make -j4 gpu GPU_BIN=franken_decode_glm_dbgskip 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -8
[ -x franken_decode_glm_dbgskip ] && cp -p franken_decode_glm_dbgskip "$OUT/" && sha256sum "$OUT/franken_decode_glm_dbgskip" | cut -c1-16
git -C "$SRC" worktree remove --force "$WT"; echo done
