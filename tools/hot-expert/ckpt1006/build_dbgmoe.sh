#!/bin/bash
# build_dbgmoe.sh -- DEBUG-ONLY build: pf-embed-kernel + `--debug-skip-moe 0|1` (a per-config run option): when 1, every routed-expert GEMM launch
# (moe_launch) is skipped while the staging copies, events and everything else still run. Output is garbage: TIMING ONLY (no oracle configs).
# Purpose: does the in-engine staged DMA rate rise toward M4's 52 GB/s when the expert kernels are not running (record §M7-HOSTSRC)?
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-dbgmoe; OUT=$HOME/bench/franken_bin
git -C "$SRC" worktree add --detach "$WT" pf-embed-kernel || exit 1
cd "$WT/franken/decode" || exit 1
python3 - <<'PY'
p='glm5_graph.cpp'; s=open(p).read()
a="    int prefill_stage = 0, stage_mb = 0, adapt_prefill = 1;"
assert s.count(a)==1; s=s.replace(a,a+"\n    int skip_moe = 0;                // --debug-skip-moe 0|1 (debug: TIMING ONLY, the expert GEMMs are not launched)")
a="    else if (a == \"--glm-stage-mb\")  { if (!(v = val())) return -1; r.stage_mb = std::atoi(v); }"
assert s.count(a)==1; s=s.replace(a,a+"\n    else if (a == \"--debug-skip-moe\") { if (!(v = val())) return -1; r.skip_moe = std::atoi(v); }")
a="    for (int d = 0; d < n_devices; ++d) gops[d]->set_prefill_stage(stage, r.stage_mb);"
assert s.count(a)==1; s=s.replace(a,a+"\n    for (int d = 0; d < n_devices; ++d) gops[d]->set_debug_skip_moe(r.skip_moe);")
open(p,'w').write(s)
p='glm5_ops.h'; s=open(p).read()
a="    virtual void set_prefill_stage(int on, int ring_mb) { (void) on; (void) ring_mb; }"
assert s.count(a)==1; s=s.replace(a,a+"\n    virtual void set_debug_skip_moe(int on) { (void) on; }   // DEBUG, TIMING ONLY: do not launch the routed-expert GEMMs")
open(p,'w').write(s)
p='glm5_gpu.inc'; s=open(p).read()
a="    int stage_mb_req_ = 0;"
assert s.count(a)==1; s=s.replace(a,"    int dbg_skip_moe_ = 0;                           // --debug-skip-moe (TIMING ONLY)\n"+a)
a="    void set_prefill_stage(int on, int ring_mb) override {"
assert s.count(a)==1; s=s.replace(a,"    void set_debug_skip_moe(int on) override { dbg_skip_moe_ = on != 0; }\n"+a)
a="        const int staged = (rb % 16 == 0) && rb <= (size_t) glm5_row_max<TY>();\n        k_glm5_moe<TY>"
assert s.count(a)==1; s=s.replace(a,"        const int staged = (rb % 16 == 0) && rb <= (size_t) glm5_row_max<TY>();\n        if (dbg_skip_moe_) return;   // --debug-skip-moe: TIMING ONLY\n        k_glm5_moe<TY>")
open(p,'w').write(s)
PY
nice -n 19 make -j4 gpu GPU_BIN=franken_decode_glm_dbgmoe 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -5
[ -x franken_decode_glm_dbgmoe ] && cp -p franken_decode_glm_dbgmoe "$OUT/" && sha256sum "$OUT/franken_decode_glm_dbgmoe" | cut -c1-16
git -C "$SRC" worktree remove --force "$WT"; echo done
