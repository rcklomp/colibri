#!/bin/bash
# glm_pf2c_quality_chain.sh -- PF2c (ACTION-LIST row 5h): the task-level quality of the small-trunk-matrix change (`FRANKEN_GEMM_LDS_MIN_ROWS=16`, `FRANKEN_GEMM_LDS_SPLITK=1` on top of the shipped
# f16 arm `FRANKEN_GEMM_LDS=3`) through the REAL gateway, same protocol as §L5-PF2 item 7 / §L5-PF14-QUALITY: franken_quality_chain.sh ARMS=franken-glm, the frozen 70-question MMLU-Pro sample,
# reasoning xhigh, 16 000-token budget, greedy, then the needle set. The engine is the PF2c serving build (NOT installed) on the hybrid GGUF; results ~/bench/franken_quality/franken_glm_pf2c.jsonl.
# Why a task-level gate: the final-position KL of 24 windows cannot separate the arms (a 1e-6 relative change of the small matrices moves it as far as f16 rounding does: routing near-ties flip,
# record §L5-PF2c); the kernel itself is checked against float64 (bench_gemm_small). Bar (O4, as applied to PF2): MMLU-Pro inside the Wilson interval of the previous binary (55/70 = 78.6 %,
# 95 % [67.6, 86.6], i.e. fewer than 48 of 70 is a fail) and no needle miss the previous binary got right (8/8 at 30k/60k/120k/200k).
# franken_quality_chain.sh starts and stops the gateway itself (serve_alt.sh franken-glm) and leaves the service OFF while ~/bench/.dev_reserved exists. Launch it DIRECTLY, not through
# run_chain.sh (serve_alt.sh takes the rig lock itself). Start thermal_guard.sh once it is up (GUARD_WATCH="glm_pf2c_quality_chai[n]"). About 3.5-5 h.
# Env: PF2CQ_EG (16.85), PF2CQ_CHUNK (1024), PF2CQ_GGUF, PF2CQ_BIN, QE_PARTS (default mmlu,needle).
set -u
export FRANKEN_DOCKER_BIN=${PF2CQ_BIN:-$HOME/bench/franken_bin/franken_dec_glm.pf2c}
export FRANKEN_GGUF=${PF2CQ_GGUF:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
export FRANKEN_EXPERT_GB=${PF2CQ_EG:-16.85}
export FRANKEN_GLM_CHUNK=${PF2CQ_CHUNK:-1024}
export FRANKEN_GEMM_LDS=3
export FRANKEN_GEMM_LDS_MIN_ROWS=16
export FRANKEN_GEMM_LDS_SPLITK=1
[ "$FRANKEN_GLM_CHUNK" -gt 1024 ] && export FRANKEN_SNAP_EVERY=$FRANKEN_GLM_CHUNK
for f in "$FRANKEN_DOCKER_BIN" "$FRANKEN_GGUF"; do [ -e "$f" ] || { echo "FATAL: missing $f"; exit 2; }; done
echo "=== glm_pf2c_quality $(date -Is) bin=$(sha256sum "$FRANKEN_DOCKER_BIN" | cut -c1-16) gguf=$(basename "$FRANKEN_GGUF") expert-gb=$FRANKEN_EXPERT_GB chunk=$FRANKEN_GLM_CHUNK gemm_lds=$FRANKEN_GEMM_LDS min_rows=$FRANKEN_GEMM_LDS_MIN_ROWS splitk=$FRANKEN_GEMM_LDS_SPLITK"
ARMS=franken-glm QE_TAG=pf2c exec "$HOME/src/colibri/tools/hot-expert/franken_quality_chain.sh"
