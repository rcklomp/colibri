#!/bin/bash
# glm_f16_quality_chain.sh -- PF2 (ACTION-LIST row 5g): the task-level quality of the f16 / rocBLAS trunk GEMM (`FRANKEN_GEMM_LDS=3`) through the REAL gateway, same protocol as §L5-PF14-QUALITY
# (the installed IQ3_XXS hybrid with the shipped f32 LDS GEMM: greedy MMLU-Pro 56/70 = 80.0 %, needle 8/8 at 30k/60k/120k/200k): franken_quality_chain.sh ARMS=franken-glm, the frozen
# 70-question sample, reasoning xhigh, 16 000-token budget, greedy, then the needle set. The engine is the f16 serving build (NOT installed) on the hybrid GGUF; results
# ~/bench/franken_quality/franken_glm_f16.jsonl. About 4-5 h (decode ~50 ms/token; prefill is a small share of it) + the needles (~1 h).
# PF2's stop rule: a drop outside the Wilson interval of the previous binary's 56/70 (80.0 %, 95 % interval 69.2-87.7 %, i.e. fewer than 49 of 70), or any needle miss the old binary got right.
# franken_quality_chain.sh starts and stops the gateway itself (serve_alt.sh franken-glm) and leaves the service OFF while ~/bench/.dev_reserved exists.
# Launch it DIRECTLY, not through run_chain.sh (serve_alt.sh takes the rig lock itself and refuses to start under another holder):
#   F16Q_EG=16.85 setsid nohup ~/bench/glm_f16_quality_chain.sh > ~/bench/glm_f16_quality_chain.log 2>&1 < /dev/null &
# Env: F16Q_EG (expert GB a card, default 17 here; the shipped config would give the f16 scratch back, see the plan), F16Q_CHUNK (1024), F16Q_GGUF, F16Q_BIN, QE_PARTS (default mmlu,needle).
set -u
export FRANKEN_DOCKER_BIN=${F16Q_BIN:-$HOME/bench/franken_bin/franken_dec_glm.f16}
export FRANKEN_GGUF=${F16Q_GGUF:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
export FRANKEN_EXPERT_GB=${F16Q_EG:-17}
export FRANKEN_GLM_CHUNK=${F16Q_CHUNK:-1024}
export FRANKEN_GEMM_LDS=3
[ "$FRANKEN_GLM_CHUNK" -gt 1024 ] && export FRANKEN_SNAP_EVERY=$FRANKEN_GLM_CHUNK
for f in "$FRANKEN_DOCKER_BIN" "$FRANKEN_GGUF"; do [ -e "$f" ] || { echo "FATAL: missing $f"; exit 2; }; done
echo "=== glm_f16_quality $(date -Is) bin=$(sha256sum "$FRANKEN_DOCKER_BIN" | cut -c1-16) gguf=$(basename "$FRANKEN_GGUF") expert-gb=$FRANKEN_EXPERT_GB chunk=$FRANKEN_GLM_CHUNK gemm_lds=$FRANKEN_GEMM_LDS"
ARMS=franken-glm QE_TAG=f16 exec "$HOME/src/colibri/tools/hot-expert/franken_quality_chain.sh"
