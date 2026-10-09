#!/bin/bash
# glm_hyb_quality_chain.sh -- PF14: the task-level quality of the IQ3_XXS-expert hybrid through the REAL gateway, same protocol as §L5-GLM-QUALITY (the served IQ4_XS: greedy MMLU-Pro 58/70 = 82.9 %,
# needle 8/8 at 30k/60k/120k/200k): franken_quality_chain.sh ARMS=franken-glm, 70-question frozen sample, reasoning xhigh, 16 000-token budget, greedy, then the needle set.
# The engine is the pf14 serving build (NOT installed) on the hybrid GGUF; results ~/bench/franken_quality/franken_glm_hyb.jsonl. About 4-5 h (decode ~51 ms/token) + the needles (~1 h).
# franken_quality_chain.sh starts and stops the gateway itself (serve_alt.sh franken-glm) and leaves the service OFF while ~/bench/.dev_reserved exists.
#   setsid nohup ~/bench/glm_hyb_quality_chain.sh > ~/bench/glm_hyb_quality_chain.log 2>&1 < /dev/null &
# Env: HYBQ_EG (expert GB a card, default 17), HYBQ_CHUNK (default 1024), HYBQ_GGUF, HYBQ_BIN, QE_PARTS (default mmlu,needle).
set -u
export FRANKEN_DOCKER_BIN=${HYBQ_BIN:-$HOME/bench/franken_bin/franken_dec_glm.pf14}
export FRANKEN_GGUF=${HYBQ_GGUF:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
export FRANKEN_EXPERT_GB=${HYBQ_EG:-17}
export FRANKEN_GLM_CHUNK=${HYBQ_CHUNK:-1024}
[ "$FRANKEN_GLM_CHUNK" -gt 1024 ] && export FRANKEN_SNAP_EVERY=$FRANKEN_GLM_CHUNK   # the prefill loop cuts chunks at snapshot points
for f in "$FRANKEN_DOCKER_BIN" "$FRANKEN_GGUF"; do [ -e "$f" ] || { echo "FATAL: missing $f"; exit 2; }; done
echo "=== glm_hyb_quality $(date -Is) bin=$(sha256sum "$FRANKEN_DOCKER_BIN" | cut -c1-16) gguf=$(basename "$FRANKEN_GGUF") expert-gb=$FRANKEN_EXPERT_GB chunk=$FRANKEN_GLM_CHUNK"
ARMS=franken-glm QE_TAG=hyb exec "$HOME/src/colibri/tools/hot-expert/franken_quality_chain.sh"
