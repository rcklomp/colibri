#!/bin/bash
# glm_d10_quality_chain.sh -- D10 (ACTION-LIST row 8): the task-level quality of the CPU lane (`FRANKEN_GLM_CPU_LANE=6` on top of the shipped config) through the REAL gateway,
# same protocol as §L5-PF2c / §L5-PF14-QUALITY: franken_quality_chain.sh ARMS=franken-glm, the frozen 70-question MMLU-Pro sample, reasoning xhigh, 16 000-token budget, greedy, then the
# needle set. The engine is the D10 serving build `franken_dec_glm_d10l` (NOT installed, franken-engine branch d10-cpu-lane) on the hybrid GGUF; results ~/bench/franken_quality/franken_glm_d10.jsonl.
# Why a task-level gate: a CPU slot runs ggml's Q8_K activation chain (CPU vs GPU row rel-L2 1.8 %, max 2.1 %, record §L5-D10-LANE); answers change.
# Bar (O4, set before the run): MMLU-Pro inside the Wilson interval of the installed .pf15 (55/70 = 78.6 %, 95 % [67.6, 86.6], i.e. fewer than 48 of 70 is a fail)
# and no needle miss the installed binary got right (8/8 at 30k/60k/120k/200k).
# franken_quality_chain.sh starts and stops the gateway itself (serve_alt.sh franken-glm) and leaves the service OFF while ~/bench/.dev_reserved exists. Launch it DIRECTLY, not through
# run_chain.sh (serve_alt.sh takes the rig lock itself). About 3.5-5 h.
# Env: D10Q_EG (16.85), D10Q_CHUNK (1024), D10Q_GGUF, D10Q_BIN, D10Q_LANE (6), QE_PARTS (default mmlu,needle).
set -u
export FRANKEN_DOCKER_BIN=${D10Q_BIN:-$HOME/bench/franken_bin/franken_dec_glm_d10l}
export FRANKEN_GGUF=${D10Q_GGUF:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
export FRANKEN_EXPERT_GB=${D10Q_EG:-16.85}
export FRANKEN_GLM_CHUNK=${D10Q_CHUNK:-1024}
export FRANKEN_GEMM_LDS=3
export FRANKEN_GEMM_LDS_MIN_ROWS=16
export FRANKEN_GEMM_LDS_SPLITK=1
export FRANKEN_GLM_ATTN_WMMA=64
export FRANKEN_GLM_CPU_LANE=${D10Q_LANE:-6}
[ "$FRANKEN_GLM_CHUNK" -gt 1024 ] && export FRANKEN_SNAP_EVERY=$FRANKEN_GLM_CHUNK
for f in "$FRANKEN_DOCKER_BIN" "$FRANKEN_GGUF"; do [ -e "$f" ] || { echo "FATAL: missing $f"; exit 2; }; done
grep -q FRANKEN_GLM_CPU_LANE "$HOME/bench/franken_decode_glm_docker.sh" || { echo "FATAL: ~/bench/franken_decode_glm_docker.sh does not pass FRANKEN_GLM_CPU_LANE (copy rig_copies/)"; exit 2; }
echo "=== glm_d10_quality $(date -Is) bin=$(sha256sum "$FRANKEN_DOCKER_BIN" | cut -c1-16) gguf=$(basename "$FRANKEN_GGUF") expert-gb=$FRANKEN_EXPERT_GB chunk=$FRANKEN_GLM_CHUNK attn_wmma=$FRANKEN_GLM_ATTN_WMMA cpu_lane=$FRANKEN_GLM_CPU_LANE"
ARMS=franken-glm QE_TAG=d10 exec "$HOME/src/colibri/tools/hot-expert/franken_quality_chain.sh"
