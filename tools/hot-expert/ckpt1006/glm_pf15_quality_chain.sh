#!/bin/bash
# glm_pf15_quality_chain.sh -- PF15 (ACTION-LIST row 5i): the task-level quality of the WMMA attention (`FRANKEN_GLM_ATTN_WMMA=64` on top of the shipped config) through the REAL gateway,
# same protocol as §L5-PF2c / §L5-PF14-QUALITY: franken_quality_chain.sh ARMS=franken-glm, the frozen 70-question MMLU-Pro sample, reasoning xhigh, 16 000-token budget, greedy, then the
# needle set. The engine is the PF15 serving build (NOT installed) on the hybrid GGUF; results ~/bench/franken_quality/franken_glm_pf15.jsonl.
# Why a task-level gate: q and P go to f16 (kernel error vs float64 ~3e-4 RMS / 1.4e-2 max of the output RMS, bench_attn_wmma), and the final-position KL cannot separate arms
# (routing near-ties, record §L5-PF2c item 7). Bar (O4): MMLU-Pro inside the Wilson interval of the installed binary (59/70 = 84.3 %, 95 % [74.0, 91.0], i.e. fewer than 52 of 70 is a fail)
# and no needle miss the installed binary got right (8/8 at 30k/60k/120k/200k).
# franken_quality_chain.sh starts and stops the gateway itself (serve_alt.sh franken-glm) and leaves the service OFF while ~/bench/.dev_reserved exists. Launch it DIRECTLY, not through
# run_chain.sh (serve_alt.sh takes the rig lock itself). About 3.5-5 h.
# Env: PF15Q_EG (16.85), PF15Q_CHUNK (1024), PF15Q_GGUF, PF15Q_BIN, PF15Q_WMMA (64), QE_PARTS (default mmlu,needle).
set -u
export FRANKEN_DOCKER_BIN=${PF15Q_BIN:-$HOME/bench/franken_bin/franken_dec_glm.pf15}
export FRANKEN_GGUF=${PF15Q_GGUF:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
export FRANKEN_EXPERT_GB=${PF15Q_EG:-16.85}
export FRANKEN_GLM_CHUNK=${PF15Q_CHUNK:-1024}
export FRANKEN_GEMM_LDS=3
export FRANKEN_GEMM_LDS_MIN_ROWS=16
export FRANKEN_GEMM_LDS_SPLITK=1
export FRANKEN_GLM_ATTN_WMMA=${PF15Q_WMMA:-64}
[ "$FRANKEN_GLM_CHUNK" -gt 1024 ] && export FRANKEN_SNAP_EVERY=$FRANKEN_GLM_CHUNK
for f in "$FRANKEN_DOCKER_BIN" "$FRANKEN_GGUF"; do [ -e "$f" ] || { echo "FATAL: missing $f"; exit 2; }; done
grep -q FRANKEN_GLM_ATTN_WMMA "$HOME/bench/franken_decode_glm_docker.sh" || { echo "FATAL: ~/bench/franken_decode_glm_docker.sh does not pass FRANKEN_GLM_ATTN_WMMA (copy rig_copies/)"; exit 2; }
echo "=== glm_pf15_quality $(date -Is) bin=$(sha256sum "$FRANKEN_DOCKER_BIN" | cut -c1-16) gguf=$(basename "$FRANKEN_GGUF") expert-gb=$FRANKEN_EXPERT_GB chunk=$FRANKEN_GLM_CHUNK attn_wmma=$FRANKEN_GLM_ATTN_WMMA"
ARMS=franken-glm QE_TAG=pf15 exec "$HOME/src/colibri/tools/hot-expert/franken_quality_chain.sh"
