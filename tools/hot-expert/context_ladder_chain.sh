#!/bin/bash
# context_ladder_chain.sh — performance vs context length, up to 64Ki tokens.
#
# Launch through run_chain.sh, never directly — it takes the rig lock, and the
# gateway watchdog (cron, every 5 min) restarts the gateway under any
# measurement that stopped it without one:
#
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh \
#       ~/src/colibri/tools/hot-expert/context_ladder_chain.sh \
#       > ~/bench/context_ladder.log 2>&1 < /dev/null &
#
# Steps:
#   1. stop the owner's gateway and WAIT for the engine to die (ETXTBSY trap)
#   2. warm the model and assert residency (the harness re-asserts per turn)
#   3. context_ladder.py — one growing conversation to 64Ki+, ~2-3 h
#   4. restart the gateway on EVERY exit path
#
# GLM53_MAXT is the engine's per-slot context (glm53.c:4697, default 8192, NOT a
# generation cap): a prompt at or above it is refused with BAD_REQUEST, and the
# slot needs room for the generated tokens on top of the prompt. 96Ki of slack
# over a 64Ki ladder at the engine's measured ~33 KB/token is ~3.2 GB for the one
# slot this conversation touches.
#
# GLM53_PREFIX_CKPT=0 with a private COLI_CKPT_DIR, as every gate on this track
# runs: with checkpoints on, a turn could restore a prefix a previous turn wrote
# and report a prefill that never happened.
set -u
TAG=ctx$(date +%m%d%H%M)
OUT=~/bench/ctx_ladder_out; mkdir -p "$OUT"
HERE=~/src/colibri/tools/hot-expert
BIN=~/src/colibri/c/glm53
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)

STEPS=${STEPS:-1024,1024,2048,4096,8192,8192,8192,8192,8192,8192,8192}
GEN=${GEN:-32}
FOLLOWUPS=${FOLLOWUPS:-2}
MAXT=${MAXT:-98304}
# 90, not ttft_serve's 96. A loaded glm53 holds ~89 GB of anon memory beside a
# 183 GiB model on a 247 GB box, so the shards CANNOT be 96% resident while the
# engine that is being measured is running: the first attempt here re-warmed
# twice (31 s each, reading 182 GiB and fighting the engine for the same pages)
# and climbed only 92.3 -> 94.0. ttft_serve.py's own comment records the same
# thing -- "the gateway's engine sits at ~91.6%". The floor is a proxy for the
# question that matters, which is whether the run is re-reading the model from
# NVMe inside the measurement; majflt per turn answers that directly and is
# recorded on every row, so it is the check to read in the results.
MIN_RESIDENT=${MIN_RESIDENT:-90}

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" \
         -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && break
    sleep 5
  done
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
}

stop_gateway() {
  pkill -f "openai_[s]erver.py" 2>/dev/null || true
  sleep 3
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 120); do pgrep -x glm53 >/dev/null || return 0; sleep 2; done
  echo "FATAL: an engine is still alive after 240 s"; return 1
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== context_ladder_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== results: $OUT/$TAG.jsonl"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== context_ladder_chain $TAG $(date -Is)"
echo "=== steps=$STEPS gen=$GEN followups=$FOLLOWUPS maxt=$MAXT"
echo "=== binary in service: $(sha256sum "$BIN" | cut -c1-16)"

stop_gateway || exit 1

echo "--- warming the model (one model at a time; 182 GiB)"
cat ~/models/GLM-5.3-Flash-colibri-int4-g64/*.safetensors > /dev/null 2>&1 || true

export GLM53_MAXT="$MAXT"
export GLM53_PREFIX_CKPT=0
export COLI_CKPT_DIR="$OUT/ckpt_$TAG"; mkdir -p "$COLI_CKPT_DIR"
export GLM53_VERBOSE=1

python3 "$HERE/context_ladder.py" \
    --engine "$BIN" \
    --steps "$STEPS" --gen "$GEN" --followups "$FOLLOWUPS" \
    --kv-slots 4 --warm --min-resident "$MIN_RESIDENT" \
    --tag "$TAG" --json "$OUT/$TAG.jsonl" \
    --engine-log "$OUT/engine_$TAG.log"
rc=$?
echo "--- context_ladder.py exit=$rc"
exit $rc
