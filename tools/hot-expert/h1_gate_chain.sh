#!/bin/bash
# h1_gate_chain.sh -- H1 gate (i): engine-mode context_ladder.py --url/--engine
# port reproduces the pre-H1 smoke run (commit eb4dd5b) within this box's
# 3-5% spread. Modelled directly on context_ladder_chain.sh (2f51379..eb4dd5b);
# the only difference is which STEPS/GEN/FOLLOWUPS/MAXT it drives and that it
# ends by proving the request AFTER the measurement (accept_live.sh), which
# context_ladder_chain.sh does not do because it is not a serving gate.
#
# Launch through run_chain.sh, never directly -- see context_ladder_chain.sh's
# own header for why (the lock, the watchdog):
#
#   setsid nohup ~/src/colibri-h1/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-h1/tools/hot-expert/h1_gate_chain.sh \
#       > ~/bench/h1_gate.log 2>&1 < /dev/null &
#
# The historical smoke run's own invocation (found in ~/bench/ctx_smoke3.log,
# ~/bench/ctx_ladder_out/ctx09151956.jsonl): STEPS=256,256,512 GEN=8
# FOLLOWUPS=1 MAXT=8192, giving turn 1 = 374 tokens cold in 45.44s
# (121.51 ms/token), turn 2 REUSE 381/731 at 110.80 ms/token (eb4dd5b's commit
# body). Reproducing those numbers is the gate; this script's only job is to
# run the SAME configuration through the NEW context_ladder.py (H1's --url
# argparse/driver-registry changes) in --engine mode, so the port is proven
# harmless on the path it did not change before it is trusted on the path it
# did.
#
# H1's engine changed nothing in glm53.c, so this gate runs the BINARY ALREADY
# IN SERVICE (~/src/colibri/c/glm53, and its built shaders) from the NEW
# harness code checked out at ~/src/colibri-h1. A fresh git clone's c/shaders
# has the .comp SOURCE (committed) but not the built .spv files, and
# ttft_serve.shaders_dir() would find the source-only directory first and
# silently point Vulkan there -- so COLI_VK_SHADERS is exported explicitly,
# below, at the pristine checkout's built path.
set -u
TAG=h1$(date +%m%d%H%M)
OUT=~/bench/ctx_ladder_out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PRISTINE=~/src/colibri
BIN="$PRISTINE/c/glm53"
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)

STEPS=${STEPS:-256,256,512}
GEN=${GEN:-8}
FOLLOWUPS=${FOLLOWUPS:-1}
MAXT=${MAXT:-8192}
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

# wait_no_engine: p7_gate.sh's pattern (CLAUDE.md) -- a chain's next step must
# not start until the previous engine is actually gone, or the pristine `cp`/
# spawn races an ETXTBSY.
wait_no_engine() {
  for _ in $(seq 1 120); do pgrep -x glm53 >/dev/null || return 0; sleep 2; done
  echo "FATAL: an engine is still alive after 240 s"; return 1
}

stop_gateway() {
  pkill -f "openai_[s]erver.py" 2>/dev/null || true
  sleep 3
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_engine
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== h1_gate_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== results: $OUT/$TAG.jsonl"
  echo "--- accept_live.sh (the chain must end with this passing, not just the ladder)"
  "$HERE/accept_live.sh"
  alive_rc=$?
  echo "--- accept_live.sh exit=$alive_rc"
  [ "$rc" -eq 0 ] && rc=$alive_rc
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== h1_gate_chain $TAG $(date -Is)"
echo "=== steps=$STEPS gen=$GEN followups=$FOLLOWUPS maxt=$MAXT"
echo "=== binary in service: $(sha256sum "$BIN" | cut -c1-16)"
echo "=== harness checkout: $(git -C "$HERE/../.." rev-parse --short HEAD 2>/dev/null) ($HERE)"

stop_gateway || exit 1

echo "--- warming the model (one model at a time; 182 GiB)"
cat ~/models/GLM-5.3-Flash-colibri-int4-g64/*.safetensors > /dev/null 2>&1 || true

# Assert residency BEFORE the engine starts (not only inside context_ladder.py,
# which can only check after EngineDriver has already spawned it): a loaded
# glm53 is ~89 GB of anon memory next to the 183 GiB model (context_ladder.py's
# own --min-resident 90 note), so this is the last point at which the floor
# means "the model, undisturbed".
PCT=$(fincore --bytes --output SIZE,RES ~/models/GLM-5.3-Flash-colibri-int4-g64/*.safetensors 2>/dev/null \
      | tail -n +2 | awk '{ts+=$1; rs+=$2} END{if (ts>0) printf "%.4f", 100*rs/ts; else print 0}')
echo "[resid pre-engine] resident=${PCT}%"
if ! awk -v p="$PCT" -v m="$MIN_RESIDENT" 'BEGIN{exit !(p>=m)}'; then
  echo "FATAL: ${PCT}% resident < ${MIN_RESIDENT}% before the engine even started"; exit 1
fi

export GLM53_MAXT="$MAXT"
export GLM53_PREFIX_CKPT=0
export COLI_CKPT_DIR="$OUT/ckpt_$TAG"; mkdir -p "$COLI_CKPT_DIR"
export GLM53_VERBOSE=1
export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
# The pristine checkout's BUILT shaders (see the header note) -- must win over
# whatever ttft_serve.shaders_dir() would find relative to $HERE.
export COLI_VK_SHADERS="$PRISTINE/c/shaders"

python3 "$HERE/context_ladder.py" \
    --engine "$BIN" \
    --steps "$STEPS" --gen "$GEN" --followups "$FOLLOWUPS" \
    --kv-slots 4 --warm --min-resident "$MIN_RESIDENT" \
    --arm h1gate --tag "$TAG" --json "$OUT/$TAG.jsonl" \
    --engine-log "$OUT/engine_$TAG.log"
rc=$?
echo "--- context_ladder.py exit=$rc"
exit $rc
