#!/bin/bash
# f1_step0_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md sec 8.3 item F1, step
# 0: split the "[PROF] eg=" window (record sec RP-F6a: "the last request's
# [PROF] line reads eg=6.913s cpu=6.512s over 69 forwards ... the two run
# concurrently and this profile cannot say how much of ffn_moe a vanished CPU
# path would return") into issue / cpu_in / take, so a later F1 step can read
# off whether the CPU-computed non-resident experts are actually on decode's
# critical path (take ~ 0 with cpu_in > 0) or whether the GPU fence wait is
# (take large). TIMERS ONLY -- c/glm53.c's ffn_layer computes the same thing
# in the same order; only new clock reads and three new accumulators were
# added (g_t_eg_issue, g_t_eg_cpu, g_t_eg_take, g_n_eg_win, g_n_eg_cpu0,
# g_t_eg_take_cpu0), printed as a second "[PROF] eg split: ..." line and, per
# request, as a second "[OPTIME req=... ] eg split: ..." line. No shader, no
# backend_vulkan.c, no streamed-prefill logic touched.
#
# Phase 1: a SHORT CLI oracle, candidate vs the served pristine
# (~/bench/glm53.f2, sha 5c01246c), same env as service (the clamp and
# streaming knobs ON on both sides, since that is what serves today), COLI_
# TIMERS=1 on both (irrelevant to text/logits, proves the knob doesn't leak
# into the accumulators some other way). gate_compare on teacher_forcing:
# any miss aborts the whole chain before the ladder runs -- a timers-only
# change has no business moving output.
#
# Phase 2: ONE ladder arm on the candidate only (no A/B: there is no second
# arm, the pristine and the candidate compute identically by the oracle
# above), steps 1024,1024,2048,4096,8192, gen=128 (>=64, so the decode
# sample per turn is not a 20-token sliver), followups=2, COLI_TIMERS=1.
# The "[PROF] eg split" / "[OPTIME ...] eg split" lines in the resulting
# engine log are the whole point of this chain; the record is written from
# them, not from a speed verdict (this step changes no numerics and adds a
# handful of clock reads -- it is not expected to move tok/s at all).
#
# Env matches service (CLAUDE.md, record sec F2-LADDER):
#   OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
#   COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
#   COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 COLI_KDA_GPU=2
#   GLM53_VK_SWIGLU_CLAMP=1 COLI_PREFILL_STREAM=1 GLM53_PREFILL_CHUNK=512
#   COLI_TIMERS=1
# plus a frozen COPY of ~/.glm53_explain.bin (never the canonical file --
# CLAUDE.md/record sec F2b: an unfrozen histogram or none at all preloads a
# tier that never serves) and GLM53_PREFIX_CKPT=0 with a private
# COLI_CKPT_DIR on every invocation (a restored checkpoint reports a prefill
# that never happened).
#
# Launch ONLY through run_chain.sh (it takes the rig lock; this script does
# not stop the gateway on its own without it):
#   setsid nohup ~/src/colibri-f1s0/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f1s0/tools/hot-expert/f1_step0_chain.sh \
#       >> ~/bench/f1_step0.log 2>&1 < /dev/null &
#
# The build (`make -C c glm53 VK=1` in ~/src/colibri-f1s0) happens BEFORE
# this is launched, not inside it, and needs no lock.
#
# Budget: oracle ~2-3 min, ladder ~26 min (record sec F2-LADDER: the same
# rung's cold prefill to 18 439 tokens is ~21-38 min depending on the arm,
# plus the decode samples) -- target total gateway-down time <= 40 min.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
F1_ROOT=$(cd "$HERE/../.." && pwd)                       # e.g. ~/src/colibri-f1s0
PRISTINE=~/src/colibri                                   # the served tree
PRISTINE_BIN=${F1_PRISTINE_BIN:-~/bench/glm53.f2}        # served since 2026-09-19 16:02 UTC, sha 5c01246c
PRISTINE_SHADERS=${F1_PRISTINE_SHADERS:-$PRISTINE/c/shaders}
CAND_BIN="$F1_ROOT/c/glm53"
CAND_SHADERS="$F1_ROOT/c/shaders"
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
OUT=~/bench/f1s0_out; mkdir -p "$OUT"
TAG=f1s0$(date +%m%d%H%M)
. "$HERE/gate_lib.sh"

A_STEPS=${F1_STEPS:-1024,1024,2048,4096,8192}
A_GEN=${F1_GEN:-128}          # decode tokens per turn -- >=64 so the sample is not a 20-token sliver
A_FOLLOWUPS=${F1_FOLLOWUPS:-2}
ORACLE_CHARS=${F1_ORACLE_CHARS:-6000}   # short: ~1.5k tokens, enough for a real teacher_forcing line

echo "=== f1_step0_chain $TAG $(date -Is)"
echo "=== candidate=$CAND_BIN ($([ -x "$CAND_BIN" ] && sha256sum "$CAND_BIN" | cut -c1-16 || echo MISSING))"
echo "=== pristine=$PRISTINE_BIN ($([ -x "$PRISTINE_BIN" ] && sha256sum "$PRISTINE_BIN" | cut -c1-16 || echo MISSING))"
echo "=== ladder steps=$A_STEPS gen=$A_GEN followups=$A_FOLLOWUPS  oracle_chars=$ORACLE_CHARS"
echo "=== THE GATEWAY IS DOWN FOR THE WHOLE CHAIN"

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }

precheck() {   # precheck <label>
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  for e in glm53 qwen38 qwen38-vk; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  echo "[$label] pgrep -f 'glm53[.]': $(pgrep -f "glm53[.]" | wc -l)"
  for c in 0 1 2; do
    echo "[$label] card$c vram_used=$(VRAM "$c")"
  done
}

assert_vram_free() {   # assert_vram_free <label>
  local label=$1 v c bad attempt
  for attempt in $(seq 1 30); do
    bad=0
    for c in 0 1 2; do
      v=$(VRAM "$c")
      [ "$v" -lt 0 ] || [ "$v" -ge 1073741824 ] && bad=1
    done
    [ "$bad" = 0 ] && { [ "$attempt" -gt 1 ] && echo "[$label] VRAM free after ${attempt} checks (~$(( (attempt-1) * 2 ))s)"; return 0; }
    sleep 2
  done
  for c in 0 1 2; do
    v=$(VRAM "$c")
    if [ "$v" -lt 0 ] || [ "$v" -ge 1073741824 ]; then
      echo "FATAL [$label]: card$c VRAM $v >= 1 GiB (or unreadable) after 60s of retries"
    fi
  done
  return 1
}

wait_no_engine() {
  for _ in $(seq 1 120); do
    if ! pgrep -x glm53 >/dev/null && ! pgrep -f "glm53[.]" >/dev/null; then return 0; fi
    sleep 2
  done
  echo "FATAL: a glm53 is still alive after 240 s"; return 1
}

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT -u COLI_TIMERS \
      -u COLI_PREFILL_STREAM -u GLM53_PREFILL_CHUNK -u GLM53_EXPERTS_CPU \
      -u GLM53_VK_SWIGLU_CLAMP -u COLI_USAGE_PATH -u COLI_KDA_GPU \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh >> "$GLOG" 2>&1 < /dev/null &
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
  wait_no_engine
}

warm_glm() {
  echo "--- warming GLM shards (one model at a time; 182 GiB)"
  cat "$GLM_SNAP"/*.safetensors > /dev/null 2>&1 || true
}

assert_glm_resident() {   # assert_glm_resident <label>
  local label=$1 pct
  pct=$(fincore --bytes --output SIZE,RES "$GLM_SNAP"/*.safetensors 2>/dev/null \
        | tail -n +2 | awk '{ts+=$1; rs+=$2} END{if (ts>0) printf "%.4f", 100*rs/ts; else print 0}')
  echo "[resid $label] resident=${pct}%"
  awk -v p="$pct" 'BEGIN{exit !(p>=90)}'
}

# --------------------------------------------------------------- main --
on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  if pgrep -x glm53 >/dev/null 2>&1 || pgrep -f "glm53[.]" >/dev/null 2>&1; then
    pkill -9 -x glm53 2>/dev/null || true
    wait_no_engine
  fi
  echo "--- final re-warm of GLM before restart"
  warm_glm
  assert_glm_resident "final" || echo "WARNING: GLM not >=90% resident at restart time"
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== f1_step0_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== results dir: $OUT (tag $TAG)"
  echo "--- accept_live.sh (the request AFTER the chain is part of the measurement)"
  "$HERE/accept_live.sh"
  alive_rc=$?
  echo "--- accept_live.sh exit=$alive_rc"
  [ "$rc" -eq 0 ] && rc=$alive_rc
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

precheck "chain-start"
[ -x "$CAND_BIN" ] || { echo "FATAL: $CAND_BIN missing -- build before launching"; exit 1; }
[ -x "$PRISTINE_BIN" ] || { echo "FATAL: $PRISTINE_BIN missing"; exit 1; }

stop_gateway || exit 1
precheck "after-stop"

warm_glm
assert_glm_resident "pre" || { echo "FATAL: GLM not >=90% resident before the chain"; exit 1; }

# Frozen usage histogram COPY -- record sec F2b: the CLI oracle path does not
# get one for free the way ttft_serve.py's engine mode does, and an unfrozen
# / canonical file is either the wrong tier or gets mutated by the run that
# reads it.
HIST="$OUT/${TAG}_hist.bin"
cp ~/.glm53_explain.bin "$HIST" || { echo "FATAL: no ~/.glm53_explain.bin to freeze"; exit 1; }
echo "--- frozen usage histogram: $HIST ($(stat -c %s "$HIST") bytes)"

ORACLE_PACKET="$OUT/${TAG}_oracle_packet.txt"
python3 - "$F1_ROOT/tools/hot-expert/ROME-3x7900XTX-2026-09-04.md" "$ORACLE_PACKET" "$ORACLE_CHARS" <<'PY'
import sys
src, dst, n = sys.argv[1:4]
text = open(src, encoding="utf-8").read()
q = "\n\nContinue summarising the notes above in a few sentences, without repeating what you already said."
open(dst, "w").write(text[:int(n)] + q)
PY
[ -s "$ORACLE_PACKET" ] || { echo "FATAL: could not build the oracle packet"; exit 1; }

# ======================================================================
# Phase 1: the oracle -- candidate vs the served pristine, SAME env as
# service on both sides (the clamp and streaming knobs are ON in service
# today; this chain does not vary them, it only proves the timers-only
# change didn't move anything).
# ======================================================================
run_oracle() {   # run_oracle <side> <bin> <shaders>
  local side=$1 bin=$2 shaders=$3
  for e in glm53 qwen38 qwen38-vk; do
    pgrep -x "$e" >/dev/null 2>&1 && { echo "FATAL: $e already running before oracle-$side"; return 9; }
  done
  rm -rf "$OUT/ckpt_${TAG}_oracle_${side}"; mkdir -p "$OUT/ckpt_${TAG}_oracle_${side}"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_KDA_GPU=2
    export GLM53_VK_SWIGLU_CLAMP=1 COLI_PREFILL_STREAM=1 GLM53_PREFILL_CHUNK=512
    export COLI_TIMERS=1
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt_${TAG}_oracle_${side}"
    export COLI_USAGE_PATH="$HIST"
    export GLM53_VERBOSE=1
    "$bin" --model "$GLM_SNAP" --prompt "$(cat "$ORACLE_PACKET")" --logits --greedy 0
  ) > "$OUT/${TAG}_oracle_${side}.out" 2> "$OUT/${TAG}_oracle_${side}.err"
  local rc=$?
  echo "[oracle] $side rc=$rc $(grep -c ^teacher_forcing "$OUT/${TAG}_oracle_${side}.out") tf-line(s)"
  echo "         tier: $(grep -a 'preload dev2:\|preload dev3:\|preload:' "$OUT/${TAG}_oracle_${side}.err" | tr '\n' ' ')"
  wait_no_engine || true
  if grep -aq 'tier empty' "$OUT/${TAG}_oracle_${side}.err"; then
    echo "FATAL [oracle/$side]: the expert tier came up EMPTY -- that is not the served"
    echo "      configuration and no comparison taken on it means anything. Refusing."
    return 8
  fi
  return $rc
}

echo "=== phase 1: oracle (candidate vs served pristine) $(date -Is)"
run_oracle cand     "$CAND_BIN"     "$CAND_SHADERS"     || exit 1
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE_SHADERS" || exit 1
gate_compare "F1s0 teacher_forcing (candidate vs pristine)" \
    "$OUT/${TAG}_oracle_cand.out" "$OUT/${TAG}_oracle_pristine.out" '^teacher_forcing'
oracle_rc=$?
if [ "$oracle_rc" != 0 ]; then
  echo "FATAL: oracle miss (rc=$oracle_rc) -- a timers-only change moved the output."
  echo "       Aborting before the ladder; nothing after this line describes a"
  echo "       trustworthy profile."
  exit 1
fi
echo "=== phase 1 PASS: candidate is numerically inert against the served pristine"

# ======================================================================
# Phase 2: ONE ladder arm, candidate only, COLI_TIMERS=1. No A/B: the
# oracle above already established the candidate computes identically to
# what serves; the ladder exists to fill the [PROF]/[OPTIME] eg-split
# accumulators at the depths the record's other profiles used, not to
# produce a speed verdict.
# ======================================================================
run_ladder() {   # run_ladder <arm-label> <binary> <shaders>
  local arm=$1 bin=$2 shaders=$3
  local json="$OUT/${TAG}_${arm}.jsonl" elog="$OUT/${TAG}_${arm}_engine.log" \
        console="$OUT/${TAG}_${arm}_console.log" rc
  echo "=== arm $arm (GLM, engine mode) bin=$bin $(date -Is)"
  precheck "$arm-pre"
  warm_glm
  assert_glm_resident "$arm-pre" || { echo "FATAL: $arm residency < 90% before engine start"; return 1; }

  export GLM53_MAXT=32768
  export GLM53_PREFIX_CKPT=0
  export COLI_CKPT_DIR="$OUT/ckpt_${TAG}_${arm}"; mkdir -p "$COLI_CKPT_DIR"
  export GLM53_VERBOSE=1
  export COLI_TIMERS=1
  export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
  export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
  export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
  export COLI_KDA_GPU=2
  export COLI_VK_SHADERS="$shaders"
  export COLI_USAGE_PATH="$HIST"
  export GLM53_VK_SWIGLU_CLAMP=1
  export COLI_PREFILL_STREAM=1
  export GLM53_PREFILL_CHUNK=512

  python3 "$HERE/context_ladder.py" \
      --engine "$bin" \
      --steps "$A_STEPS" --gen "$A_GEN" --followups "$A_FOLLOWUPS" \
      --kv-slots 4 --warm --min-resident 90 \
      --arm "$arm" --tag "$TAG" --json "$json" \
      --engine-log "$elog" 2>&1 | tee -a "$console"
  rc=${PIPESTATUS[0]}
  echo "=== arm $arm exit=$rc json=$json engine_log=$elog"
  unset GLM53_MAXT GLM53_PREFIX_CKPT COLI_CKPT_DIR COLI_TIMERS
  unset COLI_PREFILL_STREAM GLM53_PREFILL_CHUNK GLM53_VK_SWIGLU_CLAMP COLI_KDA_GPU
  wait_no_engine || rc=1
  return $rc
}

echo "=== phase 2: ladder (candidate only) $(date -Is)"
run_ladder RP "$CAND_BIN" "$CAND_SHADERS" || exit 1
precheck "post-ladder"
assert_vram_free "post-ladder" || exit 1

echo "--- eg split, every window in the engine log (per-request lines first, final destructor table last)"
grep -E '^\[OPTIME|^\[PROF\] eg' "$OUT/${TAG}_RP_engine.log" 2>/dev/null

echo "=== f1_step0_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
