#!/bin/bash
# f4_optime_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md sec 8.3 item F4, first
# half: prove the ttft_serve.EngineDriver.close() fix (close stdin, wait
# bounded, THEN terminate/kill) actually lets glm53's serve_loop return
# through main() so its COLI_TIMERS destructor runs, and that c/glm53.c now
# prints one [OPTIME req=N ctx=C] table per completed serve-mode request
# (reset before the next one) instead of the single cumulative, depth-mixing
# table an engine-mode run could never even see (record sec X3 step 0: "no
# engine-mode log ever can" carry an [OPTIME] line -- SIGTERM's default
# disposition skips destructors). Interpreting what the tables say about the
# 205 ms/token growth from 2.7k to 18k is the NEXT step, not this one.
#
# Order: preflight (prints only) -> stop_gateway -> wait_no_proc glm53 ->
# warm+assert GLM >=90% resident -> ORACLE (COLI_TIMERS unset: candidate vs
# the served pristine, same short CLI greedy prompt, gate_compare on
# teacher_forcing) -> SMOKE (COLI_TIMERS=1, candidate, tiny engine-mode
# ladder, assert the engine log now carries >=2 "[OPTIME req=" tables and a
# final untagged "[OPTIME]" table) -> the A ARM (franken_chain.sh's run_A,
# unchanged: --steps 1024,1024,2048,4096,8192 --gen 128 --followups 2,
# COLI_TIMERS=1) -> stop engine, assert VRAM free -> exit trap: re-warm GLM,
# restart the gateway, accept_live.sh.
#
# Launch ONLY through run_chain.sh (it takes the rig lock; this script does
# not stop the gateway on its own without it):
#   setsid nohup ~/src/colibri-f4/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f4/tools/hot-expert/f4_optime_chain.sh \
#       >> ~/bench/f4_optime.log 2>&1 < /dev/null &
#
# The build (`make -C c glm53 VK=1` in the ~/src/colibri-f4 clone) does not
# need the lock and happens BEFORE this is launched, not inside it.
#
# GLM53_PREFIX_CKPT=0 with a private COLI_CKPT_DIR on every GLM invocation
# here (CLAUDE.md: a restored checkpoint reports a prefill that never
# happened). Budget: oracle ~2 min, smoke ~3 min, A arm ~75 min (the 18k
# turn alone prefills ~43 min -- expected, not a stall, per franken_chain.sh
# and the plan's F2/F4 row).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
F4_ROOT=$(cd "$HERE/../.." && pwd)              # the clone under test, e.g. ~/src/colibri-f4
PRISTINE=~/src/colibri                          # binary in service; H1/franken_chain's own convention
PRISTINE_BIN="$PRISTINE/c/glm53"
CAND_BIN="$F4_ROOT/c/glm53"
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
OUT=~/bench/ctx_ladder_out; mkdir -p "$OUT"
TAG=f4$(date +%m%d%H%M)
. "$HERE/gate_lib.sh"

A_STEPS="1024,1024,2048,4096,8192"; A_GEN=128; A_FOLLOWUPS=2
SMOKE_STEPS="256,256"; SMOKE_GEN=16; SMOKE_FOLLOWUPS=1

echo "=== f4_optime_chain $TAG $(date -Is)"
echo "=== candidate=$CAND_BIN pristine=$PRISTINE_BIN"
echo "=== A_STEPS=$A_STEPS gen=$A_GEN followups=$A_FOLLOWUPS"
echo "=== THE GATEWAY IS DOWN FOR THE WHOLE CHAIN"

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }

precheck() {   # precheck <label>
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  for e in glm53 qwen38 qwen38-vk; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  for c in 0 1 2; do
    echo "[$label] card$c vram_used=$(VRAM "$c")"
  done
}

assert_vram_free() {   # assert_vram_free <label> -- copied from franken_chain.sh
  # (the amdgpu/KFD driver reclaims a large allocation's page tables over real
  # wall-clock time after the process exits, not atomically with it)
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

wait_no_proc() {   # wait_no_proc <procname>
  local p=$1
  for _ in $(seq 1 120); do pgrep -x "$p" >/dev/null || return 0; sleep 2; done
  echo "FATAL: $p still alive after 240 s"; return 1
}

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT -u COLI_TIMERS \
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
  wait_no_proc glm53
}

warm_glm() {
  echo "--- warming GLM shards (one model at a time; 182 GiB)"
  cat "$GLM_SNAP"/*.safetensors > /dev/null 2>&1 || true
}

assert_glm_resident() {   # assert_glm_resident <label> -- copied from franken_chain.sh
  local label=$1 pct
  pct=$(fincore --bytes --output SIZE,RES "$GLM_SNAP"/*.safetensors 2>/dev/null \
        | tail -n +2 | awk '{ts+=$1; rs+=$2} END{if (ts>0) printf "%.4f", 100*rs/ts; else print 0}')
  echo "[resid $label] resident=${pct}%"
  awk -v p="$pct" 'BEGIN{exit !(p>=90)}'
}

# ---- the A arm: exactly franken_chain.sh's run_A (glm53, engine mode) ------
run_A() {   # run_A <arm-label> <binary> <shaders>
  # arm must be its own `local` statement -- franken_chain.sh's run_A found
  # this live on the rig (`set -u`: "arm: unbound variable") the first smoke run.
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
  # COLI_TIMERS=1: the [OPTIME] kda/mla/ffn split. F4's whole point: this now
  # prints once per completed request into $elog (reset before the next
  # one), plus a final untagged table from the destructor for the last
  # request's own window -- see c/glm53.c serve_one()/optime_print_req().
  export COLI_TIMERS=1
  export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
  export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
  export COLI_VK_SHADERS="$shaders"

  python3 "$HERE/context_ladder.py" \
      --engine "$bin" \
      --steps "$A_STEPS" --gen "$A_GEN" --followups "$A_FOLLOWUPS" \
      --kv-slots 4 --warm --min-resident 90 \
      --arm "$arm" --tag "$TAG" --json "$json" \
      --engine-log "$elog" 2>&1 | tee -a "$console"
  rc=${PIPESTATUS[0]}
  echo "=== arm $arm exit=$rc json=$json engine_log=$elog"
  unset GLM53_MAXT GLM53_PREFIX_CKPT COLI_CKPT_DIR COLI_TIMERS
  # A still-alive glm53 here would ETXTBSY or hijack the next step's spawn --
  # fold its failure into the return code rather than discarding it.
  wait_no_proc glm53 || rc=1
  return $rc
}

# ---- smoke: candidate, engine mode, tiny ladder, COLI_TIMERS=1 -----------
run_smoke() {
  local elog="$OUT/${TAG}_smoke_engine.log" json="$OUT/${TAG}_smoke.jsonl" \
        console="$OUT/${TAG}_smoke_console.log" rc n_req n_final
  echo "=== smoke (candidate, engine mode, COLI_TIMERS=1) $(date -Is)"
  precheck "smoke-pre"
  warm_glm
  assert_glm_resident "smoke-pre" || { echo "FATAL: smoke residency < 90% before engine start"; return 1; }

  export GLM53_MAXT=32768
  export GLM53_PREFIX_CKPT=0
  export COLI_CKPT_DIR="$OUT/ckpt_${TAG}_smoke"; mkdir -p "$COLI_CKPT_DIR"
  export GLM53_VERBOSE=1
  export COLI_TIMERS=1
  export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
  export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
  export COLI_VK_SHADERS="$F4_ROOT/c/shaders"

  python3 "$HERE/context_ladder.py" \
      --engine "$CAND_BIN" \
      --steps "$SMOKE_STEPS" --gen "$SMOKE_GEN" --followups "$SMOKE_FOLLOWUPS" \
      --kv-slots 4 --warm --min-resident 90 \
      --arm smoke --tag "$TAG" --json "$json" \
      --engine-log "$elog" 2>&1 | tee -a "$console"
  rc=${PIPESTATUS[0]}
  echo "=== smoke exit=$rc json=$json engine_log=$elog"
  unset GLM53_MAXT GLM53_PREFIX_CKPT COLI_CKPT_DIR COLI_TIMERS
  wait_no_proc glm53 || rc=1
  if [ "$rc" != 0 ]; then echo "FATAL: smoke ladder exited rc=$rc"; return 1; fi

  # The candidate is the whole point of the smoke: the engine log must now
  # show per-request tables (this item's new C-side behaviour) AND a final
  # untagged destructor table (proof the EngineDriver.close() fix let
  # serve_loop see EOF and return through main() instead of dying to
  # SIGTERM). "[OPTIME req=" never appears in a destructor-only table, and
  # the destructor's own tag is the literal "[OPTIME]" with no "req=", so
  # the two greps cannot double-count each other.
  n_req=$(grep -c '^\[OPTIME req=' "$elog" 2>/dev/null); n_req=${n_req:-0}
  n_final=$(grep -c '^\[OPTIME\] forwards=' "$elog" 2>/dev/null); n_final=${n_final:-0}
  echo "=== smoke $elog: per-request [OPTIME req=] tables=$n_req  final destructor tables=$n_final"
  if [ "$n_req" -lt 2 ] || [ "$n_final" -lt 1 ]; then
    echo "FATAL: smoke did not produce the expected [OPTIME] tables (need >=2 req= tables and >=1 final destructor table)"
    echo "--- tail of $elog ---"
    tail -n 80 "$elog"
    return 1
  fi
  return 0
}

# --------------------------------------------------------------- main --
on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  pgrep -x glm53 >/dev/null 2>&1 && { pkill -9 -x glm53 2>/dev/null || true; wait_no_proc glm53; }
  echo "--- final re-warm of GLM before restart"
  warm_glm
  assert_glm_resident "final" || echo "WARNING: GLM not >=90% resident at restart time"
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== f4_optime_chain exit rc=$rc tag=$TAG $(date -Is)"
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

warm_glm
assert_glm_resident "pre" || { echo "FATAL: GLM not >=90% resident before the chain"; exit 1; }

# ---- F4_PROFILE_ONLY=1: the re-profile after a landed item ----------------
# Launched from ~/src/colibri itself, candidate == pristine == the binary in
# service, so the oracle is vacuous and the smoke's [OPTIME] feature is
# already proven. Run only the timed A arm (same steps, same env) and leave
# through the same exit trap (re-warm, restart, accept_live.sh).
if [ "${F4_PROFILE_ONLY:-0}" = 1 ]; then
  echo "=== F4_PROFILE_ONLY=1: skipping oracle and smoke; A arm on $CAND_BIN ($(sha256sum "$CAND_BIN" | cut -c1-16))"
  run_A "${F4_ARM:-RP}" "$CAND_BIN" "$F4_ROOT/c/shaders" || exit 1
  precheck "post-A"
  assert_vram_free "post-A" || exit 1
  echo "=== f4_optime_chain (profile only) body done $(date -Is)"
  exit 0
fi

# ---- oracle (COLI_TIMERS unset): candidate vs the served pristine ---------
echo "=== oracle $(date -Is)"
PACKET=$(cat "$HERE/x2_packet_450.txt")
run_oracle() {   # run_oracle <tag> <binary> <shaders>
  local tag=$1 bin=$2 shaders=$3
  for e in glm53 qwen38 qwen38-vk; do
    pgrep -x "$e" >/dev/null 2>&1 && { echo "FATAL: $e already running before oracle $tag"; return 9; }
  done
  cp -f ~/.glm53_explain.bin "/tmp/f4_hist_$tag.bin"
  rm -rf "/tmp/f4_ckpt_$tag"; mkdir -p "/tmp/f4_ckpt_$tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_USAGE_PATH="/tmp/f4_hist_$tag.bin"
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="/tmp/f4_ckpt_$tag"
    export GLM53_VERBOSE=1 COLI_KDA_GPU=2
    unset COLI_TIMERS
    "$bin" --model "$GLM_SNAP" --prompt "$PACKET" --greedy 0 ) \
      > "$OUT/${TAG}_oracle_$tag.out" 2> "$OUT/${TAG}_oracle_$tag.err"
  local rc=$?
  echo "[oracle] $tag rc=$rc $(grep -c ^teacher_forcing "$OUT/${TAG}_oracle_$tag.out") tf-line(s)"
  wait_no_proc glm53 || true
  return $rc
}
run_oracle candidate "$CAND_BIN" "$F4_ROOT/c/shaders" || exit 1
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE/c/shaders" || exit 1

gate_compare "F4 oracle teacher_forcing" \
  "$OUT/${TAG}_oracle_candidate.out" "$OUT/${TAG}_oracle_pristine.out" '^teacher_forcing'
oracle_rc=$?
echo "=== oracle gate_compare exit=$oracle_rc (0=IDENTICAL 1=DIFFERS 2=REFUSED)"
if [ "$oracle_rc" != 0 ]; then
  echo "FATAL: F4 oracle did not come back IDENTICAL -- stopping before the smoke/A arm"
  exit 1
fi

# ---- smoke (COLI_TIMERS=1): the [OPTIME] table itself ----------------------
run_smoke || exit 1

# ---- the A arm --------------------------------------------------------
warm_glm
assert_glm_resident "pre-A" || { echo "FATAL: GLM not >=90% resident before the A arm"; exit 1; }
run_A F4 "$CAND_BIN" "$F4_ROOT/c/shaders" || exit 1
precheck "post-A"
assert_vram_free "post-A" || exit 1

echo "=== f4_optime_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
