#!/bin/bash
# f9a_gate_chain.sh -- item F9a: head-lane SIMD for the DSA indexer's score
# pass (c/sparse_index.h) -- ported from P5.1's MLA attention core
# (glm_transpose/glm_lane_dots): transpose a row's queries once to qT[d][h],
# then compute every head's dot against one pool vector at a time, heads in
# the AVX2 lanes, each lane summing over d in ascending order with the SAME
# multiply-then-add (no FMA contraction) as this build's own compiled
# scalar loop -- bit-identical on real GCC with AVX2/FMA (default; verified
# by objdump, see the F9a record and sparse_index.h's own comment).
# GLM53_INDEX_LANES=0 restores the scalar dot for the A/B.
#
# Structure copied from f8_gate_chain.sh (which has everything): run_chain.sh
# takes the rig lock, the gateway comes back on EVERY exit path, the exit
# trap kills only glm53, the engine pgreps run after stop_gateway,
# wait_no_engine before any cp, the histogram is frozen per chain, an empty
# expert tier is a refusal, and accept_live.sh measures the request AFTER
# the chain. The index-dump oracle leg (GLM53_DUMP_INDEX, set + rank order)
# and the "mla split index=" ms/call report are F6a's own patterns
# (f6a_gate_chain.sh), reused verbatim.
#
# Phase 1, ORACLE, HARD GATES. Three arms, one CLI invocation each, on the
#   DEEP packet (9 115 tokens -- the sparse regime is where the indexer
#   matters) plus 32 greedy tokens:
#     pristine  served binary (~/bench/glm53.f8), no F9a knob
#     cand      candidate binary, default (lanes on where GCC+AVX2/FMA say
#               so; the rig build is exactly that)
#     candoff   candidate binary, GLM53_INDEX_LANES=0
#   Hard gates (exit on failure):
#     cand    == pristine : teacher_forcing IDENTICAL, greedy text IDENTICAL,
#                            logit dump BIT-IDENTICAL (cmp)
#     candoff == pristine : same three checks
#   Index dump (GLM53_DUMP_INDEX=1, "index q=... ->" rows, set AND rank
#   order): run on the SHORT packet (f8_gate_chain's own default packet,
#   ~3k tokens) instead of the deep one -- the deep packet's index dump is
#   tens of MB of stderr text per arm, F6a's own reason for using a
#   separate short run when the full dump is not needed for this leg.
#     cand    == pristine : index rows IDENTICAL
#     candoff == pristine : index rows IDENTICAL
#
# Phase 2, LADDER -- A,B,B,A on ONE binary (the candidate):
#   A = service env + GLM53_INDEX_LANES=0
#   B = service env (lanes on, the default)
#   Item gate: TTFT of the deepest (18 439-token) turn SEPARATED >= 1.10x,
#   decode tok/s not dropped (through gate_ab_verdict). A short rung
#   (default F9A_STEPS) is DIRECTIONAL ONLY, same caveat as every other
#   chain here -- only a rung reaching the record's usual depth decides
#   the item.
#
# Knobs:
#   F9A_ORACLE_ONLY=1   phase 1 only, then stop
#   F9A_LADDER_ONLY=1   skip phase 1 and run only the A,B,B,A ladder
#   F9A_STEPS=...       ladder steps, default "1024,1024,2048,4096,8192"
#   F9A_GEN=128         new tokens generated per ladder turn
#   F9A_FOLLOWUPS=2     follow-up turns per ladder step
#   F9A_GREEDY=32       decode tokens generated after the oracle prefill
#   F9A_REUSE_ORACLE=1  reuse a prior run's phase-1 .out/.err files (F6a's
#                       own knob name and meaning)
#
# Launch ONLY through run_chain.sh (it takes the rig lock):
#   setsid nohup ~/src/colibri-f9a/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f9a/tools/hot-expert/f9a_gate_chain.sh \
#       >> ~/bench/f9a_gate.log 2>&1 < /dev/null &
# The engine build (`make -C c qwen38 qwen38-vk glm53 VK=1` in
# ~/src/colibri-f9a) happens BEFORE this is launched, not inside it, and
# needs no lock.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
F9A_ROOT=$(cd "$HERE/../.." && pwd)             # the clone under test, ~/src/colibri-f9a
PRISTINE=~/src/colibri                          # shaders only; F8 is served
PRISTINE_BIN=${F9A_PRISTINE_BIN:-~/bench/glm53.f8}
[ -x "$PRISTINE_BIN" ] || PRISTINE_BIN="$PRISTINE/c/glm53"
PRISTINE_SHADERS=${F9A_PRISTINE_SHADERS:-$PRISTINE/c/shaders}
CAND_BIN="$F9A_ROOT/c/glm53"
CAND_SHADERS="$F9A_ROOT/c/shaders"
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
OUT=~/bench/f9a_out; mkdir -p "$OUT"
TAG=f9a$(date +%m%d%H%M)
. "$HERE/gate_lib.sh"

DEEP_PACKET=${F9A_DEEP_PACKET:-~/bench/f7_out/f709191847_deep_packet.txt}
DEEP_PACKET=$(eval echo "$DEEP_PACKET")
SHORT_PACKET=${F9A_SHORT_PACKET:-~/bench/f7_out/f709191847_shallow_packet.txt}
SHORT_PACKET=$(eval echo "$SHORT_PACKET")

A_STEPS=${F9A_STEPS:-1024,1024,2048,4096,8192}
A_GEN=${F9A_GEN:-128}
A_FOLLOWUPS=${F9A_FOLLOWUPS:-2}
GREEDY_N=${F9A_GREEDY:-32}

echo "=== f9a_gate_chain $TAG $(date -Is)"
echo "=== pristine=$PRISTINE_BIN candidate=$CAND_BIN"
echo "=== oracle deep packet=$DEEP_PACKET short packet=$SHORT_PACKET greedy=$GREEDY_N"
echo "=== ladder steps=$A_STEPS gen=$A_GEN followups=$A_FOLLOWUPS"
echo "=== gate: TTFT of the deepest ladder turn SEPARATED >= 1.10x, decode not dropped"
echo "=== THE GATEWAY IS DOWN FOR THE WHOLE CHAIN"

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
VRAM_TOTAL() { cat "/sys/class/drm/card$1/device/mem_info_vram_total" 2>/dev/null || echo -1; }

precheck() {
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  for e in glm53 qwen38 qwen38-vk; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  echo "[$label] pgrep -f 'glm53[.]': $(pgrep -f "glm53[.]" | wc -l)"
  for c in 0 1 2; do echo "[$label] card$c vram_used=$(VRAM "$c") vram_total=$(VRAM_TOTAL "$c")"; done
}

assert_vram_free() {
  local label=$1 v c bad attempt
  for attempt in $(seq 1 30); do
    bad=0
    for c in 0 1 2; do
      v=$(VRAM "$c")
      [ "$v" -lt 0 ] || [ "$v" -ge 1073741824 ] && bad=1
    done
    [ "$bad" = 0 ] && return 0
    sleep 2
  done
  for c in 0 1 2; do echo "FATAL [$label]: card$c vram_used=$(VRAM "$c")"; done
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
      -u GLM53_VK_SWIGLU_CLAMP -u GLM53_MLA_ATTN_GPU -u GLM53_MLA_ATTN_SB \
      -u GLM53_MOE_ONE_TEAM -u GLM53_I4_FAST -u GLM53_INDEX_LANES \
      -u GLM53_INDEX_SCALAR -u GLM53_DUMP_INDEX \
      -u GLM53_LOGIT_DUMP_ALL -u COLI_PREFILL_RING_SLOTS -u COLI_USAGE_PATH \
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

assert_glm_resident() {
  local label=$1 pct
  pct=$(fincore --bytes --output SIZE,RES "$GLM_SNAP"/*.safetensors 2>/dev/null \
        | tail -n +2 | awk '{ts+=$1; rs+=$2} END{if (ts>0) printf "%.4f", 100*rs/ts; else print 0}')
  echo "[resid $label] resident=${pct}%"
  awk -v p="$pct" 'BEGIN{exit !(p>=90)}'
}

# extract_greedy_text <file> -- the generated decode text only, between the
# teacher_forcing line and the "decode N token in Xs" timing line (the
# timing itself always differs A/B and must not enter an identity check).
extract_greedy_text() {
  sed -n '/^teacher_forcing /,/^decode [0-9]\+ token in/p' "$1" | sed '1d;$d'
}

gate_text() {
  local label="$1" a="$2" b="$3" ta tb sa sb rc
  ta=$(mktemp); tb=$(mktemp)
  extract_greedy_text "$a" > "$ta"
  extract_greedy_text "$b" > "$tb"
  sa=$(wc -c < "$ta"); sb=$(wc -c < "$tb")
  if [ "$sa" -eq 0 ] || [ "$sb" -eq 0 ]; then
    printf '  %-34s REFUSED: generated text empty (A=%s B=%s bytes)\n' "$label" "$sa" "$sb"
    rm -f "$ta" "$tb"; return 2
  fi
  if cmp -s "$ta" "$tb"; then
    printf '  %-34s IDENTICAL (%s bytes)\n' "$label" "$sa"
    rc=0
  else
    printf '  %-34s DIFFERS\n' "$label"
    rc=1
  fi
  rm -f "$ta" "$tb"
  return $rc
}

# gate_index_dump <label> <errA> <errB> -- F6a's own check: the "index q=..."
# stderr rows, set AND rank order, diffed whole.
gate_index_dump() {
  local label="$1" a="$2" b="$3" na
  na=$(grep -c '^index q=' "$a" 2>/dev/null || echo 0)
  if [ "$na" -gt 0 ] && diff <(grep '^index q=' "$a") <(grep '^index q=' "$b") >/dev/null 2>&1; then
    printf '  %-34s IDENTICAL (%s row(s))\n' "$label" "$na"
    return 0
  fi
  printf '  %-34s DIFFERS or empty -- first differing lines:\n' "$label"
  diff <(grep '^index q=' "$a") <(grep '^index q=' "$b") 2>/dev/null | head -20
  return 1
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
  echo "=== f9a_gate_chain exit rc=$rc tag=$TAG $(date -Is)"
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
[ -s "$DEEP_PACKET" ] || { echo "FATAL: $DEEP_PACKET missing"; exit 1; }
[ -s "$SHORT_PACKET" ] || { echo "FATAL: $SHORT_PACKET missing"; exit 1; }

stop_gateway || exit 1
precheck "after-stop"

warm_glm
assert_glm_resident "pre" || { echo "FATAL: GLM not >=90% resident before the chain"; exit 1; }

# A per-chain COPY of the usage histogram, never the canonical file --
# without one vk_preload_tier comes up with an EMPTY tier (F8's own note).
HIST="$OUT/${TAG}_hist.bin"
cp ~/.glm53_explain.bin "$HIST" || { echo "FATAL: no ~/.glm53_explain.bin to freeze"; exit 1; }
echo "--- frozen usage histogram: $HIST ($(stat -c %s "$HIST") bytes)"

# run_oracle <side> <bin> <shaders> <packet> <outtag> [extra env assignments...]
# Base env is the SERVED configuration (clamp + streaming + chunk 512 +
# batched attention + F8's one-team/gv CPU expert path), so a mismatch here
# means the F9a knob, not the rest of the service stack.
run_oracle() {
  local side=$1 bin=$2 shaders=$3 packet=$4 outtag=$5; shift 5
  if [ "${F9A_REUSE_ORACLE:-0}" = 1 ] && [ -s "$OUT/${outtag}_${side}.out" ]; then
    echo "[oracle $outtag] $side: REUSING $OUT/${outtag}_${side}.out (F9A_REUSE_ORACLE=1)"; return 0
  fi
  for e in glm53 qwen38 qwen38-vk; do
    pgrep -x "$e" >/dev/null 2>&1 && { echo "FATAL: $e already running before oracle $outtag-$side"; return 9; }
  done
  rm -rf "$OUT/ckpt_${outtag}_${side}"; mkdir -p "$OUT/ckpt_${outtag}_${side}"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_KDA_GPU=0
    export GLM53_VK_SWIGLU_CLAMP=1 COLI_PREFILL_STREAM=1 GLM53_PREFILL_CHUNK=512
    export GLM53_MLA_ATTN_GPU=1
    export GLM53_MOE_ONE_TEAM=1 GLM53_I4_FAST=2
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt_${outtag}_${side}"
    export COLI_USAGE_PATH="$HIST"
    export GLM53_VERBOSE=1
    export GLM53_LOGIT_DUMP_ALL="$OUT/${outtag}_dump_${side}.f32"
    unset COLI_TIMERS GLM53_INDEX_SCALAR GLM53_DUMP_INDEX
    for kv in "$@"; do export "${kv?}"; done
    "$bin" --model "$GLM_SNAP" --prompt "$(cat "$packet")" --logits --greedy "$GREEDY_N"
  ) > "$OUT/${outtag}_${side}.out" 2> "$OUT/${outtag}_${side}.err"
  local rc=$?
  echo "[oracle $outtag] $side rc=$rc $(grep -c ^teacher_forcing "$OUT/${outtag}_${side}.out") tf-line(s) $(grep -a -o 'decode [0-9]* token in [0-9.]*s' "$OUT/${outtag}_${side}.out" | tail -1)"
  echo "           tier: $(grep -a 'preload dev2:\|preload dev3:\|preload:' "$OUT/${outtag}_${side}.err" | tr '\n' ' ')"
  wait_no_engine || true
  if grep -aq 'tier empty' "$OUT/${outtag}_${side}.err"; then
    echo "FATAL [oracle $outtag/$side]: the expert tier came up EMPTY -- that is not the served"
    echo "      configuration and no numerics comparison taken on it means anything. Refusing."
    return 8
  fi
  return $rc
}

oracle_rc=0
if [ "${F9A_LADDER_ONLY:-0}" = 1 ]; then
  echo "=== phase 1 (oracle) SKIPPED: F9A_LADDER_ONLY=1"
else
echo "=== phase 1: oracle $(date -Is)"

echo "--- deep packet (9 115 tokens): pristine, cand (default), candoff (GLM53_INDEX_LANES=0)"
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE_SHADERS" "$DEEP_PACKET" deep || exit 1
run_oracle cand     "$CAND_BIN"     "$CAND_SHADERS"     "$DEEP_PACKET" deep || exit 1
run_oracle candoff  "$CAND_BIN"     "$CAND_SHADERS"     "$DEEP_PACKET" deep "GLM53_INDEX_LANES=0" || exit 1

echo "--- HARD GATE: cand == pristine (deep)"
gate_compare "cand==pristine teacher_forcing (deep)" "$OUT/deep_cand.out" "$OUT/deep_pristine.out" '^teacher_forcing'
g1_tf=$?
gate_text "cand==pristine greedy text (deep, $GREEDY_N tok)" "$OUT/deep_cand.out" "$OUT/deep_pristine.out"
g1_txt=$?
if [ -s "$OUT/deep_dump_cand.f32" ] && [ -s "$OUT/deep_dump_pristine.f32" ] \
   && cmp -s "$OUT/deep_dump_cand.f32" "$OUT/deep_dump_pristine.f32"; then
  echo "  cand==pristine logit dump (deep)    BIT-IDENTICAL ($(wc -c < "$OUT/deep_dump_cand.f32") bytes)"
  g1_dump=0
else
  echo "  cand==pristine logit dump (deep)    DIFFERS"
  g1_dump=1
fi

echo "--- HARD GATE: candoff == pristine (deep)"
gate_compare "candoff==pristine teacher_forcing (deep)" "$OUT/deep_candoff.out" "$OUT/deep_pristine.out" '^teacher_forcing'
g2_tf=$?
gate_text "candoff==pristine greedy text (deep, $GREEDY_N tok)" "$OUT/deep_candoff.out" "$OUT/deep_pristine.out"
g2_txt=$?
if [ -s "$OUT/deep_dump_candoff.f32" ] && [ -s "$OUT/deep_dump_pristine.f32" ] \
   && cmp -s "$OUT/deep_dump_candoff.f32" "$OUT/deep_dump_pristine.f32"; then
  echo "  candoff==pristine logit dump (deep) BIT-IDENTICAL"
  g2_dump=0
else
  echo "  candoff==pristine logit dump (deep) DIFFERS"
  g2_dump=1
fi

echo "--- short packet index dump (GLM53_DUMP_INDEX=1, the deep dump is too large for this leg)"
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE_SHADERS" "$SHORT_PACKET" short "GLM53_DUMP_INDEX=1" || exit 1
run_oracle cand     "$CAND_BIN"     "$CAND_SHADERS"     "$SHORT_PACKET" short "GLM53_DUMP_INDEX=1" || exit 1
run_oracle candoff  "$CAND_BIN"     "$CAND_SHADERS"     "$SHORT_PACKET" short "GLM53_DUMP_INDEX=1" "GLM53_INDEX_LANES=0" || exit 1
gate_index_dump "cand==pristine index dump (short)" "$OUT/short_pristine.err" "$OUT/short_cand.err"
g1_idx=$?
gate_index_dump "candoff==pristine index dump (short)" "$OUT/short_pristine.err" "$OUT/short_candoff.err"
g2_idx=$?
gate_compare "cand==pristine teacher_forcing (short)" "$OUT/short_cand.out" "$OUT/short_pristine.out" '^teacher_forcing'
g1_tf_short=$?
gate_compare "candoff==pristine teacher_forcing (short)" "$OUT/short_candoff.out" "$OUT/short_pristine.out" '^teacher_forcing'
g2_tf_short=$?

echo "=== oracle summary"
echo "  cand==pristine    (HARD GATE): tf(deep)=$g1_tf text(deep)=$g1_txt dump(deep)=$g1_dump idx(short)=$g1_idx tf(short)=$g1_tf_short"
echo "  candoff==pristine (HARD GATE): tf(deep)=$g2_tf text(deep)=$g2_txt dump(deep)=$g2_dump idx(short)=$g2_idx tf(short)=$g2_tf_short"
if [ "$g1_tf" != 0 ] || [ "$g1_txt" != 0 ] || [ "$g1_dump" != 0 ] || [ "$g1_idx" != 0 ] || [ "$g1_tf_short" != 0 ]; then
  echo "FATAL: cand is NOT bit-identical to pristine. Stopping before the ladder."
  oracle_rc=1
fi
if [ "$g2_tf" != 0 ] || [ "$g2_txt" != 0 ] || [ "$g2_dump" != 0 ] || [ "$g2_idx" != 0 ] || [ "$g2_tf_short" != 0 ]; then
  echo "FATAL: candoff (GLM53_INDEX_LANES=0) is NOT bit-identical to pristine. Stopping before the ladder."
  oracle_rc=1
fi
[ "$oracle_rc" = 0 ] || exit 1
echo "=== phase 1 PASS: teacher_forcing, ALL-position logits (deep) and the index selection dump"
echo "    (set + rank order, short) are IDENTICAL for both cand and candoff against pristine"
fi

[ "${F9A_ORACLE_ONLY:-0}" = 1 ] && { echo "=== F9A_ORACLE_ONLY=1, stopping after phase 1"; exit 0; }

# ======================================================================
# Phase 2: the ladder. A = GLM53_INDEX_LANES=0, B = default. SAME binary,
# A,B,B,A.
# ======================================================================
run_arm() {   # run_arm <arm-label> <lanes 0|1>
  local arm=$1 lanes=$2
  local json="$OUT/${TAG}_${arm}.jsonl" elog="$OUT/${TAG}_${arm}_engine.log" \
        console="$OUT/${TAG}_${arm}_console.log" rc
  echo "=== arm $arm (GLM53_INDEX_LANES=$lanes) $(date -Is)"
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
  export COLI_VK_SHADERS="$CAND_SHADERS"
  export COLI_USAGE_PATH="$HIST"      # frozen copy: both arms preload the same tier
  # The served configuration, in BOTH arms: GLM53_INDEX_LANES is the only
  # difference (candoff-shaped A vs default-shaped B).
  export GLM53_VK_SWIGLU_CLAMP=1 COLI_PREFILL_STREAM=1 GLM53_PREFILL_CHUNK=512
  export GLM53_MLA_ATTN_GPU=1
  export GLM53_MOE_ONE_TEAM=1 GLM53_I4_FAST=2
  if [ "$lanes" = 0 ]; then
    export GLM53_INDEX_LANES=0
  else
    unset GLM53_INDEX_LANES
  fi

  python3 "$HERE/context_ladder.py" \
      --engine "$CAND_BIN" \
      --steps "$A_STEPS" --gen "$A_GEN" --followups "$A_FOLLOWUPS" \
      --kv-slots 4 --warm --min-resident 90 \
      --arm "$arm" --tag "$TAG" --json "$json" \
      --engine-log "$elog" 2>&1 | tee -a "$console"
  rc=${PIPESTATUS[0]}
  echo "=== arm $arm exit=$rc json=$json engine_log=$elog"
  unset GLM53_MAXT GLM53_PREFIX_CKPT COLI_CKPT_DIR COLI_TIMERS
  unset GLM53_PREFILL_CHUNK GLM53_VK_SWIGLU_CLAMP COLI_PREFILL_STREAM
  unset GLM53_MLA_ATTN_GPU GLM53_MOE_ONE_TEAM GLM53_I4_FAST GLM53_INDEX_LANES
  wait_no_engine || rc=1
  return $rc
}

echo "=== phase 2: ladder A,B,B,A $(date -Is)  steps=$A_STEPS gen=$A_GEN followups=$A_FOLLOWUPS"
warm_glm
assert_glm_resident "pre-ladder" || { echo "FATAL: GLM not >=90% resident before the ladder"; exit 1; }

run_arm A1 0 || exit 1
run_arm B1 1 || exit 1
run_arm B2 1 || exit 1
run_arm A2 0 || exit 1
precheck "post-ladder"
assert_vram_free "post-ladder" || exit 1

echo "=== verdicts $(date -Is)"
python3 "$HERE/context_compare.py" --rows \
    A1="$OUT/${TAG}_A1.jsonl" B1="$OUT/${TAG}_B1.jsonl" \
    B2="$OUT/${TAG}_B2.jsonl" A2="$OUT/${TAG}_A2.jsonl"

echo "--- [OPTIME req=] mla split index= ms/call at the deepest ladder turn and the two follow-ups, all four arms"
python3 - "$TAG" "$OUT" A1 B1 B2 A2 <<'PY'
import re, sys
tag, out, *arms = sys.argv[1:]
pat = re.compile(r'mla split \(n=(\d+), mean ctx=(\d+)\).*index=([\d.]+)s \(([\d.]+) ms\)')
for arm in arms:
    path = f"{out}/{tag}_{arm}_engine.log"
    rows = []
    try:
        for line in open(path):
            m = pat.search(line)
            if m:
                rows.append(float(m.group(4)))
    except OSError as e:
        print(f"  {arm:<4} REFUSED: could not read {path} ({e})")
        continue
    if not rows:
        print(f"  {arm:<4} REFUSED: no 'mla split' line in {path} -- an empty comparison is not a result")
        continue
    print(f"  {arm:<4} index_ms/call (last up to 3 prints): " + ", ".join(f"{v:.3f}" for v in rows[-3:]))
PY

echo "--- decode tok/s and TTFT of the deepest ladder turn, all four arms, through gate_ab_verdict"
python3 - "$TAG" "$OUT" A1 B1 B2 A2 > "$OUT/${TAG}_deep.env" <<'PY'
import json, sys
tag, out, *arms = sys.argv[1:]
decode, ttft = {}, {}
for arm in arms:
    rows = []
    try:
        for line in open(f"{out}/{tag}_{arm}.jsonl"):
            line = line.strip()
            if line: rows.append(json.loads(line))
    except OSError:
        continue
    lad = [r for r in rows if r.get("kind") in ("ladder", "followup") and r.get("decode_tps")]
    if not lad: continue
    deepest = max(lad, key=lambda r: r.get("prompt_tokens", 0))
    decode[arm] = deepest["decode_tps"]
    if deepest.get("ttft_s") is not None:
        ttft[arm] = deepest["ttft_s"]
def emit(name, d):
    A = " ".join(f"{d[a]:.4f}" for a in ("A1","A2") if a in d)
    B = " ".join(f"{d[a]:.4f}" for a in ("B1","B2") if a in d)
    print(f"{name}_A='{A}'"); print(f"{name}_B='{B}'")
emit("DECODE", decode); emit("TTFT", ttft)
PY
cat "$OUT/${TAG}_deep.env"
. "$OUT/${TAG}_deep.env"
gate_ab_verdict "decode tok/s, deepest turn (higher better)" "${DECODE_A:-}" "${DECODE_B:-}"
gate_ab_verdict "TTFT, deepest turn (s, lower better -- B should be faster)" "${TTFT_A:-}" "${TTFT_B:-}"
python3 - "${DECODE_A:-}" "${DECODE_B:-}" "${TTFT_A:-}" "${TTFT_B:-}" <<'PY'
import sys
def mean(s):
    v = [float(x) for x in s.split()] if s.strip() else []
    return sum(v) / len(v) if v else None
da, db, ta, tb = (mean(x) for x in sys.argv[1:5])
print("=== gate arithmetic (F9a: TTFT of the deepest turn >= 1.10x faster with lanes on, decode not dropped)")
if ta is None or tb is None or ta == 0:
    print("  ttft           NO NUMBER")
else:
    speedup = ta / tb
    print(f"  ttft           A={ta:.3f}s  B={tb:.3f}s  speedup={speedup:.3f}x  bar=1.10x  "
          f"{'MET' if speedup >= 1.10 else 'NOT MET'}")
if da is None or db is None or da == 0:
    print("  decode         NO NUMBER")
else:
    print(f"  decode         A={da:.3f} tok/s  B={db:.3f} tok/s  "
          f"{'MET (not dropped)' if db >= da * 0.98 else 'NOT MET (dropped)'}")
print("  (a short rung is DIRECTIONAL ONLY -- run F9A_STEPS to the record's usual")
print("   depth, 1024,1024,2048,4096,8192 to 18k, before calling the item done.)")
PY

echo "--- index split (mla index) and decode's share of the turn, per arm, deepest turn's [OPTIME] table"
for arm in A1 B1 B2 A2; do
  echo "  [$arm]"
  grep -aE 'mla split|kda=.*mla=' "$OUT/${TAG}_${arm}_engine.log" 2>/dev/null | tail -3
done

echo "=== f9a_gate_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
