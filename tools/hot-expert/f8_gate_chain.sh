#!/bin/bash
# f8_gate_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F8 (the
# decode-window CPU-expert kernel and fusion), record §F8-STEP0.
#
# Structure copied from f7_gate_chain.sh: run_chain.sh takes the rig lock,
# the gateway comes back on EVERY exit path, the exit trap kills only
# glm53, the engine pgreps run after stop_gateway, wait_no_engine before
# any cp, the histogram is frozen per chain, an empty expert tier is a
# refusal, and accept_live.sh measures the request AFTER the chain.
#
# Two knobs, both off by default in the candidate binary:
#   GLM53_MOE_ONE_TEAM=1   one OpenMP team per decode window instead of one
#                          per expert -- must be BIT-IDENTICAL to off.
#   GLM53_I4_FAST=2        coli_i4_row_gv, reassociating -- judged like F7's
#                          reordered kernel: no hard top-1 bar, "moves no
#                          further than the eps=6e-7 jitter arm" (record
#                          §F7-VERDICT), mean KL <= 0.003 on this packet.
#
# Phase 1, ORACLE (shallow packet only, 3 013 tokens, reused from F7's own
#   oracle run -- same packet the jitter-arm KL numbers above were measured
#   on). Four arms, one CLI invocation each, GLM53_LOGIT_DUMP_ALL on the
#   whole prefill AND `--greedy N` so the decode (tokens==1) path actually
#   runs -- a streamed prefill alone barely reaches a CPU expert (record
#   §F2, "COLI_PREFILL_STREAM=1" moves the routed-expert compute onto the
#   GPU tile kernels for the resident-enough case; what stays on the CPU
#   at nr>1 is the rows path, not what this item changes at nr=1):
#     pristine  served binary, no F8 knobs
#     off       candidate binary, no F8 knobs
#     team      candidate + GLM53_MOE_ONE_TEAM=1
#     gv        candidate + GLM53_MOE_ONE_TEAM=1 GLM53_I4_FAST=2
#   Hard gates (this chain, exit on failure):
#     off   == pristine : teacher_forcing IDENTICAL, greedy text IDENTICAL,
#                          logit dump BIT-IDENTICAL
#     team  == off       : same three checks -- proves the fusion changes
#                          NOTHING, not even a jitter's worth
#   Reported, not gated here (f8_kl_report.sh, OUTSIDE the lock, same reason
#   as F7's O2: a multi-hundred-MB dump diff is minutes of pure Python and
#   needs no engine, no GPU, no lock):
#     gv vs off : mean KL over the whole prefill dump, bar 0.003
#
# Phase 2, LADDER -- A,B,B,A on ONE binary (the candidate):
#   A = service env, no F8 knobs
#   B = service env + GLM53_MOE_ONE_TEAM=1 GLM53_I4_FAST=2
#   Item gate: decode tok/s SEPARATED >= +3% at the deepest turn (through
#   gate_ab_verdict), TTFT at the deepest turn not worse (B <= A). A short
#   rung (default F8_STEPS) is DIRECTIONAL ONLY -- see f7_gate_chain.sh's
#   own comment on this; only a rung reaching the record's usual depth
#   decides the item.
#
# Knobs:
#   F8_ORACLE_ONLY=1   phase 1 only, then stop (no ladder, no gateway churn
#                      beyond phase 1's own stop/restart)
#   F8_LADDER_ONLY=1   skip phase 1 and run only the A,B,B,A ladder
#   F8_STEPS=...       ladder steps, default the SHORT rung "1024,1024,2048"
#   F8_GEN=256         new tokens generated per ladder turn (decode-heavy,
#                      per the item's own "keep prefill short, decode long")
#   F8_FOLLOWUPS=2     follow-up turns per ladder step
#   F8_GREEDY=64       decode tokens generated after the oracle prefill, in
#                      EVERY phase-1 arm (exercises tokens==1)
#
# Launch ONLY through run_chain.sh (it takes the rig lock):
#   setsid nohup ~/src/colibri-f8e/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f8e/tools/hot-expert/f8_gate_chain.sh \
#       >> ~/bench/f8_gate.log 2>&1 < /dev/null &
# The engine build (`make -C c qwen38 qwen38-vk glm53 VK=1` in
# ~/src/colibri-f8e) happens BEFORE this is launched, not inside it, and
# needs no lock.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
F8_ROOT=$(cd "$HERE/../.." && pwd)              # the clone under test, ~/src/colibri-f8e
PRISTINE=~/src/colibri                          # the served tree (shaders only; F7 is served)
PRISTINE_BIN=${F8_PRISTINE_BIN:-~/bench/glm53.f7}
[ -x "$PRISTINE_BIN" ] || PRISTINE_BIN="$PRISTINE/c/glm53"
PRISTINE_SHADERS=${F8_PRISTINE_SHADERS:-$PRISTINE/c/shaders}
CAND_BIN="$F8_ROOT/c/glm53"
CAND_SHADERS="$F8_ROOT/c/shaders"
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
OUT=~/bench/f8_out; mkdir -p "$OUT"
TAG=f8$(date +%m%d%H%M)
. "$HERE/gate_lib.sh"
KL="$HERE/kl_compare.py"

PACKET=${F8_PACKET:-~/bench/f7_out/f709191847_shallow_packet.txt}
PACKET=$(eval echo "$PACKET")

A_STEPS=${F8_STEPS:-1024,1024,2048}
A_GEN=${F8_GEN:-256}
A_FOLLOWUPS=${F8_FOLLOWUPS:-2}
GREEDY_N=${F8_GREEDY:-64}

echo "=== f8_gate_chain $TAG $(date -Is)"
echo "=== pristine=$PRISTINE_BIN candidate=$CAND_BIN"
echo "=== oracle packet=$PACKET greedy=$GREEDY_N"
echo "=== ladder steps=$A_STEPS gen=$A_GEN followups=$A_FOLLOWUPS"
echo "=== gate: decode SEPARATED >= +3% at the deepest ladder turn, TTFT there not worse"
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
      -u GLM53_MOE_ONE_TEAM -u GLM53_I4_FAST \
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

# gate_text <label> <fileA> <fileB> -- like gate_compare, but for the
# extracted decode-text block above instead of a single grep pattern.
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
  echo "=== f8_gate_chain exit rc=$rc tag=$TAG $(date -Is)"
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
[ -s "$PACKET" ] || { echo "FATAL: $PACKET missing -- refusing to regenerate F7's oracle packet"; exit 1; }

stop_gateway || exit 1
precheck "after-stop"

warm_glm
assert_glm_resident "pre" || { echo "FATAL: GLM not >=90% resident before the chain"; exit 1; }

# The CLI oracle path does not get a usage histogram for free; without one
# vk_preload_tier comes up with an EMPTY tier, which is not the served
# configuration and makes every numerics comparison meaningless. A per-chain
# COPY, never the canonical file.
HIST="$OUT/${TAG}_hist.bin"
cp ~/.glm53_explain.bin "$HIST" || { echo "FATAL: no ~/.glm53_explain.bin to freeze"; exit 1; }
echo "--- frozen usage histogram: $HIST ($(stat -c %s "$HIST") bytes)"

# run_oracle <side> <bin> <shaders> <outtag> [extra env assignments...]
# Base env is the SERVED configuration (clamp + streaming + chunk 512 +
# batched attention), so O1 proves the F8 knobs are inert where they
# actually run. `--greedy $GREEDY_N` after the prefill exercises tokens==1.
run_oracle() {
  local side=$1 bin=$2 shaders=$3 outtag=$4; shift 4
  if [ "${F8_REUSE_ORACLE:-0}" = 1 ] && [ -s "$OUT/${outtag}_${side}.out" ]; then
    echo "[oracle $outtag] $side: REUSING $OUT/${outtag}_${side}.out (F8_REUSE_ORACLE=1)"; return 0
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
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt_${outtag}_${side}"
    export COLI_USAGE_PATH="$HIST"
    export GLM53_VERBOSE=1
    export GLM53_LOGIT_DUMP_ALL="$OUT/${outtag}_dump_${side}.f32"
    unset COLI_TIMERS GLM53_EXPERTS_CPU GLM53_MOE_ONE_TEAM GLM53_I4_FAST
    for kv in "$@"; do export "${kv?}"; done
    "$bin" --model "$GLM_SNAP" --prompt "$(cat "$PACKET")" --greedy "$GREEDY_N"
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
if [ "${F8_LADDER_ONLY:-0}" = 1 ]; then
  echo "=== phase 1 (oracle) SKIPPED: F8_LADDER_ONLY=1"
else
echo "=== phase 1: oracle $(date -Is)"

echo "--- pristine and off (candidate, no F8 knobs)"
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE_SHADERS" shallow || exit 1
run_oracle off       "$CAND_BIN"     "$CAND_SHADERS"     shallow || exit 1
echo "--- team (candidate + GLM53_MOE_ONE_TEAM=1)"
run_oracle team "$CAND_BIN" "$CAND_SHADERS" shallow "GLM53_MOE_ONE_TEAM=1" || exit 1
echo "--- gv (candidate + GLM53_MOE_ONE_TEAM=1 GLM53_I4_FAST=2)"
run_oracle gv   "$CAND_BIN" "$CAND_SHADERS" shallow "GLM53_MOE_ONE_TEAM=1" "GLM53_I4_FAST=2" || exit 1

echo "--- HARD GATE: off == pristine (no F8 knob reaches anything by default)"
gate_compare "off==pristine teacher_forcing" "$OUT/shallow_off.out" "$OUT/shallow_pristine.out" '^teacher_forcing'
g1_tf=$?
gate_text "off==pristine greedy text ($GREEDY_N tok)" "$OUT/shallow_off.out" "$OUT/shallow_pristine.out"
g1_txt=$?
if [ -s "$OUT/shallow_dump_off.f32" ] && [ -s "$OUT/shallow_dump_pristine.f32" ] \
   && cmp -s "$OUT/shallow_dump_off.f32" "$OUT/shallow_dump_pristine.f32"; then
  echo "  off==pristine logit dump                    BIT-IDENTICAL ($(wc -c < "$OUT/shallow_dump_off.f32") bytes)"
  g1_dump=0
else
  echo "  off==pristine logit dump                    DIFFERS"
  python3 "$KL" "off vs pristine" "$OUT/shallow_dump_pristine.f32" "$OUT/shallow_dump_off.f32" || true
  g1_dump=1
fi

echo "--- HARD GATE: team == off (GLM53_MOE_ONE_TEAM=1 must change NOTHING)"
gate_compare "team==off teacher_forcing" "$OUT/shallow_team.out" "$OUT/shallow_off.out" '^teacher_forcing'
g2_tf=$?
gate_text "team==off greedy text ($GREEDY_N tok)" "$OUT/shallow_team.out" "$OUT/shallow_off.out"
g2_txt=$?
if [ -s "$OUT/shallow_dump_team.f32" ] && [ -s "$OUT/shallow_dump_off.f32" ] \
   && cmp -s "$OUT/shallow_dump_team.f32" "$OUT/shallow_dump_off.f32"; then
  echo "  team==off logit dump                        BIT-IDENTICAL"
  g2_dump=0
else
  echo "  team==off logit dump                        DIFFERS"
  python3 "$KL" "team vs off" "$OUT/shallow_dump_off.f32" "$OUT/shallow_dump_team.f32" || true
  g2_dump=1
fi

echo "--- gv vs off: informational here (greedy text expected to differ -- reassociation"
echo "    changes summation order once it reaches a leftover row). Not a hard gate."
gate_text "gv vs off greedy text ($GREEDY_N tok, informational)" "$OUT/shallow_gv.out" "$OUT/shallow_off.out" || true
echo "--- gv KL bar (mean KL <= 0.003, record §F7-VERDICT's jitter-arm scale): NOT run here."
echo "    The dumps are on disk. With the gateway back up, run:"
echo "      $HERE/f8_kl_report.sh $OUT"
echo "    and do not call phase 1 passed until it prints MET."
for d in shallow_dump_off shallow_dump_gv; do
  [ -s "$OUT/$d.f32" ] || { echo "FATAL: $OUT/$d.f32 missing or empty -- f8_kl_report.sh has nothing to read"; oracle_rc=1; }
done

echo "=== oracle summary"
echo "  off==pristine (HARD GATE): tf=$g1_tf text=$g1_txt dump=$g1_dump"
echo "  team==off     (HARD GATE): tf=$g2_tf text=$g2_txt dump=$g2_dump"
if [ "$g1_tf" != 0 ] || [ "$g1_txt" != 0 ] || [ "$g1_dump" != 0 ]; then
  echo "FATAL: off is NOT inert against pristine. Stopping before the ladder."
  oracle_rc=1
fi
if [ "$g2_tf" != 0 ] || [ "$g2_txt" != 0 ] || [ "$g2_dump" != 0 ]; then
  echo "FATAL: GLM53_MOE_ONE_TEAM=1 is NOT bit-identical to knob-off. Stopping before the ladder."
  oracle_rc=1
fi
[ "$oracle_rc" = 0 ] || exit 1
echo "=== phase 1 ARMS DONE (both hard gates passed; the gv KL bar is f8_kl_report.sh's, out of the lock)"
fi

[ "${F8_ORACLE_ONLY:-0}" = 1 ] && { echo "=== F8_ORACLE_ONLY=1, stopping after phase 1"; exit 0; }

# ======================================================================
# Phase 2: the ladder. A = knobs off, B = knobs on, SAME binary, A,B,B,A.
# ======================================================================
run_arm() {   # run_arm <arm-label> <f8 0|1>
  local arm=$1 f8=$2
  local json="$OUT/${TAG}_${arm}.jsonl" elog="$OUT/${TAG}_${arm}_engine.log" \
        console="$OUT/${TAG}_${arm}_console.log" rc
  echo "=== arm $arm (GLM53_MOE_ONE_TEAM=$f8 GLM53_I4_FAST=$([ "$f8" = 1 ] && echo 2 || echo 0)) $(date -Is)"
  precheck "$arm-pre"
  warm_glm
  assert_glm_resident "$arm-pre" || { echo "FATAL: $arm residency < 90% before engine start"; return 1; }

  export GLM53_MAXT=32768
  export GLM53_PREFIX_CKPT=0
  export COLI_CKPT_DIR="$OUT/ckpt_${TAG}_${arm}"; mkdir -p "$COLI_CKPT_DIR"
  export GLM53_VERBOSE=1
  export COLI_TIMERS=1
  export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
  export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
  export COLI_VK_SHADERS="$CAND_SHADERS"
  export COLI_USAGE_PATH="$HIST"      # frozen copy: both arms preload the same tier
  # The served configuration, in BOTH arms: F8 is the only difference.
  export GLM53_VK_SWIGLU_CLAMP=1 COLI_PREFILL_STREAM=1 GLM53_PREFILL_CHUNK=512
  export GLM53_MLA_ATTN_GPU=1
  if [ "$f8" = 1 ]; then
    export GLM53_MOE_ONE_TEAM=1 GLM53_I4_FAST=2
  else
    unset GLM53_MOE_ONE_TEAM GLM53_I4_FAST
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
  unset GLM53_MLA_ATTN_GPU GLM53_MOE_ONE_TEAM GLM53_I4_FAST
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
gate_ab_verdict "TTFT, deepest turn (s, lower better)"        "${TTFT_A:-}"   "${TTFT_B:-}"
python3 - "${DECODE_A:-}" "${DECODE_B:-}" "${TTFT_A:-}" "${TTFT_B:-}" <<'PY'
import sys
def mean(s):
    v = [float(x) for x in s.split()] if s.strip() else []
    return sum(v) / len(v) if v else None
da, db, ta, tb = (mean(x) for x in sys.argv[1:5])
print("=== gate arithmetic (F8: decode >= +3% at the deepest turn, TTFT there not worse)")
if da is None or db is None or da == 0:
    print("  decode         NO NUMBER")
else:
    speedup = db / da
    print(f"  decode         A={da:.3f} tok/s  B={db:.3f} tok/s  speedup={speedup:.3f}x  bar=1.03x  "
          f"{'MET' if speedup >= 1.03 else 'NOT MET'}")
if ta is None or tb is None or ta == 0:
    print("  ttft           NO NUMBER")
else:
    print(f"  ttft           A={ta:.3f}s  B={tb:.3f}s  {'MET (not worse)' if tb <= ta else 'NOT MET (worse)'}")
print("  (a short rung is DIRECTIONAL ONLY -- run F8_STEPS to the record's usual")
print("   depth, e.g. 1024,1024,2048,4096,8192, before calling the item done.)")
PY

echo "--- eg split (cpu_in) and moe split, per arm, deepest turn's [OPTIME] table"
for arm in A1 B1 B2 A2; do
  echo "  [$arm]"
  grep -aE '^\[OPTIME req=.* eg split:|^\[OPTIME req=.* moe split:' "$OUT/${TAG}_${arm}_engine.log" 2>/dev/null | tail -2
done

echo "=== f8_gate_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
