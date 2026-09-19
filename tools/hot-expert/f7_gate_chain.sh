#!/bin/bash
# f7_gate_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F7 (the batched
# MLA/DSA attention core in prefill), design note
# tools/hot-expert/F7-MLA-ATTN-GPU-DESIGN-2026-09-19.md.
#
# Structure copied from f2_gate_chain.sh: run_chain.sh takes the rig lock, the
# gateway comes back on EVERY exit path, the exit trap kills only glm53, the
# engine pgreps run after stop_gateway, wait_no_engine before any cp, the
# histogram is frozen per chain, an empty expert tier is a refusal, and
# accept_live.sh measures the request AFTER the chain.
#
# Gate for the item: >= 1.25x on the 18k ladder-turn TTFT (the design projects
# 1.43x; F7's own >= 8 ms/token target alone is 1.33x), turn 1 not slower,
# decode not dropped, gate_ab_verdict SEPARATED.
#
# Phase 0, PROBE (F7_PROBE_ONLY=1) -- the microbenchmark, which has to run
#   before any engine number is believed: f7_attn_probe at the real shapes
#   (H 64, L 512, V 256, width 2051, S 128/512, seen 2k/9k/18 439) through the
#   engine's own coli_vk_mla_attn, with a scalar CPU reference diff per row.
#   Runs with the gateway DOWN: it allocates ~150 MB on dev0 and, more to the
#   point, its timing means nothing while the served engine is using the card.
#
# Phase 1, ORACLE
#   O1  GLM53_MLA_ATTN_GPU unset (default) vs the served pristine
#       -> teacher_forcing IDENTICAL and GLM53_LOGIT_DUMP_ALL bit-identical,
#          deep AND shallow. This is the knob-is-inert proof and a HARD GATE.
#   O2  GLM53_MLA_ATTN_GPU=1 vs knob off, SAME binary, through kl_compare.py
#       -> mean KL < 0.0284 AND top-1 >= 99.0 % over every position, deep AND
#          shallow. The dense-selection regime (seen <= width = 2051, every
#          row attending to the whole prefix) is the first four chunks of ANY
#          run at chunk 512, so both packets exercise it; the shallow one is
#          there because it is mostly that regime rather than a tail of it.
#   Both arms of both comparisons carry the SERVED configuration:
#   GLM53_VK_SWIGLU_CLAMP=1 COLI_PREFILL_STREAM=1 GLM53_PREFILL_CHUNK=512.
#
# Phase 2, LADDER -- A,B,B,A on ONE binary, A = knob off, B = knob on.
#
# Knobs:
#   F7_PROBE_ONLY=1     phase 0 only, then stop
#   F7_ORACLE_ONLY=1    phases 0-1, then stop
#   F7_LADDER_ONLY=1    skip phases 0-1 and run only the A,B,B,A ladder
#   F7_REF64_ONLY=1     skip the probe and the ladder; after the usual
#                       stop/warm/frozen-histogram preamble, run ONE arm per
#                       packet with GLM53_MLA_ATTN_REF64=1 (GPU knob unset),
#                       same served config as the other oracle arms, writing
#                       shallow_dump_ref64.f32 (and deep_dump_ref64.f32 if
#                       F7_REF64_DEEP=1) into $OUT. Reuses the newest existing
#                       *_shallow_packet.txt / *_deep_packet.txt already in
#                       $OUT instead of generating a new one -- refuses if
#                       none is there. No KL comparison runs here (read the
#                       R1/R2 rows with f7_kl_report.sh, gateway back up).
#   F7_REF64_DEEP=1     with F7_REF64_ONLY=1, also run the deep packet
#   F7_STEPS=...        ladder steps. Default is the SHORT rung "1024,1024,2048"
#                       (directional only). The gate's own rung is
#                       "1024,1024,2048,4096,8192" (to 18 439) and is
#                       MULTI-HOUR: pass it explicitly.
#   F7_SB=128           GLM53_MLA_ATTN_SB on the B arm (sub-batch rows/submit)
#
# Launch ONLY through run_chain.sh (it takes the rig lock):
#   setsid nohup ~/src/colibri-f7/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f7/tools/hot-expert/f7_gate_chain.sh \
#       >> ~/bench/f7_gate.log 2>&1 < /dev/null &
# The engine build (`make -C c qwen38 qwen38-vk glm53 VK=1` in ~/src/colibri-f7)
# happens BEFORE this is launched, not inside it, and needs no lock. The PROBE
# binary is built inside this chain (it is not part of `make glm53`).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
F7_ROOT=$(cd "$HERE/../.." && pwd)              # the clone under test, ~/src/colibri-f7
PRISTINE=~/src/colibri                          # the served tree
PRISTINE_BIN=${F7_PRISTINE_BIN:-~/bench/glm53.f2}
[ -x "$PRISTINE_BIN" ] || PRISTINE_BIN="$PRISTINE/c/glm53"
PRISTINE_SHADERS=${F7_PRISTINE_SHADERS:-$PRISTINE/c/shaders}
CAND_BIN="$F7_ROOT/c/glm53"
CAND_SHADERS="$F7_ROOT/c/shaders"
PROBE="$HERE/f7_attn_probe"                     # built here, not checked in
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
OUT=~/bench/f7_out; mkdir -p "$OUT"
TAG=f7$(date +%m%d%H%M)
. "$HERE/gate_lib.sh"
KL="$HERE/kl_compare.py"

A_STEPS=${F7_STEPS:-1024,1024,2048}
A_GEN=${F7_GEN:-128}
A_FOLLOWUPS=${F7_FOLLOWUPS:-2}
B_SB=${F7_SB:-128}

echo "=== f7_gate_chain $TAG $(date -Is)"
echo "=== pristine=$PRISTINE_BIN candidate=$CAND_BIN"
echo "=== ladder steps=$A_STEPS gen=$A_GEN followups=$A_FOLLOWUPS  B sub-batch=$B_SB"
echo "=== gate: >= 1.25x on the 18k ladder-turn TTFT (only the 18k rung can decide"
echo "===       that; a short rung is DIRECTIONAL ONLY), turn 1 not slower"
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
  echo "=== f7_gate_chain exit rc=$rc tag=$TAG $(date -Is)"
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
for s in mla_attn_score mla_attn_softmax mla_attn_pool mla_attn_vproj; do
  [ -s "$CAND_SHADERS/$s.spv" ] || { echo "FATAL: $CAND_SHADERS/$s.spv missing"; exit 1; }
done

stop_gateway || exit 1
precheck "after-stop"

# ======================================================================
# Phase 0: the microbenchmark.
# ======================================================================
if [ "${F7_LADDER_ONLY:-0}" != 1 ] && [ "${F7_REF64_ONLY:-0}" != 1 ] \
   && [ "${F7_DEFECT_ONLY:-0}" != 1 ] && [ "${F7_GPU_ARMS_ONLY:-0}" != 1 ]; then
  echo "=== phase 0: microbenchmark $(date -Is)"
  echo "--- VRAM preflight (>= 1 GiB free on card0 -- the gateway is down)"
  used=$(VRAM 0); total=$(VRAM_TOTAL 0)
  if [ "$used" -lt 0 ] || [ "$total" -lt 0 ]; then echo "FATAL: unreadable VRAM"; exit 1; fi
  free=$((total - used))
  echo "card0: vram_used=$used vram_total=$total vram_free=$free"
  [ "$free" -ge $((1024 * 1024 * 1024)) ] || { echo "FATAL: card0 free=$free < 1 GiB"; exit 1; }

  echo "--- build f7_attn_probe (links the engine's own backend_vulkan.c)"
  BUILD_LOG="$OUT/${TAG}_probe_build.log"
  gcc -O2 -DCOLI_VULKAN "$HERE/f7_attn_probe.c" "$F7_ROOT/c/backend_vulkan.c" \
      -o "$PROBE" -lvulkan -lm > "$BUILD_LOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] || [ ! -x "$PROBE" ]; then
    echo "FATAL: probe build failed rc=$rc"; cat "$BUILD_LOG"; exit 1
  fi
  echo "build ok -> $PROBE"

  PROBE_OUT="$OUT/${TAG}_probe.txt"
  env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
      COLI_VK_SHADERS="$CAND_SHADERS" \
      "$PROBE" "$CAND_SHADERS" "${F7_PROBE_REPS:-3}" > "$PROBE_OUT" 2>&1
  rc=$?
  echo "probe rc=$rc -> $PROBE_OUT"
  if [ "$rc" -ne 0 ]; then echo "FATAL: probe exited non-zero"; tail -40 "$PROBE_OUT"; exit 1; fi
  echo "--- ROW lines (ms per layer-chunk, and the CPU-reference diff)"
  grep '^ROW ' "$PROBE_OUT"
  echo "--- INFO / WARN"
  grep -E '^(INFO|WARN) ' "$PROBE_OUT"
  if grep -q '^WARN ' "$PROBE_OUT"; then
    echo "FATAL: the probe fell back to the CPU somewhere -- the shapes it was given"
    echo "       are the engine's, so a fallback here is a bug, not a configuration."
    exit 1
  fi
fi
[ "${F7_PROBE_ONLY:-0}" = 1 ] && { echo "=== F7_PROBE_ONLY=1, stopping after phase 0"; exit 0; }

warm_glm
assert_glm_resident "pre" || { echo "FATAL: GLM not >=90% resident before the chain"; exit 1; }

# The CLI oracle path does not get a usage histogram for free; without one
# vk_preload_tier comes up with an EMPTY tier, which is not the served
# configuration and makes every numerics comparison meaningless. A per-chain
# COPY, never the canonical file.
HIST="$OUT/${TAG}_hist.bin"
cp ~/.glm53_explain.bin "$HIST" || { echo "FATAL: no ~/.glm53_explain.bin to freeze"; exit 1; }
echo "--- frozen usage histogram: $HIST ($(stat -c %s "$HIST") bytes)"

DEEP_PACKET="$OUT/${TAG}_deep_packet.txt"
SHALLOW_PACKET="$OUT/${TAG}_shallow_packet.txt"
if [ "${F7_REF64_ONLY:-0}" = 1 ]; then
  # Reuse the SAME packets shallow_dump_off.f32 / deep_dump_off.f32 were made
  # from -- do not regenerate, so the ref64 dump lines up with the existing
  # off/gpu dumps position-for-position.
  echo "--- F7_REF64_ONLY=1: reusing the newest existing packets in $OUT (not regenerating)"
  SHALLOW_PACKET=$(ls -t "$OUT"/*_shallow_packet.txt 2>/dev/null | head -1)
  DEEP_PACKET=$(ls -t "$OUT"/*_deep_packet.txt 2>/dev/null | head -1)
  [ -n "$SHALLOW_PACKET" ] && [ -s "$SHALLOW_PACKET" ] || {
    echo "FATAL: F7_REF64_ONLY=1 needs an existing *_shallow_packet.txt in $OUT -- none found, refusing to regenerate"
    exit 1
  }
  echo "    shallow packet: $SHALLOW_PACKET"
  if [ "${F7_REF64_DEEP:-0}" = 1 ]; then
    [ -n "$DEEP_PACKET" ] && [ -s "$DEEP_PACKET" ] || {
      echo "FATAL: F7_REF64_DEEP=1 needs an existing *_deep_packet.txt in $OUT -- none found, refusing to regenerate"
      exit 1
    }
    echo "    deep packet:    $DEEP_PACKET"
  fi
else
  python3 - "$F7_ROOT/tools/hot-expert/ROME-3x7900XTX-2026-09-04.md" "$DEEP_PACKET" "$SHALLOW_PACKET" \
          "${F7_DEEP_CHARS:-30000}" "${F7_SHALLOW_CHARS:-9000}" <<'PY'
import sys
src, deep, shallow, dn, sn = sys.argv[1:6]
text = open(src, encoding="utf-8").read()
q = "\n\nContinue summarising the notes above in a few sentences, without repeating what you already said."
open(deep, "w").write(text[:int(dn)] + q)
open(shallow, "w").write(text[:int(sn)] + q)
PY
  [ -s "$DEEP_PACKET" ] || { echo "FATAL: could not build the deep oracle packet"; exit 1; }
fi

# run_oracle <side> <bin> <shaders> <packet> <outtag> [extra env assignments...]
# The base env is the SERVED configuration (clamp + streaming + chunk 512), so
# O1 proves the knob inert where it actually runs.
run_oracle() {
  local side=$1 bin=$2 shaders=$3 packet=$4 outtag=$5; shift 5
  if [ "${F7_REUSE_ORACLE:-0}" = 1 ] && [ -s "$OUT/${outtag}_${side}.out" ]; then
    echo "[oracle $outtag] $side: REUSING $OUT/${outtag}_${side}.out (F7_REUSE_ORACLE=1)"; return 0
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
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt_${outtag}_${side}"
    export COLI_USAGE_PATH="$HIST"
    export GLM53_VERBOSE=1
    export GLM53_LOGIT_DUMP_ALL="$OUT/${outtag}_dump_${side}.f32"
    unset COLI_TIMERS GLM53_EXPERTS_CPU GLM53_MLA_ATTN_GPU GLM53_MLA_ATTN_SB GLM53_MLA_ATTN_REF64 GLM53_MLA_ATTN_JITTER
    for kv in "$@"; do export "${kv?}"; done
    "$bin" --model "$GLM_SNAP" --prompt "$(cat "$packet")" --logits --greedy 0
  ) > "$OUT/${outtag}_${side}.out" 2> "$OUT/${outtag}_${side}.err"
  local rc=$?
  echo "[oracle $outtag] $side rc=$rc $(grep -c ^teacher_forcing "$OUT/${outtag}_${side}.out") tf-line(s) $(grep -a -o 'prefill [0-9]* token in [0-9.]*s' "$OUT/${outtag}_${side}.err" | tail -1)"
  echo "           tier: $(grep -a 'preload dev2:\|preload dev3:\|preload:' "$OUT/${outtag}_${side}.err" | tr '\n' ' ')"
  wait_no_engine || true
  if grep -aq 'tier empty' "$OUT/${outtag}_${side}.err"; then
    echo "FATAL [oracle $outtag/$side]: the expert tier came up EMPTY -- that is not the served"
    echo "      configuration and no numerics comparison taken on it means anything. Refusing."
    return 8
  fi
  return $rc
}

if [ "${F7_REF64_ONLY:-0}" = 1 ]; then
  echo "=== F7_REF64_ONLY: float64 reference arm(s) $(date -Is)"
  echo "--- GLM53_MLA_ATTN_REF64=1, GPU knob unset, same served config as the other oracle arms"
  t0=$(date +%s)
  run_oracle "${F7_REF64_SIDE:-ref64}" "$CAND_BIN" "$CAND_SHADERS" "$SHALLOW_PACKET" shallow "${F7_REF64_KNOB:-GLM53_MLA_ATTN_REF64=1}" || exit 1
  t1=$(date +%s)
  echo "=== ref64 shallow arm wall: $((t1 - t0))s"
  [ -s "$OUT/shallow_dump_ref64.f32" ] || { echo "FATAL: shallow_dump_ref64.f32 missing after the ref64 arm"; exit 1; }
  if [ "${F7_REF64_DEEP:-0}" = 1 ]; then
    t0=$(date +%s)
    run_oracle "${F7_REF64_SIDE:-ref64}" "$CAND_BIN" "$CAND_SHADERS" "$DEEP_PACKET" deep "${F7_REF64_KNOB:-GLM53_MLA_ATTN_REF64=1}" || exit 1
    t1=$(date +%s)
    echo "=== ref64 deep arm wall: $((t1 - t0))s"
    [ -s "$OUT/deep_dump_ref64.f32" ] || { echo "FATAL: deep_dump_ref64.f32 missing after the ref64 arm"; exit 1; }
  fi
  echo "=== F7_REF64_ONLY done -- dumps on disk, no KL run inside the lock."
  echo "    With the gateway back up: $HERE/f7_kl_report.sh $OUT (reads R1/R2)"
  exit 0
fi

# ======================================================================
# F7_DEFECT_ONLY=1: the defect hunt (2026-09-19, after Fable's float64
# arbitration). One engine run on a SHORT packet with the CPU path and
# GLM53_MLA_ATTN_DUMP writes the real layer-chunk containing row 1420; the
# probe then replays that chunk through coli_vk_mla_attn and against a
# float64 reference, per row and per head. Everything after the engine run
# is seconds, and the whole thing is one gateway window.
# ======================================================================
if [ "${F7_DEFECT_ONLY:-0}" = 1 ]; then
  echo "=== defect hunt $(date -Is)"
  DPKT="$OUT/${TAG}_defect_packet.txt"
  python3 - "$F7_ROOT/tools/hot-expert/ROME-3x7900XTX-2026-09-04.md" "$DPKT" \
           "${F7_DEFECT_CHARS:-5200}" <<'PY'
import sys
src, out, n = sys.argv[1:4]
text = open(src, encoding="utf-8").read()
open(out, "w").write(text[:int(n)] + "\n\nSummarise the notes above in one sentence.")
PY
  DUMP="$OUT/${TAG}_chunk.bin"
  # F7_DEFECT_GPU=1: arm the dump on the KNOB-ON run, so the file holds the
  # context the GPU path actually produced IN SITU. Diffing that against the
  # CPU run's file for the same chunk separates "the core computed something
  # different in the engine" from "the core is fine and the divergence is
  # elsewhere" -- which a replay of the CPU run's inputs cannot do.
  DGPU=""
  [ "${F7_DEFECT_GPU:-0}" = 1 ] && DGPU="GLM53_MLA_ATTN_GPU=1"
  run_oracle dumprun "$CAND_BIN" "$CAND_SHADERS" "$DPKT" defect \
      "GLM53_MLA_ATTN_DUMP=$DUMP" "GLM53_MLA_ATTN_DUMP_ROW=${F7_DEFECT_ROW:-1420}" \
      "GLM53_MLA_ATTN_DUMP_N=${F7_DEFECT_N:-11}" ${DGPU:+"$DGPU"} || exit 1
  grep -a 'F7 chunk dump' "$OUT/defect_dumprun.err" || { echo "FATAL: no chunk was dumped"; exit 1; }
  set -- "$DUMP".*
  [ -s "$1" ] || { echo "FATAL: no $DUMP.<n> files -- is the ENGINE rebuilt at this commit?"; exit 1; }
  echo "--- $# dump file(s):"; ls -l "$@" | head -20

  echo "--- build f7_attn_probe"
  gcc -O2 -fopenmp -DCOLI_VULKAN "$HERE/f7_attn_probe.c" "$F7_ROOT/c/backend_vulkan.c" \
      -o "$PROBE" -lvulkan -lm > "$OUT/${TAG}_probe_build.log" 2>&1 \
      || { echo "FATAL: probe build failed"; cat "$OUT/${TAG}_probe_build.log"; exit 1; }

  if [ "${F7_DEFECT_REPLAY:-1}" != 1 ]; then
    echo "=== dump only (F7_DEFECT_REPLAY=0); files above. defect hunt done $(date -Is)"
    exit 0
  fi
  RP="$OUT/${TAG}_replay.txt"
  : > "$RP"
  for f in "$DUMP".*; do
    echo "=== replay $f" >> "$RP"
    env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
        COLI_VK_SHADERS="$CAND_SHADERS" F7_REPLAY="$f" \
        F7_REPLAY_ROWS="${F7_REPLAY_ROWS:-8}" F7_REPLAY_FOCUS="${F7_DEFECT_ROW:-1420}" \
        "$PROBE" "$CAND_SHADERS" >> "$RP" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || { echo "FATAL: replay of $f exited $rc"; tail -20 "$RP"; exit 1; }
  done
  echo "replay done -> $RP"
  grep -E '^(=== replay|RROW|ROW |WARN) ' "$RP"
  echo "=== defect hunt done $(date -Is)"
  exit 0
fi

# ======================================================================
# F7_GPU_ARMS_ONLY=1: re-run ONLY the knob-on arms after a shader fix. The
# off/pristine/ref64 dumps stand as long as the packets are unchanged, so
# this is 130 s + 480 s of engine instead of a full phase 1.
# ======================================================================
if [ "${F7_GPU_ARMS_ONLY:-0}" = 1 ]; then
  echo "=== gpu arms only $(date -Is)"
  for d in shallow_dump_off shallow_dump_ref64 deep_dump_off deep_dump_ref64; do
    [ -s "$OUT/$d.f32" ] || echo "WARN: $OUT/$d.f32 missing -- f7_kl_report.sh will refuse that line"
  done
  run_oracle gpu "$CAND_BIN" "$CAND_SHADERS" "$SHALLOW_PACKET" shallow \
      "GLM53_MLA_ATTN_GPU=1" "GLM53_MLA_ATTN_SB=$B_SB" "COLI_TIMERS=1" || exit 1
  # F7_O_PERROW=1: a second shallow arm whose o-projection is `tokens` separate
  # mv() calls -- literally the CPU path's own o -- so R3 vs R2 isolates the
  # batched o from the attention core with nothing else moving.
  if [ "${F7_O_PERROW:-0}" = 1 ]; then
    run_oracle gpuo "$CAND_BIN" "$CAND_SHADERS" "$SHALLOW_PACKET" shallow \
        "GLM53_MLA_ATTN_GPU=1" "GLM53_MLA_ATTN_SB=$B_SB" "COLI_TIMERS=1" \
        "GLM53_MLA_ATTN_O_PERROW=1" || exit 1
  fi
  echo "--- did the batched o-projection ever fall back to the CPU kernel?"
  grep -a "fell back to the CPU kernel" "$OUT/shallow_gpu.err" || echo "  no (it stayed on the GPU)"
  if [ "${F7_GPU_ARMS_DEEP:-1}" = 1 ]; then
    run_oracle gpu "$CAND_BIN" "$CAND_SHADERS" "$DEEP_PACKET" deep \
        "GLM53_MLA_ATTN_GPU=1" "GLM53_MLA_ATTN_SB=$B_SB" "COLI_TIMERS=1" || exit 1
  fi
  for sfx in shallow_gpu deep_gpu; do
    printf '  %-14s %s\n' "$sfx" "$(grep -a 'mla attn gpu (F7)' "$OUT/${sfx}.err" 2>/dev/null | tail -1)"
    printf '  %-14s %s\n' "$sfx" "$(grep -a -o 'prefill [0-9]* token in [0-9.]*s' "$OUT/${sfx}.err" 2>/dev/null | tail -1)"
  done
  echo "=== gpu arms done. The KL bar is OUT of the lock: $HERE/f7_kl_report.sh $OUT"
  exit 0
fi

oracle_rc=0
if [ "${F7_LADDER_ONLY:-0}" = 1 ]; then
  echo "=== phase 1 (oracle) SKIPPED: F7_LADDER_ONLY=1"
else
echo "=== phase 1: oracle $(date -Is)"

echo "--- O1: knob OFF must be bit-identical to the served pristine (deep + shallow)"
run_oracle off      "$CAND_BIN"     "$CAND_SHADERS"     "$DEEP_PACKET" deep || exit 1
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE_SHADERS" "$DEEP_PACKET" deep || exit 1
gate_compare "O1 teacher_forcing (deep)" "$OUT/deep_off.out" "$OUT/deep_pristine.out" '^teacher_forcing'
o1_tf=$?
if [ -s "$OUT/deep_dump_off.f32" ] && [ -s "$OUT/deep_dump_pristine.f32" ] \
   && cmp -s "$OUT/deep_dump_off.f32" "$OUT/deep_dump_pristine.f32"; then
  echo "  O1 logit dump (deep)                        BIT-IDENTICAL ($(wc -c < "$OUT/deep_dump_off.f32") bytes)"
  o1_dump=0
else
  echo "  O1 logit dump (deep)                        DIFFERS"
  python3 "$KL" "O1 deep (pristine vs knob-off)" "$OUT/deep_dump_pristine.f32" "$OUT/deep_dump_off.f32" || true
  o1_dump=1
fi

run_oracle off      "$CAND_BIN"     "$CAND_SHADERS"     "$SHALLOW_PACKET" shallow || exit 1
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE_SHADERS" "$SHALLOW_PACKET" shallow || exit 1
gate_compare "O1 teacher_forcing (shallow)" "$OUT/shallow_off.out" "$OUT/shallow_pristine.out" '^teacher_forcing'
o1_tf_sh=$?
if [ -s "$OUT/shallow_dump_off.f32" ] && [ -s "$OUT/shallow_dump_pristine.f32" ] \
   && cmp -s "$OUT/shallow_dump_off.f32" "$OUT/shallow_dump_pristine.f32"; then
  echo "  O1 logit dump (shallow)                     BIT-IDENTICAL"
  o1_dump_sh=0
else
  echo "  O1 logit dump (shallow)                     DIFFERS"
  python3 "$KL" "O1 shallow (pristine vs knob-off)" "$OUT/shallow_dump_pristine.f32" "$OUT/shallow_dump_off.f32" || true
  o1_dump_sh=1
fi

echo "--- O2: GLM53_MLA_ATTN_GPU=1 vs knob off, same binary, deep AND shallow"
run_oracle gpu "$CAND_BIN" "$CAND_SHADERS" "$DEEP_PACKET" deep \
    "GLM53_MLA_ATTN_GPU=1" "GLM53_MLA_ATTN_SB=$B_SB" "COLI_TIMERS=1" || exit 1
run_oracle gpu "$CAND_BIN" "$CAND_SHADERS" "$SHALLOW_PACKET" shallow \
    "GLM53_MLA_ATTN_GPU=1" "GLM53_MLA_ATTN_SB=$B_SB" "COLI_TIMERS=1" || exit 1
echo "--- the GPU core must actually have run (a silent CPU fallback would pass every bar)"
for sfx in deep_gpu shallow_gpu; do
  printf '  %-14s %s\n' "$sfx" "$(grep -a 'mla attn gpu (F7)' "$OUT/${sfx}.err" | tail -1)"
done
if ! grep -aq 'mla attn gpu (F7): calls=[1-9]' "$OUT/deep_gpu.err"; then
  echo "FATAL: the deep knob-on arm never entered the GPU core -- nothing was measured."
  oracle_rc=1
fi
gate_compare "O2 teacher_forcing (deep, gpu vs off)"    "$OUT/deep_gpu.out"    "$OUT/deep_off.out"    '^teacher_forcing'
o2_tf=$?
gate_compare "O2 teacher_forcing (shallow, gpu vs off)" "$OUT/shallow_gpu.out" "$OUT/shallow_off.out" '^teacher_forcing'
o2_tf_sh=$?
# The KL passes are MINUTES TO AN HOUR of pure Python per comparison (a 9 115
# position dump is 1.4 G floats a side) and they need no engine, no GPU and no
# lock. Running them here once cost 40+ minutes of gateway outage before the
# run was killed to get the rig back -- so, exactly as f2_kl_report.sh does,
# they are NOT run inside the lock. The chain stops after the arms; the bar is
# read afterwards by f7_kl_report.sh with the gateway up.
o2_kl=0; o2_kl_sh=0; o2_bar=0
echo "--- O2 KL bar (mean KL < 0.0284 AND top-1 >= 99.0 %): NOT run here."
echo "    The dumps are on disk. With the gateway back up, run:"
echo "      $HERE/f7_kl_report.sh $OUT"
echo "    and do not call phase 1 passed until it prints MET on both packets."

echo "--- prefill wall seconds, the deep arms"
for sfx in deep_off deep_pristine deep_gpu; do
  printf '  %-14s %s\n' "$sfx" "$(grep -a -o 'prefill [0-9]* token in [0-9.]*s' "$OUT/${sfx}.err" | tail -1)"
done
echo "--- mla split, knob off then knob on (deep)"
grep -aE 'mla split|mla attn gpu' "$OUT/deep_gpu.err" | tail -3 || true

echo "=== oracle summary"
echo "  O1 (knob off == pristine, HARD GATE): tf_deep=$o1_tf dump_deep=$o1_dump tf_shallow=$o1_tf_sh dump_shallow=$o1_dump_sh"
echo "  O2 (gpu vs off): tf_deep=$o2_tf tf_shallow=$o2_tf_sh kl_deep=$o2_kl kl_shallow=$o2_kl_sh bar=$o2_bar"
if [ "$o1_tf" != 0 ] || [ "$o1_dump" != 0 ] || [ "$o1_tf_sh" != 0 ] || [ "$o1_dump_sh" != 0 ]; then
  echo "FATAL: O1 failed -- GLM53_MLA_ATTN_GPU unset is NOT inert. Stopping before the ladder."
  oracle_rc=1
fi
for d in deep_dump_off deep_dump_gpu shallow_dump_off shallow_dump_gpu; do
  [ -s "$OUT/$d.f32" ] || { echo "FATAL: $OUT/$d.f32 missing or empty -- f7_kl_report.sh has nothing to read"; oracle_rc=1; }
done
[ "$oracle_rc" = 0 ] || exit 1
echo "=== phase 1 ARMS DONE (O1 passed; the KL bar is f7_kl_report.sh's, out of the lock)"
fi

[ "${F7_ORACLE_ONLY:-0}" = 1 ] && { echo "=== F7_ORACLE_ONLY=1, stopping after phase 1"; exit 0; }

# ======================================================================
# Phase 2: the ladder. A = knob off, B = knob on, SAME binary, A,B,B,A.
# ======================================================================
run_arm() {   # run_arm <arm-label> <gpu 0|1>
  local arm=$1 gpu=$2
  local json="$OUT/${TAG}_${arm}.jsonl" elog="$OUT/${TAG}_${arm}_engine.log" \
        console="$OUT/${TAG}_${arm}_console.log" rc
  echo "=== arm $arm (GLM53_MLA_ATTN_GPU=$gpu sb=$B_SB) $(date -Is)"
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
  # The served configuration, in BOTH arms: F7 is the only difference.
  export GLM53_VK_SWIGLU_CLAMP=1 COLI_PREFILL_STREAM=1 GLM53_PREFILL_CHUNK=512
  if [ "$gpu" = 1 ]; then
    export GLM53_MLA_ATTN_GPU=1 GLM53_MLA_ATTN_SB="$B_SB"
  else
    unset GLM53_MLA_ATTN_GPU GLM53_MLA_ATTN_SB
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
  unset GLM53_MLA_ATTN_GPU GLM53_MLA_ATTN_SB
  wait_no_engine || rc=1
  return $rc
}

echo "=== phase 2: ladder A,B,B,A $(date -Is)  steps=$A_STEPS"
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

echo "--- TTFT of the deepest ladder turn and of turn 1, all four arms, through gate_ab_verdict"
python3 - "$TAG" "$OUT" A1 B1 B2 A2 > "$OUT/${TAG}_ttft.env" <<'PY'
import json, sys
tag, out, *arms = sys.argv[1:]
first, deep = {}, {}
for arm in arms:
    rows = []
    try:
        for line in open(f"{out}/{tag}_{arm}.jsonl"):
            line = line.strip()
            if line: rows.append(json.loads(line))
    except OSError:
        continue
    lad = [r for r in rows if r.get("kind") in (None, "ladder", "step")]
    if not lad: lad = rows
    if not lad: continue
    def ttft(r):
        for k in ("ttft", "ttft_s", "ttft_sec"):
            if k in r: return float(r[k])
        return None
    t0 = ttft(lad[0])
    td = ttft(max(lad, key=lambda r: r.get("ctx", r.get("prompt_tokens", 0))))
    if t0 is not None: first[arm] = t0
    if td is not None: deep[arm] = td
def emit(name, d):
    A = " ".join(f"{d[a]:.3f}" for a in ("A1","A2") if a in d)
    B = " ".join(f"{d[a]:.3f}" for a in ("B1","B2") if a in d)
    print(f"{name}_A='{A}'"); print(f"{name}_B='{B}'")
emit("TURN1", first); emit("DEEP", deep)
PY
cat "$OUT/${TAG}_ttft.env"
. "$OUT/${TAG}_ttft.env"
gate_ab_verdict "TTFT turn 1 (s, lower is better)"    "${TURN1_A:-}" "${TURN1_B:-}"
gate_ab_verdict "TTFT deepest turn (s, lower better)" "${DEEP_A:-}"  "${DEEP_B:-}"
python3 - "${TURN1_A:-}" "${TURN1_B:-}" "${DEEP_A:-}" "${DEEP_B:-}" <<'PY'
import sys
def mean(s):
    v=[float(x) for x in s.split()] if s.strip() else []
    return sum(v)/len(v) if v else None
t1a,t1b,da,db = (mean(x) for x in sys.argv[1:5])
print("=== gate arithmetic (F7: >= 1.25x on the 18k turn, turn 1 not slower)")
for name,a,b,bar in (("turn 1",t1a,t1b,1.0),("deepest turn",da,db,1.25)):
    if a is None or b is None or b == 0:
        print(f"  {name:<14} NO NUMBER"); continue
    print(f"  {name:<14} A={a:.1f}s B={b:.1f}s  speedup={a/b:.2f}x  bar={bar}x  "
          f"{'MET' if a/b >= bar else 'NOT MET'}")
print("  (a short rung is DIRECTIONAL ONLY: the gate's own numbers are the 18k rung,")
print("   A=672.6 s on the deepest turn and 67.6 s on turn 1, record §F2-LADDER.)")
PY

echo "--- mla split and the F7 accounting, per arm, deepest turn"
for arm in A1 B1 B2 A2; do
  echo "  [$arm]"; grep -aE 'mla split|mla attn gpu' "$OUT/${TAG}_${arm}_engine.log" 2>/dev/null | tail -3
done

echo "=== f7_gate_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
