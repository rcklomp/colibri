#!/bin/bash
# f2_gate_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F2 (chunked
# batched prefill with expert streaming), design note
# tools/hot-expert/F2-STREAM-PREFILL-DESIGN-2026-09-19.md.
#
# Gate (plan rev 16): >= 1.6x on the 18k ladder-turn TTFT (<= 737 s against
# 1 179 s) and >= 2.0x on turn 1, A,B,B,A, decode not dropped, and the request
# AFTER the tested one measured (accept_live.sh in the exit trap).
#
# Phase 1, ORACLE -- four comparisons, all on ONE candidate binary except O1:
#   O1  COLI_PREFILL_STREAM=0 (default) vs the served pristine
#       -> teacher_forcing IDENTICAL and GLM53_LOGIT_DUMP_ALL bit-identical.
#          This is the knob-is-inert proof and it is a hard gate.
#   O2  COLI_PREFILL_STREAM=1 vs COLI_PREFILL_STREAM=0
#       -> the shipping delta. Reported (cosine / max-abs / argmax), not gated:
#          the routed-expert GPU kernel does not apply GLM-5.3's swiglu clamp
#          and the CPU path does, so moving an expert from CPU to GPU changes
#          numerics BY DESIGN on the non-resident share of the calls.
#   O3  COLI_PREFILL_STREAM=1 vs GLM53_EXPERTS_CPU=2
#       -> placement ONLY: =2 is "every routed expert on the CPU, UNCLAMPED",
#          so neither side clamps and the only difference left is where the
#          arithmetic ran and in what order it summed. This is the real
#          correctness proof of the streamed kernel; it is gated on argmax
#          agreement and a cosine bar.
#   O4  GLM53_PREFILL_CHUNK=512 with streaming OFF vs the default 128
#       -> the chunk change on its own. NOT bit-identical by construction
#          (the union is built in first-appearance order over the whole chunk,
#          so a token's 8 routed contributions are summed in a different order
#          when the chunk boundary moves); reported, not gated.
# O1 runs at both depths, O2 and O4 at the deep one (>= 4000 tokens), O3 at
# the shallow one -- its all-CPU reference costs roughly 9x the MoE bucket and
# a deep O3 is an hour of gateway outage that proves nothing the shallow one
# does not (the shallow packet is still several chunks at chunk 512).
#
# Packet sizes: F2_DEEP_CHARS (24000, ~5000 tokens) and F2_SHALLOW_CHARS
# (6000). Shrink both for a smoke run that only proves the chain and the
# engine start.
#
# Phase 2, LADDER -- A,B,B,A on ONE binary, A = knob off, B = knob on,
# COLI_TIMERS=1 on every arm, verdict through gate_lib.sh.
#
# Knobs:
#   F2_REUSE_ORACLE=1   skip phase 1 (phase 2 separately launchable)
#   F2_ORACLE_ONLY=1    run phase 1 and stop
#   F2_STEPS=...        ladder steps. Default is the SHORT rung to ~4.5k
#                       ("1024,1024,2048"), which is a directional A,B,B,A in
#                       well under an hour. The gate's own rung is
#                       "1024,1024,2048,4096,8192" (to 18k) and is MULTI-HOUR:
#                       pass it explicitly.
#   F2_CHUNK=512        the chunk the B arm uses (A always uses the default)
#
# Launch ONLY through run_chain.sh (it takes the rig lock):
#   setsid nohup ~/src/colibri-f2/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f2/tools/hot-expert/f2_gate_chain.sh \
#       >> ~/bench/f2_gate.log 2>&1 < /dev/null &
# The build (`make -C c qwen38 qwen38-vk glm53 VK=1` in ~/src/colibri-f2)
# happens BEFORE this is launched, not inside it, and needs no lock.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
F2_ROOT=$(cd "$HERE/../.." && pwd)              # the clone under test, e.g. ~/src/colibri-f2
PRISTINE=~/src/colibri                          # the served tree
PRISTINE_BIN=${F2_PRISTINE_BIN:-~/bench/glm53.f2base}
[ -x "$PRISTINE_BIN" ] || PRISTINE_BIN="$PRISTINE/c/glm53"
PRISTINE_SHADERS=${F2_PRISTINE_SHADERS:-$PRISTINE/c/shaders}
CAND_BIN="$F2_ROOT/c/glm53"
CAND_SHADERS="$F2_ROOT/c/shaders"
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
OUT=~/bench/f2_out; mkdir -p "$OUT"
TAG=f2$(date +%m%d%H%M)
. "$HERE/gate_lib.sh"

A_STEPS=${F2_STEPS:-1024,1024,2048}
A_GEN=${F2_GEN:-128}
A_FOLLOWUPS=${F2_FOLLOWUPS:-2}
B_CHUNK=${F2_CHUNK:-512}

echo "=== f2_gate_chain $TAG $(date -Is)"
echo "=== pristine=$PRISTINE_BIN candidate=$CAND_BIN"
echo "=== ladder steps=$A_STEPS gen=$A_GEN followups=$A_FOLLOWUPS  B chunk=$B_CHUNK"
echo "=== gate: >= 1.6x on the 18k ladder-turn TTFT and >= 2.0x on turn 1 (only the"
echo "===       18k rung can decide that; a short rung is DIRECTIONAL ONLY)"
echo "=== THE GATEWAY IS DOWN FOR THE WHOLE CHAIN"

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }

precheck() {
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  for e in glm53 qwen38 qwen38-vk; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  echo "[$label] pgrep -f 'glm53[.]': $(pgrep -f "glm53[.]" | wc -l)"
  for c in 0 1 2; do echo "[$label] card$c vram_used=$(VRAM "$c")"; done
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
    pgrep -x glm53 >/dev/null || ! pgrep -f "glm53[.]" >/dev/null || true
    if ! pgrep -x glm53 >/dev/null && ! pgrep -f "glm53[.]" >/dev/null; then return 0; fi
    sleep 2
  done
  echo "FATAL: a glm53 is still alive after 240 s"; return 1
}

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT -u COLI_TIMERS \
      -u COLI_PREFILL_STREAM -u GLM53_PREFILL_CHUNK -u GLM53_EXPERTS_CPU \
      -u GLM53_LOGIT_DUMP_ALL -u COLI_PREFILL_RING_SLOTS \
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
  echo "=== f2_gate_chain exit rc=$rc tag=$TAG $(date -Is)"
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

# The CLI oracle path does NOT get a usage histogram for free the way
# ttft_serve.py's engine mode does (it copies ~/.glm53_explain.bin when
# COLI_USAGE_PATH is unset). Without it vk_preload_tier prints
# "[VK] no usage history loaded -- tier empty" and the oracle measures a
# configuration that never serves: zero resident experts, so streaming moves
# 100 % of the routed calls instead of the ~11 % it moves in service, and the
# numerics delta it reports says nothing about the shipped engine. Measured
# 2026-09-19: a 1 064-token oracle prefill with no tier is 177.0 s, with the
# tier it is a different engine entirely. A per-chain COPY, never the
# canonical file, so a run cannot mutate what the next one preloads from.
HIST="$OUT/${TAG}_hist.bin"
cp ~/.glm53_explain.bin "$HIST" || { echo "FATAL: no ~/.glm53_explain.bin to freeze"; exit 1; }
echo "--- frozen usage histogram: $HIST ($(stat -c %s "$HIST") bytes)"

DEEP_PACKET="$OUT/${TAG}_deep_packet.txt"
SHALLOW_PACKET="$OUT/${TAG}_shallow_packet.txt"
python3 - "$F2_ROOT/tools/hot-expert/ROME-3x7900XTX-2026-09-04.md" "$DEEP_PACKET" "$SHALLOW_PACKET" \
        "${F2_DEEP_CHARS:-24000}" "${F2_SHALLOW_CHARS:-6000}" <<'PY'
import sys
src, deep, shallow, dn, sn = sys.argv[1:6]
text = open(src, encoding="utf-8").read()
q = "\n\nContinue summarising the notes above in a few sentences, without repeating what you already said."
open(deep, "w").write(text[:int(dn)] + q)
open(shallow, "w").write(text[:int(sn)] + q)
PY
[ -s "$DEEP_PACKET" ] || { echo "FATAL: could not build the deep oracle packet"; exit 1; }

# run_oracle <side> <bin> <shaders> <packet> <outtag> [extra env assignments...]
run_oracle() {
  local side=$1 bin=$2 shaders=$3 packet=$4 outtag=$5; shift 5
  if [ "${F2_REUSE_ORACLE:-0}" = 1 ] && [ -s "$OUT/${outtag}_${side}.out" ]; then
    echo "[oracle $outtag] $side: REUSING $OUT/${outtag}_${side}.out (F2_REUSE_ORACLE=1)"; return 0
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
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt_${outtag}_${side}"
    export COLI_USAGE_PATH="$HIST"
    export GLM53_VERBOSE=1
    export GLM53_LOGIT_DUMP_ALL="$OUT/${outtag}_dump_${side}.f32"
    unset COLI_TIMERS COLI_PREFILL_STREAM GLM53_PREFILL_CHUNK GLM53_EXPERTS_CPU
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

# logit_diff <label> <A.f32> <B.f32>  -- cosine / max-abs / argmax agreement.
#
# Two things this got wrong on its first run and does not get wrong again.
# (1) It guessed the vocab. The dump carries a 16-byte header (magic "GLKD",
#     version, count, vocab -- dump_logits_all in c/glm53.c); GLM-5.3's vocab is
#     154 880, the guess was 155 136, and a wrong vocab turns the argmax check
#     into nonsense instead of an error. It reads the header now.
# (2) It looped element-wise in Python. There is no numpy on this rig and a
#     5 000-position dump is 774M floats: everything heavy here is a C-level
#     builtin over array.array (per-row max+index for the argmaxes,
#     map(sub)/map(abs) for the largest difference), and the cosine is taken on
#     the LAST position, which is CLAUDE.md's own convention.
logit_diff() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys, array, math, struct
from operator import sub, mul
label, pa, pb = sys.argv[1:4]
try:
    a = open(pa, "rb").read(); b = open(pb, "rb").read()
except OSError as e:
    print(f"  {label:<44} REFUSED: {e}"); sys.exit(2)
if not a or not b:
    print(f"  {label:<44} REFUSED: empty dump (A={len(a)}B B={len(b)}B) is not a comparison"); sys.exit(2)
if len(a) != len(b):
    print(f"  {label:<44} REFUSED: different lengths ({len(a)} vs {len(b)} bytes)"); sys.exit(2)
if len(a) < 16 or a[:4] != b"GLKD" or b[:4] != b"GLKD":
    print(f"  {label:<44} REFUSED: not a GLKD logit dump"); sys.exit(2)
_, _, rows, V = struct.unpack("<4I", a[:16])
_, _, rows_b, V_b = struct.unpack("<4I", b[:16])
if (rows, V) != (rows_b, V_b) or rows < 1 or V < 1:
    print(f"  {label:<44} REFUSED: shapes {rows}x{V} vs {rows_b}x{V_b}"); sys.exit(2)
if a == b:
    print(f"  {label:<44} BIT-IDENTICAL ({rows} x {V})"); sys.exit(0)
fa = array.array('f'); fa.frombytes(a[16:16 + rows*V*4])
fb = array.array('f'); fb.frombytes(b[16:16 + rows*V*4])
if len(fa) != rows*V:
    print(f"  {label:<44} REFUSED: {len(fa)} floats, header says {rows}x{V}"); sys.exit(2)
mx = max(map(abs, map(sub, fa, fb)))
agree = 0
for r in range(rows):
    ra = fa[r*V:(r+1)*V]; rb = fb[r*V:(r+1)*V]
    if ra.index(max(ra)) == rb.index(max(rb)): agree += 1
la = fa[(rows-1)*V:]; lb = fb[(rows-1)*V:]
dot = sum(map(mul, la, lb))
na = math.sqrt(sum(map(mul, la, la))); nb = math.sqrt(sum(map(mul, lb, lb)))
cos = dot / (na * nb) if na > 0 and nb > 0 else float("nan")
print(f"  {label:<44} last-token cosine={cos:.8f} max_abs={mx:.6g} "
      f"argmax {agree}/{rows} positions agree")
sys.exit(0 if (cos >= 0.9999 and agree == rows) else 1)
PY
}

oracle_rc=0
if [ "${F2_LADDER_ONLY:-0}" = 1 ]; then
  echo "=== phase 1 (oracle) SKIPPED: F2_LADDER_ONLY=1"
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
  logit_diff "O1 logit dump (deep)" "$OUT/deep_dump_pristine.f32" "$OUT/deep_dump_off.f32"; o1_dump=1
fi

run_oracle off      "$CAND_BIN"     "$CAND_SHADERS"     "$SHALLOW_PACKET" shallow || exit 1
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE_SHADERS" "$SHALLOW_PACKET" shallow || exit 1
gate_compare "O1 teacher_forcing (shallow)" "$OUT/shallow_off.out" "$OUT/shallow_pristine.out" '^teacher_forcing'
o1_tf_sh=$?
if cmp -s "$OUT/shallow_dump_off.f32" "$OUT/shallow_dump_pristine.f32"; then
  echo "  O1 logit dump (shallow)                     BIT-IDENTICAL"
  o1_dump_sh=0
else
  logit_diff "O1 logit dump (shallow)" "$OUT/shallow_dump_pristine.f32" "$OUT/shallow_dump_off.f32"; o1_dump_sh=1
fi

echo "--- O2: streaming on vs knob off, at the deep prompt"
run_oracle on  "$CAND_BIN" "$CAND_SHADERS" "$DEEP_PACKET" deep \
    "COLI_PREFILL_STREAM=1" "COLI_TIMERS=1" || exit 1
gate_compare "O2 teacher_forcing (on vs off)" "$OUT/deep_on.out" "$OUT/deep_off.out" '^teacher_forcing'
o2_tf=$?
logit_diff "O2 logits (on vs off)"  "$OUT/deep_dump_off.f32"  "$OUT/deep_dump_on.f32"; o2_lg=$?

# O3 runs at the SHALLOW depth on purpose. GLM53_EXPERTS_CPU=2 forces all eight
# routed experts per token per layer onto the CPU where today only ~10.7 % go,
# which costs roughly 9x the MoE bucket -- a deep O3 is an hour of gateway
# outage on its own and proves nothing a shallow one does not. The shallow
# packet is still several chunks at chunk 512, so the ring, the waves and the
# accumulate order are all exercised; O2 above carries the depth.
echo "--- O3: streaming on vs the UNCLAMPED all-CPU reference (placement only), shallow"
run_oracle on   "$CAND_BIN" "$CAND_SHADERS" "$SHALLOW_PACKET" shallow \
    "COLI_PREFILL_STREAM=1" "COLI_TIMERS=1" || exit 1
run_oracle cpu2 "$CAND_BIN" "$CAND_SHADERS" "$SHALLOW_PACKET" shallow \
    "GLM53_EXPERTS_CPU=2" "GLM53_PREFILL_CHUNK=$B_CHUNK" || exit 1
gate_compare "O3 teacher_forcing (on vs EXPERTS_CPU=2)" "$OUT/shallow_on.out" "$OUT/shallow_cpu2.out" '^teacher_forcing'
o3_tf=$?
logit_diff "O3 logits (on vs EXPERTS_CPU=2)" "$OUT/shallow_dump_cpu2.f32" "$OUT/shallow_dump_on.f32"; o3_lg=$?

echo "--- O4: chunk $B_CHUNK with streaming OFF, against the default chunk"
run_oracle chunk "$CAND_BIN" "$CAND_SHADERS" "$DEEP_PACKET" deep \
    "GLM53_PREFILL_CHUNK=$B_CHUNK" || exit 1
gate_compare "O4 teacher_forcing (chunk vs default)" "$OUT/deep_chunk.out" "$OUT/deep_off.out" '^teacher_forcing'
o4_tf=$?
logit_diff "O4 logits (chunk vs default)" "$OUT/deep_dump_off.f32" "$OUT/deep_dump_chunk.f32"; o4_lg=$?

echo "--- streamed-expert accounting, knob ON then knob OFF (COLI_TIMERS=1 on the ON run)"
grep -E 'moe split|\[STREAM\]|\[PROF\] eg=|prefill ring' "$OUT/deep_on.err" | tail -8 || echo "(none)"
echo "--- prefill wall time, all four deep arms (GLM53_VERBOSE 'prefill N token' + the run's own timing)"
for sidep in off pristine on chunk; do
  printf '  %-9s %s\n' "$sidep" "$(grep -E 'prefill [0-9]+ token' "$OUT/deep_${sidep}.err" | tail -1)"
done

echo "=== oracle summary"
echo "  O1 (knob off == pristine, HARD GATE): tf_deep=$o1_tf dump_deep=$o1_dump tf_shallow=$o1_tf_sh dump_shallow=$o1_dump_sh"
echo "  O2 (on vs off, reported):             tf=$o2_tf logits=$o2_lg"
echo "  O3 (on vs EXPERTS_CPU=2, GATED):      tf=$o3_tf logits=$o3_lg"
echo "  O4 (chunk alone, reported):           tf=$o4_tf logits=$o4_lg"
if [ "$o1_tf" != 0 ] || [ "$o1_dump" != 0 ] || [ "$o1_tf_sh" != 0 ] || [ "$o1_dump_sh" != 0 ]; then
  echo "FATAL: O1 failed -- COLI_PREFILL_STREAM=0 is NOT inert. Stopping before the ladder."
  oracle_rc=1
fi
if [ "$o3_lg" = 2 ]; then
  echo "FATAL: O3 REFUSED (a dump was missing or empty) -- an empty comparison is not a pass."
  oracle_rc=1
fi
[ "$oracle_rc" = 0 ] || exit 1
echo "=== phase 1 PASS"
fi

[ "${F2_ORACLE_ONLY:-0}" = 1 ] && { echo "=== F2_ORACLE_ONLY=1, stopping after phase 1"; exit 0; }

# ======================================================================
# Phase 2: the ladder. A = knob off, B = knob on, SAME binary, A,B,B,A.
# ======================================================================
run_arm() {   # run_arm <arm-label> <stream 0|1>
  local arm=$1 stream=$2
  local json="$OUT/${TAG}_${arm}.jsonl" elog="$OUT/${TAG}_${arm}_engine.log" \
        console="$OUT/${TAG}_${arm}_console.log" rc
  echo "=== arm $arm (COLI_PREFILL_STREAM=$stream) $(date -Is)"
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
  if [ "$stream" = 1 ]; then export COLI_PREFILL_STREAM=1
  else unset COLI_PREFILL_STREAM; fi
  unset GLM53_PREFILL_CHUNK          # B's chunk default comes from the knob itself

  python3 "$HERE/context_ladder.py" \
      --engine "$CAND_BIN" \
      --steps "$A_STEPS" --gen "$A_GEN" --followups "$A_FOLLOWUPS" \
      --kv-slots 4 --warm --min-resident 90 \
      --arm "$arm" --tag "$TAG" --json "$json" \
      --engine-log "$elog" 2>&1 | tee -a "$console"
  rc=${PIPESTATUS[0]}
  echo "=== arm $arm exit=$rc json=$json engine_log=$elog"
  unset GLM53_MAXT GLM53_PREFIX_CKPT COLI_CKPT_DIR COLI_TIMERS COLI_PREFILL_STREAM
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
print("=== gate arithmetic (plan rev 16: >= 2.0x on turn 1, >= 1.6x on the 18k turn)")
for name,a,b,bar in (("turn 1",t1a,t1b,2.0),("deepest turn",da,db,1.6)):
    if a is None or b is None or b == 0:
        print(f"  {name:<14} NO NUMBER"); continue
    print(f"  {name:<14} A={a:.1f}s B={b:.1f}s  speedup={a/b:.2f}x  bar={bar}x  "
          f"{'MET' if a/b >= bar else 'NOT MET'}")
print("  (a short rung is DIRECTIONAL ONLY: the gate's own numbers are the 18k rung,")
print("   A=1179.4 s on the deepest turn and 136.5 s on turn 1, record §RP-F6a.)")
PY

echo "--- ffn_moe and the streamed-expert accounting, per arm, deepest turn"
for arm in A1 B1 B2 A2; do
  echo "  [$arm]"; grep -E 'moe split|\[STREAM\]' "$OUT/${TAG}_${arm}_engine.log" 2>/dev/null | tail -3
done

echo "=== f2_gate_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
