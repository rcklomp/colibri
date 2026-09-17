#!/bin/bash
# f6a_gate_chain.sh -- F6-INDEXER-DESIGN-2026-09-18.md, item F6a: the DSA
# indexer's score pass parallelised (G7's pattern: #pragma omp parallel for
# over the pools, each iteration writing only its own scores[p]) and its
# top-512 selection made exact with a bounded min-heap instead of the
# O(wanted*pools) greedy scan -- bit-identical by construction. GLM53_
# INDEX_SCALAR=1 restores the original serial scan unchanged, for the A/B.
#
# Order: preflight (prints only) -> stop_gateway -> wait_no_proc glm53 ->
# warm+assert GLM >=90% resident -> ORACLE (candidate vs the served pristine,
# CLI greedy mode, a >=4000-token prompt built from the record file --
# teacher_forcing via gate_compare, ALL logits via GLM53_LOGIT_DUMP_ALL bit-
# identical by cmp, GLM53_DUMP_INDEX rows diffed at that same deep prompt AND
# at a <=2000-token prompt for the dense-identical leg, plus a GLM53_INDEX_
# SCALAR=1 knob-routing check) -> the ladder, A,B,B,A to 9.5k (context_
# ladder.py engine mode, --steps 1024,1024,2048,4096 --gen 128 --followups 2,
# COLI_TIMERS=1, same env as f4_optime_chain.sh's run_A) -> verdicts
# (context_compare.py through gate_ab_verdict, plus the [OPTIME req=] index
# ms/call at the deepest turn and the two follow-ups, all four arms side by
# side) -> exit trap: re-warm GLM, restart the gateway, accept_live.sh.
#
# Launch ONLY through run_chain.sh (it takes the rig lock; this script does
# not stop the gateway on its own without it):
#   setsid nohup ~/src/colibri-f6a/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f6a/tools/hot-expert/f6a_gate_chain.sh \
#       >> ~/bench/f6a_gate.log 2>&1 < /dev/null &
#
# The build (`make -C c glm53 VK=1` in the ~/src/colibri-f6a clone) does not
# need the lock and happens BEFORE this is launched, not inside it.
#
# GLM53_PREFIX_CKPT=0 with a private COLI_CKPT_DIR on every GLM invocation
# here (CLAUDE.md: a restored checkpoint reports a prefill that never
# happened). Budget: oracle ~15 min + warm + 4 x ~30 min ladder arms +
# accept_live ~= 2.5-3 h. Gateway down for that window only; not 18k tonight
# (the owner needs the gateway by morning), 9.5k.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
F6A_ROOT=$(cd "$HERE/../.." && pwd)             # the clone under test, e.g. ~/src/colibri-f6a
PRISTINE=~/src/colibri                          # binary in service; f4_optime_chain's own convention
PRISTINE_BIN="$PRISTINE/c/glm53"
CAND_BIN="$F6A_ROOT/c/glm53"
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
OUT=~/bench/f6a_out; mkdir -p "$OUT"
TAG=f6a$(date +%m%d%H%M)
. "$HERE/gate_lib.sh"

A_STEPS="1024,1024,2048,4096"; A_GEN=128; A_FOLLOWUPS=2

echo "=== f6a_gate_chain $TAG $(date -Is)"
echo "=== pristine=$PRISTINE_BIN candidate=$CAND_BIN"
echo "=== A_STEPS=$A_STEPS gen=$A_GEN followups=$A_FOLLOWUPS (to ~9.5k, not 18k tonight)"
echo "=== THE GATEWAY IS DOWN FOR THE WHOLE CHAIN"

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }

precheck() {   # precheck <label> -- prints only
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  for e in glm53 qwen38 qwen38-vk; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  for c in 0 1 2; do
    echo "[$label] card$c vram_used=$(VRAM "$c")"
  done
}

assert_vram_free() {   # assert_vram_free <label> -- copied from f4_optime_chain.sh
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
      -u GLM53_INDEX_SCALAR -u GLM53_LOGIT_DUMP_ALL -u GLM53_DUMP_INDEX \
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

assert_glm_resident() {   # assert_glm_resident <label> -- copied from f4_optime_chain.sh
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
  pgrep -x glm53 >/dev/null 2>&1 && { pkill -9 -x glm53 2>/dev/null || true; wait_no_proc glm53; }
  echo "--- final re-warm of GLM before restart"
  warm_glm
  assert_glm_resident "final" || echo "WARNING: GLM not >=90% resident at restart time"
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== f6a_gate_chain exit rc=$rc tag=$TAG $(date -Is)"
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

# ======================================================================
# ORACLE: above the dense threshold. Below 2052 tokens the selection is a
# no-op as a SET (every visible complete pool fits inside `wanted`), so the
# real prize and the real numerics risk both live above it -- the oracle
# prompt must be >=4000 tokens (F6a's own instruction; the design's own
# packet, x2_packet_450.txt, tokenises to 564 positions and would print a
# guaranteed, meaningless PASS here).
# ======================================================================
echo "=== oracle $(date -Is)"
DEEP_PACKET="$OUT/${TAG}_deep_packet.txt"
SHALLOW_PACKET="$OUT/${TAG}_shallow_packet.txt"
# Built from the record file itself (ttft_serve.DEFAULT_TEXT's own corpus),
# pinned to the candidate clone's copy so the chain is reproducible from one
# tree. ~24000 chars clears 4000 tokens comfortably for this tokenizer
# (English+markdown prose runs well under 6 chars/token); the run's own
# "prefill N token" line (GLM53_VERBOSE=1) is checked below rather than
# assumed from the char count.
python3 - "$F6A_ROOT/tools/hot-expert/ROME-3x7900XTX-2026-09-04.md" "$DEEP_PACKET" "$SHALLOW_PACKET" <<'PY'
import sys
src, deep, shallow = sys.argv[1:4]
text = open(src, encoding="utf-8").read()
q = "\n\nContinue summarising the notes above in a few sentences, without repeating what you already said."
open(deep, "w").write(text[:24000] + q)
open(shallow, "w").write(text[:6000] + q)
PY
[ -s "$DEEP_PACKET" ] || { echo "FATAL: could not build the deep oracle packet"; exit 1; }

run_oracle() {   # run_oracle <side> <bin> <shaders> <packet> <outtag> [extra_env="NAME=val"]
  local side=$1 bin=$2 shaders=$3 packet=$4 outtag=$5 extra_env=${6:-}
  for e in glm53 qwen38 qwen38-vk; do
    pgrep -x "$e" >/dev/null 2>&1 && { echo "FATAL: $e already running before oracle $outtag-$side"; return 9; }
  done
  rm -rf "$OUT/ckpt_${outtag}_${side}"; mkdir -p "$OUT/ckpt_${outtag}_${side}"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    # The CLI oracle path runs the single-slot CPU KDA recurrence
    # (COLI_KDA_GPU=0), same as prefill_gate.sh's default -- §X2/§P13's
    # finding about the single-slot CLI path, not the serving default.
    export COLI_KDA_GPU=0
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt_${outtag}_${side}"
    export GLM53_VERBOSE=1
    export GLM53_LOGIT_DUMP_ALL="$OUT/${outtag}_dump_${side}.f32"
    export GLM53_DUMP_INDEX=1
    unset COLI_TIMERS GLM53_INDEX_SCALAR
    [ -n "$extra_env" ] && export "${extra_env?}"
    "$bin" --model "$GLM_SNAP" --prompt "$(cat "$packet")" --logits --greedy 0
  ) > "$OUT/${outtag}_${side}.out" 2> "$OUT/${outtag}_${side}.err"
  local rc=$?
  echo "[oracle $outtag] $side rc=$rc $(grep -c ^teacher_forcing "$OUT/${outtag}_${side}.out") tf-line(s) $(grep -o 'prefill [0-9]* token' "$OUT/${outtag}_${side}.err" | tail -1)"
  wait_no_proc glm53 || true
  return $rc
}

echo "--- deep oracle (>=4000 tokens): candidate vs pristine, timers off"
run_oracle candidate "$CAND_BIN" "$F6A_ROOT/c/shaders" "$DEEP_PACKET" deep || exit 1
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE/c/shaders" "$DEEP_PACKET" deep || exit 1

gate_compare "F6a oracle teacher_forcing (deep)" \
  "$OUT/deep_candidate.out" "$OUT/deep_pristine.out" '^teacher_forcing'
tf_rc=$?
echo "=== teacher_forcing gate_compare exit=$tf_rc (0=IDENTICAL 1=DIFFERS 2=REFUSED)"

echo "--- logit dump (GLM53_LOGIT_DUMP_ALL, every position x vocab, raw f32): cmp"
if [ -s "$OUT/deep_dump_pristine.f32" ] && [ -s "$OUT/deep_dump_candidate.f32" ] \
   && cmp -s "$OUT/deep_dump_pristine.f32" "$OUT/deep_dump_candidate.f32"; then
  echo "F6a oracle logit dump: BIT-IDENTICAL ($(wc -c < "$OUT/deep_dump_pristine.f32") bytes)"
  dump_rc=0
else
  dump_rc=1
  python3 - "$OUT/deep_dump_pristine.f32" "$OUT/deep_dump_candidate.f32" <<'PY'
import sys, array
try:
    a = open(sys.argv[1], "rb").read(); b = open(sys.argv[2], "rb").read()
except OSError as e:
    print(f"F6a oracle logit dump: REFUSED, could not read a dump ({e})"); sys.exit(0)
if not a or not b:
    print(f"F6a oracle logit dump: REFUSED, an empty dump (A={len(a)}B B={len(b)}B) is not a comparison")
    sys.exit(0)
fa = array.array('f'); fa.frombytes(a[: (len(a)//4)*4])
fb = array.array('f'); fb.frombytes(b[: (len(b)//4)*4])
n = min(len(fa), len(fb))
mx = max(abs(fa[i] - fb[i]) for i in range(n)) if n else float("nan")
print(f"F6a oracle logit dump: DIFFERS -- {len(a)} vs {len(b)} bytes, "
      f"max_abs={mx:.6g} over {n} common floats (must be exactly 0)")
PY
fi
echo "=== logit dump exit=$dump_rc"

echo "--- index selection dump (GLM53_DUMP_INDEX, stderr 'index q=... ->' rows): diff, deep prompt"
if diff <(grep '^index q=' "$OUT/deep_pristine.err") <(grep '^index q=' "$OUT/deep_candidate.err") >/dev/null 2>&1 \
   && [ -n "$(grep -c '^index q=' "$OUT/deep_pristine.err")" ] && [ "$(grep -c '^index q=' "$OUT/deep_pristine.err")" -gt 0 ]; then
  echo "F6a index dump (deep, set AND rank order): IDENTICAL ($(grep -c '^index q=' "$OUT/deep_pristine.err") row(s))"
  idx_rc=0
else
  echo "F6a index dump (deep): DIFFERS or empty -- first differing lines:"
  diff <(grep '^index q=' "$OUT/deep_pristine.err") <(grep '^index q=' "$OUT/deep_candidate.err") | head -20
  idx_rc=1
fi

echo "--- dense-identical leg (<=2000 tokens): index dump must still match (checked, not assumed)"
run_oracle candidate "$CAND_BIN" "$F6A_ROOT/c/shaders" "$SHALLOW_PACKET" shallow || exit 1
run_oracle pristine "$PRISTINE_BIN" "$PRISTINE/c/shaders" "$SHALLOW_PACKET" shallow || exit 1
gate_compare "F6a oracle teacher_forcing (shallow, dense-identical)" \
  "$OUT/shallow_candidate.out" "$OUT/shallow_pristine.out" '^teacher_forcing'
tf_shallow_rc=$?
if diff <(grep '^index q=' "$OUT/shallow_pristine.err") <(grep '^index q=' "$OUT/shallow_candidate.err") >/dev/null 2>&1 \
   && [ "$(grep -c '^index q=' "$OUT/shallow_pristine.err")" -gt 0 ]; then
  echo "F6a index dump (shallow, dense-identical): IDENTICAL ($(grep -c '^index q=' "$OUT/shallow_pristine.err") row(s))"
  idx_shallow_rc=0
else
  echo "F6a index dump (shallow): DIFFERS or empty"
  idx_shallow_rc=1
fi

echo "--- knob-routing check: candidate with GLM53_INDEX_SCALAR=1 must still match pristine"
run_oracle scalar "$CAND_BIN" "$F6A_ROOT/c/shaders" "$DEEP_PACKET" deep "GLM53_INDEX_SCALAR=1" || exit 1
gate_compare "F6a knob-routing teacher_forcing (GLM53_INDEX_SCALAR=1 vs pristine)" \
  "$OUT/deep_scalar.out" "$OUT/deep_pristine.out" '^teacher_forcing'
scalar_rc=$?

if [ "$tf_rc" != 0 ] || [ "$dump_rc" != 0 ] || [ "$idx_rc" != 0 ] || \
   [ "$tf_shallow_rc" != 0 ] || [ "$idx_shallow_rc" != 0 ] || [ "$scalar_rc" != 0 ]; then
  echo "FATAL: F6a oracle did not come back fully IDENTICAL -- stopping before the ladder"
  echo "  teacher_forcing(deep)=$tf_rc logit_dump=$dump_rc index_dump(deep)=$idx_rc"
  echo "  teacher_forcing(shallow)=$tf_shallow_rc index_dump(shallow)=$idx_shallow_rc knob_routing=$scalar_rc"
  exit 1
fi
echo "=== oracle PASS: teacher_forcing, ALL-position logits, and the index selection dump (set + rank order) are IDENTICAL at deep and shallow depths; the GLM53_INDEX_SCALAR=1 knob routes to the same pristine code path"

# ======================================================================
# LADDER: A,B,B,A to ~9.5k (not 18k tonight). Exactly f4_optime_chain.sh's
# run_A shape and env.
# ======================================================================
run_arm() {   # run_arm <arm-label> <binary> <shaders>
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
  wait_no_proc glm53 || rc=1
  return $rc
}

echo "=== ladder A,B,B,A $(date -Is)"
warm_glm
assert_glm_resident "pre-ladder" || { echo "FATAL: GLM not >=90% resident before the ladder"; exit 1; }

run_arm A1 "$PRISTINE_BIN" "$PRISTINE/c/shaders" || exit 1
run_arm B1 "$CAND_BIN"     "$F6A_ROOT/c/shaders" || exit 1
run_arm B2 "$CAND_BIN"     "$F6A_ROOT/c/shaders" || exit 1
run_arm A2 "$PRISTINE_BIN" "$PRISTINE/c/shaders" || exit 1
precheck "post-ladder"
assert_vram_free "post-ladder" || exit 1

# ======================================================================
# VERDICTS
# ======================================================================
echo "=== verdicts $(date -Is)"
echo "--- context_compare.py (decode per turn, ttft_inc for the follow-ups), through gate_ab_verdict"
python3 "$HERE/context_compare.py" --rows \
    A1="$OUT/${TAG}_A1.jsonl" B1="$OUT/${TAG}_B1.jsonl" \
    B2="$OUT/${TAG}_B2.jsonl" A2="$OUT/${TAG}_A2.jsonl"

echo "--- [OPTIME req=] mla index ms/call at the deepest ladder turn and the two follow-ups, all four arms"
python3 - "$TAG" "$OUT" A1 B1 B2 A2 <<'PY'
import re, sys
tag, out, *arms = sys.argv[1:]
pat = re.compile(r'\[OPTIME req=(\d+) ctx=(\d+)\].*mla split \(n=(\d+), mean ctx=(\d+)\).*'
                  r'index=([\d.]+)s \(([\d.]+) ms\)')
for arm in arms:
    path = f"{out}/{tag}_{arm}_engine.log"
    rows = []
    try:
        for line in open(path):
            m = pat.search(line)
            if m:
                rows.append((int(m.group(1)), int(m.group(2)), float(m.group(6))))
    except OSError as e:
        print(f"  {arm:<4} REFUSED: could not read {path} ({e})")
        continue
    if not rows:
        print(f"  {arm:<4} REFUSED: no '[OPTIME req=] mla split' line in {path} -- an empty "
              f"comparison is not a result")
        continue
    # The deepest ladder turn and the two follow-ups are the LAST three
    # requests in the log, regardless of exact req numbering (a warm-up
    # request may or may not precede the ladder).
    last3 = rows[-3:] if len(rows) >= 3 else rows
    cells = "  ".join(f"req={r} ctx={c} index_ms/call={ix:.3f}" for r, c, ix in last3)
    print(f"  {arm:<4} {cells}")
PY

echo "--- design's projection at 9.5k, for reference: index/call 12.6 -> ~1.7 ms; the two "
echo "    follow-ups (ctx ~9.6k): decode from 3.04 tok/s toward ~4.5"

echo "=== f6a_gate_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
