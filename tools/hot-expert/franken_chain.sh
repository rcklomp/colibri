#!/bin/bash
# franken_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item H2: the head-to-
# head chain (A,B,B,A + C). Full run estimate ~4 h (plan §2.2 H2 row: A1 ~70
# min + B1 ~20 + B2 ~20 + A2 ~70 + C <=30 + warm/restart ~20).
# THE GATEWAY IS DOWN FOR THE WHOLE CHAIN. Schedule at night and say so.
#
# Launch through run_chain.sh, never directly -- it takes the rig lock, and
# the gateway watchdog (cron, every 5 min) restarts the gateway under any
# measurement that stopped it without the lock:
#
#   setsid nohup ~/src/colibri-h2/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-h2/tools/hot-expert/franken_chain.sh [--smoke] [--skip-b] \
#       >> ~/bench/franken_chain.log 2>&1 < /dev/null &
#
# Flags:
#   --smoke    tiny steps for proving the chain's shape, not a measurement:
#              A/B --steps 256,256 --gen 16 --followups 1, plus a 2-size cold
#              sweep on B (1024,2048); C --steps 256 --gen 16 --followups 1
#              (2 rows: one ladder step, one followup). Budget: <= 40 min
#              (qwen36's first load is cold on the NVMe).
#   --skip-b   B (hipFire) DOES run by default now: FRANKEN-H0d (branch
#              perf/franken-h0, record §FRANKEN-H0d, merged into
#              hot-expert-tier) found the box's system ROCm 6.2.0 was the
#              blocker (no rocm-device-libs), not hipFire or the box's
#              hardware -- a user-local ROCm 10.0.0 venv (~/venvs/rocm)
#              rebuilds and runs it clean (ttft ~248 ms, decode 148-164
#              tok/s at empty context). --skip-b is kept as an escape hatch
#              for a future blocker on this arm specifically: it replaces B
#              with a logged "B SKIPPED: <reason>" line -- no hipfire process
#              started, no GPU touched -- and the chain still runs A1, A2, C.
#
# Order (plan §2.1, §2.2): stop gateway -> wait_no_engine -> warm GLM, assert
# >=90% -> A1 -> stop engine, assert VRAM<1GiB on every card, record DPM
# level per card -> B1 (hipFire, throwaway + ladder to 64Ki + cold sweep) ->
# stop, assert VRAM free -> B2 (fresh process, identical) -> assert VRAM
# free, re-warm GLM, assert -> A2 (identical to A1) -> C (`coli serve` on
# qwen36, HTTP ladder, one repeat, 30 min cap) -> re-warm GLM -> restart
# gateway on every exit path -> accept_live.sh (the request AFTER the chain
# is part of the measurement, CLAUDE.md).
#
# An A arm is INVALID only on residency: below 90% at arm start, or any
# turn's residency check settling below 90% (checked below from each A
# run's own console log, not assumed -- context_ladder.py aborts outright
# on a residency floor breach mid-ladder, so an A run that DID complete but
# logged a borderline value is what this re-checks, the H1 gate's own
# majflt=7805-at-90.25%-resident case). majflt does NOT invalidate an arm:
# commit c790851 established that a loaded glm53's ~89 GB anon memory beside
# the 183 GiB model on 247 GiB caps residency near 92% with the engine up,
# so every turn major-faults reading its own working set in and majflt>0 is
# this box's own floor, not a defect -- the reference ladder ctx09152003
# (majflt 21663/6611/4521/4176, turns 1-4) is what majflt is compared
# against instead: a turn is flagged MAJFLT-HIGH, not invalid, when it
# exceeds 2x that same-turn reference (turns beyond 4 compare to turn 4),
# and the summary names the flagged turns so H3's table can annotate rather
# than drop them.
#
# GLM53_PREFIX_CKPT=0 with a private COLI_CKPT_DIR on every GLM invocation
# (CLAUDE.md: a restored checkpoint reports a prefill that never happened).
#
# THE THINKING DECISION (coordinator amendment, 2026-09-16, after FRANKEN-H0d
# found the B arm runs): hipFire ERRORS ("open think span at end of
# generation") when --max-tokens ends inside <think> -- measured at 16 and 64
# tokens, clean at 900 (FRANKEN-H0d2). B disables thinking: every B request
# carries chat_template_kwargs={"enable_thinking": false}
# ($THINK_OFF_CTK below), hipFire's own documented mechanism for this exact
# arch (docs/SERVE.md line ~229 "Qwen3.8 -- disable thinking natively (empty
# closed think block)"; docs/CONFIG.md's family table: "Qwen3.6
# (non-effort-native template) | same on/off"). With it, the model never
# opens a think span at all, so no --max-tokens value can end inside one --
# this is *why* thinking-off is the fix, not merely a workaround for it, and
# it is also why B's own max_tokens can stay small in --smoke (16) rather
# than needing H0d2's 900-token margin.
#
# The SAME switch is NOT applied to A: it does not exist there. glm53's own
# renderer (c/openai_server.py, render_chat_glm53) is explicit that GLM-5.3's
# chat template has no thinking-off form -- the generation prompt always
# opens a bare <think>, and enable_thinking=False (already what EngineDriver
# has sent on every A request since before this item, ttft_serve.py's
# EngineDriver.render) does not turn reasoning off, it selects "Reasoning
# Effort: Low", the template's minimum -- reasoning still happens; only a
# fabricated closed-think form (once tried, #1278/#1282, measured false and
# withdrawn) could hide it, and it does not exist in what the model was
# trained on. This is a DELIBERATE, DOCUMENTED ASYMMETRY, not a gap: A always
# reasons (at the template's minimum effort, unavoidably), B never does. Both
# arms still count "first delta of any kind" -- content, reasoning_content,
# or tool_calls -- as TTFT (ttft_serve.HttpDriver already does; EngineDriver
# reads the engine's raw token stream, which starts with A's own <think>
# tokens), because that is what the owner's screen shows streaming, on
# either side of the asymmetry.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
H2_ROOT=$(cd "$HERE/../.." && pwd)
PRISTINE=~/src/colibri                       # binary in service; H1's own convention
GLM_BIN="$PRISTINE/c/glm53"
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
C_MODEL=${C_MODEL:-/home/ronald/models/qwen36_i4_gs64}
# FRANKEN-H0d: system ROCm 6.2.0 lacks rocm-device-libs; hipFire is built and
# run here against a user-local ROCm 10.0.0 venv instead (TheRock stable pip
# wheels, ~/venvs/rocm) -- a different binary, a different root, not a system
# change. ROCM_ROOT is resolved at run time (the venv's own tool), not
# hardcoded, in case the venv path changes.
HIPFIRE_BIN=~/src/hipfire/target-rocm7/release/hipfire
HIPFIRE_MODEL="qwen3.6:35b-a3b-mq4r"
HIPFIRE_MODEL_DIR=~/.hipfire/models
HIPFIRE_PORT=${HIPFIRE_PORT:-11436}
HIPFIRE_DEV=${HIPFIRE_DEV:-1}                 # dev3's HIP/physical index, §FRANKEN-H0
THINK_OFF_CTK='{"enable_thinking": false}'    # B only -- see the header note
C_PORT=${C_PORT:-8600}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
OUT=~/bench/ctx_ladder_out; mkdir -p "$OUT"
TAG=fk$(date +%m%d%H%M)

SMOKE=0
SKIP_B=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --skip-b) SKIP_B=1 ;;
    *) echo "franken_chain: unknown arg $a"; exit 2 ;;
  esac
done

if [ "$SMOKE" = 1 ]; then
  A_STEPS="256,256"; A_GEN=16; A_FOLLOWUPS=1
  # 2-size cold sweep even in --smoke (coordinator amendment): this is B's
  # first real execution and the sweep is part of what --smoke has to prove,
  # not only the ladder. Small sizes -- hipFire's own numbers (~248 ms ttft,
  # 148-164 tok/s at empty context) make even 2048 cheap.
  B_STEPS="256,256"; B_GEN=16; B_FOLLOWUPS=1; DO_COLD_SWEEP=1; COLD_SWEEP_SIZES="1024,2048"
  C_STEPS="256";     C_GEN=16; C_FOLLOWUPS=1; C_TIMEOUT=1200
else
  A_STEPS="1024,1024,2048,4096,8192"; A_GEN=128; A_FOLLOWUPS=2
  B_STEPS="$A_STEPS,8192,8192,8192,8192,8192,8192"; B_GEN=128; B_FOLLOWUPS=2
  DO_COLD_SWEEP=1; COLD_SWEEP_SIZES="2048,4096,8192,16384"
  C_STEPS="512,1536"; C_GEN=128; C_FOLLOWUPS=0; C_TIMEOUT=1800
fi

echo "=== franken_chain $TAG $(date -Is) smoke=$SMOKE skip_b=$SKIP_B"
echo "=== A_STEPS=$A_STEPS gen=$A_GEN followups=$A_FOLLOWUPS"
echo "=== B_STEPS=$B_STEPS gen=$B_GEN followups=$B_FOLLOWUPS cold_sweep=$COLD_SWEEP_SIZES"
echo "=== C_STEPS=$C_STEPS gen=$C_GEN followups=$C_FOLLOWUPS timeout=${C_TIMEOUT}s"
echo "=== THE GATEWAY IS DOWN FOR THE WHOLE CHAIN"

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
DPMLVL() { cat "/sys/class/drm/card$1/device/power_dpm_force_performance_level" 2>/dev/null; }

precheck() {   # precheck <label>
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  for e in glm53 qwen38 qwen38-vk qwen36; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  if pgrep -f "release/hip[f]ire" >/dev/null; then echo "[$label] hipfire: RUNNING"; else echo "[$label] hipfire: none"; fi
  for c in 0 1 2; do
    echo "[$label] card$c vram_used=$(VRAM "$c") dpm=$(DPMLVL "$c")"
  done
}

assert_vram_free() {   # assert_vram_free <label>
  local label=$1 v c bad=0
  for c in 0 1 2; do
    v=$(VRAM "$c")
    if [ "$v" -lt 0 ] || [ "$v" -ge 1073741824 ]; then
      echo "FATAL [$label]: card$c VRAM $v >= 1 GiB (or unreadable)"; bad=1
    fi
  done
  return $bad
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

assert_glm_resident() {   # assert_glm_resident <label> -- preflight only;
  # context_ladder.py re-asserts (with re-warm retries) on every turn.
  local label=$1 pct
  pct=$(fincore --bytes --output SIZE,RES "$GLM_SNAP"/*.safetensors 2>/dev/null \
        | tail -n +2 | awk '{ts+=$1; rs+=$2} END{if (ts>0) printf "%.4f", 100*rs/ts; else print 0}')
  echo "[resid $label] resident=${pct}%"
  awk -v p="$pct" 'BEGIN{exit !(p>=90)}'
}

# ---- A arm (glm53, engine mode) --------------------------------------
run_A() {   # run_A <arm-label>
  # arm must be its own `local` statement: bash expands every word of a
  # single `local a=1 b=$a` line before any assignment takes effect, so a
  # same-line `${arm}` reads the OUTER (unset) arm, not this one -- found live
  # on the rig (`set -u`: "arm: unbound variable") the first smoke run.
  local arm=$1
  local json="$OUT/${TAG}_${arm}.jsonl" elog="$OUT/${TAG}_${arm}_engine.log" \
        console="$OUT/${TAG}_${arm}_console.log" rc
  echo "=== arm $arm (GLM, engine mode) $(date -Is)"
  precheck "$arm-pre"
  warm_glm
  assert_glm_resident "$arm-pre" || { echo "FATAL: $arm residency < 90% before engine start"; return 1; }

  export GLM53_MAXT=32768
  export GLM53_PREFIX_CKPT=0
  export COLI_CKPT_DIR="$OUT/ckpt_${TAG}_${arm}"; mkdir -p "$COLI_CKPT_DIR"
  export GLM53_VERBOSE=1
  # COLI_TIMERS=1: the [OPTIME] kda/mla/ffn split, printed by glm53's own
  # destructor at engine exit into $elog -- X3 step 0's "which bucket grows
  # from 2.5k to 18k" question reads this log (plan §2.2 H2 row).
  export COLI_TIMERS=1
  export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
  export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
  export COLI_VK_SHADERS="$PRISTINE/c/shaders"

  python3 "$HERE/context_ladder.py" \
      --engine "$GLM_BIN" \
      --steps "$A_STEPS" --gen "$A_GEN" --followups "$A_FOLLOWUPS" \
      --kv-slots 4 --warm --min-resident 90 \
      --arm "$arm" --tag "$TAG" --json "$json" \
      --engine-log "$elog" 2>&1 | tee -a "$console"
  rc=${PIPESTATUS[0]}
  echo "=== arm $arm exit=$rc json=$json engine_log=$elog"
  unset GLM53_MAXT GLM53_PREFIX_CKPT COLI_CKPT_DIR COLI_TIMERS
  # A still-alive glm53 here would ETXTBSY or hijack the next arm's spawn --
  # fold its failure into the return code rather than discarding it.
  wait_no_proc glm53 || rc=1
  return $rc
}

# Reference per-turn majflt from the recorded baseline ladder (tag
# ctx09152003), turns 1-4. commit c790851: a loaded glm53 is ~89 GB of anon
# memory beside a 183 GiB model on a 247 GiB box, so residency tops out near
# 92% with the engine up and turn 1 always major-faults reading its own
# working set in -- majflt>0 is this box's own floor, not a defect, and
# cannot be the thing that invalidates an arm (it always fires). Turns
# beyond 4 compare against turn 4's reference value.
REF_MAJFLT_TAG="ctx09152003"
REF_MAJFLT=(21663 6611 4521 4176)

check_a_arm_valid() {   # check_a_arm_valid <arm> <json> <console>
  local arm=$1
  local json=$2 console=$3 bad_resid py_out majflt_report flagged
  [ -f "$json" ] || { echo "SUMMARY: arm $arm INVALID -- no jsonl produced"; return 1; }

  # (1) The ONLY invalidating condition: residency at arm start (this
  # label's own "$arm-pre" pre-check, printed by assert_glm_resident above,
  # bash-level) or any turn's residency check settling below 90%. The
  # python-side "[resid <label>] files=N resident=X% short=S" summary line
  # is the only one carrying a literal "resident=" substring (the
  # "re-warm N -> X%" transient probe line does not), so this greps exactly
  # the settled/gating values, not every intermediate re-warm attempt; and
  # since context_ladder.py itself sys.exit()s (aborting the whole run) on
  # a settled value under the floor, a completed run reaching this function
  # should never actually trip it -- this re-checks the same floor rather
  # than re-deriving a new one, per CLAUDE.md "do not re-derive".
  bad_resid=$(grep -oE 'resident=[0-9.]+' "$console" 2>/dev/null | cut -d= -f2 \
              | awk '{if ($1+0<90) print}')

  # (2) majflt is reported per turn, never invalidating: flagged
  # MAJFLT-HIGH when it exceeds 2x the reference ladder's same-turn value
  # (REF_MAJFLT/REF_MAJFLT_TAG above), so H3's table can annotate a turn
  # rather than the whole arm being dropped.
  py_out=$(python3 -c "
import json
ref = [21663, 6611, 4521, 4176]
rows = []
for line in open('$json'):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    if r.get('kind') in ('ladder', 'followup'):
        rows.append((r.get('turn'), r.get('majflt') or 0))
parts, flagged = [], []
for turn, mf in rows:
    idx = min(max((turn or 1) - 1, 0), len(ref) - 1)
    hi = mf > 2 * ref[idx]
    parts.append(f'{turn}={mf}' + ('(MAJFLT-HIGH)' if hi else ''))
    if hi:
        flagged.append(str(turn))
print(' '.join(parts))
print(','.join(flagged))
")
  majflt_report=$(printf '%s\n' "$py_out" | sed -n '1p')
  flagged=$(printf '%s\n' "$py_out" | sed -n '2p')

  if [ -n "$bad_resid" ]; then
    echo "SUMMARY: arm $arm INVALID -- residency below 90% seen: $bad_resid"; return 1
  fi
  if [ -n "$flagged" ]; then
    echo "SUMMARY: arm $arm valid (residency>=90% throughout); majflt per turn: $majflt_report; flagged turns: $flagged (>2x $REF_MAJFLT_TAG's same-turn reference [${REF_MAJFLT[*]}], box floor per c790851 -- not invalid)"
  else
    echo "SUMMARY: arm $arm valid (residency>=90% throughout); majflt per turn: $majflt_report; flagged turns: none (all <=2x $REF_MAJFLT_TAG's same-turn reference [${REF_MAJFLT[*]}])"
  fi
  return 0
}

# ---- B arm (hipFire, HTTP mode) -- FRANKEN-H0d/H0d2 verified it serves ----
HIPFIRE_PID=""
stop_hipfire() {
  [ -n "$HIPFIRE_PID" ] && kill -0 "$HIPFIRE_PID" 2>/dev/null && {
    kill "$HIPFIRE_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$HIPFIRE_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$HIPFIRE_PID" 2>/dev/null || true
  }
  pkill -f "release/hip[f]ire serve" 2>/dev/null || true
  HIPFIRE_PID=""
}

run_B() {   # run_B <arm-label>
  local arm=$1   # own statement -- see run_A's comment on this exact bug
  local json="$OUT/${TAG}_${arm}.jsonl" hlog="$OUT/${TAG}_${arm}_hipfire.log" rc
  echo "=== arm $arm (hipFire, HTTP mode) $(date -Is)"
  precheck "$arm-pre"
  assert_vram_free "$arm-pre" || return 1
  [ -x "$HIPFIRE_BIN" ] || { echo "FATAL: $HIPFIRE_BIN missing"; return 1; }

  # Exact env from h0d2_chain.sh (FRANKEN-H0d/H0d2, the branch this box's
  # first working hipFire serve came from) -- venv ROCm 10.0.0's own root,
  # not the system 6.2.0 this arm used to point at; CPLUS_INCLUDE_PATH so
  # hipFire's child clang++ finds GCC 15's headers instead of autodetecting
  # the headerless GCC 16; LD_LIBRARY_PATH at the venv root's lib dir, NO
  # libxml2 shim (that was a 6.2-only ld.lld dependency, absent here).
  local rocm_root; rocm_root=$(~/venvs/rocm/bin/rocm-sdk path --root)
  rm -f ~/.hipfire_kernels/gfx1100/*.tmp 2>/dev/null || true
  ROCM_PATH="$rocm_root" HIP_PATH="$rocm_root" HIPFIRE_DEVICES=$HIPFIRE_DEV \
    CPLUS_INCLUDE_PATH=/usr/include/c++/15:/usr/include/x86_64-linux-gnu/c++/15 \
    LD_LIBRARY_PATH="$rocm_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$HIPFIRE_BIN" serve "$HIPFIRE_MODEL" 0.0.0.0:"$HIPFIRE_PORT" \
    >> "$hlog" 2>&1 < /dev/null &
  HIPFIRE_PID=$!
  echo "hipfire serve pid=$HIPFIRE_PID port=$HIPFIRE_PORT dev=$HIPFIRE_DEV rocm_root=$rocm_root"

  local up=0
  for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$HIPFIRE_PORT/v1/models" 2>/dev/null)
    [ "$code" = 200 ] && { up=1; break; }
    kill -0 "$HIPFIRE_PID" 2>/dev/null || { echo "FATAL: hipfire exited before answering, see $hlog"; break; }
    sleep 5
  done
  if [ "$up" != 1 ]; then echo "$arm: FAILED -- hipfire never answered /v1/models"; stop_hipfire; return 1; fi

  # enable_thinking=false on every B request, throwaway included: hipFire's
  # own off switch (see the header note), and what keeps a small --max-tokens
  # (16 here) from ending inside an open <think> span.
  echo "throwaway request (D_B(~0) preview, not scored; thinking off):"
  curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":16,\"chat_template_kwargs\":${THINK_OFF_CTK},\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}" \
    >> "$OUT/${TAG}_${arm}_throwaway.json" 2>&1

  local cs_args=()
  if [ "$DO_COLD_SWEEP" = 1 ]; then
    cs_args=(--cold-sweep "$COLD_SWEEP_SIZES" --sweep-offset-chars 300000)
  fi
  python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$HIPFIRE_PORT" --model-id "$HIPFIRE_MODEL" \
      --steps "$B_STEPS" --gen "$B_GEN" --followups "$B_FOLLOWUPS" \
      "${cs_args[@]}" \
      --chat-template-kwargs "$THINK_OFF_CTK" \
      --snap "$HIPFIRE_MODEL_DIR" --min-resident 90 --warm \
      --arm "$arm" --tag "$TAG" --json "$json" \
      --server-log "$hlog"
  rc=$?
  echo "=== arm $arm exit=$rc json=$json"
  stop_hipfire
  assert_vram_free "$arm-post"
  return $rc
}

# ---- C arm (qwen36 via `coli serve`, HTTP mode) -----------------------
C_PID=""
stop_c() {
  [ -n "$C_PID" ] && kill -0 "$C_PID" 2>/dev/null && {
    kill "$C_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$C_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$C_PID" 2>/dev/null || true
  }
  pkill -9 -x qwen36 2>/dev/null || true
  C_PID=""
  wait_no_proc qwen36
}

run_C() {
  local json="$OUT/${TAG}_C.jsonl" clog="$OUT/${TAG}_C_serve.log" rc
  echo "=== arm C (qwen36 via 'coli serve', HTTP mode) $(date -Is)"
  precheck "C-pre"
  pgrep -x qwen36 >/dev/null && { echo "FATAL: qwen36 already running"; return 1; }

  echo "--- building qwen36 (CPU-only; no-op if already built)"
  make -C "$H2_ROOT/c" qwen36 >> "$OUT/${TAG}_C_build.log" 2>&1
  [ -x "$H2_ROOT/c/qwen36" ] || { echo "FATAL: c/qwen36 did not build, see $OUT/${TAG}_C_build.log"; return 1; }

  echo "--- warming qwen36 container ($C_MODEL)"
  find "$C_MODEL" -type f -print0 2>/dev/null | xargs -0 cat > /dev/null 2>&1 || true

  COLI_MODEL="$C_MODEL" OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    setsid "$H2_ROOT/c/coli" serve --host 127.0.0.1 --port "$C_PORT" --cap 256 \
    >> "$clog" 2>&1 < /dev/null &
  C_PID=$!
  echo "coli serve (qwen36) pid=$C_PID port=$C_PORT"

  local up=0
  for _ in $(seq 1 120); do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$C_PORT/v1/models" 2>/dev/null)
    [ "$code" = 200 ] && { up=1; break; }
    kill -0 "$C_PID" 2>/dev/null || { echo "FATAL: qwen36 exited before answering, see $clog"; break; }
    sleep 5
  done
  if [ "$up" != 1 ]; then echo "C: FAILED -- coli serve never answered /v1/models"; stop_c; return 1; fi

  timeout "$C_TIMEOUT" python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$C_PORT" \
      --steps "$C_STEPS" --gen "$C_GEN" --followups "$C_FOLLOWUPS" \
      --snap "$C_MODEL" --min-resident 90 --warm \
      --arm C --tag "$TAG" --json "$json" \
      --server-log "$clog"
  rc=$?
  echo "=== arm C exit=$rc json=$json (30-min cap: $C_TIMEOUT s; timeout's own rc is 124)"
  stop_c
  if [ -f "$json" ]; then
    echo "--- arm C rows (tok/s, ttft_s):"
    python3 -c "
import json
for line in open('$json'):
    line=line.strip()
    if not line: continue
    r=json.loads(line)
    print(f\"  turn={r.get('turn')} kind={r.get('kind')} prompt_tokens={r.get('prompt_tokens')} \"
          f\"decode_tps={r.get('decode_tps')} ttft_s={r.get('ttft_s')}\")
"
  fi
  return $rc
}

# --------------------------------------------------------------- main --
on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  stop_hipfire
  stop_c
  pgrep -x glm53 >/dev/null 2>&1 && { pkill -9 -x glm53 2>/dev/null || true; wait_no_proc glm53; }
  echo "--- final re-warm of GLM before restart"
  warm_glm
  assert_glm_resident "final" || echo "WARNING: GLM not >=90% resident at restart time"
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== franken_chain exit rc=$rc tag=$TAG $(date -Is)"
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
stop_gateway || exit 1

# ---- A1 ----------------------------------------------------------------
run_A A1 || exit 1
check_a_arm_valid A1 "$OUT/${TAG}_A1.jsonl" "$OUT/${TAG}_A1_console.log"
precheck "post-A1"
assert_vram_free "post-A1" || exit 1

# ---- B1, B2 (or skipped) ------------------------------------------------
if [ "$SKIP_B" = 1 ]; then
  echo "B SKIPPED: --skip-b was passed. hipFire itself is not blocked on this box "\
"any more (FRANKEN-H0d, branch perf/franken-h0, record section FRANKEN-H0d: the system "\
"ROCm 6.2.0 install lacked rocm-device-libs; a user-local ROCm 10.0.0 venv "\
"rebuilds and serves it clean). This flag is now only an operator escape hatch "\
"for a future blocker on this arm specifically. No hipfire process started, "\
"no GPU touched by this arm."
else
  run_B B1
  precheck "post-B1"
  assert_vram_free "post-B1" || exit 1
  run_B B2
  precheck "post-B2"
  assert_vram_free "post-B2" || exit 1
fi

# ---- A2 ------------------------------------------------------------------
warm_glm
assert_glm_resident "pre-A2" || { echo "FATAL: GLM not >=90% resident before A2"; exit 1; }
run_A A2 || exit 1
check_a_arm_valid A2 "$OUT/${TAG}_A2.jsonl" "$OUT/${TAG}_A2_console.log"

# ---- C ---------------------------------------------------------------
run_C
c_rc=$?
[ "$c_rc" != 0 ] && echo "WARNING: arm C exited rc=$c_rc (see $OUT/${TAG}_C_serve.log)"

# ---- comparison ----------------------------------------------------------
echo "--- context_compare.py over the ladder arms produced this run"
rows=(A1="$OUT/${TAG}_A1.jsonl")
[ "$SKIP_B" = 1 ] || rows+=(B1="$OUT/${TAG}_B1.jsonl" B2="$OUT/${TAG}_B2.jsonl")
rows+=(A2="$OUT/${TAG}_A2.jsonl")
have_rows=()
for spec in "${rows[@]}"; do
  path=${spec#*=}
  [ -f "$path" ] && have_rows+=("$spec")
done
if [ "${#have_rows[@]}" -ge 1 ]; then
  python3 "$HERE/context_compare.py" --rows "${have_rows[@]}"
  compare_rc=$?
  echo "--- context_compare.py exit=$compare_rc (REFUSED/NO VERDICT is an accepted outcome on smoke data)"
else
  echo "context_compare.py: SKIPPED -- no ladder jsonl files were produced"
fi

echo "=== franken_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
