#!/bin/bash
# f11_quality_chain.sh -- Franken plan item F11's open question: "nobody has measured ANSWER
# QUALITY of these quantizations." Runs quality_eval.py (MMLU-Pro + a needle-in-haystack test,
# see that file's own docstring) against all three models serve_alt.sh can put behind port
# 8081, one at a time, and prints a paired comparison at the end.
#
# ORDER: qwen38 (already serving -- do NOT restart it), deepseek, glm. This is NOT the
# A,B,B,A interleave CLAUDE.md asks for numerics gates: this is not a numerics oracle, each
# model answers its OWN frozen question set once, and quality_eval.py's own McNemar test
# (not a warm-cache-confounded latency reading) is what judges the difference -- CLAUDE.md's
# interleave rule exists because a warm page cache changes a SPEED reading; it has no
# analogous effect on whether a model gets a multiple-choice question right.
#
# WHY THIS DOES NOT GO THROUGH run_chain.sh (read before changing that). run_chain.sh's first
# line is `rig_lock_take "$NAME"`, which REFUSES outright if the lock is held by anyone else --
# it has no notion of serve_alt.sh's own lock scheme (serve_alt-<model>, rebound to a `docker
# wait` keeper pid that outlives the setup script, CLAUDE.md/serve_alt.sh's own header). Right
# now, and in general, this chain's very first precondition is "qwen38 may already be serving",
# i.e. the lock may already be legitimately held by `serve_alt-qwen38` -- run_chain.sh would
# refuse to even start. run_chain.sh also does not know about the swap: at its own exit it is
# satisfied by ANY 200 from /v1/models, which is true while an alt model serves, so it would
# not notice or fix a chain that left something other than GLM in service. For those two
# reasons this chain manages the lock itself, matching serve_alt.sh's own scheme exactly:
#   - during the qwen38 and deepseek phases, serve_alt.sh's OWN lock (a keeper pid tied to the
#     alt server's container lifetime) is the only lock in play; this chain neither takes nor
#     releases it, only calls `serve_alt.sh <model>` and `serve_alt.sh status` as ordinary
#     subprocesses, exactly as the owner would run them by hand.
#   - during the glm phase, nothing holds the lock any more (serve_alt.sh's `restore_glm`
#     releases it on PASS -- serve_alt.sh's own header, point 2 near the end), and the GLM
#     phase sends dozens of long chat completions to the OWNER'S DAILY GATEWAY over what the
#     wall-time estimate below shows can be hours; this chain takes the rig lock itself
#     (rig_lock.sh, sourced directly, `rig_lock_take` bound to THIS script's own $$, which
#     lives for the whole synchronous eval call) so a second session's chain or a stray
#     restart cannot land on top of it, and releases it right after that eval call returns,
#     before summarize (which touches nothing on the rig).
# Launch directly, detached (NOT through run_chain.sh, per the above):
#   ssh -n -f rome 'setsid nohup ~/src/colibri/tools/hot-expert/f11_quality_chain.sh \
#       > ~/bench/f11_quality/chain.log 2>&1 < /dev/null &'
#
# EVERY EXIT PATH LEAVES GLM SERVING AND VERIFIED (requirement: "a trap that, if an
# alternative is still serving or nothing is, runs serve_alt.sh glm"). on_exit below: release
# our own rig lock if we are holding it (so serve_alt.sh's own lock-take, which only
# recognises "serve_alt*"-prefixed holders as safe to take over, is not refused by a foreign
# name); read `serve_alt.sh status`; if it does not say GLM is serving, run `serve_alt.sh glm`
# (which itself warms GLM, restarts the gateway, runs accept_live.sh, and reverts to leaving
# the lock HELD -- not released -- on failure, exactly as its own header documents, so a
# stuck state is loud, not silently papered over); re-check status once more and fail loudly
# (nonzero exit) if GLM is still not confirmed serving.
#
# DEEPSEEK FAILURE HANDLING (requirement: "if serve_alt.sh deepseek fails, serve_alt itself
# falls back to GLM: detect that (status) and continue with the GLM phase instead of
# aborting"). `cmd_alt`'s every failure branch in serve_alt.sh calls `restore_glm` before
# `exit 1` (see its ladder of `restore_glm "..."; exit 1` calls). So: if `serve_alt.sh
# deepseek` returns nonzero, `serve_alt.sh status` is read; if it now says GLM is serving,
# the deepseek eval is skipped (no deepseek.jsonl -- summarize below simply has one fewer
# model) and the chain proceeds straight into the GLM phase WITHOUT calling `serve_alt.sh
# glm` again (GLM is already confirmed in service, and restore_glm already ran accept_live
# and released its lock on that PASS -- re-running it would just repeat that work). If status
# does NOT confirm GLM either (restore_glm itself failed, lock left held on purpose per its
# own header), the chain exits nonzero at that point; the exit trap still attempts the same
# recovery (`serve_alt.sh glm`) as a backstop.
#
# ESTIMATED WALL TIME (labelled: this is arithmetic on the F11 record's own measured rates,
# nobody has measured MMLU-Pro/needle costs on this box -- record sections F11-STEP0/-DEPTH/
# -SERVE, ROME-3x7900XTX-2026-09-04.md). Two unknowns dominate and are both guessed
# explicitly: (1) how many tokens a model spends on the CoT before "Answer: X" -- MMLU-Pro's
# own max_tokens budget is 3500; (2) needle-test prefill cost past the depths already measured
# (F11-DEPTH only goes to 257k for Qwen, nothing for DeepSeek/GLM past ~19k) -- extrapolated
# linearly from the nearest two measured points, labelled as extrapolation.
#   decode rates (tok/s): qwen38=15, deepseek=8, glm=5 (record's own headline figures).
#   REALISTIC (avg completion 500 tokens/mmlu-item, 150 tokens/needle-reply; needle prefill
#   ms/token interpolated from F11-DEPTH's cumulative-prefill table for Qwen, from
#   F11-STEP0's single 18-19k point held constant for DeepSeek/GLM):
#     qwen38 mmlu:  70 * 500/15 s                       =  2333 s (38.9 min)
#     qwen38 needle: prefill(30k,60k,120k,200k) approx
#            30k@1.75ms/tok=52s 60k@2.05ms/tok=123s 120k@2.65ms/tok=318s 200k@3.5ms/tok=700s
#            + 4*(150/15 s gen) = 1193s + 40s           =  1234 s (20.6 min)
#     deepseek mmlu: 70 * 500/8 s                       =  4375 s (72.9 min)
#     deepseek needle: prefill@5.1ms/tok (F11-STEP0's 18-19k rate, held constant, NOT
#            measured deeper) * (30k,60k,120k,200k) = 2091s + 4*(150/8 s)=75s
#                                                     =  2166 s (36.1 min)
#     deepseek swap overhead (serve_alt.sh, 90.9 GB warm + fit load)  ~=   900 s (15 min)
#     glm mmlu: 70 * 500/5 s                            =  7000 s (116.7 min)
#     glm needle (only 30k,60k allowed under --max-context 60000):
#            prefill@19.3ms/tok (F11-STEP0's 18k rate, held constant) * (30k,60k) = 1737s
#            + 2*(150/5 s gen)=60s                      =  1797 s (30.0 min)
#     glm swap overhead (serve_alt.sh glm: 182 GiB two-pass warm + restart + accept_live,
#            CLAUDE.md/f11_ladder_chain.sh's own ~10 min figure + ~2 min accept_live)
#                                                        =   720 s (12 min)
#   REALISTIC TOTAL = (2333+1234) + (4375+2166+900) + (7000+1797+720) + fetch(~180s)
#                    =  3567 + 7441 + 9517 + 180 = 20705 s =~ 5.75 h
#   WORST CASE (every reply hits its max_tokens cap: 3500 mmlu, 1500 needle):
#     qwen38: mmlu 70*3500/15=16333s + needle (prefill 1193s + 4*1500/15=400s)=1593s
#             = 17926 s (4.98 h)
#     deepseek: mmlu 70*3500/8=30625s + needle (2091s+4*1500/8=750s)=2841s + swap 900s
#             = 34366 s (9.55 h)
#     glm: mmlu 70*3500/5=49000s + needle (1737s+2*1500/5=600s)=2337s + swap 720s
#             = 52057 s (14.46 h)
#   WORST TOTAL = 17926+34366+52057+180 = 104529 s =~ 29.0 h
#   Neither bound has been checked against a real run; the realistic case is the planning
#   number, the worst case is why this chain must be launched detached and polled, not waited
#   on in a foreground shell.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/rig_lock.sh"
QE="$HERE/quality_eval.py"
SERVE_ALT="$HERE/serve_alt.sh"
KEY_FILE="$HOME/.colibri_api_key"
URL="http://127.0.0.1:8081"
MODEL_ID="glm-5.3-flash"     # the gateway's own alias -- what every backend answers to, see
                              # serve_alt.sh's header and quality_eval.py's identity-check note
OUT_DIR="$HOME/bench/f11_quality"
mkdir -p "$OUT_DIR"
MMLU_SAMPLE="$OUT_DIR/mmlu_pro_sample.json"
Q_JSONL="$OUT_DIR/qwen38.jsonl"
D_JSONL="$OUT_DIR/deepseek.jsonl"
G_JSONL="$OUT_DIR/glm.jsonl"

echo "=== f11_quality_chain $(date -Is)"

# ------------------------------------------------------------------------------- primitives --
status_says() {   # status_says <substring> -- true if `serve_alt.sh status` output contains it
  "$SERVE_ALT" status 2>&1 | grep -qi -- "$1"
}

run_eval() {   # run_eval <expect> <out_jsonl> <max_context> <parts>
  local expect=$1 out=$2 max_context=$3 parts=${4:-mmlu,needle}
  echo "--- quality_eval.py run --expect $expect --out $out --max-context $max_context --parts $parts $(date -Is)"
  python3 "$QE" run --expect "$expect" --url "$URL" --key-file "$KEY_FILE" \
      --model-id "$MODEL_ID" --out "$out" --mmlu "$MMLU_SAMPLE" \
      --max-context "$max_context" --parts "$parts"
  local rc=$?
  echo "--- quality_eval.py run --expect $expect exit=$rc $(date -Is)"
  return $rc
}

# ------------------------------------------------------------------------------------ on_exit --
GLM_LOCK_HELD_BY_US=0
on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  echo "=== f11_quality_chain exit trap rc=$rc $(date -Is)"
  if [ "$GLM_LOCK_HELD_BY_US" = 1 ]; then
    rig_lock_release
    GLM_LOCK_HELD_BY_US=0
  fi
  local status
  status=$("$SERVE_ALT" status 2>&1)
  echo "$status"
  if echo "$status" | grep -qi "serving: GLM-5.3"; then
    echo "on_exit: GLM already confirmed serving -- leaving it"
  else
    echo "on_exit: an alternative is still serving (or nothing recognizable is) -- running 'serve_alt.sh glm'"
    "$SERVE_ALT" glm
  fi
  status=$("$SERVE_ALT" status 2>&1)
  echo "$status"
  if echo "$status" | grep -qi "serving: GLM-5.3"; then
    echo "on_exit: GLM confirmed serving"
  else
    echo "on_exit: FATAL -- GLM is still NOT confirmed serving after recovery; manual intervention needed"
    [ "$rc" -eq 0 ] && rc=1
  fi
  echo "=== f11_quality_chain done rc=$rc $(date -Is)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

# ------------------------------------------------------------------------------- preflight ----
[ -x "$SERVE_ALT" ] || { echo "FATAL: $SERVE_ALT missing or not executable"; exit 1; }
[ -s "$KEY_FILE" ] || { echo "FATAL: $KEY_FILE missing or empty"; exit 1; }

if [ ! -f "$MMLU_SAMPLE" ]; then
  echo "--- fetching the frozen MMLU-Pro sample (not present yet): $MMLU_SAMPLE"
  python3 "$QE" fetch --out "$MMLU_SAMPLE" || { echo "FATAL: quality_eval.py fetch failed"; exit 1; }
else
  echo "--- MMLU-Pro sample already present: $MMLU_SAMPLE ($(python3 -c "import json;print(len(json.load(open('$MMLU_SAMPLE'))['items']))" 2>/dev/null || echo '?') items)"
fi

echo "=== initial serve_alt status"
"$SERVE_ALT" status

# ---------------------------------------------------------------------------- phase 1: qwen38 --
echo "=== phase 1/3: qwen38"
if status_says "Qwen3.8-Flash-Next"; then
  echo "qwen38 already serving -- NOT restarting it (per brief)"
else
  echo "qwen38 not currently serving -- starting it"
  "$SERVE_ALT" qwen38 || { echo "FATAL: serve_alt.sh qwen38 failed (its own fallback should have restored GLM; on_exit will verify)"; exit 1; }
fi
run_eval qwen38 "$Q_JSONL" 250000 "mmlu,needle"
QWEN_EVAL_RC=$?
[ "$QWEN_EVAL_RC" -ne 0 ] && echo "WARNING: qwen38 eval had incomplete items (rc=$QWEN_EVAL_RC) -- resumable, continuing chain"

# --------------------------------------------------------------------------- phase 2: deepseek --
echo "=== phase 2/3: deepseek"
SKIP_GLM_SWITCH=0
"$SERVE_ALT" deepseek
DS_RC=$?
if [ "$DS_RC" -ne 0 ]; then
  echo "serve_alt.sh deepseek FAILED rc=$DS_RC -- checking whether its own fallback restored GLM"
  if status_says "serving: GLM-5.3"; then
    echo "confirmed via status: GLM is back in service (serve_alt.sh's own fallback) -- skipping the deepseek eval, proceeding to the GLM phase without another 'serve_alt.sh glm' call"
    SKIP_GLM_SWITCH=1
  else
    echo "FATAL: deepseek failed AND status does not confirm GLM either -- letting the exit trap attempt recovery"
    exit 1
  fi
else
  run_eval deepseek "$D_JSONL" 250000 "mmlu,needle"
  DS_EVAL_RC=$?
  [ "$DS_EVAL_RC" -ne 0 ] && echo "WARNING: deepseek eval had incomplete items (rc=$DS_EVAL_RC) -- resumable, continuing chain"
fi

# -------------------------------------------------------------------------------- phase 3: glm --
echo "=== phase 3/3: glm"
if [ "$SKIP_GLM_SWITCH" != 1 ]; then
  "$SERVE_ALT" glm || { echo "FATAL: serve_alt.sh glm failed"; exit 1; }
fi
echo "--- taking the rig lock for the GLM eval phase (nothing else holds it now: serve_alt.sh's own lock is released on a PASS)"
rig_lock_take "f11-quality-glm" || { echo "FATAL: could not take the rig lock for the GLM eval phase"; exit 1; }
GLM_LOCK_HELD_BY_US=1
run_eval glm "$G_JSONL" 60000 "mmlu,needle"
GLM_EVAL_RC=$?
[ "$GLM_EVAL_RC" -ne 0 ] && echo "WARNING: glm eval had incomplete items (rc=$GLM_EVAL_RC) -- resumable, continuing chain"
rig_lock_release
GLM_LOCK_HELD_BY_US=0

# -------------------------------------------------------------------------------- summarize ----
echo "=== summarize"
SUMMARIZE_ARGS=()
[ -f "$Q_JSONL" ] && SUMMARIZE_ARGS+=(--model "qwen38=$Q_JSONL")
[ -f "$D_JSONL" ] && SUMMARIZE_ARGS+=(--model "deepseek=$D_JSONL")
[ -f "$G_JSONL" ] && SUMMARIZE_ARGS+=(--model "glm=$G_JSONL")
if [ "${#SUMMARIZE_ARGS[@]}" -ge 2 ]; then
  python3 "$QE" summarize "${SUMMARIZE_ARGS[@]}"
else
  echo "summarize: fewer than 2 models produced a jsonl -- nothing to compare"
  [ "${#SUMMARIZE_ARGS[@]}" -eq 1 ] && python3 "$QE" summarize "${SUMMARIZE_ARGS[@]}"
fi

echo "=== f11_quality_chain body done $(date -Is) -- exit trap verifies/restores GLM next"
exit 0
