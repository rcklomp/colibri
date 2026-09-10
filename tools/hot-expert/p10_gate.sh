#!/bin/bash
# P10 — the executable gate for "value-based checkpoint eviction".
#
#   p10_gate.sh <pristine-glm53> <candidate-glm53> [tag]
#
# Exit 0 = the item is done and its tables belong in the commit body.
# Exit 1 = an oracle failed (teacher forcing, logits, tworeq) or the flood
#          evicted the tool block, or the control arm failed to reproduce the
#          bug (an item that passes without its control passes for an unknown
#          reason).
# Exit 2 = a harness refusal (residency, another engine running, no gateway, or
#          a flood that stored nothing and therefore tested nothing).
# Exit 3 = a speed check failed (step 1's neutrality).
#
# P10 changes ONE policy in c/glm53.c: which checkpoint slot dies. It cannot
# change what any request computes -- ckpt_restore accepts a slot only after a
# memcmp of the full prefix -- so step 1 runs with GLM53_PREFIX_CKPT=0, where
# none of the new code executes at all, and must come back bit-identical.
#
#  (1) prefill_gate.sh, GLM53_PREFIX_CKPT=0, MIN_SPEEDUP=0.97, private
#      COLI_CKPT_DIR, PROFILE_MIN_RESIDENT=97.
#  (2) tworeq.py at TWOREQ_SLOTS=4, COLI_KDA_GPU=2 and =0: IDENTICAL, and no
#      "forcing COLI_KDA_GPU=0" line at knob 2.
#  (3) THE FLOOD, in TWO arms, both at GLM53_PREFIX_CKPT_MIN=128:
#        on   GLM53_CKPT_VALUE_EVICT=1   (the item)
#        off  GLM53_CKPT_VALUE_EVICT=0   -- today's pure LRU, the control
#      Each arm, from a byte-identical seeded checkpoint directory:
#        a. restart the gateway on a PRIVATE COLI_CKPT_DIR seeded from the live
#           one, so the owner's own checkpoints are never at risk and both arms
#           start from the same four slots;
#        b. drive $P10_WARM UI-shaped chats, so the tool block EARNS the hits
#           the policy turns on (on disk every slot loads at hits=1, which is
#           the length-only rule, and the tool block is not even the longest);
#        c. flood: $P10_FLOOD short API chats, each with a unique ~200-token
#           system block, so the gateway's prefix hint makes each one STORE.
#           A flood that stores nothing is a harness refusal, not a pass;
#        d. one UI-shaped new chat.
#      Expected: `on` still logs `CKPT hit prefix>=4300`, `off` does not.
#  (4) accept_live.sh on the SERVED gateway (the owner's own script, no arm
#      knobs): the four checks plus the ledger's alarm.
#
# GATE_STEPS selects: "12" is the offline half (gateway must be DOWN), "34" the
# live half. Default "1234". The Mac-side half of the gate -- accept_ui.sh and
# ui/ui_matrix.sh --sizes 0,500,2000 with the turn-2 rows unchanged -- cannot
# run here (the rig has no browser and no Node) and is owed separately.
set -u
PRISTINE=${1:?pristine binary}; CAND=${2:?candidate binary}; TAG=${3:-p10gate$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/bench/p10_gate_$TAG
LOG=${GLM53_LOG:-$HOME/glm53_server.log}
STEPS=${GATE_STEPS:-1234}
WARM=${P10_WARM:-3}
FLOOD=${P10_FLOOD:-20}
FLOOD_TOK=${P10_FLOOD_TOK:-200}
MIN_STORES=${P10_MIN_STORES:-4}
TOOLBLOCK_MIN=${P10_TOOLBLOCK_MIN:-4300}
LIVE_CKPT=${P10_LIVE_CKPT:-}
K=$(cat "$HOME/.colibri_api_key")
URL=http://127.0.0.1:8081
mkdir -p "$OUT"
run_step() { case "$STEPS" in *"$1"*) return 0;; *) return 1;; esac; }
say() { printf '  %-42s %s\n' "$1" "$2"; }

wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running after 240 s: $(pgrep -x glm53 | tr '\n' ' ')"
  return 1
}
gw_stop() { "$HOME/bench/p7_stop.sh"; wait_no_engine; }

# Every refusal path below (`exit 2`, `wait_no_engine || exit 2`, a Ctrl-C, a
# dropped ssh) used to return with whatever an ARM last started still serving:
# the owner's daily service left running from $ARM_SCRIPT, with
# GLM53_CKPT_VALUE_EVICT possibly OFF and its checkpoints going to a
# ~/bench/p10_gate_* temp dir. p10_chain.sh does not save you either -- its
# on_exit deliberately skips the revert while an openai_server.py is up, which
# is exactly the case an arm leaves behind. p10_hits_case.sh got this right;
# this gate did not. Restore the owner's own script on every exit path.
GATE_ARM_RUNNING=0
restore_service() {
  rc=$?
  trap - EXIT INT TERM HUP            # never run this handler twice
  if [ "$GATE_ARM_RUNNING" = 1 ]; then
    echo "[p10_gate] an arm gateway is up -- restoring the owner's service"
    gw_stop || true
    gw_start || echo "[p10_gate] WARNING: the owner's gateway did not come back"
    GATE_ARM_RUNNING=0
  fi
  exit "$rc"
}
trap restore_service EXIT INT TERM HUP

# ~/start_glm53.sh EXPORTS GLM53_PREFIX_CKPT_MIN=1024 unconditionally, and the
# whole point of P10 is that the minimum can come back down once eviction can
# rank. An arm therefore runs a copy of the owner's own script with that one
# line made overridable, generated at run time from the original so it cannot
# drift from what is in service. The FINAL restart uses the owner's file
# verbatim. (The pattern, and the reason for it, are p9_gate.sh's.)
ARM_SCRIPT="$OUT/start_glm53_arm.sh"
make_arm_script() {
  sed -e 's/^export GLM53_PREFIX_CKPT_MIN=1024$/export GLM53_PREFIX_CKPT_MIN=${GLM53_PREFIX_CKPT_MIN:-1024}/' \
      "$HOME/start_glm53.sh" > "$ARM_SCRIPT"
  grep -q 'GLM53_PREFIX_CKPT_MIN:-1024' "$ARM_SCRIPT" || {
    echo "the arm script did not take -- start_glm53.sh no longer sets MIN=1024 the same way"; return 1; }
  chmod +x "$ARM_SCRIPT"
}
gw_start() {   # gw_start [VAR=VAL ...] — arm knobs make it use the arm script
  local script="$HOME/start_glm53.sh"
  [ $# -gt 0 ] && { make_arm_script || return 1; script="$ARM_SCRIPT"; }
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_PREFIX_CKPT_MIN \
      -u GLM53_CKPT_VALUE_EVICT -u COLI_MLA_POOL -u COLI_REPLY_PIN \
      -u COLI_LEDGER -u COLI_PREFIX_PIN \
      "$@" SKIP_WARM=1 setsid nohup "$script" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $K" \
         -w '%{http_code}' $URL/v1/models 2>/dev/null)" = 200 ] && { sleep 2; return 0; }
    sleep 5
  done
  echo "gateway did not answer /v1/models within 10 min"; return 1
}

# `grep -c X f || echo 0` prints TWO zeros when there is no match (grep prints 0
# AND exits 1). That parked one chain for an hour on 2026-09-09 and aborted
# another under `set -u`. Count without the fallback, tolerate the status.
count() { local n; n=$(grep -ac "$1" "$LOG" 2>/dev/null) || true; echo "${n:-0}"; }

echo "=== p10_gate $TAG $(date -Is) steps=$STEPS"
echo "    pristine=$PRISTINE candidate=$CAND"
echo "    outputs in $OUT"

rc1=0; rc2=0; rc3=0; rc4=0
ran1=0; ran2=0; ran3=0; ran4=0
verdict_line() { if [ "$2" = 1 ]; then printf "  %-40s rc=%s\n" "$1" "$3"
                 else printf "  %-40s -- (not run)\n" "$1"; fi; }

# ---------------------------------------------------------------- (1)
if run_step 1; then
  echo
  echo "### step 1 — prefill_gate.sh, checkpoints OFF (P10's code must not exist there)"
  if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running -- stop the gateway first"; exit 2; fi
  if [ "$(sha256sum "$PRISTINE" | cut -d" " -f1)" = "$(sha256sum "$CAND" | cut -d" " -f1)" ]; then
    echo "    NOTE: pristine and candidate are THE SAME BYTES -- P10 changes c/glm53.c,"
    echo "          so an identical binary means the build did not take"
  else
    echo "    pristine and candidate are different binaries, as expected"
  fi
  GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt/step1" MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
    PROFILE_MIN_RESIDENT=${PROFILE_MIN_RESIDENT:-97} \
    "$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG-off" 2>&1 | tee "$OUT/step1.txt"
  rc1=${PIPESTATUS[0]}; ran1=1
  wait_no_engine || exit 2
  echo "step 1 rc=$rc1"
fi

# ---------------------------------------------------------------- (2)
# Verbatim p9_gate.sh's step 2, including every environment variable. The first
# run of this gate (2026-09-10 07:20) invoked tworeq.py with none of them: no
# COLI_VULKAN, no COLI_VK_DEV2/3, no COLI_VK_EXPERTS2/3=1695, no shader path, no
# thread pinning, and the binary passed as argv where tworeq.py reads TWOREQ_EXE.
# The KDA device slot pool then held 0, the P6b guard printed
# "forcing COLI_KDA_GPU=0", and the gate failed the CANDIDATE for a defect in
# itself -- after measuring the CPU recurrence instead of the regime every
# recorded number was taken in. CLAUDE.md names both of those (the 1695 caps for
# comparability, the 8-thread pinning); this block does not get to paraphrase it.
if run_step 2; then
  echo
  echo "### step 2 — tworeq.py at 4 slots, COLI_KDA_GPU=2 and =0"
  if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running"; exit 2; fi
  ran2=1
  # A COPY of the histogram, never the canonical file: the engine rewrites
  # COLI_USAGE_PATH at exit and a benchmark must not teach the serving tier.
  cp -f "$HOME/.glm53_explain.bin" "$OUT/hist_tworeq.bin" 2>/dev/null || true
  for knob in 2 0; do
    echo "--- tworeq COLI_KDA_GPU=$knob TWOREQ_SLOTS=4"
    env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
        COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
        COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
        COLI_VK_SHADERS="$HERE/../../c/shaders" \
        COLI_USAGE_PATH="$OUT/hist_tworeq.bin" \
        GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt/step2" \
        COLI_KDA_GPU=$knob TWOREQ_SLOTS=4 TWOREQ_EXE="$CAND" GLM53_VERBOSE=1 \
        python3 "$HERE/tworeq.py" > "$OUT/tworeq_kda$knob.txt" 2>&1
    r=$?
    forced=$(grep -c "forcing COLI_KDA_GPU=0" "$OUT/tworeq_kda$knob.txt") || true
    echo "tworeq COLI_KDA_GPU=$knob: $(grep '^RESULT:' "$OUT/tworeq_kda$knob.txt" || echo 'NO RESULT') (rc=$r)"
    echo "    forcing lines: ${forced:-0}"
    [ "$r" = 0 ] || rc2=1
    grep -q "^RESULT: IDENTICAL" "$OUT/tworeq_kda$knob.txt" || {
      echo "    FAIL: not IDENTICAL at knob $knob"; rc2=1; }
    if [ "$knob" = 2 ] && [ "${forced:-0}" != 0 ]; then
      echo "    FAIL: the guard fired at knob 2 -- this run measured the CPU recurrence"; rc2=1
    fi
    wait_no_engine || exit 2
  done
  echo "step 2 rc=$rc2"
fi

# ---------------------------------------------------------------- (3)
# One arm of the flood. Prints one summary line and returns 0 if the tool block
# survived, 1 if it did not, 2 if the arm did not actually test anything.
flood_arm() {   # flood_arm <label> <value_evict 0|1> <seed-dir>
  local label=$1 evict=$2 seed=$3
  local dir="$OUT/ckpt/arm-$label"
  echo
  echo "--- arm $label: GLM53_CKPT_VALUE_EVICT=$evict GLM53_PREFIX_CKPT_MIN=128"
  rm -rf "$dir"; mkdir -p "$dir"
  if ! cp -p "$seed"/*.bin "$dir"/ 2>/dev/null; then
    echo "    REFUSED: could not seed the arm's checkpoint dir from $seed"; return 2
  fi
  say "seeded from $seed" "$(ls "$dir" | wc -l) files, $(du -sh "$dir" | cut -f1)"
  gw_stop || return 2
  GATE_ARM_RUNNING=1   # from here the service is an ARM, not the owner's -- see restore_service
  gw_start COLI_CKPT_DIR="$dir" GLM53_PREFIX_CKPT_MIN=128 GLM53_CKPT_VALUE_EVICT=$evict || return 2
  echo "    slots as loaded:"
  grep -a "CKPT disk load" "$LOG" | tail -8 | sed 's/^/      /' | cut -c1-140

  # (b) the tool block earns its hits — every UI chat is one vote for it
  local i
  for i in $(seq 1 "$WARM"); do
    "$HERE/owui_ui_turn.sh" "Which day comes after Tuesday? One word. [$TAG-$label-w$i]" \
      > "$OUT/arm-$label-warm$i.txt" 2>&1
    say "warm-up $i/$WARM" "$(grep '^RESULT' "$OUT/arm-$label-warm$i.txt" || echo 'no RESULT')"
  done
  # This count is the whole point of step (b): the arm exists to show that the
  # HITS term defends the tool block, so if the warm-ups never restored it, the
  # on/off verdict that follows is about something else (length alone) and is
  # not evidence for this item. It used to be assigned and never read -- an
  # assertion written down but not made. Fail loudly instead.
  local tb_hits
  tb_hits=$(grep -a "CKPT hit prefix=" "$LOG" | tail -n "$WARM" | grep -c "hits=") || true
  tb_hits=${tb_hits:-0}
  grep -a "CKPT hit prefix=" "$LOG" | tail -3 | sed 's/^/      /' | cut -c1-120
  say "tool block earned hits" "$tb_hits of $WARM warm-ups"
  if [ "$tb_hits" -lt 2 ]; then
    echo "    REFUSED: the tool block was restored $tb_hits time(s) in $WARM warm-ups."
    echo "             Without hits on it there is no hits term to test, and the"
    echo "             arm would be measuring the length-only rule instead."
    return 2
  fi

  # (c) the flood: unique ~FLOOD_TOK-token system blocks, so the gateway's
  #     prefix hint lands on each one and the engine STORES it.
  local mark_store mark_hit stores
  mark_store=$(count "CKPT store")
  for i in $(seq 1 "$FLOOD"); do
    python3 - "$i" "$FLOOD_TOK" "$TAG-$label" > "$OUT/arm-$label-flood.json" <<'PY'
import sys, json, random
i, ntok, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
# Deterministic per (tag, i) and DISTINCT between floods: a shared opening would
# make ckpt_have() suppress every capture after the first, and the flood would
# flood nothing. ~3.6 chars per token on this prose (the record's calibration).
r = random.Random(f"{tag}-{i}")
words = ["ledger","prefix","expert","router","tensor","kernel","window","segment","adapter",
         "residue","channel","budget","harness","canary","rotate","shard","latent","gate"]
filler = " ".join(r.choice(words) for _ in range(int(ntok * 3.6 / 6.5)))
sysmsg = f"Case {tag}-{i}. Reference notes, do not summarise: {filler}"
print(json.dumps({"model": "glm-5.3-flash", "max_tokens": 8, "messages": [
    {"role": "system", "content": sysmsg},
    {"role": "user", "content": f"Reply with the single word OK. [{tag}-{i}]"}]}))
PY
    curl -s -m 600 -o /dev/null -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
      $URL/v1/chat/completions -d @"$OUT/arm-$label-flood.json"
    [ $((i % 5)) = 0 ] && say "flood $i/$FLOOD" "CKPT store +$(( $(count "CKPT store") - mark_store ))"
  done
  stores=$(( $(count "CKPT store") - mark_store ))
  say "flood done" "$FLOOD short chats, CKPT store +$stores"
  grep -a "CKPT store" "$LOG" | tail -6 | sed 's/^/      /' | cut -c1-150
  if [ "$stores" -lt "$MIN_STORES" ]; then
    echo "    REFUSED: the flood stored $stores checkpoints (< $MIN_STORES) -- it tested nothing."
    echo "             Raise P10_FLOOD_TOK, or the prefix hint is not landing."
    return 2
  fi

  # (d) the request the flood was aimed at
  mark_hit=$(grep -ac "CKPT hit prefix=" "$LOG" 2>/dev/null) || true; mark_hit=${mark_hit:-0}
  "$HERE/owui_ui_turn.sh" "Name one prime number greater than ten. One word. [$TAG-$label-after]" \
    > "$OUT/arm-$label-after.txt" 2>&1
  local res big
  res=$(grep '^RESULT' "$OUT/arm-$label-after.txt" || echo "RESULT error")
  big=$(grep -a "CKPT hit prefix=" "$LOG" | tail -n +$((mark_hit + 1)) \
        | sed -n 's/.*CKPT hit prefix=\([0-9]*\).*/\1/p' | awk -v m="$TOOLBLOCK_MIN" '$1>=m' | tail -1)
  say "UI chat after the flood" "$res"
  say "tool-block restore" "${big:-none} (want >= $TOOLBLOCK_MIN)"
  ARM_RESULT="stores=$stores hit=${big:-none} ${res#RESULT }"
  [ -n "$big" ] && return 0 || return 1
}

if run_step 3; then
  echo
  echo "### step 3 — the flood: 20 short chats must not evict the tool block"
  # $CAND is used by steps 1 and 2 only. Steps 3 and 4 drive a GATEWAY, and the
  # gateway serves whatever is at ~/src/colibri/c/glm53 -- not $CAND. Run this
  # gate standalone (GATE_STEPS=34, which its own header invites) against a tree
  # that has not been rebuilt, and it will flood-test the PRISTINE binary and
  # report PASS for the candidate. The chain happens to install first; a gate
  # whose exit 0 means "the item is done" must not depend on its caller.
  gw_bin="$HOME/src/colibri/c/glm53"
  if [ "$(sha256sum "$gw_bin" | cut -d" " -f1)" != "$(sha256sum "$CAND" | cut -d" " -f1)" ]; then
    echo "REFUSED: the gateway would serve $(sha256sum "$gw_bin" | cut -c1-16), but the"
    echo "         candidate under test is $(sha256sum "$CAND" | cut -c1-16). Steps 3-4 drive"
    echo "         the gateway, so they would measure the wrong binary. Install the"
    echo "         candidate first (the chain does this) or pass the served binary."
    exit 2
  fi
  echo "    gateway will serve the candidate: $(sha256sum "$gw_bin" | cut -c1-16) ✓"
  if [ -z "$LIVE_CKPT" ]; then
    LIVE_CKPT=$(ls -d "$HOME"/models/*/.coli_ckpt 2>/dev/null | head -1)
  fi
  if [ -z "$LIVE_CKPT" ] || [ ! -d "$LIVE_CKPT" ]; then
    echo "REFUSED: no live checkpoint directory found (set P10_LIVE_CKPT)"; exit 2
  fi
  if ! ls "$LIVE_CKPT"/*.bin >/dev/null 2>&1; then
    echo "REFUSED: $LIVE_CKPT holds no checkpoint -- there is no tool block to defend"; exit 2
  fi
  # A frozen copy: both arms start from the SAME four slots, and the owner's own
  # checkpoints are never the thing being destroyed by the control arm.
  SEED="$OUT/ckpt/seed"; rm -rf "$SEED"; mkdir -p "$SEED"
  cp -p "$LIVE_CKPT"/*.bin "$SEED"/ || { echo "REFUSED: could not snapshot $LIVE_CKPT"; exit 2; }
  echo "    seed: $SEED ($(ls "$SEED" | wc -l) files, $(du -sh "$SEED" | cut -f1)) from $LIVE_CKPT"

  ARM_RESULT=""
  flood_arm on 1 "$SEED"; on_rc=$?; on_res=$ARM_RESULT
  ARM_RESULT=""
  flood_arm off 0 "$SEED"; off_rc=$?; off_res=$ARM_RESULT
  ran3=1
  echo
  echo "  | arm | GLM53_CKPT_VALUE_EVICT | after the flood |"
  echo "  |---|---|---|"
  echo "  | on  | 1 | $on_res |"
  echo "  | off | 0 | $off_res |"
  if [ "$on_rc" = 2 ] || [ "$off_rc" = 2 ]; then
    echo "  REFUSED: an arm did not test anything (rc on=$on_rc off=$off_rc)"; exit 2
  fi
  if [ "$on_rc" != 0 ]; then
    echo "  FAIL: the flood evicted the tool block WITH value eviction on -- the item does not work"
    rc3=1
  elif [ "$off_rc" = 0 ]; then
    echo "  FAIL: the control arm ALSO kept the tool block, so this flood does not reproduce the"
    echo "        bug and the 'on' arm passed for an unknown reason. Raise P10_FLOOD."
    rc3=1
  else
    echo "  PASS: value eviction kept the tool block ($on_res); pure LRU lost it ($off_res)"
  fi
  echo "step 3 rc=$rc3"
fi

# ---------------------------------------------------------------- (4)
if run_step 4; then
  echo
  echo "### step 4 — accept_live.sh on the served gateway (the owner's own script)"
  gw_stop || exit 2
  gw_start || exit 2
  bash "$HERE/accept_live.sh" 2>&1 | tee "$OUT/step4-accept_live.txt"
  rc4=${PIPESTATUS[0]}; ran4=1
  echo "step 4 rc=$rc4"
fi

echo
echo "=== p10_gate $TAG verdict"
verdict_line "1 prefill_gate (checkpoints off)" "$ran1" "$rc1"
verdict_line "2 tworeq at both KDA knobs"       "$ran2" "$rc2"
verdict_line "3 the flood, two arms"            "$ran3" "$rc3"
verdict_line "4 accept_live"                    "$ran4" "$rc4"
[ "$rc1" = 3 ] && { echo "=== FAIL (speed)"; exit 3; }
if [ "$rc1" != 0 ] || [ "$rc2" != 0 ] || [ "$rc3" != 0 ] || [ "$rc4" != 0 ]; then
  echo "=== FAIL"; exit 1
fi
echo "=== PASS"
exit 0
