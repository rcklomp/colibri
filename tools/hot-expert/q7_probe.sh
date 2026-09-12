#!/bin/bash
# q7_probe.sh -- roadmap item Q7-gpu: the in-engine legs.
#
# Spec: tools/hot-expert/Q7-DENSE-GPU-SPEC-2026-09-12.md. The isolated shader
# microbenchmark is `rome_vkbench <spv> q7` (step 0, no engine); THIS runs the
# model, and it exists for three things the microbenchmark cannot answer:
#
#   step0e  the `qsa-proj` sub-timer's real value -- §Q-PROFILE's ~16.4 ms/token
#           for the QSA projections is a SUBTRACTION ESTIMATE, and step 2's gate
#           ("half its step-0 figure") cannot be stated against one. Also the
#           knob-off bit-identity, and the denormal scan on the real checkpoint
#           (rome_vkbench proved the shader flushes bf16 denormals to zero; only
#           the checkpoint can say whether any exist).
#   step1   DeltaNet on dev0: the oracle at both prompt lengths and the [OPTIME]
#           A/B, three fresh-process repeats a side.
#   base    the same knob-off identity and [OPTIME] baseline for the steps-2/3
#           candidate, against a pristine rebuilt from the CURRENT branch point.
#   step2   QSA on dev0 (mask 2 alone and mask 3 = with step 1).
#   step3   the LM head (mask 4 alone and mask 7 = the whole arm).
#
# tools/hot-expert/qp_probe.sh is the pattern, including the two things that
# pattern got right and a probe gets wrong at its peril:
#
#   R2  the candidate with EVERY knob off must equal the pristine byte for byte.
#       A probe that perturbs the default path is not a probe.
#   the histogram  every run gets a FRESH copy of the same .coli_usage, because
#       qwen38 REWRITES it at exit (§RP1's confound). Without this no two runs
#       preload the same tier and no two numbers are comparable.
#
# The pristine is hot-expert-tier at branch time (313f1f5, i.e. WITH Q4 merged),
# built and installed as ~/bench/qwen38-vk.q7base + ~/bench/shaders_q7base. Each
# side runs with ITS OWN shaders: the fmt=9 addition is purely additive, and
# proving that on the pristine's own .spv is part of the point.
#
# Usage (on the rig):  q7_probe.sh [step0e|step1|all]
#   WT=<worktree>  the candidate tree (default /home/ronald/q7wt)
#
# Runs the owner's gateway down for its duration: takes the rig lock first so
# gateway_watchdog.sh cannot race back in, and -- via q7_lib.sh -- restarts the
# gateway ONLY if this chain is the one that stopped it.
set -u
. /home/ronald/bench/q7_lib.sh

PHASE="${1:-all}"
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
OUT=~/bench/q7_probe_out; mkdir -p "$OUT"
# Overridable, because the branch point moved: ~/bench/qwen38-vk.q7base is a
# 313f1f5 build and steps 2-3 are gated against a pristine rebuilt from a19b938
# (~/bench/qwen38-vk.q7base2 + ~/bench/shaders_q7base2), which is what Fable's
# merge criterion 3 asks for.
PRIS=${PRIS:-~/bench/qwen38-vk.q7base}
PRIS_SH=${PRIS_SH:-~/bench/shaders_q7base}
CAND="$WT/c/qwen38-vk"
CAND_SH="$WT/c/shaders"
HIST_SRC="$QSNAP/.coli_usage"

# q7_on_exit restarts the gateway only if THIS shell stopped it and releases the
# lock only if THIS shell took it, so it is safe on both branches below.
trap q7_on_exit EXIT INT TERM HUP

# Q7_PARENT_LOCK=<pid>: a chain that already holds the rig lock and has already
# stopped the gateway is running us as a step, and owns both on every exit path.
# NOT a bypass -- the declared pid is checked against the LIVE holder, so a stale
# or foreign value refuses exactly as rig_lock_take would. (The bypass this
# guards against is the 2026-09-12 07:39 incident: a chain that restarted the
# gateway on an exit path where the lock had been REFUSED.)
if [ -n "${Q7_PARENT_LOCK:-}" ]; then
  holder=$(rig_lock_holder) || { log "REFUSED: Q7_PARENT_LOCK=$Q7_PARENT_LOCK but no live lock holder"; exit 3; }
  hpid=$(echo "$holder" | awk '{print $2}')
  [ "$hpid" = "$Q7_PARENT_LOCK" ] || { log "REFUSED: rig lock held by [$holder], not by the declared parent $Q7_PARENT_LOCK"; exit 3; }
  log "rig lock and gateway delegated by parent pid $Q7_PARENT_LOCK [$holder]"
else
  q7_take_lock "q7-engine" 120 || exit 3
fi

# The bench copies live under a different name, so `pgrep -x` does not see them.
# Bracket form, per CLAUDE.md: an unbracketed `pkill -f` matches the ssh command
# line that carries the pattern and kills the session.
any_engine_up(){ pgrep -x qwen38 >/dev/null || pgrep -x qwen38-vk >/dev/null || \
                 pgrep -f "bench/qwen38-[v]k\." >/dev/null; }
if any_engine_up; then log "REFUSED: a qwen engine is already running"; exit 1; fi
stop_gateway || exit 1

# Rebuild the candidate BEFORE measuring it. The first run of this script
# measured a binary two commits old, because the step-0 chain had rebuilt only
# qmatmul.spv and `make` on a fresh worktree will not rebuild a .spv whose mtime
# ties its .comp. A chain rebuilds everything it is about to measure.
q7_build_engines || { log "REFUSED: build failed"; exit 1; }

[ -x "$PRIS" ] || { log "REFUSED: no pristine at $PRIS"; exit 1; }
[ -x "$CAND" ] || { log "REFUSED: no candidate at $CAND"; exit 1; }
log "pristine  $(sha256sum "$PRIS" | cut -c1-16)  shaders $(sha256sum "$PRIS_SH/qmatmul.spv" | cut -c1-16)"
log "candidate $(sha256sum "$CAND" | cut -c1-16)  shaders $(sha256sum "$CAND_SH/qmatmul.spv" | cut -c1-16)"

# ---- the prompts (identical to qp_probe.sh's, so the two items' numbers are
# ---- read against the same text) ---------------------------------------------
SHORT=/tmp/q7_prompt_short.txt
LONG=/tmp/q7_prompt_long.txt
cat > "$SHORT" <<'EOF'
Explain how a database transaction can deadlock, give a concrete example with two transactions and two rows, and compare two practical prevention strategies in detail.
EOF
: > "$LONG"; for _ in $(seq 1 40); do cat "$SHORT" >> "$LONG"; done
log "prompt bytes: short=$(wc -c < "$SHORT") long=$(wc -c < "$LONG")"

# ---- page cache: GLM out, Qwen in, ASSERTED before the first run -------------
python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(GLM) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY
free -g

warm(){ find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done; }
resid(){   # $1 label  $2 assert|report
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/q7_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/q7_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
pct=res/tot*100
print(f'[resid $1] shards={n} resident={pct:.4f}% short={short}')
if '$2'=='assert': assert n==131 and short==0, 'NOT 100% RESIDENT'
"
}
log "warming Qwen"; warm
resid boot assert || { log "REFUSED: Qwen not fully resident"; exit 1; }

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto

# run <bin> <shaders> <tag> <prompt> <n_new> [KEY=VAL ...]
run(){
  local bin="$1" sh="$2" tag="$3" prompt="$4" nn="$5"; shift 5
  cp -f "$HIST_SRC" /tmp/q7_hist.bin
  log "--- $tag  n_new=$nn  $*"
  local t0=$(date +%s)
  env SNAP="$QSNAP" COLI_USAGE=/tmp/q7_hist.bin N_NEW="$nn" NOSTREAM=1 \
      COLI_VK_SHADERS="$sh" \
      Q38_TF=1 Q38_TF_DUMP="$OUT/$tag.tf.f32" DUMP="$OUT/$tag.last.f32" \
      "$@" "$bin" 512 8 "$prompt" > "$OUT/$tag.log" 2>&1
  local rc=$?
  log "  rc=$rc  wall=$(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { log "REFUSED: $tag exited $rc"; tail -25 "$OUT/$tag.log"; exit 1; }
  grep -E "Vulkan tier preloaded|Q7 dense|Q7 denormal|^Speed:|placement" "$OUT/$tag.log" | sed 's/^/  /'
  resid "$tag" report
}

# ---- step 0, in-engine -------------------------------------------------------
phase_step0e(){
  run "$PRIS" "$PRIS_SH" s0-R1 "$SHORT" 128          # the reference
  run "$CAND" "$CAND_SH" s0-R2 "$SHORT" 128          # must be BYTE-IDENTICAL
  for r in 1 2 3; do                                  # qsa-proj + timers-on A/B
    run "$PRIS" "$PRIS_SH" s0-base-$r "$SHORT" 32 COLI_TIMERS=1
    run "$CAND" "$CAND_SH" s0-cand-$r "$SHORT" 32 COLI_TIMERS=1
  done
  # the denormal question, on the real checkpoint, over exactly the set this
  # item uploads (mask 15 = every set, so the scan covers all of it)
  run "$CAND" "$CAND_SH" s0-scan "$SHORT" 1 Q38_DENSE_GPU=15 Q38_DENSE_GPU_SCAN=1
}
# ---- step 1: DeltaNet on dev0 ------------------------------------------------
phase_step1(){
  run "$CAND" "$CAND_SH" s1-R3 "$SHORT" 128 Q38_DENSE_GPU=1
  run "$PRIS" "$PRIS_SH" l-R1  "$LONG"  1
  run "$CAND" "$CAND_SH" l-R2  "$LONG"  1
  run "$CAND" "$CAND_SH" l-R3  "$LONG"  1 Q38_DENSE_GPU=1
  for r in 1 2 3; do
    run "$CAND" "$CAND_SH" s1-gpu-$r "$SHORT" 32 COLI_TIMERS=1 Q38_DENSE_GPU=1
  done
}

# ---- the knob-off baseline for the steps-2/3 candidate ------------------------
# Re-run, not inherited: the candidate binary has changed (two more dispatch call
# sites), so "knob off is bit-identical" is a claim about THIS binary and the
# qsa-proj / lm-head buckets the arm is gated on are its own. The pristine here
# is the a19b938 rebuild (criterion 3), not the 313f1f5 q7base.
phase_base(){
  run "$PRIS" "$PRIS_SH" b2-R1 "$SHORT" 128          # reference
  run "$CAND" "$CAND_SH" b2-R2 "$SHORT" 128          # must be BYTE-IDENTICAL
  run "$PRIS" "$PRIS_SH" b2-lR1 "$LONG" 1
  run "$CAND" "$CAND_SH" b2-lR2 "$LONG" 1
  for r in 1 2 3; do
    run "$PRIS" "$PRIS_SH" b2-pris-$r "$SHORT" 32 COLI_TIMERS=1
    run "$CAND" "$CAND_SH" b2-off-$r  "$SHORT" 32 COLI_TIMERS=1
  done
}
# ---- step 2: QSA on dev0 (|= 2) ----------------------------------------------
# Oracle per the spec: PER BIT (mask 2 alone) and with ALL LANDED BITS TOGETHER
# (mask 3), both against the knob-OFF binary -- which is the same candidate, so
# the reference rows s0-R2 / l-R2 from phase_step0e are what these compare to.
phase_step2(){
  run "$CAND" "$CAND_SH" s2-R4 "$SHORT" 128 Q38_DENSE_GPU=2
  run "$CAND" "$CAND_SH" s2-R5 "$SHORT" 128 Q38_DENSE_GPU=3
  run "$CAND" "$CAND_SH" l-R4  "$LONG"  1   Q38_DENSE_GPU=2
  run "$CAND" "$CAND_SH" l-R5  "$LONG"  1   Q38_DENSE_GPU=3
  for r in 1 2 3; do
    run "$CAND" "$CAND_SH" s2-qsa-$r  "$SHORT" 32 COLI_TIMERS=1 Q38_DENSE_GPU=2
    run "$CAND" "$CAND_SH" s2-both-$r "$SHORT" 32 COLI_TIMERS=1 Q38_DENSE_GPU=3
  done
}
# ---- step 3: the LM head (|= 4), and the whole arm (mask 7) ------------------
phase_step3(){
  run "$CAND" "$CAND_SH" s3-R6 "$SHORT" 128 Q38_DENSE_GPU=4
  run "$CAND" "$CAND_SH" s3-R7 "$SHORT" 128 Q38_DENSE_GPU=7
  run "$CAND" "$CAND_SH" l-R6  "$LONG"  1   Q38_DENSE_GPU=4
  run "$CAND" "$CAND_SH" l-R7  "$LONG"  1   Q38_DENSE_GPU=7
  for r in 1 2 3; do
    run "$CAND" "$CAND_SH" s3-head-$r "$SHORT" 32 COLI_TIMERS=1 Q38_DENSE_GPU=4
    run "$CAND" "$CAND_SH" s3-arm-$r  "$SHORT" 32 COLI_TIMERS=1 Q38_DENSE_GPU=7
  done
}

case "$PHASE" in
  step0e) phase_step0e ;;
  step1)  phase_step1 ;;
  base)   phase_base ;;
  step2)  phase_step2 ;;
  step3)  phase_step3 ;;
  s23)    phase_base; phase_step2; phase_step3 ;;
  all)    phase_step0e; phase_step1; phase_step2; phase_step3 ;;
  *) log "unknown phase $PHASE"; exit 2 ;;
esac
log "engine legs done; outputs in $OUT -- compare with tools/hot-expert/qp_compare.py"
