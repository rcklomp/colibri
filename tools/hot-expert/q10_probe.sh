#!/bin/bash
# q10_probe.sh -- roadmap item Q10: the DeltaNet recurrence's loop order, in
# the engine.  The isolated step-0 microbenchmark is `rome_dnbench recur`
# (no engine); THIS runs the model, for the two things the microbenchmark
# cannot answer:
#
#   oracle  the knob-OFF identity and the knob-ON identity, separately.  The
#           OFF leg is not a formality here: Q10 moves the base recurrence out
#           of q38_deltanet's `omp for` into a noinline function so the
#           build-time self-check can compare the code the engine actually
#           runs, and that extraction has to be proved neutral BEFORE the
#           folding is judged.  Two legs, two different failures.
#   optime  [OPTIME] dn-recur and deltanet, three fresh-process repeats each of
#           pristine / candidate-OFF / candidate-ON, plus every neighbouring
#           bucket (§Q2's rule: an item that changes a loop reports the buckets
#           that neighbour the buffers it touches).
#
# tools/hot-expert/q7_probe.sh is the pattern and ~/bench/q7_lib.sh is its
# library; the two things that pattern got right and a probe gets wrong at its
# peril are kept verbatim:
#   R2  the candidate with the knob off must equal the pristine byte for byte.
#   the histogram  every run gets a FRESH copy of the same .coli_usage, because
#       qwen38 REWRITES it at exit (§RP1's confound, MEASURING.md).
#
# Q10 changes no shader and no file glm53 includes (qwen38_core.h is included
# by qwen38.c and the qwen38 tests only), so both sides run with the PRISTINE
# shaders and glm53 is not part of this probe.
#
# Usage (on the rig):  q10_probe.sh [oracle|optime|all]
#   WT=<worktree>    the candidate tree (default /home/ronald/src/colibri-q10)
#   PRIS=<binary>    the pristine qwen38-vk (default ~/bench/qwen38-vk.q10base)
#
# Runs the owner's gateway down for its duration: takes the rig lock first so
# gateway_watchdog.sh cannot race back in, and -- via q7_lib.sh -- restarts the
# gateway ONLY if this chain is the one that stopped it.
set -u
WT=${WT:-/home/ronald/src/colibri-q10}
OUT=${OUT:-~/bench/q10_probe_out}
. /home/ronald/bench/q7_lib.sh

PHASE="${1:-all}"
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
mkdir -p "$OUT"
PRIS=${PRIS:-~/bench/qwen38-vk.q10base}
PRIS_SH=${PRIS_SH:-~/bench/shaders_q10base}
CAND="$WT/c/qwen38-vk"
CAND_SH="$WT/c/shaders"
HIST_SRC="$QSNAP/.coli_usage"

trap q7_on_exit EXIT INT TERM HUP

any_engine_up(){ pgrep -x qwen38 >/dev/null || pgrep -x qwen38-vk >/dev/null || \
                 pgrep -f "bench/qwen38-[v]k\." >/dev/null; }

if [ -n "${Q10_PARENT_LOCK:-}" ]; then
  holder=$(rig_lock_holder) || { log "REFUSED: Q10_PARENT_LOCK set but no live holder"; exit 3; }
  hpid=$(echo "$holder" | awk '{print $2}')
  [ "$hpid" = "$Q10_PARENT_LOCK" ] || { log "REFUSED: rig lock held by [$holder]"; exit 3; }
  log "rig lock and gateway delegated by parent pid $Q10_PARENT_LOCK"
else
  q7_take_lock "q10-engine" 120 || exit 3
fi

if any_engine_up; then log "REFUSED: a qwen engine is already running"; exit 1; fi
stop_gateway || exit 1

log "building the candidate in $WT"
make -C "$WT/c" qwen38 qwen38-vk VK=1 -j8 > /tmp/q10_build.log 2>&1 || {
  log "REFUSED: build failed"; tail -25 /tmp/q10_build.log; exit 1; }
sha256sum "$WT/c/qwen38-vk" "$WT/c/qwen38"

[ -x "$PRIS" ] || { log "REFUSED: no pristine at $PRIS"; exit 1; }
log "pristine  $(sha256sum "$PRIS" | cut -c1-16)"
log "candidate $(sha256sum "$CAND" | cut -c1-16)"

# Same prompts as qp_probe.sh / q7_probe.sh, so the three items' numbers are
# read against the same text.
SHORT=/tmp/q10_prompt_short.txt
LONG=/tmp/q10_prompt_long.txt
cat > "$SHORT" <<'EOF'
Explain how a database transaction can deadlock, give a concrete example with two transactions and two rows, and compare two practical prevention strategies in detail.
EOF
: > "$LONG"; for _ in $(seq 1 40); do cat "$SHORT" >> "$LONG"; done
log "prompt bytes: short=$(wc -c < "$SHORT") long=$(wc -c < "$LONG")"

python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(GLM) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY
free -g

warm(){ find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done; }
resid(){
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/q10_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/q10_fincore.txt').read().splitlines()[1:]:
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
  cp -f "$HIST_SRC" /tmp/q10_hist.bin
  log "--- $tag  n_new=$nn  $*"
  local t0=$(date +%s)
  env SNAP="$QSNAP" COLI_USAGE=/tmp/q10_hist.bin N_NEW="$nn" NOSTREAM=1 \
      COLI_VK_SHADERS="$sh" Q38_VERBOSE=1 \
      Q38_TF=1 Q38_TF_DUMP="$OUT/$tag.tf.f32" DUMP="$OUT/$tag.last.f32" \
      "$@" "$bin" 512 8 "$prompt" > "$OUT/$tag.log" 2>&1
  local rc=$?
  log "  rc=$rc  wall=$(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { log "REFUSED: $tag exited $rc"; tail -25 "$OUT/$tag.log"; exit 1; }
  grep -E "Vulkan tier preloaded|\[DN\] recurrence|^Speed:" "$OUT/$tag.log" | sed 's/^/  /'
}

cmpf(){  # cmpf <a> <b> <label>
  if cmp -s "$OUT/$1" "$OUT/$2"; then
    log "ORACLE $3: BYTE-IDENTICAL ($(stat -c%s "$OUT/$1") bytes)"
  else
    log "ORACLE $3: ****DIFFER**** $(cmp "$OUT/$1" "$OUT/$2" 2>&1 | head -1)"
  fi
}

phase_oracle(){
  # 128 tokens, greedy continuation + last-token logits + teacher forcing
  run "$PRIS" "$PRIS_SH" o-pris-128 "$SHORT" 128
  run "$CAND" "$CAND_SH" o-off-128  "$SHORT" 128 Q38_DN_RECUR_FOLD=0
  run "$CAND" "$CAND_SH" o-on-128   "$SHORT" 128
  # 32 tokens, timers off and on -- the timers change the region's shape
  run "$PRIS" "$PRIS_SH" o-pris-32  "$SHORT" 32
  run "$CAND" "$CAND_SH" o-on-32    "$SHORT" 32
  run "$PRIS" "$PRIS_SH" o-pris-32t "$SHORT" 32 COLI_TIMERS=1
  run "$CAND" "$CAND_SH" o-on-32t   "$SHORT" 32 COLI_TIMERS=1
  # the long prompt, one token: 1230 teacher-forcing positions
  run "$PRIS" "$PRIS_SH" o-pris-long "$LONG" 1
  run "$CAND" "$CAND_SH" o-on-long   "$LONG" 1
  echo "=================== ORACLE ==================="
  cmpf o-pris-128.last.f32 o-off-128.last.f32 "knob OFF (extraction only), last logits @128"
  cmpf o-pris-128.tf.f32   o-off-128.tf.f32   "knob OFF (extraction only), teacher forcing @128"
  cmpf o-pris-128.last.f32 o-on-128.last.f32  "knob ON, last logits @128"
  cmpf o-pris-128.tf.f32   o-on-128.tf.f32    "knob ON, teacher forcing @128"
  cmpf o-pris-32.last.f32  o-on-32.last.f32   "knob ON, last logits @32 timers off"
  cmpf o-pris-32t.last.f32 o-on-32t.last.f32  "knob ON, last logits @32 timers on"
  cmpf o-pris-long.last.f32 o-on-long.last.f32 "knob ON, last logits @long"
  cmpf o-pris-long.tf.f32   o-on-long.tf.f32   "knob ON, teacher forcing @long (1230 pos)"
  # greedy text: the WHOLE continuation, not its first line (§QP's own bug --
  # its greedy comparison read one line of a multi-line answer and called two
  # divergent 128-token outputs identical).
  for t in o-off-128 o-on-128; do
    if diff -q <(sed -n '/^Generated (/,/^TTFT:/p' "$OUT/o-pris-128.log") \
               <(sed -n '/^Generated (/,/^TTFT:/p' "$OUT/$t.log") >/dev/null; then
      log "ORACLE greedy text $t vs pristine @128: IDENTICAL"
    else
      log "ORACLE greedy text $t vs pristine @128: ****DIFFER****"
    fi
  done
}

phase_optime(){
  for r in 1 2 3; do
    run "$PRIS" "$PRIS_SH" t-pris-$r "$SHORT" 32 COLI_TIMERS=1
    run "$CAND" "$CAND_SH" t-off-$r  "$SHORT" 32 COLI_TIMERS=1 Q38_DN_RECUR_FOLD=0
    run "$CAND" "$CAND_SH" t-on-$r   "$SHORT" 32 COLI_TIMERS=1
  done
}

case "$PHASE" in
  oracle) phase_oracle ;;
  optime) phase_optime ;;
  all)    phase_oracle; phase_optime ;;
  *) echo "usage: $0 [oracle|optime|all]"; exit 2 ;;
esac
log "phase $PHASE complete"
