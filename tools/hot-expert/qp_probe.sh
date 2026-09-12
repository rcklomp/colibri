#!/bin/bash
# qp_probe.sh -- roadmap item QP, sub-items (a), (b) and (c): does Qwen3.8
# survive int4-g64 experts, int8 dense projections, an int8 LM head?
#
# The item is explicit that this is a NUMERICS probe and that speed is
# irrelevant if accuracy fails, so this script measures no speed. It runs the
# same two prompts through eight configurations of the same model and diffs
# their output. §G15's g15_probe.sh is the pattern, including the two things
# that pattern got right and a probe gets wrong at its peril:
#
#   R2  the candidate with EVERY knob off must equal the pristine byte for byte.
#       A probe that perturbs the default path is not a probe.
#   R3  the placement half of (a) isolated from the precision half. Forcing the
#       routed experts onto the CPU changes WHERE they run; R3-vs-R1 prices that
#       on its own, and R4-vs-R3 is then int4 and nothing else. §G15 needed this
#       and found GLM's missing swiglu clamp with it.
#
#   R1  pristine              tier on, no knobs                  -- the reference
#   R2  candidate             no knobs                           -- must equal R1
#   R3  candidate  EXPERTS_CPU=1                                 -- placement only
#   R4  candidate  EXPERTS_CPU=1 I4_SIM=1                        -- (a) int4-g64
#   R5  candidate  I8_DENSE=1                                    -- (b) int8 per-row
#   R6  candidate  I8_DENSE=2                                    -- (b) int8 g64
#   R7  candidate  I8_HEAD=1                                     -- (c) int8 per-row
#   R8  candidate  I8_HEAD=2                                     -- (c) int8 g64
#
# R4's control is R3 (identically placed, per the item's gate). R5-R8's control
# is R2: the dense path and the head run on the CPU in every configuration this
# engine has, so nothing about placement changes there, and R2 is the same
# binary with the same tier and the same frozen histogram.
#
# Oracles per configuration: `teacher_forcing` over the short prompt and over
# >=1000 positions of the long one (the line qwen38 did not have before QP --
# see q38_tf_emit), greedy text over 128 tokens, and the last prompt position's
# logit row by cosine / max-abs / argmax.
#
# Runs the owner's gateway down for its duration: takes ~/bench/.rig.lock first
# so gateway_watchdog.sh cannot race back in, and restarts it on every exit
# path. One benchmark at a time -- refuses if any engine is already up.
#
# Usage (on the rig):  tools/hot-expert/qp_probe.sh [smoke|short|long|all]
set -u

PHASE="${1:-all}"
SRC=~/src/colibri
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
OUT=~/bench/qp_probe_out; mkdir -p "$OUT"
PRIS=~/bench/qwen38-vk.qppristine
CAND="$SRC/c/qwen38-vk"
SHADERS="$SRC/c/shaders"
HIST_SRC="$QSNAP/.coli_usage"
GATEWAY_UP=0

log(){ echo "[$(date -Is)] $*"; }

. "$SRC/tools/hot-expert/rig_lock.sh"

wait_no_engine(){ for _ in $(seq 1 240); do pgrep -x "$1" >/dev/null || return 0; sleep 1; done
  echo "REFUSED: $1 still running"; return 1; }
# The pristine is a COPY under a different name (~/bench/qwen38-vk.qppristine),
# so `pgrep -x qwen38-vk` does not see it and neither does `pkill -9 -x`. Match
# the bench copies by path, in the bracket form CLAUDE.md requires -- `pkill -f`
# with an unbracketed pattern matches the ssh command line that carries it and
# kills the session.
kill_bench_engines(){ pkill -9 -f "bench/qwen38-[v]k\." 2>/dev/null; pkill -9 -f "bench/qwen3[8]\." 2>/dev/null; true; }
any_engine_up(){ pgrep -x qwen38 >/dev/null || pgrep -x qwen38-vk >/dev/null || \
                 pgrep -f "bench/qwen38-[v]k\." >/dev/null || pgrep -f "bench/qwen3[8]\." >/dev/null; }

start_gateway(){
  log "restarting the gateway (warms GLM first -- minutes)"
  SKIP_WARM=1 setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &
  for _ in $(seq 1 90); do
    pgrep -f "openai_[s]erver.py" >/dev/null && pgrep -x glm53 >/dev/null && break
    sleep 10
  done
  local svc eng code
  svc=$(pgrep -f "openai_[s]erver.py" | wc -l); eng=$(pgrep -x glm53 | wc -l)
  log "gateway processes: server=$svc engine=$eng"
  if [ "$svc" -ge 1 ] && [ "$eng" -ge 1 ]; then
    GATEWAY_UP=1
    code=$(curl -s -o /dev/null -m 30 -w '%{http_code}' -H "Authorization: Bearer $(cat ~/.colibri_api_key)" http://127.0.0.1:8081/v1/models) || true
    log "curl /v1/models -> ${code:-?}"
  else
    log "WARNING: gateway did not come up cleanly"
  fi
}

on_exit(){
  rc=$?
  log "=== qp_probe exiting rc=$rc"
  pkill -9 -x qwen38-vk 2>/dev/null; pkill -9 -x qwen38 2>/dev/null; kill_bench_engines
  wait_no_engine qwen38-vk; wait_no_engine qwen38
  if [ "${KEEP_QWEN:-0}" = 1 ]; then
    log "KEEP_QWEN=1: leaving Qwen resident and the gateway DOWN"
  else
    log "dropping Qwen from the page cache so GLM fits"
    python3 - <<'PY' || true
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(Qwen) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/Qwen3.8-Flash-Next-FP8")))
PY
    start_gateway
  fi
  rig_lock_release
  log "=== qp_probe done"
}
trap on_exit EXIT INT TERM HUP

rig_lock_take "qp-probe" || { echo "rig busy"; exit 3; }

if any_engine_up; then log "REFUSED: a qwen engine is already running"; pgrep -af qwen38 | head -3; exit 1; fi
if pgrep -x glm53 >/dev/null; then
  log "stopping the owner's gateway"
  pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53
  wait_no_engine glm53 || { log "REFUSED: glm53 would not die"; exit 1; }
fi

[ -x "$PRIS" ] || { log "REFUSED: no pristine at $PRIS"; exit 1; }
[ -x "$CAND" ] || { log "REFUSED: no candidate at $CAND"; exit 1; }
log "pristine  sha256=$(sha256sum "$PRIS" | cut -c1-16)"
log "candidate sha256=$(sha256sum "$CAND" | cut -c1-16)"
log "shaders   sha256=$(sha256sum "$SHADERS/qmatmul.spv" | cut -c1-16)"

# ---- the prompts --------------------------------------------------------------
# Short: one self-contained question, the shape §G1 used for GLM. Long: the same
# text repeated until the prefill has well over 1000 positions, which is §G15's
# construction (short x30) and gives >=1000 INDEPENDENT teacher-forced
# predictions rather than one.
SHORT=/tmp/qp_prompt_short.txt
LONG=/tmp/qp_prompt_long.txt
cat > "$SHORT" <<'EOF'
Explain how a database transaction can deadlock, give a concrete example with two transactions and two rows, and compare two practical prevention strategies in detail.
EOF
: > "$LONG"; for _ in $(seq 1 40); do cat "$SHORT" >> "$LONG"; done   # 40 x 30 tokens = ~1200 positions, >= the item's 1000
log "prompt bytes: short=$(wc -c < "$SHORT") long=$(wc -c < "$LONG")"

# ---- page cache: GLM out, Qwen in, asserted ----------------------------------
python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(GLM) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY
free -g

warm(){ find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done; }
resid(){  # $1 label  $2 assert|report
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/qp_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/qp_fincore.txt').read().splitlines()[1:]:
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
export COLI_VK_SHADERS="$SHADERS"

# run <bin> <tag> <prompt> <n_new> [KEY=VAL ...]
# Every run gets a FRESH copy of the same routing history: qwen38 REWRITES
# COLI_USAGE at exit (§RP1's histogram confound), so without this each run would
# preload a tier the previous run had edited and no two runs would be comparable.
run(){
  local bin="$1" tag="$2" prompt="$3" nn="$4"; shift 4
  cp -f "$HIST_SRC" /tmp/qp_hist.bin
  log "--- $tag  n_new=$nn  $*"
  local t0=$(date +%s)
  env SNAP="$QSNAP" COLI_USAGE=/tmp/qp_hist.bin N_NEW="$nn" NOSTREAM=1 \
      Q38_TF=1 Q38_TF_DUMP="$OUT/$tag.tf.f32" DUMP="$OUT/$tag.last.f32" \
      "$@" "$bin" 512 8 "$prompt" > "$OUT/$tag.log" 2>&1
  local rc=$?
  log "  rc=$rc  wall=$(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { log "REFUSED: $tag exited $rc"; tail -20 "$OUT/$tag.log"; exit 1; }
  grep -E "Vulkan tier preloaded|q38sim|^Speed:|^TTFT:|placement|q38tf" "$OUT/$tag.log" | sed 's/^/  /'
}

phase_smoke(){
  run "$PRIS" smoke-R1 "$SHORT" 8
  run "$CAND" smoke-R2 "$SHORT" 8
  run "$CAND" smoke-R3 "$SHORT" 8 Q38_EXPERTS_CPU=1
  run "$CAND" smoke-R4 "$SHORT" 8 Q38_EXPERTS_CPU=1 Q38_I4_SIM=1
  run "$CAND" smoke-R5 "$SHORT" 8 Q38_I8_DENSE=1
  run "$CAND" smoke-R7 "$SHORT" 8 Q38_I8_HEAD=1
}
phase_short(){
    run "$PRIS" s-R1 "$SHORT" 128
    run "$CAND" s-R2 "$SHORT" 128
    run "$CAND" s-R3 "$SHORT" 128 Q38_EXPERTS_CPU=1
    run "$CAND" s-R4 "$SHORT" 128 Q38_EXPERTS_CPU=1 Q38_I4_SIM=1
    run "$CAND" s-R5 "$SHORT" 128 Q38_I8_DENSE=1
    run "$CAND" s-R6 "$SHORT" 128 Q38_I8_DENSE=2
    run "$CAND" s-R7 "$SHORT" 128 Q38_I8_HEAD=1
    run "$CAND" s-R8 "$SHORT" 128 Q38_I8_HEAD=2
}
phase_long(){
    run "$PRIS" l-R1 "$LONG" 1
    run "$CAND" l-R2 "$LONG" 1
    run "$CAND" l-R3 "$LONG" 1 Q38_EXPERTS_CPU=1
    run "$CAND" l-R4 "$LONG" 1 Q38_EXPERTS_CPU=1 Q38_I4_SIM=1
    run "$CAND" l-R5 "$LONG" 1 Q38_I8_DENSE=1
    run "$CAND" l-R6 "$LONG" 1 Q38_I8_DENSE=2
    run "$CAND" l-R7 "$LONG" 1 Q38_I8_HEAD=1
    run "$CAND" l-R8 "$LONG" 1 Q38_I8_HEAD=2
}

# The phases are functions and `all` calls them in this process: re-invoking $0
# would try to take the rig lock a second time and refuse itself.
case "$PHASE" in
  smoke) phase_smoke ;;
  short) phase_short ;;
  long)  phase_long ;;
  all)   phase_smoke; phase_short; phase_long ;;
  *) log "unknown phase $PHASE"; exit 2 ;;
esac

log "outputs in $OUT; compare with tools/hot-expert/qp_compare.py"
