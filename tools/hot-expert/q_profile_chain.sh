#!/bin/bash
# qprof_chain.sh -- the Q-PROFILE campaign: build the instrumented qwen38-vk in
# a separate worktree, prove it bit-identical to the binary the fresh baseline
# was measured on, then take the per-op table and a perf flat profile.
#
# Pattern copied from ~/bench/q38remap_chain.sh: take the rig lock, stop the
# gateway, do the work, and restore the owner's service on EVERY exit path.
set -u
OUT=~/bench/qprof_out; mkdir -p "$OUT"
SRC=~/src/colibri
WT=~/src/colibri-qprof
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
GSNAP=~/models/GLM-5.3-Flash-colibri-int4-g64
PRIS=~/bench/qwen38-vk.qprofbase
SHADERS=$SRC/c/shaders
GATEWAY_UP=0

log(){ echo "[$(date -Is)] $*"; }

. "$SRC/tools/hot-expert/rig_lock.sh"

wait_no_engine(){ for _ in $(seq 1 240); do pgrep -x "$1" >/dev/null || return 0; sleep 1; done
  echo "REFUSED: $1 still running"; return 1; }

start_gateway(){
  log "restarting the gateway (warms GLM first -- minutes)"
  setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &
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
  log "=== chain exiting rc=$rc"
  pkill -9 -x qwen38-vk 2>/dev/null; pkill -9 -x qwen38 2>/dev/null
  wait_no_engine qwen38-vk; wait_no_engine qwen38
  if [ "$GATEWAY_UP" != 1 ]; then
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
  log "=== qprof chain done"
}
trap on_exit EXIT INT TERM HUP

rig_lock_take "qprof-chain" || { echo "rig busy"; exit 3; }

# ---- step 0: nothing else may be running -------------------------------------
for e in qwen38 qwen38-vk; do
  pgrep -x "$e" >/dev/null && { log "REFUSED: $e already running"; exit 1; }
done
if pgrep -x glm53 >/dev/null; then
  log "stopping the owner's gateway"
  pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53
  wait_no_engine glm53 || { log "REFUSED: glm53 would not die"; exit 1; }
fi

# ---- step 1: freeze the pristine binary BEFORE anything rebuilds --------------
cp -f "$SRC/c/qwen38-vk" "$PRIS"
log "pristine  sha256=$(sha256sum "$PRIS" | cut -c1-16)  (the binary the 2026-09-11 baseline rows were taken on)"

# ---- step 2: build the candidate in its own worktree --------------------------
if [ ! -d "$WT" ]; then
  git -C "$SRC" worktree add "$WT" perf/q-profile-and-roadmap || exit 1
else
  git -C "$WT" fetch "$SRC" perf/q-profile-and-roadmap && git -C "$WT" reset --hard FETCH_HEAD || exit 1
fi
log "worktree HEAD: $(git -C "$WT" log --oneline -1)"
make -C "$WT/c" qwen38 qwen38-vk VK=1 > "$OUT/build.log" 2>&1 || { log "REFUSED: build failed"; tail -30 "$OUT/build.log"; exit 1; }
CAND="$WT/c/qwen38-vk"
log "candidate sha256=$(sha256sum "$CAND" | cut -c1-16)"
grep -iE 'warning|error' "$OUT/build.log" | head -20

for t in test_qwen38_prefix test_qwen38_config test_qwen38_metrics test_qwen38_serve_framing; do
  ( cd "$WT/c" && make tests/$t >/dev/null 2>&1 && ./tests/$t ) > "$OUT/$t.log" 2>&1
  log "$t exit=$?"
done

# Both binaries must load the SAME shader binaries, or the oracle compares two
# variables at once. They are unchanged on this branch; assert it.
log "shader sha: $(sha256sum "$SHADERS"/qmatmul.spv | cut -c1-16) vs worktree $(sha256sum "$WT/c/shaders/qmatmul.spv" | cut -c1-16)"

# ---- step 3: page cache -- GLM out, Qwen in, asserted -------------------------
python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(GLM) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY
free -g

warm(){ find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done; }
resid(){  # $1 label  $2 assert|report
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/qprof_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/qprof_fincore.txt').read().splitlines()[1:]:
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

# ---- step 4: the run harness --------------------------------------------------
export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
export COLI_VK_SHADERS="$SHADERS"
HIST_SRC="$QSNAP/.coli_usage"
PROMPT=/tmp/q_profile_prompt.txt
cat > "$PROMPT" <<'EOF'
Explain how a database transaction can deadlock, give a concrete example, and compare two practical prevention strategies in detail.
EOF

# Every run gets a FRESH copy of the same history: qwen38 rewrites COLI_USAGE at
# exit, so without this each run would preload a different tier and neither the
# oracle nor the repeats would be comparable (record §RP1's histogram confound).
engine(){   # $1 bin  $2 tag  $3 n_new  $4 timers(0|1)
  cp -f "$HIST_SRC" /tmp/qprof_hist.bin
  warm; resid "$2-pre" assert || { log "ABORT before $2"; exit 1; }
  log "--- run $2 ($(basename "$(dirname "$(dirname "$1")")")) n_new=$3 timers=$4"
  env SNAP="$QSNAP" COLI_USAGE=/tmp/qprof_hist.bin N_NEW="$3" \
      $( [ "$4" = 1 ] && echo COLI_TIMERS=1 ) \
      DUMP="/tmp/qprof_$2.f32" NOSTREAM=1 \
      "$1" 512 8 "$PROMPT" > "$OUT/$2.log" 2>&1
  local rc=$?
  log "  rc=$rc"
  [ $rc -eq 0 ] || { log "REFUSED: run $2 exited $rc"; tail -5 "$OUT/$2.log"; exit 1; }
  grep -E "Vulkan tier preloaded|^Speed:|^TTFT:|^Text      :|^Expert cache hit" "$OUT/$2.log" | sed 's/^/  /'
  resid "$2-post" report
}

# ---- step 5: the oracle -------------------------------------------------------
log "===== ORACLE ====="
engine "$PRIS" oracle-pristine   32 0
engine "$CAND" oracle-cand       32 0
engine "$CAND" oracle-cand-timed 32 1

for pair in "oracle-pristine oracle-cand" "oracle-pristine oracle-cand-timed"; do
  set -- $pair
  if [ ! -s "/tmp/qprof_$1.f32" ] || [ ! -s "/tmp/qprof_$2.f32" ]; then L="MISSING(no dump written)"
  elif cmp -s "/tmp/qprof_$1.f32" "/tmp/qprof_$2.f32"; then L="IDENTICAL ($(stat -c%s "/tmp/qprof_$1.f32") bytes)"
  else L="DIFFER"; fi
  a=$(grep "^Text      :" "$OUT/$1.log"); b=$(grep "^Text      :" "$OUT/$2.log")
  [ "$a" = "$b" ] && T="IDENTICAL" || T="DIFFER"
  log "ORACLE $1 vs $2: last-token logits $L, greedy text $T"
done

# ---- step 6: the per-op profile ----------------------------------------------
log "===== PROFILE ====="
for i in 1 2 3; do
  engine "$CAND" prof-$i 80 1
  grep -E "^\[OPTIME\]" "$OUT/prof-$i.log"
done

# ---- step 7: perf flat --------------------------------------------------------
log "===== PERF ====="
cp -f "$HIST_SRC" /tmp/qprof_hist.bin
warm; resid perf-pre assert || exit 1
( env SNAP="$QSNAP" COLI_USAGE=/tmp/qprof_hist.bin N_NEW=300 COLI_TIMERS=1 NOSTREAM=1 \
      "$CAND" 512 8 "$PROMPT" > "$OUT/perf-engine.log" 2>&1 ) &
EPID=$!
log "engine pid $EPID; waiting for prefill to finish before sampling"
# "TTFT:" is printed AFTER generate() returns, so it is useless as a start
# signal. The tier-preload line is printed before generation begins.
for _ in $(seq 1 120); do
  grep -q "Vulkan tier preloaded" "$OUT/perf-engine.log" 2>/dev/null && break
  kill -0 $EPID 2>/dev/null || break
  sleep 5
done
sleep 20   # let prefill finish; decode is ~60 s at 300 tokens
if kill -0 $EPID 2>/dev/null; then
  perf record -e cycles:u -F 999 -p $EPID -o /tmp/qprof_perf.data -- sleep 30 > "$OUT/perf-record.log" 2>&1
  log "perf record rc=$?"
else
  log "WARNING: engine exited before sampling"
fi
wait $EPID 2>/dev/null
grep -E "^\[OPTIME\]|^Speed:|^TTFT:" "$OUT/perf-engine.log" | head -40
log "--- perf by dso ---"
perf report -i /tmp/qprof_perf.data --stdio --sort=dso --percent-limit 0.2 2>/dev/null | head -20
log "--- perf by symbol ---"
perf report -i /tmp/qprof_perf.data --stdio --sort=symbol --percent-limit 0.2 2>/dev/null | head -45
resid perf-post report

log "===== END OF MEASUREMENTS ====="
