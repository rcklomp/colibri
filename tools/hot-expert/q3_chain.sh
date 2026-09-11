#!/bin/bash
# q3_chain.sh -- Q3: the shared expert into the GPU gap (qwen38-vk).
#
# Builds the pristine (branch point) and the candidate, proves the candidate
# BIT-IDENTICAL (last-token logits by cmp, greedy text by string compare), and
# takes the per-op [OPTIME] table for both in the SAME fresh-process regime
# §Q-PROFILE used, three repeats each. The serving A/B (rome_bench.sh) is a
# separate step and is NOT run from here.
#
# Copied from q_profile_chain.sh, which is the pattern for this engine; the
# three Qwen traps it documents apply unchanged (131 shards, COLI_USAGE not
# COLI_USAGE_PATH and rewritten at exit, positional CLI). Takes the rig lock and
# restores the owner's gateway on every exit path.
set -u
OUT=~/bench/q3_out; mkdir -p "$OUT"
SRC=~/src/colibri
WT=~/src/colibri-q3
BRANCH=perf/q3-shared-expert-into-gap
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
PRIS=~/bench/qwen38-vk.q3base
SHADERS=$SRC/c/shaders
GATEWAY_UP=0
REPS=${REPS:-3}

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
  if [ "${KEEP_QWEN:-0}" = 1 ]; then
    log "KEEP_QWEN=1: leaving Qwen resident and the gateway DOWN for the serving A/B"
  elif [ "$GATEWAY_UP" != 1 ]; then
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
  log "=== q3 chain done"
}
trap on_exit EXIT INT TERM HUP

rig_lock_take "q3-chain" || { echo "rig busy"; exit 3; }

# ---- step 0: nothing else may be running -------------------------------------
for e in qwen38 qwen38-vk; do
  pgrep -x "$e" >/dev/null && { log "REFUSED: $e already running"; exit 1; }
done
if pgrep -x glm53 >/dev/null; then
  log "stopping the owner's gateway"
  pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53
  wait_no_engine glm53 || { log "REFUSED: glm53 would not die"; exit 1; }
fi

# ---- step 1: the pristine is the branch point, built from the main tree -------
# NOT ~/bench/qwen38-vk.qprofbase: that is the pre-merge binary. The branch point
# is hot-expert-tier @ the merge of perf/q-profile-and-roadmap, which carries the
# [OPTIME] counters this item is gated on.
log "main tree HEAD: $(git -C "$SRC" log --oneline -1)"
make -C "$SRC/c" qwen38 qwen38-vk VK=1 > "$OUT/build-pristine.log" 2>&1 || { log "REFUSED: pristine build failed"; tail -30 "$OUT/build-pristine.log"; exit 1; }
cp -f "$SRC/c/qwen38-vk" "$PRIS"
log "pristine  sha256=$(sha256sum "$PRIS" | cut -c1-16)"

# ---- step 2: the candidate in its own worktree --------------------------------
if [ ! -d "$WT" ]; then
  git -C "$SRC" worktree add "$WT" "$BRANCH" || exit 1
else
  git -C "$WT" fetch "$SRC" "$BRANCH" && git -C "$WT" reset --hard FETCH_HEAD || exit 1
fi
log "worktree HEAD: $(git -C "$WT" log --oneline -1)"
make -C "$WT/c" qwen38 qwen38-vk VK=1 > "$OUT/build-cand.log" 2>&1 || { log "REFUSED: candidate build failed"; tail -30 "$OUT/build-cand.log"; exit 1; }
CAND="$WT/c/qwen38-vk"
log "candidate sha256=$(sha256sum "$CAND" | cut -c1-16)"
grep -iE 'warning|error' "$OUT/build-cand.log" | head -20

for t in test_qwen38_prefix test_qwen38_config test_qwen38_metrics test_qwen38_serve_framing; do
  ( cd "$WT/c" && make tests/$t >/dev/null 2>&1 && ./tests/$t ) > "$OUT/$t.log" 2>&1
  log "$t exit=$?"
done

# Both binaries must load the SAME shaders or the oracle compares two variables.
log "shader sha: $(sha256sum "$SHADERS"/qmatmul.spv | cut -c1-16) vs worktree $(sha256sum "$WT/c/shaders/qmatmul.spv" | cut -c1-16)"

# ---- step 2.5: the isolated microbenchmark ------------------------------------
# What Q3 moves is one expert-shaped FP8 triple per layer. rome_cpubench's K=1
# row is that op at the engine's real shapes, 8 threads, out of a 630 MB pool --
# past the 128 MB L3 and the 96 MB Infinity Cache, so it streams from DRAM. Run
# with no engine up and the lock held, like every other row it has produced.
if [ "${SKIP_MICRO:-0}" != 1 ]; then
  if gcc -O3 -march=native -fopenmp -I"$WT/c" -o /tmp/q3_cpubench \
        "$WT/tools/hot-expert/rome_cpubench.c" -lm > "$OUT/cpubench-build.log" 2>&1; then
    log "--- rome_cpubench (8 threads)"
    OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close /tmp/q3_cpubench 2>&1 \
      | tee "$OUT/cpubench.log" | grep -E "FP8|threads"
  else
    log "WARNING: cpubench did not build"; tail -5 "$OUT/cpubench-build.log"
  fi
fi

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
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/q3_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/q3_fincore.txt').read().splitlines()[1:]:
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

# Every run gets a FRESH copy of the same history (qwen38 rewrites COLI_USAGE at
# exit; §RP1's histogram confound). The tier must therefore come up identical in
# every run, and the oracle would be meaningless if it did not.
engine(){   # $1 bin  $2 tag  $3 n_new  $4 timers(0|1)
  cp -f "$HIST_SRC" /tmp/q3_hist.bin
  warm; resid "$2-pre" assert || { log "ABORT before $2"; exit 1; }
  log "--- run $2 n_new=$3 timers=$4"
  env SNAP="$QSNAP" COLI_USAGE=/tmp/q3_hist.bin N_NEW="$3" \
      $( [ "$4" = 1 ] && echo COLI_TIMERS=1 ) \
      DUMP="/tmp/q3_$2.f32" NOSTREAM=1 \
      "$1" 512 8 "$PROMPT" > "$OUT/$2.log" 2>&1
  local rc=$?
  log "  rc=$rc"
  [ $rc -eq 0 ] || { log "REFUSED: run $2 exited $rc"; tail -5 "$OUT/$2.log"; exit 1; }
  grep -E "Vulkan tier preloaded|^Speed:|^TTFT:|^Expert cache hit" "$OUT/$2.log" | sed 's/^/  /'
}

oracle(){  # $1 tag-a  $2 tag-b
  local L T a b
  if [ ! -s "/tmp/q3_$1.f32" ] || [ ! -s "/tmp/q3_$2.f32" ]; then L="MISSING(no dump written)"
  elif cmp -s "/tmp/q3_$1.f32" "/tmp/q3_$2.f32"; then L="IDENTICAL ($(stat -c%s "/tmp/q3_$1.f32") bytes)"
  else L="DIFFER -- first byte $(cmp "/tmp/q3_$1.f32" "/tmp/q3_$2.f32" 2>&1 | head -1)"; fi
  a=$(grep "^Text      :" "$OUT/$1.log"); b=$(grep "^Text      :" "$OUT/$2.log")
  [ "$a" = "$b" ] && T="IDENTICAL" || T="DIFFER"
  log "ORACLE $1 vs $2: last-token logits $L, greedy text $T"
}

# ---- step 5: the oracle (bit-identical is this item's gate) -------------------
log "===== ORACLE ====="
engine "$PRIS" oracle-pristine     32 0
engine "$CAND" oracle-cand         32 0
engine "$CAND" oracle-cand-timed   32 1
oracle oracle-pristine oracle-cand
oracle oracle-pristine oracle-cand-timed
# 128 tokens: a longer greedy run, so a divergence that needs a few tokens of
# context to appear cannot hide behind a 32-token window.
engine "$PRIS" oracle128-pristine 128 0
engine "$CAND" oracle128-cand     128 0
oracle oracle128-pristine oracle128-cand

# ---- step 6: the per-op table, both binaries, same regime ---------------------
log "===== PROFILE ====="
for i in $(seq 1 "$REPS"); do
  engine "$PRIS" prof-pristine-$i 80 1
  grep -E "^\[OPTIME\]" "$OUT/prof-pristine-$i.log"
  engine "$CAND" prof-cand-$i 80 1
  grep -E "^\[OPTIME\]" "$OUT/prof-cand-$i.log"
done

log "===== SUMMARY ====="
python3 - "$OUT" "$REPS" <<'PY'
import re,sys,statistics as st
out,reps=sys.argv[1],int(sys.argv[2])
# ONLY the "decode only" bank: the engine prints a second [OPTIME] table for the
# whole run, prefill included, and averaging the two would silently mix regimes.
def table(tag):
    rows={}
    for i in range(1,reps+1):
        keep=False
        for line in open(f"{out}/prof-{tag}-{i}.log"):
            if "[OPTIME] ===" in line:
                keep="decode only" in line; continue
            if not keep: continue
            m=re.match(r"\[OPTIME\]\s+(\S.*?)\s{2,}([\d.]+)\s*ms",line)
            if m: rows.setdefault(m.group(1).strip(),[]).append(float(m.group(2)))
    return rows
p,c=table("pristine"),table("cand")
print(f"{'op (decode only)':22}{'pristine':>12}{'candidate':>12}{'delta':>10}")
for k in p:
    if k not in c: continue
    a,b=st.median(p[k]),st.median(c[k])
    print(f"{k:22}{a:12.2f}{b:12.2f}{b-a:+10.2f}")
print("\nper-run spread (ms/token):")
for k in ("moe","vk-take","shared","cpu-experts","vk-issue","step" ):
    if k in p: print(f"  {k:14} pristine {['%.2f'%v for v in p[k]]}  candidate {['%.2f'%v for v in c.get(k,[])]}")
PY
log "===== END OF MEASUREMENTS ====="
