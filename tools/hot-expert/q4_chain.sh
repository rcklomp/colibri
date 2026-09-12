#!/bin/bash
# q4_chain.sh -- Q4: four accumulators in the BF16 dense GEMV at S=1 (qwen38-vk).
#
# q3_chain.sh is the pattern and the three Qwen traps it documents apply
# unchanged (131 shards, COLI_USAGE not COLI_USAGE_PATH and rewritten at exit,
# positional CLI). What is DIFFERENT here, and it is the whole point of the
# item: Q4 is NOT bit-identical by design. It changes the order the partial sums
# of a dot product are added, so the gate has two halves and they are not the
# same test:
#
#   knob OFF   must be BIT-IDENTICAL to the pristine. A knob that perturbs the
#              default path is not a knob. `cmp` on the logit dump, not a cosine.
#   knob ON    the greedy 128-token text must be IDENTICAL, and the last-token
#              logit cosine is REPORTED whatever it is. §G14's GLM53_I4_FAST is
#              the precedent: it shipped opt-in at relL2 3.0e-6 with identical
#              text. A changed text is the failure; a nonzero diff is not.
#
# So qp_compare.py is used for the numbers (teacher_forcing, greedy block,
# cosine/relL2/max-abs/argmax) but NOT for its VERDICT lines -- those apply QP's
# kill line for a FORMAT change (any short-prompt teacher_forcing change is a
# rejection), which is a different and stricter question than this item's gate.
# Read the numbers, not the verdicts.
#
# Also takes the per-op [OPTIME] table for all three configurations in the SAME
# fresh-process regime §Q-PROFILE used, three repeats each, and re-runs the
# isolated microbenchmark. The serving A/B (rome_bench.sh, from the Mac) is a
# separate step and is NOT run from here.
#
# Takes the rig lock and restores the owner's gateway on every exit path.
#
# Usage (on the rig):  tools/hot-expert/q4_chain.sh [all|oracle|profile|micro]
set -u
PHASE="${1:-all}"
OUT=~/bench/q4_out; mkdir -p "$OUT"
SRC=~/src/colibri
WT=~/src/colibri-q4
BRANCH=perf/q4-bf16-four-accumulators
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
PRIS=~/bench/qwen38-vk.q4base
SHADERS=$SRC/c/shaders
HIST_SRC="$QSNAP/.coli_usage"
GATEWAY_UP=0
REPS=${REPS:-3}

log(){ echo "[$(date -Is)] $*"; }

. "$SRC/tools/hot-expert/rig_lock.sh"

wait_no_engine(){ for _ in $(seq 1 240); do pgrep -x "$1" >/dev/null || return 0; sleep 1; done
  echo "REFUSED: $1 still running"; return 1; }
# The pristine is a COPY under another name, so `pgrep -x qwen38-vk` misses it.
# Bracket form per CLAUDE.md: an unbracketed `pkill -f` matches the ssh command
# line carrying the pattern and kills the session.
kill_bench_engines(){ pkill -9 -f "bench/qwen38-[v]k\." 2>/dev/null; pkill -9 -f "bench/qwen3[8]\." 2>/dev/null; true; }
any_engine_up(){ pgrep -x qwen38 >/dev/null || pgrep -x qwen38-vk >/dev/null || \
                 pgrep -f "bench/qwen38-[v]k\." >/dev/null || pgrep -f "bench/qwen3[8]\." >/dev/null; }

start_gateway(){
  log "restarting the gateway"
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
  log "=== q4 chain exiting rc=$rc"
  pkill -9 -x qwen38-vk 2>/dev/null; pkill -9 -x qwen38 2>/dev/null; kill_bench_engines
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
  log "=== q4 chain done"
}
trap on_exit EXIT INT TERM HUP

rig_lock_take "q4-chain" || { echo "rig busy"; exit 3; }

# ---- step 0: nothing else may be running -------------------------------------
if any_engine_up; then log "REFUSED: a qwen engine is already running"; pgrep -af qwen38 | head -3; exit 1; fi
if pgrep -x glm53 >/dev/null; then
  log "stopping the owner's gateway"
  pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53
  wait_no_engine glm53 || { log "REFUSED: glm53 would not die"; exit 1; }
fi

# ---- step 1: the pristine is the branch point, rebuilt and sha-checked --------
# ~/bench/qwen38-vk.q4base was frozen from hot-expert-tier HEAD BEFORE the
# candidate was written. Rebuilding it here and comparing the sha proves the
# main tree is still the branch point and that nothing edited it in between.
log "main tree HEAD: $(git -C "$SRC" log --oneline -1)"
make -C "$SRC/c" qwen38 qwen38-vk VK=1 > "$OUT/build-pristine.log" 2>&1 || { log "REFUSED: pristine build failed"; tail -30 "$OUT/build-pristine.log"; exit 1; }
log "pristine (tree)   sha256=$(sha256sum "$SRC/c/qwen38-vk" | cut -c1-16)"
log "pristine (frozen) sha256=$(sha256sum "$PRIS" | cut -c1-16)"
cmp -s "$SRC/c/qwen38-vk" "$PRIS" && log "  pristine binaries MATCH" || log "  WARNING: frozen pristine differs from the tree build"

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
# The candidate's rome_cpubench adds the gated residual's two shapes (I=320 and
# O=320), 1.28 of the 6.81 GB the dense-matmul counter streams; every pool is
# past the 128 MB L3 and the 96 MB Infinity Cache, so every row is DRAM.
phase_micro(){
  # The transform oracle first (§G15's rule): prove the candidate kernel sums
  # the SAME products as the pristine one before believing any timing from it.
  if gcc -O3 -march=native -fopenmp -o /tmp/rome_q4acc \
        "$WT/tools/hot-expert/rome_q4acc.c" -lm > "$OUT/q4acc-build.log" 2>&1; then
    log "--- rome_q4acc (transform oracle)"
    OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close /tmp/rome_q4acc 2>&1 \
      | tee "$OUT/q4acc.log"
    log "  rome_q4acc rc=${PIPESTATUS[0]}"
  else
    log "REFUSED: rome_q4acc did not build"; tail -5 "$OUT/q4acc-build.log"; exit 1
  fi
  if gcc -O3 -march=native -fopenmp -I"$WT/c" -o /tmp/q4_cpubench \
        "$WT/tools/hot-expert/rome_cpubench.c" -lm > "$OUT/cpubench-build.log" 2>&1; then
    for r in 1 2; do
      log "--- rome_cpubench repeat $r (8 threads)"
      OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close /tmp/q4_cpubench 2>&1 \
        | tee "$OUT/cpubench-cand-$r.log" | grep -E "BF16|threads"
    done
  else
    log "WARNING: cpubench did not build"; tail -5 "$OUT/cpubench-build.log"
  fi
}

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
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/q4_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/q4_fincore.txt').read().splitlines()[1:]:
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

# QP's two prompts, verbatim, so §Q4's oracle rows sit next to §QP's on the same
# text: short = one self-contained question (~30 tokens), long = the same x40,
# ~1200 positions, i.e. >=1000 INDEPENDENT teacher-forced predictions.
SHORT=/tmp/q4_prompt_short.txt
LONG=/tmp/q4_prompt_long.txt
cat > "$SHORT" <<'EOF'
Explain how a database transaction can deadlock, give a concrete example with two transactions and two rows, and compare two practical prevention strategies in detail.
EOF
: > "$LONG"; for _ in $(seq 1 40); do cat "$SHORT" >> "$LONG"; done
# The [OPTIME] table uses §Q-PROFILE's OWN prompt, not QP's, so the pristine
# column can be read against the record's 90.0 / 15.8 rather than only against
# the candidate column next to it. §Q1/§Q2/§Q3 all used this file.
PROF=/tmp/q_profile_prompt.txt
cat > "$PROF" <<'EOF'
Explain how a database transaction can deadlock, give a concrete example, and compare two practical prevention strategies in detail.
EOF
log "prompt bytes: short=$(wc -c < "$SHORT") long=$(wc -c < "$LONG") prof=$(wc -c < "$PROF")"

# run <bin> <tag> <prompt> <n_new> <timers 0|1> [KEY=VAL ...]
# Every run gets a FRESH copy of the same routing history: qwen38 REWRITES
# COLI_USAGE at exit (§RP1's histogram confound), so without this each run would
# preload a tier the previous run had edited.
run(){
  local bin="$1" tag="$2" prompt="$3" nn="$4" tm="$5"; shift 5
  cp -f "$HIST_SRC" /tmp/q4_hist.bin
  warm; resid "$tag-pre" assert || { log "ABORT before $tag"; exit 1; }
  log "--- $tag  n_new=$nn timers=$tm  $*"
  local t0=$(date +%s)
  env SNAP="$QSNAP" COLI_USAGE=/tmp/q4_hist.bin N_NEW="$nn" NOSTREAM=1 \
      $( [ "$tm" = 1 ] && echo COLI_TIMERS=1 ) \
      Q38_TF=1 Q38_TF_DUMP="$OUT/$tag.tf.f32" DUMP="$OUT/$tag.last.f32" \
      "$@" "$bin" 512 8 "$prompt" > "$OUT/$tag.log" 2>&1
  local rc=$?
  log "  rc=$rc  wall=$(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { log "REFUSED: $tag exited $rc"; tail -20 "$OUT/$tag.log"; exit 1; }
  grep -E "Vulkan tier preloaded|q38sim|Q4: BF16|^Speed:|^TTFT:|^Expert cache hit|q38tf" "$OUT/$tag.log" | sed 's/^/  /'
}

# A1 pristine | A2 candidate knob OFF (must be bit-identical to A1)
# A3 candidate knob ON  (greedy text must be identical; cosine reported)
phase_oracle(){
  log "===== ORACLE: short prompt, 128 greedy tokens ====="
  run "$PRIS" s-A1 "$SHORT" 128 0
  run "$CAND" s-A2 "$SHORT" 128 0
  run "$CAND" s-A3 "$SHORT" 128 0 Q38_BF16_ACC4=1
  log "===== ORACLE: long prompt, ~1200 teacher-forced positions ====="
  run "$PRIS" l-A1 "$LONG" 1 0
  run "$CAND" l-A2 "$LONG" 1 0
  run "$CAND" l-A3 "$LONG" 1 0 Q38_BF16_ACC4=1
  log "===== COMPARE (numbers from qp_compare.py; its VERDICT lines are QP's"
  log "      format kill line, NOT this item's gate -- see the header) ====="
  for p in s l; do
    python3 "$WT/tools/hot-expert/qp_compare.py" "$OUT" "$p-A1" "$p-A2" "knob OFF vs pristine -- must be BIT-IDENTICAL"
    python3 "$WT/tools/hot-expert/qp_compare.py" "$OUT" "$p-A1" "$p-A3" "knob ON vs pristine -- text identical, cosine reported"
  done
}

# ---- step 5: the per-op table, three configurations, same regime -------------
phase_profile(){
  log "===== PROFILE ====="
  for i in $(seq 1 "$REPS"); do
    run "$PRIS" prof-pristine-$i "$PROF" 80 1
    grep -E "^\[OPTIME\]" "$OUT/prof-pristine-$i.log" | head -40
    run "$CAND" prof-candoff-$i "$PROF" 80 1
    grep -E "^\[OPTIME\]" "$OUT/prof-candoff-$i.log" | head -40
    run "$CAND" prof-candon-$i "$PROF" 80 1 Q38_BF16_ACC4=1
    grep -E "^\[OPTIME\]" "$OUT/prof-candon-$i.log" | head -40
  done
  log "===== SUMMARY ====="
  python3 - "$OUT" "$REPS" <<'PY'
import re,sys,statistics as st
out,reps=sys.argv[1],int(sys.argv[2])
# ONLY the "decode only" bank: the engine prints a second [OPTIME] table for the
# whole run, prefill included, and averaging the two would mix regimes silently.
def table(tag):
    rows={}
    for i in range(1,reps+1):
        keep=False
        try: fh=open(f"{out}/prof-{tag}-{i}.log")
        except OSError: continue
        for line in fh:
            if "[OPTIME] ===" in line:
                keep="decode only" in line; continue
            if not keep: continue
            m=re.match(r"\[OPTIME\]\s+(\S.*?)\s{2,}([\d.]+)\s*ms",line)
            if m: rows.setdefault(m.group(1).strip(),[]).append(float(m.group(2)))
    return rows
p,o,n=table("pristine"),table("candoff"),table("candon")
print(f"{'op (decode only)':22}{'pristine':>11}{'knob off':>11}{'knob ON':>11}{'ON-pris':>10}")
for k in p:
    a=st.median(p[k])
    b=st.median(o[k]) if k in o else float('nan')
    c=st.median(n[k]) if k in n else float('nan')
    print(f"{k:22}{a:11.2f}{b:11.2f}{c:11.2f}{c-a:+10.2f}")
print("\nper-run spread (ms/token):")
for k in ("dense-matmul","lm-head","step","moe","deltanet"):
    if k in p:
        print(f"  {k:14} pris {['%.2f'%v for v in p[k]]}  off {['%.2f'%v for v in o.get(k,[])]}  ON {['%.2f'%v for v in n.get(k,[])]}")
PY
}

case "$PHASE" in
  micro)   phase_micro ;;
  oracle)  phase_oracle ;;
  profile) phase_profile ;;
  all)     phase_micro; phase_oracle; phase_profile ;;
  *) log "unknown phase $PHASE"; exit 2 ;;
esac
log "===== END OF MEASUREMENTS ====="
