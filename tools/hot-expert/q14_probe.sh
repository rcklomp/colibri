#!/bin/bash
# q14_probe.sh -- roadmap item Q14 step 0: the CPU-side twin of Q13.
#
# The question (record §Q7 steps 2-3 "What the arm costs on the CPU side"): with
# the whole Q7 arm on dev0 the three moved buckets fall 84.70 -> 38.88 (-45.82)
# but the token falls only -43.12, and the difference sits in two CPU BF16 GEMV
# buckets the item never touches -- `gr-read` +6.35 and `shared` +2.23 ms/token.
# It tracks neither expert eviction nor submit count; it tracks how long the
# token spends waiting on a GPU fence. Q13 then showed 0.93 of the 6.35 is dev0's
# power state, leaving ~5.4 unexplained. The hypothesis: while the main thread
# waits on the fence the CPU goes idle in short bursts, the OpenMP workers park,
# the cores drop to `scaling_min_freq` (2 500 of 3 700-3 914 MHz on this box,
# `schedutil` + `acpi-cpufreq`) and/or enter C2 (400 us exit latency, 800 us
# target residency), and every one of the ~97 sites a token pays a ramp.
#
# This is a ZERO-CODE item like Q13: nothing is built and nothing is edited. All
# three legs the roadmap names are pure environment, because the third one -- the
# fence spin -- is ALREADY a knob: vk_fence_wait() in c/backend_vulkan.c spins on
# vkGetFenceStatus for COLI_VK_SPIN_US (default 300) before it blocks.
#
#   base      today's defaults, i.e. the row that must reproduce the record
#   active    OMP_WAIT_POLICY=active                  (leg 1)
#   actinf    OMP_WAIT_POLICY=active + GOMP_SPINCOUNT=infinite  (leg 2)
#   spin0     COLI_VK_SPIN_US=0        -- the fence blocks immediately
#   spin3000  COLI_VK_SPIN_US=3000     -- the fence never blocks in practice
#   passive   OMP_WAIT_POLICY=passive  -- RP1's -7.5% control, sanity only
#
# Usage on the rig:
#   ~/bench/q14_probe.sh                                   # legs 1 and 2
#   Q14_VARIANTS="base spin0 spin3000" Q14_MASKS="7" ~/bench/q14_probe.sh
#   Q14_REPEATS=3 Q14_TAG=r2 ...                           # a second campaign
#
# Runs are INTERLEAVED by round (r1: every variant, then r2, ...) so that drift
# over the campaign hits every variant equally; Q13's block order could not do
# that because its variable was a rig-global root-owned setting.
#
# Cost: ~110 s per run plus the Qwen warm; the owner's gateway is down for the
# whole campaign, so the script takes the rig lock and restarts the gateway only
# if it is the thing that stopped it (q7_lib.sh).
set -u

# The binary: Q13's, so the `base` row is read against §Q13's paired `auto` leg
# (sha256 5a9a24953b33f509, tree a934285 = the tip with Q10). ~/src/colibri's
# qwen38-vk is still the Q10 PRISTINE (0da58e9a) and would measure the wrong
# engine -- that trap cost §Q13 a paragraph.
WT=${WT:-/home/ronald/src/colibri-q10}
BIN=${BIN:-$WT/c/qwen38-vk}
SH=${SH:-$WT/c/shaders}
WANT_SHA=${Q14_WANT_SHA:-5a9a24953b33f509}

OUT=~/bench/q14_out; mkdir -p "$OUT"
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
HIST_SRC="$QSNAP/.coli_usage"
VARIANTS=${Q14_VARIANTS:-"base active actinf"}
MASKS=${Q14_MASKS:-"7 3"}
BASE_MASKS=${Q14_BASE_MASKS:-"0"}     # extra masks run for `base` only (the control)
REPEATS=${Q14_REPEATS:-3}
TAG=${Q14_TAG:-}

. /home/ronald/bench/q7_lib.sh          # log(), rig lock, stop_gateway, q7_on_exit
trap q7_on_exit EXIT INT TERM HUP

q7_take_lock "q14-step0" 120 || exit 3
for e in qwen38 qwen38-vk; do pgrep -x "$e" >/dev/null && { log "REFUSED: $e running"; exit 1; }; done
pgrep -f "bench/qwen38-[v]k\." >/dev/null && { log "REFUSED: a bench qwen copy is running"; exit 1; }

git -C "$WT" status --short | grep -q . && { log "REFUSED: $WT is dirty -- Q14 measures the shipped binary"; exit 1; }
log "tree: $(git -C "$WT" log --oneline -1)"
[ -x "$BIN" ] || { log "REFUSED: no $BIN"; exit 1; }
GOT=$(sha256sum "$BIN" | cut -c1-16)
log "binary  $GOT   shaders $(sha256sum "$SH/qmatmul.spv" | cut -c1-16)"
[ "$GOT" = "$WANT_SHA" ] || { log "REFUSED: binary $GOT != expected $WANT_SHA (Q13's binary)"; exit 1; }

# Q13 left all three cards at `auto` and every number here is read against that.
for p in 0000:83:00.0 0000:48:00.0 0000:86:00.0; do
  L=$(cat /sys/bus/pci/devices/$p/power_dpm_force_performance_level)
  log "dpm $p = $L"
  [ "$L" = auto ] || { log "REFUSED: $p is at '$L', not auto -- Q14 is not comparable to §Q13's auto leg"; exit 1; }
done
log "cpufreq: governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) driver=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver) min=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq) max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"

stop_gateway || exit 1

# ---- the prompt: q7_probe.sh's / q13_probe.sh's, byte for byte ---------------
SHORT=/tmp/q14_prompt_short.txt
cat > "$SHORT" <<'EOF'
Explain how a database transaction can deadlock, give a concrete example with two transactions and two rows, and compare two practical prevention strategies in detail.
EOF

# ---- page cache: GLM out, Qwen in, asserted ----------------------------------
python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(GLM) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY
free -g
warm(){ find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done; }
resid(){
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/q14_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/q14_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
print(f'[resid $1] shards={n} resident={res/tot*100:.4f}% short={short}')
if '$2'=='assert': assert n==131 and short==0, 'NOT 100% RESIDENT'
"
}
log "warming Qwen"; warm
resid boot assert || { log "REFUSED: Qwen not fully resident"; exit 1; }

# ---- CPU sampling: core frequency, and the cpuidle counters -------------------
# ONE python process at 2 Hz (q13_probe.sh's rule: a shell sampler forks ~120
# processes a second and would steal from the 8 pinned engine threads). The
# fields are the two the hypothesis is about: per-core scaling_cur_freq, and
# dev0's power so that a GPU-side confound would be visible.
pm_start(){ PM_CSV="$OUT/$1.pm.csv"; : > "$PM_CSV"
  python3 - "$PM_CSV" <<'PY' &
import sys, time, glob, re
csv = sys.argv[1]
cpus = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"),
              key=lambda p: int(re.search(r"/cpu(\d+)/", p).group(1)))
d0 = "/sys/bus/pci/devices/0000:83:00.0"
h = (glob.glob(d0+"/hwmon/hwmon*")+[d0])[0]
gpu = [h+"/freq1_input", h+"/power1_average", d0+"/gpu_busy_percent"]
def rd(f):
    try:
        with open(f) as fh: return fh.read().strip()
    except Exception: return "-1"
with open(csv,"w",buffering=1) as out:
    while True:
        out.write(f"{time.time():.3f} " + " ".join(rd(f) for f in cpus+gpu) + "\n")
        time.sleep(0.5)
PY
  PM_PID=$!; }
pm_stop(){ [ -n "${PM_PID:-}" ] && kill $PM_PID 2>/dev/null; wait $PM_PID 2>/dev/null; PM_PID=; }
pm_report(){ python3 - "$1" <<'PY'
import sys, statistics
rows=[l.split() for l in open(sys.argv[1]) if l.strip()]
rows=[r for r in rows if len(r)>=5]
if not rows: print("  [pm] no samples"); raise SystemExit
ncpu=len(rows[0])-4
mean=[statistics.mean(int(x) for x in r[1:1+ncpu])/1e3 for r in rows]
mx  =[max(int(x) for x in r[1:1+ncpu])/1e3 for r in rows]
gs  =[int(r[1+ncpu])/1e6 for r in rows]; gp=[int(r[2+ncpu])/1e6 for r in rows]
print(f"  [pm cpu] freq mean-of-cores: med {statistics.median(mean):7.1f} min {min(mean):7.1f} max {max(mean):7.1f} MHz | "
      f"hottest core med {statistics.median(mx):7.1f} | dev0 sclk med {statistics.median(gs):6.1f} power med {statistics.median(gp):5.1f} W  n={len(rows)}")
PY
}
# cpuidle: usage/time per state, summed over all CPUs, delta across the run.
ci_snap(){ python3 - <<'PY'
import glob
tot={}
for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpuidle/state*"):
    st=p.rsplit("/",1)[1]
    try:
        u=int(open(p+"/usage").read()); t=int(open(p+"/time").read())
    except Exception: continue
    a=tot.setdefault(st,[0,0]); a[0]+=u; a[1]+=t
print(" ".join(f"{k}:{v[0]}:{v[1]}" for k,v in sorted(tot.items())))
PY
}
ci_report(){ python3 - "$1" "$2" "$3" <<'PY'
import sys
def parse(s): return {k:(int(u),int(t)) for k,u,t in (x.split(":") for x in s.split())}
a,b=parse(sys.argv[1]),parse(sys.argv[2]); wall=float(sys.argv[3])
out=[]
for k in sorted(b):
    du=b[k][0]-a[k][0]; dt=(b[k][1]-a[k][1])/1e6
    out.append(f"{k} n={du:9d} ({du/wall:8.0f}/s) t={dt:8.2f}s")
print("  [cpuidle] " + " | ".join(out))
PY
}

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto

variant_env(){   # echo the extra KEY=VAL for a variant name
  case "$1" in
    base)     echo "" ;;
    active)   echo "OMP_WAIT_POLICY=active" ;;
    actinf)   echo "OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite" ;;
    passive)  echo "OMP_WAIT_POLICY=passive" ;;
    spin0)    echo "COLI_VK_SPIN_US=0" ;;
    spin3000) echo "COLI_VK_SPIN_US=3000" ;;
    spininf)  echo "COLI_VK_SPIN_US=100000" ;;
    actspin)  echo "OMP_WAIT_POLICY=active GOMP_SPINCOUNT=infinite COLI_VK_SPIN_US=3000" ;;
    *) echo "UNKNOWN_VARIANT_$1"; return 1 ;;
  esac
}

# run <tag> <mask> <variant>
run(){
  local tag="$1" mask="$2" var="$3"
  local extra; extra=$(variant_env "$var") || { log "REFUSED: unknown variant $var"; exit 1; }
  case "$extra" in UNKNOWN_VARIANT_*) log "REFUSED: unknown variant $var"; exit 1;; esac
  cp -f "$HIST_SRC" /tmp/q14_hist.bin
  log "--- $tag  mask=$mask  variant=$var  env: ${extra:-<defaults>}"
  local ci0 ci1; ci0=$(ci_snap)
  pm_start "$tag"
  local t0=$(date +%s)
  env SNAP="$QSNAP" COLI_USAGE=/tmp/q14_hist.bin N_NEW=32 NOSTREAM=1 \
      COLI_VK_SHADERS="$SH" COLI_TIMERS=1 Q38_DENSE_GPU="$mask" \
      Q38_TF=1 Q38_TF_DUMP="$OUT/$tag.tf.f32" DUMP="$OUT/$tag.last.f32" \
      $extra "$BIN" 512 8 "$SHORT" > "$OUT/$tag.log" 2>&1
  local rc=$?
  local wall=$(( $(date +%s) - t0 ))
  pm_stop
  ci1=$(ci_snap)
  log "  rc=$rc  wall=${wall}s"
  [ $rc -eq 0 ] || { log "REFUSED: $tag exited $rc"; tail -25 "$OUT/$tag.log"; exit 1; }
  grep -E "Vulkan tier preloaded|Q7 dense|^Speed:" "$OUT/$tag.log" | sed 's/^/  /'
  pm_report "$OUT/$tag.pm.csv"
  ci_report "$ci0" "$ci1" "${wall:-1}"
}

# ---- a 30 s no-engine idle sample, for the freq/C-state floor ----------------
log "--- idle sample (30 s, no engine)"
ci0=$(ci_snap); pm_start "idle${TAG}"; sleep 30; pm_stop; ci1=$(ci_snap)
pm_report "$OUT/idle${TAG}.pm.csv"; ci_report "$ci0" "$ci1" 30

# ---- the campaign, interleaved by round --------------------------------------
log "variants: $VARIANTS   masks: $MASKS   base-only masks: $BASE_MASKS   repeats: $REPEATS"
for r in $(seq 1 "$REPEATS"); do
  for v in $VARIANTS; do
    for m in $MASKS; do run "$v$TAG-m$m-$r" "$m" "$v"; done
    if [ "$v" = base ]; then
      for m in $BASE_MASKS; do run "$v$TAG-m$m-$r" "$m" "$v"; done
    fi
  done
done

# ---- the table ---------------------------------------------------------------
ALLMASKS="$MASKS $BASE_MASKS"
python3 - "$OUT" "$TAG" "$ALLMASKS" $VARIANTS <<'PY'
import sys, os, glob, statistics
D, TAG, MASKS = sys.argv[1], sys.argv[2], sys.argv[3].split()
VARS = sys.argv[4:]
def bank(p):
    out, on = {}, False
    for line in open(p, errors="replace"):
        if "[OPTIME] === decode only" in line:
            on = True; out["token"] = float(line.split("forwards,")[1].split("ms/forward")[0]); continue
        if on and "[OPTIME] === whole run" in line: break
        if on and line.startswith("[OPTIME]"):
            f = line.split()
            if len(f) >= 3:
                try: out[f[1]] = float(f[2])
                except ValueError: pass
    return out
def group(pat):
    bs = [b for b in (bank(p) for p in sorted(glob.glob(os.path.join(D, pat + ".log")))) if b]
    if not bs: return {}, [], 0
    keys = set().union(*map(set, bs))
    med = {k: statistics.median([b[k] for b in bs if k in b]) for k in keys}
    return med, [b.get("token") for b in bs], len(bs)
rows = ["token", "gr-read", "gr-down", "gr-up", "gr-rms", "gr-mix", "shared", "cpu-experts",
        "router", "moe", "vk-dense", "vk-take", "vk-issue", "dn-proj", "qsa-proj", "lm-head",
        "dn-recur", "dense-matmul"]
for mask in MASKS:
    data = [(v,) + group(f"{v}{TAG}-m{mask}-*") for v in VARS]
    data = [d for d in data if d[3]]
    if not data: continue
    print(f"\n=== Q38_DENSE_GPU={mask}   " + ", ".join(f"{d[3]}x {d[0]}" for d in data))
    w = max(len(r) for r in rows) + 2
    print("bucket".ljust(w) + "".join(d[0].rjust(12) for d in data) +
          "".join(("d:" + d[0]).rjust(12) for d in data[1:]))
    for r in rows:
        vals = [d[1].get(r) for d in data]
        line = r.ljust(w) + "".join((f"{v:12.3f}" if v is not None else "".rjust(12)) for v in vals)
        for v in vals[1:]:
            line += (f"{v-vals[0]:+12.3f}" if (v is not None and vals[0] is not None) else "".rjust(12))
        print(line)
    print("token repeats".ljust(w) + "".join(("/".join(f"{t:.2f}" for t in d[2] if t is not None)).rjust(24) for d in data))
PY

# ---- the oracle: an environment variable changes no arithmetic ---------------
set -- $VARIANTS; REF=$1; shift
log "--- oracle: teacher-forcing and last-token dumps, $REF vs $*"
NID=0; NDIFF=0
for v in "$@"; do
  for mask in $MASKS; do
    for r in $(seq 1 "$REPEATS"); do
      a=$(sha256sum "$OUT/$REF$TAG-m$mask-$r.tf.f32" 2>/dev/null | cut -c1-16)
      h=$(sha256sum "$OUT/$v$TAG-m$mask-$r.tf.f32" 2>/dev/null | cut -c1-16)
      al=$(sha256sum "$OUT/$REF$TAG-m$mask-$r.last.f32" 2>/dev/null | cut -c1-16)
      hl=$(sha256sum "$OUT/$v$TAG-m$mask-$r.last.f32" 2>/dev/null | cut -c1-16)
      if [ -n "$a" ] && [ -n "$h" ] && [ "$a" = "$h" ] && [ "$al" = "$hl" ]; then
        res=IDENTICAL; NID=$((NID+1))
      else res="DIFFERS  tf $a/$h  last $al/$hl"; NDIFF=$((NDIFF+1)); fi
      echo "  $v m$mask r$r: $res"
    done
  done
done
log "oracle: $NID identical, $NDIFF differing"
log "q14 step 0 done; logs in $OUT"
