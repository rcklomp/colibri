#!/bin/bash
# q13_probe.sh -- roadmap item Q13 step 0: dev0's idle clocks between dense submits.
#
# The question (record §Q7 "Why it undershoots ... it is NOT the kernel", and
# §Q7 `### Arbitration`): the identical two DeltaNet submits cost 18.61 ms/token
# back-to-back and 27.46-34.67 with the engine's own inter-submit gaps, and the
# effect is non-monotonic in gap length -- a power-state signature. If dev0 is
# de-clocking in those sub-millisecond idle bursts, forcing its DPM performance
# level to `high` should move `dn-proj` toward the back-to-back figure.
#
# THIS SCRIPT NEVER WRITES power_dpm_force_performance_level. That file is
# root-owned; the level is changed by the owner in his own interactive shell,
# and the script WAITS for the sysfs file to read the value it needs. Nothing
# here needs sudo and no password is ever stored (CLAUDE.md).
#
# dev0 is 0000:83:00.0 == /sys/class/drm/card1 on this box, established three
# ways and NOT by the card number (the card numbers are scrambled: card0=86:00.0,
# card1=83:00.0, card2=48:00.0):
#   1. the engine's own selection rule replayed against VK_EXT_pci_bus_info --
#      coli_vk_init() takes the highest-ranked device with a strict '>' , so with
#      three tied discrete GPUs it keeps enumeration index 0, which is 0000:83:00.0
#      (dev2=auto -> 0000:48:00.0, dev3=auto -> 0000:86:00.0);
#   2. the live gateway's VRAM ledger: 83:00.0 holds 22.85 GB (dense 3.78 GB +
#      KDA pool 624 MB + 1248 experts) while 48:00.0 and 86:00.0 hold exactly
#      24.19 GB each (1695 experts apiece) -- dev0 is the odd one and it is 83;
#   3. `coli_vk_matmul_multi()`, the only dense dispatch path and the one Q7's
#      `Q38_DENSE_GPU` uses, touches G (= dev0) exclusively.
#
# Usage on the rig:
#   ~/bench/q13_probe.sh                # both levels, waits for the owner between them
#   Q13_ORACLE128=1 ~/bench/q13_probe.sh   # + a 128-token greedy/TF oracle per level (+11 min)
#   Q13_WAIT_MIN=45 ...                 # how long to wait for the owner's sudo line
#
# Cost: ~12 min of engine time per level plus the Qwen warm; the owner's gateway
# is down for the whole run, so the script takes the rig lock and restarts the
# gateway only if it is the thing that stopped it (q7_lib.sh).
set -u

DEV0_PCI=${DEV0_PCI:-0000:83:00.0}
DEV2_PCI=${DEV2_PCI:-0000:48:00.0}
DEV3_PCI=${DEV3_PCI:-0000:86:00.0}
DEV0_SYS=/sys/bus/pci/devices/$DEV0_PCI
DPM=$DEV0_SYS/power_dpm_force_performance_level

WT=${WT:-/home/ronald/src/colibri}
OUT=~/bench/q13_out; mkdir -p "$OUT"
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
HIST_SRC="$QSNAP/.coli_usage"
BIN=${BIN:-$WT/c/qwen38-vk}
SH=${SH:-$WT/c/shaders}
WAIT_MIN=${Q13_WAIT_MIN:-45}
ORACLE128=${Q13_ORACLE128:-0}

. /home/ronald/bench/q7_lib.sh          # log(), rig lock, stop_gateway, q7_on_exit
trap q7_on_exit EXIT INT TERM HUP

q7_take_lock "q13-step0" 120 || exit 3
for e in qwen38 qwen38-vk; do pgrep -x "$e" >/dev/null && { log "REFUSED: $e running"; exit 1; }; done
pgrep -f "bench/qwen38-[v]k\." >/dev/null && { log "REFUSED: a bench qwen copy is running"; exit 1; }

# Q13 is a ZERO-CODE item: nothing is built and nothing is edited. The tree must
# be clean at the commit whose binary is in service, and the binary is the one
# already on disk -- a rebuild would change the served glm53's bytes for nothing.
git -C "$WT" status --short | grep -q . && { log "REFUSED: $WT is dirty -- Q13 measures the shipped binary"; exit 1; }
log "tree: $(git -C "$WT" log --oneline -1)"
[ -x "$BIN" ] || { log "REFUSED: no $BIN"; exit 1; }
log "binary  $(sha256sum "$BIN" | cut -c1-16)   shaders $(sha256sum "$SH/qmatmul.spv" | cut -c1-16)"

stop_gateway || exit 1

# ---- the prompt: q7_probe.sh's, so the buckets are read against the same text --
SHORT=/tmp/q13_prompt_short.txt
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
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/q13_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/q13_fincore.txt').read().splitlines()[1:]:
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

# ---- power / clock sampling (read-only sysfs, no sudo) ------------------------
# ONE python process, not a fork per field: a shell sampler at 4 Hz forks ~120
# processes a second and would steal from the 8 pinned engine threads on an
# 8-core box -- i.e. it would perturb the very number being measured.
pm_start(){ PM_CSV="$OUT/$1.pm.csv"; : > "$PM_CSV"
  python3 - "$PM_CSV" "$DEV0_PCI" "$DEV2_PCI" "$DEV3_PCI" <<'PY' &
import sys, time, glob, os
csv, pcis = sys.argv[1], sys.argv[2:]
paths=[]
for p in pcis:
    d="/sys/bus/pci/devices/"+p
    h=(glob.glob(d+"/hwmon/hwmon*")+[d])[0]
    paths.append([h+"/freq1_input", h+"/freq2_input", h+"/power1_average", d+"/gpu_busy_percent"])
def rd(f):
    try:
        with open(f) as fh: return fh.read().strip()
    except Exception: return "-1"
with open(csv,"w",buffering=1) as out:
    while True:
        row=[f"{time.time():.3f}"]
        for i,fs in enumerate(paths):
            row.append(f"dev{[0,2,3][i]}"); row += [rd(f) for f in fs]
        out.write(" ".join(row)+"\n")
        time.sleep(0.5)
PY
  PM_PID=$!; }
pm_stop(){ [ -n "${PM_PID:-}" ] && kill $PM_PID 2>/dev/null; wait $PM_PID 2>/dev/null; PM_PID=; }
pm_report(){ python3 - "$1" <<'PY'
import sys, statistics
rows=[l.split() for l in open(sys.argv[1]) if l.strip()]
if not rows: print("  [pm] no samples"); raise SystemExit
for k,off in (("dev0",2),("dev2",7),("dev3",12)):
    s=[int(r[off]) for r in rows]; m=[int(r[off+1]) for r in rows]
    p=[int(r[off+2]) for r in rows]; b=[int(r[off+3]) for r in rows]
    print(f"  [pm {k}] sclk med {statistics.median(s)/1e6:7.1f} max {max(s)/1e6:7.1f} MHz | "
          f"mclk med {statistics.median(m)/1e6:7.1f} max {max(m)/1e6:7.1f} MHz | "
          f"power med {statistics.median(p)/1e6:6.1f} max {max(p)/1e6:6.1f} W | "
          f"busy med {statistics.median(b):3.0f} max {max(b):3.0f} %  n={len(rows)}")
PY
}

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto

# run <tag> <n_new> [KEY=VAL ...]
run(){
  local tag="$1" nn="$2"; shift 2
  cp -f "$HIST_SRC" /tmp/q13_hist.bin
  log "--- $tag  n_new=$nn  $*"
  pm_start "$tag"
  local t0=$(date +%s)
  env SNAP="$QSNAP" COLI_USAGE=/tmp/q13_hist.bin N_NEW="$nn" NOSTREAM=1 \
      COLI_VK_SHADERS="$SH" \
      Q38_TF=1 Q38_TF_DUMP="$OUT/$tag.tf.f32" DUMP="$OUT/$tag.last.f32" \
      "$@" "$BIN" 512 8 "$SHORT" > "$OUT/$tag.log" 2>&1
  local rc=$?
  pm_stop
  log "  rc=$rc  wall=$(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { log "REFUSED: $tag exited $rc"; tail -25 "$OUT/$tag.log"; exit 1; }
  grep -E "Vulkan tier preloaded|Q7 dense|^Speed:|placement" "$OUT/$tag.log" | sed 's/^/  /'
  pm_report "$OUT/$tag.pm.csv"
}

# ---- one level's worth of measurement ----------------------------------------
idle_sample(){   # 30 s of no-engine idle at the current level: the gate's "idle power draw"
  log "--- idle sample (30 s, no engine) at level=$(cat $DPM)"
  pm_start "idle-$1"; sleep 30; pm_stop; pm_report "$OUT/idle-$1.pm.csv"
}
level_phase(){   # $1 = tag prefix (auto|high)
  idle_sample "$1"
  for r in 1 2 3; do run "$1-m1-$r" 32 COLI_TIMERS=1 Q38_DENSE_GPU=1; done
  for r in 1 2 3; do run "$1-m7-$r" 32 COLI_TIMERS=1 Q38_DENSE_GPU=7; done
  [ "$ORACLE128" = 1 ] && run "$1-oracle" 128 Q38_DENSE_GPU=7
  return 0
}
wait_for_level(){  # $1 = wanted value
  local want="$1" i=0 now
  now=$(cat "$DPM")
  [ "$now" = "$want" ] && { log "level already $want"; return 0; }
  echo
  echo "==================================================================="
  echo " Q13 needs dev0 ($DEV0_PCI) at power_dpm_force_performance_level=$want"
  echo " Run THIS, in your own shell, on the rig:"
  echo
  echo "     echo $want | sudo tee $DPM"
  echo
  echo " (equivalently: echo $want | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level)"
  echo " The script is polling that file and continues by itself. Nothing else"
  echo " is touched; the other two cards stay at auto."
  echo "==================================================================="
  while [ $i -lt $((WAIT_MIN*6)) ]; do
    now=$(cat "$DPM"); [ "$now" = "$want" ] && { log "dev0 level is now $want"; sleep 5; return 0; }
    i=$((i+1)); sleep 10
  done
  log "REFUSED: dev0 still at '$now' after ${WAIT_MIN}m -- not measuring"; return 3
}

log "dev0=$DEV0_PCI level=$(cat $DPM)  dev2=$DEV2_PCI level=$(cat /sys/bus/pci/devices/$DEV2_PCI/power_dpm_force_performance_level)  dev3=$DEV3_PCI level=$(cat /sys/bus/pci/devices/$DEV3_PCI/power_dpm_force_performance_level)"

# Q13_ORDER: the levels to measure, in order. Default is the natural one, but on
# 2026-09-12 the owner had already set `high` before the run started, so the run
# was taken as "high auto" -- which also means the LAST thing the script waits
# for is the revert, and the card is back at its default before the gateway
# comes up. The order is recorded with the numbers; it is a confound like any
# other and three fresh-process repeats a side is what bounds it.
ORDER=${Q13_ORDER:-"auto high"}
for lvl in $ORDER; do
  wait_for_level "$lvl" || exit 3
  level_phase "$lvl"
done
# Q13_FINAL_AUTO=0 when the two legs are run as two invocations (gateway up in
# between, so the owner's service is not down while a human is being waited on):
# the revert is then chased outside the script, with nothing stopped.
if [ "${Q13_FINAL_AUTO:-1}" = 1 ]; then
  [ "$(cat $DPM)" = auto ] || wait_for_level auto || log "WARNING: dev0 NOT back at auto -- TELL THE OWNER"
else
  [ "$(cat $DPM)" = auto ] || log "NOTE: dev0 is at '$(cat $DPM)', NOT auto -- the revert is owed"
fi

# ---- the table ---------------------------------------------------------------
# Q13_CMP: which levels the table and the oracle compare, first one is the
# reference. Defaults to the two the item started with; "auto manual" is the
# narrower mclk-only variant, whose runs land in the same output directory and
# are read against the same auto leg.
CMP=${Q13_CMP:-"auto high"}
python3 - "$OUT" $CMP <<'PY'
import sys, os, glob, statistics
D = sys.argv[1]; LEVELS = sys.argv[2:] or ["auto", "high"]
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
    if not bs: return {}, 0
    keys = set().union(*map(set, bs))
    return {k: statistics.median([b[k] for b in bs if k in b]) for k in keys}, len(bs)
rows = ["token", "dn-proj", "qsa-proj", "lm-head", "vk-dense", "vk-take", "vk-issue",
        "dense-matmul", "deltanet", "attention", "moe", "cpu-experts", "gr-read", "shared", "dn-recur"]
for mask in ("m1", "m7"):
    data = [group(f"{l}-{mask}-*") for l in LEVELS]
    print(f"\n=== Q38_DENSE_GPU={'1' if mask=='m1' else '7'}   " +
          ", ".join(f"{d[1]} {l} repeats" for l, d in zip(LEVELS, data)))
    w = max(len(r) for r in rows) + 2
    print("bucket".ljust(w) + "".join(l.rjust(12) for l in LEVELS) +
          "".join(("d:" + l).rjust(12) for l in LEVELS[1:]))
    for r in rows:
        vals = [d[0].get(r) for d in data]
        line = r.ljust(w) + "".join((f"{v:12.3f}" if v is not None else "".rjust(12)) for v in vals)
        for v in vals[1:]:
            line += (f"{v-vals[0]:+12.3f}" if (v is not None and vals[0] is not None) else "".rjust(12))
        print(line)
PY

# ---- the oracle: a DPM level changes no arithmetic ---------------------------
set -- $CMP; REF=$1; shift
log "--- oracle: teacher-forcing and last-token dumps, $REF vs $*"
for lvl in "$@"; do
  for mask in m1 m7; do
    for r in 1 2 3; do
      a=$(sha256sum "$OUT/$REF-$mask-$r.tf.f32" 2>/dev/null | cut -c1-16)
      h=$(sha256sum "$OUT/$lvl-$mask-$r.tf.f32" 2>/dev/null | cut -c1-16)
      al=$(sha256sum "$OUT/$REF-$mask-$r.last.f32" 2>/dev/null | cut -c1-16)
      hl=$(sha256sum "$OUT/$lvl-$mask-$r.last.f32" 2>/dev/null | cut -c1-16)
      [ -n "$a" ] && [ -n "$h" ] && [ "$a" = "$h" ] && [ "$al" = "$hl" ] \
        && v=IDENTICAL || v="DIFFERS  tf $a/$h  last $al/$hl"
      echo "  $lvl $mask r$r: $v"
    done
  done
done
log "q13 step 0 done; logs in $OUT"
