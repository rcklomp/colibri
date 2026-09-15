#!/bin/bash
# q9_probe.sh -- roadmap item Q9 step 1: the chunk probe, Qwen3.8 analog of §SPEC-PROBE.
#
# Every technique in the AngelSpec/MTP family verifies a BLOCK of K tokens in
# one forward. The ceiling on this box is set by how that verify pass scales
# with K, and it is measurable with NO drafter: run a FIXED long prompt with
# prefill batching OFF (row-at-a-time, K=1) and ON (chunked, distinct-expert
# dedup), and read [OPTIME] placement. GLM's §SPEC-PROBE (record §SPEC-PROBE)
# found the CPU expert count FLAT (35 252 at every K from 1 to 16) -- cold
# experts are never shared across adjacent tokens -- which is the refusal line
# this probe checks for Qwen. Flat = MTP refused (~1.2x cap, not the 5-8 day
# build); a real dedup = a Fable spec is worth it.
#
# Q38_PREFILL_BATCH is a BOOL (qwen38 has no integer chunk knob like GLM's
# GLM53_PREFILL_CHUNK): 0 = row-at-a-time (K=1), 1 = bounded chunk (dedup, up
# to Q38_PREFILL_BATCH_ROWS=32). The probe reports the prefill CPU expert count
# for both states; the difference IS the union effect.
#
# Cost: one Qwen warm + two short runs (~20 min of rig time). The gateway is
# down for the run, so the script takes the rig lock and restarts it on exit
# (q7_lib.sh). No sudo, no code change, no numerics risk (both batch paths are
# bit-identical -- Q38_PREFILL_BATCH=0 is the shipped A/B diagnosis knob).
set -u

WT=${WT:-/home/ronald/src/colibri}
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
BIN=${BIN:-$WT/c/qwen38-vk}
SH=${SH:-$WT/c/shaders}
OUT=~/bench/q9_out; mkdir -p "$OUT"

. /home/ronald/bench/q7_lib.sh          # log(), rig lock, stop_gateway, q7_on_exit
trap q7_on_exit EXIT INT TERM HUP

q7_take_lock "q9-step1" 120 || exit 3
for e in qwen38 qwen38-vk; do pgrep -x "$e" >/dev/null && { log "REFUSED: $e running"; exit 1; }; done
[ -x "$BIN" ] || { log "REFUSED: no $BIN"; exit 1; }

log "tree: $(git -C "$WT" log --oneline -1)"
log "binary  $(sha256sum "$BIN" | cut -c1-16)   shaders $(sha256sum "$SH/qmatmul.spv" | cut -c1-16)"
stop_gateway || exit 1

# ---- GLM out of the page cache first (Qwen is 173 GiB, RAM 247 GiB; the two
#      models do not fit together). fadvise, no sudo. ------------------------
python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("[evict] GLM ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY

# ---- a long deterministic prompt: the SPEC-PROBE shape (repetition is fine,
#      we measure routing breadth + batching, not output quality) ------------
P=/tmp/q9_prompt.txt
python3 -c "print((open('$HOME/bench/prompt_glm.txt').read().strip()+' ')*6)" > "$P"

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
export COLI_VK_SHADERS="$SH"
export COLI_TIMERS=1

warm(){ find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done; }
resid(){
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/q9_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/q9_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
print(f'[resid $1] shards={n} resident={res/tot*100:.4f}% short={short}')
assert n==131 and short==0, 'NOT 100% RESIDENT'
"
}

run(){
  local batch="$1"
  cp -f "$QSNAP/.coli_usage" /tmp/q9_hist.bin
  log "--- Q38_PREFILL_BATCH=$batch  ($([ "$batch" = 0 ] && echo 'K=1 row-at-a-time' || echo 'chunked dedup'))"
  local t0=$(date +%s)
  env SNAP="$QSNAP" COLI_USAGE=/tmp/q9_hist.bin N_NEW=1 NOSTREAM=1 \
      Q38_PREFILL_BATCH="$batch" \
      "$BIN" 512 8 "$P" > "$OUT/b$batch.log" 2>&1
  local rc=$?
  log "  rc=$rc  wall=$(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] || { tail -30 "$OUT/b$batch.log"; exit 1; }
  grep -E "resident weights|Vulkan tier|placement" "$OUT/b$batch.log" | sed 's/^/  /'
}

warm
resid boot assert || { log "REFUSED: Qwen not fully resident"; exit 1; }
run 0
warm
resid b1 assert || { log "REFUSED: Qwen not fully resident (pre-b1)"; exit 1; }
run 1

# ---- the table: prefill CPU expert count at K=1 (batch 0) vs chunked (batch 1).
#      placement prints total (prefill+decode) and decode; prefill = total - decode.
python3 - "$OUT" <<'PY'
import sys, os, re, glob
D = sys.argv[1]
def bank(p):
    out = {}
    for line in open(p, errors="replace"):
        m = re.search(r"placement\s+(total|decode):\s+gpu=(\d+)\s+cpu=(\d+)", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out
rows = {}
for b in ("b0", "b1"):
    p = os.path.join(D, b + ".log")
    if not os.path.exists(p):
        print(f"{b}: MISSING"); continue
    bk = bank(p)
    rows[b] = bk
    tot_g, tot_c = bk.get("total", (0, 0))
    dec_g, dec_c = bk.get("decode", (0, 0))
    print(f"{b} (Q38_PREFILL_BATCH={b[1]})  total: gpu={tot_g} cpu={tot_c}  decode: gpu={dec_g} cpu={dec_c}  => prefill cpu={tot_c-dec_c}")
if "b0" in rows and "b1" in rows:
    p0 = rows["b0"].get("total", (0,0))[1] - rows["b0"].get("decode", (0,0))[1]
    p1 = rows["b1"].get("total", (0,0))[1] - rows["b1"].get("decode", (0,0))[1]
    d = p1 - p0
    pct = 100.0*d/p0 if p0 else 0.0
    print(f"\nprefill CPU experts:  K=1 -> {p0},  chunked -> {p1},  delta {d:+d} ({pct:+.2f}%)")
    print("VERDICT: " + ("FLAT -- cold experts not shared, MTP refused (~1.2x cap)" if abs(d) <= max(2, 0.01*p0)
          else ("DEDUP -- verify-of-K amortises the miss, MTP worth a Fable spec" if d < 0
                else "INCREASED -- investigate (should not happen)")))
PY

log "q9 step 1 (chunk probe) done; logs in $OUT"
