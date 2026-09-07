#!/bin/bash
# P0 — the prefill DIAGNOSTIC: where does one prompt token's prefill time go?
#
# Teacher-forcing CLI run (`--greedy 0`: the whole run is the prefill) with the
# per-op timers, on a fixed prompt file, at the chunk size the serve path uses
# (GLM53_PREFILL_CHUNK, default 128 = the engine's default). Prints the
# [OPTIME]/[PROF] split as ms per prompt token -- the table in
# PREFILL-ROADMAP-2026-09.md ("The diagnosis") -- and the teacher_forcing line
# so the same run doubles as the oracle input for prefill_gate.sh.
#
# This is a diagnosis instrument, not the gate: TTFT is gated on the serve
# path by ttft_serve.py. Use this to see WHICH bucket an item moved.
#
#   prefill_profile.sh <glm53-binary> <prompt-file> [tag] [chunk]
#
# Residency is re-warmed and asserted first (profile_run.sh's lesson: a fresh
# glm53 process evicts ~8% of the model; assert before EVERY run). Output goes
# to ~/bench/prefill_profile_<tag>.log; the parsed table to stdout.
set -u
BIN=${1:?binary}; PF=${2:?prompt file}; TAG=${3:-$(date +%H%M%S)}; CHUNK=${4:-128}
M=${SNAP:-$HOME/models/GLM-5.3-Flash-colibri-int4-g64}
LOG=$HOME/bench/prefill_profile_$TAG.log
mkdir -p "$HOME/bench"

if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running (one engine at a time)"; exit 3; fi

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
# shaders live in the repo's c/shaders, not beside the binary (a pristine copy
# in ~/bench ran without Vulkan on 2026-09-06 and produced CPU-only numbers)
HERE=$(cd "$(dirname "$0")" && pwd)
export COLI_VK_SHADERS="${COLI_VK_SHADERS:-$HERE/../../c/shaders}"
[ -f "$COLI_VK_SHADERS/qmatmul.comp" ] || { echo "REFUSED: no shaders at $COLI_VK_SHADERS -- the run would be CPU-only"; exit 3; }
export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
export COLI_KDA_GPU=${COLI_KDA_GPU:-2}
export COLI_TIMERS=1 GLM53_VERBOSE=1 GLM53_PREFILL_CHUNK=$CHUNK
# RP4: 0 = off (no query pool, no device extension, nothing in the command
# buffer); 1 = phase timestamps on the expert group; 2 = + per expert.
export COLI_VK_TIMESTAMPS=${COLI_VK_TIMESTAMPS:-0}
cp "$HOME/.glm53_explain.bin" /tmp/prefill_hist.$$.bin
export COLI_USAGE_PATH=/tmp/prefill_hist.$$.bin

resid() {
  fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors > /tmp/prefill_fincore.$$.txt
  python3 - "$1" "/tmp/prefill_fincore.$$.txt" <<'PY'
import sys
PAGE=4096; tot=res=0; short=0; n=0
for line in open(sys.argv[2]).read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
print(f'[resid {sys.argv[1]}] shards={n} resident={res/tot*100:.4f}% short={short}')
sys.exit(0 if (n==62 and short==0) else 1)
PY
}
warm() { find "$M" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done; }

warm
resid "$TAG-pre" || { echo "ABORT: not 100% resident after warm"; exit 1; }
P=$(cat "$PF")
echo "### prefill_profile tag=$TAG bin=$BIN prompt=$PF chunk=$CHUNK $(date -Is)" | tee "$LOG"
t0=$(date +%s.%N)
# Positional cap = expert-cache slots per layer, the same 512 the serve
# harness and tworeq.py use. Without it the CLI sizes the cache from
# MemAvailable and evicts the model under itself (p0self pristine run: 100% ->
# 91.7% during the prefill, 414 ms/token instead of the serve path's 171-196).
"$BIN" --model "$M" --prompt "$P" --greedy 0 --logits "${CAP:-512}" 2>&1 | tee -a "$LOG" \
  | grep -E "^\[OPTIME\]|^\[PROF\]|^\[VKTS|^prefill|caricamento|teacher_forcing" | cut -c1-200
echo "### wall(load+prefill)=$(echo "$(date +%s.%N) - $t0" | bc)s" | tee -a "$LOG"
resid "$TAG-post" || true
rm -f /tmp/prefill_hist.$$.bin /tmp/prefill_fincore.$$.txt

# ms per prompt token, per bucket
python3 - "$LOG" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r'prefill (\d+) token in ([\d.]+)s', s)
if not m: sys.exit("no 'prefill N token' line (GLM53_VERBOSE missing?)")
N, T = int(m.group(1)), float(m.group(2))
def g(p):
    r = re.search(p, s); return float(r.group(1)) if r else float('nan')
rows = [("TOTAL prefill", T), ("layers", g(r'layers=([\d.]+)s')),
        ("ffn_moe", g(r'ffn_moe=([\d.]+)s')), ("  cpu experts", g(r'cpu=([\d.]+)s\(n=')),
        ("  eg (gpu)", g(r'eg=([\d.]+)s\(disp')), ("  router", g(r'router=([\d.]+)s')),
        ("  shared", g(r'shared=([\d.]+)s')), ("kda", g(r'kda=([\d.]+)s')),
        ("  kda.proj", g(r'proj=([\d.]+)s')), ("  kda.step", g(r'step=([\d.]+)s')),
        ("  kda.ko", g(r'ko=([\d.]+)s')), ("mla", g(r'mla=([\d.]+)s')),
        ("  mla.attn", g(r'attn=([\d.]+)s')), ("hc+norm", g(r'hc\+norm=([\d.]+)s')),
        ("head", g(r'head=([\d.]+)s'))]
print(f"\n{N} prompt tokens in {T:.1f}s = {N/T:.2f} tok/s prefill ({T/N*1000:.1f} ms/token)")
print(f"{'bucket':16s} {'ms/token':>9s} {'share':>6s}")
for name, v in rows:
    print(f"{name:16s} {v/N*1000:9.1f} {100*v/T:5.0f}%")
eg = re.search(r'eg=[\d.]+s\(disp=(\d+) experts=(\d+)\) cpu=[\d.]+s\(n=(\d+)\)', s)
if eg: print(f"gpu expert-calls {int(eg.group(2))/N:.1f}/token, cpu (token,expert) pairs {int(eg.group(3))/N:.1f}/token")

# RP4: the GPU-side row group, printed only when COLI_VK_TIMESTAMPS was set.
# It sits next to the CPU-side "eg (gpu)" bucket above: that one is the wall
# time the engine blocks, this one is what the GPU actually did in it.
ts  = re.findall(r'^\[VKTS\] dev(\d+) chunks=(\d+) experts=(\d+) gate_up=([\d.-]+) ms down=([\d.-]+) ms '
                 r'per_expert=([\d.-]+) us queue_lat=([\d.-]+) ms fence_tail=([\d.-]+) ms', s, re.M)
tsp = {m[0]: m for m in re.findall(r'^\[VKTS\+\] dev(\d+) level=(\d+) busy=([\d.-]+) ms \(([\d.-]+)% of submit->fence '
                                   r'([\d.-]+) ms\) rows=([\d.-]+) \(([\d.-]+)/expert\) cpu_overlap=([\d.-]+) ms '
                                   r'join_wait=([\d.-]+) ms experts/chunk=([\d.-]+) gpu_late=([\d.-]+) ms in (\d+)/(\d+)', s, re.M)}
if ts:
    print(f"\n[VKTS] GPU-side expert group, ms per prompt token (N={N})")
    print(f"{'dev':>4} {'chunks':>7} {'experts':>8} {'gate_up':>8} {'down':>7} {'busy':>7} "
          f"{'queue':>7} {'span':>7} {'cpu_ovl':>8} {'join':>7} {'gpu_late':>9} {'us/exp':>7} {'r/exp':>6} {'busy%':>6}")
    for d, ch, ex, gu, dn, pe, ql, ft in ts:
        p2 = tsp.get(d)
        busy = float(gu) + float(dn)
        span = float(p2[4]) if p2 else float('nan')
        ovlp = float(p2[7]) if p2 else float('nan')
        join = float(p2[8]) if p2 else float('nan')
        rpe  = float(p2[6]) if p2 else float('nan')
        late = float(p2[10]) if p2 else float('nan')
        lat_n = f"{p2[11]}/{p2[12]}" if p2 else "?"
        print(f"{d:>4} {int(ch):>7} {int(ex):>8} {float(gu)/N:>8.2f} {float(dn)/N:>7.2f} "
              f"{busy/N:>7.2f} {float(ql)/N:>7.2f} {span/N:>7.2f} {ovlp/N:>8.2f} {join/N:>7.2f} "
              f"{late/N:>9.3f} {float(pe):>7.1f} {rpe:>6.2f} {(100*busy/span if span else 0):>5.1f}%")
        print(f"{'':>4} {'':>7} {'':>8} (ms/prompt-token; gpu_late in {lat_n} chunks -- "
              f"the only part of the eg wait the GPU owns)")
    tot = sum(float(m[3]) + float(m[4]) for m in ts)
    latot = sum(float(v[10]) for v in tsp.values()) if tsp else 0.0
    print(f"  three devices: GPU busy {tot/N:.2f} ms/token summed (they run concurrently), "
          f"gpu_late {latot/N:.3f} ms/token summed")
rows = re.findall(r'^\[VKTSROW\] dev(\d+) rows=(\d+)\.\.(\d+) n=(\d+) gate_up=([\d.-]+) us down=([\d.-]+) us total=([\d.-]+) us', s, re.M)
if rows:
    print(f"\n[VKTSROW] per-expert GPU time by row count (level 2; dispatches serialized)")
    print(f"{'dev':>4} {'rows':>9} {'n':>8} {'gate_up us':>11} {'down us':>9} {'total us':>9} {'us/row':>8}")
    for d, lo, hi, n, gu, dn, to in rows:
        mid = (int(lo) + int(hi)) / 2
        print(f"{d:>4} {lo+'..'+hi:>9} {int(n):>8} {float(gu):>11.1f} {float(dn):>9.1f} {float(to):>9.1f} {float(to)/mid:>8.1f}")
PY
