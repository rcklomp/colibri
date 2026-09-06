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
export COLI_VK_SHADERS="$(dirname "$BIN")/shaders"
export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
export COLI_KDA_GPU=${COLI_KDA_GPU:-2}
export COLI_TIMERS=1 GLM53_VERBOSE=1 GLM53_PREFILL_CHUNK=$CHUNK
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
"$BIN" --model "$M" --prompt "$P" --greedy 0 --logits 2>&1 | tee -a "$LOG" \
  | grep -E "^\[OPTIME\]|^\[PROF\]|^prefill|caricamento|teacher_forcing" | cut -c1-200
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
PY
