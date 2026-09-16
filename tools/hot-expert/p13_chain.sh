#!/bin/bash
# p13_chain.sh -- P13 bisect: for each candidate binary, run the 564-token
# X2 packet at COLI_KDA_GPU=2 and COLI_KDA_GPU=0 and gate_compare the
# teacher_forcing lines. IDENTICAL = good (this sha does not carry the
# defect), DIFFERS = bad (the CLI/=2 prefill path is already broken here).
#
# Modelled on x2_iso_chain.sh (§X2, ~/bench/x2_iso_out/): same stop-gateway /
# wait_no_engine / warm+resident / trap-restarts-gateway / accept_live.sh
# shape, generalised to an arbitrary list of candidates instead of three
# fixed binaries, because P13 bisects between G12's landing (dd8fc3c) and
# the pristine serving binary over ~30 first-parent commits touching the
# KDA prefill path (c/glm53.c, c/backend_vulkan.c, c/shaders,
# c/sparse_index.h) and needs to test an arbitrary subset per lock window.
#
# Usage:
#   p13_chain.sh 'label1=/path/to/binary1=/path/to/shaders1' 'label2=...' ...
#
# For each candidate: warm+resident check (fincore >= 90%, PROFILE_MIN_RESIDENT
# to override), then two CLI runs on the fixed 564-token packet
# (x2_packet_450.txt), COLI_KDA_GPU=2 then COLI_KDA_GPU=0, then
# gate_compare on the teacher_forcing line; a DIFFERS pair also gets the
# first divergent position (0-based) via a small python diff, so the Opus
# fix has a concrete location to start from.
#
# Budget: ~7 min/candidate (two prefill-only runs on one packet); batch
# 2-3 candidates per window (run_chain.sh call), windows <= 30 min.
set -u
TAG=p13$(date +%m%d%H%M)
OUT=~/bench/p13_out; mkdir -p "$OUT"
SRC_P13=~/src/colibri-p13
HERE="$SRC_P13/tools/hot-expert"
M="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
FLOOR="${PROFILE_MIN_RESIDENT:-90}"

RESTART_GATEWAY=0

start_gateway() {
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 90); do
    sleep 10
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
             http://127.0.0.1:8081/v1/models 2>/dev/null) || true
    [ "${code:-}" = 200 ] && break
  done
  echo "[p13] gateway back, /v1/models=${code:-?}"
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[p13] stopping the owner's gateway for the duration of this run"
    pkill -f "openai_[s]erver.py" || true
    RESTART_GATEWAY=1
  fi
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_engine || return 1
}

wait_no_engine() {
  for _ in $(seq 1 120); do
    pgrep -x glm53 >/dev/null 2>&1 || pgrep -x qwen38 >/dev/null 2>&1 || pgrep -x qwen38-vk >/dev/null 2>&1 || return 0
    sleep 2
  done
  echo "[p13] FATAL: an engine is still alive after 240s"; return 1
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_engine || true
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "[p13] accept_live.sh"
  "$HOME/src/colibri/tools/hot-expert/accept_live.sh"
  echo "=== p13_chain exit rc=$rc tag=$TAG $(date -Is)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

resid() {                    # resid <tag>
  fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors > "/tmp/p13_fincore.$$.txt"
  python3 - "$1" "/tmp/p13_fincore.$$.txt" "$FLOOR" <<'PY'
import sys
PAGE=4096; tot=res=0; n=0
for line in open(sys.argv[2]).read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
pct = res/tot*100 if tot else 0
floor = float(sys.argv[3])
print(f"[p13 resid {sys.argv[1]}] shards={n} resident={pct:.2f}% floor={floor}")
sys.exit(0 if (n==62 and pct >= floor) else 1)
PY
  rc=$?
  rm -f "/tmp/p13_fincore.$$.txt"
  return $rc
}

echo "=== p13_chain $TAG $(date -Is)"
for _eng in glm53 qwen38 qwen38-vk; do
  if pgrep -x "$_eng" >/dev/null 2>&1; then echo "[p13] $_eng is running -- refusing"; exit 1; fi
done

[ $# -ge 1 ] || { echo "[p13] usage: p13_chain.sh 'label=binary=shaders' ..."; exit 2; }

stop_gateway || exit 1

echo "--- warming the model (assume warm; cat is cheap if so)"
cat "$M"/*.safetensors > /dev/null 2>&1 || true
resid pre || { echo "[p13] FATAL: model not >= ${FLOOR}% resident before the chain"; exit 1; }

PACKET=$(cat "$HERE/x2_packet_450.txt")

run() {                      # run <tag> <binary> <shaders> <kda_gpu>
  local tag="$1" bin="$2" shaders="$3" kda="$4"
  for _eng in glm53 qwen38 qwen38-vk; do
    pgrep -x "$_eng" >/dev/null 2>&1 && { echo "[p13] $_eng already running before $tag -- refusing"; return 9; }
  done
  echo "[p13] $(date +%H:%M:%S) run $tag  bin=$bin  KDA_GPU=$kda"
  cp -f "$HOME/.glm53_explain.bin" "/tmp/p13_hist_$tag.bin"
  rm -rf "/tmp/p13_ckpt_$tag"; mkdir -p "/tmp/p13_ckpt_$tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_USAGE_PATH="/tmp/p13_hist_$tag.bin"
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="/tmp/p13_ckpt_$tag"
    export GLM53_VERBOSE=1 COLI_KDA_GPU="$kda"
    "$bin" --model "$M" --prompt "$PACKET" --greedy 0 ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  echo "[p13] $(date +%H:%M:%S) run $tag rc=$rc  $(grep -c ^teacher_forcing "$OUT/$tag.out") tf-line(s)"
  wait_no_engine || true
  return $rc
}

first_diverge() {            # first_diverge <fileA> <fileB>
  python3 - "$1" "$2" <<'PY'
import sys
def toks(path):
    for line in open(path):
        if line.startswith('teacher_forcing'):
            return line.split()[1:]
    return []
a, b = toks(sys.argv[1]), toks(sys.argv[2])
n = min(len(a), len(b))
for i in range(n):
    if a[i] != b[i]:
        print(f"    first divergent position: {i} (A={a[i]} B={b[i]}, {n} positions compared)")
        break
else:
    print(f"    no token-level divergence in the first {n} positions (length mismatch: A={len(a)} B={len(b)})")
PY
}

. "$HERE/gate_lib.sh"

echo
echo "=== P13 bisect: $TAG ==="
declare -A RESULT
for spec in "$@"; do
  label="${spec%%=*}"; rest="${spec#*=}"
  bin="${rest%%=*}"; shaders="${rest#*=}"
  bin="${bin/#\~/$HOME}"; shaders="${shaders/#\~/$HOME}"
  [ -x "$bin" ] || { echo "[p13] REFUSED: $bin missing or not executable"; RESULT[$label]="REFUSED(missing bin)"; continue; }
  [ -d "$shaders" ] || { echo "[p13] REFUSED: $shaders missing"; RESULT[$label]="REFUSED(missing shaders)"; continue; }
  sha=$(sha256sum "$bin" | cut -c1-16)
  echo "--- candidate $label  bin=$bin ($sha)  shaders=$shaders"
  run "${label}_gpu2" "$bin" "$shaders" 2
  run "${label}_gpu0" "$bin" "$shaders" 0
  gate_compare "$label: GPU=2 vs GPU=0" "$OUT/${label}_gpu2.out" "$OUT/${label}_gpu0.out" '^teacher_forcing'
  gc_rc=$?
  case $gc_rc in
    0) RESULT[$label]="IDENTICAL (good)" ;;
    1) RESULT[$label]="DIFFERS (bad)"; first_diverge "$OUT/${label}_gpu2.out" "$OUT/${label}_gpu0.out" ;;
    *) RESULT[$label]="REFUSED (gate_compare rc=$gc_rc)" ;;
  esac
done

echo
echo "=== P13 bisect summary ($TAG) ==="
for spec in "$@"; do
  label="${spec%%=*}"
  printf '  %-20s %s\n' "$label" "${RESULT[$label]:-?}"
done

RESTART_GATEWAY=1
exit 0
