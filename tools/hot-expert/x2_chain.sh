#!/bin/bash
# x2_chain.sh -- FRANKEN-ENGINE-PLAN X2: a per-position KL oracle for glm53.
#
# Modelled on context_ladder_chain.sh's shape (stop gateway -> wait for the
# engine to die -> warm the model, assert residency -> do the work -> restart
# the gateway on EVERY exit path). Launch only through run_chain.sh, which
# takes the rig lock and refuses a second chain rather than interleaving:
#
#   setsid nohup ~/src/colibri-x2/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-x2/tools/hot-expert/x2_chain.sh \
#       >> ~/bench/x2_run.log 2>&1 < /dev/null &
#
# This chain never touches the SERVING tree (~/src/colibri). It works
# entirely inside the dedicated clone ~/src/colibri-x2 (perf/x2-kl-oracle,
# `git clone ~/src/colibri ~/src/colibri-x2 -b perf/x2-kl-oracle` once,
# `git -C ~/src/colibri-x2 pull --ff-only` after every push). The gateway
# still has to come down for the duration, because glm53 pins 3 GPUs and
# 1695/1695 of expert VRAM and only one engine may run at a time.
#
# TWO binaries are built here, both from the dedicated clone:
#   glm53.x2cand -- perf/x2-kl-oracle at its tip. Carries GLM53_LOGIT_DUMP_ALL
#     plus every knob already in the tree (COLI_KDA_GPU, GLM53_EXPERTS_CPU,
#     GLM53_I3_SIM) -- three of the four oracle pairs run on this one binary,
#     env-knob only.
#   glm53.x2g14 -- commit be95eb5 (the G14 int8-activation kernel,
#     `GLM53_I4_INT8`, later removed at b216dae because it was rejected) with
#     ONLY this chain's dump patch (the diff of c/glm53.c between the branch
#     base 116d2f5 and perf/x2-kl-oracle's tip) applied on top, so it can
#     write the same per-position dump. That patch applies with a clean
#     `patch -p1` (checked on the Mac before this chain was written) because
#     the insertion point -- just above main() -- has not moved in a
#     conflicting way in the 34 commits between be95eb5 and 116d2f5. Nothing
#     else about be95eb5 is touched; GLM53_I4_INT8=0 (the default) there is
#     documented in the record as bit-identical to pristine, so that binary
#     with the knob OFF stands in for "pristine" in the G14 pair without a
#     second checkout. This binary is built, used, and discarded -- it never
#     leaves ~/src/colibri-x2 and is not part of what gets pushed.
#
# The four oracle pairs (env-var pairs unless noted), all on ONE fixed
# 450-token packet (tools/hot-expert/x2_packet_450.txt, sliced deterministically
# from ROME-3x7900XTX-2026-09-04.md's own text -- see that script's header):
#   g12   glm53.x2cand  COLI_KDA_GPU=0        vs  COLI_KDA_GPU=2
#   g14   glm53.x2g14   GLM53_I4_INT8=0       vs  GLM53_I4_INT8=1
#   g15   glm53.x2cand  GLM53_EXPERTS_CPU=1   vs  GLM53_EXPERTS_CPU=1 GLM53_I3_SIM=1
#   clamp glm53.x2cand  GLM53_EXPERTS_CPU=2   vs  GLM53_EXPERTS_CPU=1
#         (isolates the swiglu clamp exactly as §G15 did: =2 forces CPU
#         placement with the clamp OFF, matching what the GPU kernel computes;
#         =1 is CPU placement with the clamp ON. This is the pair that
#         reproduces CLAUDE.md's own "6 of 42 and 8 of 1232" figures.)
#
# Every run is --greedy 0 (prefill only, teacher_forcing's positions are the
# ones the dump and the KL oracle both need) on the same packet, 100% of the
# generated dump feeding tools/hot-expert/kl_compare.py.
set -u
TAG=x2$(date +%m%d%H%M)
OUT=~/bench/x2_out; mkdir -p "$OUT"
SRC=~/src/colibri-x2
HERE="$SRC/tools/hot-expert"
M="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"
BRANCH=perf/x2-kl-oracle
BASE=116d2f5
G14SHA=be95eb5
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)

RESTART_GATEWAY=0

start_gateway() {
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 90); do
    sleep 10
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
             http://127.0.0.1:8081/v1/models 2>/dev/null) || true
    [ "${code:-}" = 200 ] && break
  done
  echo "[x2] gateway back, /v1/models=${code:-?}  gateway=$(pgrep -f "openai_[s]erver.py" | wc -l) engine=$(pgrep -x glm53 | wc -l)"
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[x2] stopping the owner's gateway for the duration of this run"
    pkill -f "openai_[s]erver.py" || true
    RESTART_GATEWAY=1
  fi
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 120); do pgrep -x glm53 >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[x2] FATAL: a glm53 is still alive after 240s"; return 1
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  # Kill glm53 ONLY when the gateway is about to be restarted here. On the normal
  # path the body has already restarted it (RESTART_GATEWAY=0) and that glm53 IS the
  # owner's served engine: killing it left openai_server.py up with a defunct engine,
  # /v1/models=200 and every chat a 500 "engine dispatcher stopped" (2026-09-20).
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    pkill -9 -x glm53 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "=== x2_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== results: $OUT/$TAG.*"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== x2_chain $TAG $(date -Is)"

for _eng in qwen38 qwen38-vk; do
  if pgrep -x "$_eng" >/dev/null 2>&1; then echo "[x2] $_eng is running -- refusing"; exit 1; fi
done

stop_gateway || exit 1

echo "[x2] refreshing $SRC"
if [ ! -d "$SRC/.git" ]; then
  git clone "$HOME/src/colibri" "$SRC" -b "$BRANCH" || exit 1
else
  git -C "$SRC" checkout -q "$BRANCH" && git -C "$SRC" pull --ff-only || exit 1
fi
git -C "$SRC" log -1 --oneline

echo "--- warming the model (one model at a time; 182 GiB)"
cat "$M"/*.safetensors > /dev/null 2>&1 || true

resid() {
  fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors > /tmp/x2_fincore.txt
  python3 - /tmp/x2_fincore.txt <<'PY'
import sys
PAGE = 4096
tot = res = 0
for line in open(sys.argv[1]).read().splitlines()[1:]:
    f = line.split()
    if len(f) < 3:
        continue
    size, r = int(f[-2]), int(f[-1])
    tot += -(-size // PAGE) * PAGE
    res += r
pct = res / tot * 100 if tot else 0.0
print(f"{pct:.1f}")
if pct < 90.0:
    sys.exit(1)
PY
}
PCT=$(resid) || { echo "[x2] REFUSED: page-cache residency $PCT% < 90%"; exit 1; }
echo "[x2] resident: $PCT%"

echo "[x2] building candidate ($BRANCH)"
( cd "$SRC" && make -C c glm53 VK=1 ) > "$OUT/build_cand.log" 2>&1 \
  || { echo "[x2] candidate build FAILED"; tail -40 "$OUT/build_cand.log"; exit 1; }
cp -f "$SRC/c/glm53" ~/bench/glm53.x2cand
echo "[x2] glm53.x2cand sha256=$(sha256sum ~/bench/glm53.x2cand | cut -c1-16)"

echo "[x2] building the G14 reference ($G14SHA + this chain's dump patch, dedicated clone only)"
git -C "$SRC" diff "$BASE" "$BRANCH" -- c/glm53.c > /tmp/x2_dump.patch
( cd "$SRC" \
  && git checkout -q --detach "$G14SHA" \
  && patch -p1 < /tmp/x2_dump.patch \
  && make -C c glm53 VK=1 ) > "$OUT/build_g14.log" 2>&1
RC=$?
if [ $RC -eq 0 ]; then
  cp -f "$SRC/c/glm53" ~/bench/glm53.x2g14
  echo "[x2] glm53.x2g14 sha256=$(sha256sum ~/bench/glm53.x2g14 | cut -c1-16)"
else
  echo "[x2] G14 reference build FAILED -- reporting the other three pairs only"
  tail -40 "$OUT/build_g14.log"
fi
( cd "$SRC" && git checkout -q -- c/glm53.c 2>/dev/null; git checkout -q "$BRANCH" )
echo "[x2] $SRC back on $(git -C "$SRC" rev-parse --abbrev-ref HEAD)"

PACKET=$(cat "$HERE/x2_packet_450.txt")

run() {                      # run <tag> <binary> <dumpfile> [ENV=V ...]
  local tag="$1" bin="$2" dump="$3"; shift 3
  echo "[x2] $(date +%H:%M:%S) run $tag  ($*)"
  cp -f "$HOME/.glm53_explain.bin" "/tmp/x2_hist_$tag.bin"
  rm -rf "/tmp/x2_ckpt_$tag"; mkdir -p "/tmp/x2_ckpt_$tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_USAGE_PATH="/tmp/x2_hist_$tag.bin"
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="/tmp/x2_ckpt_$tag"
    export GLM53_VERBOSE=1
    export GLM53_LOGIT_DUMP_ALL="$dump"
    for kv in "$@"; do export "${kv?}"; done
    "$bin" --model "$M" --prompt "$PACKET" --greedy 0 ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  local ntok
  ntok=$(awk '/^teacher_forcing/{print NF-1}' "$OUT/$tag.out")
  echo "[x2] $(date +%H:%M:%S) run $tag rc=$rc positions=$ntok dump=$(wc -c < "$dump" 2>/dev/null || echo 0) bytes"
  grep -E '^\[VK\] preload:|^\[VK\] preload dev' "$OUT/$tag.err" | tail -3
  return $rc
}

CAND=~/bench/glm53.x2cand
G14=~/bench/glm53.x2g14

run g12_ref  "$CAND" "$OUT/g12_ref.dump"  COLI_KDA_GPU=0
run g12_cand "$CAND" "$OUT/g12_cand.dump" COLI_KDA_GPU=2

run g15_ref  "$CAND" "$OUT/g15_ref.dump"  GLM53_EXPERTS_CPU=1
run g15_cand "$CAND" "$OUT/g15_cand.dump" GLM53_EXPERTS_CPU=1 GLM53_I3_SIM=1

run clamp_ref  "$CAND" "$OUT/clamp_ref.dump"  GLM53_EXPERTS_CPU=2
run clamp_cand "$CAND" "$OUT/clamp_cand.dump" GLM53_EXPERTS_CPU=1

if [ -f "$G14" ]; then
  run g14_ref  "$G14" "$OUT/g14_ref.dump"  GLM53_I4_INT8=0
  run g14_cand "$G14" "$OUT/g14_cand.dump" GLM53_I4_INT8=1
else
  echo "[x2] SKIPPING g14 pair -- glm53.x2g14 was not built"
fi

echo
echo "=== X2 KL oracle: G12 / G14 / G15 in order, plus the swiglu clamp ==="
. "$HERE/gate_lib.sh"
gate_kl "G12  COLI_KDA_GPU=2 vs 0"                       "$OUT/g12_ref.dump"   "$OUT/g12_cand.dump"
gate_kl "G14  GLM53_I4_INT8=1 vs 0 (be95eb5)"             "$OUT/g14_ref.dump"   "$OUT/g14_cand.dump"
gate_kl "G15  I3_SIM=1 vs EXPERTS_CPU=1 alone"            "$OUT/g15_ref.dump"   "$OUT/g15_cand.dump"
gate_kl "clamp  EXPERTS_CPU=1 vs =2 (swiglu clamp only)"  "$OUT/clamp_ref.dump" "$OUT/clamp_cand.dump"

echo
echo "[x2] restoring the gateway before accept_live.sh"
start_gateway
RESTART_GATEWAY=0

echo "[x2] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ACCEPT_RC=$?
echo "[x2] accept_live.sh rc=$ACCEPT_RC"

exit $ACCEPT_RC
