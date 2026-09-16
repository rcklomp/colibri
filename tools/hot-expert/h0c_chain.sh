#!/bin/bash
# h0c_chain.sh -- H0's last window: is it MTP or Redline?
#
# "forward failed" with no HipError reads like an internal check on the
# forward's output failing (kernels run, values wrong), not a launch
# failure. Two default-on paths could be misbehaving on this ROCm: the MTP
# speculator and Redline's retained replay. Try, in order, stopping at the
# first that answers: (a) MTP off, Redline default; (b) MTP off and Redline
# off. --kv-backend contiguous throughout (b09160707's variant a already
# showed vmm vs contiguous makes no difference). Knobs from docs/env-vars.md:
# HIPFIRE_MTP_MODE (auto/3 default; off disables), HIPFIRE_REPLAY_BACKEND
# (hip/off/shadow/auto; off disables the retained-replay/Redline route).
#
# Fixes h0b_chain.sh's race: waits for the previous hipfire pid to die AND
# card0 VRAM < 1 GB before launching the next variant.
#
# Budget: <= 15 min, then H0 closes as O3 regardless.
set -u
TAG=h0c_$(date +%m%d%H%M)
OUT=~/bench/h0_out; mkdir -p "$OUT"
HERE=~/src/colibri-h0/tools/hot-expert
HIPFIRE_BIN=~/src/hipfire/target/release/hipfire
HIPFIRE_MODEL="qwen3.6:35b-a3b-mq4r"
HIPFIRE_PORT=${HIPFIRE_PORT:-11436}
HIPFIRE_DEV=${HIPFIRE_DEV:-1}
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
QUESTION="Continue summarising the notes above in a few sentences, without repeating what you already said."
export CPLUS_INCLUDE_PATH=/usr/include/c++/15:/usr/include/x86_64-linux-gnu/c++/15
export LD_LIBRARY_PATH="$HOME/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_PATH=/opt/rocm-6.2.0 HIP_PATH=/opt/rocm-6.2.0 HIPFIRE_DEVICES=$HIPFIRE_DEV

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" \
         -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && break
    sleep 5
  done
}

stop_gateway() {
  pkill -f "openai_[s]erver.py" 2>/dev/null || true
  sleep 3
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 120); do pgrep -x glm53 >/dev/null || return 0; sleep 2; done
  echo "FATAL: an engine is still alive after 240 s"; return 1
}

HIPFIRE_PID=""
stop_hipfire() {
  [ -n "$HIPFIRE_PID" ] && kill -0 "$HIPFIRE_PID" 2>/dev/null && {
    kill "$HIPFIRE_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$HIPFIRE_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$HIPFIRE_PID" 2>/dev/null || true
  }
  pkill -f "target/release/hip[f]ire serve" 2>/dev/null || true
  # wait for the pid AND for card0 VRAM to actually drop before returning --
  # h0b_chain.sh's race was launching the next variant before either had
  # happened.
  for _ in $(seq 1 30); do
    pgrep -f "target/release/hip[f]ire serve" >/dev/null || break
    sleep 1
  done
  for _ in $(seq 1 30); do
    [ "$(VRAM 0)" -lt 1073741824 ] && break
    sleep 1
  done
  HIPFIRE_PID=""
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  stop_hipfire
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== h0c_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "--- accept_live.sh ---"
  "$HERE/accept_live.sh" || echo "ACCEPT_LIVE FAILED"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== h0c_chain $TAG $(date -Is)"
echo "--- step 1: stop gateway"
stop_gateway || exit 1
for e in glm53 qwen38 qwen38-vk; do
  pgrep -x "$e" >/dev/null && { echo "FATAL: $e still running"; exit 1; }
done
for c in 0 1 2; do
  v=$(VRAM "$c")
  echo "card$c: vram_used=$v"
  [ "$v" -lt 1073741824 ] || { echo "FATAL: card$c VRAM $v >= 1 GiB after stop"; exit 1; }
done
rm -f ~/.hipfire_kernels/gfx1100/*.tmp 2>/dev/null || true

try_variant() {
  local name="$1"; shift
  echo "--- variant $name: $* ---"
  local hlog="$OUT/${TAG}_${name}_hipfire.log"
  "$@" "$HIPFIRE_BIN" serve "$HIPFIRE_MODEL" 0.0.0.0:"$HIPFIRE_PORT" --kv-backend contiguous \
    > "$hlog" 2>&1 < /dev/null &
  HIPFIRE_PID=$!
  local up=0
  for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$HIPFIRE_PORT/v1/models" 2>/dev/null)
    [ "$code" = 200 ] && { up=1; break; }
    kill -0 "$HIPFIRE_PID" 2>/dev/null || { echo "variant $name: hipfire exited before answering"; break; }
    sleep 5
  done
  if [ "$up" != 1 ]; then
    echo "variant $name: FAILED (no /v1/models)"
    stop_hipfire
    return 1
  fi
  curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}" \
    > "$OUT/${TAG}_${name}_throwaway.json" 2>&1
  cat "$OUT/${TAG}_${name}_throwaway.json"; echo
  if grep -q '"error"' "$OUT/${TAG}_${name}_throwaway.json" 2>/dev/null; then
    echo "variant $name: throwaway still errors"
    stop_hipfire
    return 1
  fi
  echo "variant $name: SUCCESS"
  return 0
}

WINNER=""
if try_variant a env HIPFIRE_MTP_MODE=off; then
  WINNER=a
elif try_variant b env HIPFIRE_MTP_MODE=off HIPFIRE_REPLAY_BACKEND=off; then
  WINNER=b
fi

if [ -n "$WINNER" ]; then
  echo "=== WINNER: variant $WINNER ==="
  t0=$(date +%s.%N)
  curl -s -N -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
    > "$OUT/h0c_first.sse" 2>&1
  t1=$(date +%s.%N)
  echo "wall time req1: $(echo "$t1 - $t0" | bc 2>/dev/null || echo "${t0}->${t1}")"
  t2=$(date +%s.%N)
  curl -s -N -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
    > "$OUT/h0c_second.sse" 2>&1
  t3=$(date +%s.%N)
  echo "wall time req2: $(echo "$t3 - $t2" | bc 2>/dev/null || echo "${t2}->${t3}")"
  curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
    > "$OUT/${TAG}_usage.json" 2>&1
  cat "$OUT/${TAG}_usage.json"; echo
  stop_hipfire
else
  echo "=== NO WINNER: H0 closes as O3 on this box: hipFire v0.3.1 forward fails on"
  echo "ROCm 6.2.0 (24.04 build on Ubuntu 26.04) with variants vmm/contiguous,"
  echo "prewarm on/off, MTP on/off, Redline on/off; cause not exposed by hipFire;"
  echo "owner's options are a ROCm release built for this OS or debugging"
  echo "hipFire's source."
fi

for c in 0 1 2; do
  v=$(VRAM "$c")
  echo "card$c: vram_used=$v (post)"
  [ "$v" -lt 1073741824 ] || echo "WARNING: card$c VRAM not free ($v bytes)"
done

exit 0
