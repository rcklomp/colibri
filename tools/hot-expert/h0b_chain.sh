#!/bin/bash
# h0b_chain.sh -- H0 follow-up: name the "bench_decode forward failed" cause.
#
# hipFire's own log calls its KV cache "Fwht3 vmm" -- HIP virtual-memory
# management APIs a ROCm 6.2 userspace may not support on this Ubuntu 26.04
# kernel, which would fail exactly at the first forward pass. Try, in order,
# stopping at the first that answers a request: (a) --kv-backend contiguous;
# (b) same + --no-prewarm; (c) same as (b) with HIP-level logging so the
# failing call is named. If one answers, run the full request set (repeat
# for the two TTFTs, non-streaming usage) and compute D_B(~=0). If none
# answers, quote the first HIP error line from the AMD_LOG output verbatim.
#
# Launch only via run_chain.sh. Budget: <= 25 min, stop regardless.
set -u
TAG=h0b_$(date +%m%d%H%M)
OUT=~/bench/h0_out; mkdir -p "$OUT"
HERE=~/src/colibri-h0/tools/hot-expert
HIPFIRE_BIN=~/src/hipfire/target/release/hipfire
HIPFIRE_MODEL="qwen3.6:35b-a3b-mq4r"
HIPFIRE_PORT=${HIPFIRE_PORT:-11436}
HIPFIRE_DEV=${HIPFIRE_DEV:-1}
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
QUESTION="Continue summarising the notes above in a few sentences, without repeating what you already said."
# Standing workaround (see FRANKEN-H0/M0 record, 2026-09-16): this box's
# ROCm 6.2 clang picks headerless GCC 16 and its ld.lld wants libxml2.so.2,
# which is only libxml2.so.16 here.
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
  HIPFIRE_PID=""
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  stop_hipfire
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== h0b_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "--- accept_live.sh ---"
  "$HERE/accept_live.sh" || echo "ACCEPT_LIVE FAILED"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== h0b_chain $TAG $(date -Is)"
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
  "$HIPFIRE_BIN" serve "$HIPFIRE_MODEL" 0.0.0.0:"$HIPFIRE_PORT" "$@" \
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
  echo "variant $name: throwaway request"
  curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}" \
    > "$OUT/${TAG}_${name}_throwaway.json" 2>&1
  cat "$OUT/${TAG}_${name}_throwaway.json"; echo
  echo "variant $name: greedy 64-token request"
  curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
    > "$OUT/${TAG}_${name}_greedy.json" 2>&1
  cat "$OUT/${TAG}_${name}_greedy.json"; echo
  if grep -q '"error"' "$OUT/${TAG}_${name}_throwaway.json" "$OUT/${TAG}_${name}_greedy.json" 2>/dev/null; then
    echo "variant $name: request(s) still error"
    stop_hipfire
    return 1
  fi
  echo "variant $name: SUCCESS"
  return 0
}

WINNER=""
if try_variant a --kv-backend contiguous; then
  WINNER=a
elif try_variant b --kv-backend contiguous --no-prewarm; then
  WINNER=b
else
  echo "--- variant c: contiguous + no-prewarm + HIP-level logging ---"
  hlog="$OUT/${TAG}_c_hipfire.log"
  AMD_LOG_LEVEL=4 AMD_LOG_MASK=0xFFFFFFFF RUST_BACKTRACE=1 HIPFIRE_LOG=trace \
    "$HIPFIRE_BIN" serve "$HIPFIRE_MODEL" 0.0.0.0:"$HIPFIRE_PORT" \
      --kv-backend contiguous --no-prewarm \
    > "$hlog" 2>&1 < /dev/null &
  HIPFIRE_PID=$!
  up=0
  for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$HIPFIRE_PORT/v1/models" 2>/dev/null)
    [ "$code" = 200 ] && { up=1; break; }
    kill -0 "$HIPFIRE_PID" 2>/dev/null || break
    sleep 5
  done
  if [ "$up" = 1 ]; then
    curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}" \
      > "$OUT/${TAG}_c_throwaway.json" 2>&1
    cat "$OUT/${TAG}_c_throwaway.json"; echo
    if ! grep -q '"error"' "$OUT/${TAG}_c_throwaway.json"; then
      curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
        > "$OUT/${TAG}_c_greedy.json" 2>&1
      cat "$OUT/${TAG}_c_greedy.json"; echo
      grep -q '"error"' "$OUT/${TAG}_c_greedy.json" || WINNER=c
    fi
  fi
  echo "--- first HIP-ish error line from variant c's log ---"
  grep -m1 -iE "hip error|HipError|panicked|error\[|hsa_|amdgpu|Aborted|SIGSEGV|SIGABRT" "$hlog" || echo "(no matching line found in $hlog)"
  [ -n "$WINNER" ] || stop_hipfire
fi

if [ -n "$WINNER" ]; then
  echo "=== WINNER: variant $WINNER ==="
  echo "greedy 64-token streamed request #1 (D_B(~0) preview):"
  t0=$(date +%s.%N)
  curl -s -N -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
    > "$OUT/h0b_first.sse" 2>&1
  t1=$(date +%s.%N)
  echo "wall time req1: $(echo "$t1 - $t0" | bc 2>/dev/null || echo "${t0}->${t1}")"
  echo "same request again (repeat, for cache/prefix evidence):"
  t2=$(date +%s.%N)
  curl -s -N -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
    > "$OUT/h0b_second.sse" 2>&1
  t3=$(date +%s.%N)
  echo "wall time req2: $(echo "$t3 - $t2" | bc 2>/dev/null || echo "${t2}->${t3}")"
  echo "non-streaming call for usage:"
  curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
    > "$OUT/${TAG}_usage.json" 2>&1
  cat "$OUT/${TAG}_usage.json"; echo
  stop_hipfire
fi

sleep 2
for c in 0 1 2; do
  v=$(VRAM "$c")
  echo "card$c: vram_used=$v (post)"
  [ "$v" -lt 1073741824 ] || echo "WARNING: card$c VRAM not free ($v bytes)"
done

exit 0
