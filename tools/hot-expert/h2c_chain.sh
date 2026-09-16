#!/bin/bash
# h2c_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item H2c: a second
# hipFire-class arm, B' = hipEngine (shisa-ai/hipEngine, PyPI `hipengine`),
# on the same card and the same Qwen3.6-35B-A3B model family as arm B
# (hipFire). Modeled on tools/hot-expert/h0d2_chain.sh (the venv-ROCm serve
# recipe: one HTTP engine on dev3, throwaway + repeat + usage, stop, assert
# VRAM, restart gateway, accept_live.sh) and the B-arm functions in
# franken_chain.sh (run_B: assert_vram_free before/after, the smoke-shaped
# context_ladder.py --url invocation, thinking off via chat_template_kwargs).
#
# Launch through run_chain.sh, never directly -- it takes the rig lock, and
# the gateway watchdog (cron, every 5 min) restarts the gateway under any
# measurement that stopped it without one:
#
#   setsid nohup ~/src/colibri-h2c/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-h2c/tools/hot-expert/h2c_chain.sh \
#       >> ~/bench/h2c_run.log 2>&1 < /dev/null &
#
# Facts recorded during install (record §FRANKEN-H2c) that shape this script:
#   - hipEngine JIT-compiles its HIP kernels via `hipcc` on first model load
#     and caches the .so under ~/.cache/hipengine/build/ (docs/KERNELS.md
#     "hipengine.core.build calls hipcc or nvcc ... caches by
#     source/flags/compiler/target"; README "The first model load compiles
#     and caches kernels, so it takes longer than later starts."). That
#     means the SAME box trap as hipFire applies: this box's clang
#     autodetects the newest, headerless GCC 16 unless CPLUS_INCLUDE_PATH
#     points it at GCC 15's headers (agent-chain-traps-2026-09-16 item 7).
#     Reused verbatim from h0d2_chain.sh/franken_chain.sh's run_B.
#   - Device selection is HIP_VISIBLE_DEVICES (docs/ENVS.md); hipEngine does
#     not have hipFire's HIPFIRE_DEVICES name. dev3 = PCI 0000:86:00.0 =
#     /sys/class/drm/card0 = HIP index 1 (record §Q13, same device map used
#     by arm B) -- verify the index with the venv's rocminfo before trusting
#     it blindly on a driver/topology change.
#   - Because the first request pays the JIT-compile cost inside model load
#     (serve's default --eager-load / --startup-chat-smoke run at startup,
#     per `hipengine serve --help`), the /v1/models readiness poll below is
#     far more generous than hipFire's (hipFire ships prebuilt kernels and
#     only JITs a small AOTriton image cache; hipEngine's own README singles
#     out "the first model load" as the slow one). 40 min bound (480 x 5s).
#   - Model: unsloth/Qwen3.6-35B-A3B-MTP-GGUF, file
#     Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (22 663 387 424 bytes, sha256
#     0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b),
#     fetched to /home/ronald/models/hipengine_qwen36_gguf/ -- the same repo
#     the hipFire arm's own MTP sidecar comes from (docs/MTP-gguf.md), the
#     GGUF family hipEngine's README lists as tested for gfx1100, and the
#     "Qwen3.6-35B-A3B-Q4_K_M ... public bring-up" fixture in docs/GGUF.md.
#   - Thinking off: hipEngine documents `chat_template_kwargs.enable_thinking`
#     (docs/API.md line ~1128) "accepted for Qwen-compatible clients" -- the
#     exact same mechanism arm B already uses via context_ladder.py's
#     --chat-template-kwargs, reused verbatim as THINK_OFF_CTK below. Unlike
#     GLM-5.3 (arm A: no thinking-off form, see franken_chain.sh's header),
#     Qwen3.6 has one, same as hipFire.
#   - Prefix/KV-cache reuse: hipEngine's server ships a `--prefix-cache
#     {off,radix}` flag, default off. Passed as `radix` here so the second,
#     identical 64-token request (D_B'(~0) evidence) actually exercises
#     reuse instead of measuring an engine that cannot reuse anything.
#   - Speculative decoding (MTP): `--speculative-mtp-serving` default `auto`
#     (fail-closed to AR outside a measured/qualified shape) is left
#     unchanged -- same posture as arm B's own default, no override asked
#     for by the plan.
#
# Order: stop gateway -> assert VRAM free on all 3 cards -> start hipEngine
# on dev3 -> GET /v1/models -> throwaway (thinking off) -> greedy 64-token
# streamed request #1 (raw SSE saved) -> same request repeated (2nd TTFT,
# prefix-cache evidence) -> non-streaming usage call -> serve --help saved ->
# smoke ladder + 2-size cold sweep via context_ladder.py --url --arm Bp ->
# stop hipEngine -> assert VRAM released (60s bounded) -> exit trap restarts
# the gateway on every exit path and runs accept_live.sh (CLAUDE.md: the
# request AFTER the one under test is part of the measurement).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=~/bench/ctx_ladder_out; mkdir -p "$OUT"
H2C_OUT=~/bench/h2c_out; mkdir -p "$H2C_OUT"
TAG=h2c$(date +%m%d%H%M)

HIPENGINE_BIN=~/venvs/rocm/bin/hipengine
HIPENGINE_MODEL=/home/ronald/models/hipengine_qwen36_gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
HIPENGINE_MODEL_DIR=/home/ronald/models/hipengine_qwen36_gguf
HIPENGINE_MODEL_ID=qwen36-hipengine
HIPENGINE_PORT=${HIPENGINE_PORT:-11437}     # hipFire uses 11436, gateway 8081, C-arm 8600
HIPENGINE_DEV=${HIPENGINE_DEV:-1}           # dev3's HIP index, see header
# Run 1 (tag h2c09162052, 2026-09-16 20:52 UTC) loaded the weights to 21.23 of
# 23.98 GiB and then failed: "automatic GGUF resident context sizing found no
# allocatable context tokens", followed by HIP OOM in the runtime-workspace
# preparation on every request (hipEngine's own 35B-A3B rows were taken on a
# 48 GB W7900). --max-context-tokens pins the KV pool instead of letting the
# estimator ask for the model's full window; override or extend here.
HIPENGINE_EXTRA_ARGS=${HIPENGINE_EXTRA_ARGS:---max-context-tokens 16384}
THINK_OFF_CTK='{"enable_thinking": false}'  # hipEngine's documented Qwen-compatible off switch
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
QUESTION="Continue summarising the notes above in a few sentences, without repeating what you already said."

# --- smoke shape (task phase 2 window, <= 30 min budget) -------------------
B_STEPS="256,256"; B_GEN=16; B_FOLLOWUPS=1
DO_COLD_SWEEP=1; COLD_SWEEP_SIZES="1024,2048"

echo "=== h2c_chain $TAG $(date -Is)"
echo "=== model=$HIPENGINE_MODEL port=$HIPENGINE_PORT dev=$HIPENGINE_DEV extra_args=[$HIPENGINE_EXTRA_ARGS]"
echo "=== B_STEPS=$B_STEPS gen=$B_GEN followups=$B_FOLLOWUPS cold_sweep=$COLD_SWEEP_SIZES"

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
DPMLVL() { cat "/sys/class/drm/card$1/device/power_dpm_force_performance_level" 2>/dev/null; }

precheck() {   # precheck <label>
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  for e in glm53 qwen38 qwen38-vk qwen36; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  if pgrep -f "hip[e]ngine serve" >/dev/null; then echo "[$label] hipengine: RUNNING"; else echo "[$label] hipengine: none"; fi
  if pgrep -f "release/hip[f]ire" >/dev/null; then echo "[$label] hipfire: RUNNING"; else echo "[$label] hipfire: none"; fi
  for c in 0 1 2; do
    echo "[$label] card$c vram_used=$(VRAM "$c") dpm=$(DPMLVL "$c")"
  done
}

# Verbatim from franken_chain.sh's assert_vram_free -- the amdgpu/KFD driver
# reclaims a large allocation's page tables over real wall-clock time after
# the process exits, not atomically with it (found live on B1, tag
# fk09160840). 60 s bound in 2 s steps.
assert_vram_free() {   # assert_vram_free <label>
  local label=$1 v c bad attempt
  for attempt in $(seq 1 30); do
    bad=0
    for c in 0 1 2; do
      v=$(VRAM "$c")
      [ "$v" -lt 0 ] || [ "$v" -ge 1073741824 ] && bad=1
    done
    [ "$bad" = 0 ] && { [ "$attempt" -gt 1 ] && echo "[$label] VRAM free after ${attempt} checks (~$(( (attempt-1) * 2 ))s)"; return 0; }
    sleep 2
  done
  for c in 0 1 2; do
    v=$(VRAM "$c")
    if [ "$v" -lt 0 ] || [ "$v" -ge 1073741824 ]; then
      echo "FATAL [$label]: card$c VRAM $v >= 1 GiB (or unreadable) after 60s of retries"
    fi
  done
  return 1
}

wait_no_proc() {   # wait_no_proc <procname>
  local p=$1
  for _ in $(seq 1 120); do pgrep -x "$p" >/dev/null || return 0; sleep 2; done
  echo "FATAL: $p still alive after 240 s"; return 1
}

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh >> "$GLOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" \
         -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && break
    sleep 5
  done
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
}

stop_gateway() {
  pkill -f "openai_[s]erver.py" 2>/dev/null || true
  sleep 3
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_proc glm53
}

HIPENGINE_PID=""
stop_hipengine() {
  [ -n "$HIPENGINE_PID" ] && kill -0 "$HIPENGINE_PID" 2>/dev/null && {
    kill "$HIPENGINE_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$HIPENGINE_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$HIPENGINE_PID" 2>/dev/null || true
  }
  pkill -f "hip[e]ngine serve" 2>/dev/null || true
  HIPENGINE_PID=""
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  stop_hipengine
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== h2c_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "--- accept_live.sh (the request AFTER the one under test is part of the measurement, CLAUDE.md)"
  "$HERE/accept_live.sh"
  alive_rc=$?
  echo "--- accept_live.sh exit=$alive_rc"
  [ "$rc" -eq 0 ] && rc=$alive_rc
  echo "=== results dir: $OUT (tag $TAG), $H2C_OUT"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

precheck "chain-start"
stop_gateway || exit 1
assert_vram_free "pre-hipengine" || exit 1

echo "--- start hipEngine (venv ROCm 10, same CPLUS_INCLUDE_PATH recipe as hipFire -- box trap, see header)"
[ -x "$HIPENGINE_BIN" ] || { echo "FATAL: $HIPENGINE_BIN missing"; exit 1; }
[ -f "$HIPENGINE_MODEL" ] || { echo "FATAL: $HIPENGINE_MODEL missing"; exit 1; }
ROCM_ROOT=$(~/venvs/rocm/bin/rocm-sdk path --root)
HLOG="$H2C_OUT/${TAG}_hipengine.log"
ROCM_PATH="$ROCM_ROOT" HIP_PATH="$ROCM_ROOT" HIP_VISIBLE_DEVICES=$HIPENGINE_DEV \
  CPLUS_INCLUDE_PATH=/usr/include/c++/15:/usr/include/x86_64-linux-gnu/c++/15 \
  LD_LIBRARY_PATH="$ROCM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$HIPENGINE_BIN" serve \
    --model "$HIPENGINE_MODEL" \
    --served-model-name "$HIPENGINE_MODEL_ID" \
    --host 0.0.0.0 --port "$HIPENGINE_PORT" \
    --prefix-cache radix \
    $HIPENGINE_EXTRA_ARGS \
    >> "$HLOG" 2>&1 < /dev/null &
HIPENGINE_PID=$!
echo "hipengine serve pid=$HIPENGINE_PID port=$HIPENGINE_PORT dev=$HIPENGINE_DEV (HIP_VISIBLE_DEVICES=$HIPENGINE_DEV) rocm_root=$ROCM_ROOT"

# Generous bound: hipEngine JIT-compiles kernels into the model-load path on
# first start (README: "The first model load compiles and caches kernels, so
# it takes longer than later starts."), unlike hipFire which ships prebuilt
# kernels. 480 x 5s = 40 min.
up=0
for _ in $(seq 1 480); do
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$HIPENGINE_PORT/v1/models" 2>/dev/null)
  [ "$code" = 200 ] && { up=1; break; }
  kill -0 "$HIPENGINE_PID" 2>/dev/null || { echo "FATAL: hipengine exited before answering, see $HLOG"; break; }
  sleep 5
done
[ "$up" = 1 ] || { echo "H2c: FAILED -- hipengine never answered /v1/models (40 min bound), see $HLOG"; exit 1; }
echo "hipengine ready at $(date -Is)"

echo "GET /v1/models:"
curl -s "http://127.0.0.1:$HIPENGINE_PORT/v1/models" | tee "$H2C_OUT/${TAG}_models.json"; echo

echo "GET /v1/hipengine/capabilities (MTP/speculative/prefix-cache/context posture):"
curl -s "http://127.0.0.1:$HIPENGINE_PORT/v1/hipengine/capabilities" | tee "$H2C_OUT/${TAG}_capabilities.json"; echo

echo "throwaway request (thinking off):"
curl -s -m 120 "http://127.0.0.1:$HIPENGINE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HIPENGINE_MODEL_ID\",\"temperature\":0,\"max_tokens\":16,\"chat_template_kwargs\":${THINK_OFF_CTK},\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}" \
  > "$H2C_OUT/${TAG}_throwaway.json" 2>&1
cat "$H2C_OUT/${TAG}_throwaway.json"; echo

echo "greedy 64-token streamed request #1 (D_B'(~0) preview, thinking off):"
t0=$(date +%s.%N)
curl -s -N -m 120 "http://127.0.0.1:$HIPENGINE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HIPENGINE_MODEL_ID\",\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"chat_template_kwargs\":${THINK_OFF_CTK},\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
  > "$H2C_OUT/${TAG}_first.sse" 2>&1
t1=$(date +%s.%N)
echo "wall time req1: $(echo "$t1 - $t0" | bc 2>/dev/null || echo "${t0}->${t1}")"

echo "same request again (repeat, prefix-cache=radix reuse evidence):"
t2=$(date +%s.%N)
curl -s -N -m 120 "http://127.0.0.1:$HIPENGINE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HIPENGINE_MODEL_ID\",\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"chat_template_kwargs\":${THINK_OFF_CTK},\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
  > "$H2C_OUT/${TAG}_second.sse" 2>&1
t3=$(date +%s.%N)
echo "wall time req2: $(echo "$t3 - $t2" | bc 2>/dev/null || echo "${t2}->${t3}")"

echo "non-streaming call for usage:"
curl -s -m 120 "http://127.0.0.1:$HIPENGINE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HIPENGINE_MODEL_ID\",\"temperature\":0,\"max_tokens\":64,\"stream\":false,\"chat_template_kwargs\":${THINK_OFF_CTK},\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
  > "$H2C_OUT/${TAG}_usage.json" 2>&1
cat "$H2C_OUT/${TAG}_usage.json"; echo

echo "--- serve --help (flags actually offered by this install):"
"$HIPENGINE_BIN" serve --help > "$H2C_OUT/${TAG}_serve_help.txt" 2>&1
cat "$H2C_OUT/${TAG}_serve_help.txt"

echo "--- smoke ladder + 2-size cold sweep via context_ladder.py --url"
json="$OUT/${TAG}_Bp.jsonl"
python3 "$HERE/context_ladder.py" \
    --url "http://127.0.0.1:$HIPENGINE_PORT" --model-id "$HIPENGINE_MODEL_ID" \
    --steps "$B_STEPS" --gen "$B_GEN" --followups "$B_FOLLOWUPS" \
    --cold-sweep "$COLD_SWEEP_SIZES" --sweep-offset-chars 300000 \
    --chat-template-kwargs "$THINK_OFF_CTK" \
    --snap "$HIPENGINE_MODEL_DIR" --min-resident 90 --warm \
    --arm Bp --tag "$TAG" --json "$json" \
    --server-log "$HLOG"
ladder_rc=$?
echo "=== arm Bp exit=$ladder_rc json=$json"

echo "--- stop hipEngine"
stop_hipengine
assert_vram_free "post-hipengine"
vram_rc=$?

echo "=== h2c_chain body done $(date -Is) -- exit trap runs restart/accept_live.sh next"
[ "$ladder_rc" = 0 ] && [ "$vram_rc" = 0 ]
exit $?
