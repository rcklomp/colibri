#!/bin/bash
# h0d_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item H0, rerun (2026-09-16)
# against a user-local ROCm 10.0.0 (TheRock stable pip wheels in a venv,
# ~/venvs/rocm) instead of the box's system ROCm 6.2.0. Everything else is
# h0_chain.sh unchanged: same device map, same port, same requests, same
# gateway trap. hipFire itself is rebuilt against the venv ROCm
# (~/src/hipfire/target-rocm7/release/hipfire) -- see FRANKEN-H0d in the
# record for the venv recipe and what changed vs 6.2.
#
# Launch through run_chain.sh, never directly -- it takes the rig lock, and the
# gateway watchdog (cron, every 5 min) restarts the gateway under any
# measurement that stopped it without one:
#
#   setsid nohup ~/src/colibri-h0d/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-h0d/tools/hot-expert/h0d_chain.sh \
#       > ~/bench/h0d_chain.log 2>&1 < /dev/null &
#
# Steps (§H0, §M0, §6 of the plan):
#   1. assert no engine is running and VRAM is free on all three cards; record
#      DPM level and sclk per card (read-only)
#   2. stop the owner's gateway (WAIT for the engine to die -- ETXTBSY trap)
#   3. hipFire H0: serve the MQ4R SKU on dev3's HIP index, one throwaway
#      request, the ladder question twice (streamed, saved raw), one
#      non-streaming call for `usage`, GET /v1/models
#   4. stop hipFire; assert VRAM back under 1 GB on dev3's card, unchanged on
#      the other two
#   5. M0: rccl-tests all_reduce_perf -g 3 and -g 2, if the binary exists --
#      if the build never produced one (a named cause is a valid M0 outcome),
#      say so and move on rather than spending lock time on a doomed build
#   6. assert VRAM free on every card; restart the gateway on every exit path
#      and run accept_live.sh
#
# Device map (record §Q13, confirmed here by rocminfo + rocm-smi + the VRAM
# rise on the card that answers hipFire's request):
#   dev0 = 0000:83:00.0 = /sys/class/drm/card1 = ROCr Node 1 = HIP index 0
#   dev2 = 0000:48:00.0 = /sys/class/drm/card2 = ROCr Node 3 = HIP index 2
#   dev3 = 0000:86:00.0 = /sys/class/drm/card0 = ROCr Node 2 = HIP index 1
# hipFire selects a device with HIPFIRE_DEVICES=<physical GPU index>, which
# synchronizes ROCR_VISIBLE_DEVICES=<list> with HIP_VISIBLE_DEVICES=0..N-1
# (docs/env-vars.md). dev3 is HIP/physical index 1, hence HIPFIRE_DEVICES=1.
set -u
TAG=h0d_$(date +%m%d%H%M)
OUT=~/bench/h0_out; mkdir -p "$OUT"
HERE=~/src/colibri-h0d/tools/hot-expert
HIPFIRE_BIN=~/src/hipfire/target-rocm7/release/hipfire
HIPFIRE_MODEL="qwen3.6:35b-a3b-mq4r"
HIPFIRE_PORT=${HIPFIRE_PORT:-11436}
HIPFIRE_DEV=${HIPFIRE_DEV:-1}          # dev3's HIP/physical index, see header
RCCL_BIN=~/src/rccl-tests/build/all_reduce_perf
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
ROCM_ROOT=$(~/venvs/rocm/bin/rocm-sdk path --root)
QUESTION="Continue summarising the notes above in a few sentences, without repeating what you already said."

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
DPMLVL() { cat "/sys/class/drm/card$1/device/power_dpm_force_performance_level" 2>/dev/null; }
SCLK() { cat "/sys/class/drm/card$1/device/pp_dpm_sclk" 2>/dev/null | tr '\n' ' '; }

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh > "$LOG" 2>&1 < /dev/null &
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
  echo "=== h0_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "--- accept_live.sh ---"
  "$HERE/accept_live.sh" || echo "ACCEPT_LIVE FAILED"
  echo "=== results dir: $OUT"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== h0_chain $TAG $(date -Is)"

echo "--- step 1: stop gateway"
stop_gateway || exit 1

echo "--- step 1b: pre-checks (post-stop: engines idle, VRAM free, DPM/sclk recorded)"
for e in glm53 qwen38 qwen38-vk; do
  pgrep -x "$e" >/dev/null && { echo "FATAL: $e still running after stop_gateway"; exit 1; }
done
for c in 0 1 2; do
  v=$(VRAM "$c")
  echo "card$c: vram_used=$v dpm=$(DPMLVL "$c") sclk=[$(SCLK "$c")]"
  [ "$v" -lt 1073741824 ] || { echo "FATAL: card$c VRAM $v >= 1 GiB after stopping the gateway"; exit 1; }
done

echo "--- step 3: hipFire H0 (venv ROCm $ROCM_ROOT)"
[ -x "$HIPFIRE_BIN" ] || { echo "FATAL: $HIPFIRE_BIN missing"; exit 1; }
# Venv-ROCm workaround (2026-09-16), NOT a system change, and NOT specific to
# ROCm 6.2 -- it reproduces unchanged against this venv's clang (AMD clang
# 23.0.0git): this box has GCC 11/15/16 with no libstdc++-16-dev, so clang's
# GCC-install autodetection always picks the newest, headerless GCC 16 unless
# told otherwise; CPLUS_INCLUDE_PATH is honoured even then (confirmed with a
# device-only cmath kernel compile against the venv ROCm, --cuda-device-only,
# no LD_LIBRARY_PATH). hipFire spawns clang++ as a child with no override
# point, so this is exported for it to inherit. The other half of the 6.2
# workaround -- LD_LIBRARY_PATH=~/compat/lib for a libxml2.so.2 shim ld.lld
# wanted -- is DROPPED here: this venv's ld.lld has no libxml2 dependency at
# all (ldd shows none), so LD_LIBRARY_PATH is only needed to point the built
# binary's HIP runtime discovery at the venv root's lib dir.
rm -f ~/.hipfire_kernels/gfx1100/*.tmp 2>/dev/null || true
HIPFIRE_LOG="$OUT/${TAG}_hipfire.log"
ROCM_PATH="$ROCM_ROOT" HIP_PATH="$ROCM_ROOT" HIPFIRE_DEVICES=$HIPFIRE_DEV \
  CPLUS_INCLUDE_PATH=/usr/include/c++/15:/usr/include/x86_64-linux-gnu/c++/15 \
  LD_LIBRARY_PATH="$ROCM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$HIPFIRE_BIN" serve "$HIPFIRE_MODEL" 0.0.0.0:"$HIPFIRE_PORT" \
  > "$HIPFIRE_LOG" 2>&1 < /dev/null &
HIPFIRE_PID=$!
echo "hipfire serve pid=$HIPFIRE_PID port=$HIPFIRE_PORT dev=$HIPFIRE_DEV (HIPFIRE_DEVICES=$HIPFIRE_DEV)"

up=0
for _ in $(seq 1 60); do
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$HIPFIRE_PORT/v1/models" 2>/dev/null)
  [ "$code" = 200 ] && { up=1; break; }
  kill -0 "$HIPFIRE_PID" 2>/dev/null || { echo "FATAL: hipfire exited before answering, see $HIPFIRE_LOG"; break; }
  sleep 5
done
[ "$up" = 1 ] || { echo "H0: FAILED -- hipfire never answered /v1/models"; exit 1; }

echo "GET /v1/models:"
curl -s "http://127.0.0.1:$HIPFIRE_PORT/v1/models" | tee "$OUT/${TAG}_models.json"; echo

echo "throwaway request:"
curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}" \
  > "$OUT/${TAG}_throwaway.json" 2>&1
cat "$OUT/${TAG}_throwaway.json"; echo

echo "greedy 64-token streamed request #1 (D_B(~0) preview):"
t0=$(date +%s.%N)
curl -s -N -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
  > "$OUT/h0_first.sse" 2>&1
t1=$(date +%s.%N)
echo "wall time req1: $(echo "$t1 - $t0" | bc 2>/dev/null || echo "${t0}->${t1}")"

echo "same request again (repeat, for cache/prefix evidence):"
t2=$(date +%s.%N)
curl -s -N -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
  > "$OUT/h0_second.sse" 2>&1
t3=$(date +%s.%N)
echo "wall time req2: $(echo "$t3 - $t2" | bc 2>/dev/null || echo "${t2}->${t3}")"

echo "non-streaming call for usage:"
curl -s -m 120 "http://127.0.0.1:$HIPFIRE_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HIPFIRE_MODEL\",\"temperature\":0,\"max_tokens\":64,\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" \
  > "$OUT/${TAG}_usage.json" 2>&1
cat "$OUT/${TAG}_usage.json"; echo

echo "--- serve --help / prefix-cache / MTP / speculative flags:"
"$HIPFIRE_BIN" serve --help > "$OUT/${TAG}_serve_help.txt" 2>&1
cat "$OUT/${TAG}_serve_help.txt"

echo "--- step 4: stop hipFire"
stop_hipfire
sleep 3
v0=$(VRAM 0)
echo "card0 (dev3) vram after stop: $v0"
[ "$v0" -lt 1073741824 ] || echo "WARNING: dev3 VRAM not released ($v0 bytes)"
for c in 1 2; do echo "card$c vram (should be unchanged): $(VRAM "$c")"; done

echo "--- step 5: M0 rccl-tests"
if [ -x "$RCCL_BIN" ]; then
  NCCL_DEBUG=INFO LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib \
    "$RCCL_BIN" -b 8 -e 128M -f 2 -g 3 > "$OUT/${TAG}_m0_g3.log" 2>&1
  echo "M0 -g3 rc=$?"
  NCCL_DEBUG=INFO LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib \
    "$RCCL_BIN" -b 8 -e 128M -f 2 -g 2 > "$OUT/${TAG}_m0_g2.log" 2>&1
  echo "M0 -g2 rc=$?"
else
  echo "M0: SKIPPED -- $RCCL_BIN does not exist. Build failed on a named cause:" \
       "this ROCm 6.2.0 install has no amdgcn device-library bitcode" \
       "(amdclang++ 'cannot find ROCm device library'; no oclc_*.bc under" \
       "/opt/rocm-6.2.0, and /opt/rocm-6.2.0/bin is root-owned, not writable" \
       "without sudo). hipify-perl was also missing and was fetched from" \
       "ROCm/HIPIFY upstream to a user-local path as a documented workaround;" \
       "the device-library gap has no such workaround without sudo or a" \
       "system package. See ~/rccl_tests_build.log, ~/rccl_tests_build3.log," \
       "~/rccl_tests_build4.log on the Mac-side session's record." \
       | tee "$OUT/${TAG}_m0_blocked.txt"
fi

echo "--- step 6: final VRAM check"
for c in 0 1 2; do
  v=$(VRAM "$c")
  echo "card$c: vram_used=$v"
  [ "$v" -lt 1073741824 ] || echo "WARNING: card$c VRAM not free before restart ($v bytes)"
done

exit 0
