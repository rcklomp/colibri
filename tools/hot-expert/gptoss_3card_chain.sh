#!/bin/bash
# gptoss_3card_chain.sh -- probe (2026-09-16, owner's question): a model
# bigger than one card, spread over the three RX 7900 XTX with experts or KV
# offloaded to RAM, at usable speed at realistic context depth. Everything
# tried on this box so far drops under 2 tok/s at depth (Colibri GLM-5.3:
# 2.27/2.21/2.24 tok/s at ~18.4-18.7k, record section FRANKEN-H2). This
# chain measures a ~60 GB MoE (gpt-oss-120b, MXFP4, 128 experts top-4) fully
# resident across the three cards on llama.cpp -- the multi-GPU engine
# already proven working on this box -- against the same model with its
# experts pushed to the CPU in two steps, on the SAME context_ladder.py
# harness used for FRANKEN-H2 so the rows are directly comparable.
#
# Model: ~/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf (ggml-org/gpt-oss-
# 120b-GGUF, 63.4 GB, 36 layers, 128 experts, top-4). Downloaded by a
# separate `hf download` already running when this chain is written; the
# chain waits (bounded) for ~/models/gpt-oss-120b/.download_done before
# touching the gateway.
#
# Arms (fresh process each; --n-cpu-moe counts LAYERS whose MoE experts move
# to the CPU, out of 36 total):
#   RV   = Vulkan (RADV), --n-gpu-layers 99 --n-cpu-moe 0, fully resident,
#          three devices, the owner's own working flag set
#          (~/bin/llama-profile.sh).
#   RH   = HIP, inside the rocm/dev-ubuntu-24.04:7.14.0-full container (the
#          native build-hip binary is missing libhipblas), same placement,
#          the owner's own docker pattern (~/qwen38-flash-next.sh).
#   OV18 = Vulkan, --n-cpu-moe 18 -- half the MoE layers' experts on the
#          CPU, attention and the rest on the GPUs.
#   OV36 = Vulkan, --n-cpu-moe 36 -- ALL experts on the CPU, the regime the
#          owner's GLM-5.3 lives in.
# Order (mirrored, CLAUDE.md "run arms interleaved A,B,B,A", extended to
# four conditions): RV RH OV18 OV36 OV36 OV18 RH RV.
#
# context_compare.py's arm_of() groups a --rows LABEL by its leading
# letters, stopping at the first non-letter character. "OV18"/"OV36" each
# already contain a digit before any run-number suffix could be appended
# (OV18_1 -> leading letters "OV", same as OV36_1 -> "OV" -- the two
# different arms would silently merge into one group). The --arm value
# written into every json row (and every filename/tag/log path below) is
# the spec's own vocabulary, exactly RV/RH/OV18/OV36; the LABELS handed to
# context_compare.py's --rows for grouping are a separate, letter-safe set,
# used only in that one place, never written to any file or the record:
#   RV1/RV2 (arm RV), RH1/RH2 (arm RH), HO1/HO2 (arm OV18, "half offload"),
#   FO1/FO2 (arm OV36, "full offload").
#
# Launch only through run_chain.sh (rig lock; restarts the gateway on every
# exit path):
#   setsid nohup ~/src/colibri-gptoss/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-gptoss/tools/hot-expert/gptoss_3card_chain.sh \
#       >> ~/bench/gptoss_3card.log 2>&1 < /dev/null &
#
# Budget: 8 arms, each up to a 15-min readiness poll (first load reads
# 63 GB from NVMe) + a 25-min-capped ladder (`timeout 1500`). An arm that
# times out is marked TIMEOUT; the chain continues to the next arm rather
# than aborting the whole run over one slow condition.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PRISTINE=~/src/colibri                        # binary in service; H2's own convention
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)

MODEL=~/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
DL_MARKER=~/models/gpt-oss-120b/.download_done
SNAP_DIR=~/models/gpt-oss-120b

VK_BIN=/home/ronald/src/llama.cpp/build/bin/llama-server
HIP_BIN_DIR=/home/ronald/src/llama-latest/build-hip/bin
HIP_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full

VK_PORT=8090
HIP_PORT=8091
CONTAINER_NAME=gptoss_rh_probe

OUT=~/bench/ctx_ladder_out; mkdir -p "$OUT"
GOUT=~/bench/gptoss_out; mkdir -p "$GOUT"
TAG=go$(date +%m%d%H%M)

REASON_FLAGS=(--reasoning-format none --chat-template-kwargs '{"reasoning_effort":"low"}')

echo "=== gptoss_3card_chain $TAG $(date -Is)"
echo "=== model=$MODEL"
echo "=== order: RV RH OV18 OV36 OV36 OV18 RH RV"

# ------------------------------------------------------------- primitives --
VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
DPMLVL() { cat "/sys/class/drm/card$1/device/power_dpm_force_performance_level" 2>/dev/null; }

precheck() {   # precheck <label> -- prints only, no side effects
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  for e in glm53 qwen38 qwen38-vk; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  echo "[$label] pgrep -f llama-[s]erver: $(pgrep -f "llama-[s]erver" | wc -l)"
  echo "[$label] docker ps: $(docker ps --format '{{.Image}} {{.Names}}' 2>/dev/null | tr '\n' ';')"
  for c in 0 1 2; do
    echo "[$label] card$c vram_used=$(VRAM "$c") dpm=$(DPMLVL "$c")"
  done
}

# Found live on the rig for FRANKEN-H2's B1 (tag fk09160840, 2026-09-16): a
# GPU process confirmed dead by pgrep still read VRAM used dropping across
# several seconds of checks (the driver reclaims page tables over real
# wall-clock time, not atomically with process exit). Same bound here: 60 s
# in 2 s steps. Copied from franken_chain.sh, not re-derived (CLAUDE.md).
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

wait_no_proc() {   # wait_no_proc <procname> (pgrep -x)
  local p=$1
  for _ in $(seq 1 120); do pgrep -x "$p" >/dev/null || return 0; sleep 2; done
  echo "FATAL: $p still alive after 240 s"; return 1
}

# Copied from franken_chain.sh's warm_glm/assert_glm_resident -- the exit
# trap must re-warm GLM before the gateway restarts, or the owner's first
# request after the chain pays minutes reading 182 GiB off NVMe.
warm_glm() {
  echo "--- warming GLM shards (one model at a time; 182 GiB)"
  cat "$GLM_SNAP"/*.safetensors > /dev/null 2>&1 || true
}

assert_glm_resident() {   # assert_glm_resident <label>
  local label=$1 pct
  pct=$(fincore --bytes --output SIZE,RES "$GLM_SNAP"/*.safetensors 2>/dev/null \
        | tail -n +2 | awk '{ts+=$1; rs+=$2} END{if (ts>0) printf "%.4f", 100*rs/ts; else print 0}')
  echo "[resid $label] resident=${pct}%"
  awk -v p="$pct" 'BEGIN{exit !(p>=90)}'
}

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT -u COLI_TIMERS \
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

wait_for_download() {
  echo "--- waiting for gpt-oss-120b download marker $DL_MARKER (bounded 40 min, 30s steps)"
  for i in $(seq 1 80); do
    [ -f "$DL_MARKER" ] && { echo "download marker present after $((i*30))s"; return 0; }
    sleep 30
  done
  echo "FATAL: download marker never appeared after 2400s -- see ~/bench/gptoss_download.log"
  return 1
}

# ------------------------------------------------------------- arm runner --
# State of whatever this chain itself started, for the exit trap. Never
# touch glm53 here -- it is already down from stop_gateway, and this chain
# never restarts it mid-run the way franken_chain.sh's A2 does.
VK_PID=""
HIP_PID=""
HIP_CONTAINER_UP=0

stop_vk() {
  [ -n "$VK_PID" ] && kill -0 "$VK_PID" 2>/dev/null && {
    kill "$VK_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$VK_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$VK_PID" 2>/dev/null || true
  }
  VK_PID=""
}

stop_hip() {
  if [ "$HIP_CONTAINER_UP" = 1 ]; then
    docker stop -t 10 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    HIP_CONTAINER_UP=0
  fi
  [ -n "$HIP_PID" ] && { wait "$HIP_PID" 2>/dev/null || true; }
  HIP_PID=""
}

# model_id_of <port> -- best-effort; empty means "let context_ladder.py
# auto-detect from /v1/models" (ttft_serve.HttpDriver already does this).
model_id_of() {
  curl -s -m 10 "http://127.0.0.1:$1/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["data"][0]["id"])
except Exception:
    pass' 2>/dev/null
}

wait_ready() {   # wait_ready <port> <pid-var-name> <label> -- bounded 15 min
  local port=$1 pidvar=$2 label=$3 pid
  for _ in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)" = 200 ] && return 0
    pid="${!pidvar}"
    [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { echo "$label: FATAL -- process died before /health=200"; return 1; }
    sleep 5
  done
  echo "$label: FATAL -- /health never returned 200 after 900s"
  return 1
}

# run_ladder <port> <arm> <n> <cmp_label> <elog> -- shared by every arm
# (Vulkan and HIP) once the server is up. Returns context_ladder.py's rc,
# or 124 (timeout(1)'s own code) if the 25-min cap was hit.
run_ladder() {
  local port=$1 arm=$2 n=$3 cmp=$4 elog=$5
  local tagarm="${arm}${n}" model_id json console rc
  json="$OUT/${TAG}_${tagarm}.jsonl"
  console="$GOUT/${TAG}_${tagarm}_ladder_console.log"
  model_id=$(model_id_of "$port")
  echo "$arm run $n: model-id=${model_id:-<auto>} port=$port json=$json"

  echo "throwaway request (not scored):"
  curl -s -m 120 "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model_id:-gpt-oss-120b}\",\"temperature\":0,\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}" \
    >> "$GOUT/${TAG}_${tagarm}_throwaway.json" 2>&1

  local model_id_args=()
  [ -n "$model_id" ] && model_id_args=(--model-id "$model_id")

  timeout 1500 python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$port" "${model_id_args[@]}" \
      --steps 1024,1024,2048,4096,8192 --gen 128 --followups 2 \
      --cold-sweep 2048,4096,8192,16384 --sweep-offset-chars 300000 \
      --snap "$SNAP_DIR" --min-resident 90 --warm \
      --arm "$arm" --tag "$TAG" --json "$json" \
      --server-log "$elog" 2>&1 | tee -a "$console"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 124 ]; then
    echo "=== $tagarm TIMEOUT (25 min cap, timeout(1) rc=124) -- continuing chain"
  else
    echo "=== $tagarm exit=$rc json=$json"
  fi
  return "$rc"
}

run_vulkan_arm() {   # run_vulkan_arm <arm:RV|OV18|OV36> <n> <ncmoe> <cmp_label>
  local arm=$1 n=$2 ncmoe=$3 cmp=$4
  local tagarm="${arm}${n}" elog rc
  elog="$GOUT/${TAG}_${tagarm}.log"
  echo "=== arm $tagarm (Vulkan, n-cpu-moe=$ncmoe) $(date -Is)"
  precheck "${tagarm}-pre"
  assert_vram_free "${tagarm}-pre" || { echo "FATAL: $tagarm VRAM not free before start -- aborting chain (box-safety, not a per-arm failure)"; exit 1; }

  local extra=()
  if [ "$ncmoe" != 0 ]; then
    extra=(-t 8 -tb 8)
  fi
  local envs=(VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json)
  if [ "$ncmoe" != 0 ]; then
    envs+=(OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close)
  fi

  env "${envs[@]}" setsid "$VK_BIN" \
    --model "$MODEL" --host 127.0.0.1 --port "$VK_PORT" \
    --n-gpu-layers 99 --device Vulkan0,Vulkan1,Vulkan2 --split-mode layer --tensor-split 1,1,1 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cont-batching \
    --n-cpu-moe "$ncmoe" -c 32768 --parallel 1 "${extra[@]}" \
    "${REASON_FLAGS[@]}" \
    >> "$elog" 2>&1 < /dev/null &
  VK_PID=$!
  echo "$tagarm: llama-server (Vulkan) pid=$VK_PID port=$VK_PORT ncmoe=$ncmoe"

  wait_ready "$VK_PORT" VK_PID "$tagarm" || { stop_vk; assert_vram_free "${tagarm}-post-fail" || { echo "FATAL: $tagarm VRAM stuck after a failed start -- aborting chain"; exit 1; }; return 1; }
  echo "$tagarm VRAM-AT-LOAD card0=$(VRAM 0) card1=$(VRAM 1) card2=$(VRAM 2)"

  run_ladder "$VK_PORT" "$arm" "$n" "$cmp" "$elog"
  rc=$?
  stop_vk
  assert_vram_free "${tagarm}-post" || { echo "FATAL: $tagarm VRAM stuck after stop -- aborting chain (box-safety)"; exit 1; }
  return "$rc"
}

run_hip_arm() {   # run_hip_arm <n>
  local n=$1 arm=RH
  local tagarm="RH${n}" elog rc
  elog="$GOUT/${TAG}_${tagarm}.log"
  echo "=== arm $tagarm (HIP, docker) $(date -Is)"
  precheck "${tagarm}-pre"
  assert_vram_free "${tagarm}-pre" || { echo "FATAL: $tagarm VRAM not free before start -- aborting chain (box-safety, not a per-arm failure)"; exit 1; }

  if [ "${HIP_REFUSED:-0}" = 1 ]; then
    echo "$tagarm: REFUSED -- docker unavailable ($HIP_REFUSE_REASON)"
    return 1
  fi
  if [ ! -x "${HIP_BIN_DIR}/llama-server" ]; then
    echo "$tagarm: REFUSED -- ${HIP_BIN_DIR}/llama-server missing or not executable"
    return 1
  fi

  docker run --rm --name "$CONTAINER_NAME" \
    -p "127.0.0.1:${HIP_PORT}:${HIP_PORT}" \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc=host \
    -e "LD_LIBRARY_PATH=/opt/rocm/lib:${HIP_BIN_DIR}" \
    -v /home/ronald:/home/ronald \
    "$HIP_IMAGE" \
    "${HIP_BIN_DIR}/llama-server" \
    -m "$MODEL" --host 0.0.0.0 --port "$HIP_PORT" \
    --n-gpu-layers 99 --device ROCm0,ROCm1,ROCm2 --split-mode layer --tensor-split 1,1,1 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cont-batching \
    --n-cpu-moe 0 -c 32768 --parallel 1 \
    "${REASON_FLAGS[@]}" \
    >> "$elog" 2>&1 < /dev/null &
  HIP_PID=$!
  HIP_CONTAINER_UP=1
  echo "$tagarm: docker run pid=$HIP_PID container=$CONTAINER_NAME port=$HIP_PORT"

  wait_ready "$HIP_PORT" HIP_PID "$tagarm" || { stop_hip; assert_vram_free "${tagarm}-post-fail" || { echo "FATAL: $tagarm VRAM stuck after a failed start -- aborting chain"; exit 1; }; return 1; }
  echo "$tagarm VRAM-AT-LOAD card0=$(VRAM 0) card1=$(VRAM 1) card2=$(VRAM 2)"

  run_ladder "$HIP_PORT" "$arm" "$n" "" "$elog"
  rc=$?
  stop_hip
  assert_vram_free "${tagarm}-post" || { echo "FATAL: $tagarm VRAM stuck after stop -- aborting chain (box-safety)"; exit 1; }
  return "$rc"
}

# ------------------------------------------------------------------- exit --
on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  stop_vk
  stop_hip
  echo "--- final re-warm of GLM before restart"
  warm_glm
  assert_glm_resident "final" || echo "WARNING: GLM not >=90% resident at restart time"
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== gptoss_3card_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== results: $OUT (jsonl), $GOUT (server logs/consoles)"
  echo "--- accept_live.sh (the request AFTER the chain is part of the measurement)"
  "$HERE/accept_live.sh"
  alive_rc=$?
  echo "--- accept_live.sh exit=$alive_rc"
  [ "$rc" -eq 0 ] && rc=$alive_rc
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

# --------------------------------------------------------------- preflight --
precheck "chain-start"

# Owner-safety (spec): a llama-server the owner is running, native or in a
# container, is not ours to touch. This check runs ONCE, before this chain
# has started anything of its own.
if pgrep -f "llama-[s]erver" >/dev/null; then
  echo "FATAL: a llama-server process is already running -- not ours, not touching it:"
  pgrep -af "llama-[s]erver"
  exit 1
fi

HIP_REFUSED=0
HIP_REFUSE_REASON=""
if ! docker info >/dev/null 2>&1; then
  HIP_REFUSED=1
  HIP_REFUSE_REASON="docker info failed: $(docker info 2>&1 | head -1)"
  echo "NOTE: $HIP_REFUSE_REASON -- HIP arms (RH) will be REFUSED, Vulkan arms still run"
fi

wait_for_download || exit 1

[ -x "$VK_BIN" ] || { echo "FATAL: $VK_BIN missing or not executable"; exit 1; }

stop_gateway || exit 1
assert_vram_free "post-stop-gateway" || exit 1

# ------------------------------------------------------------------- arms --
run_vulkan_arm RV 1 0 RV1
run_hip_arm 1
run_vulkan_arm OV18 1 18 HO1
run_vulkan_arm OV36 1 36 FO1
run_vulkan_arm OV36 2 36 FO2
run_vulkan_arm OV18 2 18 HO2
run_hip_arm 2
run_vulkan_arm RV 2 0 RV2

# --------------------------------------------------------------- compare --
echo "--- context_compare.py: RV vs RH, RV vs OV18, RV vs OV36, OV18 vs OV36"
echo "--- (labels below are the letter-safe RV1/RV2/RH1/RH2/HO1/HO2/FO1/FO2 --"
echo "--- see the header note; the json 'arm' field is the spec's own RV/RH/OV18/OV36)"

RV1_J="$OUT/${TAG}_RV1.jsonl"; RV2_J="$OUT/${TAG}_RV2.jsonl"
RH1_J="$OUT/${TAG}_RH1.jsonl"; RH2_J="$OUT/${TAG}_RH2.jsonl"
HO1_J="$OUT/${TAG}_OV181.jsonl"; HO2_J="$OUT/${TAG}_OV182.jsonl"
FO1_J="$OUT/${TAG}_OV361.jsonl"; FO2_J="$OUT/${TAG}_OV362.jsonl"

compare_pair() {   # compare_pair <name> "label=path label=path ..."
  local name=$1; shift
  local rows=() spec label path
  for spec in "$@"; do
    label=${spec%%=*}; path=${spec#*=}
    [ -f "$path" ] && rows+=("$label=$path")
  done
  if [ "${#rows[@]}" -ge 1 ]; then
    echo "=== context_compare.py: $name"
    python3 "$HERE/context_compare.py" --rows "${rows[@]}"
    echo "--- $name exit=$? (REFUSED/NO VERDICT is an accepted outcome)"
  else
    echo "=== context_compare.py: $name -- SKIPPED, no jsonl files present"
  fi
}

compare_pair "RV vs RH"   "RV1=$RV1_J" "RV2=$RV2_J" "RH1=$RH1_J" "RH2=$RH2_J"
compare_pair "RV vs OV18" "RV1=$RV1_J" "RV2=$RV2_J" "HO1=$HO1_J" "HO2=$HO2_J"
compare_pair "RV vs OV36" "RV1=$RV1_J" "RV2=$RV2_J" "FO1=$FO1_J" "FO2=$FO2_J"
compare_pair "OV18 vs OV36" "HO1=$HO1_J" "HO2=$HO2_J" "FO1=$FO1_J" "FO2=$FO2_J"

echo "=== gptoss_3card_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
