#!/bin/bash
# glm53flash_ladder_chain.sh -- the owner's other GLM-5.3-Flash deployment,
# on the same context_ladder.py harness FRANKEN-H2 used, so its rows sit
# beside Colibri's own A1/A2 (record section FRANKEN-H2).
#
# THE QUESTION. The owner's daily model is GLM-5.3-Flash. Colibri serves it
# (int4-gs64, three cards + RAM offload) at 2.27/2.21/2.24 tok/s at ~18.4k
# depth (record §FRANKEN-H2, arms A1/A2, jsonl
# ~/bench/ctx_ladder_out/fk09161755_A1.jsonl / _A2.jsonl). The owner ALSO
# runs the same model through llama.cpp under CPU offload,
# ~/glm53-flash.sh (read-only reference -- this chain does not call that
# script, it copies its docker command so the two ports/names/flags this
# chain adds do not collide with the owner's own daily use of it):
#   UD-IQ4_XS GGUF, 149 GB, 5 shards under
#   ~/models/GLM-5.3-Flash/UD-IQ4_XS/, -ngl 18 (19 OOMs -- the owner's own
#   comment: "max that fits VRAM"), --tensor-split 1,1,1 --split-mode layer
#   --device ROCm0,ROCm1,ROCm2 -fa on -ctk q8_0 -ctv q8_0 -t 16 -tb 8
#   --reasoning-effort low, inside rocm/dev-ubuntu-24.04:7.14.0-full,
#   BIN=/home/ronald/src/llama-glm53/build-hip/bin. Decode at empty context
#   3.15 tok/s (the owner's own comment; all-CPU --cpu-moe = 0.71 tok/s).
# Nobody has measured this deployment on the ladder. This chain does, two
# fresh-process runs (GF1, GF2), so the rows are directly comparable to A.
#
# NOT INTERLEAVED WITH A. CLAUDE.md's "run arms interleaved A,B,B,A" rule
# cannot be satisfied here: A (fk09161755) was already measured in a
# SEPARATE chain earlier the same day (2026-09-16), and re-running it would
# cost another ~2.3 h and hold the lock again for a number this session
# already has. The comparison at the end says so explicitly and the verdict
# carries that caveat -- it is a weaker claim than an interleaved pair, not
# a hidden one.
#
# THINKING: no thinking-off kwargs on this arm. The owner's own script
# passes --reasoning-effort low as a llama-server CLI flag (baked into the
# docker command below); this is not a thinking-OFF switch, the same
# asymmetry franken_chain.sh's header describes for Colibri's A arm (GLM-5.3
# has no thinking-off template form). Every GF request reasons at low
# effort, same as A; neither arm disables it.
#
# CPU THREADS: OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close (the
# rest of this tree's convention for a host-side engine) is NOT applicable
# here -- GF's compute runs inside the container on the owner's own
# -t 16 -tb 8, which is what he runs daily; this chain does not touch it.
# The host side only runs context_ladder.py, a thin HTTP client.
#
# PAGE CACHE. The 149 GB GGUF evicts most of Colibri's 182 GiB GLM-5.3
# shards from page cache (CLAUDE.md: "Qwen3.8 and GLM-5.3 do not fit in RAM
# together" -- this box has the same problem between Colibri's own GLM-5.3
# and a second, differently-quantized copy of the same model). The exit
# trap's re-warm of GLM_SNAP before the gateway restarts is therefore not
# optional here, same as every other chain that displaces Colibri's shards.
#
# --snap DELIBERATELY OVERRIDDEN, not left at context_ladder.py's default.
# ttft_serve.DEFAULT_SNAP is Colibri's OWN GLM_SNAP
# (~/models/GLM-5.3-Flash-colibri-int4-g64) -- leaving --snap unset while
# talking to a completely different server (GF, llama.cpp, a different
# quantization) would make every turn's residency check re-warm (cat)
# COLIBRI's 182 GiB shards mid-GF-run, fighting the 149 GB GF model for the
# same page cache the "page cache" note above just said doesn't fit two
# ways at once, and asserting a floor on memory the engine under test never
# touches. GF_SNAP_DIR (the 5 GGUF shards) is passed explicitly instead, so
# --warm/--min-resident 90 apply to the model actually being measured.
#
# SPLITTING THE LADDER FROM THE COLD SWEEP. context_ladder.py's --cold-sweep
# is HTTP-mode-only and, read closely, unconditional on --steps being
# non-empty: `steps = [int(s) for s in args.steps.split(",") if s]` on an
# empty string is `[]`, and with --followups 0 the turn loop
# (`for i, step in enumerate(steps + [0]*args.followups)`) runs zero times,
# falling straight through to the cold-sweep loop below it. So
# `--steps "" --followups 0 --cold-sweep ...` is a real, working
# sweep-only invocation, and both calls can append to the SAME --json file
# (emit() opens it "a"), same --tag, same --arm -- the rows land together
# exactly as a single invocation's would.
#
# This chain always uses two invocations per arm (ladder+followups, then
# cold-sweep-only), never one combined call, and this is unconditional, not
# "only if it looks like it would run long": nobody has ever measured this
# deployment's prefill rate, so there is no principled way to predict which
# side of 90 minutes a single call would land on before running it, and the
# split costs nothing (same file, same arm) while unconditionally protecting
# the already-good ladder rows if the sweep half is the one that overruns.
# The two invocations split the brief's 5400 s (90 min) total budget evenly,
# 2700 s each ($LADDER_TIMEOUT / $SWEEP_TIMEOUT below) -- an even split, not
# a derived one; see the commit/report for the arithmetic this number rests
# on and its honestly wide error bars.
#
# ORDER: GF1, then GF2 (fresh docker process each; two-invocation ladder+
# sweep each). Not interleaved with A (see above -- A already exists).
#
# Launch only through run_chain.sh (rig lock; restarts the gateway on every
# exit path):
#   setsid nohup ~/src/colibri-glmflash/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-glmflash/tools/hot-expert/glm53flash_ladder_chain.sh \
#       >> ~/bench/glm53flash_ladder.log 2>&1 < /dev/null &
#
# Budget per arm: readiness up to 20 min (1200 s; 149 GB from NVMe at
# ~0.7 GB/s is ~4 min, plus HIP init and container startup) + throwaway
# (<1 min) + ladder (capped 2700 s) + cold sweep (capped 2700 s) + docker
# stop / VRAM settle (bounded 60 s). An arm whose ladder or sweep half times
# out is marked TIMEOUT in the log; the chain continues (ladder rows already
# written are kept either way) rather than aborting the whole run over one
# slow half.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)

GF_MODEL=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
GF_SNAP_DIR=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS
GF_BIN_DIR=/home/ronald/src/llama-glm53/build-hip/bin
GF_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full
GF_PORT=8092
GF_MODEL_ID=glm53-flash-ladder

LADDER_TIMEOUT=2700   # 45 min -- see the header note on the 90-min split
SWEEP_TIMEOUT=2700    # 45 min

OUT=~/bench/ctx_ladder_out; mkdir -p "$OUT"
GOUT=~/bench/glm53flash_out; mkdir -p "$GOUT"
TAG=gf$(date +%m%d%H%M)

# A's own files -- fk09161755, FRANKEN-H2 -- read explicitly, not by tag:
# context_compare.py's arm_of() groups --rows labels by their leading
# letters only, never by filename or tag, so a label from one tag and a
# label from another mix into one comparison exactly as intended here.
A1_J=~/bench/ctx_ladder_out/fk09161755_A1.jsonl
A2_J=~/bench/ctx_ladder_out/fk09161755_A2.jsonl

echo "=== glm53flash_ladder_chain $TAG $(date -Is)"
echo "=== GF_MODEL=$GF_MODEL"
echo "=== order: GF1 GF2 (each: ladder+followups, then cold-sweep-only, same json)"
echo "=== A (fk09161755) was measured in a SEPARATE chain earlier 2026-09-16, NOT interleaved with GF"

# ------------------------------------------------------------- primitives --
# VRAM/DPMLVL/precheck/assert_vram_free/wait_no_proc/warm_glm/
# assert_glm_resident/start_gateway/stop_gateway copied from
# gptoss_3card_chain.sh (CLAUDE.md: "do not re-derive anything that is in
# it" -- these are already proven live on this box, 2026-09-16).
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

assert_vram_free() {   # assert_vram_free <label> -- bounded 60 s, 2 s steps
  # Copied from franken_chain.sh/gptoss_3card_chain.sh, not re-derived: the
  # amdgpu/KFD driver reclaims a large allocation's page tables over real
  # wall-clock time after the process exits, not atomically with it.
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

wait_no_proc() {   # wait_no_proc <procname> (pgrep -x) -- bounded 240 s
  local p=$1
  for _ in $(seq 1 120); do pgrep -x "$p" >/dev/null || return 0; sleep 2; done
  echo "FATAL: $p still alive after 240 s"; return 1
}

wait_no_container() {   # wait_no_container <name> -- bounded 60 s, 2 s steps
  local name=$1
  for _ in $(seq 1 30); do
    docker ps -q -f "name=^/${name}\$" 2>/dev/null | grep -q . || return 0
    sleep 2
  done
  echo "FATAL: container $name still present after 60 s"
  return 1
}

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

wait_ready_gf() {   # wait_ready_gf <port> <label> -- bounded 20 min, 5 s steps
  local port=$1 label=$2
  for _ in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)" = 200 ] && return 0
    [ -n "$GF_PID" ] && ! kill -0 "$GF_PID" 2>/dev/null && { echo "$label: FATAL -- docker run exited before /health=200"; return 1; }
    sleep 5
  done
  echo "$label: FATAL -- /health never returned 200 after 1200s (20 min)"
  return 1
}

# --------------------------------------------------------------- GF arm --
# State of whatever THIS chain started -- the exit trap must stop only this,
# never touch glm53 by name (the brief's own instruction: this chain never
# spawned it, so it must never `pkill -9 -x glm53`; only start_gateway does
# that, and only through the owner's own ~/start_glm53.sh).
GF_PID=""
GF_CONTAINER_UP=0
GF_CONTAINER_NAME=""

stop_gf() {
  if [ "$GF_CONTAINER_UP" = 1 ]; then
    docker stop -t 10 "$GF_CONTAINER_NAME" >/dev/null 2>&1 || true
    wait_no_container "$GF_CONTAINER_NAME"
    GF_CONTAINER_UP=0
  fi
  [ -n "$GF_PID" ] && { wait "$GF_PID" 2>/dev/null || true; }
  GF_PID=""
  GF_CONTAINER_NAME=""
}

run_gf_arm() {   # run_gf_arm <n> -- fresh docker process each call
  local n=$1 arm=GF tagarm name elog console json rc_ladder rc_sweep
  tagarm="${arm}${n}"
  name="glm53flash_ladder_${n}"
  elog="$GOUT/${TAG}_${tagarm}.log"
  console="$GOUT/${TAG}_${tagarm}_ladder_console.log"
  json="$OUT/${TAG}_${tagarm}.jsonl"
  echo "=== arm $tagarm (llama.cpp GLM-5.3-Flash UD-IQ4_XS, docker HIP, -ngl 18) $(date -Is)"
  precheck "${tagarm}-pre"
  assert_vram_free "${tagarm}-pre" || { echo "FATAL: $tagarm VRAM not free before start -- aborting chain (box-safety, not a per-arm failure)"; exit 1; }

  # Exactly the docker command ~/glm53-flash.sh launches (read on the rig,
  # read-only) -- same image, bind mount, ROCm devices, -t 16 -tb 8,
  # --reasoning-effort low -- with a dedicated name/port and the extra
  # serving flags the brief calls for (--parallel 1 --ctx-size 32768
  # --alias) so this run never collides with the owner's own daily use of
  # that script.
  docker run --rm --name "$name" \
    -p "127.0.0.1:${GF_PORT}:${GF_PORT}" \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc=host \
    -e "LD_LIBRARY_PATH=/opt/rocm/lib:${GF_BIN_DIR}" \
    -v /home/ronald:/home/ronald \
    "$GF_IMAGE" \
    "${GF_BIN_DIR}/llama-server" \
    -m "$GF_MODEL" -ngl 18 \
    --tensor-split 1,1,1 --split-mode layer --device ROCm0,ROCm1,ROCm2 \
    -fa on -ctk q8_0 -ctv q8_0 -t 16 -tb 8 \
    --reasoning-effort low \
    --host 0.0.0.0 --port "$GF_PORT" \
    --parallel 1 --ctx-size 32768 --alias "$GF_MODEL_ID" \
    >> "$elog" 2>&1 < /dev/null &
  GF_PID=$!
  GF_CONTAINER_UP=1
  GF_CONTAINER_NAME="$name"
  echo "$tagarm: docker run pid=$GF_PID container=$name port=$GF_PORT"

  wait_ready_gf "$GF_PORT" "$tagarm" || { stop_gf; assert_vram_free "${tagarm}-post-fail" || { echo "FATAL: $tagarm VRAM stuck after a failed start -- aborting chain"; exit 1; }; return 1; }
  echo "$tagarm VRAM-AT-LOAD card0=$(VRAM 0) card1=$(VRAM 1) card2=$(VRAM 2)"

  echo "throwaway request (not scored):"
  curl -s -m 120 "http://127.0.0.1:$GF_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$GF_MODEL_ID\",\"temperature\":0,\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}" \
    >> "$GOUT/${TAG}_${tagarm}_throwaway.json" 2>&1

  # 1/2: ladder + followups. No --cold-sweep here -- see the header note on
  # why this is always split into two invocations.
  echo "--- $tagarm ladder+followups (timeout ${LADDER_TIMEOUT}s)"
  timeout "$LADDER_TIMEOUT" python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$GF_PORT" --model-id "$GF_MODEL_ID" \
      --steps 1024,1024,2048,4096,8192 --gen 128 --followups 2 \
      --snap "$GF_SNAP_DIR" --min-resident 90 --warm \
      --arm "$arm" --tag "$TAG" --json "$json" \
      --server-log "$elog" 2>&1 | tee -a "$console"
  rc_ladder=${PIPESTATUS[0]}
  if [ "$rc_ladder" = 124 ]; then
    echo "=== $tagarm ladder TIMEOUT (${LADDER_TIMEOUT}s cap) -- rows already written are kept; sweep half still attempted"
  else
    echo "=== $tagarm ladder exit=$rc_ladder"
  fi

  # 2/2: cold-sweep-only, same json, same tag/arm (--steps "" --followups 0
  # skips the ladder+followup loop entirely and falls through to the sweep
  # loop -- see the header note).
  echo "--- $tagarm cold-sweep-only, second invocation, same json (timeout ${SWEEP_TIMEOUT}s)"
  timeout "$SWEEP_TIMEOUT" python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$GF_PORT" --model-id "$GF_MODEL_ID" \
      --steps "" --followups 0 --gen 128 \
      --cold-sweep 2048,4096,8192,16384 --sweep-offset-chars 300000 \
      --snap "$GF_SNAP_DIR" --min-resident 90 --warm \
      --arm "$arm" --tag "$TAG" --json "$json" \
      --server-log "$elog" 2>&1 | tee -a "$console"
  rc_sweep=${PIPESTATUS[0]}
  if [ "$rc_sweep" = 124 ]; then
    echo "=== $tagarm cold-sweep TIMEOUT (${SWEEP_TIMEOUT}s cap) -- continuing chain"
  else
    echo "=== $tagarm cold-sweep exit=$rc_sweep"
  fi

  echo "=== $tagarm json=$json ladder_rc=$rc_ladder sweep_rc=$rc_sweep"
  stop_gf
  assert_vram_free "${tagarm}-post" || { echo "FATAL: $tagarm VRAM stuck after stop -- aborting chain (box-safety)"; exit 1; }
  [ "$rc_ladder" = 0 ] && [ "$rc_sweep" = 0 ]
}

# ------------------------------------------------------------------- exit --
on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  stop_gf   # idempotent no-op if the last arm already stopped cleanly
  echo "--- final re-warm of GLM before restart"
  warm_glm
  assert_glm_resident "final" || echo "WARNING: GLM not >=90% resident at restart time"
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== glm53flash_ladder_chain exit rc=$rc tag=$TAG $(date -Is)"
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

docker info >/dev/null 2>&1 || { echo "FATAL: docker unavailable ($(docker info 2>&1 | head -1))"; exit 1; }
[ -f "$GF_MODEL" ] || { echo "FATAL: $GF_MODEL missing"; exit 1; }
[ -x "${GF_BIN_DIR}/llama-server" ] || { echo "FATAL: ${GF_BIN_DIR}/llama-server missing or not executable on the host (bind-mounted into the container)"; exit 1; }

# Owner-safety: a llama-server the owner (or llama-swap) is already running,
# native or in a container, is not ours to touch -- this check runs ONCE,
# before this chain stops the gateway or starts anything of its own.
if pgrep -f "llama-[s]erver" >/dev/null; then
  echo "FATAL: a llama-server process is already running -- not ours, not touching it:"
  pgrep -af "llama-[s]erver"
  exit 1
fi
llama_containers=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -i llama || true)
if [ -n "$llama_containers" ]; then
  echo "FATAL: a docker container with 'llama' in its name is already running -- not ours, not touching it:"
  echo "$llama_containers"
  exit 1
fi

stop_gateway || exit 1
assert_vram_free "post-stop-gateway" || exit 1

# ------------------------------------------------------------------- arms --
run_gf_arm 1
run_gf_arm 2

# --------------------------------------------------------------- compare --
echo "--- context_compare.py: A (Colibri GLM-5.3, fk09161755) vs GF (llama.cpp CPU-offload, this chain, tag $TAG)"
echo "--- CAVEAT: A was measured in a SEPARATE chain earlier the same day (2026-09-16), NOT"
echo "--- interleaved with GF -- CLAUDE.md's 'run arms interleaved A,B,B,A' rule is not met by"
echo "--- this comparison, and the verdict below should be read with that caveat, not as an"
echo "--- interleaved A/B."

GF1_J="$OUT/${TAG}_GF1.jsonl"
GF2_J="$OUT/${TAG}_GF2.jsonl"

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

compare_pair "A (fk09161755, not interleaved) vs GF ($TAG)" \
  "A1=$A1_J" "A2=$A2_J" "GF1=$GF1_J" "GF2=$GF2_J"

echo "=== glm53flash_ladder_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
