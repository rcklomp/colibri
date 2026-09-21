#!/bin/bash
# f11_ladder_chain.sh -- Franken plan item F11 step 0 (plan Rev 38,
# 2026-09-21): the owner closed F3 (Qwen3.6) and named the acceptable model
# families -- Qwen3.8, DeepSeek V4.x, GLM-5.3 and successors. Two files have
# sat on the rig's disk since 2026-09-01, never measured:
#   Q = ~/models/Qwen3.8-Flash-Next/UD-IQ4_XS (93.7 GB, arch qwen4exp)
#   D = ~/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M (90.9 GB, arch
#       deepseek4)
# Both would be ~70-80% VRAM-resident on this box's three 24 GiB cards
# (GLM-5.3 in both its deployments is only 30-40% resident and pays 3-5
# tok/s at 18k -- record §FRANKEN-H2, glm53flash_ladder_chain.sh). This
# chain measures decode tok/s and TTFT for both, at the same context depths
# as the GF/A arms, on the same harness. It is derived from
# glm53flash_ladder_chain.sh (read in full before writing this one) for its
# primitives, its exit-trap contract, and its two-invocation ladder/sweep
# split; gptoss_3card_chain.sh for --n-cpu-moe and the tensor-split pattern.
#
# BACKEND: HIP, via docker, NOT Vulkan -- verified, not assumed.
#   ~/src/llama.cpp/build (Vulkan, libggml-vulkan.so, mtime 2026-08-07
#   03:38) predates the qwen4exp arch commits (llama.cpp upstream added
#   qwen4exp starting 2026-08-27, e.g. 6c84c7d5d "add Qwen3.8-Flash-Next");
#   `strings .../build/bin/libllama.so.0.0.1 | grep -x qwen4exp` finds
#   NOTHING (deepseek4 is present, qwen4exp is not -- deepseek4 landed
#   earlier). Every *build-hip* tree on the rig (llama.cpp/build-hip,
#   llama-glm53/build-hip, llama-mtp/build-hip, llama-27977/build-hip,
#   llama-latest/build-hip), all rebuilt 2026-08-31/09-01, carries BOTH
#   `strings .../libllama.so.0.3.0 | grep -x qwen4exp` and `deepseek4`.
#   Q therefore cannot run on the Vulkan binary the gpt-oss chain used --
#   HIP is the only backend on this rig that can serve it, so this chain
#   uses HIP for both Q and D (one backend, one docker image, comparable
#   rows). Bin dir and docker image are copied verbatim from
#   glm53flash_ladder_chain.sh's GF arm (GF_BIN_DIR/GF_IMAGE), which is
#   already proven live on this box (same rocm/dev-ubuntu-24.04:7.14.0-full
#   image, same LD_LIBRARY_PATH/-v/--device pattern) -- not re-derived.
#   `--n-cpu-moe` exists in this tree: common/arg.cpp:2789 (llama-glm53),
#   "keep the Mixture of Experts (MoE) weights of the first N layers in the
#   CPU" (common/common.h:1142 `llm_add_n_cpu_ffn_overrides` -- it is the
#   FIRST N layers by block index, not the last N; the direction does not
#   matter for a residency search, only the count does, and this comment
#   corrects an initial assumption that it was the last N).
#
# PLACEMENT: -ngl 999 (every layer's attention/dense/embedding on the GPUs)
# + --n-cpu-moe N (only the routed-expert FFN weights of the first N layers
# move to the CPU) -- the Franken hypothesis from the plan's own next
# paragraph: place experts by residency budget, keep the trunk resident.
# --tensor-split 1,1,1 --split-mode layer --device ROCm0,ROCm1,ROCm2 is kept
# from both GF and gpt-oss's RH arm (same 3-way even split, proven live).
#
# N0 -- THE STARTING GUESS, ITS OWN ARITHMETIC, LABELLED AS A GUESS.
# N cannot be known without loading (the brief's own words); this chain
# finds it per model by trying to load, before any timed arm, bounded 6
# tries, step bump on failure. N0 only decides where the search STARTS.
#
# The byte counts below were read from each GGUF's own header (magic,
# version, tensor_count, kv_count, KV pairs, then each tensor's dims/type/
# offset) with a ~150-line pure-stdlib python parser (no numpy -- the rig
# has none, CLAUDE.md) run read-only over the shard files on the rig; NO
# engine, GPU or docker container was touched to get them. Full per-tensor
# dump is in this branch's commit body, not repeated here; the results:
#
#   Q (qwen4exp): block_count=48, expert_count=512, expert_used=10,
#     expert_ffn_len=640, embedding_length=2560. Total file bytes (3
#     shards) = 93,682,584,224. Expert tensor bytes (blk.N.ffn_{gate,up,
#     down}_exps.weight, mixed iq3_s/iq4_nl/q8_0 -- an Unsloth UD mixed
#     quant) summed over all 48 layers = 59,519,795,200 (63.5% of the
#     file). Non-expert (attention incl. the indexer, PLE embedding table,
#     SSM/hyper-connection state, lm_head, norms) = 34,162,789,024.
#     Average expert bytes/layer = 59,519,795,200 / 48 = 1,239,995,733.
#   D (deepseek4): block_count=43, expert_count=256, expert_used=6,
#     shared_count=1, expert_ffn_len=2048, embedding_length=4096. Total
#     file bytes (3 shards) = 90,926,928,288. Expert tensor bytes (mixed
#     iq2_xxs/iq2_s/iq3_xxs/mxfp4) summed over 43 layers = 83,869,302,784
#     (92.2% of the file -- IQ2_M's low bit width sits almost entirely in
#     the experts). Non-expert = 7,057,625,504. Average expert bytes/layer
#     = 83,869,302,784 / 43 = 1,950,449,basically-computed-in-bash-below.
#
# Both archs use a compressed/MLA-style KV cache (deepseek4:
# head_count_kv=1, key/value_length=512; qwen4exp: head_count_kv=2,
# key/value_length=256 -- both far below a plain GQA cache of this size)
# and BOTH have a `full_attention_interval`/`compress_ratios` field, i.e.
# only a fraction of layers keep a real per-token KV cache at all (the rest
# are compressed/SSM-style, ~O(1) state). A naive upper bound that ignores
# that sparsity and prices EVERY layer as full dense attention at q8_0 (~1
# byte/elem after quantization) and ctx 32768 is still only ~1.6 GB total
# for either model (48*2*(256+256)*32768 ≈ 1.61 GB for Q; 43*1*(512+512)*
# 32768 ≈ 1.44 GB for D) -- small next to a 24 GiB card. The reserve below
# is therefore generous on purpose; the load-attempt search is the actual
# test, this arithmetic only picks where it starts.
#
# RESERVE: 3 GiB per card (9 GiB total) for KV cache (bounded above, see
# arithmetic just above) + llama.cpp's per-device compute/graph buffers +
# allocator slack. A stated guess, not a measurement.
#
# STEP: on a failed try, N grows by enough to free ~2 GiB/card (6 GiB
# total) worth of expert weight, rounded up to whole layers: STEP =
# ceil(6 GiB / avg_expert_bytes_per_layer) -- computed in bash below from
# the same byte constants, so it is reproducible from this file alone
# (Q: ~6 layers, D: ~4 layers). Bounded 6 tries (N0, N0+STEP, ..., capped
# at the model's own block_count).
#
# This is a MONOTONIC UPWARD search from a conservative N0: if N0 itself
# loads (expected, since the reserve above is generous relative to the
# ~1.6 GB naive KV bound), that is what this chain reports and it is NOT
# claimed to be the true minimum -- refining downward from a successful N0
# is future work, out of scope for step 0 (the brief's own wording: "the
# SMALLEST N that loads" as found by this bounded upward search, not by an
# exhaustive one).
#
# PAGE CACHE: Q (93.7 GB) + D (90.9 GB) = 184.6 GB fit together in 247 GiB
# (265.6 GB) RAM; GLM-5.3's 182 GiB shards do not fit alongside either (the
# three-way sum is 366+ GB) -- CLAUDE.md's page-cache note, restated for a
# three-way conflict instead of GF's two-way one. Order Q1,D1,D2,Q2 needs
# only ONE warm of each file (Q once before the Q-search+Q1, D once before
# the D-search+D1; D2/Q2 re-verify only, matching GF's own residency check
# which re-warms just the shortfall, not unconditionally) -- both files
# should stay resident together throughout since 184.6 GB < 265.6 GB.
# Every warm is two-pass with a >=90% fincore check each pass (record bug
# 19, 2026-09-16: one `cat` under a nearly-full page cache can leave a file
# ~85% resident; a second pass finds the rest). context_ladder.py's own
# --warm --min-resident 90 is the second line of defense inside every arm
# and inside the N-search's OWN pre-loop warm this chain adds (bash-native,
# fincore-based, since context_ladder.py is not invoked during the search).
# The GF chain's final GLM re-warm before the gateway restarts is kept
# unconditionally.
#
# ARMS: Q1, D1, D2, Q2 -- this chain's own order IS the interleave
# (CLAUDE.md "run arms interleaved A,B,B,A"), unlike glm53flash_ladder_chain
# .sh's GF-vs-A comparison which could not interleave against an
# already-completed run. context_compare.py's arm_of() groups a --rows
# label by its leading letters only (regex ^([A-Za-z]+)) -- "Q1"/"Q2" ->
# group "Q", "D1"/"D2" -> group "D", no collision (unlike gpt-oss's
# OV18/OV36, which shared no digit-free prefix and needed a letter-safe
# alias table; Q/D need none, checked, not invented).
#
# Launch only through run_chain.sh (rig lock; restarts the gateway on
# every exit path). The script must be present on the rig's ~/src/colibri
# tree (this branch merged/synced there) before this is run -- that sync
# is the orchestrator's job, not this chain's:
#   ssh -n -f rome 'setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh \
#       ~/src/colibri/tools/hot-expert/f11_ladder_chain.sh \
#       > ~/bench/f11_ladder.log 2>&1 < /dev/null &'
#
# ESTIMATED WALL TIME (labelled: nobody has measured either model on this
# box, so this is arithmetic on the timeouts below, not a result).
#   Worst case (every bound hit): 2 model searches x 6 tries x 480s
#   (2880s=48min) = 96min, + 4 arms x (1200s ready + 2700s ladder + 2700s
#   sweep + 60s settle = 6660s=111min) = 444min, + final GLM re-warm
#   (~182 GiB / ~0.7 GB/s, two passes worst case ~10min), + accept_live
#   (~2min) ~= 552 min ~= 9.2 h.
#   Realistic case (first search try succeeds each model, ladder/sweep
#   finish well inside their caps -- both models are far more VRAM-resident
#   than GLM-5.3, so prefill/decode should be faster than GLM's mostly-CPU
#   regime the caps were sized for): ~16min (searches, dominated by the two
#   warm passes) + ~90min (4 arms) + ~10min (final rewarm) + ~2min
#   (accept_live) ~= 2 h.
#
# LIVENESS CHECKS REPLACED (CLAUDE.md, 2026-09-20 incident: a killed engine
# child is a ZOMBIE and still matches `pgrep -x glm53`). Three places in the
# inherited pattern used `pgrep -x glm53` as a liveness signal; all three
# are replaced here with `engine_alive()` (`ps -C glm53 -o stat= | grep -qv
# '^Z'`, verbatim from run_chain.sh's own fix):
#   1. precheck()'s per-engine loop -- glm53's count now comes from
#      engine_alive, not `pgrep -x glm53 | wc -l` (qwen38/qwen38-vk keep
#      plain pgrep -x; they have no documented zombie-under-supervisor
#      pattern).
#   2. start_gateway()'s status line -- was `pgrep -x glm53 | wc -l`, now
#      `engine_alive && echo 1 || echo 0`.
#   3. stop_gateway()'s wait for glm53 to be gone -- was `wait_no_proc
#      glm53` (built on `pgrep -x`), now `wait_no_glm53` (built on
#      engine_alive). `wait_no_proc` is kept, unused by this script, only
#      because it is a harmless generic primitive copied along with the
#      rest of the primitives block; nothing calls it with "glm53" anymore.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)

# --- models ---------------------------------------------------------------
Q_MODEL=/home/ronald/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
Q_SNAP_DIR=/home/ronald/models/Qwen3.8-Flash-Next/UD-IQ4_XS
Q_MODEL_ID=f11-qwen38flash-ladder
Q_LAYERS=48
Q_TOTAL_BYTES=93682584224
Q_EXPERT_BYTES=59519795200

D_MODEL=/home/ronald/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
D_SNAP_DIR=/home/ronald/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M
D_MODEL_ID=f11-deepseek4flash-ladder
D_LAYERS=43
D_TOTAL_BYTES=90926928288
D_EXPERT_BYTES=83869302784

# --- backend: HIP docker, same bin dir/image glm53flash_ladder_chain.sh's
# GF arm used (see header for why the Vulkan build is unusable here) ------
F11_BIN_DIR=/home/ronald/src/llama-glm53/build-hip/bin
F11_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full
F11_PORT=8093

LADDER_TIMEOUT=2700   # 45 min -- see the header note (no prior measurement)
SWEEP_TIMEOUT=2700    # 45 min
SEARCH_TIMEOUT=480    # 8 min per --n-cpu-moe try (warm cache; see header)
READY_TIMEOUT=1200    # 20 min -- matches GF's own bound (149 GB cold read)

OUT=~/bench/ctx_ladder_out; mkdir -p "$OUT"
GOUT=~/bench/f11_out; mkdir -p "$GOUT"
TAG=f11$(date +%m%d%H%M)

# --- N0/STEP arithmetic (bash integers, bytes; see header for the source
# of the byte constants and the reasoning) ---------------------------------
GIB=1073741824
VRAM_PER_CARD_GIB=24
CARDS=3
RESERVE_PER_CARD_GIB=3
VRAM_TOTAL_BYTES=$(( VRAM_PER_CARD_GIB * CARDS * GIB ))
RESERVE_BYTES=$(( RESERVE_PER_CARD_GIB * CARDS * GIB ))
VRAM_USABLE_BYTES=$(( VRAM_TOTAL_BYTES - RESERVE_BYTES ))
STEP_HEADROOM_BYTES=$(( 2 * CARDS * GIB ))   # ~2 GiB/card freed per retry

ceil_div() { # ceil_div <num> <den> -- integer ceiling division, num/den >= 0
  local num=$1 den=$2
  [ "$num" -le 0 ] && { echo 0; return; }
  echo $(( (num + den - 1) / den ))
}

Q_AVG_EXPERT_BYTES=$(( Q_EXPERT_BYTES / Q_LAYERS ))
D_AVG_EXPERT_BYTES=$(( D_EXPERT_BYTES / D_LAYERS ))
Q_EXCESS=$(( Q_TOTAL_BYTES - VRAM_USABLE_BYTES ))
D_EXCESS=$(( D_TOTAL_BYTES - VRAM_USABLE_BYTES ))
Q_N0=$(ceil_div "$Q_EXCESS" "$Q_AVG_EXPERT_BYTES")
D_N0=$(ceil_div "$D_EXCESS" "$D_AVG_EXPERT_BYTES")
Q_STEP=$(ceil_div "$STEP_HEADROOM_BYTES" "$Q_AVG_EXPERT_BYTES")
D_STEP=$(ceil_div "$STEP_HEADROOM_BYTES" "$D_AVG_EXPERT_BYTES")
[ "$Q_STEP" -lt 1 ] && Q_STEP=1
[ "$D_STEP" -lt 1 ] && D_STEP=1

echo "=== f11_ladder_chain $TAG $(date -Is)"
echo "=== Q_MODEL=$Q_MODEL"
echo "=== D_MODEL=$D_MODEL"
echo "=== VRAM budget: total=$VRAM_TOTAL_BYTES reserve=$RESERVE_BYTES usable=$VRAM_USABLE_BYTES bytes"
echo "=== Q: avg_expert_bytes/layer=$Q_AVG_EXPERT_BYTES N0=$Q_N0 STEP=$Q_STEP (of $Q_LAYERS layers)"
echo "=== D: avg_expert_bytes/layer=$D_AVG_EXPERT_BYTES N0=$D_N0 STEP=$D_STEP (of $D_LAYERS layers)"
echo "=== order: Q1 D1 D2 Q2 (this chain's own order is the A,B,B,A interleave)"

# ------------------------------------------------------------- primitives --
# VRAM/DPMLVL/assert_vram_free/wait_no_container/warm_glm/assert_glm_resident
# copied from glm53flash_ladder_chain.sh / gptoss_3card_chain.sh (CLAUDE.md:
# "do not re-derive anything that is in it" -- proven live on this box).
VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
DPMLVL() { cat "/sys/class/drm/card$1/device/power_dpm_force_performance_level" 2>/dev/null; }

# Non-zombie liveness check for glm53, copied from run_chain.sh's own fix
# (CLAUDE.md, 2026-09-20: a killed engine child is a ZOMBIE and still
# matches `pgrep -x glm53`). Every liveness check on glm53 in this script
# uses this, never a bare `pgrep -x glm53` (see the header's replacement
# list).
engine_alive() { ps -C glm53 -o stat= 2>/dev/null | grep -qv '^Z'; }

precheck() {   # precheck <label> -- prints only, no side effects
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  echo "[$label] glm53 engine_alive (non-zombie): $(engine_alive && echo 1 || echo 0)"
  for e in qwen38 qwen38-vk; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  echo "[$label] pgrep -f llama-[s]erver: $(pgrep -f "llama-[s]erver" | wc -l)"
  echo "[$label] docker ps: $(docker ps --format '{{.Image}} {{.Names}}' 2>/dev/null | tr '\n' ';')"
  for c in 0 1 2; do
    echo "[$label] card$c vram_used=$(VRAM "$c") dpm=$(DPMLVL "$c")"
  done
}

assert_vram_free() {   # assert_vram_free <label> -- bounded 60 s, 2 s steps
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

wait_no_proc() {   # wait_no_proc <procname> (pgrep -x) -- bounded 240 s;
                    # generic primitive, kept for parity with the source
                    # chains, NOT used on glm53 here (see wait_no_glm53)
  local p=$1
  for _ in $(seq 1 120); do pgrep -x "$p" >/dev/null || return 0; sleep 2; done
  echo "FATAL: $p still alive after 240 s"; return 1
}

wait_no_glm53() {   # bounded 240 s, 2 s steps; zombie-safe replacement for
                     # `wait_no_proc glm53` (see header's replacement list)
  for _ in $(seq 1 120); do engine_alive || return 0; sleep 2; done
  echo "FATAL: glm53 still alive (non-zombie) after 240 s"; return 1
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

# warm_and_verify <dir> <label> -- two passes, fincore-verified >=90% each
# (record bug 19, 2026-09-16: one `cat` under a near-full page cache can
# leave a file ~85% resident). Generalised to *.gguf, not *.safetensors --
# used only for the pre-search warm; every real ladder arm's own residency
# check is context_ladder.py's --warm/--min-resident 90 (generic to every
# file under --snap, ttft_serve's own second line of defense).
warm_and_verify() {
  local dir=$1 label=$2 pass pct
  for pass in 1 2; do
    echo "--- warming $label ($dir) pass $pass $(date -Is)"
    cat "$dir"/*.gguf > /dev/null 2>&1 || true
    pct=$(fincore --bytes --output SIZE,RES "$dir"/*.gguf 2>/dev/null \
          | tail -n +2 | awk '{ts+=$1; rs+=$2} END{if (ts>0) printf "%.4f", 100*rs/ts; else print 0}')
    echo "[resid $label-p$pass] resident=${pct}%"
    awk -v p="$pct" 'BEGIN{exit !(p>=90)}' && return 0
  done
  echo "WARNING: $label not >=90% resident after 2 warm passes"
  return 1
}

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT -u COLI_TIMERS \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh >> "$GLOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" \
         -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && break
    sleep 5
  done
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(engine_alive && echo 1 || echo 0)"
}

stop_gateway() {
  pkill -f "openai_[s]erver.py" 2>/dev/null || true
  sleep 3
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_glm53
}

wait_ready_f11() {   # wait_ready_f11 <port> <label> <timeout_s>
  local port=$1 label=$2 cap=$3 steps i
  steps=$(( (cap + 4) / 5 ))
  for i in $(seq 1 "$steps"); do
    [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)" = 200 ] && return 0
    [ -n "$F11_PID" ] && ! kill -0 "$F11_PID" 2>/dev/null && { echo "$label: FATAL -- docker run exited before /health=200"; return 1; }
    sleep 5
  done
  echo "$label: FATAL -- /health never returned 200 after ${cap}s"
  return 1
}

# --------------------------------------------------------------- F11 arm --
# State of whatever THIS chain started -- the exit trap must stop only this,
# never touch glm53 by name (only start_gateway's own stop_gateway call,
# run ONCE at preflight, ever does that -- see the header note).
F11_PID=""
F11_CONTAINER_UP=0
F11_CONTAINER_NAME=""

start_f11() {   # start_f11 <model> <ncmoe> <name> <alias>
  local model=$1 ncmoe=$2 name=$3 alias=$4
  docker run --rm --name "$name" \
    -p "127.0.0.1:${F11_PORT}:${F11_PORT}" \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc=host \
    -e "LD_LIBRARY_PATH=/opt/rocm/lib:${F11_BIN_DIR}" \
    -v /home/ronald:/home/ronald \
    "$F11_IMAGE" \
    "${F11_BIN_DIR}/llama-server" \
    -m "$model" -ngl 999 --n-cpu-moe "$ncmoe" \
    --tensor-split 1,1,1 --split-mode layer --device ROCm0,ROCm1,ROCm2 \
    -fa on -ctk q8_0 -ctv q8_0 -t 16 -tb 8 \
    --reasoning-effort low \
    --host 0.0.0.0 --port "$F11_PORT" \
    --parallel 1 --ctx-size 32768 --alias "$alias" \
    >> "$GOUT/${TAG}_${name}.log" 2>&1 < /dev/null &
  F11_PID=$!
  F11_CONTAINER_UP=1
  F11_CONTAINER_NAME="$name"
  echo "$name: docker run pid=$F11_PID container=$name port=$F11_PORT ncmoe=$ncmoe"
}

stop_f11() {
  if [ "$F11_CONTAINER_UP" = 1 ]; then
    docker stop -t 10 "$F11_CONTAINER_NAME" >/dev/null 2>&1 || true
    wait_no_container "$F11_CONTAINER_NAME"
    F11_CONTAINER_UP=0
  fi
  [ -n "$F11_PID" ] && { wait "$F11_PID" 2>/dev/null || true; }
  F11_PID=""
  F11_CONTAINER_NAME=""
}

# find_n_cpu_moe <model> <snap_dir> <n0> <step> <total_layers> <label>
# Sets FOUND_N/FOUND_VRAM0/FOUND_VRAM1/FOUND_VRAM2/FOUND_TRIES on success
# (return 0); on exhaustion prints a per-model (non chain-aborting) failure
# and returns 1 -- that model's arms are then skipped, the chain continues.
find_n_cpu_moe() {
  local model=$1 snap=$2 n0=$3 step=$4 total=$5 label=$6
  local try n name rc summary
  FOUND_N=""; FOUND_VRAM0=""; FOUND_VRAM1=""; FOUND_VRAM2=""; FOUND_TRIES=""
  echo "=== $label: --n-cpu-moe search, n0=$n0 step=$step total_layers=$total (max 6 tries) $(date -Is)"
  warm_and_verify "$snap" "$label-presearch"
  for try in $(seq 1 6); do
    n=$(( n0 + (try - 1) * step ))
    [ "$n" -gt "$total" ] && n=$total
    name="f11_search_${label}_try${try}"
    echo "--- $label search try $try/6: n-cpu-moe=$n $(date -Is)"
    precheck "${name}-pre"
    assert_vram_free "${name}-pre" || { echo "FATAL: $label search VRAM not free before try $try -- aborting chain (box-safety)"; exit 1; }
    start_f11 "$model" "$n" "$name" "$name"
    wait_ready_f11 "$F11_PORT" "$name" "$SEARCH_TIMEOUT"
    rc=$?
    if [ "$rc" = 0 ]; then
      FOUND_VRAM0=$(VRAM 0); FOUND_VRAM1=$(VRAM 1); FOUND_VRAM2=$(VRAM 2)
      echo "$label search try $try: n-cpu-moe=$n LOADED. VRAM card0=$FOUND_VRAM0 card1=$FOUND_VRAM1 card2=$FOUND_VRAM2"
      stop_f11
      assert_vram_free "${name}-post" || { echo "FATAL: $label search VRAM stuck after stop -- aborting chain (box-safety)"; exit 1; }
      FOUND_N=$n; FOUND_TRIES=$try
      summary="{\"tag\":\"$TAG\",\"model\":\"$label\",\"n0\":$n0,\"step\":$step,\"total_layers\":$total,\"chosen_n\":$n,\"tries\":$try,\"vram_used_bytes\":[$FOUND_VRAM0,$FOUND_VRAM1,$FOUND_VRAM2]}"
      echo "$summary" > "$OUT/${TAG}_${label}_search.json"
      echo "$label search summary: $summary"
      return 0
    fi
    echo "$label search try $try: n-cpu-moe=$n FAILED (no /health=200 within ${SEARCH_TIMEOUT}s)"
    stop_f11
    assert_vram_free "${name}-post-fail" || { echo "FATAL: $label search VRAM stuck after a failed try -- aborting chain (box-safety)"; exit 1; }
  done
  echo "$label: no working --n-cpu-moe found in 6 tries (up to N=$n of $total) -- $label's arms are SKIPPED, chain continues"
  summary="{\"tag\":\"$TAG\",\"model\":\"$label\",\"n0\":$n0,\"step\":$step,\"total_layers\":$total,\"chosen_n\":null,\"tries\":6}"
  echo "$summary" > "$OUT/${TAG}_${label}_search.json"
  return 1
}

# run_f11_arm <key:Q|D> <n> <model> <snap> <model_id> <ncmoe> <vram_note>
run_f11_arm() {
  local key=$1 n=$2 model=$3 snap=$4 model_id=$5 ncmoe=$6 vram_note=$7
  local tagarm="${key}${n}" name elog console json rc_ladder rc_sweep
  tagarm="${key}${n}"
  name="f11_arm_${tagarm}"
  elog="$GOUT/${TAG}_${tagarm}.log"
  console="$GOUT/${TAG}_${tagarm}_ladder_console.log"
  json="$OUT/${TAG}_${tagarm}.jsonl"
  echo "=== arm $tagarm ($model_id, HIP docker, -ngl 999 --n-cpu-moe $ncmoe) $(date -Is)"
  precheck "${tagarm}-pre"
  assert_vram_free "${tagarm}-pre" || { echo "FATAL: $tagarm VRAM not free before start -- aborting chain (box-safety)"; exit 1; }

  start_f11 "$model" "$ncmoe" "$name" "$model_id"
  wait_ready_f11 "$F11_PORT" "$tagarm" "$READY_TIMEOUT" || { stop_f11; assert_vram_free "${tagarm}-post-fail" || { echo "FATAL: $tagarm VRAM stuck after a failed start -- aborting chain"; exit 1; }; return 1; }
  echo "$tagarm VRAM-AT-LOAD card0=$(VRAM 0) card1=$(VRAM 1) card2=$(VRAM 2) (search chose n-cpu-moe=$ncmoe: $vram_note)"

  echo "throwaway request (not scored):"
  curl -s -m 180 "http://127.0.0.1:$F11_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_id\",\"temperature\":0,\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}" \
    >> "$GOUT/${TAG}_${tagarm}_throwaway.json" 2>&1

  # 1/2: ladder + followups (see glm53flash_ladder_chain.sh's header for why
  # this is always split into two invocations rather than one combined call
  # with --cold-sweep -- same reasoning applies here: nobody has measured
  # either model's prefill rate, so there is no principled way to predict
  # which side of the budget a single call would land on).
  echo "--- $tagarm ladder+followups (timeout ${LADDER_TIMEOUT}s)"
  timeout "$LADDER_TIMEOUT" python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$F11_PORT" --model-id "$model_id" \
      --steps 1024,1024,2048,4096,8192 --gen 128 --followups 2 \
      --snap "$snap" --min-resident 90 --warm \
      --arm "$key" --tag "$TAG" --json "$json" \
      --server-log "$elog" 2>&1 | tee -a "$console"
  rc_ladder=${PIPESTATUS[0]}
  if [ "$rc_ladder" = 124 ]; then
    echo "=== $tagarm ladder TIMEOUT (${LADDER_TIMEOUT}s cap) -- rows already written are kept; sweep half still attempted"
  else
    echo "=== $tagarm ladder exit=$rc_ladder"
  fi

  # 2/2: cold-sweep-only, second invocation, same json/tag/arm.
  echo "--- $tagarm cold-sweep-only, second invocation, same json (timeout ${SWEEP_TIMEOUT}s)"
  timeout "$SWEEP_TIMEOUT" python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$F11_PORT" --model-id "$model_id" \
      --steps "" --followups 0 --gen 128 \
      --cold-sweep 2048,4096,8192,16384 --sweep-offset-chars 300000 \
      --snap "$snap" --min-resident 90 --warm \
      --arm "$key" --tag "$TAG" --json "$json" \
      --server-log "$elog" 2>&1 | tee -a "$console"
  rc_sweep=${PIPESTATUS[0]}
  if [ "$rc_sweep" = 124 ]; then
    echo "=== $tagarm cold-sweep TIMEOUT (${SWEEP_TIMEOUT}s cap) -- continuing chain"
  else
    echo "=== $tagarm cold-sweep exit=$rc_sweep"
  fi

  echo "=== $tagarm json=$json ladder_rc=$rc_ladder sweep_rc=$rc_sweep"
  stop_f11
  assert_vram_free "${tagarm}-post" || { echo "FATAL: $tagarm VRAM stuck after stop -- aborting chain (box-safety)"; exit 1; }
  [ "$rc_ladder" = 0 ] && [ "$rc_sweep" = 0 ]
}

# print_summary_table <tagarm> <json> -- turn/depth/decode/ttft, stdlib only
print_summary_table() {
  local tagarm=$1 json=$2
  if [ ! -f "$json" ]; then
    echo "  ($tagarm: no jsonl at $json -- arm did not run or produced no rows)"
    return
  fi
  python3 - "$tagarm" "$json" <<'PYEOF'
import json, sys
tagarm, path = sys.argv[1], sys.argv[2]
print(f"--- summary: {tagarm} ({path})")
print(f"{'turn':>4} {'kind':>10} {'depth_tok':>10} {'decode_tps':>11} {'ttft_s':>8}")
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('kind') not in ('ladder', 'followup', 'cold-sweep'):
            continue
        turn = r.get('turn', '-')
        kind = r.get('kind', '-')
        depth = r.get('prompt_tokens')
        if depth is None:
            depth = r.get('target_new', '-')
        dtps = r.get('decode_tps')
        ttft = r.get('ttft_s')
        print(f"{str(turn):>4} {kind:>10} {str(depth):>10} "
              f"{(f'{dtps:.2f}' if dtps is not None else '-'):>11} "
              f"{(f'{ttft:.2f}' if ttft is not None else '-'):>8}")
PYEOF
}

# ------------------------------------------------------------------- exit --
on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  stop_f11   # idempotent no-op if the last arm/search try already stopped cleanly
  echo "--- final re-warm of GLM before restart"
  warm_glm
  assert_glm_resident "final" || echo "WARNING: GLM not >=90% resident at restart time"
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== f11_ladder_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== results: $OUT (jsonl/search-json), $GOUT (server logs/consoles)"
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
[ -f "$Q_MODEL" ] || { echo "FATAL: $Q_MODEL missing"; exit 1; }
[ -f "$D_MODEL" ] || { echo "FATAL: $D_MODEL missing"; exit 1; }
[ -x "${F11_BIN_DIR}/llama-server" ] || { echo "FATAL: ${F11_BIN_DIR}/llama-server missing or not executable on the host (bind-mounted into the container)"; exit 1; }

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

# --------------------------------------------------------- n-cpu-moe search --
Q_N=""; D_N=""
Q_VRAM_NOTE=""; D_VRAM_NOTE=""
if find_n_cpu_moe "$Q_MODEL" "$Q_SNAP_DIR" "$Q_N0" "$Q_STEP" "$Q_LAYERS" "q"; then
  Q_N=$FOUND_N
  Q_VRAM_NOTE="card0=$FOUND_VRAM0 card1=$FOUND_VRAM1 card2=$FOUND_VRAM2 (try $FOUND_TRIES/6)"
fi
if find_n_cpu_moe "$D_MODEL" "$D_SNAP_DIR" "$D_N0" "$D_STEP" "$D_LAYERS" "d"; then
  D_N=$FOUND_N
  D_VRAM_NOTE="card0=$FOUND_VRAM0 card1=$FOUND_VRAM1 card2=$FOUND_VRAM2 (try $FOUND_TRIES/6)"
fi
echo "=== search done: Q_N=${Q_N:-SKIPPED} ($Q_VRAM_NOTE)  D_N=${D_N:-SKIPPED} ($D_VRAM_NOTE)"

# ------------------------------------------------------------------- arms --
[ -n "$Q_N" ] && run_f11_arm Q 1 "$Q_MODEL" "$Q_SNAP_DIR" "$Q_MODEL_ID" "$Q_N" "$Q_VRAM_NOTE"
[ -n "$D_N" ] && run_f11_arm D 1 "$D_MODEL" "$D_SNAP_DIR" "$D_MODEL_ID" "$D_N" "$D_VRAM_NOTE"
[ -n "$D_N" ] && run_f11_arm D 2 "$D_MODEL" "$D_SNAP_DIR" "$D_MODEL_ID" "$D_N" "$D_VRAM_NOTE"
[ -n "$Q_N" ] && run_f11_arm Q 2 "$Q_MODEL" "$Q_SNAP_DIR" "$Q_MODEL_ID" "$Q_N" "$Q_VRAM_NOTE"

# --------------------------------------------------------------- summary --
echo "--- per-arm summary tables"
Q1_J="$OUT/${TAG}_Q1.jsonl"; Q2_J="$OUT/${TAG}_Q2.jsonl"
D1_J="$OUT/${TAG}_D1.jsonl"; D2_J="$OUT/${TAG}_D2.jsonl"
print_summary_table "Q1 (n-cpu-moe=${Q_N:-n/a})" "$Q1_J"
print_summary_table "D1 (n-cpu-moe=${D_N:-n/a})" "$D1_J"
print_summary_table "D2 (n-cpu-moe=${D_N:-n/a})" "$D2_J"
print_summary_table "Q2 (n-cpu-moe=${Q_N:-n/a})" "$Q2_J"
echo "--- chosen N / VRAM-at-load"
echo "  Q: n-cpu-moe=${Q_N:-SKIPPED} $Q_VRAM_NOTE"
echo "  D: n-cpu-moe=${D_N:-SKIPPED} $D_VRAM_NOTE"

echo "--- context_compare.py: Q vs D (arm_of() groups Q1/Q2 -> Q, D1/D2 -> D by leading letters -- checked, not invented)"
rows=()
[ -f "$Q1_J" ] && rows+=("Q1=$Q1_J")
[ -f "$Q2_J" ] && rows+=("Q2=$Q2_J")
[ -f "$D1_J" ] && rows+=("D1=$D1_J")
[ -f "$D2_J" ] && rows+=("D2=$D2_J")
if [ "${#rows[@]}" -ge 1 ]; then
  python3 "$HERE/context_compare.py" --rows "${rows[@]}"
  echo "--- context_compare.py exit=$? (REFUSED/NO VERDICT is an accepted outcome)"
else
  echo "=== context_compare.py: SKIPPED, no jsonl files present (both models' searches failed)"
fi

echo "=== f11_ladder_chain body done $(date -Is) -- exit trap runs re-warm/restart/accept_live.sh next"
exit 0
