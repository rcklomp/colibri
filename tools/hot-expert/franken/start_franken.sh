#!/bin/bash
# start_franken.sh -- serve the Franken engine over Colibri's OpenAI gateway,
# on the owner's own port, key and model id.
#
# This is ~/start_glm53.sh with three things changed and nothing else: the
# engine binary, the model, and the family. Everything the owner's Open WebUI
# depends on is kept BYTE-IDENTICAL to that file -- port 8081, the key in
# ~/.colibri_api_key, --model-id glm-5.3-flash, --max-tokens 16384 (FRANKEN_MAX_TOKENS;
# GLM's launcher uses 4096 -- a thinking Qwen3.8 answer needs room: record §L0-QUALITY
# scored 10 truncations at 4096 as wrong), the same
# --allowed-host list -- so the UI reaches whatever is behind the port with no
# change at all (serve_alt.sh's header documents why that works: Open WebUI's
# connection #1 is http://host.docker.internal:8081/v1 with that key, and its
# single configured model id is "glm-5.3-flash").
#
# NOT RUN by the agent that wrote it: it stops nothing and starts nothing on
# its own, but the moment it runs it takes the port and the three cards. The
# orchestrating session launches it, under the rig lock, through
# `tools/hot-expert/serve_alt.sh franken` -- which is also what puts the lock
# in place and what hands the box back to GLM afterwards.
#
# --- WHY --arch qwen38 AND --model-id glm-5.3-flash AT THE SAME TIME --------
# The gateway's --arch decides which chat template renders the prompt
# (render_chat_for_arch, c/openai_server.py:1985-2004). The model this engine
# runs IS Qwen3.8-Flash-Next, so its own template is the only correct one:
# rendering GLM-5.3's <|user|>/<|assistant|> markers into a Qwen tokenizer
# would produce a prompt the model has never seen. --model-id is a separate
# thing entirely -- the name the API advertises -- and it stays the name Open
# WebUI already has. One is correctness, the other is plumbing.
#
# --arch is resolved from the MODEL DIRECTORY's config.json (resolve_model,
# c/family_registry.py:1283), and a GGUF directory has none. So --model points
# at the FP8 checkpoint directory, which carries `"model_type": "qwen4_exp"`
# and therefore resolves to the qwen38 family, while the engine reads its
# weights from the GGUF named by FRANKEN_GGUF. The gateway never opens either.
#
# --- WHAT THE ENGINE TAKES FROM THE ENVIRONMENT ----------------------------
# The gateway launches the engine as `[binary, <cap>]` and passes no flags, so
# every setting is an env var (decode/franken_serve.cpp):
#   FRANKEN_GGUF     the shard to load (the other two are found beside it)
#   FRANKEN_CTX      cells per KV slot -- 262144, the design's target window
#   FRANKEN_CHUNK    prefill chunk (256; the cap is 512, decode is chunk 1)
#   FRANKEN_DEVICES  3
#   FRANKEN_GEMM_LDS 0 = the wave-per-row trunk GEMM, bit-identical to decode.
#                    1 and 2 reassociate K and are a knob, not a default
#                    (record §L0-PREFILL-2)
#   KV_SLOTS         set by the gateway from --kv-slots
# Slots are allocated lazily and one slot is 1.21 GB a card at 262144 cells,
# against the 2.4-3.3 GB each card has left after the weights -- which is why
# --kv-slots is 1 here. It is also the maximum the qwen38 family allows
# (FamilyLimits max_kv_slots=1, c/family_registry.py:1156), so the gateway
# would refuse a larger number anyway.
set -u

# FRANKEN_BIN may be the engine itself or a wrapper that execs it (the docker
# wrapper the first served run used, ~/bench/franken_decode_docker.sh, is one:
# the gateway only ever runs `<BIN> <cap>` with SERVE=1 in the environment, so
# anything that forwards stdin/stdout/stderr and the environment will do).
# Whatever it is, the PROCESS the engine runs as must still be called
# franken_decode: serve_alt.sh's `ps -C franken_decode` is how the rig lock's
# keeper and `serve_alt.sh status` find it.
BIN=${FRANKEN_BIN:-$HOME/src/colibri/tools/hot-expert/franken/decode/franken_decode}
GGUF=${FRANKEN_GGUF:-$HOME/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
GGUF_DIR=$(dirname "$GGUF")
# The directory with a config.json, for the family resolver only.
FAMILY_DIR=${FRANKEN_FAMILY_DIR:-$HOME/models/Qwen3.8-Flash-Next-FP8}

[ -x "$BIN" ]  || { echo "[start] FATAL: no engine at $BIN (set FRANKEN_BIN)"; exit 1; }
[ -s "$GGUF" ] || { echo "[start] FATAL: no model at $GGUF"; exit 1; }
[ -s "$FAMILY_DIR/config.json" ] || {
  echo "[start] FATAL: $FAMILY_DIR/config.json is what --arch qwen38 is resolved from"; exit 1; }

cd "$HOME/src/colibri/c" || exit 1

# CLAUDE.md: every recorded number on this box is an 8-thread number. The host
# work in this engine is the PLE gather and the sampler, but the CPU backend
# and any future host path must see the same pinning as everything else.
export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close

export FRANKEN_GGUF="$GGUF"
export FRANKEN_CTX=${FRANKEN_CTX:-262144}
export FRANKEN_CHUNK=${FRANKEN_CHUNK:-256}
export FRANKEN_DEVICES=${FRANKEN_DEVICES:-3}
# --gemm-lds 1 IN SERVICE (2026-09-23). 0 is the wave-per-row trunk GEMM, which
# is bit-identical to the decode kernel; 1 is the LDS-tiled one, which decodes a
# weight block once per 64 token columns instead of per 8 and reassociates K.
# It is the difference between llama.cpp's prefill rate and half of it (record
# §L0-PREFILL-2: 3.0 -> 1.63 ms a token), it applies to T > 1 only -- decode is
# untouched and stays bit-identical either way -- and its divergence has not yet
# been measured against the CPU reference. That is the open item: if the quality
# harness ever says this arm is not acceptable, this is the one line to change.
export FRANKEN_GEMM_LDS=${FRANKEN_GEMM_LDS:-1}

# Design 9.1's PLE table, resident. The gather reads 16 rows a token scattered
# over 28.8 GB; through the GGUF's mmap a row that is not page-cached costs
# 4.2 ms against 9.5 us (record §PLE-GATHER), and the first served run spent
# 3.8 s of a 42 s prefill there. Read once at start into memory this process
# owns, after which no gather can fault. ~29 GB of host RAM, reported with its
# load time at boot. FRANKEN_PLE_PINNED=1 uses hipHostMalloc instead of malloc;
# it is not the default because nothing DMAs from this table (the gather
# dequantises on the CPU), so pinning costs registration time for a transfer
# that never happens.
export FRANKEN_PLE_RESIDENT=${FRANKEN_PLE_RESIDENT:-1}
export FRANKEN_PLE_PINNED=${FRANKEN_PLE_PINNED:-0}
# Prefix reuse. A checkpoint is taken at every prefill chunk boundary and once
# at the end of every request (the point a follow-up turn resumes from), eight
# kept per slot inside a host-memory budget. Each costs ~118 MB of recurrent
# state, plus ~13.8 kB a token of CELLS if and when something is about to
# overwrite them -- see franken_serve.cpp's `struct Snapshot`. They are what
# lets a second conversation reuse a shared system/tool block after another
# request has used the same slot; accept_live.sh check 2 wants
# reused >= prompt_tokens - 256, which is why the interval is the chunk.
# Halving FRANKEN_SNAP_EVERY is the lever if a real prompt's shared prefix ends
# just above a boundary; it costs one more 118 MB download per interval.
export FRANKEN_SNAP_EVERY=${FRANKEN_SNAP_EVERY:-512}
export FRANKEN_SNAP_KEEP=${FRANKEN_SNAP_KEEP:-8}
export FRANKEN_SNAP_BUDGET_MB=${FRANKEN_SNAP_BUDGET_MB:-6144}
export FRANKEN_SNAP_TURNS=${FRANKEN_SNAP_TURNS:-3}
# --- THE ENV DIFF AGAINST ~/start_glm53.sh ----------------------------------
# Everything that file exports falls into three groups. The engine-specific
# ones (COLI_VULKAN, COLI_VK_DEV*/EXPERTS*/SHADERS, COLI_USAGE_PATH,
# COLI_KDA_GPU, GLM53_VK_SWIGLU_CLAMP, GLM53_PREFILL_CHUNK, GLM53_MLA_ATTN_GPU,
# GLM53_MOE_ONE_TEAM, GLM53_I4_FAST, GLM53_PREFIX_CKPT, GLM53_MAXT,
# GLM53_VERBOSE) are read by glm53.c and by nothing else; this engine's
# equivalents are the FRANKEN_* block above. That leaves the GATEWAY-side ones,
# which are family-neutral and are the ones that had to be copied:
#
#   COLI_REQ_LOG=1     the one [req] line a request, which ~/bench/owui_report.sh
#                      and accept_live.sh's idle check both read.
#   COLI_PREFIX_PIN=1  P7. Open WebUI rebuilds its `memory_context` system block
#                      every turn; pin_context_blocks (openai_server.py:2371)
#                      replaces it with the conversation's FIRST one, so the
#                      shared prefix stays byte-identical across turns. Without
#                      it the prompt changes near its head and prefix reuse --
#                      the thing this engine's checkpoints exist for -- is lost
#                      on every turn. It is a no-op while the ledger is on, and
#                      the ledger is off here (see COLI_LEDGER).
#   COLI_THINK=0       Qwen3.8's gateway default is xhigh thinking
#                      (openai_server.py:5041), which is why the first served
#                      run answered every UI turn with a reasoning_content
#                      block in front of it. The GLM path the owner is used to
#                      serves with thinking off; this is the same switch, and
#                      the renderer closes the block in the prompt
#                      (`<think>\n\n</think>`) rather than hoping the model
#                      stops on its own. An explicit client reasoning_effort
#                      still wins.
#   COLI_LEDGER=1      set for parity and NOT sufficient: ledger_enabled()
#                      (openai_server.py:2703-2706) is `ARCH == "glm53" and
#                      COLI_LEDGER != "0"`, so the conversation ledger -- and
#                      with it accept_live.sh check 2b -- is off BY CODE for
#                      every other family, whatever this variable says. 2b
#                      reporting SKIP against this engine is that, not a
#                      configuration mistake. Widening the gate is a gateway
#                      change with a real risk attached: _ledger_record reads
#                      `plan.parts`, which only render_chat_glm53 produces
#                      (openai_server.py:1998), so a ledger on qwen38 would
#                      record part-less entries and can log `ledger=broken` --
#                      which check 2b fails on, turning a SKIP into a FAIL.
export COLI_REQ_LOG=1
export COLI_PREFIX_PIN=1
export COLI_THINK=0
export COLI_LEDGER=1

# The gateway refuses to bind 0.0.0.0 without a key -- same key file, same
# permissions rule as start_glm53.sh.
if [ ! -s "$HOME/.colibri_api_key" ]; then
  umask 077; head -c 24 /dev/urandom | base64 | tr -d "=+/" | cut -c1-32 > "$HOME/.colibri_api_key"
fi
export COLI_API_KEY="$(cat "$HOME/.colibri_api_key")"

if [ "${SKIP_WARM:-0}" != "1" ]; then
  echo "[start] warming $GGUF_DIR -- 87 GiB, a few minutes"
  cat "$GGUF_DIR"/*.gguf > /dev/null 2>&1 || true
fi
echo "[start] engine   : $BIN"
echo "[start] model    : $GGUF"
echo "[start] family   : qwen38 (from $FAMILY_DIR/config.json), advertised as 'glm-5.3-flash'"
echo "[start] ctx=$FRANKEN_CTX chunk=$FRANKEN_CHUNK devices=$FRANKEN_DEVICES gemm_lds=$FRANKEN_GEMM_LDS"
# The log this gateway writes to, printed in a form `serve_alt.sh status` and
# accept_live.sh's caller can both pick up: accept_live.sh reads $GLM53_LOG (it
# defaults to ~/glm53_server.log, which is GLM's and is NOT this one -- its
# check 1 polled that file for 18 minutes on 2026-09-23 because nobody told it).
echo "[start] gateway log ${FRANKEN_LOG:-$HOME/bench/serve_alt_franken.log}"
echo "[start] launching gateway on 0.0.0.0:8081"
# NB (inherited from start_glm53.sh, and it cost a night there): do NOT put a
# comment line BETWEEN the backslash-continued arguments below. bash joins the
# logical line, the # comments out everything after it, and the gateway comes
# up silently missing every argument that followed -- `bash -n` still passes.
python3 -u openai_server.py \
  --model "$FAMILY_DIR" \
  --engine "$BIN" \
  --arch qwen38 \
  --host 0.0.0.0 --port 8081 \
  --model-id glm-5.3-flash \
  --max-tokens "${FRANKEN_MAX_TOKENS:-16384}" \
  --kv-slots 1 \
  --allowed-host 127.0.0.1 --allowed-host localhost \
  --allowed-host host.docker.internal \
  --allowed-host rome.local 2>&1 | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }'
