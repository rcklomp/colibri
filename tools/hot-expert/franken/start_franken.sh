#!/bin/bash
# start_franken.sh -- serve the Franken engine over Colibri's OpenAI gateway,
# on the owner's own port, key and model id.
#
# This is ~/start_glm53.sh with three things changed and nothing else: the
# engine binary, the model, and the family. Everything the owner's Open WebUI
# depends on is kept BYTE-IDENTICAL to that file -- port 8081, the key in
# ~/.colibri_api_key, --model-id glm-5.3-flash, --max-tokens 4096, the same
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
export FRANKEN_GEMM_LDS=${FRANKEN_GEMM_LDS:-0}
# Prefix reuse: one snapshot of the recurrent state per prefill chunk, eight
# kept per slot. A snapshot is ~118 MB of HOST memory and one download a card;
# they are what lets a second conversation reuse a shared system/tool block
# (accept_live.sh check 2 wants reused >= prompt_tokens - 256, which is the
# chunk). FRANKEN_SNAP_EVERY in TOKENS, 0 keeps the default.
export FRANKEN_SNAP_EVERY=${FRANKEN_SNAP_EVERY:-256}
export FRANKEN_SNAP_KEEP=${FRANKEN_SNAP_KEEP:-8}
# The [req] accounting line, same as the GLM gateway.
export COLI_REQ_LOG=1

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
  --max-tokens 4096 \
  --kv-slots 1 \
  --allowed-host 127.0.0.1 --allowed-host localhost \
  --allowed-host host.docker.internal \
  --allowed-host rome.local 2>&1 | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }'
