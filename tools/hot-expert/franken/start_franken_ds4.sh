#!/bin/bash
# start_franken_ds4.sh -- serve the Franken engine on DeepSeek-V4-Flash over
# Colibri's OpenAI gateway (M1), on the owner's own port, key and model id.
#
# This is tools/hot-expert/franken/start_franken.sh with the same three
# things changed and nothing else, a second time: the engine binary
# (franken_decode_ds4), the model (DeepSeek-V4-Flash-0731 UD-IQ2_M), and the
# family (deepseek_v4 instead of qwen38). Everything the owner's Open WebUI
# depends on stays byte-identical to that file and to ~/start_glm53.sh --
# port 8081, the key in ~/.colibri_api_key, --model-id glm-5.3-flash, the
# same --allowed-host list -- so the UI reaches whatever is behind the port
# with no change at all. --max-tokens is 16384 here too: DeepSeek-V4's own
# template has a thinking mode (render_chat_v4, c/openai_server.py) and a
# thinking answer needs the same room Qwen3.8's did (start_franken.sh's own
# comment, record §L0-QUALITY).
#
# --- WHY --arch deepseek_v4 AND --model-id glm-5.3-flash AT THE SAME TIME --
# Same reasoning as start_franken.sh's own header, restated for this model:
# --arch decides the chat template (render_chat_for_arch), and the model this
# engine runs IS DeepSeek-V4-Flash, so render_chat_v4 is the only correct
# template -- rendering GLM-5.3's or Qwen's markers into this tokenizer would
# produce a prompt the model has never seen. --model-id is unrelated plumbing:
# it stays the name Open WebUI already has.
#
# --arch is resolved from the MODEL DIRECTORY's config.json (resolve_model,
# c/family_registry.py:1283), and a GGUF directory has none -- exactly
# start_franken.sh's problem, with one difference worth being honest about:
# there is no real upstream DeepSeek-V4-Flash-0731 config.json anywhere on
# this rig (only the GGUF was ever ingested; DEEPSEEK4.md section 1 is where
# its architecture numbers actually come from, read out of the GGUF header).
# FRANKEN_FAMILY_DIR below therefore points at a STUB config.json this task
# wrote (`{"model_type": "deepseek_v4", ...}`, with the geometry fields
# family_registry.py's planner would want populated from ds4_shapes.h's own
# already-asserted constants, and a long "_colibri_note" explaining exactly
# which fields are load-bearing for SERVING (only model_type is) versus
# populated-but-not-required (everything _dsv4_geometry() would read) versus
# deliberately left out (anything whose real HF field name isn't verified).
# Read that note before trusting any other field in it. The gateway never
# opens the GGUF either way -- FRANKEN_GGUF below names the shard the engine
# reads its weights from.
#
# --- WHAT THE ENGINE TAKES FROM THE ENVIRONMENT -----------------------------
# The gateway launches the engine as `[binary, <cap>]` and passes no flags,
# so every setting is an env var (decode/ds4_serve.cpp):
#   FRANKEN_GGUF       the shard to load (the other two are found beside it)
#   FRANKEN_CTX        cells per KV slot -- 262144, same target window as Qwen
#   FRANKEN_CHUNK      prefill chunk (256; cap 512, decode is chunk 1)
#   FRANKEN_DEVICES    3
#   FRANKEN_GEMM_LDS   default 1 here (ds4_serve.cpp), NOT 0 as Qwen's franken
#                      serve defaults to -- DEEPSEEK4.md section 13.3 measured
#                      it as the batched-prefill rate on this engine; override
#                      down to 0 (bit-identical to decode) if a quality gate
#                      ever says otherwise, exactly start_franken.sh's own
#                      open item for Qwen's LDS arm.
#   FRANKEN_PLACEMENT  ~/bench/m2/deepseek_mix (M2 histogram; DEEPSEEK4.md
#                      section 5/12) -- which experts start resident per card
#   FRANKEN_EXPERT_GB  20 -- VRAM a card spends on resident experts
#   FRANKEN_ADAPT      1 -- adaptive placement (design L2, DEEPSEEK4.md
#                      section 12) learns the hot experts from THIS
#                      conversation's own routing and swaps them in, on top
#                      of the M2 histogram's static starting point
#   KV_SLOTS           set by the gateway from --kv-slots (1: the
#                      deepseek_v4 family's own max_kv_slots, same cap as
#                      qwen38's -- family_registry.py)
# Same reason as Qwen's launcher: one slot is far larger than the 2-3 GB a
# card has left after the weights at full context, so --kv-slots is 1.
set -u

# FRANKEN_BIN may be the engine itself or the docker wrapper
# (~/bench/franken_decode_docker.sh) that runs it inside the ROCm 7.14 image
# this box's host ROCm (6.2) cannot satisfy libllama's tokenizer dependency
# on (same reason start_franken.sh's FRANKEN_BIN can be that wrapper). The
# PROCESS must still be called franken_decode_ds4: serve_alt.sh's
# franken_ds4_alive/pgrep -f "franken_decode_[d]s4" is how the rig lock's
# keeper and `serve_alt.sh status` find it (its comm truncates at 15 chars,
# see serve_alt.sh's D4_* header comment -- pgrep -f matches the untruncated
# argv instead of comm for exactly that reason).
BIN=${FRANKEN_DS4_BIN:-$HOME/bench/franken_decode_ds4_docker.sh}   # the engine lives in frankenstack/franken-engine since 2026-09-24; this tree has no binary
GGUF=${FRANKEN_GGUF:-$HOME/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf}
GGUF_DIR=$(dirname "$GGUF")
# The directory with a (stub) config.json, for the family resolver only.
FAMILY_DIR=${FRANKEN_FAMILY_DIR:-$HOME/models/DeepSeek-V4-Flash-family}

[ -x "$BIN" ]  || { echo "[start] FATAL: no engine at $BIN (set FRANKEN_DS4_BIN)"; exit 1; }
[ -s "$GGUF" ] || { echo "[start] FATAL: no model at $GGUF"; exit 1; }
[ -s "$FAMILY_DIR/config.json" ] || {
  echo "[start] FATAL: $FAMILY_DIR/config.json is what --arch deepseek_v4 is resolved from"; exit 1; }

cd "$HOME/src/colibri/c" || exit 1

# CLAUDE.md: every recorded number on this box is an 8-thread number.
export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close

export FRANKEN_GGUF="$GGUF"
export FRANKEN_CTX=${FRANKEN_CTX:-262144}
export FRANKEN_CHUNK=${FRANKEN_CHUNK:-256}
export FRANKEN_DEVICES=${FRANKEN_DEVICES:-3}
export FRANKEN_GEMM_LDS=${FRANKEN_GEMM_LDS:-1}
export FRANKEN_PLACEMENT=${FRANKEN_PLACEMENT:-$HOME/bench/m2/deepseek_mix}
export FRANKEN_EXPERT_GB=${FRANKEN_EXPERT_GB:-20}
export FRANKEN_ADAPT=${FRANKEN_ADAPT:-1}
# DO NOT set FRANKEN_HIP_GRAPH=1 while FRANKEN_ADAPT=1 (and FRANKEN_MISS_STAGE is at its default 1): the three together run
# 81 % slower (51 -> 92 ms a token at depth 53, any depth); any two are fine, and the engine's own default is graph OFF. Record
# §DS4-ADAPT-GRAPH (2026-10-06); the mechanism is not isolated, so this is a rule, not a fix.

# Prefix reuse (ds4_serve.cpp's Slot/Snapshot, ported from franken_serve.cpp's
# own -- see that file's header comment for why checkpointing is not optional
# with one KV slot). Same defaults as Qwen's launcher for the same reason:
# every number the design targeted was taken at these.
export FRANKEN_SNAP_EVERY=${FRANKEN_SNAP_EVERY:-512}
export FRANKEN_SNAP_KEEP=${FRANKEN_SNAP_KEEP:-8}
export FRANKEN_SNAP_BUDGET_MB=${FRANKEN_SNAP_BUDGET_MB:-6144}
export FRANKEN_SNAP_TURNS=${FRANKEN_SNAP_TURNS:-3}

# --- THE ENV DIFF AGAINST start_franken.sh ----------------------------------
# Same three gateway-side, family-neutral knobs Qwen's launcher sets, for the
# same reasons (that file's own comment on each):
#   COLI_REQ_LOG=1     the [req] line owui_report.sh/accept_live.sh read.
#   COLI_PREFIX_PIN=1  P7's pinned per-turn context block; a no-op while the
#                      ledger is on (COLI_LEDGER below), harmless either way.
#   COLI_LEDGER=1      set for parity, NOT sufficient: ledger_enabled() is
#                      ARCH == "glm53" only, so the ledger -- and check 2b --
#                      is off BY CODE for deepseek_v4 too, exactly as it is
#                      for qwen38 (start_franken.sh's own comment explains
#                      the gateway-side risk of widening that gate).
# COLI_THINK is NOT forced to 0 here, unlike Qwen's launcher: ARCH ==
# "deepseek_v4" never hits the qwen38-only xhigh-default branch in
# c/openai_server.py (that branch is literally `if ARCH == "qwen38" and
# COLI_THINK != "0"`), so a client that sends neither reasoning_effort nor
# enable_thinking already gets thinking OFF by default on this family
# (the generic `elif COLI_THINK == "1"` branch, default "0") -- matching what
# the owner is used to from GLM without needing an explicit override. Set
# COLI_THINK=1 to make thinking the default here instead, the same switch
# Qwen's launcher uses to opt OUT.
export COLI_REQ_LOG=1
export COLI_PREFIX_PIN=1
export COLI_LEDGER=1

if [ ! -s "$HOME/.colibri_api_key" ]; then
  umask 077; head -c 24 /dev/urandom | base64 | tr -d "=+/" | cut -c1-32 > "$HOME/.colibri_api_key"
fi
export COLI_API_KEY="$(cat "$HOME/.colibri_api_key")"

if [ "${SKIP_WARM:-0}" != "1" ]; then
  echo "[start] warming $GGUF_DIR -- 85 GB, several minutes"
  cat "$GGUF_DIR"/*.gguf > /dev/null 2>&1 || true
fi
echo "[start] engine   : $BIN"
echo "[start] model    : $GGUF"
echo "[start] family   : deepseek_v4 (from $FAMILY_DIR/config.json), advertised as 'glm-5.3-flash'"
echo "[start] ctx=$FRANKEN_CTX chunk=$FRANKEN_CHUNK devices=$FRANKEN_DEVICES gemm_lds=$FRANKEN_GEMM_LDS placement=$FRANKEN_PLACEMENT expert_gb=$FRANKEN_EXPERT_GB adapt=$FRANKEN_ADAPT"
echo "[start] gateway log ${FRANKEN_LOG:-$HOME/bench/serve_alt_franken_ds4.log}"
echo "[start] launching gateway on 0.0.0.0:8081"
# NB (inherited from start_glm53.sh/start_franken.sh, and it cost a night the
# first time): do NOT put a comment line BETWEEN the backslash-continued
# arguments below -- bash joins the logical line and the # comments out
# everything after it silently.
python3 -u openai_server.py \
  --model "$FAMILY_DIR" \
  --engine "$BIN" \
  --arch deepseek_v4 \
  --host 0.0.0.0 --port 8081 \
  --model-id glm-5.3-flash \
  --max-tokens "${FRANKEN_MAX_TOKENS:-16384}" \
  --kv-slots 1 \
  --allowed-host 127.0.0.1 --allowed-host localhost \
  --allowed-host host.docker.internal \
  --allowed-host rome.local 2>&1 | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }'
