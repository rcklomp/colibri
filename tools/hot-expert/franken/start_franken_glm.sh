#!/bin/bash
# start_franken_glm.sh -- serve the Franken engine on GLM-5.3-Flash over Colibri's OpenAI
# gateway, exactly as start_franken_ds4.sh does for DeepSeek-V4-Flash (that file's own header
# explains the shape this one copies: port 8081, the key in ~/.colibri_api_key, --model-id
# glm-5.3-flash, the same --allowed-host list -- so Open WebUI reaches whatever is behind the
# port with no change at all).
#
# --- WHY --arch glm53, UNLIKE start_franken_ds4.sh's --arch deepseek_v4 -----------------------
# start_franken_ds4.sh needs a DIFFERENT arch than the model it serves is ACTUALLY named,
# because deepseek_v4 has no real HF config.json on this rig (that file's own header explains
# the synthetic-stub problem in full). GLM-5.3-Flash has no such problem: the model THIS engine
# runs (~/models/GLM-5.3-Flash/UD-IQ4_XS, general.architecture glm5next in the GGUF header) is
# the exact same model the COLIBRI glm53 engine already serves, just a different quantisation
# and a different C++ engine behind the same gateway line protocol. So --arch glm53 here is not
# a workaround -- it IS the correct architecture, and it buys the Franken GLM arm the SAME
# chat template, stop tokens and thinking-mode handling (render_chat_for_arch, c/openai_server.py)
# the owner already gets from the real glm53 engine, with no template code to port or maintain
# separately. --model-id stays glm-5.3-flash so Open WebUI's model picker needs no change.
#
# FRANKEN_FAMILY_DIR points at ~/models/GLM-5.3-Flash-colibri-int4-g64 -- an EXISTING container
# of this same model that already has a real config.json (model_type "glm5_next", matching
# family_registry.py's glm53 registration `model_types=("glm5_next", "glm5_next_text")`). This
# is read-only, for family/arch resolution ONLY (resolve_model, c/family_registry.py:1283); the
# gateway never opens its weights -- FRANKEN_GGUF below names the actual shard this engine reads.
# Unlike start_franken_ds4.sh's stub, no synthetic config.json was written for this: a real one
# for this real model already existed on the rig.
#
# --- WHAT THE ENGINE TAKES FROM THE ENVIRONMENT (glm5_serve.cpp) ---------------------------
#   FRANKEN_GGUF          the shard to load (the other four shards are found beside it)
#   FRANKEN_CTX            cells per KV slot -- 262144, same target window as DS4/Qwen
#   FRANKEN_GLM_CHUNK       prefill chunk, 1024 since 2026-10-06 (record §L5-GLM-CHUNK: every chunk streams ALL the staged experts, so DMA bytes a
#                           token fall as 1/chunk; -10 % at the served context, bit-exact; needs franken-engine glm-prefill-final, an older binary
#                           clamps it to 512 and still runs); decode is chunk 1
#   FRANKEN_DEVICES         3
#   FRANKEN_GEMM_LDS        1 here since 2026-09-25 (served ON, as Qwen3.8 is: prefill 16.9 -> 9.3 ms/token, record
#                           §L5-GLM-LDS; a last-bits order change, judged by the quality run). The engine's own
#                           default (glm5_serve.cpp) is 0; the export below overrides it.
#   FRANKEN_GEMV_FUSED_REDUCE  0 since 2026-10-07 (record §L5-GLM-GEMV-REDUCE: -4.7 % decode, bit-exact; 1 = the in-kernel fused reduce)
#   FRANKEN_GLM_MOE_G       8 since 2026-10-07 (4 before; 1|2|4|8|16 are valid; record §L5-GLM-G8: prefill -2.6 %, bit-exact). Only the docker wrapper's
#                           allowlist carries it into the container (rig_copies/franken_decode_glm_docker.sh: it forwards FRANKEN_GLM_MOE_G and the FRANKEN_GEMV_* /
#                           FRANKEN_GLM_FETCH_ASSIGN knobs since 2026-10-07; before that a knob not on the list was DROPPED silently)
#   FRANKEN_GLM_PREFILL_STAGE  0 since 2026-10-06 (was 1: GLM5.md section 12's prefill miss path; a chunk's missed
#                           experts DMA'd into a VRAM ring instead of read in place)
#   FRANKEN_ADAPT_PREFILL   0 -- GLM5.md section 13: hold adaptation swaps during a chunked
#                           prefill, resume at the first decode token (this port's own served
#                           brief: "adapt-prefill 0 for prefill, adaptation on for decode")
#   FRANKEN_PLACEMENT       ~/bench/m2/glm (M2 histogram; GLM5.md section 5/9) -- which experts
#                           start resident per card
#   FRANKEN_EXPERT_GB       17 -- VRAM a card spends on resident experts (CSV, one per card;
#                           a single value applies to every card)
#   FRANKEN_ADAPT           1 -- adaptive placement learns the hot experts from THIS
#                           conversation's own routing and swaps them in
#   FRANKEN_ADAPT_PREFILL_CAP 64 since 2026-10-07, FRANKEN_ADAPT_HALFLIFE 512 (record §L5-GLM-ADAPT): the placement policy; 0 = the old behaviour of the cap
#   KV_SLOTS                forced to 1 below, REGARDLESS of glm53's family registration
#                           (FamilyLimits max_kv_slots 16, c/family_registry.py) -- that cap is
#                           for the COLIBRI glm53 engine, which has per-slot KDA device state
#                           (P6b) and prefix checkpoints; this engine's Adapter is untested with
#                           more than one slot live at once (glm5_serve.cpp's own comment on
#                           ensure_slot). Prefix checkpoints (the ds4_serve.cpp machinery, franken-
#                           engine b5cf4e0, in service since 2026-10-06, record §L5-GLM-CKPT-E2E)
#                           are per slot too, so a second slot would cost VRAM and ~1.4 GB of host
#                           buffers for a workload (one owner, one conversation at a time) that
#                           does not need it. NB: the engine binary must be the checkpoint build
#                           (franken_dec_glm since 2026-10-06; ~/bench/franken_bin/franken_dec_glm.nockpt
#                           is the live-prefix-only build before it).
set -u

# FRANKEN_BIN may be the engine itself or the docker wrapper
# (~/bench/franken_decode_glm_docker.sh) that runs it inside the ROCm 7.14 image this box's host
# ROCm (6.2) cannot satisfy libllama's tokenizer dependency on (same reason start_franken_ds4.sh's
# FRANKEN_BIN can be that wrapper). The PROCESS name is franken_dec_glm -- serve_alt.sh's
# franken_glm_alive/pgrep -f "franken_dec_[g]lm" is how the rig lock's keeper and `serve_alt.sh
# status` find it; see that binary's own header comment (franken_dec_glm_main.cpp, in the
# frankenstack/franken-engine repo) for why it is named 15 characters and not franken_decode_glm.
BIN=${FRANKEN_GLM_BIN:-$HOME/bench/franken_decode_glm_docker.sh}
GGUF=${FRANKEN_GGUF:-$HOME/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf}
GGUF_DIR=$(dirname "$GGUF")
# A real config.json for this model, for the family resolver only (see this file's own header).
FAMILY_DIR=${FRANKEN_FAMILY_DIR:-$HOME/models/GLM-5.3-Flash-colibri-int4-g64}

[ -x "$BIN" ]  || { echo "[start] FATAL: no engine at $BIN (set FRANKEN_GLM_BIN)"; exit 1; }
[ -s "$GGUF" ] || { echo "[start] FATAL: no model at $GGUF"; exit 1; }
[ -s "$FAMILY_DIR/config.json" ] || {
  echo "[start] FATAL: $FAMILY_DIR/config.json is what --arch glm53 is resolved from"; exit 1; }

cd "$HOME/src/colibri/c" || exit 1

# CLAUDE.md: every recorded number on this box is an 8-thread number.
export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close

export FRANKEN_GGUF="$GGUF"
export FRANKEN_CTX=${FRANKEN_CTX:-262144}
export FRANKEN_GLM_CHUNK=${FRANKEN_GLM_CHUNK:-1024}   # 2026-10-06: was 512; 1024 + the blocked kernels = prefill 9.5 -> 6.0 ms/token at 262 144 cells (records §L5-GLM-CHUNK, §L5-GLM-MOEBLK, §L5-GLM-HGBLK)
export FRANKEN_DEVICES=${FRANKEN_DEVICES:-3}
export FRANKEN_GEMM_LDS=${FRANKEN_GEMM_LDS:-1}   # 2026-09-25: served ON, as Qwen3.8 is -- prefill 16.9 -> 9.3 ms/token (record §L5-GLM-LDS); a last-bits order change, judged by the quality run
export FRANKEN_GEMV_FUSED_REDUCE=${FRANKEN_GEMV_FUSED_REDUCE:-0}   # 2026-10-07: a split GEMV's partials summed by a SEPARATE launch, not in-kernel behind one atomic counter a row: decode 69.9 -> 66.7 ms/token, bit-exact (record §L5-GLM-GEMV-REDUCE); needs franken-engine fused-reduce-split, an older binary ignores it
export FRANKEN_GLM_MOE_G=${FRANKEN_GLM_MOE_G:-8}   # 2026-10-07: routed-expert assignments taken in groups of 8 by a prefill chunk's kernel (was 4): prefill 5.769 -> 5.621 ms/token (-2.6 %, 3 processes G=4,8,4), bit-exact at chunk 1024 / 512 / 32 in place (record §L5-GLM-G8); decode keeps the old kernel
export FRANKEN_GLM_PREFILL_STAGE=${FRANKEN_GLM_PREFILL_STAGE:-0}   # 2026-10-06: was 1. In place (the chunk reads only the experts its rows pick) beats staging at every chunk size: -4 % at 1024 rows, -23..-77 % at 16-256 rows (a short follow-up paid ~2 s of staging), bit-identical (record §L5-GLM-STAGECROSS)
export FRANKEN_GLM_STAGE_MB=${FRANKEN_GLM_STAGE_MB:-256}   # staging ring a card, MB: a 1024-row chunk doubles the per-chunk scratch and a 400 MB ring OOMs dev 2 at 262 144 cells; the ring was measured flat from 400 MB up
export FRANKEN_ADAPT_PREFILL=${FRANKEN_ADAPT_PREFILL:-0}
export FRANKEN_HIP_GRAPH=${FRANKEN_HIP_GRAPH:-0}
export FRANKEN_PLACEMENT=${FRANKEN_PLACEMENT:-$HOME/bench/m2/glm}
export FRANKEN_EXPERT_GB=${FRANKEN_EXPERT_GB:-17}
export FRANKEN_ADAPT=${FRANKEN_ADAPT:-1}
export FRANKEN_ADAPT_PREFILL_CAP=${FRANKEN_ADAPT_PREFILL_CAP:-64}   # 2026-10-07: a held prefill span counts as at most 64 tokens in the placement average (was: the whole span, so after an 8 192-token prompt the average WAS the prompt and decode's routing could not move it): decode 64.8 -> 49.5-56.7 ms after a long prompt, 53.8 -> 51.2-51.5 after a short one, missed MB a token -40..-66 %, bit-exact, placement only (record §L5-GLM-ADAPT; engine flag --adapt-prefill-cap, franken_dec_glm.cap and newer; an older binary ignores it)
export FRANKEN_ADAPT_HALFLIFE=${FRANKEN_ADAPT_HALFLIFE:-512}   # 2026-10-07: tokens for a count to weigh half (was 2048); with the cap, 128 / 512 / 2048 measured alike on a short prompt and 512 best on a long one (trajectory noise past 2 051 tokens: not a ranking)

# Prefix checkpoints (glm5_serve.cpp, franken-engine b5cf4e0): the same knobs and defaults the DeepSeek
# and Qwen launchers set, written out so the served values are visible here and so the docker
# wrapper (which passes them through since 2026-10-06) cannot drop them silently. Boundary tokens
# default to GLM's own turn markers ("<|user|>,<|assistant|>", FRANKEN_BOUNDARY_TOKENS). One
# checkpoint is ~156 MB of recurrent state plus ~1 kB a token of cells (record §L5-GLM-CKPT-E2E).
export FRANKEN_SNAP_EVERY=${FRANKEN_SNAP_EVERY:-1024}   # 2026-10-06: was 512; the prefill loop cuts chunks at snapshot points, 512 would cut every 1024-row chunk back to 512
export FRANKEN_SNAP_KEEP=${FRANKEN_SNAP_KEEP:-8}
export FRANKEN_SNAP_BUDGET_MB=${FRANKEN_SNAP_BUDGET_MB:-6144}
export FRANKEN_SNAP_TURNS=${FRANKEN_SNAP_TURNS:-3}

# --- THE ENV DIFF AGAINST start_franken_ds4.sh ----------------------------------------------
# Same three gateway-side, family-neutral knobs both other launchers set, for the same reasons
# (start_franken.sh's own comment on each): COLI_REQ_LOG, COLI_PREFIX_PIN, COLI_LEDGER (ACTIVE
# here, unlike Qwen/DS4 -- ledger_enabled() is ARCH == "glm53" only, which THIS process's --arch
# actually IS; with the checkpoint build it does real work: the second `[ledger]` line of a
# follow-up reads `state=continuation expect_reuse=146 engine_reuse=146 ok`, and accept_live's
# check 2b judges exactly that -- PASS, 0 MISMATCH on 2026-10-06. Before the checkpoint build
# this comment called the ledger a no-op because the engine reported no checkpoint at all).
# COLI_THINK is NOT forced here, for a third reason again: --arch glm53 means this process hits
# EXACTLY the same COLI_THINK default branch the real glm53 engine hits (unlike DS4's
# deepseek_v4, which needed its own explanation) -- whatever the owner is used to from GLM stays
# true here with no override.
export COLI_REQ_LOG=1
export COLI_PREFIX_PIN=1
export COLI_LEDGER=1

if [ ! -s "$HOME/.colibri_api_key" ]; then
  umask 077; head -c 24 /dev/urandom | base64 | tr -d "=+/" | cut -c1-32 > "$HOME/.colibri_api_key"
fi
export COLI_API_KEY="$(cat "$HOME/.colibri_api_key")"

if [ "${SKIP_WARM:-0}" != "1" ]; then
  echo "[start] warming $GGUF_DIR -- ~110 GB, several minutes"
  cat "$GGUF_DIR"/*.gguf > /dev/null 2>&1 || true
fi
echo "[start] engine   : $BIN"
echo "[start] model    : $GGUF"
echo "[start] family   : glm53 (from $FAMILY_DIR/config.json), advertised as 'glm-5.3-flash'"
echo "[start] ctx=$FRANKEN_CTX chunk=$FRANKEN_GLM_CHUNK devices=$FRANKEN_DEVICES gemm_lds=$FRANKEN_GEMM_LDS glm_prefill_stage=$FRANKEN_GLM_PREFILL_STAGE adapt_prefill=$FRANKEN_ADAPT_PREFILL placement=$FRANKEN_PLACEMENT expert_gb=$FRANKEN_EXPERT_GB adapt=$FRANKEN_ADAPT"
echo "[start] gateway log ${FRANKEN_LOG:-$HOME/bench/serve_alt_franken_glm.log}"
echo "[start] launching gateway on 0.0.0.0:8081"
# NB (inherited from start_glm53.sh/start_franken.sh/start_franken_ds4.sh, and it cost a night
# the first time): do NOT put a comment line BETWEEN the backslash-continued arguments below --
# bash joins the logical line and the # comments out everything after it silently.
python3 -u openai_server.py \
  --model "$FAMILY_DIR" \
  --engine "$BIN" \
  --arch glm53 \
  --host 0.0.0.0 --port 8081 \
  --model-id glm-5.3-flash \
  --max-tokens "${FRANKEN_MAX_TOKENS:-16384}" \
  --kv-slots 1 \
  --allowed-host 127.0.0.1 --allowed-host localhost \
  --allowed-host host.docker.internal \
  --allowed-host rome.local 2>&1 | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }'
