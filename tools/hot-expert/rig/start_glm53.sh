#!/bin/bash
# Serve GLM-5.3 over Colibri's OpenAI-compatible gateway, for OpenWebUI.
#
# Env mirrors tools/rome_bench.sh's glm53 case, including CLAUDE.md's mandatory
# tier caps, with one deliberate difference: COLI_USAGE_PATH points at the
# CANONICAL histogram rather than a frozen copy. Benchmarks freeze it so runs
# stay comparable; a serving process should learn from real traffic, which is
# what makes the expert tier get better the more you use it.
#
# (2026-09-06 morning) COLI_KDA_GPU=2 was ON here for +11.7% decode; superseded the same night by P6 below.
# bit-identical (cosine 0.99992 at 1260 positions, no token selection changed
# anywhere tested); unset it to get the shipped default back.
set -u
M="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"
cd "$HOME/src/colibri/c"
export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
export COLI_VK_SHADERS="$HOME/src/colibri/c/shaders"
export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
export COLI_USAGE_PATH="$HOME/.glm53_explain.bin"
# P6 (2026-09-06): Open WebUI sends a title request after every reply; on ONE KV slot it evicts the
# conversation (turn 2 re-prefilled: 191.6 s). With 4 slots the gateway routes turns to their slot and
# turn 2 takes 3.2 s. Multi-slot needs the CPU recurrence: COLI_KDA_GPU=2 keeps one device state per layer.
# Cost ~10% decode, ~5% prefill; P6b (per-slot device state) is what makes 2 safe again.
# P6b (2026-09-07): the engine allocates ONE KDA state set per KV slot on dev0,
# before the expert preload, so slots no longer share a recurrence and the CPU
# fallback is gone. 4 slots x 156 MB = 625 MB of dev0, paid in the coldest ~44
# of its 1 296 preloaded experts; it buys back the ~10 % decode / ~5 % prefill
# P6 gave up. --kv-slots is 4, not 8, because that is the pool the gate measured
# and because 8 would cost 88 experts for slots Open WebUI does not use.
# If the pool ever comes up short the engine says so and forces 0 by itself.
export COLI_KDA_GPU=2   # P6b: per-slot device state (was 0 from P6, 2026-09-06)
# P1 (2026-09-06): REUSE <id> <reused> <prompt_tokens> per turn on stderr -> the server log tells whether a turn reused its prefix
export GLM53_VERBOSE=1
# Context per KV slot: the engine reads GLM53_MAXT as the slot context (glm53.c:4696,
# default 8192), not a generation cap. 64k = 65536; at the measured 33 KB/token that is
# ~2.2 GB per slot, ~8.8 GB for the 4 slots -- fine on 247 GB.
export GLM53_MAXT=65536
export COLI_REQ_LOG=1   # gateway prints one [req] line per request: ttft, total, prompt/gen tokens
# P7 (2026-09-07): prefix checkpoints in the engine and the gateway-side pin.
# The engine snapshots the shared system+tools prefix at the boundary the
# gateway sends in the 8th SUBMIT field and persists it under
# $M/.coli_ckpt, so a NEW conversation -- and the first request after a
# gateway restart -- starts warm. COLI_PREFIX_PIN keeps Open WebUI's
# <memory_context> block byte-identical for the life of a conversation
# (with more than one memory it re-ranks every turn and turn 2 reused 0/195).
export GLM53_PREFIX_CKPT=1
# A short conversation must not evict the tool-block checkpoint: 4 slots, and any prefix over
# the minimum is captured. At 128 a 170-token API chat stored ~160 MB and knocked out a
# 4 300-token capture (measured in the P9 gate, ~24 min lost). 1024 keeps the slots for
# prefixes worth minutes; a 500-token one re-prefills in ~40 s.
export GLM53_PREFIX_CKPT_MIN=128
export COLI_PREFIX_PIN=1

# The gateway refuses to bind 0.0.0.0 without a key, which is correct -- this
# listens on the LAN. Key lives in ~/.colibri_api_key (mode 600), generated once.
if [ ! -s "$HOME/.colibri_api_key" ]; then
  umask 077; head -c 24 /dev/urandom | base64 | tr -d "=+/" | cut -c1-32 > "$HOME/.colibri_api_key"
fi
export COLI_API_KEY="$(cat "$HOME/.colibri_api_key")"

if [ "${SKIP_WARM:-0}" != "1" ]; then
  echo "[start] warming $M -- ~182 GiB, a few minutes"
  find "$M" -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done
fi
echo "[start] warm complete; launching gateway on 0.0.0.0:8081"
# --max-tokens (2026-09-13): was 256 -- the only knob in this file that carried no
# justification, and it is rome_bench's GENERATION LENGTH, not a serving budget.
# openai_server.py clamps every request DOWN to it (`if maximum > limit: maximum = limit`)
# and Open WebUI sends no max_tokens of its own, so 256 was the hard ceiling on every
# reply the owner ever got. Incident 2026-09-13 22:06: a tool-calling turn spent all 256
# opening a <tool_call> box it could never close; the strict parser needs both tags and
# _unclosed_tail cannot recover a cut inside <arg_value>, so the client got zero
# tool_calls plus raw markers, Open WebUI stored no assistant message, and the chat
# showed NOTHING. A cap, not a target: generation still ends at EOS. 4096 tokens is
# ~135 MB of the 2.2 GB slot; at this box's ~3.4 tok/s a runaway costs ~20 min, which is
# why it is not higher.
# NB: this comment lives ABOVE the command on purpose. Put a # line BETWEEN the
# backslash-continued argument lines below and bash joins the logical line, treats the #
# as a comment, and SILENTLY DROPS EVERY ARGUMENT AFTER IT -- `bash -n` still passes.
# Done exactly that on 2026-09-13 22:54: the gateway came up with no --max-tokens, no
# --kv-slots 4 (KV_SLOTS=1) and no --allowed-host at all.
python3 -u openai_server.py \
  --model "$M" \
  --engine "$HOME/src/colibri/c/glm53" \
  --arch glm53 \
  --host 0.0.0.0 --port 8081 \
  --model-id glm-5.3-flash \
  --max-tokens 4096 \
  --kv-slots 4 \
  --allowed-host 127.0.0.1 --allowed-host localhost \
  --allowed-host host.docker.internal \
  --allowed-host rome.local 2>&1 | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }'
