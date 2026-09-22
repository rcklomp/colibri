#!/bin/bash
# F11 quality, second run (2026-09-22): the comparison the first one was not.
#   - ONE engine for all three models: llama.cpp HIP via serve_alt.sh (GLM-5.3-Flash UD-IQ4_XS
#     added as `glm-llama`), each model at its chat template's default reasoning -- what Open
#     WebUI would give the owner on that engine.
#   - 210 MMLU-Pro questions (15 per category, same seed/offsets scheme, frozen file
#     mmlu_pro_210.json), zero-shot CoT, temperature 0.
#   - answer budget 16000 tokens (the first run's 3500 cut off 25 of 140 answers); a cut-off
#     answer is still reported separately, not hidden.
#   - no needle test (24/24 in the first run: it discriminates nothing here).
# Ends with the Colibri GLM gateway back in service (serve_alt.sh glm runs accept_live).
# Wall time: unknown until measured; the first run's per-question medians (Qwen 20 s, DeepSeek
# 30 s, but 30 MINUTES on the questions DeepSeek reasons longest about) say "a day or two".
# Launch detached:
#   ssh -n -f rome 'setsid nohup ~/src/colibri/tools/hot-expert/f11_quality2_chain.sh > ~/bench/f11_quality/chain2.log 2>&1 < /dev/null &'
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
QE="$HERE/quality_eval.py"; SA="$HERE/serve_alt.sh"
OUT=~/bench/f11_quality; mkdir -p "$OUT"; cd "$OUT" || exit 1
SAMPLE="$OUT/mmlu_pro_210.json"
export MMLU_PER_CATEGORY=15

on_exit() {
  trap - EXIT INT TERM HUP
  echo "--- on_exit $(date -Is): making sure the GLM gateway serves"
  "$SA" status 2>&1 | grep -qi "serving: GLM-5.3 (model id" || "$SA" glm
  "$SA" status
}
trap on_exit EXIT INT TERM HUP

echo "=== f11_quality2_chain $(date -Is)"
[ -f "$SAMPLE" ] || python3 "$QE" fetch --out "$SAMPLE" || { echo "FATAL: fetch failed"; exit 1; }
for m in qwen38 deepseek glm-llama; do
  echo "=== $m $(date -Is)"
  "$SA" "$m" || { echo "=== $m: serve_alt failed (it has fallen back to GLM by itself); skipping"; continue; }
  python3 "$QE" run --expect "$m" --url http://127.0.0.1:8081 --key-file ~/.colibri_api_key \
      --model-id glm-5.3-flash --out "$OUT/q2_$m.jsonl" --mmlu "$SAMPLE" \
      --max-context 250000 --parts mmlu --mmlu-max-tokens 16000
  echo "=== $m eval exit=$? $(date -Is)"
done
echo "=== back to GLM $(date -Is)"; "$SA" glm
args=(); for m in qwen38 deepseek glm-llama; do [ -f "$OUT/q2_$m.jsonl" ] && args+=(--model "$m=$OUT/q2_$m.jsonl"); done
[ ${#args[@]} -gt 0 ] && python3 "$QE" summarize "${args[@]}"
echo "=== f11_quality2_chain done $(date -Is)"
