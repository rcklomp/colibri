#!/bin/bash
# franken_quality_chain.sh -- the L0 quality number (design §9.5 step 4): the Franken engine
# as served (FRANKEN_GEMM_LDS=1, three cards, 256k) against llama.cpp's Qwen3.8 on the same
# 70-question MMLU-Pro sample and the needle set, both with the same explicit reasoning
# setting (QE_REASONING_EFFORT, default xhigh = llama.cpp's template default for Qwen3.8), the
# same 16k answer budget. Ends with GLM back in service whatever happened. Launch detached:
#   setsid nohup ~/src/colibri/tools/hot-expert/franken_quality_chain.sh > ~/bench/franken_quality/chain.log 2>&1 < /dev/null &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
QE="$HERE/quality_eval.py"; SERVE_ALT="$HERE/serve_alt.sh"
KEY_FILE="$HOME/.colibri_api_key"; URL="http://127.0.0.1:8081"; MODEL_ID="glm-5.3-flash"
OUT_DIR="$HOME/bench/franken_quality"; mkdir -p "$OUT_DIR"
MMLU_SAMPLE="${QE_MMLU_SAMPLE:-$HOME/bench/f11_quality/mmlu_pro_sample.json}"   # the 2026-09-21 sample, seed-fixed
export QE_REASONING_EFFORT=${QE_REASONING_EFFORT:-xhigh}
BUDGET=${QE_BUDGET:-16000}
# engine paths are per arm: FRANKEN_BIN is the Qwen3.8 engine only (2026-09-25: exported globally it
# was picked up by start_franken_glm.sh and started the Qwen engine on the GLM model)
ARMS=${ARMS:-franken,qwen38}

echo "=== franken_quality_chain $(date -Is) arms=$ARMS reasoning=$QE_REASONING_EFFORT budget=$BUDGET temperature=${QE_TEMPERATURE:-0 (greedy)} top_p=${QE_TOP_P:-server default} tag=${QE_TAG:-none}"
on_exit() {
  local rc=$?; trap - EXIT INT TERM HUP
  echo "=== chain exit rc=$rc $(date -Is); restoring GLM"
  "$SERVE_ALT" glm 2>&1 | tail -3
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

run_eval() {   # run_eval <expect> <out_jsonl>
  echo "--- eval --expect $1 $(date -Is)"
  python3 "$QE" run --expect "$1" --url "$URL" --key-file "$KEY_FILE" --model-id "$MODEL_ID" \
      --out "$2" --mmlu "$MMLU_SAMPLE" --max-context 262144 --parts "${QE_PARTS:-mmlu,needle}" \
      --mmlu-max-tokens "$BUDGET"
  echo "--- eval --expect $1 exit=$? $(date -Is)"
}

SUMMARY_ARGS=()
for arm in ${ARMS//,/ }; do
  case $arm in
    franken)
      FRANKEN_BIN=${FRANKEN_BIN:-$HOME/bench/franken_decode_docker.sh} "$SERVE_ALT" franken 2>&1 | grep "now serving\|FATAL\|accept_live" || { echo "FATAL: franken did not come up"; exit 1; }
      "$SERVE_ALT" status 2>&1 | grep -q "Franken engine" || { echo "FATAL: not the Franken engine"; exit 1; }
      run_eval franken "$OUT_DIR/franken_lds1.jsonl"
      SUMMARY_ARGS+=(--model "franken_lds1=$OUT_DIR/franken_lds1.jsonl") ;;
    qwen38)
      "$SERVE_ALT" qwen38 2>&1 | grep "now serving\|FATAL" || { echo "FATAL: qwen38 did not come up"; exit 1; }
      run_eval qwen38 "$OUT_DIR/qwen38_llama.jsonl"
      SUMMARY_ARGS+=(--model "qwen38_llama=$OUT_DIR/qwen38_llama.jsonl") ;;
    franken-ds4)
      # M1: the Franken engine on DeepSeek-V4-Flash (franken_decode_ds4), same
      # shape as the `franken` arm above -- serve_alt.sh's D4_LABEL also
      # carries "(Franken engine)", so the same status grep tells the two apart
      # from a llama.cpp arm without needing a second pattern.
      "$SERVE_ALT" franken-ds4 2>&1 | grep "now serving\|FATAL\|accept_live" || { echo "FATAL: franken-ds4 did not come up"; exit 1; }
      "$SERVE_ALT" status 2>&1 | grep -q "Franken engine" || { echo "FATAL: not the Franken engine"; exit 1; }
      run_eval franken-ds4 "$OUT_DIR/franken_ds4.jsonl"
      SUMMARY_ARGS+=(--model "franken_ds4=$OUT_DIR/franken_ds4.jsonl") ;;
    franken-glm)
      # the Franken engine on GLM-5.3-Flash (franken_dec_glm, FRANKEN_GEMM_LDS=1 as served),
      # against llama.cpp's own GLM-5.3-Flash UD-IQ4_XS below (serve_alt.sh glm-llama)
      "$SERVE_ALT" franken-glm 2>&1 | grep "now serving\|FATAL\|accept_live" || { echo "FATAL: franken-glm did not come up"; exit 1; }
      "$SERVE_ALT" status 2>&1 | grep -q "Franken engine" || { echo "FATAL: not the Franken engine"; exit 1; }
      run_eval franken-glm "$OUT_DIR/franken_glm${QE_TAG:+_$QE_TAG}.jsonl"
      SUMMARY_ARGS+=(--model "franken_glm=$OUT_DIR/franken_glm${QE_TAG:+_$QE_TAG}.jsonl") ;;
    glm-llama)
      "$SERVE_ALT" glm-llama 2>&1 | grep "now serving\|FATAL" || { echo "FATAL: glm-llama did not come up"; exit 1; }
      run_eval glm-llama "$OUT_DIR/glm_llama${QE_TAG:+_$QE_TAG}.jsonl"
      SUMMARY_ARGS+=(--model "glm_llama=$OUT_DIR/glm_llama${QE_TAG:+_$QE_TAG}.jsonl") ;;
    deepseek)
      # llama.cpp's own DeepSeek-V4-Flash-0731 UD-IQ2_M arm (serve_alt.sh's
      # pre-existing `deepseek` target, cmd_alt D) -- the comparison arm
      # franken-ds4 is measured against, same shape as qwen38/franken above.
      "$SERVE_ALT" deepseek 2>&1 | grep "now serving\|FATAL" || { echo "FATAL: deepseek did not come up"; exit 1; }
      run_eval deepseek "$OUT_DIR/deepseek_llama.jsonl"
      SUMMARY_ARGS+=(--model "deepseek_llama=$OUT_DIR/deepseek_llama.jsonl") ;;
  esac
done
echo "--- summarize"
python3 "$QE" summarize "${SUMMARY_ARGS[@]}" 2>&1 | tail -60
echo "=== franken_quality_chain done $(date -Is)"
