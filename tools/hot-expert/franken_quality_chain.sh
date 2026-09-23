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
MMLU_SAMPLE="$HOME/bench/f11_quality/mmlu_pro_sample.json"   # the 2026-09-21 sample, seed-fixed
export QE_REASONING_EFFORT=${QE_REASONING_EFFORT:-xhigh}
BUDGET=${QE_BUDGET:-16000}
export FRANKEN_BIN=${FRANKEN_BIN:-$HOME/bench/franken_decode_docker.sh}
ARMS=${ARMS:-franken,qwen38}

echo "=== franken_quality_chain $(date -Is) arms=$ARMS reasoning=$QE_REASONING_EFFORT budget=$BUDGET"
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
      --out "$2" --mmlu "$MMLU_SAMPLE" --max-context 262144 --parts mmlu,needle \
      --mmlu-max-tokens "$BUDGET"
  echo "--- eval --expect $1 exit=$? $(date -Is)"
}

for arm in ${ARMS//,/ }; do
  case $arm in
    franken)
      "$SERVE_ALT" franken 2>&1 | grep "now serving\|FATAL\|accept_live" || { echo "FATAL: franken did not come up"; exit 1; }
      "$SERVE_ALT" status 2>&1 | grep -q "Franken engine" || { echo "FATAL: not the Franken engine"; exit 1; }
      run_eval franken "$OUT_DIR/franken_lds1.jsonl" ;;
    qwen38)
      "$SERVE_ALT" qwen38 2>&1 | grep "now serving\|FATAL" || { echo "FATAL: qwen38 did not come up"; exit 1; }
      run_eval qwen38 "$OUT_DIR/qwen38_llama.jsonl" ;;
  esac
done
echo "--- summarize"
python3 "$QE" summarize --model "franken_lds1=$OUT_DIR/franken_lds1.jsonl" --model "qwen38_llama=$OUT_DIR/qwen38_llama.jsonl" 2>&1 | tail -60
echo "=== franken_quality_chain done $(date -Is)"
