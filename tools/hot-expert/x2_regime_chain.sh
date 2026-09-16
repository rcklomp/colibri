#!/bin/bash
# x2_regime_chain.sh -- X2 follow-up 2: which regime does COLI_KDA_GPU=2's
# degenerate output actually affect?
#
# x2_iso_chain.sh showed the PRISTINE serving binary's CLI teacher_forcing
# line is degenerate at COLI_KDA_GPU=2 on the 564-token packet, and that this
# branch introduced no regression. But the coordinator points out the live
# gateway (same binary, same knob, `~/start_glm53.sh`'s production config)
# has answered coherently all night (accept_live.sh PASS repeatedly:
# "Wednesday", a prime > 10, a 286-token finish=stop reply). Two candidate
# explanations, from `c/glm53.c`'s "G12 sync points" comment (~line 6124,
# segment-adapter code -- the CPU cannot see the GPU's kda_state/kda_window
# except at two explicit sync points, because reading them every token would
# cost ~5 ms/token, a third of what the chain saves):
#   (i)  the CLI's --prompt/teacher_forcing prefill path is itself broken at
#        =2 (a real generation bug the server happens not to hit), or
#   (ii) the teacher_forcing readout races the GPU (reads the logits buffer
#        before the chained command buffer's fence signals), and DECODE --
#        which must wait for each step's argmax before it can pick the next
#        token, so it is naturally fenced -- is fine regardless of regime.
#
# This does not choose between (i)/(ii) by reading the shader (that is
# bisection, explicitly out of scope here). It separates CLI generation from
# CLI teacher_forcing/readout from the served path, on the PRISTINE binary
# only, and reports which one disagrees with which.
#
#   A. CLI, COLI_KDA_GPU=2, greedy 64, short question -- text + teacher_forcing
#   B. CLI, COLI_KDA_GPU=2, greedy 64, the 450-row packet -- text
#   C. CLI, COLI_KDA_GPU=0, greedy 64, short question -- text + teacher_forcing
#   D. CLI, COLI_KDA_GPU=0, greedy 64, the 450-row packet -- text
#   [gateway restarted here -- production config, COLI_KDA_GPU=2]
#   E. live gateway, same short question, max_tokens=64 temperature=0 -- text
#   F. live gateway, the 450-row packet as the user turn, same params -- text
#
# Same rig rules as every other X2 chain: run only through run_chain.sh,
# stop gateway / wait_no_engine, trap restarts the gateway on every exit
# path, accept_live.sh at the end. Budget <= 20 min -- six short
# (<=64-token) generations, no dumps.
set -u
TAG=x2reg$(date +%m%d%H%M)
OUT=~/bench/x2_regime_out; mkdir -p "$OUT"
SRC_X2=~/src/colibri-x2
HERE="$SRC_X2/tools/hot-expert"
M="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"
PRISTINE=~/src/colibri/c/glm53
PRISTINE_SHA=a8e10ecf09edc05c3049c667ffcba05beb73b30415409a40214a4ba0e1f565db
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
SHORT_Q="Which day comes after Tuesday? Answer in one sentence."

RESTART_GATEWAY=0

start_gateway() {
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 90); do
    sleep 10
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
             http://127.0.0.1:8081/v1/models 2>/dev/null) || true
    [ "${code:-}" = 200 ] && break
  done
  echo "[x2reg] gateway back, /v1/models=${code:-?}"
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[x2reg] stopping the owner's gateway for the duration of the CLI runs"
    pkill -f "openai_[s]erver.py" || true
    RESTART_GATEWAY=1
  fi
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 120); do pgrep -x glm53 >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[x2reg] FATAL: a glm53 is still alive after 240s"; return 1
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "=== x2_regime_chain exit rc=$rc tag=$TAG $(date -Is)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== x2_regime_chain $TAG $(date -Is)"
for _eng in qwen38 qwen38-vk; do
  if pgrep -x "$_eng" >/dev/null 2>&1; then echo "[x2reg] $_eng is running -- refusing"; exit 1; fi
done

stop_gateway || exit 1

[ "$(sha256sum "$PRISTINE" | cut -d' ' -f1)" = "$PRISTINE_SHA" ] || {
  echo "[x2reg] REFUSED: $PRISTINE sha does not match the expected pristine ($PRISTINE_SHA)"; exit 2; }
echo "[x2reg] pristine=$(sha256sum "$PRISTINE" | cut -c1-16)"

echo "--- warming the model (cheap if already warm)"
cat "$M"/*.safetensors > /dev/null 2>&1 || true

PACKET=$(cat "$HERE/x2_packet_450.txt")

run_cli() {                  # run_cli <tag> <prompt> [ENV=V ...]
  local tag="$1" prompt="$2"; shift 2
  echo "[x2reg] $(date +%H:%M:%S) run $tag  ($*)"
  cp -f "$HOME/.glm53_explain.bin" "/tmp/x2reg_hist_$tag.bin"
  rm -rf "/tmp/x2reg_ckpt_$tag"; mkdir -p "/tmp/x2reg_ckpt_$tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$SRC_X2/c/shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_USAGE_PATH="/tmp/x2reg_hist_$tag.bin"
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="/tmp/x2reg_ckpt_$tag"
    export GLM53_VERBOSE=1
    for kv in "$@"; do export "${kv?}"; done
    "$PRISTINE" --model "$M" --prompt "$prompt" --greedy 64 ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  echo "[x2reg] $(date +%H:%M:%S) run $tag rc=$rc"
  return $rc
}

# extract_text <out-file>: everything between the teacher_forcing line and the
# "decode N token" stats line -- the streamed generated text, nothing else.
extract_text() {
  awk '/^teacher_forcing/{tf=1; next} /^decode [0-9]+ token/{exit} tf{printf "%s", $0 "\n"}' "$1"
}

run_cli qA "$SHORT_Q" COLI_KDA_GPU=2
run_cli qB "$PACKET"  COLI_KDA_GPU=2
run_cli qC "$SHORT_Q" COLI_KDA_GPU=0
run_cli qD "$PACKET"  COLI_KDA_GPU=0

echo "[x2reg] restoring the gateway (production config, COLI_KDA_GPU=2) before the served comparison"
start_gateway
RESTART_GATEWAY=0

query_gateway() {            # query_gateway <tag> <content>
  local tag="$1" content="$2"
  python3 - "$KEY" "$content" > "$OUT/$tag.json" <<'PY'
import sys, json, urllib.request
key, content = sys.argv[1], sys.argv[2]
req = urllib.request.Request(
    "http://127.0.0.1:8081/v1/chat/completions",
    data=json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": content}],
                      "max_tokens": 64, "temperature": 0}).encode(),
    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    print(json.dumps(r))
except Exception as e:
    print(json.dumps({"error": str(e)}))
PY
  python3 -c "
import json
d = json.load(open('$OUT/$tag.json'))
try:
    print(d['choices'][0]['message']['content'])
except Exception:
    print('ERROR:', d)
" > "$OUT/$tag.txt"
  echo "[x2reg] gateway $tag saved -> $OUT/$tag.txt"
}

query_gateway qE "$SHORT_Q"
query_gateway qF "$PACKET"

echo
echo "=== X2 regime check: text and teacher_forcing across CLI =2 / =0 / served ==="
echo "-- qA (CLI =2, short) teacher_forcing:"
grep ^teacher_forcing "$OUT/qA.out" | cut -c1-200
echo "-- qA (CLI =2, short) generated text:"; extract_text "$OUT/qA.out"
echo "-- qC (CLI =0, short) teacher_forcing:"
grep ^teacher_forcing "$OUT/qC.out" | cut -c1-200
echo "-- qC (CLI =0, short) generated text:"; extract_text "$OUT/qC.out"
echo "-- qE (gateway, short) generated text:"; cat "$OUT/qE.txt"
echo
echo "-- qB (CLI =2, packet) generated text:"; extract_text "$OUT/qB.out"
echo "-- qD (CLI =0, packet) generated text:"; extract_text "$OUT/qD.out"
echo "-- qF (gateway, packet) generated text:"; cat "$OUT/qF.txt"

echo
echo "[x2reg] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ACCEPT_RC=$?
echo "[x2reg] accept_live.sh rc=$ACCEPT_RC"

exit $ACCEPT_RC
