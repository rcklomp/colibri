#!/bin/bash
# glm_serve_ab_chain.sh -- is the checkpoint build of the Franken GLM serve engine slower at decode
# than the build before it? (record §L5-GLM-CKPT-E2E: served decode was 11.4-11.8 tok/s against
# 12.7-14.8 in §L5-GLM-SERVE, unexplained, other sessions were compiling on the host.)
#   A = ~/bench/franken_bin/franken_dec_glm.nockpt  (franken-engine b9bd25d, live-prefix reuse only)
#   B = ~/bench/franken_bin/franken_dec_glm.ckpt    (franken-engine b5cf4e0, prefix checkpoints)
# Each arm: a FRESH gateway + engine (same start script, same env as serve_alt.sh franken-glm, only
# FRANKEN_DOCKER_BIN differs), one warm-up request, then two timed requests; greedy (temperature 0) and
# the same three prompts in the same order in every arm, so the routing and the adaptation trajectory
# are the same. The number is the engine's own `tok/s=` for the request (decode only), from the
# gateway log. Order A,B,B,A. Launch ONLY through run_chain.sh (rig lock):
#   ssh -n -f rome 'setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glmserve_ab/glm_serve_ab_chain.sh \
#       > ~/bench/glmserve_ab/chain.log 2>&1 < /dev/null &'
# Refuses to start if anything is compiling on the host (a build competes for the host issue thread).
set -u
OUT=$HOME/bench/glmserve_ab; mkdir -p "$OUT"
HE=$HOME/src/colibri/tools/hot-expert
. "$HE/gate_lib.sh"
KEY=$(cat "$HOME/.colibri_api_key")
BIN_A=$HOME/bench/franken_bin/franken_dec_glm.nockpt
BIN_B=$HOME/bench/franken_bin/franken_dec_glm.ckpt
START=$HE/franken/start_franken_glm.sh
P1="Explain in detail how a hash table handles collisions, then compare open addressing with chaining."
P2="Write a detailed explanation of how TCP congestion control works, covering slow start, congestion avoidance and fast recovery."
P3="Describe step by step how a compiler turns source code into machine code, covering lexing, parsing, optimization and code generation."
MAXTOK=${MAXTOK:-256}

vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; echo "vram still $(vram_max) MiB"; return 1; }
stop_all() {
  pkill -f "openai_[s]erver.py" 2>/dev/null; sleep 2; pkill -9 -f "openai_[s]erver.py" 2>/dev/null
  pkill -9 -f "franken_dec_[g]lm" 2>/dev/null
  docker stop -t 5 franken_engine >/dev/null 2>&1 || true
}
trap 'stop_all; echo "=== chain exit trap $(date -Is): gateway and engine stopped, port 8081 left empty (rig reserved)"' EXIT

if pgrep -f "[m]ake .*glm-serve|[h]ipcc|[c]c1plus|[c]lang.*offload" > /dev/null; then
  echo "REFUSED: a compile is running on the host -- it would compete with the decode loop; retry later"; exit 3
fi
[ -x "$BIN_A" ] && [ -x "$BIN_B" ] || { echo "FATAL: binaries missing"; exit 2; }
echo "=== glm_serve_ab start $(date -Is) A=$(sha256sum $BIN_A | cut -c1-16) B=$(sha256sum $BIN_B | cut -c1-16)"
stop_all; vram_wait || exit 2

declare -A TPS
arm() {   # arm <A|B> <bin>
  local name=$1 bin=$2 log="$OUT/gw_${1}_$(date +%H%M%S).log"
  vram_wait || return 1
  echo "=== arm $name ($(basename "$bin")) $(date +%T)"
  FRANKEN_DOCKER_BIN=$bin FRANKEN_LOG=$log SKIP_WARM=1 setsid nohup "$START" > "$log" 2>&1 < /dev/null &
  local up=0
  for i in $(seq 1 120); do
    sleep 5
    if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" -w '%{http_code}' http://127.0.0.1:8081/v1/models)" = 200 ] \
       && pgrep -f "franken_dec_[g]lm" > /dev/null; then up=1; break; fi
  done
  [ $up = 1 ] || { echo "FAIL: arm $name did not come up"; tail -5 "$log"; stop_all; return 1; }
  echo "up after $((i*5)) s"
  local n=0
  for P in "$P1" "$P2" "$P3"; do
    n=$((n+1))
    curl -s -m 900 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
      http://127.0.0.1:8081/v1/chat/completions \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"glm-5.3-flash","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$P" "$MAXTOK")" > /dev/null
    sleep 1
    local line; line=$(grep -a "serve-glm5\] req=" "$log" | tail -1)
    local t; t=$(echo "$line" | sed -n 's/.*tok\/s=\([0-9.]*\).*/\1/p'); local e; e=$(echo "$line" | sed -n 's/.*emitted=\([0-9]*\).*/\1/p')
    echo "  req $n: emitted=$e tok/s=$t ($(echo "$line" | sed -n 's/.*\(prefill_s=[0-9.]*\).*/\1/p'))"
    [ $n -ge 2 ] && TPS[$name]="${TPS[$name]:-}$t "
  done
  stop_all
  for _ in $(seq 1 30); do pgrep -f "franken_dec_[g]lm" > /dev/null || break; sleep 1; done
}
arm A "$BIN_A"; arm B "$BIN_B"; arm B "$BIN_B"; arm A "$BIN_A"
echo "=== verdict (decode tok/s, requests 2 and 3 of each arm; A = no-checkpoint build, B = checkpoint build)"
gate_ab_verdict "decode tok/s, 256 greedy tokens" "${TPS[A]:-}" "${TPS[B]:-}"
echo "=== glm_serve_ab end $(date -Is)"
