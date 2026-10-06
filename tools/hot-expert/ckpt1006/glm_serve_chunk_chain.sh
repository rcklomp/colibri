#!/bin/bash
# glm_serve_chunk_chain.sh -- the combined build IN THE SERVING PATH: first-token cost of cold ~8k-token prompts through the real gateway, A,B,B,A.
#   A = franken_dec_glm.integ  (franken-engine 8256d53, the merged main: chunk 512, snapshots every 512, stock expert kernel)
#   B = franken_dec_glm.comb   (moe-regblock + chunk cap 1024 + serve cap 1024; FRANKEN_GLM_CHUNK=1024 FRANKEN_SNAP_EVERY=1024 FRANKEN_GLM_STAGE_MB=256)
# Each arm: a FRESH gateway + engine (start_franken_glm.sh, only FRANKEN_DOCKER_BIN and B's three knobs differ), one short warm-up request, then three cold ~8k-token prompts (slices of three
# different documents, greedy, 4 tokens). The number is the engine's own prefill_s / prompt from the request log line (chunks= shows the cuts), VRAM used on each card is read after the
# last request (the margin of record §L5-GLM-CHUNK was 0.28 GB on dev 2). Launch ONLY through run_chain.sh (rig lock); it leaves nothing serving (the rig is reserved).
set -u
OUT=$HOME/bench/glmserve_chunk; rm -rf "$OUT"; mkdir -p "$OUT"
HE=$HOME/src/colibri/tools/hot-expert
. "$HE/gate_lib.sh"
KEY=$(cat "$HOME/.colibri_api_key")
BIN_A=$HOME/bench/franken_bin/franken_dec_glm.integ
BIN_B=$HOME/bench/franken_bin/franken_dec_glm.comb
START=$HE/franken/start_franken_glm.sh
DOC=$HOME/src/colibri/tools/hot-expert/ROADMAP-2026-09.md
python3 - "$DOC" "$OUT" <<'PY'
import sys
t = open(sys.argv[1], errors="ignore").read()
for i, off in enumerate((0, 70000, 140000), 1):
    open("%s/prompt%d.txt" % (sys.argv[2], i), "w").write("Document %d follows.\n\n" % i + t[off:off + 27000] + "\n\nReply with the single word OK.")
PY
vram_each() { local o=""; for d in /sys/class/drm/card[0-9]/device; do o="$o $(( $(cat $d/mem_info_vram_used)/1048576 ))"; done; echo "$o MiB (cards 0,1,2 by drm order)"; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; echo "vram still $(vram_max) MiB"; return 1; }
stop_all() {
  pkill -f "openai_[s]erver.py" 2>/dev/null; sleep 2; pkill -9 -f "openai_[s]erver.py" 2>/dev/null
  pkill -9 -f "franken_dec_[g]lm" 2>/dev/null
  docker stop -t 5 franken_engine >/dev/null 2>&1 || true
  # the container of a 144 GB-pinned engine lingers for minutes while the kernel unpins: a new `docker run --name franken_engine` then fails with "name already in use"
  # (the second B arm of the first run died that way): wait until the container is gone, not just the process
  for _ in $(seq 1 120); do [ -z "$(docker ps -aq -f name='^franken_engine$' 2>/dev/null)" ] && break; sleep 5; done
}
trap 'stop_all; echo "=== chain exit trap $(date -Is): gateway and engine stopped, port 8081 left empty (rig reserved)"' EXIT
if pgrep -f "[m]ake .*glm-serve|[h]ipcc|[c]c1plus|[c]lang.*offload" > /dev/null; then
  echo "REFUSED: a compile is running on the host -- it would compete with the engine; retry later"; exit 3; fi
[ -x "$BIN_A" ] && [ -x "$BIN_B" ] || { echo "FATAL: binaries missing ($BIN_A / $BIN_B)"; exit 2; }
echo "=== glm_serve_chunk start $(date -Is) A=$(sha256sum $BIN_A | cut -c1-16) B=$(sha256sum $BIN_B | cut -c1-16)"
stop_all; vram_wait || exit 2
declare -A MSTOK
arm() {   # arm <A|B> <bin> [env assignments...]
  local name=$1 bin=$2; shift 2
  local log="$OUT/gw_${name}_$(date +%H%M%S).log"
  vram_wait || return 1
  echo "=== arm $name ($(basename "$bin")) env: ${*:-defaults} $(date +%T)"
  env "$@" FRANKEN_DOCKER_BIN=$bin FRANKEN_LOG=$log SKIP_WARM=1 setsid nohup "$START" > "$log" 2>&1 < /dev/null &
  local up=0 i
  for i in $(seq 1 180); do
    sleep 5
    if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" -w '%{http_code}' http://127.0.0.1:8081/v1/models)" = 200 ] && pgrep -f "franken_dec_[g]lm" > /dev/null; then up=1; break; fi
  done
  [ $up = 1 ] || { echo "FAIL: arm $name did not come up"; tail -6 "$log" | cut -c1-200; stop_all; return 1; }
  echo "up after $((i*5)) s; VRAM after load: $(vram_each)"
  grep -aE "glm5_stage dev|chunk=|serve-glm5\] .*(chunk|snap)" "$log" | head -4 | cut -c1-200
  local n=0 P
  for P in "warm: say hi" "$(cat $OUT/prompt1.txt)" "$(cat $OUT/prompt2.txt)" "$(cat $OUT/prompt3.txt)"; do
    n=$((n+1))
    curl -s -m 1500 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' http://127.0.0.1:8081/v1/chat/completions \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"glm-5.3-flash","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":4,"temperature":0}))' "$P")" > "$OUT/resp_${name}_$n.json"
    sleep 1
    local line; line=$(grep -a "serve-glm5\] req=" "$log" | tail -1)
    local pr; pr=$(echo "$line" | sed -n 's/.* prompt=\([0-9]*\).*/\1/p'); local ps; ps=$(echo "$line" | sed -n 's/.*prefill_s=\([0-9.]*\).*/\1/p')
    local ch; ch=$(echo "$line" | sed -n 's/.* chunks=\([0-9]*\).*/\1/p'); local re; re=$(echo "$line" | sed -n 's/.* reused=\([0-9]*\).*/\1/p')
    local mt="n/a"; [ -n "$pr" ] && [ "${pr:-0}" -gt 0 ] && mt=$(python3 -c "print('%.3f' % (1000.0 * $ps / $pr))")
    echo "  req $n: prompt=$pr reused=$re chunks=$ch prefill_s=$ps -> $mt ms/token"
    [ $n -ge 2 ] && [ "${re:-1}" = 0 ] && MSTOK[$name]="${MSTOK[$name]:-}$mt "
  done
  echo "  VRAM after the long prompts: $(vram_each)"
  grep -aE "HIP error|Memory access|out of memory" "$log" | head -2 | cut -c1-160
  stop_all
  for _ in $(seq 1 30); do pgrep -f "franken_dec_[g]lm" > /dev/null || break; sleep 1; done
}
arm A "$BIN_A"
arm B "$BIN_B" FRANKEN_GLM_CHUNK=1024 FRANKEN_SNAP_EVERY=1024 FRANKEN_GLM_STAGE_MB=256
arm B "$BIN_B" FRANKEN_GLM_CHUNK=1024 FRANKEN_SNAP_EVERY=1024 FRANKEN_GLM_STAGE_MB=256
arm A "$BIN_A"
echo "=== verdict (cold ~8k-token prompts: prefill ms/token; A = merged main, B = combined)"
echo "A: ${MSTOK[A]:-none}"; echo "B: ${MSTOK[B]:-none}"
gate_ab_verdict "prefill ms/token, cold ~8k prompts" "${MSTOK[A]:-}" "${MSTOK[B]:-}"
echo "=== glm_serve_chunk end $(date -Is)"
