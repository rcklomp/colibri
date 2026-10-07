#!/bin/bash
# glm_serve_curve_chain.sh -- the placement policy in a PERSISTENT serving session (2026-10-07; record section L5-GLM-ADAPT-MONO follow-up).
# Why: the policy sweeps restarted every configuration from the generic placement; the real service keeps its average across chats. Each arm = a fresh serving process (the monotonic-position
#  build franken_dec_glm, the installed one) with one policy setting, driven by serve_curve_driver.py: 14 different short fresh greedy chats, 300 tokens each. The curve of decode tok/s over the
#  chats says how fast a session converges and which policy it converges to; greedy text below 2 051 tokens of depth is identical in every arm (compare the `chars` columns).
# Arms (env on top of the start script's shipped defaults G=8, cap 64, half-life 512): shipped (nothing), hl128, hl2048 (cap 64), nocap (cap 0, half-life 2048: the old policy, position fix kept).
# Launch through run_chain.sh (the reservation flag stays in place), the chain stops the engine after each arm and at its end:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_serve_curve_chain.sh > ~/bench/glm_serve_curve_chain.log 2>&1 < /dev/null &
# Env: SC_BIN (default the installed ~/bench/franken_bin/franken_dec_glm), SC_N (chats per arm, default 14), SC_ARMS ("shipped hl128 hl2048 nocap"), SC_OUT (default ~/bench/franken/glm5/serve_curve).
set -u
. "$HOME/bench/chain_preflight.sh"
HE=$HOME/src/colibri/tools/hot-expert; START=${SC_START:-$HE/franken/start_franken_glm.sh}; PG=${SC_PGREP:-franken_dec_[g]lm}
BIN=${SC_BIN:-$HOME/bench/franken_bin/franken_dec_glm}; N=${SC_N:-14}
O=${SC_OUT:-$HOME/bench/franken/glm5/serve_curve}; rm -rf "$O" 2>/dev/null; mkdir -p "$O"
KEY=$(cat "$HOME/.colibri_api_key")
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_serve_curve exit rc=2 $(date -Is)"; exit 2; }
[ -e "$HOME/bench/.dev_reserved" ] || { echo "FATAL: the reservation flag is gone: this chain starts an engine and must run under it"; echo "=== glm_serve_curve exit rc=2 $(date -Is)"; exit 2; }
echo "=== glm_serve_curve start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) chats=$N"
trap 'rig_stop_serving; echo "=== glm_serve_curve chain end $(date -Is): everything stopped"' EXIT
rcs=0
for arm in ${SC_ARMS:-shipped hl128 hl2048 nocap}; do
  case $arm in
    shipped) E="";;
    hl128)   E="FRANKEN_ADAPT_HALFLIFE=128";;
    hl2048)  E="FRANKEN_ADAPT_HALFLIFE=2048";;
    nocap)   E="FRANKEN_ADAPT_PREFILL_CAP=0 FRANKEN_ADAPT_HALFLIFE=2048";;
    *) echo "unknown arm $arm"; rcs=1; continue;;
  esac
  echo "=== arm $arm env='$E' $(date -Is)"
  rig_stop_serving; rig_quiet_wait 900 || { echo "arm $arm: rig not quiet"; rcs=1; continue; }
  log="$O/$arm.gw.log"
  env FRANKEN_DOCKER_BIN=$BIN FRANKEN_LOG=$log SKIP_WARM=1 $E setsid nohup "$START" > "$log" 2>&1 < /dev/null &
  up=0
  for i in $(seq 1 240); do
    sleep 5
    if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" -w '%{http_code}' http://127.0.0.1:8081/v1/models)" = 200 ] && pgrep -f "$PG" > /dev/null; then up=1; break; fi
  done
  [ $up = 1 ] || { echo "arm $arm: the engine did not come up"; tail -6 "$log" | cut -c1-200; rcs=1; rig_stop_serving; continue; }
  echo "arm $arm up after $((i*5)) s; container env: $(docker inspect franken_engine --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -E 'ADAPT_(HALFLIFE|PREFILL_CAP)|MOE_G' | tr '\n' ' ')"
  python3 -I "$HOME/bench/serve_curve_driver.py" "$log" "$O/$arm.res.json" "$N" 2>&1 | tee "$O/$arm.res.txt"
  echo "arm $arm adapter windows in the whole log: $(grep -a -c 'adapt_all pos=' "$log")"
  rig_stop_serving
done
echo "=== SUMMARY decode tok/s per chat (chats 1..$N), mean of the last 7"
for arm in ${SC_ARMS:-shipped hl128 hl2048 nocap}; do
  f="$O/$arm.res.txt"; [ -f "$f" ] || continue
  awk -v a="$arm" '/^chat/ { for (i = 1; i <= NF; i++) if ($i ~ /^tok\/s=/) { split($i, t, "="); v[++n] = t[2] } }
    END { s = ""; for (i = 1; i <= n; i++) s = s sprintf(" %5.2f", v[i]); m = 0; c = 0; for (i = n - 6; i <= n; i++) if (i >= 1) { m += v[i]; c++ }
          printf "%-8s%s | last-7 mean %.2f\n", a, s, (c ? m / c : 0) }' "$f"
done
echo "=== glm_serve_curve exit rc=$rcs $(date -Is)"
exit $rcs
