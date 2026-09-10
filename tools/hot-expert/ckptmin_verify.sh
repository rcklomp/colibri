#!/bin/bash
# Verify GLM53_PREFIX_CKPT_MIN=1024 (already patched into ~/start_glm53.sh and live):
# a short conversation must not capture a checkpoint, and accept_live must pass. Revert if not.
#
# `grep -c X f || echo 0` prints TWO zeros when there is no match (grep prints 0 AND exits 1).
# That parked one chain for an hour tonight and aborted this one under `set -u`; both times the
# same idiom. Count without the fallback and tolerate the exit status.
set -u
LOG=$HOME/glm53_server.log; K=$(cat ~/.colibri_api_key); BK=$HOME/bench/start_glm53.sh.ckptmin_base
count() { local n; n=$(grep -ac "$1" "$LOG" 2>/dev/null) || true; echo "${n:-0}"; }
say() { echo "$(date -u +%H:%M:%S) $*"; }

say "min in service: $(grep -o 'GLM53_PREFIX_CKPT_MIN=[0-9]*' ~/start_glm53.sh)"
mark=$(count "CKPT store")
for q in '{"role":"user","content":"Name three colours, comma separated."}' ; do :; done
curl -s -m 300 -o /dev/null -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  http://127.0.0.1:8081/v1/chat/completions \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Name three colours, comma separated."}],"max_tokens":24}'
curl -s -m 300 -o /dev/null -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  http://127.0.0.1:8081/v1/chat/completions \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Name three colours, comma separated."},{"role":"assistant","content":"Red, blue, green."},{"role":"user","content":"And one more?"}],"max_tokens":24}'
stores=$(( $(count "CKPT store") - mark ))
say "short-chat probe: CKPT store lines +$stores (want 0)"

~/src/colibri/tools/hot-expert/accept_live.sh > /tmp/ckptmin_accept.txt 2>&1; rc=$?
grep -a "^[0-9]\.\|^2b\.\|accept_live " /tmp/ckptmin_accept.txt | cut -c1-125
if [ "$stores" = 0 ] && [ "$rc" = 0 ]; then
  say "KEEPING GLM53_PREFIX_CKPT_MIN=1024 — no small capture, accept_live PASS"
  exit 0
fi
say "REVERTING (stores=$stores accept_live rc=$rc)"
cp -p "$BK" ~/start_glm53.sh
pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53 2>/dev/null
for _ in $(seq 1 60); do pgrep -x glm53 >/dev/null || break; sleep 2; done
SKIP_WARM=1 setsid nohup ~/start_glm53.sh > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 90); do sleep 10
  [ "$(curl -s -o /dev/null -m 15 -w '%{http_code}' -H "Authorization: Bearer $K" http://127.0.0.1:8081/v1/models)" = 200 ] && break
done
say "reverted; gateway $(curl -s -o /dev/null -m 15 -w '%{http_code}' -H "Authorization: Bearer $K" http://127.0.0.1:8081/v1/models)"
exit 1
