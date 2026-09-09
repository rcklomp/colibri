#!/bin/bash
# accept_live.sh -- the acceptance gate on the USER'S path, run on the rig against the SERVED
# gateway. A binary is not "in service" until this exits 0; every chain calls it after the
# restart and reverts on failure. It exists because P7b: every P7 gate proved a checkpoint
# restore worked and none looked at the request AFTER the restore, and the owner found the
# 380-second new chat himself. So this gate always measures the request after the one under
# test, on Open WebUI's own backend (saved chat + session + memory = what the browser sends).
#
#   accept_live.sh            all four checks (exit 1 on any miss)
#   accept_live.sh --canary   checks 1-2 only: keeps the tool-block checkpoint current when
#                             Open WebUI's tool list drifts (cron, daily); exit 1 on a miss
#
# Checks (bounds are the measured values of 2026-09-09 with headroom, not aspirations):
#  1. UI-shaped new chat A            -> may be cold (a changed tool block is captured here)
#  2. UI-shaped new chat B (other Q)  -> reused >= prompt-256 and ttft <= 60 s  (the P7b case)
#  3. API two-turn, memory on         -> turn 2 reused >= prompt-32 and ttft <= 15 s (P6/P7 pin)
#  4. abandoned 1 000-token request   -> the short request behind it answers in <= 45 s (CANCEL)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); LOG="${GLM53_LOG:-$HOME/glm53_server.log}"; MODE="${1:-full}"
K=$(cat "$HOME/.colibri_api_key"); URL=http://127.0.0.1:8081; FAIL=0
say() { printf '%-44s %s\n' "$1" "$2"; }
pgrep -f "openai_[s]erver.py" >/dev/null || { echo "REFUSED: gateway not running"; exit 2; }
[ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 -H "Authorization: Bearer $K" $URL/v1/models)" = 200 ] || { echo "REFUSED: /v1/models not 200"; exit 2; }
echo "=== accept_live $MODE $(date -Is) binary=$(sha256sum "$HOME/src/colibri/c/glm53" | cut -c1-12)"
A=$("$HERE/owui_ui_turn.sh" "Which day comes after Tuesday? One word." | tee /dev/stderr | grep "^RESULT")
B=$("$HERE/owui_ui_turn.sh" "Name one prime number greater than ten. One word." | tee /dev/stderr | grep "^RESULT")
v() { echo "$2" | sed -n "s/.*$1=\([0-9.]*\).*/\1/p"; }
pa=$(v prompt "$A"); ra=$(v reused "$A"); ta=$(v ttft "$A")
pb=$(v prompt "$B"); rb=$(v reused "$B"); tb=$(v ttft "$B")
say "1. UI new chat A (may be cold)" "prompt=$pa reused=$ra ttft=${ta}s"
if [ -n "$pb" ] && [ "$rb" -ge $((pb - 256)) ] && [ "${tb%.*}" -le 60 ]; then say "2. UI new chat B warm" "PASS prompt=$pb reused=$rb ttft=${tb}s";
else say "2. UI new chat B warm" "FAIL prompt=$pb reused=$rb ttft=${tb}s (need reused>=$((pb-256)), ttft<=60)"; FAIL=1; fi
[ "$MODE" = "--canary" ] && { echo "=== accept_live canary $([ $FAIL = 0 ] && echo PASS || echo FAIL)"; exit $FAIL; }
# 3. API two-turn with memory on (the pin), through Open WebUI's backend, no chat_id (no tools: fast)
T=$(docker exec -i "${OWUI_CONTAINER:-open-webui-new}" python3 - <<'PY' 2>/dev/null
import sqlite3, json, os, jwt, urllib.request, time
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
def chat(msgs):
    t0 = time.time(); req = urllib.request.Request("http://127.0.0.1:8080/api/chat/completions", data=json.dumps({"model": "glm-5.3-flash", "messages": msgs, "stream": False, "features": {"memory": True}, "max_tokens": 24}).encode(), headers=H)
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read()); return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"], time.time() - t0
a, p1, t1 = chat([{"role": "user", "content": "What is two plus two? One word."}])
b, p2, t2 = chat([{"role": "user", "content": "What is two plus two? One word."}, {"role": "assistant", "content": a}, {"role": "user", "content": "And three plus three? One word."}])
print(f"p2={p2} t2={t2:.1f}")
PY
)
r2=$(grep " REUSE " "$LOG" | tail -1 | awk '{print $(NF-1)}'); p2=$(v p2 "$T"); t2=$(v t2 "$T")
if [ -n "$p2" ] && [ "$r2" -ge $((p2 - 32)) ] && [ "${t2%.*}" -le 15 ]; then say "3. API follow-up turn (pin)" "PASS prompt=$p2 reused=$r2 total=${t2}s";
else say "3. API follow-up turn (pin)" "FAIL prompt=$p2 reused=$r2 total=${t2}s (need reused>=$((p2-32)), <=15 s)"; FAIL=1; fi
# 4. an abandoned long request must not hold the engine: curl gives up at 8 s, the next short request must answer
BIG=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * 130))")
curl -s -m 8 -o /dev/null -H "Authorization: Bearer $K" -H "Content-Type: application/json" $URL/v1/chat/completions \
  -d "{\"model\":\"glm-5.3-flash\",\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$BIG Summarise this in one word." | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],\"max_tokens\":32}" || true
t0=$(date +%s.%N)
curl -s -m 300 -o /dev/null -H "Authorization: Bearer $K" -H "Content-Type: application/json" $URL/v1/chat/completions \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}'
t4=$(python3 -c "print(round($(date +%s.%N) - $t0, 1))")
if [ "${t4%.*}" -le 45 ]; then say "4. request behind an abandoned one" "PASS answered in ${t4}s"; else say "4. request behind an abandoned one" "FAIL ${t4}s (need <=45)"; FAIL=1; fi
grep "CANCEL" "$LOG" | tail -1 | cut -c1-120
echo "=== accept_live $([ $FAIL = 0 ] && echo PASS || echo FAIL) $(date -Is)"
exit $FAIL
