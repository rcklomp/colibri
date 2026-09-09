#!/bin/bash
# p8_reasoning_case.sh -- THE regression that defines P8, scripted so it runs unattended.
#
# Drives a TWO-TURN conversation through Open WebUI's own backend (saved chat +
# session_id + chat_id + memory on: the request the owner's browser sends, minus
# the browser) whose FIRST reply contains a <think> block, and then sends the
# follow-up exactly as the browser would -- with the assistant turn's VISIBLE
# content, read back out of webui.db, which is all Open WebUI stores.
#
#   p8_reasoning_case.sh [--seed N] [--filler-tokens 500] [--max-tokens 128]
#                        [--question TEXT] [--follow-up TEXT]
#
# PASS iff the follow-up REUSEs prompt_tokens(turn 1) + gen(turn 1), +/- 2, and
# answers in under 15 s. Before P8 those turns reused only the shared prefix and
# cost 45-321 s (record, "Response-time matrix on Open WebUI", 2026-09-09).
#
# The script asserts nothing about WHY, on purpose: the control is the same
# script with COLI_REPLY_PIN=0, which must fail. It does report `restored=`
# (the [reply-pin] line's character count) next to `stored=` (what Open WebUI
# kept), because their difference IS the reasoning that used to be lost.
#
# One machine-readable line at the end:
#   RESULT seed=.. p1=.. gen1=.. t1=.. p2=.. reused2=.. t2=.. expect=.. \
#          stored=.. restored=.. verdict=PASS|FAIL
set -u
C="${OWUI_CONTAINER:-open-webui-new}"; LOG="${GLM53_LOG:-$HOME/glm53_server.log}"
SEED=1; FILLER=500; MAXTOK=128
# The instruction the filler ends in. It has to be one the model REASONS about,
# and that is not a free choice: measured against the served gateway on
# 2026-09-09, the matrix's own "Reply with the single word OK." after 500 tokens
# of filler answers with gen=2 and an EMPTY <think> block at temperature 0
# (the browser rows that reasoned did so at Open WebUI's own sampling settings,
# 3 of 5 -- reasoning is a sampling event there, not a property of the prompt),
# and a turn with no reasoning reuses its prefix perfectly with or without the
# pin, so it would prove nothing. This little system of equations reasons every
# time, at temperature 0 and at 0.8: gen=29, 34 B of reasoning, 6 B of visible
# answer. That is the shape the bug needs -- generated tokens the client cannot
# give back -- with a one-line answer that keeps the arithmetic simple.
ASK=${P8_QUESTION:-"There are 3 boxes. Box A has twice as many balls as box B. Box C has 5 fewer than box A. Together they have 45. How many are in box B? Give only the number."}
Q2=${P8_FOLLOWUP:-"In one word: what colour is a clear sky?"}
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --filler-tokens) FILLER="$2"; shift 2;;
    --max-tokens) MAXTOK="$2"; shift 2;;
    --question) ASK="$2"; shift 2;;
    --follow-up) Q2="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
pgrep -f "openai_[s]erver.py" >/dev/null || { echo "RESULT error=no-gateway verdict=FAIL"; exit 2; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
# The matrix's own first-turn shape: ~500 tokens of filler ending in a one-word
# instruction. It reliably produces a short reply that is mostly thinking, which
# is the only kind of turn this bug touches. Distinct text per seed, for the
# reason ui_matrix.sh records: a shared opening lets an earlier row's prefix
# checkpoint fire on a later one.
python3 - "$FILLER" "$SEED" "$WORK/q1.txt" "$ASK" <<'PY'
import sys, random
n, seed = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(20260909 + 1000 * seed + n)
words = ("rig expert prefill token latency memory shard cache kernel routing budget lane "
         "vector matrix residency window recurrence attention gateway checkpoint slot "
         "throughput profile histogram device queue submit fence tile accumulator").split()
chars, text = int(n * 3.6), []
while sum(len(w) + 1 for w in text) < chars:
    text.append(rng.choice(words))
body = " ".join(text)[:chars]
open(sys.argv[3], "w").write((body + "\n\n" if body else "") + sys.argv[4])
PY

wait_req() {   # wait_req <count-before>  -> the new [req] line
  for _ in $(seq 1 720); do            # up to 60 min: a cold prefix is minutes
    [ "$(grep -c "\[req\] " "$LOG")" -gt "$1" ] && { grep "\[req\] " "$LOG" | tail -1; return 0; }
    sleep 5
  done
  return 1
}

M0=$(grep -c "\[req\] " "$LOG")
CID=$(docker exec -i "$C" python3 - "$(cat "$WORK/q1.txt")" "$MAXTOK" <<'PY' 2>/dev/null | tail -1
import sqlite3, json, os, jwt, urllib.request, uuid, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
def api(method, path, body=None):
    req = urllib.request.Request("http://127.0.0.1:8080" + path,
        data=json.dumps(body).encode() if body is not None else None, headers=H, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=600).read())
chat = api("POST", "/api/v1/chats/new", {"chat": {"title": "p8 reasoning case (temporary)",
    "models": ["glm-5.3-flash"], "messages": [], "history": {"messages": {}, "currentId": None}}})
api("POST", "/api/chat/completions", {"model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": sys.argv[1]}], "stream": False,
    "features": {"memory": True}, "session_id": "p8-" + chat["id"][:8],
    "chat_id": chat["id"], "id": str(uuid.uuid4()), "max_tokens": int(sys.argv[2]),
    "temperature": 0})
print(chat["id"])
PY
)
[ -n "$CID" ] || { echo "RESULT error=dispatch verdict=FAIL"; exit 2; }
cleanup_chat() {
  docker exec -i "$C" python3 - "$CID" <<'PY' >/dev/null 2>&1
import sqlite3, os, jwt, urllib.request, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8080/api/v1/chats/" + sys.argv[1],
    headers={"Authorization": "Bearer " + tok}, method="DELETE"), timeout=60)
PY
}
trap 'cleanup_chat; rm -rf "$WORK"' EXIT

L1=$(wait_req "$M0") || { echo "RESULT error=turn1-timeout verdict=FAIL"; exit 2; }
sleep 4                       # the backend writes the assistant message after the stream ends
# What Open WebUI STORED -- the exact string its front end sends back on the next
# turn. Reading it (rather than the gateway's response body) is the whole point:
# the record's finding is that this string has lost the reasoning.
docker exec -i "$C" python3 - "$CID" > "$WORK/reply.txt" <<'PY' 2>/dev/null
import sqlite3, json, sys
c = sqlite3.connect("/app/backend/data/webui.db")
row = c.execute("select chat from chat where id=?", (sys.argv[1],)).fetchone()
d = json.loads(row[0]) if row else {}
msgs = (d.get("history") or {}).get("messages") or {}
best = ""
for m in msgs.values():
    if m.get("role") == "assistant" and isinstance(m.get("content"), str):
        best = m["content"]
sys.stdout.write(best)
PY
STORED=$(wc -c < "$WORK/reply.txt" | tr -d ' ')
if [ "$STORED" = 0 ]; then echo "RESULT error=empty-reply verdict=FAIL"; exit 2; fi

M1=$(grep -c "\[req\] " "$LOG")
docker exec -i "$C" python3 - "$(cat "$WORK/q1.txt")" "$(cat "$WORK/reply.txt")" "$Q2" "$CID" "$MAXTOK" <<'PY' >/dev/null 2>&1
import sqlite3, json, os, jwt, urllib.request, uuid, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
q1, reply, q2, cid, mt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
body = {"model": "glm-5.3-flash", "messages": [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": q2}],
        "stream": False, "features": {"memory": True},
        "session_id": "p8-" + cid[:8], "chat_id": cid, "id": str(uuid.uuid4()),
        "max_tokens": mt, "temperature": 0}
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/api/chat/completions",
    data=json.dumps(body).encode(), headers=H), timeout=3600).read()
PY
L2=$(wait_req "$M1") || { echo "RESULT error=turn2-timeout verdict=FAIL"; exit 2; }

RID1=$(echo "$L1" | sed -n 's/.*id=\([0-9]*\).*/\1/p')
RID2=$(echo "$L2" | sed -n 's/.*id=\([0-9]*\).*/\1/p')
R2=$(grep " REUSE $RID2 " "$LOG" | tail -1 | awk '{print $(NF-1)}')
RESTORED=$(grep "\[reply-pin\] " "$LOG" | tail -1 | sed -n 's/.*restored=\([0-9]*\).*/\1/p')
echo "turn 1: $L1"
echo "turn 2: $L2  reused=$R2"
grep -E "\[reply-pin\] |\[pin\] |CKPT " "$LOG" | tail -4 | cut -c1-150
python3 - "$L1" "$L2" "${R2:-0}" "$SEED" "$STORED" "${RESTORED:-0}" <<'PY'
import re, sys
d1 = dict(re.findall(r"(\w+)=([\d.]+)", sys.argv[1]))
d2 = dict(re.findall(r"(\w+)=([\d.]+)", sys.argv[2]))
p1, g1 = int(d1["prompt_tokens"]), int(d1["gen"])
p2, t2 = int(d2["prompt_tokens"]), float(d2["ttft"])
reused = int(sys.argv[3]); expect = p1 + g1
ok = abs(reused - expect) <= 2 and t2 < 15.0
print(f"RESULT seed={sys.argv[4]} p1={p1} gen1={g1} t1={d1['ttft']} p2={p2} "
      f"reused2={reused} t2={t2:.2f} expect={expect} stored={sys.argv[5]} "
      f"restored={sys.argv[6]} verdict={'PASS' if ok else 'FAIL'}")
print(f"  bound: |reused - (p1+gen1)| <= 2  -> |{reused} - {expect}| = {abs(reused-expect)}; "
      f"ttft < 15 s -> {t2:.2f}")
sys.exit(0 if ok else 1)
PY
