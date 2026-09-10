#!/bin/bash
# p9_memory_case.sh -- client behaviour (a): the re-ranked memory block.
#
# Open WebUI appends its <memory_context> block to the END of the leading system
# message and rebuilds it from a vector search over the last seven user
# messages, so as soon as the account holds more than one memory the block's
# ITEM ORDER moves every turn. The rendered prompt then stops matching the token
# sequence the KV slot holds and the conversation re-prefills: measured
# 2026-09-07, turn 2 reused 0/195 and cost 21.6 s where it should cost ~2 s.
# P7's pin patched that one behaviour; P9's ledger subsumes it by replaying the
# head system message the renderer wrote on turn 1.
#
#   p9_memory_case.sh [--seed N] [--max-tokens 96] [--memories 4]
#
# Two turns through Open WebUI's own backend (saved chat + session_id + chat_id +
# features.memory, i.e. the request the owner's browser sends, minus the
# browser), with four TEMPORARY `context` memories added before and deleted
# after -- on every exit path, including a failure.
#
# PASS iff turn 2 REUSEs prompt_tokens(turn 1) + gen(turn 1), +/- 2, and answers
# in under 15 s. The script asserts nothing about WHY: the control is the same
# script with the mechanism off, which must fail, or the block did not re-rank
# and the case proves nothing.
#
#   RESULT seed=.. p1=.. gen1=.. t1=.. p2=.. reused2=.. t2=.. expect=.. \
#          memories=.. verdict=PASS|FAIL
set -u
C="${OWUI_CONTAINER:-open-webui-new}"; LOG="${GLM53_LOG:-$HOME/glm53_server.log}"
SEED=1; MAXTOK=96; NMEM=4
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --max-tokens) MAXTOK="$2"; shift 2;;
    --memories) NMEM="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
pgrep -f "openai_[s]erver.py" >/dev/null || { echo "RESULT error=no-gateway verdict=FAIL"; exit 2; }

# A nonce per run. A fixed pair of questions makes a second run send an
# IDENTICAL turn 2, which the engine cannot reuse by design (kv_prefix_reuse
# needs at least one new token) and which reads as a regression that is really
# the harness repeating itself -- 2026-09-09, accept_live check 3.
N="$$-$(date +%s)-$SEED"
# Four unrelated topics, so the top-k retrieval genuinely re-orders between a
# question about food and a question about travel. One memory is byte-stable and
# would prove nothing (P7 step 0: with one memory turn 2 already reused 162/177).
Q1="I am planning dinner for friends this weekend. What should I cook? [$N]"
Q2="In one word: what colour is a clear sky? [$N]"

wait_req() {   # wait_req <count-before> -> the new [req] line
  for _ in $(seq 1 720); do
    [ "$(grep -c "\[req\] " "$LOG")" -gt "$1" ] && { grep "\[req\] " "$LOG" | tail -1; return 0; }
    sleep 5
  done
  return 1
}

STATE=$(mktemp); trap 'rm -f "$STATE"' EXIT
# --- add the temporary memories, create the chat, drive turn 1 --------------
M0=$(grep -c "\[req\] " "$LOG")
docker exec -i "$C" python3 - "$Q1" "$MAXTOK" "$N" "$NMEM" > "$STATE" <<'PY' 2>/dev/null
import sqlite3, json, os, jwt, urllib.request, uuid, sys
q1, mt, nonce, nmem = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
def api(method, path, body=None, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:8080" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=H, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
FACTS = [f"The user is allergic to shellfish [{nonce}]",
         f"The user's favourite city to visit is Lisbon [{nonce}]",
         f"The user plays the cello on Thursday evenings [{nonce}]",
         f"The user prefers vegetarian food when cooking for friends [{nonce}]",
         f"The user's car is a blue estate [{nonce}]",
         f"The user reads science fiction before sleeping [{nonce}]"]
ids = []
for fact in FACTS[:nmem]:
    row = api("POST", "/api/v1/memories/add", {"content": fact})
    ids.append(row["id"])
chat = api("POST", "/api/v1/chats/new", {"chat": {
    "title": "p9 memory case (temporary)", "models": ["glm-5.3-flash"],
    "messages": [], "history": {"messages": {}, "currentId": None}}})
api("POST", "/api/chat/completions", {"model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": q1}], "stream": False,
    "features": {"memory": True}, "session_id": "p9-" + chat["id"][:8],
    "chat_id": chat["id"], "id": str(uuid.uuid4()), "max_tokens": mt,
    "temperature": 0}, timeout=3600)
print(json.dumps({"chat": chat["id"], "memories": ids}))
PY
CID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['chat'])" "$STATE" 2>/dev/null)
MIDS=$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['memories']))" "$STATE" 2>/dev/null)
[ -n "$CID" ] || { echo "RESULT error=dispatch verdict=FAIL"; exit 2; }

cleanup() {
  docker exec -i "$C" python3 - "$CID" $MIDS <<'PY' >/dev/null 2>&1
import sqlite3, os, jwt, urllib.request, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok}
def delete(path):
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8080" + path, headers=H, method="DELETE"), timeout=60)
    except Exception:
        pass
delete("/api/v1/chats/" + sys.argv[1])
for mid in sys.argv[2:]:
    delete("/api/v1/memories/" + mid)
PY
  rm -f "$STATE"
}
trap cleanup EXIT

L1=$(wait_req "$M0") || { echo "RESULT error=turn1-timeout verdict=FAIL"; exit 2; }
sleep 4                       # the backend writes the assistant message after the stream ends

# What Open WebUI STORED for turn 1 -- the exact string its front end sends back.
REPLY=$(docker exec -i "$C" python3 - "$CID" <<'PY' 2>/dev/null
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
)
[ -n "$REPLY" ] || { echo "RESULT error=empty-reply verdict=FAIL"; exit 2; }

# --- turn 2: the SAME conversation, with the block re-ranked by the backend --
M1=$(grep -c "\[req\] " "$LOG")
docker exec -i "$C" python3 - "$Q1" "$REPLY" "$Q2" "$CID" "$MAXTOK" >/dev/null 2>&1 <<'PY'
import sqlite3, json, os, jwt, urllib.request, uuid, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
q1, reply, q2, cid, mt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/api/chat/completions",
    data=json.dumps({"model": "glm-5.3-flash", "messages": [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": q2}],
        "stream": False, "features": {"memory": True},
        "session_id": "p9-" + cid[:8], "chat_id": cid, "id": str(uuid.uuid4()),
        "max_tokens": mt, "temperature": 0}).encode(), headers=H), timeout=3600).read()
PY
L2=$(wait_req "$M1") || { echo "RESULT error=turn2-timeout verdict=FAIL"; exit 2; }

RID2=$(echo "$L2" | sed -n 's/.*id=\([0-9]*\).*/\1/p')
R2=$(grep " REUSE $RID2 " "$LOG" | tail -1 | awk '{print $(NF-1)}')
echo "turn 1: $L1"
echo "turn 2: $L2  reused=$R2"
grep -E "\[ledger\] |\[pin\] |\[reply-pin\] " "$LOG" | tail -4 | cut -c1-160
python3 - "$L1" "$L2" "${R2:-0}" "$SEED" "$NMEM" <<'PY'
import re, sys
d1 = dict(re.findall(r"(\w+)=([\d.]+)", sys.argv[1]))
d2 = dict(re.findall(r"(\w+)=([\d.]+)", sys.argv[2]))
p1, g1 = int(d1["prompt_tokens"]), int(d1["gen"])
p2, t2 = int(d2["prompt_tokens"]), float(d2["ttft"])
reused = int(sys.argv[3]); expect = p1 + g1
ok = abs(reused - expect) <= 2 and t2 < 15.0
print(f"RESULT seed={sys.argv[4]} p1={p1} gen1={g1} t1={d1['ttft']} p2={p2} "
      f"reused2={reused} t2={t2:.2f} expect={expect} memories={sys.argv[5]} "
      f"verdict={'PASS' if ok else 'FAIL'}")
print(f"  bound: |reused - (p1+gen1)| <= 2 -> |{reused} - {expect}| = {abs(reused-expect)}; "
      f"ttft < 15 s -> {t2:.2f}")
sys.exit(0 if ok else 1)
PY
