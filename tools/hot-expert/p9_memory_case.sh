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
#   p9_memory_case.sh [--seed N] [--max-tokens 48] [--memories 4]
#
# Two turns through Open WebUI's own backend with `features.memory` on, four
# TEMPORARY `context` memories added before and deleted after -- on every exit
# path, including a failure. This is the shape §P7 measured for this behaviour
# (turn 1 177 tokens, turn 2 205): no `session_id`, so no 24-tool block, because
# the tool block is P7b's subject and not this one. It also keeps the case
# CHEAP: a 4 400-token UI-shaped turn 1 whose memory block has changed matches
# no checkpoint and costs ~11 minutes of prefill per run, measured here
# 2026-09-10 -- nine runs of that would be two hours of gate for a behaviour a
# 200-token prompt shows exactly as well.
#
# The memories' TEXT is fixed (only the questions carry the run's nonce), so the
# block is byte-stable between runs and the engine's checkpoints are not churned
# by the gate itself.
#
# PASS iff turn 2 REUSEs prompt_tokens(turn 1) + gen(turn 1), +/- 2. The script
# asserts nothing about WHY: the control is the same script with the mechanism
# off, which must fail, or the block did not re-rank and the case proves nothing.
#
#   RESULT seed=.. p1=.. gen1=.. t1=.. p2=.. reused2=.. t2=.. expect=.. \
#          memories=.. verdict=PASS|FAIL
set -u
C="${OWUI_CONTAINER:-open-webui-new}"; LOG="${GLM53_LOG:-$HOME/glm53_server.log}"
SEED=1; MAXTOK=48; NMEM=4
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --max-tokens) MAXTOK="$2"; shift 2;;
    --memories) NMEM="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
pgrep -f "openai_[s]erver.py" >/dev/null || { echo "RESULT error=no-gateway verdict=FAIL"; exit 2; }

# A nonce per run, in the QUESTIONS only. A fixed pair makes a second run send an
# IDENTICAL turn 2, which the engine cannot reuse by design (kv_prefix_reuse
# needs at least one new token) and which reads as a regression that is really
# the harness repeating itself -- 2026-09-09, accept_live check 3.
N="$$-$(date +%s)-$SEED"

MEMFILE=$(mktemp); trap 'rm -f "$MEMFILE"' EXIT
docker exec -i "$C" python3 - "$NMEM" > "$MEMFILE" <<'PY' 2>/dev/null
import sqlite3, json, os, jwt, urllib.request, sys
nmem = int(sys.argv[1])
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
def api(method, path, body=None):
    req = urllib.request.Request("http://127.0.0.1:8080" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=H, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
# Four unrelated topics, so the top-k retrieval genuinely re-orders between a
# question about food and a question about the sky. ONE memory would be
# byte-stable and prove nothing (P7 step 0: with one memory turn 2 already
# reused 162/177). The text never changes between runs, so the block does not.
FACTS = ["The user is allergic to shellfish",
         "The user's favourite city to visit is Lisbon",
         "The user plays the cello on Thursday evenings",
         "The user prefers vegetarian food when cooking for friends",
         "The user's car is a blue estate",
         "The user reads science fiction before sleeping"]
print(json.dumps([api("POST", "/api/v1/memories/add", {"content": f})["id"]
                  for f in FACTS[:nmem]]))
PY
MIDS=$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))))" "$MEMFILE" 2>/dev/null)
[ -n "$MIDS" ] || { echo "RESULT error=memories verdict=FAIL"; exit 2; }
cleanup() {
  docker exec -i "$C" python3 - $MIDS <<'PY' >/dev/null 2>&1
import sqlite3, os, jwt, urllib.request, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
for mid in sys.argv[1:]:
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8080/api/v1/memories/" + mid,
            headers={"Authorization": "Bearer " + tok}, method="DELETE"), timeout=60)
    except Exception:
        pass
PY
  rm -f "$MEMFILE"
}
trap cleanup EXIT

M0=$(grep -c "\[req\] " "$LOG")
docker exec -i -e P9_NONCE="$N" -e P9_MAXTOK="$MAXTOK" "$C" python3 - >/dev/null 2>&1 <<'PY'
import sqlite3, json, os, jwt, urllib.request
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
n, mt = os.environ["P9_NONCE"], int(os.environ["P9_MAXTOK"])
def chat(msgs):
    req = urllib.request.Request("http://127.0.0.1:8080/api/chat/completions",
        data=json.dumps({"model": "glm-5.3-flash", "messages": msgs, "stream": False,
                         "features": {"memory": True}, "max_tokens": mt,
                         "temperature": 0}).encode(), headers=H)
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return r["choices"][0]["message"]["content"]
# Turn 1 asks about food, turn 2 about the sky: two different vector queries over
# a growing history, which is what makes the block re-rank between them.
q1 = f"I am planning dinner for friends this weekend. What should I cook? [{n}]"
a1 = chat([{"role": "user", "content": q1}])
# What the front end stores and sends back is the VISIBLE content, exactly this.
chat([{"role": "user", "content": q1},
      {"role": "assistant", "content": a1},
      {"role": "user", "content": f"In one word: what colour is a clear sky? [{n}]"}])
PY
rc=$?
[ "$rc" = 0 ] || { echo "RESULT error=dispatch verdict=FAIL"; exit 2; }
for _ in $(seq 1 720); do
  [ "$(grep -c "\[req\] " "$LOG")" -ge $((M0 + 2)) ] && break
  sleep 5
done
L1=$(grep "\[req\] " "$LOG" | tail -2 | head -1)
L2=$(grep "\[req\] " "$LOG" | tail -1)
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
ok = abs(reused - expect) <= 2
print(f"RESULT seed={sys.argv[4]} p1={p1} gen1={g1} t1={d1['ttft']} p2={p2} "
      f"reused2={reused} t2={t2:.2f} expect={expect} memories={sys.argv[5]} "
      f"verdict={'PASS' if ok else 'FAIL'}")
print(f"  bound: |reused - (p1+gen1)| <= 2 -> |{reused} - {expect}| = {abs(reused-expect)}")
sys.exit(0 if ok else 1)
PY
