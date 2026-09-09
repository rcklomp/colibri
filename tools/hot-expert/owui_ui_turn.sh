#!/bin/bash
# One UI-shaped turn through Open WebUI's own backend, on the rig: saved chat + session_id +
# memory on, i.e. the request the owner's browser sends, minus the browser. With chat_id the
# backend runs the completion as a background task, so this waits for the gateway's next
# [req] line, prints it, deletes the throwaway chat, and ends with one machine-readable line:
#   RESULT prompt=<tokens> reused=<tokens> ttft=<s> gen=<tokens>
# Usage: owui_ui_turn.sh ["question"]     (container: $OWUI_CONTAINER, default open-webui-new)
Q="${1:-Say OK.}"; C="${OWUI_CONTAINER:-open-webui-new}"; LOG="${GLM53_LOG:-$HOME/glm53_server.log}"
LAST=$(grep -c "\[req\] " "$LOG")
CID=$(docker exec -i "$C" python3 - "$Q" <<'PY' 2>/dev/null | tail -1
import sqlite3, json, os, jwt, urllib.request, uuid, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
def api(method, path, body=None):
    req = urllib.request.Request("http://127.0.0.1:8080" + path, data=json.dumps(body).encode() if body is not None else None, headers=H, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
chat = api("POST", "/api/v1/chats/new", {"chat": {"title": "acceptance turn (temporary)", "models": ["glm-5.3-flash"], "messages": [], "history": {"messages": {}, "currentId": None}}})
mid = str(uuid.uuid4())
api("POST", "/api/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": sys.argv[1]}], "stream": False,
    "features": {"memory": True}, "session_id": "accept-" + mid[:8], "chat_id": chat["id"], "id": mid, "max_tokens": 16})
print(chat["id"])
PY
)
if [ -z "$CID" ]; then echo "RESULT error=dispatch"; exit 2; fi
for i in $(seq 1 360); do   # up to 60 min: a changed tool block is a cold prefill
  n=$(grep -c "\[req\] " "$LOG"); [ "$n" -gt "$LAST" ] && break; sleep 10
done
LINE=$(grep "\[req\] " "$LOG" | tail -1)
RID=$(echo "$LINE" | sed -n 's/.*id=\([0-9]*\).*/\1/p')
REUSED=$(grep " REUSE $RID " "$LOG" | tail -1 | awk '{print $(NF-1)}')
grep "CKPT\|\[pin\]" "$LOG" | tail -3 | cut -c1-140
echo "$LINE reused=$REUSED" | cut -c1-160
docker exec -i "$C" python3 - "$CID" <<'PY' 2>/dev/null
import sqlite3, os, jwt, urllib.request, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
req = urllib.request.Request("http://127.0.0.1:8080/api/v1/chats/" + sys.argv[1], headers={"Authorization": "Bearer " + tok}, method="DELETE")
urllib.request.urlopen(req, timeout=60)
PY
python3 - "$LINE reused=$REUSED" <<'PY'
import re, sys
m = {k: v for k, v in re.findall(r"(\w+)=([\d.]+)", sys.argv[1])}
print(f"RESULT prompt={m.get('prompt_tokens','?')} reused={m.get('reused', '0')} ttft={m.get('ttft','?')} gen={m.get('gen','?')}")
PY
