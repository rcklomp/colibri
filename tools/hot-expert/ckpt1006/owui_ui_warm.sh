#!/bin/bash
# One UI-shaped turn through Open WebUI's own backend (saved chat + session_id, memory on):
# the request the owner's browser sends, minus the browser. With chat_id the backend runs the
# completion as a background task, so the HTTP call returns at once; this waits for the
# gateway's REUSE line of the request it triggered, prints it, and deletes the throwaway chat.
#   owui_ui_warm.sh ["question"]
Q="${1:-Say OK.}"
LAST=$(grep -c "^.*REUSE " ~/glm53_server.log)
docker exec -i open-webui-new python3 - "$Q" <<'PY' 2>&1 | grep -v "^INFO\|^WARNING\|^DEBUG"
import sqlite3, json, os, jwt, urllib.request, uuid, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
def api(method, path, body=None):
    req = urllib.request.Request("http://127.0.0.1:8080" + path, data=json.dumps(body).encode() if body is not None else None, headers=H, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
chat = api("POST", "/api/v1/chats/new", {"chat": {"title": "p7 warm-up (temporary)", "models": ["glm-5.3-flash"], "messages": [], "history": {"messages": {}, "currentId": None}}})
mid = str(uuid.uuid4())
r = api("POST", "/api/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": sys.argv[1]}], "stream": False,
        "features": {"memory": True}, "session_id": "p7-warm-" + mid[:8], "chat_id": chat["id"], "id": mid, "max_tokens": 16})
print("dispatched:", {k: r.get(k) for k in ("status", "task_id", "task_ids")}, "chat", chat["id"][:8])
open("/tmp/owui_warm_chat_id", "w").write(chat["id"])
PY
CID=$(docker exec open-webui-new cat /tmp/owui_warm_chat_id)
for i in $(seq 1 240); do
  n=$(grep -c "^.*REUSE " ~/glm53_server.log); [ "$n" -gt "$LAST" ] && break; sleep 10
done
grep "REUSE \|\[req\]\|CKPT" ~/glm53_server.log | tail -4 | cut -c1-150
docker exec -i open-webui-new python3 - "$CID" <<'PY' 2>&1 | grep -v "^INFO\|^WARNING\|^DEBUG"
import sqlite3, os, jwt, urllib.request, sys
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
req = urllib.request.Request("http://127.0.0.1:8080/api/v1/chats/" + sys.argv[1], headers={"Authorization": "Bearer " + tok}, method="DELETE")
print("throwaway chat deleted:", urllib.request.urlopen(req, timeout=60).status, "left:", c.execute("select count(*) from chat where title like 'p7 warm-up%'").fetchone()[0])
PY
