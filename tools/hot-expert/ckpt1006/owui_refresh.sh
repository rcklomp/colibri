#!/bin/bash
# owui_refresh.sh -- make Open WebUI re-read its model list (GET /api/models from inside its container with a minted admin token) and wait until glm-5.3-flash is in it.
# Open WebUI caches the list: after the gateway has been down, the first chat gets "Model not found" (HTTP 400) until something refreshes it (a person's page load does).
for i in $(seq 1 40); do
  ok=$(docker exec -i "${OWUI_CONTAINER:-open-webui-new}" python3 - <<'PY' 2>/dev/null
import sqlite3, os, jwt, urllib.request, json
c = sqlite3.connect("/app/backend/data/webui.db"); uid, = c.execute("select id from user where role='admin' limit 1").fetchone()
tok = jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256")
r = urllib.request.Request("http://127.0.0.1:8080/api/models", headers={"Authorization": "Bearer " + tok})
d = json.loads(urllib.request.urlopen(r, timeout=30).read())
print("yes" if any(m.get("id") == "glm-5.3-flash" for m in d.get("data", [])) else "no")
PY
)
  [ "$ok" = yes ] && { echo "open-webui lists glm-5.3-flash after $i tries"; exit 0; }
  sleep 3
done
echo "open-webui still does not list glm-5.3-flash"; exit 1
