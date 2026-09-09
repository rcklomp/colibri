#!/bin/bash
# accept_ui.sh -- the gate that runs a REAL BROWSER against the owner's Open WebUI.
#
# Runs on the Mac (the rig has no browser and no Node), driving http://<rig>:3000 exactly as a
# person does: open a new chat, type, send, wait for the first token to appear ON SCREEN.
# It exists because every gate on this track imitated the client and both regressions the
# owner hit lived in the gap between the imitation and the real front end (P7b, 2026-09-09).
#
#   accept_ui.sh [--url http://rome.local:3000] [--max-first-token 25] [--shot dir]
#
# Checks, each a fresh chat with its own question:
#   1. new chat A -> a reply appears, first token within --max-first-token
#   2. new chat B -> same, and within 1.5x of A (a checkpoint that fires once is not a fix)
# Then it runs the rig-side accept_live.sh over ssh, so one command covers both sides.
# Exit 0 only if everything passed. The probe deletes the chats it creates.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
URL="http://rome.local:3000"; MAXFT=25; SHOTDIR=""; RIG="${RIG_HOST:-rome}"
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2;; --max-first-token) MAXFT="$2"; shift 2;;
    --shot) SHOTDIR="$2"; shift 2;; *) echo "unknown argument: $1"; exit 2;;
  esac
done
# rome.local resolves through mDNS to a link-local IPv6 address; curl copes, Chrome does not
# (ERR_NAME_NOT_RESOLVED). Ask the rig for its IPv4 and drive that, keeping any explicit --url.
case "$URL" in
  *.local:*|*.local)
    RIGIP=$(ssh "$RIG" "hostname -I | awk '{print \$1}'" 2>/dev/null | tr -d '\r')
    [ -n "$RIGIP" ] && URL=$(echo "$URL" | sed -E "s#//[^/:]+#//$RIGIP#") ;;
esac
# PW_VERSION is pinned to the driver whose browser build is already in
# ~/Library/Caches/ms-playwright. Unpinned, npm gives the newest driver, its browser revision
# is missing, and the probe falls back to the Mac's Google Chrome -- which runs the same turn
# at 13.6 s against 3.6 s, i.e. it measures Chrome-headless overhead, not the service. Bump
# this and `npx playwright install chromium` together, never one alone.
MODS="$HOME/.cache/colibri-ui/node_modules"
PW_VERSION="${PW_VERSION:-1.62.1}"
have=$(node -e "try{console.log(require('$MODS/playwright-core/package.json').version)}catch(e){console.log('none')}" 2>/dev/null)
if [ "$have" != "$PW_VERSION" ]; then
  echo "--- installing playwright-core@$PW_VERSION into ~/.cache/colibri-ui (have: $have)"
  mkdir -p "$HOME/.cache/colibri-ui" && (cd "$HOME/.cache/colibri-ui" && npm i -q --prefer-offline --no-audit --no-fund "playwright-core@$PW_VERSION") || { echo "npm install failed"; exit 2; }
fi
export COLIBRI_UI_MODULES="$MODS"
command -v node >/dev/null || { echo "REFUSED: no node on this machine"; exit 2; }
curl -s -o /dev/null -m 8 -w '%{http_code}' "$URL/health" | grep -q 200 || { echo "REFUSED: $URL/health is not 200 (wrong network?)"; exit 2; }

# A session token, minted inside the container from its own secret and never printed.
TOKENFILE=$(mktemp); chmod 600 "$TOKENFILE"; trap 'rm -f "$TOKENFILE"' EXIT
ssh "$RIG" 'docker exec -i '"${OWUI_CONTAINER:-open-webui-new}"' python3 - <<'"'"'PY'"'"' 2>/dev/null
import sqlite3, os, jwt
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role = ? limit 1", ("admin",)).fetchone()
print(jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256"))
PY' > "$TOKENFILE"
[ -s "$TOKENFILE" ] || { echo "REFUSED: could not mint a session token on $RIG"; exit 2; }

FAIL=0; A=""; B=""
run() {   # run <label> <question> -> echoes the RESULT line
  local shot=""; [ -n "$SHOTDIR" ] && shot="--shot $SHOTDIR/ui_$1.png"
  OWUI_TOKEN=$(cat "$TOKENFILE") node "$HERE/ui/ui_probe.mjs" --url "$URL" --question "$2" $shot | grep "^RESULT"
}
A=$(run A "Which day comes after Tuesday? Answer in one word.")
B=$(run B "Name one prime number greater than ten. One word.")
v() { echo "$2" | sed -n "s/.*[ ]$1=\([0-9.]*\).*/\1/p"; }
fa=$(v first_token_s "$A"); fb=$(v first_token_s "$B")
oa=$(v ok "$A"); ob=$(v ok "$B")
echo "browser A: $A"
echo "browser B: $B"
chk() { # chk <label> <ok> <first_token>
  if [ "$2" = 1 ] && [ -n "$3" ] && [ "$(echo "$3 <= $MAXFT" | bc -l)" = 1 ]; then
    printf '%-38s %s\n' "$1" "PASS first token ${3}s"
  else printf '%-38s %s\n' "$1" "FAIL ok=$2 first token=${3:-none}s (bound ${MAXFT}s)"; FAIL=1; fi
}
chk "1. browser: new chat A" "$oa" "$fa"
chk "2. browser: new chat B" "$ob" "$fb"
if [ -n "$fa" ] && [ -n "$fb" ] && [ "$(echo "$fb > 1.5 * $fa + 2" | bc -l)" = 1 ]; then
  printf '%-38s %s\n' "   B against A" "FAIL B ${fb}s vs A ${fa}s -- the second chat is not warm"; FAIL=1
fi
echo "--- rig-side acceptance (accept_live.sh)"
ssh "$RIG" '~/src/colibri/tools/hot-expert/accept_live.sh' 2>&1 | grep -v "^\[pin\]\|^CKPT\|^20[0-9][0-9]-" || FAIL=1
echo "=== accept_ui $([ $FAIL = 0 ] && echo PASS || echo FAIL) $(date +%Y-%m-%dT%H:%M:%S%z)"
exit $FAIL
