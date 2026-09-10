#!/bin/bash
# p9_ui_multiturn.sh -- step 5: the invariant in PRODUCTION SHAPE.
#
# Runs on the Mac (the rig has no browser and no Node). One conversation of ten
# turns in the owner's Open WebUI, in a real Chromium at the rig's IPv4, tools
# and memory ON -- and then the rig's own log for exactly those requests: zero
# `MISMATCH`, zero `ledger=broken`, and `expect_reuse == engine_reuse` on every
# continuation.
#
# Ten turns, not two, because every gate on this track so far has proved one
# request and looked away. A ledger that drifts by one byte on turn 7 costs the
# owner a full re-prefill and, before P9, said nothing at all about it.
#
#   p9_ui_multiturn.sh [--turns 10] [--url http://rome.local:3000] [--out FILE]
#
# Exit 0 only if the browser finished every turn AND the log is clean.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
URL="http://rome.local:3000"; RIG="${RIG_HOST:-rome}"; TURNS=10; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --turns) TURNS="$2"; shift 2;;
    --url) URL="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
# rome.local resolves through mDNS to a link-local IPv6 address; curl copes,
# Chrome does not. Ask the rig for its IPv4 and drive that.
case "$URL" in *.local:*|*.local)
  RIGIP=$(ssh "$RIG" "hostname -I | awk '{print \$1}'" 2>/dev/null | tr -d '\r')
  [ -n "$RIGIP" ] && URL=$(echo "$URL" | sed -E "s#//[^/:]+#//$RIGIP#") ;;
esac
MODS="$HOME/.cache/colibri-ui/node_modules"; export COLIBRI_UI_MODULES="$MODS"
[ -d "$MODS/playwright-core" ] || { echo "run accept_ui.sh once first (installs the driver)"; exit 2; }
command -v node >/dev/null || { echo "REFUSED: no node on this machine"; exit 2; }
curl -s -o /dev/null -m 8 -w '%{http_code}' "$URL/health" | grep -q 200 \
  || { echo "REFUSED: $URL/health is not 200 (wrong network?)"; exit 2; }

TOKENFILE=$(mktemp); chmod 600 "$TOKENFILE"; trap 'rm -f "$TOKENFILE"' EXIT
ssh "$RIG" 'docker exec -i '"${OWUI_CONTAINER:-open-webui-new}"' python3 - <<'"'"'PY'"'"' 2>/dev/null
import sqlite3, os, jwt
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role = ? limit 1", ("admin",)).fetchone()
print(jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256"))
PY' > "$TOKENFILE"
[ -s "$TOKENFILE" ] || { echo "REFUSED: could not mint a session token on $RIG"; exit 2; }

# A nonce, carried into every question: a fixed set makes a second run send an
# identical turn, which the engine cannot reuse by design.
NONCE=$(date +%H%M%S)
MARK=$(ssh "$RIG" 'grep -c "\[ledger\] " ~/glm53_server.log 2>/dev/null || echo 0')
echo "=== p9_ui_multiturn $TURNS turns $(date +%Y-%m-%dT%H:%M:%S%z) nonce=$NONCE ledger_mark=$MARK"
R=$(OWUI_TOKEN=$(cat "$TOKENFILE") node "$HERE/ui/ui_probe.mjs" --url "$URL" \
      --question "Hello. Answer each of my next questions in one word. [$NONCE]" \
      --follow-ups $((TURNS - 1)) --nonce "$NONCE" --timeout 1800 | tee /dev/stderr | grep "^RESULT")
FAIL=0
ok=$(echo "$R" | sed -n 's/.*[ ]ok=\([0-9]*\).*/\1/p')
got=$(echo "$R" | sed -n 's/.*[ ]turns=\([0-9]*\).*/\1/p')
[ "$ok" = 1 ] || { echo "browser: FAIL ($R)"; FAIL=1; }
[ "${got:-0}" -ge "$TURNS" ] || { echo "browser: only ${got:-0} of $TURNS turns completed"; FAIL=1; }

echo "--- the rig's [ledger] lines for those $TURNS turns"
ssh "$RIG" "python3 - ~/glm53_server.log $MARK" <<'PY' | tee "${OUT:-/dev/null}"
import sys
log, mark = sys.argv[1], int(sys.argv[2])
# The server log is timestamped by the awk pipe in ~/start_glm53.sh, so every line reads
# "2026-09-10 01:44:53 [ledger] …". startswith() therefore matched NOTHING and this check
# could only ever report zero rows — it happened to fail closed, but a real MISMATCH would
# have been just as invisible. Match the marker anywhere in the line.
rows = [l.strip() for l in open(log, errors="replace") if "[ledger] " in l][mark:]
for l in rows:
    print("  " + l[:150])
mism = [l for l in rows if "MISMATCH" in l]
broken = [l for l in rows if "ledger=broken" in l]
checked = [l for l in rows if "expect_reuse=" in l and "expect_reuse=-" not in l]
print(f"LEDGER lines={len(rows)} checked={len(checked)} MISMATCH={len(mism)} broken={len(broken)}")
sys.exit(1 if (mism or broken or not rows) else 0)
PY
[ "${PIPESTATUS[0]}" = 0 ] || FAIL=1
echo "=== p9_ui_multiturn $([ $FAIL = 0 ] && echo PASS || echo FAIL) $(date +%Y-%m-%dT%H:%M:%S%z)"
exit $FAIL
