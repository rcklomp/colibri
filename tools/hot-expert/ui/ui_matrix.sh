#!/bin/bash
# ui_matrix.sh -- wall time against context size, measured in the owner's browser.
#
# Tools and memory ON, i.e. what Open WebUI actually sends: a fixed ~4 000-token prefix
# (24 builtin tools + the memory block) in front of every first turn. Context grows the way a
# person grows it -- by pasting text into a new chat -- so each row is: new chat with N pasted
# tokens, then a follow-up turn in the same chat.
#
#   ui_matrix.sh [--sizes 0,500,1000,2000,3500] [--out matrix.tsv]
#
# Each row records what the person waits for (browser first token, browser total) next to what
# the engine reports for the same two requests (prompt tokens, reused, ttft, decode).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); TOP=$(cd "$HERE/.." && pwd)
SIZES="0,500,1000,2000,3500"; OUT=""; RIG="${RIG_HOST:-rome}"; URL="http://rome.local:3000"
while [ $# -gt 0 ]; do
  case "$1" in --sizes) SIZES="$2"; shift 2;; --out) OUT="$2"; shift 2;; --url) URL="$2"; shift 2;;
               *) echo "unknown argument: $1"; exit 2;; esac
done
case "$URL" in *.local:*|*.local)
  RIGIP=$(ssh "$RIG" "hostname -I | awk '{print \$1}'" 2>/dev/null | tr -d '\r')
  [ -n "$RIGIP" ] && URL=$(echo "$URL" | sed -E "s#//[^/:]+#//$RIGIP#") ;;
esac
MODS="$HOME/.cache/colibri-ui/node_modules"; export COLIBRI_UI_MODULES="$MODS"
[ -d "$MODS/playwright-core" ] || { echo "run accept_ui.sh once first (installs the driver)"; exit 2; }
TOKENFILE=$(mktemp); chmod 600 "$TOKENFILE"; WORK=$(mktemp -d); trap 'rm -rf "$TOKENFILE" "$WORK"' EXIT
ssh "$RIG" 'docker exec -i '"${OWUI_CONTAINER:-open-webui-new}"' python3 - <<'"'"'PY'"'"' 2>/dev/null
import sqlite3, os, jwt
c = sqlite3.connect("/app/backend/data/webui.db")
uid, = c.execute("select id from user where role = ? limit 1", ("admin",)).fetchone()
print(jwt.encode({"id": uid}, os.environ["WEBUI_SECRET_KEY"], algorithm="HS256"))
PY' > "$TOKENFILE"
[ -s "$TOKENFILE" ] || { echo "could not mint a session token"; exit 2; }

hdr=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "pasted_tok" "turn" "engine_prompt" "engine_reused" "engine_new" "engine_ttft_s" "browser_first_s" "browser_total_s" "gen_tok")
echo "$hdr"; [ -n "$OUT" ] && echo "$hdr" > "$OUT"
for n in $(echo "$SIZES" | tr ',' ' '); do
  # deterministic filler: ~3.6 chars per token on this English prose (the record's calibration)
  python3 - "$n" "$WORK/paste.txt" <<'PY'
import sys, random
# Distinct text per size, not a longer cut of the same passage: with a shared opening the
# engine's prefix checkpoint from an earlier row fires on a later one and the table reads
# warmer than a person pasting an unrelated document ever would. Caught on the first run
# (reused 4 781 and 5 141 where the tool prefix alone is 4 419).
n = int(sys.argv[1]); chars = int(n * 3.6)
rng = random.Random(20260909 + n)
words = ("rig expert prefill token latency memory shard cache kernel routing budget lane "
         "vector matrix residency window recurrence attention gateway checkpoint slot "
         "throughput profile histogram device queue submit fence tile accumulator").split()
text = []
while sum(len(w) + 1 for w in text) < chars:
    text.append(rng.choice(words))
body = " ".join(text)[:chars]
open(sys.argv[2], "w").write((body + "\n\n" if body else "") + "Reply with the single word OK.")
PY
  mark=$(ssh "$RIG" 'grep -c "\[req\] " ~/glm53_server.log')
  R=$(OWUI_TOKEN=$(cat "$TOKENFILE") node "$HERE/ui_probe.mjs" --url "$URL" --text-file "$WORK/paste.txt" \
        --follow-up "In one word: what colour is a clear sky?" --timeout 1800 | grep "^RESULT")
  v() { echo "$R" | sed -n "s/.* $1=\([0-9.-]*\).*/\1/p"; }
  # the engine's own two lines for the two turns we just drove
  E=$(ssh "$RIG" "grep '\[req\] \| REUSE ' ~/glm53_server.log | tail -20")
  read -r p1 r1 t1 g1 p2 r2 t2 g2 <<<"$(python3 - "$mark" <<PY
import re, subprocess, sys
log = """$E"""
reqs = [l for l in log.splitlines() if "[req] " in l][-2:]
reuse = {m.group(1): m.group(2) for m in (re.search(r"REUSE (\d+) (\d+) (\d+)", l) for l in log.splitlines()) if m}
out = []
for l in reqs:
    d = dict(re.findall(r"(\w+)=([\d.]+)", l))
    rid = re.search(r"id=(\d+)", l).group(1)
    out += [d.get("prompt_tokens", "?"), reuse.get(rid, "?"), d.get("ttft", "?"), d.get("gen", "?")]
print(" ".join(out) if len(out) == 8 else "? ? ? ? ? ? ? ?")
PY
)"
  row1=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$n" 1 "$p1" "$r1" "$((${p1:-0} - ${r1:-0}))" "$t1" "$(v first_token_s)" "$(v done_s)" "$g1")
  row2=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$n" 2 "$p2" "$r2" "$((${p2:-0} - ${r2:-0}))" "$t2" "$(v follow_first_s)" "$(v follow_done_s)" "$g2")
  echo "$row1"; echo "$row2"
  [ -n "$OUT" ] && { echo "$row1" >> "$OUT"; echo "$row2" >> "$OUT"; }
done
