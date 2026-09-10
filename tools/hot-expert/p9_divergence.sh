#!/bin/bash
# p9_divergence.sh -- step 4: slow is allowed, wrong is not.
#
# The ledger's three non-continuation verdicts are the ones that can be WRONG in
# a way nobody would notice: a regenerate, an edit of an earlier message and a
# branch all render a transcript that is not the one the engine holds, and if the
# ledger replayed a piece there the user would read an answer to a conversation
# he never had. So each of the three must produce text BYTE-IDENTICAL to what a
# gateway that has never seen that transcript produces for it, and must say
# `ledger=reset` while doing it.
#
#   p9_divergence.sh --phase warm --out DIR [--nonce N]   # build, diverge, save
#   p9_divergence.sh --phase cold --out DIR               # replay on a fresh gateway
#   p9_divergence.sh --phase compare --out DIR            # verdict
#
# "Cold" is a real cold: the gate restarts the gateway between the two phases, so
# the cold run's ledger is empty and its render is the client's own. Greedy
# (temperature 0) throughout, so the only thing that can move the text is the
# prompt.
set -u
LOG="${GLM53_LOG:-$HOME/glm53_server.log}"
PHASE=""; OUT=""; NONCE="$$-$(date +%s)"
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --nonce) NONCE="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
[ -n "$OUT" ] || { echo "usage: p9_divergence.sh --phase warm|cold|compare --out DIR"; exit 2; }
mkdir -p "$OUT"

if [ "$PHASE" = compare ]; then
  rc=0
  for case in regenerate edit branch; do
    w="$OUT/$case.warm.txt"; c="$OUT/$case.cold.txt"
    if [ ! -s "$w" ] || [ ! -s "$c" ]; then
      echo "  $case: NO DATA (warm $(wc -c < "$w" 2>/dev/null || echo 0) B, cold $(wc -c < "$c" 2>/dev/null || echo 0) B)"
      rc=1; continue
    fi
    if cmp -s "$w" "$c"; then
      echo "  $case: IDENTICAL ($(wc -c < "$w" | tr -d ' ') B) $(cat "$OUT/$case.reset" 2>/dev/null || echo 'no ledger=reset line')"
    else
      echo "  $case: DIFFERS"
      diff <(cat "$w") <(cat "$c") | head -6
      rc=1
    fi
    grep -q . "$OUT/$case.reset" 2>/dev/null || { echo "  $case: no \`ledger=reset\` was logged"; rc=1; }
  done
  exit $rc
fi

[ "$PHASE" = warm ] || [ "$PHASE" = cold ] || { echo "unknown phase $PHASE"; exit 2; }
pgrep -f "openai_[s]erver.py" >/dev/null || { echo "REFUSED: the gateway is not running"; exit 2; }

python3 - "$PHASE" "$OUT" "$NONCE" "$LOG" <<'PY'
import json, os, sys, time, urllib.request

phase, out, nonce, log = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
key = open(os.path.expanduser("~/.colibri_api_key")).read().strip()
H = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
URL = os.environ.get("COLI_URL", "http://127.0.0.1:8081") + "/v1/chat/completions"


def marks():
    try:
        return sum(1 for line in open(log, errors="replace") if "[ledger] " in line)
    except OSError:
        return 0


def ledger_lines(since):
    try:
        rows = [l for l in open(log, errors="replace") if "[ledger] " in l]
    except OSError:
        return []
    return rows[since:]


def chat(messages, max_tokens=48):
    body = {"model": "glm-5.3-flash", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=H)
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return r["choices"][0]["message"].get("content") or ""


SEEDS = {
    # Each case gets its own opening, so the three do not share a ledger key and
    # one case's record cannot decide another's verdict.
    "regenerate": "List three colours, comma separated.",
    "edit":       "List three metals, comma separated.",
    "branch":     "List three rivers, comma separated.",
}
FOLLOW = "Now name one animal. One word."
EDITED = "Actually, name one city instead. One word."
BRANCHED = "Instead, name one planet. One word."

for case, opening in SEEDS.items():
    q1 = f"{opening} [{nonce}-{case}]"
    path = os.path.join(out, case + ".json")
    if phase == "warm":
        # Build the conversation the ledger will hold: two full turns.
        a1 = chat([{"role": "user", "content": q1}])
        m2 = [{"role": "user", "content": q1},
              {"role": "assistant", "content": a1},
              {"role": "user", "content": FOLLOW}]
        a2 = chat(m2)
        if case == "regenerate":
            # the client re-sends the conversation WITHOUT the last reply
            diverged = list(m2)
        elif case == "edit":
            diverged = [{"role": "user", "content": q1},
                        {"role": "assistant", "content": a1},
                        {"role": "user", "content": EDITED}]
        else:
            diverged = [{"role": "user", "content": q1},
                        {"role": "assistant", "content": a1},
                        {"role": "user", "content": BRANCHED}]
        json.dump(diverged, open(path, "w"))
        before = marks()
        text = chat(diverged)
        open(os.path.join(out, case + ".warm.txt"), "w").write(text)
        time.sleep(1)
        resets = [l.strip() for l in ledger_lines(before) if "ledger=reset" in l]
        open(os.path.join(out, case + ".reset"), "w").write(
            (resets[-1][-110:] if resets else "") + ("\n" if resets else ""))
        print(f"  {case}: warm reply {len(text.encode())} B "
              f"{'(' + resets[-1].split('[ledger]')[-1].strip() + ')' if resets else '-- NO ledger=reset'}")
    else:
        diverged = json.load(open(path))
        text = chat(diverged)
        open(os.path.join(out, case + ".cold.txt"), "w").write(text)
        print(f"  {case}: cold reply {len(text.encode())} B")
PY
