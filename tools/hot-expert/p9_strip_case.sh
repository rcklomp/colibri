#!/bin/bash
# p9_strip_case.sh -- client behaviour (c): a client that trims whitespace.
#
# SYNTHETIC, on purpose, and it is the case that makes the point of P9: it is a
# behaviour no client on this box has yet shown, it costs a full re-prefill when
# it appears, and neither existing pin covers it.
#
#   * P7's pin holds the <memory_context> block steady, and this is not that;
#   * P8's reply pin matches an assistant turn on its STRIPPED visible text, so
#     it does cover a trimmed REPLY -- but not a trimmed QUESTION. The user turn
#     is rendered from the client's bytes, so one trailing space added or removed
#     there and the prompt diverges at that position; the KDA state cannot
#     rewind past it and the whole conversation re-prefills.
#
# So the case sends a first user message that ENDS in whitespace and a follow-up
# request that hands both the question and the reply back trimmed -- exactly what
# a front end that calls `.strip()` on what it stores would do. Driven straight
# at the gateway (no Open WebUI, no memory, no tools): the behaviour is a client
# behaviour and needs nothing else to be true.
#
#   p9_strip_case.sh [--seed N] [--max-tokens 64]
#
# PASS iff turn 2 REUSEs prompt_tokens(turn 1) + gen(turn 1), +/- 2. There is no
# time bound: the prompt here is small on purpose, so a re-prefill is cheap and
# the REUSE count is the only honest discriminator.
#
#   RESULT seed=.. p1=.. gen1=.. p2=.. reused2=.. t2=.. expect=.. verdict=PASS|FAIL
set -u
LOG="${GLM53_LOG:-$HOME/glm53_server.log}"; SEED=1; MAXTOK=64
URL=${COLI_URL:-http://127.0.0.1:8081}
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --max-tokens) MAXTOK="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
pgrep -f "openai_[s]erver.py" >/dev/null || { echo "RESULT error=no-gateway verdict=FAIL"; exit 2; }
N="$$-$(date +%s)-$SEED"

M0=$(grep -c "\[req\] " "$LOG")
python3 - "$N" "$MAXTOK" "$URL" <<'PY'
import json, os, sys, urllib.request
nonce, mt, url = sys.argv[1], int(sys.argv[2]), sys.argv[3]
key = open(os.path.expanduser("~/.colibri_api_key")).read().strip()
H = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
def chat(messages):
    req = urllib.request.Request(url + "/v1/chat/completions",
        data=json.dumps({"model": "glm-5.3-flash", "messages": messages,
                         "max_tokens": mt, "temperature": 0, "stream": False}).encode(),
        headers=H)
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
# The trailing whitespace is the fixture. A client that stores `content.strip()`
# gives back neither this nor the reply's own trailing bytes.
q1 = f"Answer with one word: what is the capital of Portugal? [{nonce}]   \n\n"
r = chat([{"role": "user", "content": q1}])
msg = r["choices"][0]["message"]
visible = msg.get("content") or ""
reasoning = msg.get("reasoning_content")
# What the trimming client sends back: the question WITHOUT its trailing bytes
# and the reply stripped. Note it keeps `reasoning_content` -- so this case is
# not P8's case wearing a different hat: even a client that hands the reasoning
# back still breaks the prefix on the user turn alone.
body = [{"role": "user", "content": q1.strip()},
        {"role": "assistant", "content": visible.strip(),
         **({"reasoning_content": reasoning} if isinstance(reasoning, str) else {})},
        {"role": "user", "content": f"And of Spain? One word. [{nonce}]"}]
chat(body)
print(f"  turn 1 visible={len(visible.encode())}B reasoning="
      f"{len((reasoning or '').encode())}B; turn 2 sent trimmed")
PY
rc=$?
[ "$rc" = 0 ] || { echo "RESULT error=dispatch verdict=FAIL"; exit 2; }
for _ in $(seq 1 360); do
  [ "$(grep -c "\[req\] " "$LOG")" -ge $((M0 + 2)) ] && break
  sleep 2
done
L1=$(grep "\[req\] " "$LOG" | tail -2 | head -1)
L2=$(grep "\[req\] " "$LOG" | tail -1)
RID2=$(echo "$L2" | sed -n 's/.*id=\([0-9]*\).*/\1/p')
R2=$(grep " REUSE $RID2 " "$LOG" | tail -1 | awk '{print $(NF-1)}')
echo "turn 1: $L1"
echo "turn 2: $L2  reused=$R2"
grep -E "\[ledger\] |\[reply-pin\] " "$LOG" | tail -3 | cut -c1-160
python3 - "$L1" "$L2" "${R2:-0}" "$SEED" <<'PY'
import re, sys
d1 = dict(re.findall(r"(\w+)=([\d.]+)", sys.argv[1]))
d2 = dict(re.findall(r"(\w+)=([\d.]+)", sys.argv[2]))
p1, g1 = int(d1["prompt_tokens"]), int(d1["gen"])
p2, t2 = int(d2["prompt_tokens"]), float(d2["ttft"])
reused = int(sys.argv[3]); expect = p1 + g1
ok = abs(reused - expect) <= 2
print(f"RESULT seed={sys.argv[4]} p1={p1} gen1={g1} t1={d1['ttft']} p2={p2} "
      f"reused2={reused} t2={t2:.2f} expect={expect} "
      f"verdict={'PASS' if ok else 'FAIL'}")
print(f"  bound: |reused - (p1+gen1)| <= 2 -> |{reused} - {expect}| = {abs(reused-expect)}")
sys.exit(0 if ok else 1)
PY
