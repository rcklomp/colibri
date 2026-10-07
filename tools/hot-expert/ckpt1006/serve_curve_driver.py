#!/usr/bin/env python3
# serve_curve_driver.py LOG OUT [N] -- a PERSISTENT-SESSION curve on the live gateway (127.0.0.1:8081): N different short fresh chats, greedy, 300 tokens each, one after the other in
# one serving process (the adapter's average persists across them, as it does for the owner). Per chat: what the engine's own `[serve-glm5] req=` line says (decode tok/s, emitted)
# and the number of adapter windows (`adapt_all`) logged during it. Greedy below 2 051 tokens of depth is deterministic: every arm must emit the same text (compare `chars`).
# Stdlib only. Never prints the API key.
import json, os, re, sys, time, urllib.request

LOG, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
KEY = open(os.path.expanduser("~/.colibri_api_key")).read().strip()
PROMPTS = [
 "Explain, step by step, how a hash table handles collisions, with a short Python example.",
 "Write a bash script that watches a directory and prints a line whenever a file in it changes. Explain each part.",
 "Compare B-trees and LSM trees for a write-heavy workload and say when you would pick each.",
 "A train leaves at 9:40 and travels 180 km at 72 km/h, then waits 25 minutes and travels another 90 km at 60 km/h. When does it arrive? Show the arithmetic.",
 "Summarize the causes of the 1929 stock market crash in about 250 words for a high school student.",
 "Write a regular expression that matches ISO 8601 dates (YYYY-MM-DD), explain it, and list three strings it wrongly accepts.",
 "Give me a SQL query that returns the three customers with the highest total order value last month, and explain how you would index the tables.",
 "Translate into French and then explain the grammar of: 'If I had known that you were coming, I would have baked a cake.'",
 "This Python function is slow on large lists, explain why and fix it: def dedupe(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out",
 "Write a short poem about a lighthouse keeper who collects broken clocks, then explain the imagery you used.",
 "Explain how TCP congestion control reacts to packet loss, covering slow start, congestion avoidance and fast retransmit.",
 "Describe a good weekly meal prep plan for a vegetarian who trains four times a week, with a shopping list.",
 "Explain the difference between a mutex and a semaphore, with an example where using the wrong one causes a bug.",
 "Outline the main arguments for and against a four-day work week, and say what evidence would change your mind.",
]
REQ_RE = re.compile(r"\[serve-(?:glm5|ds4)\] req=(\d+) slot=\d+ prompt=(\d+) reused=(\d+) .*?emitted=(\d+) .*?prefill_s=([\d.]+) .*?decode_s=([\d.]+) tok/s=([\d.]+)")
def log_text():
    with open(LOG, errors="replace") as f: return f.read()
def post(content, max_tokens):
    body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.loads(r.read())
rows = []; seen_adapt = len(re.findall(r"adapt_all pos=", log_text()))
for i in range(N):
    content = PROMPTS[i % len(PROMPTS)]
    n_before = len(REQ_RE.findall(log_text()))
    j = post(content, 300)
    m = None
    for _ in range(40):
        found = REQ_RE.findall(log_text())
        if len(found) > n_before: m = found[-1]; break
        time.sleep(0.5)
    txt = (j.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    adapt_now = len(re.findall(r"adapt_all pos=", log_text()))
    row = {"chat": i + 1, "chars": len(txt), "adapt_windows": adapt_now - seen_adapt}; seen_adapt = adapt_now
    if m: row.update(prompt=int(m[1]), emitted=int(m[3]), decode_s=float(m[5]), tok_s=float(m[6]))
    rows.append(row)
    print("chat %-2d prompt=%-4s emitted=%-4s decode_s=%-6s tok/s=%-6s adapt_windows=%-3s chars=%d" % (
        i + 1, row.get("prompt", "?"), row.get("emitted", "?"), row.get("decode_s", "?"), row.get("tok_s", "?"), row["adapt_windows"], row["chars"]), flush=True)
with open(OUT, "w") as f: json.dump(rows, f, indent=1)
