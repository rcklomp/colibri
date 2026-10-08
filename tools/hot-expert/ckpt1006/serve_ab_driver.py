#!/usr/bin/env python3
# serve_ab_driver.py LOG OUT -- drive the live gateway (127.0.0.1:8081) with a fixed greedy request sequence and report, per request, what the engine's own
# `[serve-glm5] req=` line says (prompt, reused, prefill_s, emitted, decode_s, tok/s) and how many adapter windows (`adapt_all`) the engine logged since the last request.
# The sequence: W1 a long document (pos climbs to ~8.9k), then short FRESH chats (pos restarts near 0: the case the monotonic-position fix is about):
# W2 hash table, W3 bash script, W4 B-tree vs LSM, W5 = W2's prompt again (same text under greedy decoding below 2 051 tokens of depth, so a placement that has learned must do better).
# Stdlib only. Never prints the API key.
# D6 (2026-10-08): every row of OUT also carries t_start / t_end (epoch seconds, request sent / reply received; the decode ran in [t_end - decode_s, t_end]) so the clock
# sampler's rows (gpu_sampler.py arm) can be matched to the request; and with env SAB_MARK=FILE the driver writes the running request's tag into FILE (and "-" between
# requests), which the sampler copies into the `req` column of its rows. Unset / empty: nothing is written. The printed lines are unchanged.
import json, os, re, sys, time, urllib.request

LOG, OUT = sys.argv[1], sys.argv[2]
KEY = open(os.path.expanduser("~/.colibri_api_key")).read().strip()
DOC = open(os.path.expanduser("~/bench/franken/glm5/prose8400.txt")).read()
W2 = "Explain, step by step, how a hash table handles collisions, with a short Python example."
REQS = [("W1_long_doc", DOC + "\n\nSummarize the passage above in about 300 words.", 300),
        ("W2_hash", W2, 400),
        ("W3_bash", "Write a bash script that watches a directory and prints a line whenever a file in it changes. Explain each part.", 400),
        ("W4_btree", "Compare B-trees and LSM trees for a write-heavy workload and say when you would pick each.", 400),
        ("W5_hash_again", W2, 400)]
REQ_RE = re.compile(r"\[serve-(?:glm5|ds4)\] req=(\d+) slot=\d+ prompt=(\d+) reused=(\d+) .*?emitted=(\d+) .*?prefill_s=([\d.]+) .*?decode_s=([\d.]+) tok/s=([\d.]+)")

def log_text():
    with open(LOG, errors="replace") as f:
        return f.read()

def post(content, max_tokens):
    body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read())
    return time.time() - t0, j

MARK = os.environ.get("SAB_MARK", "")

def mark(tag):
    if not MARK:
        return
    try:
        with open(MARK + ".tmp", "w") as f:       # replace, so the sampler never reads a half-written name
            f.write(tag + "\n")
        os.replace(MARK + ".tmp", MARK)
    except OSError:
        pass

rows = []
seen_adapt = len(re.findall(r"adapt_all pos=", log_text()))
for tag, content, mt in REQS:
    n_before = len(REQ_RE.findall(log_text()))
    mark(tag); t_start = time.time()
    dt, j = post(content, mt)
    t_end = time.time(); mark("-")
    m = None
    for _ in range(40):                     # the engine's req line can trail the HTTP reply by a moment
        found = REQ_RE.findall(log_text())
        if len(found) > n_before:
            m = found[-1]; break
        time.sleep(0.5)
    txt = (j.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    adapt_now = len(re.findall(r"adapt_all pos=", log_text()))
    row = {"tag": tag, "http_s": round(dt, 2), "chars": len(txt), "adapt_windows": adapt_now - seen_adapt,
           "t_start": round(t_start, 2), "t_end": round(t_end, 2)}
    seen_adapt = adapt_now
    if m:
        row.update(req=int(m[0]), prompt=int(m[1]), reused=int(m[2]), emitted=int(m[3]), prefill_s=float(m[4]),
                   decode_s=float(m[5]), tok_s=float(m[6]))
    rows.append(row)
    print("%-14s prompt=%-5s reused=%-5s emitted=%-4s prefill_s=%-6s decode_s=%-6s tok/s=%-6s adapt_windows=%-3s chars=%d" % (
        tag, row.get("prompt", "?"), row.get("reused", "?"), row.get("emitted", "?"), row.get("prefill_s", "?"),
        row.get("decode_s", "?"), row.get("tok_s", "?"), row["adapt_windows"], row["chars"]), flush=True)
with open(OUT, "w") as f:
    json.dump(rows, f, indent=1)
