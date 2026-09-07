#!/usr/bin/env python3
"""Minimal terminal chat against the Colibri gateway -- the client that sends
only what you typed.

Written 2026-09-07 after a day of Open WebUI fighting the engine: a title
request fired at the same instant as the chat, 24-34 builtin tools attached by
a capability flag, max_tokens overridden to 25100, memory context injected
into the system prompt, and a spinner that hides whether the engine is
prefilling or thinking. This sends: your messages, max_tokens (default 256),
enable_thinking (default off), nothing else -- and prints per turn what the
gateway measured (time to first token, tokens, and the engine's REUSE line is
in the server log).

    chat.py [--url http://rome.local:8081] [--api-key-file ~/.colibri_api_key]
            [--max-tokens 256] [--think] [--system FILE]

Commands inside the chat: /reset (new conversation), /quit.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("COLI_URL", "http://127.0.0.1:8081"))
    ap.add_argument("--api-key-file", default=os.path.expanduser("~/.colibri_api_key"))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--think", action="store_true", help="let the model reason before answering (slow at 3.5 tok/s)")
    ap.add_argument("--system", help="text file used as the system message")
    args = ap.parse_args()

    headers = {"Content-Type": "application/json"}
    key = os.environ.get("COLI_API_KEY")
    if not key and os.path.exists(args.api_key_file):
        key = open(args.api_key_file).read().strip()
    if key:
        headers["Authorization"] = "Bearer " + key
    base = args.url.rstrip("/")
    req = urllib.request.Request(base + "/v1/models", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    print(f"[chat] {base} model={model} max_tokens={args.max_tokens} thinking={'on' if args.think else 'off'}; /reset, /quit")

    def fresh():
        m = []
        if args.system:
            m.append({"role": "system", "content": open(args.system, encoding="utf-8").read()})
        return m

    messages = fresh()
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            messages = fresh(); print("[chat] new conversation"); continue
        messages.append({"role": "user", "content": line})
        body = {"model": model, "messages": messages, "stream": True,
                "max_tokens": args.max_tokens, "temperature": 0, "enable_thinking": bool(args.think)}
        t0 = time.time(); first = None; out = []; reasoning = 0
        try:
            rq = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                        headers=headers, method="POST")
            with urllib.request.urlopen(rq, timeout=7200) as r:
                sys.stdout.write("glm> "); sys.stdout.flush()
                for raw in r:
                    s = raw.decode("utf-8", "replace").strip()
                    if not s.startswith("data:"):
                        continue
                    d = s[5:].strip()
                    if d == "[DONE]":
                        break
                    j = json.loads(d)
                    for ch in j.get("choices", []):
                        delta = ch.get("delta", {})
                        if delta.get("reasoning_content"):
                            reasoning += 1
                            if first is None:
                                first = time.time(); sys.stdout.write("(thinking…) "); sys.stdout.flush()
                        if delta.get("content"):
                            if first is None:
                                first = time.time()
                            sys.stdout.write(delta["content"]); sys.stdout.flush(); out.append(delta["content"])
        except urllib.error.HTTPError as e:
            print(f"\n[chat] HTTP {e.code}: {e.read(300).decode('utf-8', 'replace')}")
            messages.pop(); continue
        except KeyboardInterrupt:
            print("\n[chat] interrupted (note: the engine finishes the turn anyway -- CANCEL is not honoured yet)")
            messages.pop(); continue
        total = time.time() - t0
        ttft = (first - t0) if first else total
        print(f"\n[turn] ttft={ttft:.1f}s total={total:.1f}s reasoning_deltas={reasoning} history={len(messages)+1} msgs")
        messages.append({"role": "assistant", "content": "".join(out)})


if __name__ == "__main__":
    main()
