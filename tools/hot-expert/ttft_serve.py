#!/usr/bin/env python3
"""P0 — serve-path time-to-first-token harness for glm53 on rome.

PREFILL-ROADMAP-2026-09.md item P0. Every prefill item on that track is gated
on the numbers this prints, and nothing else: a prefill change validated on the
`--greedy` CLI path is not validated (the G16 CANCEL fix passed a CLI oracle and
wedged the server).

Two ways to reach the engine, both the real path:

  --engine EXE     spawn the engine through openai_server.Engine -- the very
                   class the gateway uses, the same SUBMIT/DATA/DONE pipe
                   protocol, the same env (SERVE=1 SNAP=... SERVE_BATCH=1).
                   This is the mode prefill_gate.sh uses to compare binaries.
  --url URL        talk HTTP to a LIVE gateway (the user's path end to end,
                   Open WebUI's wire format). Needs the API key.

What it measures, per prompt size, `--repeat` times:
  accept   time from SUBMIT to the engine's ACCEPT frame (tokenisation)
  ttft     time from SUBMIT to the first DATA frame = the prefill
  decode   tok/s over the few tokens generated afterwards (sanity only)
and two protocol checks:
  --multiturn   submit A, then A+B in the same cache slot: if the prefix was
                reused, ttft(A+B) is a small fraction of ttft(A).
  --cancel      submit a long prompt, CANCEL after --cancel seconds, and time
                how long the engine takes to confirm and to serve the next
                request. This is the G16 gate.

Discipline baked in (the 2026-09-06 lessons, see the roadmap):
  * residency of the model's shards is asserted before EVERY run (fincore);
    below --min-resident the run is refused unless --warm re-warms first;
  * in --engine mode no other engine may be running (pgrep) -- one engine,
    one in-flight request, or the number is somebody else's;
  * every number is printed with the actual prompt_tokens the engine
    reported, never the size we asked for.

Prompts are deterministic: slices of a fixed text file (default: the
measurement record in this directory), so before/after runs see identical
bytes. --tools FILE renders a real tool list (Open WebUI's 34 builtin tools as
dumped from its request) into the chat template -- the ~6 000-token prompt
that made "hi" a 30-minute request.
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEXT = os.path.join(HERE, "ROME-3x7900XTX-2026-09-04.md")
DEFAULT_SNAP = os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")
DEFAULT_EXE = os.path.expanduser("~/src/colibri/c/glm53")


# ---------------------------------------------------------------- residency
def residency(snap):
    """(percent resident, shards short of 100%) for the model's safetensors."""
    out = subprocess.run(["fincore", "--bytes", "--output", "FILE,SIZE,RES"]
                         + sorted(os.path.join(snap, f) for f in os.listdir(snap)
                                  if f.endswith(".safetensors")),
                         capture_output=True, text=True, check=True).stdout
    page, tot, res, short, n = 4096, 0, 0, 0, 0
    for line in out.splitlines()[1:]:
        f = line.split()
        if len(f) < 3:
            continue
        size, r = int(f[-2]), int(f[-1])
        want = -(-size // page) * page
        n += 1; tot += want; res += r
        if r < want:
            short += 1
    return 100.0 * res / tot, short, n


def warm(snap):
    for f in sorted(os.listdir(snap)):
        if f.endswith(".safetensors"):
            with open(os.path.join(snap, f), "rb") as fh:
                while fh.read(1 << 24):
                    pass


def assert_resident(args, label):
    pct, short, n = residency(args.snap)
    # A running engine holds tens of GB of anon memory next to a 181 GiB model
    # on a 247 GB box: a fresh process evicts ~3% and the gateway's engine sits
    # at ~91.6%. Warm up to three times (each pass recovers what the previous
    # allocation burst evicted), then report what the serving regime actually
    # gives -- the number is printed before EVERY request so both sides of a
    # gate are compared at the same residency.
    for attempt in range(3):
        if pct >= args.min_resident or not args.warm:
            break
        t0 = time.time()
        warm(args.snap)
        pct, short, n = residency(args.snap)
        print(f"[resid {label}] re-warm {attempt+1} in {time.time()-t0:.0f}s -> {pct:.4f}%", flush=True)
    print(f"[resid {label}] shards={n} resident={pct:.4f}% short={short}", flush=True)
    if pct < args.min_resident:
        sys.exit(f"REFUSED: model {pct:.2f}% resident < {args.min_resident}% "
                 f"(re-run with --warm, or warm it yourself and verify with fincore)")


def other_engines():
    r = subprocess.run(["pgrep", "-x", "glm53"], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]


# ---------------------------------------------------------------- prompts
def filler_tokens_guess(text, target):
    # ~3.6 chars/token on this record's English+markdown (calibrated on rome
    # 2026-09-06 through the ACCEPT frame); the engine's own count is printed.
    return text[: int(target * 3.6)]


def build_messages(args, size, suffix=None):
    text = open(args.text, encoding="utf-8").read()
    if size <= 40:
        user = "What is your name, and in one sentence, what can you do?"
    else:
        user = ("Read the following notes and answer in one sentence: what machine "
                "are they about?\n\n" + filler_tokens_guess(text, size - 30))
    if suffix:
        user += "\n\n" + suffix
    return [{"role": "user", "content": user}]


def load_tools(path):
    if not path:
        return None
    t = json.load(open(path))
    # Open WebUI sends OpenAI-style {"type":"function","function":{...}} entries;
    # accept a bare list of functions too.
    return t if isinstance(t, list) else t.get("tools")


# ---------------------------------------------------------------- engine mode
def engine_env(exe):
    """The serving env (~/start_glm53.sh) for anything not already exported, so a
    bare invocation measures the configuration the gateway actually runs: 8
    physical cores, three devices, the 1695 tier caps (CLAUDE.md), KDA on the
    GPU, and a COPY of the histogram -- COLI_USAGE_PATH is rewritten at exit
    with the run's own counts and must never point at the canonical file."""
    d = {"OMP_NUM_THREADS": "8", "OMP_PLACES": "cores", "OMP_PROC_BIND": "close",
         "COLI_VULKAN": "1", "COLI_VK_DEV2": "auto", "COLI_VK_DEV3": "auto",
         "COLI_VK_EXPERTS2": "1695", "COLI_VK_EXPERTS3": "1695",
         "COLI_VK_SHADERS": os.path.join(os.path.dirname(exe), "shaders"),
         "COLI_KDA_GPU": "2",
         # the engine prints "REUSE <id> <reused> <prompt_tokens>" on stderr per
         # turn -- the authoritative answer to "did the prefix get reused?"
         "GLM53_VERBOSE": "1"}
    for k, v in d.items():
        os.environ.setdefault(k, v)
    if "COLI_USAGE_PATH" not in os.environ:
        src = os.path.expanduser("~/.glm53_explain.bin")
        if os.path.exists(src):
            dst = f"/tmp/ttft_hist_{os.getpid()}.bin"
            with open(src, "rb") as a, open(dst, "wb") as b:
                b.write(a.read())
            os.environ["COLI_USAGE_PATH"] = dst


class EngineDriver:
    def __init__(self, args):
        engine_env(args.exe)
        # openai_server lives in the repo's c/, not necessarily next to the
        # binary (a pristine copy in ~/bench is the normal gate input)
        sys.path.insert(0, os.path.dirname(args.exe))
        sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "c")))
        import openai_server as rt
        self.rt = rt
        res = rt.resolve_model(args.snap)
        rt.ARCH = res.descriptor.id
        self.tools = load_tools(args.tools)
        t0 = time.time()
        self.eng = rt.Engine(args.exe, args.snap, cap=args.cap, max_tokens=args.gen)
        # The engine loads the model lazily on the first SUBMIT in some builds
        # and eagerly in others; either way the first request pays it. A
        # throwaway 4-token request makes every measured number a warm-process
        # number, which is the serving regime.
        self._one(self.render([{"role": "user", "content": "hi"}]), 1, 0)
        print(f"[engine] up and warm in {time.time()-t0:.1f}s (pid {self.eng.process.pid})",
              flush=True)

    def render(self, messages):
        return self.rt.render_chat_for_arch(messages, enable_thinking=False, tools=self.tools)

    def _one(self, prompt, gen, slot, cancel_after=None):
        ev = {"submit": time.time(), "accept": None, "first": None, "done": None,
              "ntok": 0, "prompt_tokens": None, "cancelled": False, "error": None}
        stop_flag = {"v": False}

        ev["text"] = []

        def on_text(t):
            if ev["first"] is None:
                ev["first"] = time.time()
            ev["ntok"] += 1
            ev["text"].append(t)

        def on_accept(info):
            ev["accept"] = time.time()
            ev["prompt_tokens"] = info.get("prompt_tokens")

        def cancelled():
            return cancel_after is not None and time.time() - ev["submit"] >= cancel_after

        try:
            stats = self.eng.generate(prompt, gen, 0.0, 1.0, on_text, cache_slot=slot,
                                      cancelled=cancelled, on_accept=on_accept)
            # an engine without an ACCEPT frame (glm53 2026-09) reports the
            # prompt length only in the DONE STAT fields generate() returns
            if ev["prompt_tokens"] is None and isinstance(stats, dict):
                ev["prompt_tokens"] = stats.get("prompt_tokens")
        except Exception as e:  # ClientCancelled is the expected outcome of --cancel
            ev["error"] = type(e).__name__
            ev["cancelled"] = "Cancel" in type(e).__name__
        ev["done"] = time.time()
        return ev

    def run(self, messages, gen, slot=0, cancel_after=None):
        return self._one(self.render(messages), gen, slot, cancel_after)

    def close(self):
        try:
            self.eng.close()
        except Exception:
            pass
        try:
            self.eng.process.wait(timeout=30)
        except Exception:
            self.eng.process.kill()


# ---------------------------------------------------------------- http mode
class HttpDriver:
    def __init__(self, args):
        self.url = args.url.rstrip("/") + "/v1/chat/completions"
        key = args.api_key
        if not key and os.path.exists(os.path.expanduser("~/.colibri_api_key")):
            key = open(os.path.expanduser("~/.colibri_api_key")).read().strip()
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = "Bearer " + key
        self.tools = load_tools(args.tools)
        self.model = args.model_id

    def run(self, messages, gen, slot=0, cancel_after=None):
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": gen, "temperature": 0, "stream_options": {"include_usage": True}}
        if self.tools:
            body["tools"] = self.tools
        ev = {"submit": time.time(), "accept": None, "first": None, "done": None,
              "ntok": 0, "prompt_tokens": None, "cancelled": False, "error": None, "text": []}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    j = json.loads(data)
                    if j.get("usage"):
                        ev["prompt_tokens"] = j["usage"].get("prompt_tokens")
                    for ch in j.get("choices", []):
                        d = ch.get("delta", {})
                        if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                            if ev["first"] is None:
                                ev["first"] = time.time()
                            ev["ntok"] += 1
                            if d.get("content"):
                                ev["text"].append(d["content"])
                    if cancel_after is not None and time.time() - ev["submit"] >= cancel_after:
                        ev["cancelled"] = True
                        break   # closing the socket is how a browser cancels
        except Exception as e:
            ev["error"] = type(e).__name__
        ev["done"] = time.time()
        return ev

    def close(self):
        pass


# ---------------------------------------------------------------- report
def fmt(ev, label):
    # no ACCEPT frame -> "accept" was set on the first DATA and says nothing
    acc = (ev["accept"] - ev["submit"]) if ev["accept"] and ev["accept"] != ev["first"] else float("nan")
    ttft = (ev["first"] - ev["submit"]) if ev["first"] else float("nan")
    dec = ""
    if ev["first"] and ev["ntok"] > 1:
        dec = f"{(ev['ntok']-1)/(ev['done']-ev['first']):.2f} tok/s"
    pt = ev["prompt_tokens"]
    rate = f"{pt/ttft:.2f} tok/s" if pt and ev["first"] else "?"
    err = f" ERROR={ev['error']}" if ev["error"] and not ev["cancelled"] else ""
    return (f"{label:<22} prompt_tokens={pt!s:>6} accept={acc:6.2f}s "
            f"ttft={ttft:8.2f}s ({rate:>11}) gen={ev['ntok']:<4} decode={dec}{err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--engine", metavar="EXE", help="spawn this glm53 through openai_server.Engine")
    m.add_argument("--url", help="live gateway base URL, e.g. http://127.0.0.1:8081")
    ap.add_argument("--snap", default=DEFAULT_SNAP)
    ap.add_argument("--cap", type=int, default=512, help="engine cache/layer cap (argv[1])")
    ap.add_argument("--sizes", default="30,300,1000", help="target prompt tokens, comma-separated")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--gen", type=int, default=8, help="tokens to generate after first (decode sanity)")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="deterministic filler text")
    ap.add_argument("--tools", help="JSON list of OpenAI-style tools to attach (realistic prompt)")
    ap.add_argument("--multiturn", action="store_true", help="prefix-reuse check: A then A+B, same slot")
    ap.add_argument("--cancel", type=float, metavar="SECONDS", help="CANCEL check after this many seconds")
    ap.add_argument("--min-resident", type=float, default=96.0,
                    help="refuse below this; 100%% is unreachable with an engine up (see assert_resident)")
    ap.add_argument("--warm", action="store_true", help="re-warm the shards if below --min-resident")
    ap.add_argument("--allow-other-engines", action="store_true")
    ap.add_argument("--api-key")
    ap.add_argument("--model-id", default="glm53")
    ap.add_argument("--json", help="append one JSON record per measurement to this file")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.exe = args.engine

    sizes = [int(s) for s in args.sizes.split(",") if s]
    others = other_engines()
    if args.engine and others and not args.allow_other_engines:
        sys.exit(f"REFUSED: another glm53 is running (pids {others}); one engine at a time. "
                 f"Stop the gateway (or pass --allow-other-engines and own the consequences).")
    assert_resident(args, "start")

    drv = EngineDriver(args) if args.engine else HttpDriver(args)
    records = []

    def record(kind, size, ev, label):
        print(fmt(ev, label), flush=True)
        rec = {"tag": args.tag, "mode": "engine" if args.engine else "http", "kind": kind,
               "target": size, "prompt_tokens": ev["prompt_tokens"],
               "accept_s": (ev["accept"] - ev["submit"]) if ev["accept"] else None,
               "ttft_s": (ev["first"] - ev["submit"]) if ev["first"] else None,
               "gen": ev["ntok"], "cancelled": ev["cancelled"], "error": ev["error"],
               "tools": bool(args.tools), "t": time.time()}
        records.append(rec)
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(rec) + "\n")

    try:
        for size in sizes:
            for rep in range(args.repeat):
                assert_resident(args, f"{size}/{rep+1}")
                ev = drv.run(build_messages(args, size), args.gen)
                record("ttft", size, ev, f"size {size} run {rep+1}")
        if args.multiturn:
            # The real chat flow, the way Open WebUI drives it: turn 1 is
            # [user A]; turn 2 resends the history, [user A, assistant reply,
            # user B]. The engine can resume a slot only from the exact position
            # it stepped to (prompt + generated tokens; KDA state cannot rewind,
            # glm53.c serve_turn), so turn 2 reuses iff the template re-renders
            # the reply into exactly the generated tokens. Turn 3 resends turn 1's
            # prompt unchanged ("regenerate"): by design that cannot reuse.
            base = sizes[1] if len(sizes) > 1 else sizes[0]
            assert_resident(args, "multiturn")
            msgs = build_messages(args, base)
            a = drv.run(msgs, 48, slot=0)
            record("turn1", base, a, "turn 1 [A]")
            reply = "".join(a.get("text", [])).strip() or "I cannot tell."
            follow = msgs + [{"role": "assistant", "content": reply},
                             {"role": "user", "content": "Thanks. In one sentence: which day comes after Tuesday?"}]
            b = drv.run(follow, args.gen, slot=0)
            record("turn2", base, b, "turn 2 [A, reply, B]")
            c = drv.run(msgs, args.gen, slot=0)
            record("regen", base, c, "regenerate [A] again")
            if a["first"] and b["first"]:
                ta, tb = a["first"] - a["submit"], b["first"] - b["submit"]
                new = (b["prompt_tokens"] or 0) - (a["prompt_tokens"] or 0) - a["ntok"]
                print(f"multiturn: ttft(turn2)/ttft(turn1) = {tb/ta:.2f}; turn 2 added ~{new} new tokens "
                      f"after the reply -> {'PREFIX REUSED' if tb < 0.35 * ta else 'NO REUSE (turn 2 re-prefilled the history)'}; "
                      f"the engine's own verdict is the REUSE line above (reused-token count)", flush=True)
        if args.cancel is not None:
            big = max(sizes)
            assert_resident(args, "cancel")
            ev = drv.run(build_messages(args, big), 256, cancel_after=args.cancel)
            confirm = ev["done"] - ev["submit"]
            print(f"cancel: sent at {args.cancel:.1f}s, engine confirmed after {confirm:.1f}s "
                  f"(cancelled={ev['cancelled']} error={ev['error']})", flush=True)
            nxt = drv.run(build_messages(args, 30), args.gen)
            record("after-cancel", 30, nxt, "next request after cancel")
            ok = ev["cancelled"] and confirm < args.cancel + 30 and nxt["first"] is not None
            print(f"cancel: {'PASS' if ok else 'FAIL'} -- a cancelled request must free the engine "
                  f"within seconds, not run to max_tokens", flush=True)
            records.append({"tag": args.tag, "kind": "cancel", "confirm_s": confirm,
                            "pass": ok, "t": time.time()})
    finally:
        drv.close()
        if args.engine:
            left = other_engines()
            if left:
                print(f"WARNING: glm53 still running after close: {left}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
