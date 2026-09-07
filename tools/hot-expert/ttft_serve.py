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
import argparse, json, math, os, re, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error

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
def shaders_dir(exe):
    """The compiled shaders live in the repo's c/shaders, not beside the binary:
    a pristine copy in ~/bench ran WITHOUT Vulkan (p0self, 2026-09-06) and every
    number it produced was a CPU-only number. Refuse rather than measure that."""
    for d in (os.path.normpath(os.path.join(HERE, "..", "..", "c", "shaders")),
              os.path.join(os.path.dirname(os.path.abspath(exe)), "shaders")):
        if os.path.isfile(os.path.join(d, "qmatmul.comp")):
            return d
    sys.exit("REFUSED: no shaders directory found (c/shaders) -- the engine would run without Vulkan")


def engine_env(exe):
    """The serving env (~/start_glm53.sh) for anything not already exported, so a
    bare invocation measures the configuration the gateway actually runs: 8
    physical cores, three devices, the 1695 tier caps (CLAUDE.md), KDA on the
    GPU, and a COPY of the histogram -- COLI_USAGE_PATH is rewritten at exit
    with the run's own counts and must never point at the canonical file."""
    d = {"OMP_NUM_THREADS": "8", "OMP_PLACES": "cores", "OMP_PROC_BIND": "close",
         "COLI_VULKAN": "1", "COLI_VK_DEV2": "auto", "COLI_VK_DEV3": "auto",
         "COLI_VK_EXPERTS2": "1695", "COLI_VK_EXPERTS3": "1695",
         "COLI_VK_SHADERS": shaders_dir(exe),
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


# ------------------------------------------------------- engine stderr (P7)
# The engine writes REUSE / CKPT lines to stderr, and openai_server.Engine lets
# the child INHERIT fd 2. A checkpoint test whose whole verdict is "how many
# tokens did the engine reuse" cannot read that off the console, so engine mode
# points the child's fd 2 at a file (dup2 around the spawn, restored right
# after) and parses it. HTTP mode reads the gateway's own log for the same
# lines -- that is where they land in service.
REUSE_RE = re.compile(r"REUSE (\d+) (\d+) (\d+)")
CKPT_RE = re.compile(r"^(?:.*\s)?(CKPT .*)$")


def scan_engine_log(path, start):
    """(reuse list, ckpt lines, new offset) for whatever the engine wrote since `start`."""
    if not path or not os.path.exists(path):
        return [], [], start
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read()
        end = f.tell()
    text = chunk.decode("utf-8", "replace")
    reuse = [(int(a), int(b), int(c)) for a, b, c in REUSE_RE.findall(text)]
    ckpt = [m.group(1).strip() for m in (CKPT_RE.match(l) for l in text.splitlines()) if m]
    return reuse, ckpt, end


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
        self.args = args
        self.tools = load_tools(args.tools)
        self.kv_slots = max(1, args.kv_slots)
        # P7: pin_context_blocks lives in the gateway; in engine mode the
        # harness is the gateway, so --pin-block applies it here, in render().
        self.pin = False
        self.log_path = args.engine_log or os.path.join(
            tempfile.gettempdir(), f"ttft_engine_{os.getpid()}.log")
        open(self.log_path, "w").close()
        self.log_at = 0
        self.spawn(args.gen)

    def spawn(self, max_tokens, extra_env=None):
        """(Re)start the engine with its stderr in self.log_path.

        openai_server.Engine lets the child inherit fd 2, so the only way to
        read the engine's own REUSE/CKPT lines is to point fd 2 at a file for
        the duration of the spawn and put the console's back afterwards."""
        t0 = time.time()
        for key, value in (extra_env or {}).items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        logf = open(self.log_path, "ab", buffering=0)
        saved = os.dup(2)
        try:
            os.dup2(logf.fileno(), 2)
            self.eng = self.rt.Engine(self.args.exe, self.args.snap, cap=self.args.cap,
                                      max_tokens=max_tokens, kv_slots=self.kv_slots)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            logf.close()
        # The engine loads the model lazily on the first SUBMIT in some builds
        # and eagerly in others; either way the first request pays it. A
        # throwaway 4-token request makes every measured number a warm-process
        # number, which is the serving regime.
        #
        # WITHOUT the tool list (P7, 2026-09-07): rendering the warm-up through
        # self.render() put the whole 5 400-token tool block in front of "hi"
        # and cost an 18-minute prefill before the first measurement, and it
        # seeded the checkpoint planner's "previous fresh prompt" with the very
        # prefix under test. The warm-up is a throwaway; it must stay one.
        self._one(self.rt.render_chat_for_arch([{"role": "user", "content": "hi"}],
                                               enable_thinking=False), 1, 0)
        self.mark()
        print(f"[engine] up and warm in {time.time()-t0:.1f}s (pid {self.eng.process.pid})"
              + (f" env {extra_env}" if extra_env else ""), flush=True)

    def respawn(self, max_tokens=None, extra_env=None):
        self.close()
        time.sleep(2)
        self.spawn(max_tokens if max_tokens is not None else self.args.gen, extra_env)

    # ---- the engine's own stderr, the authority on reuse and checkpoints
    def mark(self):
        _, _, self.log_at = scan_engine_log(self.log_path, self.log_at)
        return self.log_at

    def since_mark(self):
        reuse, ckpt, self.log_at = scan_engine_log(self.log_path, self.log_at)
        return reuse, ckpt

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
            # P6b step 6: the engine's own expert-cache hit rate, from the same
            # DONE STAT line. The pool costs dev0 experts, so the gate has to
            # show that routing did not move; without this the claim rests on
            # the preload count alone, which is an allocation, not a hit.
            if isinstance(stats, dict):
                ev["hit_pct"] = stats.get("cache_hit_percent")
        except Exception as e:  # ClientCancelled is the expected outcome of --cancel
            ev["error"] = type(e).__name__
            ev["cancelled"] = "Cancel" in type(e).__name__
        ev["done"] = time.time()
        return ev

    def faults(self):
        """(major, minor) page faults of the engine process so far -- the
        upstream datapoint on PR #1324 (2026-09-06) is that below ~1:1
        model:RAM the kernel's page LRU and the engine's slot LRU fight and a
        'hit' re-faults inside the matmul; the delta per request says whether
        that is happening here."""
        try:
            f = open(f"/proc/{self.eng.process.pid}/stat").read().split()
            return int(f[11]), int(f[9])
        except Exception:
            return 0, 0

    def slot_for(self, messages):
        """The gateway's own routing: with --kv-slots > 1 a conversation is
        keyed by its system messages + first user message (openai_server.
        conversation_cache_slot), so its turns land on one slot and a side
        request (title generation) lands elsewhere."""
        return self.rt.conversation_cache_slot(messages, self.kv_slots) if self.kv_slots > 1 else 0

    def run(self, messages, gen, slot=None, cancel_after=None):
        # P7: the pin runs BEFORE the slot hash, exactly where the gateway puts
        # it (chat_completion, before render and before conversation_cache_slot).
        # Pinning inside render() instead left turn 2 hashing the UNPINNED
        # system message, which sent it to another slot and re-prefilled the
        # whole history -- reused 0/115 in the 2026-09-07 smoke run.
        if self.pin:
            messages = self.rt.pin_context_blocks(list(messages))
        if slot is None:
            slot = self.slot_for(messages)
        f0 = self.faults()
        self.mark()
        rid = self.eng.next_request_id
        ev = self._one(self.render(messages), gen, slot, cancel_after)
        f1 = self.faults()
        ev["majflt"], ev["minflt"] = f1[0] - f0[0], f1[1] - f0[1]
        ev["slot"] = slot
        ev["req_id"] = rid
        reuse, ckpt = self.since_mark()
        ev["reused"] = reuse[-1][1] if reuse else None
        ev["ckpt"] = ckpt
        for line in ckpt:
            print(f"    [engine] {line}", flush=True)
        return ev

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
        # The gateway's log carries the engine's REUSE/CKPT lines: in service
        # that is where they land, and the live check needs the same verdict
        # the engine-mode check gets.
        self.log_path = args.server_log if args.server_log and os.path.exists(args.server_log) else None
        self.log_at = os.path.getsize(self.log_path) if self.log_path else 0
        self.model = args.model_id
        if not self.model:
            # ask the gateway rather than guess (2026-09-06: 'glm53' vs the
            # served 'glm-5.3-flash' turned every request into a 404)
            req = urllib.request.Request(args.url.rstrip("/") + "/v1/models", headers=self.headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                self.model = json.load(r)["data"][0]["id"]
            print(f"[http] model id from /v1/models: {self.model}", flush=True)

    def run(self, messages, gen, slot=None, cancel_after=None):
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
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read(300).decode("utf-8", "replace")
            except Exception:
                pass
            ev["error"] = f"HTTP {e.code}: {body.strip()}"
        except Exception as e:
            ev["error"] = f"{type(e).__name__}: {e}"
        ev["done"] = time.time()
        # give the gateway a moment to flush its log line for this request
        if self.log_path:
            time.sleep(1.0)
            reuse, ckpt, self.log_at = scan_engine_log(self.log_path, self.log_at)
            ev["reused"] = reuse[-1][1] if reuse else None
            ev["ckpt"] = ckpt
            for line in ckpt:
                print(f"    [engine] {line}", flush=True)
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
    flt = f" majflt={ev['majflt']}" if ev.get("majflt") is not None else ""
    reused = f" reused={ev['reused']}" if ev.get("reused") is not None else ""
    return (f"{label:<22} prompt_tokens={pt!s:>6}{reused} accept={acc:6.2f}s "
            f"ttft={ttft:8.2f}s ({rate:>11}) gen={ev['ntok']:<4} decode={dec}{flt}{err}")


# ------------------------------------------------------------------ P7 modes
CKPT_LEN_RE = re.compile(r"CKPT (?:store|hit|disk (?:load|write)) prefix=(\d+)")
DEFAULT_PIN_SYSTEM = ("You are a careful assistant running on a small machine. "
                      "Answer briefly and never invent facts.")


def ckpt_prefix_len(events):
    """The prefix length the engine itself printed on a CKPT line, else None."""
    best = None
    for ev in events:
        for line in ev.get("ckpt") or ():
            m = CKPT_LEN_RE.search(line)
            if m:
                best = int(m.group(1))
    return best


def read_logits(path):
    import array
    a = array.array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


def compare_logits(pa, pb):
    a, b = read_logits(pa), read_logits(pb)
    if len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    cos = dot / (na * nb) if na and nb else 0.0
    mx = max(abs(x - y) for x, y in zip(a, b))
    ia = max(range(len(a)), key=a.__getitem__); ib = max(range(len(b)), key=b.__getitem__)
    return cos, mx, ia, ib


def prefix_ckpt_mode(args, drv, record):
    """Does a NEW conversation start warm because an OLD one left a checkpoint?

    Three fresh conversations that share the whole system+tools prefix S and
    differ only in the user turn, so slot reuse (which is per-conversation and
    forward-only) cannot fire for any of them and the ONLY thing that can make
    turn 2 fast is a prefix checkpoint. The third runs in a NEW engine process:
    that one is the disk-persistence check, which is what makes a gateway
    restart warm."""
    system = (open(args.system, encoding="utf-8").read() if args.system
              else DEFAULT_PIN_SYSTEM)
    users = ["In one sentence: what is this assistant for?",
             "In one sentence: what is the difference between prefill and decode?",
             "In one sentence: why can a recurrent state not be rewound?"]

    def conv(i):
        return [{"role": "system", "content": system},
                {"role": "user", "content": users[i]}]

    # A char-count estimate of len(S), printed next to the engine's own number
    # because neither is the whole truth: the estimate has no tokenizer, and
    # the engine's number only exists once something was captured.
    est_chars = len(system) + (len(json.dumps(load_tools(args.tools))) if args.tools else 0)
    est = int(est_chars / 3.6)

    gen2 = 128 if args.oracle else 48
    assert_resident(args, "ckpt-1")
    e1 = drv.run(conv(0), 48)
    record("ckpt-conv1", 0, e1, "conv 1 [S, A]")
    assert_resident(args, "ckpt-2")
    e2 = drv.run(conv(1), gen2)
    record("ckpt-conv2", 0, e2, "conv 2 [S, B]")

    on_dump = None
    if args.oracle and args.engine and args.logit_dir:
        src = os.path.join(args.logit_dir, "on", f"req_{e2['req_id']}.f32")
        on_dump = os.path.join(args.logit_dir, "conv2_on.f32")
        if os.path.exists(src):
            with open(src, "rb") as a, open(on_dump, "wb") as b:
                b.write(a.read())
        else:
            on_dump = None

    e3 = None
    if args.engine:
        print("--- restarting the engine (disk persistence)", flush=True)
        drv.respawn(48)
        assert_resident(args, "ckpt-3")
        e3 = drv.run(conv(2), 48)
        record("ckpt-conv3", 0, e3, "conv 3 [S, C] after restart")

    lens = ckpt_prefix_len([e for e in (e1, e2, e3) if e])
    ref = lens if lens else est
    t1 = (e1["first"] - e1["submit"]) if e1["first"] else None
    t2 = (e2["first"] - e2["submit"]) if e2["first"] else None
    r1, r2 = e1.get("reused"), e2.get("reused")
    r3 = e3.get("reused") if e3 else None
    ratio = (t1 / t2) if t1 and t2 else 0.0
    # A checkpoint that already exists ON DISK (written by an earlier gate, then
    # read by the gateway) makes conversation 1 warm too -- which is the whole
    # point of persisting it, and it also makes t1/t2 a warm/warm ratio that
    # says nothing. When conversation 1 was itself a restore, the speed claim is
    # carried by conversation 1's own ttft and the reuse counts are the verdict.
    warm_start = r1 is not None and r1 >= ref - 32
    print(f"prefix-ckpt: len(S) engine={lens} estimate={est} (using {ref}), "
          f"turn 1 REUSE {r1}/{e1['prompt_tokens']}, turn 2 REUSE {r2}/{e2['prompt_tokens']}, "
          f"t1/t2 = {ratio:.2f}x, after restart REUSE {r3}", flush=True)
    if warm_start:
        print("prefix-ckpt: conversation 1 ALSO restored a checkpoint (disk "
              f"persistence: {t1:.2f}s for {e1['prompt_tokens']} tokens); the "
              "t1/t2 ratio is warm/warm and is not the speed criterion here",
              flush=True)
    ok = (r2 is not None and r2 >= ref - 32
          and (warm_start or (t1 and t2 and t2 < 0.25 * t1)))
    if args.engine:
        ok = ok and (r3 is not None and r3 >= ref - 32)
    print(f"prefix-ckpt: {'PASS' if ok else 'FAIL'}", flush=True)

    if args.oracle:
        if not args.engine:
            print("prefix-ckpt oracle: SKIPPED (engine mode only)", flush=True)
            return ok
        print("--- oracle: the same request on an engine with GLM53_PREFIX_CKPT=0", flush=True)
        off_dir = os.path.join(args.logit_dir, "off") if args.logit_dir else None
        drv.respawn(128, {"GLM53_PREFIX_CKPT": "0",
                          "GLM53_LOGIT_DUMP": off_dir})
        assert_resident(args, "oracle")
        e4 = drv.run(conv(1), 128)
        record("ckpt-oracle", 0, e4, "conv 2 [S, B] ckpt OFF")
        ton = "".join(e2.get("text", []))
        toff = "".join(e4.get("text", []))
        same = ton == toff
        print(f"oracle text: {'IDENTICAL' if same else 'DIFFERS'} "
              f"({len(ton)} vs {len(toff)} bytes over {e2['ntok']}/{e4['ntok']} tokens)", flush=True)
        if not same:
            print(f"  ON : {ton[:300]!r}\n  OFF: {toff[:300]!r}", flush=True)
        lok = False
        if on_dump and off_dir:
            other = os.path.join(off_dir, f"req_{e4['req_id']}.f32")
            cmp = compare_logits(on_dump, other) if os.path.exists(other) else None
            if cmp:
                cos, mx, ia, ib = cmp
                lok = cos >= 1 - 1e-4 and ia == ib
                print(f"oracle logits: cosine={cos:.7f} max_abs={mx:.4g} "
                      f"argmax {ia} vs {ib} {'OK' if ia == ib else 'DIFFERS'}", flush=True)
            else:
                print(f"oracle logits: MISSING ({on_dump} / {other})", flush=True)
        else:
            print("oracle logits: MISSING (no --logit-dir)", flush=True)
        ok = ok and same and lok
        print(f"prefix-ckpt --oracle: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def pin_block_mode(args, drv, record):
    """Does a re-ranked context block still cost a re-prefill?

    Turn 2 carries the same memory block with its items in another order --
    exactly what Open WebUI produced on 2026-09-07 with four memories. With the
    pin, turn 2's prompt is byte-identical up to the new user turn and the slot
    reuses the whole history; without it, everything re-prefills."""
    system = (open(args.system, encoding="utf-8").read().strip() if args.system
              else DEFAULT_PIN_SYSTEM)
    block_x = ("<memory_context>\n[Relevant Context]\n- the user runs a small "
               "machine called rome\n- the user prefers short answers\n- the user "
               "works on an inference engine\n</memory_context>")
    block_y = ("<memory_context>\n[Relevant Context]\n- the user prefers short "
               "answers\n- the user works on an inference engine\n- the user runs a "
               "small machine called rome\n</memory_context>")
    a_msg = "In one sentence: what is a KV cache?"
    b_msg = "Thanks. In one sentence: which day comes after Tuesday?"
    if args.engine:
        os.environ["COLI_PREFIX_PIN"] = "1"
        drv.pin = True
    assert_resident(args, "pin-1")
    turn1 = [{"role": "system", "content": system + "\n\n" + block_x},
             {"role": "user", "content": a_msg}]
    e1 = drv.run(list(turn1), 24)
    record("pin-turn1", 0, e1, "pin turn 1 [S+X, A]")
    reply = "".join(e1.get("text", [])).strip() or "I cannot tell."
    turn2 = [{"role": "system", "content": system + "\n\n" + block_y},
             {"role": "user", "content": a_msg},
             {"role": "assistant", "content": reply},
             {"role": "user", "content": b_msg}]
    assert_resident(args, "pin-2")
    e2 = drv.run(list(turn2), 24)
    record("pin-turn2", 0, e2, "pin turn 2 [S+Y, A, r, B]")
    need = (e1["prompt_tokens"] or 0) + e1["ntok"] - 1
    t1 = (e1["first"] - e1["submit"]) if e1["first"] else None
    t2 = (e2["first"] - e2["submit"]) if e2["first"] else None
    r1, r2 = e1.get("reused"), e2.get("reused")
    fresh1 = (e1["prompt_tokens"] or 0) - (r1 or 0)
    fresh2 = (e2["prompt_tokens"] or 0) - (r2 or 0)
    # What the pin claims is a REUSE count: turn 2 must prefill only its new
    # tokens instead of the whole history. The ttft ratio is a proxy for that
    # and it is only meaningful when turn 1 was a COLD prefill. Through the live
    # gateway turn 1 is itself restored from the prefix checkpoint on disk
    # (measured 2026-09-07: 911 of 965 reused, 8.76 s instead of ~127 s), so the
    # ratio compares warm with warm and cannot reach 0.25 however well the pin
    # works. When turn 1 was warm the reuse bound alone decides; the ratio is
    # printed either way.
    cold1 = not r1
    ok = r2 is not None and r2 >= need
    if cold1:
        ok = ok and bool(t1 and t2 and t2 < 0.25 * t1)
    print(f"pin-block: turn 1 REUSE {r1}/{e1['prompt_tokens']} ({fresh1} prefilled), "
          f"turn 2 REUSE {r2}/{e2['prompt_tokens']} ({fresh2} prefilled, need reuse >= {need}), "
          f"ttft {t1:.2f}s -> {t2:.2f}s = {(t2/t1 if t1 and t2 else 0):.3f}x", flush=True)
    if not cold1:
        print("pin-block: turn 1 was itself a checkpoint restore, so the ttft "
              "ratio is warm/warm; the reuse count is the criterion here", flush=True)
    print(f"pin-block: {'PASS' if ok else 'FAIL'}", flush=True)
    if args.engine:
        drv.pin = False
    return ok


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
    ap.add_argument("--side-request", action="store_true",
                    help="multiturn: interleave an Open WebUI-style title-generation request between turn 1 and 2")
    ap.add_argument("--system", help="text file used as a system message on every multiturn prompt (a stable prefix)")
    ap.add_argument("--kv-slots", type=int, default=1, help="engine KV slots; >1 routes conversations like the gateway does")
    ap.add_argument("--cancel", type=float, metavar="SECONDS", help="CANCEL check after this many seconds")
    ap.add_argument("--min-resident", type=float, default=96.0,
                    help="refuse below this; 100%% is unreachable with an engine up (see assert_resident)")
    ap.add_argument("--warm", action="store_true", help="re-warm the shards if below --min-resident")
    ap.add_argument("--allow-other-engines", action="store_true")
    ap.add_argument("--api-key")
    ap.add_argument("--model-id", default=None, help="HTTP mode: served model id (default: the first entry of /v1/models)")
    ap.add_argument("--json", help="append one JSON record per measurement to this file")
    ap.add_argument("--tag", default="")
    # ---- P7
    ap.add_argument("--prefix-ckpt", action="store_true",
                    help="P7: a NEW conversation sharing the system+tools prefix must start warm "
                         "(and stay warm across an engine restart)")
    ap.add_argument("--oracle", action="store_true",
                    help="--prefix-ckpt: re-serve the same request on an engine with "
                         "GLM53_PREFIX_CKPT=0 and compare text and first-token logits")
    ap.add_argument("--pin-block", action="store_true",
                    help="P7: a re-ranked <memory_context> block must not cost a re-prefill "
                         "(COLI_PREFIX_PIN)")
    ap.add_argument("--engine-log", help="engine mode: file for the engine's stderr "
                                         "(REUSE/CKPT lines); default a temp file")
    ap.add_argument("--server-log", default=os.path.expanduser("~/glm53_server.log"),
                    help="HTTP mode: gateway log to read REUSE/CKPT lines from")
    ap.add_argument("--logit-dir", help="--oracle: directory for GLM53_LOGIT_DUMP vectors")
    args = ap.parse_args()
    args.exe = args.engine
    if args.oracle and not args.prefix_ckpt:
        sys.exit("--oracle only means something with --prefix-ckpt")
    if args.oracle and args.engine:
        if not args.logit_dir:
            args.logit_dir = os.path.join(tempfile.gettempdir(), f"p7_logits_{os.getpid()}")
        for sub in ("on", "off"):
            os.makedirs(os.path.join(args.logit_dir, sub), exist_ok=True)
        # the ON side dumps from the very first spawn, so this must be set
        # before EngineDriver builds the engine
        os.environ["GLM53_LOGIT_DUMP"] = os.path.join(args.logit_dir, "on")

    sizes = [int(s) for s in args.sizes.split(",") if s]
    others = other_engines()
    if args.engine and others and not args.allow_other_engines:
        sys.exit(f"REFUSED: another glm53 is running (pids {others}); one engine at a time. "
                 f"Stop the gateway (or pass --allow-other-engines and own the consequences).")
    assert_resident(args, "start")

    drv = EngineDriver(args) if args.engine else HttpDriver(args)
    records = []
    verdicts = []

    def record(kind, size, ev, label):
        print(fmt(ev, label), flush=True)
        rec = {"tag": args.tag, "mode": "engine" if args.engine else "http", "kind": kind,
               "target": size, "prompt_tokens": ev["prompt_tokens"],
               "accept_s": (ev["accept"] - ev["submit"]) if ev["accept"] else None,
               "ttft_s": (ev["first"] - ev["submit"]) if ev["first"] else None,
               "gen": ev["ntok"], "cancelled": ev["cancelled"], "error": ev["error"],
               "decode_tps": ((ev["ntok"] - 1) / (ev["done"] - ev["first"])) if ev["first"] and ev["ntok"] > 1 else None,
               "majflt": ev.get("majflt"), "minflt": ev.get("minflt"),
               "reused": ev.get("reused"), "slot": ev.get("slot"),
               "hit_pct": ev.get("hit_pct"),
               "ckpt": ev.get("ckpt"),
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
            if args.system:
                msgs = [{"role": "system", "content": open(args.system, encoding="utf-8").read()}] + msgs
            a = drv.run(msgs, 48)
            record("turn1", base, a, "turn 1 [A]")
            reply = "".join(a.get("text", [])).strip() or "I cannot tell."
            if args.side_request:
                # what Open WebUI does after every reply: a separate short
                # request (title / tags / follow-ups) on a different prompt
                side = [{"role": "user", "content": "Create a concise, 3-5 word title for this chat, "
                         "no quotes: 'User asked what machine some notes are about.'"}]
                sd = drv.run(side, 12)
                record("side", 0, sd, "side request (title)")
            follow = msgs + [{"role": "assistant", "content": reply},
                             {"role": "user", "content": "Thanks. In one sentence: which day comes after Tuesday?"}]
            b = drv.run(follow, args.gen)
            record("turn2", base, b, "turn 2 [A, reply, B]")
            c = drv.run(msgs, args.gen)
            record("regen", base, c, "regenerate [A] again")
            if a["first"] and b["first"]:
                ta, tb = a["first"] - a["submit"], b["first"] - b["submit"]
                new = (b["prompt_tokens"] or 0) - (a["prompt_tokens"] or 0) - a["ntok"]
                print(f"multiturn: ttft(turn2)/ttft(turn1) = {tb/ta:.2f}; turn 2 added ~{new} new tokens "
                      f"after the reply -> {'PREFIX REUSED' if tb < 0.35 * ta else 'NO REUSE (turn 2 re-prefilled the history)'}; "
                      f"the engine's own verdict is the REUSE line above (reused-token count)", flush=True)
        if args.prefix_ckpt:
            verdicts.append(("prefix-ckpt", prefix_ckpt_mode(args, drv, record)))
        if args.pin_block:
            verdicts.append(("pin-block", pin_block_mode(args, drv, record)))
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
    # A named check that fails must fail the process: p7_gate.sh reads the exit
    # code, and a harness that always exits 0 is how a gate stops gating.
    for name, ok in verdicts:
        print(f"VERDICT {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if all(ok for _, ok in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
