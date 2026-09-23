#!/usr/bin/env python3
"""L0 step 4's conformance driver: speak c/openai_server.py's line protocol to
`franken_decode_cpu --serve-test` and check what comes back, frame by frame.

WHY NOT c/tests/test_openai_server.py: that suite's FakeProcess replaces the
ENGINE with a script of bytes and exercises the GATEWAY. What needs checking
here is the other direction -- that a real franken_decode process produces the
bytes GATEWAY-PROTOCOL.md specifies -- so this driver plays the gateway's half
by hand: it writes the exact SUBMIT/CANCEL frames openai_server.py writes
(Engine.generate, openai_server.py:3929-3948) and parses the replies the way
Engine._dispatch_stdout parses them (openai_server.py:3730-3849), including the
DATA terminator check and the positional DONE STAT fields.

It also reads the engine's STDERR, because two things the acceptance gates on
the rig depend on live there and nowhere else:

  * the `REUSE <id> <reused> <prompt_tokens>` line. `accept_live.sh` check 3
    and `owui_ui_turn.sh` both take `reused` from `grep " REUSE <id> "` +
    `awk '{print $(NF-1)}'` -- NOT from the DONE frame. The first served run
    failed both for exactly this reason: the number was right on the wire and
    the line the gates read did not exist.
  * the `[serve] req=... from=N chunks=N` line, which is how "the rollback
    re-prefilled only the tail" becomes a checked fact rather than a claim
    about a wall-clock number.

CPU ONLY, and that is enforced twice: the binary is franken_decode_cpu (which
links no HIP runtime -- `make ldd-check`), and HIP_VISIBLE_DEVICES is set to
empty in the child's environment, because the tokenizer's libllama is dlopened
and its libggml pulls libggml-hip into the address space even though nothing
ever calls into it.

Usage:
  python3 serve_conformance.py [--engine ./franken_decode_cpu]
                               [--gguf <shard.gguf>] [--layers 0-3] [--chunk 8]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

READY = b"\x01\x01READY\x01\x01\n"


class Wire:
    """The gateway's half of the pipe."""

    def __init__(self, proc, log):
        self.p = proc
        self.log = log

    # ---- reading (Engine._dispatch_stdout's rules) ------------------------
    def read_exact(self, n):
        out = b""
        while len(out) < n:
            chunk = self.p.stdout.read(n - len(out))
            if not chunk:
                raise EOFError("engine closed stdout")
            out += chunk
        return out

    def read_line(self):
        out = b""
        while True:
            c = self.p.stdout.read(1)
            if not c:
                raise EOFError("engine closed stdout")
            out += c
            if c == b"\n":
                return out

    def wait_ready(self, timeout=1800):
        """Discard everything up to and including the READY sentinel, then read
        the one STAT line -- openai_server.py:3448-3466, byte for byte."""
        t0 = time.time()
        seen = b""
        while not seen.endswith(READY):
            c = self.p.stdout.read(1)
            if not c:
                raise EOFError("engine died before READY")
            seen += c
            if time.time() - t0 > timeout:
                raise TimeoutError("no READY in %ds" % timeout)
        stat = self.read_line().decode().strip()
        self.log("<- %s" % stat)
        fields = stat.split()
        assert len(fields) >= 5 and fields[0] == "STAT", "invalid engine status: %r" % stat
        return fields

    def frame(self):
        """One reply frame, as (verb, id, payload_or_fields)."""
        line = self.read_line().decode("utf-8", "replace").rstrip("\n")
        fields = line.split()
        if not fields:
            return ("NONE", "", [])
        verb = fields[0]
        if verb == "DATA":
            n = int(fields[2])
            assert 0 <= n <= 65536, "DATA length out of bounds: %d" % n
            body = self.read_exact(n)
            term = self.read_exact(1)
            assert term == b"\n", "invalid engine DATA terminator: %r" % term
            self.log("<- DATA %s %d %r" % (fields[1], n, body))
            return ("DATA", fields[1], body)
        self.log("<- %s" % line)
        return (verb, fields[1] if len(fields) > 1 else "", fields[2:])

    # ---- writing (Engine.generate's frames) -------------------------------
    def submit(self, rid, slot, prompt, max_tokens, temp=0.0, top_p=1.0):
        payload = prompt.encode("utf-8")
        header = "SUBMIT %d %d %d %d %.8g %.8g\n" % (
            rid, slot, len(payload), max_tokens, temp, top_p)
        self.log("-> %s" % header.strip())
        self.p.stdin.write(header.encode() + payload + b"\n")
        self.p.stdin.flush()

    def cancel(self, rid):
        self.log("-> CANCEL %d" % rid)
        self.p.stdin.write(("CANCEL %d\n" % rid).encode())
        self.p.stdin.flush()


class Turn:
    """What one request produced, in the gateway's own terms."""

    def __init__(self):
        self.accept_prompt = None
        self.text = b""
        self.n_data = 0
        self.stat = None
        self.error = None
        self.frames = []

    @property
    def completion_tokens(self):
        return int(self.stat[0])

    @property
    def prompt_tokens(self):
        return int(self.stat[4])

    @property
    def limited(self):
        return int(self.stat[5])

    @property
    def reused(self):
        return int(self.stat[6])


def collect(wire, rid, cancel_after=None):
    """Read one request to its terminal frame. `cancel_after` sends a CANCEL
    once that many DATA frames have arrived -- the gateway's own behaviour when
    its client disconnects mid-stream."""
    t = Turn()
    while True:
        verb, fid, rest = wire.frame()
        t.frames.append(verb)
        if verb == "ACCEPT":
            assert fid == str(rid), "ACCEPT for %s, expected %s" % (fid, rid)
            t.accept_prompt = int(rest[0])
        elif verb == "DATA":
            assert fid == str(rid)
            t.text += rest
            t.n_data += 1
            if cancel_after is not None and t.n_data == cancel_after:
                wire.cancel(rid)
        elif verb == "DONE":
            assert fid == str(rid)
            assert rest[0] == "STAT", "DONE without STAT: %r" % rest
            t.stat = rest[1:]
            assert len(t.stat) >= 4, "DONE STAT needs >= 4 fields: %r" % t.stat
            if cancel_after is None:
                return t
        elif verb == "ERROR":
            assert fid == str(rid)
            t.error = rest[0] if rest else ""
            return t
        else:
            raise AssertionError("unexpected frame %r" % verb)


class Engine:
    """The child process, its wire, and its stderr."""

    def __init__(self, args, env_extra, log):
        self.log = log
        env = dict(os.environ)
        env.update({
            "FRANKEN_GGUF": args.gguf,
            "FRANKEN_LAYERS": args.layers,
            "FRANKEN_CHUNK": str(args.chunk),
            "FRANKEN_CTX": str(args.ctx),
            "FRANKEN_THREADS": str(args.threads),
            "KV_SLOTS": "1",
            # The schedule's FLOOR, pinned to the chunk so the expectations
            # below stay arithmetic. In service it is 512 and the template's
            # turn boundaries carry the precision (phase 3 covers those).
            "FRANKEN_SNAP_EVERY": str(args.chunk),
            # The gateway always sets these two; the engine must not care.
            "SERVE_BATCH": "1",
            "NGEN": "4096",
            # Second guard: nothing in this process may see a card (see the
            # module docstring). The first is the binary itself.
            "HIP_VISIBLE_DEVICES": "",
        })
        env.update(env_extra)
        env.pop("SERVE", None)          # --serve-test is the switch here
        self.err = tempfile.NamedTemporaryFile(prefix="franken_serve_err_", suffix=".log",
                                               delete=False)
        # openai_server.py launches the child as [executable, str(cap)] with
        # unbuffered pipes (bufsize=0) -- so does this.
        self.p = subprocess.Popen([args.engine, "1", "--serve-test"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.err, bufsize=0, env=env)
        self.wire = Wire(self.p, log)

    def stderr_text(self):
        self.err.flush()
        with open(self.err.name, "r", errors="replace") as fh:
            return fh.read()

    def reuse_line(self, rid):
        """Exactly what accept_live.sh:121 and owui_ui_turn.sh do:
        grep " REUSE <id> " | tail -1 | awk '{print $(NF-1)}'."""
        hit = None
        for line in self.stderr_text().splitlines():
            if (" REUSE %s " % rid) in line:
                hit = line
        if hit is None:
            return None
        return hit.split()[-2]

    def req_line(self, rid):
        """The engine's own accounting line for one request, as a dict."""
        hit = None
        for line in self.stderr_text().splitlines():
            if ("req=%s " % rid) in line:
                hit = line
        if hit is None:
            return {}
        return {k: v for k, v in re.findall(r"(\w+)=(-?[\d.]+)", hit)}

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=120)
        except Exception:
            self.p.kill()
        self.err.close()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(here, "franken_decode_cpu"))
    ap.add_argument("--gguf", default=os.environ.get("FRANKEN_GGUF",
                    "/home/ronald/models/Qwen3.8-Flash-Next/UD-IQ4_XS/"
                    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"))
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    failures = []

    def log(msg):
        print(msg, flush=True)

    def check(name, ok, detail=""):
        print("%-52s %s %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)
        if not ok:
            failures.append(name)

    # The engine tokenizes, so the driver never knows a prompt's token count in
    # advance -- every expectation below is written against numbers the engine
    # itself reported for an earlier turn, which is also how the gateway's own
    # ledger predicts `reused` (openai_server.py:2842-2857).
    SHARED = ("You are a careful assistant. Facts you must remember for this "
              "conversation: the capital of France is Paris, the capital of Italy "
              "is Rome, the capital of Japan is Tokyo, the capital of Peru is Lima, "
              "the capital of Kenya is Nairobi, the capital of Norway is Oslo, and "
              "the capital of Chile is Santiago. Answer only from that list, in one "
              "word, and never explain your answer.\n")

    def ceil_div(a, b):
        return (a + b - 1) // b

    # ---- phase 1: chunk-boundary checkpoints ------------------------------
    log("== phase 1: %s --serve-test, chunk=%d ==" % (args.engine, args.chunk))
    eng = Engine(args, {}, log)
    w = eng.wire
    turns = {}
    try:
        t0 = time.time()
        stat = w.wait_ready()
        log("== READY after %.1fs ==" % (time.time() - t0))
        check("boot: READY + STAT with >= 5 fields", len(stat) >= 5 and stat[0] == "STAT")

        # ---- 1. a prompt that does not fit the context ---------------------
        w.submit(1, 0, "hello", 100000)
        t = collect(w, 1)
        check("context overflow -> ERROR CONTEXT_EXCEEDED",
              t.error == "CONTEXT_EXCEEDED", str(t.frames))

        # ---- 2. an empty prompt -------------------------------------------
        w.submit(2, 0, "", 8)
        t = collect(w, 2)
        check("empty prompt -> ERROR EMPTY_PROMPT", t.error == "EMPTY_PROMPT", str(t.frames))

        # ---- 3. a CANCEL for nothing in flight ----------------------------
        w.cancel(99)
        verb, fid, rest = w.frame()
        check("stray CANCEL -> ERROR <id> NOT_FOUND",
              verb == "ERROR" and fid == "99" and rest[0] == "NOT_FOUND")

        # ---- 4. one full round trip ---------------------------------------
        w.submit(3, 0, SHARED + "Q: What is the capital of Japan?\nA:", 4)
        a = collect(w, 3); turns[3] = a
        check("turn A: ACCEPT carries prompt_tokens",
              a.accept_prompt is not None and a.accept_prompt == a.prompt_tokens,
              "accept=%s done=%s" % (a.accept_prompt, a.prompt_tokens))
        check("turn A: one DATA per emitted token, DONE last",
              a.n_data == a.completion_tokens and a.frames[-1] == "DONE",
              "data=%d emitted=%d" % (a.n_data, a.completion_tokens))
        check("turn A: fresh slot reports reused=0", a.reused == 0, "reused=%d" % a.reused)
        check("turn A: budget of 4 is honoured and flagged",
              a.completion_tokens == 4 and a.limited == 1,
              "emitted=%d limited=%d" % (a.completion_tokens, a.limited))
        ra = eng.req_line(3)
        check("turn A: a fresh prefill runs every chunk",
              int(ra.get("chunks", -1)) == ceil_div(a.prompt_tokens, args.chunk),
              "chunks=%s of %d" % (ra.get("chunks"), ceil_div(a.prompt_tokens, args.chunk)))

        # ---- 5. a continuation: the ledger's own prediction ----------------
        # ledger_expect_reuse (openai_server.py:2842-2857) = the previous
        # turn's prompt_tokens + completion_tokens, minus one if that turn hit
        # its budget (the last generated token was never fed back). The engine
        # must report EXACTLY that or accept_live.sh check 2b fails on MISMATCH.
        expect = a.prompt_tokens + a.completion_tokens - a.limited
        p2 = SHARED + "Q: What is the capital of Japan?\nA:" + a.text.decode("utf-8", "replace")
        w.submit(4, 0, p2, 2)
        b = collect(w, 4); turns[4] = b
        check("turn B: reused == ledger_expect_reuse",
              b.reused == expect, "reused=%d expected=%d" % (b.reused, expect))
        check("turn B: hit%% follows reused/prompt",
              abs(float(b.stat[2]) - 100.0 * b.reused / b.prompt_tokens) < 0.1,
              "hit=%s" % b.stat[2])
        rb = eng.req_line(4)
        check("turn B: only the tail is re-prefilled",
              int(rb.get("from", -1)) == b.reused and
              int(rb.get("chunks", -1)) == ceil_div(b.prompt_tokens - b.reused, args.chunk),
              "from=%s chunks=%s of %d" % (rb.get("from"), rb.get("chunks"),
                                           ceil_div(b.prompt_tokens - b.reused, args.chunk)))

        # ---- 6. a rollback: the same prompt again --------------------------
        # The live state now sits PAST this prompt's end, so reuse has to come
        # from a checkpoint: the last chunk boundary at or below prompt-1 (a
        # turn must run at least one token to have logits to sample from).
        pa = SHARED + "Q: What is the capital of Japan?\nA:"
        expect_roll = ((a.prompt_tokens - 1) // args.chunk) * args.chunk
        w.submit(5, 0, pa, 2)
        c = collect(w, 5); turns[5] = c
        check("turn C: rollback to the checkpoint below the LCP",
              c.reused == expect_roll and c.prompt_tokens == a.prompt_tokens,
              "reused=%d expected=%d" % (c.reused, expect_roll))
        rc = eng.req_line(5)
        check("turn C: only the tail is re-prefilled",
              int(rc.get("chunks", -1)) == ceil_div(c.prompt_tokens - c.reused, args.chunk),
              "chunks=%s of %d" % (rc.get("chunks"),
                                   ceil_div(c.prompt_tokens - c.reused, args.chunk)))

        # ---- 7. THE ONE THAT FAILED IN SERVICE -----------------------------
        # An unrelated short request lands on the same slot (with one KV slot
        # every conversation does) and re-prefills from its own zero, so the
        # cells the long prefix lived in are gone. A checkpoint that carried
        # only the recurrent state would now be worthless -- which is exactly
        # what the first served run reported: reused=0 on the next UI chat.
        w.submit(6, 0, "Say OK.", 2)
        x = collect(w, 6); turns[6] = x
        check("turn X: an unrelated short request reuses nothing",
              x.reused == 0, "reused=%d" % x.reused)
        w.submit(7, 0, pa, 2)
        d = collect(w, 7); turns[7] = d
        check("turn D: the shared prefix survives an interleaved request",
              d.reused == expect_roll,
              "reused=%d expected=%d (this is the 2026-09-22 failure)" % (d.reused, expect_roll))
        rd = eng.req_line(7)
        check("turn D: only the tail is re-prefilled after the rollback",
              int(rd.get("chunks", -1)) == ceil_div(d.prompt_tokens - d.reused, args.chunk),
              "chunks=%s of %d" % (rd.get("chunks"),
                                   ceil_div(d.prompt_tokens - d.reused, args.chunk)))

        # ---- 8. CANCEL in flight -------------------------------------------
        # DONE (with the partial counts) and only THEN ERROR CANCELLED: the
        # gateway pops the pending entry on ERROR, so the other order loses the
        # turn's accounting (GATEWAY-PROTOCOL.md section 3).
        w.submit(8, 0, pa + " one two three four\n", 64)
        e = collect(w, 8, cancel_after=1); turns[8] = e
        check("turn E: cancelled turn ends DONE then ERROR CANCELLED",
              e.frames[-2:] == ["DONE", "ERROR"] and e.error == "CANCELLED",
              str(e.frames))
        check("turn E: DONE carries the partial count",
              e.stat is not None and e.completion_tokens == e.n_data,
              "emitted=%s data=%d" % (e.stat[0] if e.stat else "-", e.n_data))

        # ---- 9. still serving ----------------------------------------------
        w.submit(9, 0, pa, 1)
        f = collect(w, 9); turns[9] = f
        check("turn F: the next request is served normally",
              f.stat is not None and f.completion_tokens == 1 and f.error is None,
              str(f.frames))

        # ---- 10. the fields and the line the GATES read --------------------
        bad_stat = [rid for rid, t in turns.items()
                    if t.stat is None or len(t.stat) != 7 or not t.stat[6].lstrip("-").isdigit()]
        check("every DONE STAT has 7 fields, reused an integer",
              not bad_stat, "offenders: %s" % bad_stat)
        bad_reuse = []
        for rid, t in turns.items():
            got = eng.reuse_line(rid)
            if got is None or not got.lstrip("-").isdigit() or int(got) != t.reused:
                bad_reuse.append((rid, got, t.reused))
        check("every request logs ' REUSE <id> <reused> <prompt>'",
              not bad_reuse, "offenders: %s" % bad_reuse)
    finally:
        eng.close()
        log("== engine stderr: %s ==" % eng.err.name)

    # ---- phase 2: NO chunk checkpoints, only the end-of-request one --------
    # "snapshots only at 256-token chunk boundaries mean nothing under 256 can
    # be reused" -- the end-of-request checkpoint is what answers that, and
    # this phase proves it by switching the chunk-boundary ones off entirely.
    log("\n== phase 2: FRANKEN_SNAP_EVERY=1000000 (end-of-request checkpoint only) ==")
    eng2 = Engine(args, {"FRANKEN_SNAP_EVERY": "1000000"}, log)
    w2 = eng2.wire
    try:
        w2.wait_ready()
        short = "Q: capital of Norway? A:"
        w2.submit(1, 0, short, 3)
        g = collect(w2, 1)
        expect2 = g.prompt_tokens + g.completion_tokens - g.limited
        w2.submit(2, 0, short + g.text.decode("utf-8", "replace") + "\nQ: and Chile? A:", 2)
        h = collect(w2, 2)
        check("phase 2: a follow-up far under one chunk reuses exactly",
              h.reused == expect2 and h.reused > 0,
              "prompt=%d reused=%d expected=%d" % (h.prompt_tokens, h.reused, expect2))
        r = eng2.req_line(2)
        check("phase 2: the follow-up re-prefills one chunk at most",
              int(r.get("chunks", -1)) == ceil_div(h.prompt_tokens - h.reused, args.chunk),
              "chunks=%s" % r.get("chunks"))
    finally:
        eng2.close()
        log("== engine stderr: %s ==" % eng2.err.name)

    # ---- phase 3: the template's turn boundaries are the schedule ---------
    # accept_live.sh check 2 wants reuse within 256 tokens of where two UI
    # chats diverge, and that point is not a multiple of anything -- it is the
    # start of the last user turn. With the interval far too coarse to help,
    # the boundary token is the only thing that can carry it.
    log("\n== phase 3: FRANKEN_SNAP_EVERY=1000000, boundary checkpoints only ==")
    eng3 = Engine(args, {"FRANKEN_SNAP_EVERY": "1000000"}, log)
    w3 = eng3.wire
    try:
        w3.wait_ready()
        # <|im_start|> is a real special token of this vocab, so the engine
        # tokenizes it as one and takes a checkpoint there.
        head = ("<|im_start|>system\n" + SHARED + "<|im_end|>\n")
        qa = head + "<|im_start|>user\nWhat is the capital of Kenya?<|im_end|>\n<|im_start|>assistant\n"
        qb = head + "<|im_start|>user\nWhat is the capital of Chile, and why?<|im_end|>\n<|im_start|>assistant\n"
        w3.submit(1, 0, qa, 2)
        a3 = collect(w3, 1)
        w3.submit(2, 0, qb, 2)
        b3 = collect(w3, 2)
        # The two prompts share everything up to the user turn's content, so a
        # checkpoint at the boundary token reuses all of it: well inside
        # accept_live check 2's `prompt_tokens - 256`.
        # accept_live check 2's own bar is prompt-256, which at this test's
        # scale would pass with no reuse at all; -32 (check 3's tolerance) is
        # the same statement made meaningful for a 100-token prompt.
        check("phase 3: a second chat reuses the shared head (check 2's shape)",
              b3.reused >= b3.prompt_tokens - 32 and b3.reused > 0,
              "prompt=%d reused=%d (bar %d)" % (b3.prompt_tokens, b3.reused,
                                                b3.prompt_tokens - 32))
        r3 = eng3.req_line(2)
        check("phase 3: only the diverging tail is re-prefilled",
              int(r3.get("chunks", -1)) == ceil_div(b3.prompt_tokens - b3.reused, args.chunk),
              "chunks=%s of %d" % (r3.get("chunks"),
                                   ceil_div(b3.prompt_tokens - b3.reused, args.chunk)))
    finally:
        eng3.close()
        log("== engine stderr: %s ==" % eng3.err.name)

    print("\n%d check(s) failed%s" % (len(failures),
          (": " + ", ".join(failures)) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
