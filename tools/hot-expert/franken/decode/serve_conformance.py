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
import subprocess
import sys
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(here, "franken_decode_cpu"))
    ap.add_argument("--gguf", default=os.environ.get("FRANKEN_GGUF",
                    "/home/ronald/models/Qwen3.8-Flash-Next/UD-IQ4_XS/"
                    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"))
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    env = dict(os.environ)
    env.update({
        "FRANKEN_GGUF": args.gguf,
        "FRANKEN_LAYERS": args.layers,
        "FRANKEN_CHUNK": str(args.chunk),
        "FRANKEN_CTX": str(args.ctx),
        "FRANKEN_THREADS": str(args.threads),
        "KV_SLOTS": "1",
        # The gateway always sets these two; the engine must not care.
        "SERVE_BATCH": "1",
        "NGEN": "4096",
        # Second guard: nothing in this process may see a card (see the
        # docstring). The first is the binary itself.
        "HIP_VISIBLE_DEVICES": "",
    })
    env.pop("SERVE", None)          # --serve-test is the switch here

    failures = []

    def log(msg):
        print(msg, flush=True)

    def check(name, ok, detail=""):
        print("%-46s %s %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)
        if not ok:
            failures.append(name)

    log("== launching %s --serve-test (cap argv, as the gateway does) ==" % args.engine)
    # openai_server.py launches the child as [executable, str(cap)] with
    # unbuffered pipes (bufsize=0) -- so does this.
    proc = subprocess.Popen([args.engine, "1", "--serve-test"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            bufsize=0, env=env)
    wire = Wire(proc, log)
    try:
        t0 = time.time()
        stat = wire.wait_ready()
        log("== READY after %.1fs ==" % (time.time() - t0))
        check("boot: READY + STAT with >= 5 fields", len(stat) >= 5 and stat[0] == "STAT")

        # ---- 1. a prompt that does not fit the context ---------------------
        wire.submit(1, 0, "hello", 10000)
        t = collect(wire, 1)
        check("context overflow -> ERROR CONTEXT_EXCEEDED",
              t.error == "CONTEXT_EXCEEDED", str(t.frames))

        # ---- 2. an empty prompt -------------------------------------------
        wire.submit(2, 0, "", 8)
        t = collect(wire, 2)
        check("empty prompt -> ERROR EMPTY_PROMPT", t.error == "EMPTY_PROMPT", str(t.frames))

        # ---- 3. a CANCEL for nothing in flight ----------------------------
        wire.cancel(99)
        verb, fid, rest = wire.frame()
        check("stray CANCEL -> ERROR <id> NOT_FOUND",
              verb == "ERROR" and fid == "99" and rest[0] == "NOT_FOUND")

        # ---- 4. one full round trip ---------------------------------------
        p1 = ("The capital of France is Paris, and the capital of Italy is Rome.\n"
              "Here is a list of three more capitals:\n")
        wire.submit(3, 0, p1, 4)
        a = collect(wire, 3)
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

        # ---- 5. a continuation: the ledger's own prediction ----------------
        # ledger_expect_reuse (openai_server.py:2842-2857) = the previous
        # turn's prompt_tokens + completion_tokens, minus one if that turn hit
        # its budget (the last generated token was never fed back). The engine
        # must report EXACTLY that or accept_live.sh check 2b fails on MISMATCH.
        expect = a.prompt_tokens + a.completion_tokens - a.limited
        p2 = p1 + a.text.decode("utf-8", "replace")
        wire.submit(4, 0, p2, 2)
        b = collect(wire, 4)
        check("turn B: reused == ledger_expect_reuse",
              b.reused == expect, "reused=%d expected=%d" % (b.reused, expect))
        check("turn B: hit%% follows reused/prompt",
              abs(float(b.stat[2]) - 100.0 * b.reused / b.prompt_tokens) < 0.1,
              "hit=%s" % b.stat[2])

        # ---- 6. a rollback: the same prompt again --------------------------
        # The live state now sits PAST this prompt's end, so reuse has to come
        # from a snapshot: the last chunk boundary at or below prompt_tokens-1
        # (a turn must run at least one token to have logits to sample from).
        expect_roll = ((a.prompt_tokens - 1) // args.chunk) * args.chunk
        wire.submit(5, 0, p1, 2)
        c = collect(wire, 5)
        check("turn C: rollback to the snapshot below the LCP",
              c.reused == expect_roll and c.prompt_tokens == a.prompt_tokens,
              "reused=%d expected=%d" % (c.reused, expect_roll))

        # ---- 7. CANCEL in flight -------------------------------------------
        # DONE (with the partial counts) and only THEN ERROR CANCELLED: the
        # gateway pops the pending entry on ERROR, so the other order loses the
        # turn's accounting (GATEWAY-PROTOCOL.md section 3).
        wire.submit(6, 0, p1 + "one two three four\n", 64)
        d = collect(wire, 6, cancel_after=1)
        check("turn D: cancelled turn ends DONE then ERROR CANCELLED",
              d.frames[-2:] == ["DONE", "ERROR"] and d.error == "CANCELLED",
              str(d.frames))
        check("turn D: DONE carries the partial count",
              d.stat is not None and d.completion_tokens == d.n_data,
              "emitted=%s data=%d" % (d.stat[0] if d.stat else "-", d.n_data))

        # ---- 8. the engine still serves after a cancellation ---------------
        wire.submit(7, 0, p1, 1)
        e = collect(wire, 7)
        check("turn E: the next request is served normally",
              e.stat is not None and e.completion_tokens == 1 and e.error is None,
              str(e.frames))
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=60)

    print("\n%d check(s) failed%s" % (len(failures),
          (": " + ", ".join(failures)) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
