#!/usr/bin/env python3
"""Inference performance vs CONTEXT LENGTH for glm53 on rome.

Neither existing harness answers this question, and MEASURING.md's rule is that
using one track's tool for the other track's question produces a number that is
internally consistent and meaningless. `rome_bench.sh` is decode tok/s on 30-43
token prompts; `prefill_snapshot.sh` is TTFT at 27/384/1230/3462. Both stop four
orders of magnitude short of "how does this engine behave at 64Ki tokens".

WHY A LADDER AND NOT A SWEEP OF COLD PROMPTS
--------------------------------------------
Prefill on this box runs at ~112 ms/token (P11 gate, 2026-09-10, three sizes).
A single cold 64Ki prompt is therefore a ~2 hour operation, and a sweep of
independent cold prompts at 2/4/8/16/32/64Ki costs the sum of all of them --
about 4 hours for one pass, 8 for the two repeats CLAUDE.md requires. That is
not a better measurement, only a more expensive one: every point re-pays for the
tokens the point below it already ground.

This drives ONE growing conversation instead, exactly the shape Open WebUI
sends. Turn k appends a fresh chunk of text to the transcript, so the engine
reuses the whole KV cache it already holds (the REUSE line proves it per turn)
and prefills only the new chunk -- at depth. That yields, for one pass over
65 536 tokens (~2 h):

  * the LOCAL prefill rate at each depth (ms per new token at depth d), which
    is the derivative the cold-sweep never shows -- it only ever reports the
    average from 0 to N;
  * decode tok/s at each depth, the number that says what generation costs
    once the KV cache is deep;
  * the cold TTFT at any depth, by integration: the cumulative sum of the step
    prefill times IS what one cold prompt of that length would pay, and is
    reported as such (marked derived);
  * the incremental TTFT -- what the owner actually waits for on turn k of a
    long chat, which is the number the whole P-track optimises.

THE CONTRACT THIS RESTS ON, AND WHY IT IS CHECKED EVERY TURN
------------------------------------------------------------
Reuse is all-or-nothing in the engine (glm53.c serve_one: `shared` is taken only
when the new sequence matches EVERY cached position). If the assistant turn does
not render back to the exact tokens the engine generated, `shared` is 0, the
turn re-prefills the entire history, and the ladder silently degenerates into
the expensive sweep it exists to avoid -- while still printing plausible
numbers. That is the P8 failure mode (a reply reassembled as
`<think>...</think>{content}` cost 54 s where 2.7 s was expected) and it is why
the raw generated text is fed back verbatim, never stripped or reassembled.

So every turn asserts the engine's own REUSE line: reuse must cover the whole
previous context, or the run ABORTS. A ladder that stopped reusing is not a
degraded measurement, it is a different one.

HTTP MODE (H1)
--------------
--url reuses ttft_serve.HttpDriver instead of spawning an engine, so the same
ladder can be pointed at any OpenAI-compatible server (the live gateway for a
read-only check, or a second server entirely -- H0/H2's whole point). Three
things do NOT carry over from engine mode, and are handled explicitly rather
than silently degraded:

  * the reuse-based ABORT is an engine-mode-only contract (it is `glm53`'s own
    serve_one guarantee, read off `glm53`'s stderr through openai_server's
    inherited fd 2). A server we did not write makes no such promise, so HTTP
    mode never aborts on it -- it prints a loud "REUSE UNVERIFIED (http)" note
    with whatever the gateway's log happened to report, and the JSON row's own
    `reused` field is left None so nothing downstream mistakes a log-scrape for
    an asserted guarantee. --cold-sweep (below) is how reuse gets an answer in
    HTTP mode: a genuinely fresh, disjoint prompt at the same depth.
  * P8's reply pin (`REPLY_PIN_FIELD`) is an artifact of glm53's own chat
    template, applied here because engine mode IS the gateway. A server this
    harness did not write may reject an unknown message field outright, so the
    pin is engine-mode only; HTTP mode feeds back exactly what it received.
  * residency is checked over every file under --snap regardless of extension,
    not only `.safetensors` -- a future arm (hipFire's MQ4R, qwen36's
    int4-gs64 container) is not a Colibri shard set at all.
"""
import argparse, json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ttft_serve as T                                          # noqa: E402

# The default steps: fine at shallow depth where the curve bends, even at deep
# where each point costs ~15 minutes. Cumulative user-text tokens land on
# 1/2/4/8/16/24/32/40/48/56/64 Ki; template and replies push the engine's own
# count above 65 536, which is the point -- the ask was "64 KiB tokens or it is
# not interesting".
DEFAULT_STEPS = "1024,1024,2048,4096,8192,8192,8192,8192,8192,8192,8192"
QUESTION = ("\n\nContinue summarising the notes above in a few sentences, "
            "without repeating what you already said.")

# Driver registry, kept as a dict rather than an if/else chain so a second HTTP
# driver (Ollama-style NDJSON, if H0 finds one) is a new entry and a --driver
# value, not a rewrite of main(). Only registers what exists today.
HTTP_DRIVERS = {"http": T.HttpDriver}

# The plan's own fit (FRANKEN-ENGINE-PLAN-2026-09-15.md §1), five points
# 643..13628, within ~1.5%: ms/token(depth d) = 113.9 + 0.01258*d, and its
# integral 0..N is the projected cost of ONE COLD prompt of N tokens. Used only
# to print the arithmetic behind refusing --cold-sweep in engine mode -- never
# to substitute for a measurement.
def cold_prefill_s(n):
    return (113.9 * n + 0.00629 * n * n) / 1000.0


def chunk_text(text, start_char, want_tokens):
    """A deterministic slice of the corpus, ~3.6 chars/token (ttft_serve's
    calibration on this same file). The engine's own count is what gets
    reported; this only has to land close enough to make the ladder even."""
    n = int(want_tokens * 3.6)
    if start_char + n > len(text):
        sys.exit(f"REFUSED: corpus is {len(text)} chars, the ladder needs "
                 f"{start_char + n}. Pass a bigger --text.")
    return text[start_char:start_char + n], start_char + n


# --------------------------------------------------------- generic residency
# ttft_serve.residency()/warm() hardcode ".safetensors" -- right for every
# glm53/qwen38 snap today, wrong for a directory this harness has never been
# pointed at (hipFire's MQ4R, qwen36's int4-gs64 container, neither of which
# is a Colibri shard). Same fincore-based check, same re-warm loop, generalised
# to every regular file under --snap.
def list_snap_files(snap):
    out = []
    for dirpath, _, names in os.walk(snap):
        for name in names:
            p = os.path.join(dirpath, name)
            if os.path.isfile(p):
                out.append(p)
    return sorted(out)


def snap_residency(snap):
    files = list_snap_files(snap)
    if not files:
        sys.exit(f"REFUSED: no files under --snap {snap}")
    out = subprocess.run(["fincore", "--bytes", "--output", "FILE,SIZE,RES"] + files,
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
    return 100.0 * res / max(1, tot), short, n


def snap_warm(snap):
    for f in list_snap_files(snap):
        with open(f, "rb") as fh:
            while fh.read(1 << 24):
                pass


def assert_snap_resident(args, label):
    pct, short, n = snap_residency(args.snap)
    for attempt in range(3):
        if pct >= args.min_resident or not args.warm:
            break
        t0 = time.time()
        snap_warm(args.snap)
        pct, short, n = snap_residency(args.snap)
        print(f"[resid {label}] re-warm {attempt+1} in {time.time()-t0:.0f}s -> {pct:.4f}%", flush=True)
    print(f"[resid {label}] files={n} resident={pct:.4f}% short={short}", flush=True)
    if pct < args.min_resident:
        sys.exit(f"REFUSED: {pct:.2f}% resident < {args.min_resident}% "
                 f"(re-run with --warm, or warm it yourself and verify with fincore)")


def refuse_cold_sweep_in_engine_mode(sizes):
    parts_s = [cold_prefill_s(n) for n in sizes]
    sys.exit(
        "REFUSED: --cold-sweep is HTTP mode only. Engine mode already reuses the "
        "whole context turn over turn (that is what the ladder measures); running "
        "these sizes as fresh COLD prompts here would pay the full cold prefill for "
        "each, on GLM's own fit (FRANKEN-ENGINE-PLAN-2026-09-15.md \xa71): "
        + "+".join(str(n) for n in sizes) + " tok = "
        + "+".join(f"{s:.0f}" for s in parts_s) + f" s = {sum(parts_s)/60:.0f} min per "
        "pass. Run it with --url against a server whose reuse is unverified -- that "
        "is the point of the sweep.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--engine", dest="exe", help="spawn this glm53 through openai_server.Engine")
    g.add_argument("--url", help="an OpenAI-compatible server, e.g. http://127.0.0.1:8081 "
                                 "(H1: the live gateway, or any second engine H0 stands up)")
    ap.add_argument("--driver", choices=sorted(HTTP_DRIVERS), default="http",
                    help="--url only: which driver class serves it. Registry is "
                         "HTTP_DRIVERS above; add an entry there for an Ollama-style "
                         "NDJSON server rather than branching main() on it")
    ap.add_argument("--snap", default=T.DEFAULT_SNAP)
    ap.add_argument("--text", default=T.DEFAULT_TEXT)
    ap.add_argument("--steps", default=DEFAULT_STEPS)
    ap.add_argument("--gen", type=int, default=32,
                    help="tokens generated per turn = the decode sample at that depth")
    ap.add_argument("--followups", type=int, default=2,
                    help="short turns at FULL depth after the ladder: the repeat of the "
                         "headline decode and incremental-TTFT numbers (CLAUDE.md)")
    ap.add_argument("--cold-sweep", default=None,
                    help="HTTP mode only: comma-separated token sizes, each a fresh "
                         "single-message conversation from a disjoint corpus region "
                         "(--sweep-offset-chars). Answers Q2 (FRANKEN-ENGINE-PLAN \xa72.4): "
                         "does this server reuse prefixes at all, or does the ladder's "
                         "low incremental TTFT just mean it is fast at everything")
    ap.add_argument("--sweep-offset-chars", type=int, default=300000,
                    help="start of the disjoint corpus region for --cold-sweep, so it "
                         "never reads text the ladder already consumed")
    ap.add_argument("--kv-slots", type=int, default=4, help="4 = what the gateway serves")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--tools", default=None)
    ap.add_argument("--api-key", default=None, help="HTTP mode: default ~/.colibri_api_key")
    ap.add_argument("--model-id", default=None, help="HTTP mode: default the first /v1/models entry")
    ap.add_argument("--server-log", default=os.path.expanduser("~/glm53_server.log"),
                    help="HTTP mode: gateway/server log to read REUSE/CKPT lines from")
    ap.add_argument("--arm", default=None,
                    help="which head-to-head arm this run is (A1/B1/B2/A2/... "
                         "FRANKEN-ENGINE-PLAN \xa72.2 H2) -- written into every row so "
                         "context_compare can pair runs without relying on file names")
    # 90, not ttft_serve's 96: the engine under measurement is itself ~89 GB of
    # anon memory on this box, so the shards cannot be 96% resident while it
    # runs. majflt per turn is the check that the model is not being re-read
    # from NVMe inside the measurement, and it is recorded on every row.
    ap.add_argument("--min-resident", type=float, default=90.0)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default="ctx")
    ap.add_argument("--engine-log", default=None)
    args = ap.parse_args()

    is_http = bool(args.url)
    steps = [int(s) for s in args.steps.split(",") if s]
    cold_sweep_sizes = [int(s) for s in args.cold_sweep.split(",") if s] if args.cold_sweep else []
    text = open(args.text, encoding="utf-8").read()

    if cold_sweep_sizes and not is_http:
        refuse_cold_sweep_in_engine_mode(cold_sweep_sizes)

    if not is_http:
        # HTTP mode's entire point is talking to an engine that is already running
        # (the live gateway, or a second server H0 stood up) -- this check is
        # engine mode's own "one engine at a time", not a claim about --url.
        others = T.other_engines()
        if others:
            sys.exit(f"REFUSED: another glm53 is running (pids {others}); one engine at a "
                     f"time, or the number is somebody else's. Stop the gateway.")
    assert_snap_resident(args, "start")

    if is_http:
        drv = HTTP_DRIVERS[args.driver](args)
        pin_field = None  # P8's pin is glm53's own chat-template artifact; an
                          # unknown message field may be rejected by a server
                          # this harness did not write (see the module docstring)
    else:
        drv = T.EngineDriver(args)
        pin_field = getattr(drv.rt, "REPLY_PIN_FIELD", "_coli_reply_pin")
    records, messages, cursor, cum_prefill = [], [], 0, 0.0
    prev_tokens = 0

    def emit(rec):
        rec.setdefault("mode", "http" if is_http else "engine")
        rec.setdefault("arm", args.arm)
        records.append(rec)
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(rec) + "\n")

    print(f"\n{'turn':>4} {'depth_in':>9} {'new':>7} {'total':>8} {'reused':>8} "
          f"{'ttft_s':>9} {'ms/newtok':>10} {'decode':>8} {'cum_prefill_s':>14}", flush=True)

    try:
        for i, step in enumerate(steps + [0] * args.followups):
            if step:
                body, cursor = chunk_text(text, cursor, step)
                content = body + QUESTION
            else:
                # A follow-up turn adds almost nothing: it measures what the NEXT
                # user turn costs once the context is already 64Ki deep, and
                # gives the repeated decode sample at that depth.
                content = ("In one sentence, and without repeating yourself: what "
                           "machine are these notes about?")
                body = ""
            messages.append({"role": "user", "content": content})

            assert_snap_resident(args, f"turn{i+1}")
            ev = drv.run(messages, args.gen)
            if ev["error"]:
                print(f"\nABORT: turn {i+1} failed: {ev['error']}", flush=True)
                emit({"tag": args.tag, "turn": i + 1, "error": ev["error"],
                      "side": "ladder" if step else "followup",
                      "target_new": step, "t": time.time()})
                return 2

            total = ev["prompt_tokens"] or 0
            reused_seen = ev.get("reused")  # None: driver has no signal at all
            reused = reused_seen or 0
            new = total - reused
            ttft = (ev["first"] - ev["submit"]) if ev["first"] else None

            if is_http:
                # Engine mode's ABORT is glm53's own contract, read off its
                # stderr. A server this harness did not write makes no such
                # promise -- the note is loud, the row's `reused` stays None so
                # nothing downstream treats a log-scrape as an assertion.
                print(f"    REUSE UNVERIFIED (http): server log reports "
                      f"reused={reused_seen if reused_seen is not None else 'n/a'}"
                      f"/{total} (not asserted; --cold-sweep is the oracle for reuse)",
                      flush=True)
            else:
                # The contract. Turn 1 reuses nothing by construction; after that
                # a turn that does not reuse the whole previous context is
                # measuring something else, and the ladder must not average the
                # two together.
                #
                # The floor is the previous turn's PROMPT length, not
                # prompt+reply. Reuse is all-or-nothing, so a turn that reuses at
                # all reuses everything the slot holds (prompt + generated - 1,
                # the last token being emitted but never re-ground); the only two
                # outcomes are ">= prev_tokens" and "0". Asserting the exact
                # prompt+reply total would additionally assume every generation
                # ran to --gen rather than stopping on EOS, and would abort a
                # perfectly good ladder over a one-token miscount hours in.
                if i > 0 and reused < prev_tokens:
                    print(f"\nABORT at turn {i+1}: the engine reused {reused} of the "
                          f"{prev_tokens} tokens it held, so this turn re-prefilled the "
                          f"history instead of extending it. Every number from here on "
                          f"would be a cold prompt wearing a ladder's label -- see the P8 "
                          f"note at the top of this file.", flush=True)
                    emit({"tag": args.tag, "turn": i + 1, "error": "reuse_broken",
                          "side": "ladder" if step else "followup",
                          "reused": reused, "expected_at_least": prev_tokens, "t": time.time()})
                    return 3

            comp = ev.get("completion_tokens")
            if comp:
                n_dec, decode_src = comp, "usage"
            else:
                n_dec, decode_src = ev["ntok"], "delta"

            cum_prefill += ttft or 0.0
            rec = {"tag": args.tag, "turn": i + 1, "kind": "ladder" if step else "followup",
                   "side": "ladder" if step else "followup",
                   "target_new": step, "depth_in": reused, "prompt_tokens": total,
                   "reused": None if is_http else reused, "new_tokens": new, "ttft_s": ttft,
                   "ms_per_new_token": (1000.0 * ttft / new) if (ttft and new > 0) else None,
                   "gen": ev["ntok"], "decode_src": decode_src,
                   "decode_tps": ((n_dec - 1) / (ev["done"] - ev["first"]))
                                 if ev["first"] and n_dec > 1 else None,
                   "cum_prefill_s": cum_prefill, "majflt": ev.get("majflt"),
                   "minflt": ev.get("minflt"), "hit_pct": ev.get("hit_pct"),
                   "slot": ev.get("slot"), "new_chars": len(body), "cursor_chars": cursor,
                   "t": time.time()}
            emit(rec)
            print(f"{i+1:>4} {reused:>9} {new:>7} {total:>8} {reused:>8} "
                  f"{(ttft or 0):>9.2f} {(rec['ms_per_new_token'] or 0):>10.2f} "
                  f"{(rec['decode_tps'] or 0):>8.2f} {cum_prefill:>14.1f}", flush=True)

            raw = "".join(ev["text"])
            if pin_field is not None:
                # P8's reply pin, by its own field name, because this harness IS
                # the gateway in engine mode and nothing else will apply it.
                # Without it render_chat_glm53's assistant branch emits
                #     <|assistant|><think></think>{content.strip()}
                # while the engine's KV holds <|assistant|><think> followed by
                # the RAW generation -- they diverge on the token after <think>,
                # reuse is 0, and every later turn is a cold re-prefill.
                # `content` is still carried for a reader; the pin is what
                # renders.
                messages.append({"role": "assistant", "content": raw, pin_field: raw})
            else:
                messages.append({"role": "assistant", "content": raw})
            prev_tokens = total

        # The cold sweep: HTTP mode only (refused above in engine mode), run
        # after the ladder+followups, each size its own throwaway conversation
        # from a disjoint region of the corpus so it shares no tokens with what
        # the ladder above already read.
        sweep_cursor = args.sweep_offset_chars
        for sz in cold_sweep_sizes:
            body, sweep_cursor = chunk_text(text, sweep_cursor, sz)
            content = body + QUESTION
            sweep_messages = [{"role": "user", "content": content}]
            assert_snap_resident(args, f"sweep{sz}")
            ev = drv.run(sweep_messages, args.gen)
            if ev["error"]:
                print(f"\nsweep {sz}: FAILED: {ev['error']}", flush=True)
                emit({"tag": args.tag, "kind": "cold-sweep", "side": "cold",
                      "target_new": sz, "error": ev["error"], "t": time.time()})
                continue
            total = ev["prompt_tokens"] or 0
            ttft = (ev["first"] - ev["submit"]) if ev["first"] else None
            comp = ev.get("completion_tokens")
            n_dec, decode_src = (comp, "usage") if comp else (ev["ntok"], "delta")
            rec = {"tag": args.tag, "kind": "cold-sweep", "side": "cold",
                   "target_new": sz, "prompt_tokens": total, "reused": None,
                   "ttft_s": ttft,
                   "ms_per_new_token": (1000.0 * ttft / total) if (ttft and total > 0) else None,
                   "gen": ev["ntok"], "decode_src": decode_src,
                   "decode_tps": ((n_dec - 1) / (ev["done"] - ev["first"]))
                                 if ev["first"] and n_dec > 1 else None,
                   "new_chars": len(body), "cursor_chars": sweep_cursor, "t": time.time()}
            emit(rec)
            # cold_sweep_sizes is only ever non-empty in HTTP mode (engine mode
            # refuses above), so this row is always a T_cold(sz) reference point.
            print(f"sweep {sz:>6} tok  total={total:>7}  ttft={ttft or 0:>8.2f}s  "
                  f"-- this row IS the cold reference T_cold({sz})", flush=True)
    finally:
        drv.close()

    ladder = [r for r in records if r.get("kind") == "ladder" and r.get("ms_per_new_token")]
    deep = [r for r in records if r.get("decode_tps") and r.get("prompt_tokens")
            and r.get("kind") in ("ladder", "followup")
            and r["prompt_tokens"] >= 0.9 * max(
                (x["prompt_tokens"] for x in records if x.get("kind") in ("ladder", "followup")),
                default=0)]
    if ladder:
        top = ladder[-1]
        print(f"\ncontext reached: {top['prompt_tokens']} tokens "
              f"({top['prompt_tokens']/1024:.1f} Ki)", flush=True)
        print(f"prefill rate: {ladder[0]['ms_per_new_token']:.1f} ms/token at depth "
              f"{ladder[0]['depth_in']} -> {top['ms_per_new_token']:.1f} ms/token at depth "
              f"{top['depth_in']} ({top['ms_per_new_token']/ladder[0]['ms_per_new_token']:.2f}x)",
              flush=True)
        print(f"cold TTFT at {top['prompt_tokens']} tokens (derived, = sum of the step "
              f"prefills): {cum_prefill:.0f} s = {cum_prefill/60:.1f} min", flush=True)
    if len(deep) > 1:
        print(f"decode at full depth, {len(deep)} samples: "
              + " / ".join(f"{r['decode_tps']:.2f}" for r in deep)
              + f" tok/s (median {statistics.median(r['decode_tps'] for r in deep):.2f})",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
