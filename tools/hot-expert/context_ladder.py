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
"""
import argparse, json, os, statistics, sys, time

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


def chunk_text(text, start_char, want_tokens):
    """A deterministic slice of the corpus, ~3.6 chars/token (ttft_serve's
    calibration on this same file). The engine's own count is what gets
    reported; this only has to land close enough to make the ladder even."""
    n = int(want_tokens * 3.6)
    if start_char + n > len(text):
        sys.exit(f"REFUSED: corpus is {len(text)} chars, the ladder needs "
                 f"{start_char + n}. Pass a bigger --text.")
    return text[start_char:start_char + n], start_char + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", dest="exe", required=True)
    ap.add_argument("--snap", default=T.DEFAULT_SNAP)
    ap.add_argument("--text", default=T.DEFAULT_TEXT)
    ap.add_argument("--steps", default=DEFAULT_STEPS)
    ap.add_argument("--gen", type=int, default=32,
                    help="tokens generated per turn = the decode sample at that depth")
    ap.add_argument("--followups", type=int, default=2,
                    help="short turns at FULL depth after the ladder: the repeat of the "
                         "headline decode and incremental-TTFT numbers (CLAUDE.md)")
    ap.add_argument("--kv-slots", type=int, default=4, help="4 = what the gateway serves")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--tools", default=None)
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

    steps = [int(s) for s in args.steps.split(",") if s]
    text = open(args.text, encoding="utf-8").read()

    others = T.other_engines()
    if others:
        sys.exit(f"REFUSED: another glm53 is running (pids {others}); one engine at a "
                 f"time, or the number is somebody else's. Stop the gateway.")
    T.assert_resident(args, "start")

    drv = T.EngineDriver(args)
    pin_field = getattr(drv.rt, "REPLY_PIN_FIELD", "_coli_reply_pin")
    records, messages, cursor, cum_prefill = [], [], 0, 0.0
    prev_tokens = 0

    def emit(rec):
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
            messages.append({"role": "user", "content": content})

            T.assert_resident(args, f"turn{i+1}")
            ev = drv.run(messages, args.gen)
            if ev["error"]:
                print(f"\nABORT: turn {i+1} failed: {ev['error']}", flush=True)
                emit({"tag": args.tag, "turn": i + 1, "error": ev["error"],
                      "target_new": step, "t": time.time()})
                return 2

            total = ev["prompt_tokens"] or 0
            reused = ev["reused"] or 0
            new = total - reused
            ttft = (ev["first"] - ev["submit"]) if ev["first"] else None

            # The contract. Turn 1 reuses nothing by construction; after that a
            # turn that does not reuse the whole previous context is measuring
            # something else, and the ladder must not average the two together.
            #
            # The floor is the previous turn's PROMPT length, not prompt+reply.
            # Reuse is all-or-nothing, so a turn that reuses at all reuses
            # everything the slot holds (prompt + generated - 1, the last token
            # being emitted but never re-ground); the only two outcomes are
            # ">= prev_tokens" and "0". Asserting the exact prompt+reply total
            # would additionally assume every generation ran to --gen rather
            # than stopping on EOS, and would abort a perfectly good ladder over
            # a one-token miscount hours in.
            if i > 0 and reused < prev_tokens:
                print(f"\nABORT at turn {i+1}: the engine reused {reused} of the "
                      f"{prev_tokens} tokens it held, so this turn re-prefilled the "
                      f"history instead of extending it. Every number from here on "
                      f"would be a cold prompt wearing a ladder's label -- see the P8 "
                      f"note at the top of this file.", flush=True)
                emit({"tag": args.tag, "turn": i + 1, "error": "reuse_broken",
                      "reused": reused, "expected_at_least": prev_tokens, "t": time.time()})
                return 3

            cum_prefill += ttft or 0.0
            rec = {"tag": args.tag, "turn": i + 1, "kind": "ladder" if step else "followup",
                   "target_new": step, "depth_in": reused, "prompt_tokens": total,
                   "reused": reused, "new_tokens": new, "ttft_s": ttft,
                   "ms_per_new_token": (1000.0 * ttft / new) if (ttft and new > 0) else None,
                   "gen": ev["ntok"],
                   "decode_tps": ((ev["ntok"] - 1) / (ev["done"] - ev["first"]))
                                 if ev["first"] and ev["ntok"] > 1 else None,
                   "cum_prefill_s": cum_prefill, "majflt": ev.get("majflt"),
                   "minflt": ev.get("minflt"), "hit_pct": ev.get("hit_pct"),
                   "slot": ev.get("slot"), "t": time.time()}
            emit(rec)
            print(f"{i+1:>4} {reused:>9} {new:>7} {total:>8} {reused:>8} "
                  f"{(ttft or 0):>9.2f} {(rec['ms_per_new_token'] or 0):>10.2f} "
                  f"{(rec['decode_tps'] or 0):>8.2f} {cum_prefill:>14.1f}", flush=True)

            # P8's reply pin, by its own field name, because this harness IS the
            # gateway in engine mode and nothing else will apply it. Without it
            # render_chat_glm53's assistant branch emits
            #     <|assistant|><think></think>{content.strip()}
            # while the engine's KV holds <|assistant|><think> followed by the
            # RAW generation -- they diverge on the token after <think>, reuse is
            # 0, and every later turn is a cold re-prefill. `content` is still
            # carried for a reader; the pin is what renders.
            raw = "".join(ev["text"])
            messages.append({"role": "assistant", "content": raw, pin_field: raw})
            prev_tokens = total
    finally:
        drv.close()

    ladder = [r for r in records if r["kind"] == "ladder" and r["ms_per_new_token"]]
    deep = [r for r in records if r["decode_tps"] and r["prompt_tokens"]
            and r["prompt_tokens"] >= 0.9 * max(x["prompt_tokens"] for x in records)]
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
