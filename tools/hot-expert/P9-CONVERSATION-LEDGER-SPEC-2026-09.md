# P9 — the conversation ledger: the gateway owns the rendering (spec, 2026-09-09)

Design pass for the build that follows. It replaces the pattern this week has
been repeating — find a client behaviour that breaks prefix reuse, patch that
one behaviour — with a single mechanism that makes the whole class impossible
to hit silently.

## 0. Why, from the measurements

Prefix reuse rests on a contract nobody enforces: **the transcript the client
re-sends must re-render to exactly the token sequence the engine already
holds.** Nothing checks that, and Open WebUI has broken it three times in
three days, each time costing the owner minutes per turn:

| # | what the client did | cost, measured | patched by |
|---|---|---:|---|
| 1 | re-ranked the `<memory_context>` block every turn (vector search over the last 7 user messages) | turn 2 reused **0 / 195**, 21.6 s | P7 pin (`COLI_PREFIX_PIN`) |
| 2 | the UI's browser prompt shares only 2 163 tokens with an API-shaped one (a tool group attaches only from a saved chat), and the engine never planned a capture beyond a restore | every new chat **363 s** | P7b |
| 3 | stored an assistant turn as visible `content` only, dropping the model's `<think>` block | follow-up **54.07 s** against 2.7 s (29 generated tokens, 6 characters stored) | P8 reply pin |

Three patches, three knobs, and no reason to think there is not a fourth
behaviour: a client that trims whitespace, renumbers tool calls, truncates
history, or changes its tool list mid-conversation costs the same minutes and
looks identical from the gateway — a slow turn with no error anywhere.

**What P9 changes.** The gateway keeps, per conversation, the exact text it
already caused the engine to hold, and renders every later turn from that
record rather than from the client's re-derivation of it. It also *checks*
the prefix property before it submits, and reports when the engine disagrees.
The failure mode stops being "silently minutes slower" and becomes "one line
in the log that says the ledger diverged, and why".

## 1. The safety rule this design does not break

**The ledger decorates the client's transcript; it never replaces it.** For
each turn the client sends, the gateway may substitute its own exact
rendering *only when the visible content still matches what it recorded*.
It never injects a turn the client did not send, never changes what the user
sees, and never carries content from one conversation into another.

This matters because the conversation key is derived, not given: Open WebUI
pops `metadata` (with `chat_id`) before forwarding upstream
(`routers/openai.py:1490`), so the gateway sees only the OpenAI body. Two
chats that open with the same first message therefore share a key by
construction — and the match rule above is what makes that harmless: the
shorter/different transcript does not match the record, so the record is
dropped rather than injected.

## 2. What the ledger holds

Keyed like `conversation_cache_slot` today (leading system messages + first
user message), LRU, 64 conversations, in memory, process lifetime:

```python
Ledger = {
  "key": str,                 # sha1 of the stable head, as today
  "slot": int,                # the KV slot this conversation owns
  "prefix": str,              # rendered head: tool block + base system + pinned block
  "tools_sig": str,           # hash of the tool list the conversation started with
  "turns": [                  # in order, one per turn the engine has actually seen
     {"role": "user",      "rendered": str},
     {"role": "assistant", "rendered": str, "visible": str},   # raw incl. <think>…</think>
  ],
  "prompt": str,              # the exact prompt string of the last submit
  "generated": str,           # the exact text the engine generated for it
  "used": float,
}
```

`prefix` subsumes P7's pin (the memory block captured at turn 1) and adds the
tool block: a conversation keeps the tools it started with, so a mid-session
tool-list change costs nothing until the next new chat. `turns[].rendered`
subsumes P8's reply pin and generalises it — whitespace, tool-call ordering
and any other client normalisation stop mattering, because the client's copy
is only ever used to *identify* the turn, not to render it.

## 3. The request path

1. **Match.** Walk the client's messages against `turns` in order. A user
   turn matches on normalised text; an assistant turn matches when its
   `content` equals the recorded `visible` (the same normalisation P8 uses).
   Stop at the first mismatch.
2. **Decide**, by where the walk stopped:
   - *exact continuation* (all recorded turns matched, the client adds new
     ones): render `prefix + turns + the new turns` — the prompt is then
     `ledger.prompt + ledger.generated + the new user turn`, by construction.
   - *client is shorter* (regenerate, edit, branch): truncate the ledger to
     the matched turns and render from there. The engine cannot rewind, so
     this re-prefills; that is correct, not a regression.
   - *content diverges* mid-history: drop the ledger, render the client's
     version verbatim, start a new record. Log `ledger=reset reason=diverged
     at=<turn>`.
   - *unknown key*: new ledger, render the client's version, record it.
3. **Check before submitting.** In the continuation case the gateway asserts
   `new_prompt.startswith(ledger.prompt + ledger.generated)`. It is a string
   comparison over text it produced itself, so it is cheap and total. On
   failure: do not submit a prompt that claims a prefix it does not have —
   log `ledger=broken` with the first differing offset and fall back to the
   client's rendering. **This is the check that would have caught all three
   defects above before a user saw them.**
4. **Record after.** On DONE, append the user turn and the assistant turn
   with the raw generated text (before the reasoning splitter, before any
   strip), and store `prompt`/`generated`. On CANCEL, record exactly what the
   engine kept (P7's cancel path already leaves the slot consistent at
   `filled`), so the resumable-prefill behaviour survives.
5. **Slot.** The ledger owns its slot for the conversation's life, replacing
   today's hash-modulo routing: two conversations that hash alike stop
   evicting each other.

## 4. Making the engine's agreement observable

Today the engine's `REUSE <id> <reused> <prompt>` line goes to stderr and
nothing programmatic reads it. Add `reused` as a trailing field of the
engine's `DONE … STAT` line (append-only, older gateways ignore extra
fields, and `openai_server.Engine` already parses positionally). The gateway
then logs, per request:

```
[ledger] key=8f1c turn=4 expect_reuse=6142 engine_reuse=6142 ok
[ledger] key=8f1c turn=5 expect_reuse=6180 engine_reuse=4419 MISMATCH
```

A mismatch is the alarm this project has been missing: every regression this
week was invisible except as latency. `accept_live.sh` gains a check that
the last N requests carry no `MISMATCH`, so the daily canary catches it
without anybody watching.

## 5. Knobs and blast radius

| env | default | meaning |
|---|---|---|
| `COLI_LEDGER` | 1 | 0 = today's path exactly (pins included), for A/B and for a fast rollback |
| `COLI_LEDGER_MAX_CONV` | 64 | LRU size; ~50 KB of text per conversation |
| `COLI_LEDGER_STRICT` | 0 | 1 = refuse the request on a broken invariant instead of falling back (gates only) |

Gateway-only: `c/openai_server.py` plus the one-field `STAT` addition in
`c/glm53.c`. No numerics change, no shader change, the engine's binary stays
bit-identical apart from that field. **Any exception inside ledger code falls
back to today's rendering and logs** — the ledger may never be the reason a
request fails. P7's pin and P8's reply pin remain as the fallback path when
`COLI_LEDGER=0`; they are dead code when it is on, and the build should say
so rather than leaving two mechanisms silently fighting.

## 6. Gate (executable; `p9_gate.sh`)

1. `prefill_gate.sh <served> <candidate>` with `MIN_SPEEDUP=0.97`,
   `GLM53_PREFIX_CKPT=0`, a private `COLI_CKPT_DIR`: bit-identical, neutral
   (a gateway change plus one STAT field must not move a serve row).
2. `tworeq.py TWOREQ_SLOTS=4` at both KDA knobs: IDENTICAL, no forcing line.
3. **The three known client behaviours, as scripted regressions** — each must
   reuse `prompt + gen` with `COLI_LEDGER=1` and fail with `COLI_LEDGER=0`,
   proving the mechanism is what fixed it:
   a. re-ranked memory block (P7's case, four temporary memories);
   b. a reply containing reasoning (P8's case, the three-box word problem —
      it reasons at temperature 0 and 0.8, unlike the matrix's prompt);
   c. a client that strips trailing whitespace from the reply (synthetic;
      the class neither pin covers).
4. **Divergence correctness**: regenerate, edit-an-earlier-message and
   branch each produce the same text as a cold render of that transcript
   (greedy, byte-identical), and log `ledger=reset`. Slow is allowed; wrong
   is not.
5. **The invariant, in production shape**: a 10-turn browser conversation
   (`ui_probe.mjs --follow-up` extended to N turns) with zero `MISMATCH`
   lines and `expect_reuse == engine_reuse` on every turn.
6. `accept_live.sh` (rig) and `accept_ui.sh` (Mac, real browser) exit 0, and
   `ui_matrix.sh --sizes 0,500,2000` shows **every turn-2 row reusing
   `prompt + gen`** — the headline P8 was reaching for, now by construction.

## 7. What this does not fix, stated plainly

- New tokens still cost ~170 ms each. A 3 500-token paste is minutes, ledger
  or not; only capacity changes that (see the roadmap's standing note).
- A client that genuinely sends a different conversation gets a re-prefill.
  That is the KDA recurrence, not a policy choice.
- Two chats opening with an identical first message still share a key; the
  match rule makes that harmless, and `chat_id` would make it exact if a
  client ever forwards one (Open WebUI does not).

## 8. Effort

Two to three days: the ledger and its state machine with unit tests (the
match/decide table is where the bugs will be, so it is tested at that level,
not through the engine), the STAT field, the gate, the chain. Opus.
