# Prefill / TTFT roadmap — GLM-5.3 on rome (opened 2026-09-06)

A separate track, because it has a different goal, a different gate, and a
different bottleneck from everything in `ROADMAP-2026-09.md`. That roadmap
optimised **decode throughput** (tok/s on short prompts). This one is about
**time-to-first-token on real prompts** — the number a person actually waits on
in an interactive UI. Nothing in the decode roadmap moves it.

## Why this exists (read before planning anything)

On 2026-09-06 GLM-5.3 was served to Open WebUI for the first time. It was
effectively unusable, and the reason was never a bug in any landed item — it
was a **scope gap the whole project shared**: every optimisation and every gate
measured decode on ~40-token prompts. Prefill of a real prompt was never
measured, never optimised, never tested. The first interactive use exposed it.

**Measured, clean, 2026-09-06** (direct CLI, `COLI_TIMERS`, engine hard-killed
first so no orphan contaminated it):

| prompt | wall (incl. ~77 s one-time load) | isolated prefill |
|---|---:|---|
| ~4 tokens | 76.8 s | — |
| ~602 tokens | 259.6 s | **598 tokens in 183 s → ~3.3 tok/s** |

Prefill runs at **decode speed** (~3.3 tok/s), which is the whole problem. On a
normal transformer prefill is far faster per token than decode because N prompt
tokens batch through one matmul. Here they do not, and the record already said
why in one line everyone (including this session, for hours) glossed over:
**"prefill is dominated by expert reads."** The RAM-bandwidth wall that bounds
decode bounds prefill too, multiplied by prompt length.

Consequences, all measured or derived:
- a plain short chat prompt (~30–100 tokens): **10–30 s** to first token
- a prompt carrying Open WebUI's 34 builtin tools (~6 000 tokens):
  **~30 minutes** to first token, every turn

The 34 tools are ~5 600 tokens of JSON schema attached to *every* message. That
is what turned "hi" into a 30-minute request.

### Why prefill was never optimised (the honest three)

1. **The benchmark defined the target and the benchmark measured decode.**
   Every gate in `ROADMAP-2026-09.md` is rotating-median tok/s on
   `datapoint.py`, which uses ~30–40-token prompts. An item was only justified
   if it moved that gate; nothing measured large-prompt TTFT, so no prefill
   item was ever justifiable. `G0` even recorded "cold TTFT 17.9 s" as one of
   four numbers — and nothing was ever built against it.
2. **A prefill item was specced and never started.** `ROADMAP-2026-09.md`'s
   Q5 — "4-accumulator BF16 matmul for prefill rows, 9.0 → 5.1 ms at S=32,
   *TTFT not decode*, half a day" — has sat unstarted on the Qwen track. It also
   only speeds the *compute*, which §below argues is not the bottleneck.
3. **The easy prefill wins were measured and don't beat this box's wall.**
   Commit `21999ff` measured GPU prefill as "not helping prefill-bound long
   context"; the record calls prefill "dominated by expert reads". SPEC-PROBE
   (`ROME-3x7900XTX-2026-09-04.md`) proved the killer detail: **cold experts do
   not dedup across tokens** — the CPU expert count was identical at every
   prefill chunk size 1→16. So batching N prefill tokens streams ~N× the expert
   volume; the GEMM batching that makes prefill fast elsewhere buys almost
   nothing here.

## The one lever that matters, and why it is hard

Given that prefill is streaming-bound and batching does not reduce the stream,
a faster kernel cannot fix TTFT. **The only lever that makes interactive use
viable is not computing the same prefix twice.**

The stable part of every turn — the system prompt and the ~5 600-token tool
block — is *identical* across turns and across conversations. If it is
prefilled **once** and reused, a follow-up turn only prefills the new user
message (tens of tokens = seconds). That is the whole game.

**It does not currently work.** Measured 2026-09-06: the same 806-token prompt
sent twice took 184 s then 179 s — no reuse. Whatever prefix-reuse machinery
exists (`slot_remember`, session state) is not firing on the serve path, and
the cause is unknown as of this writing.

**It is genuinely hard here, harder than normal KV caching**, because GLM-5.3
is not a plain transformer:
- 11 MLA layers hold a KV cache — standard, snapshot/restore is well-understood.
- **34 KDA layers hold recurrent (gated-delta) state.** A prefix cache must
  snapshot and restore *that* at the cache boundary, not just attention keys.
  G12 built `coli_vk_kda_sync` / `coli_vk_kda_upload` for exactly this class of
  state migration, so the machinery is not greenfield — but it is precisely the
  kind of subtle, serve-path, correctness-sensitive work that produced two
  separate bugs this session (the G12 cross-session state leak, the CANCEL
  wedge). Neither was caught by a decode oracle.

## The prerequisite nobody built (this comes first, always)

**P0 — a serve-path TTFT oracle and harness.** The root cause of 2026-09-06 is
that the engine was validated only as a decode-throughput benchmark, never as
an interactive server. Every gate below is stated in TTFT terms, and none of
them can be trusted without a harness that:

- drives the **actual serve protocol** (SUBMIT/DATA/STOP/CANCEL over the pipe,
  or the HTTP gateway), not the `--greedy` CLI path. The CANCEL fix this session
  passed a `--greedy 64` oracle that never executed the serve loop it changed,
  and shipped broken. **A prefill item validated on the CLI path is not
  validated.**
- measures **first-token latency**, separated from decode, at several prompt
  sizes including a realistic tool-laden one (~6 000 tokens).
- runs multi-turn: submit A, submit A+B, and prove the shared prefix was reused
  (turn 2 TTFT ≪ turn 1).
- asserts **residency before every run** (the discipline `profile_run.sh`
  already encodes and this session repeatedly ignored — a 12-minute "hi" was a
  cold-cache artifact, not the engine).

Until P0 exists, every item below is unmeasurable. Half a day to a day. Haiku
to build the harness once the shape is specced; the spec is Opus.

## The items

Effort and effect are honest, not optimistic. The ceiling is stated with each.

| id | item | mechanism | expected effect on TTFT | effort | tier | gate |
|---|---|---|---|---|---|---|
| **P0** | Serve-path TTFT harness | drive the real protocol, measure first-token latency, multi-turn reuse check, residency asserted | none directly — it is the instrument every other item is gated on | ½–1 day | Opus spec, Haiku build | first-token latency measured at 30 / 300 / 3 000 / 6 000-token prompts, reproducible ±10% |
| **P1** | Diagnose why prefix reuse does not fire | read the serve path: how the gateway maps a request to a slot, whether `slot_remember` matches the prefix across turns, what invalidates it | none — it is the design study P2 needs | 1–2 days | Opus | a written account of the current reuse logic and exactly why identical prompts re-prefill |
| **P2** | Prefix caching that works, KV **and** KDA state | at a turn boundary, reuse the cached forward state for the longest matching prefix; snapshot/restore MLA KV **and** KDA recurrent state (reuse G12's `kda_sync`/`kda_upload`) | turn 2+ of a conversation: **~30 min → seconds** for a tool prompt | **1–3 weeks**, high risk | Opus, Fable for the state-boundary design | P0 multi-turn: turn-2 TTFT within a few × turn-1's *new-token* count, **and** greedy text bit-identical to no-cache over 512 tokens (this is where a subtle state bug hides — the oracle must be the serve-path one, not decode) |
| **P3** | Warm the stable prefix at startup | prefill the system+tool block once when the server learns it (cache-on-first-sight), so even turn 1 of a new conversation reuses it | first turn of every conversation: **~30 min → seconds** (after a one-time warm) | +2–4 days on top of P2 | Opus | a fresh conversation's first turn reuses the pre-warmed prefix; TTFT independent of tool-block size |
| **P4** | Faster prefill compute kernel (the Q5 item) | 4-accumulator BF16 matmul for S>1 prefill rows | small — compute is not the bottleneck; ~1.1–1.2× overall at best | ½ day, low risk | Sonnet | TTFT on the 690-token document vs baseline; land only if P0 shows it is worth the numerics knob |
| **P5** | Capacity (cross-ref, not owned here) | int3 experts (`ROADMAP-2026-09.md` item 4h / G15) shrink the streamed volume for prefill and decode alike | partial — the prefill expert union is huge, so relief is bounded | weeks | Opus | its own numerics gate; tracked on the main roadmap |

## The honest ceiling, stated up front

Even if P2 and P3 both land perfectly:
- The **very first prefill of a given prefix is still paid once.** P3 hides it
  behind a startup warm for the *stable* block, but any prompt that changes the
  prefix — editing an earlier message, toggling a tool, a new system prompt —
  re-pays it at ~3.3 tok/s.
- This is a **latency** fix, not a throughput fix. It makes repeated prefixes
  free; it does not make the engine fast at genuinely novel long context. A
  32k-token document summarised cold is still ~2.7 hours of prefill. Nothing on
  this track changes that; only capacity (P5 / more VRAM) does, and only partly.

## The alternative this track must not bury

Recorded so a future session weighs it honestly rather than defaulting to the
hard path: **this box may not be the right home for low-latency interactive use
with large prompts.** Colibri's real strength is running a model too big to fit
in VRAM at all, for batch and throughput work. A model that *fits* — a smaller
model, or Qwen3.8 on the GPU tier — gives interactive tools that respond in
seconds today, with none of P0–P3. If the requirement is "tools that work,"
that is the faster answer; if the requirement is specifically "GLM-5.3 with
tools on this hardware," this track is the way, and it is weeks with real risk.

## Discipline this track inherits from 2026-09-06 (non-negotiable)

The interactive failure was compounded by measurement mistakes this session
made repeatedly. They are rules here, not suggestions:

1. **Validate on the path the user uses.** A prefill/serve item proven on the
   `--greedy` CLI path is not proven. Build and use P0 first.
2. **Measure the user's request, not a proxy.** Traffic generated from inside
   the Open WebUI container shares its source IP with the app — do not mistake
   your own test for the user's. Read the app's own record (its DB, its logs).
3. **Assert residency before every measurement.** A cold cache alone turned a
   2.8-second "hi" into 12 minutes. `profile_run.sh` already encodes this.
4. **One in-flight request while diagnosing.** This engine serves one at a time;
   an orphaned generation (e.g. a timed-out probe with the CANCEL bug unfixed)
   blocks the queue and poisons every later measurement. Kill hard and confirm
   `pgrep` is empty before trusting a number.
5. **No claim of "it works" without an end-to-end pass on the real path**,
   shown to the user, reproducible.

## Open dependency: CANCEL (from the decode session, still unfixed)

`glm53` does not honour CANCEL between generated tokens — an aborted request
runs to `max_tokens` holding the single engine slot (`ROADMAP-2026-09.md` /
this session's revert of the G16 attempt). Prefill work makes this worse, not
better: a 30-minute prefill that a user gives up on holds the engine for the
full 30 minutes. **P0's harness is also what a correct CANCEL fix needs** — the
G16 attempt failed because it had no serve-path oracle. Fix CANCEL as part of,
or immediately after, P0.
