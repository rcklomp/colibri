# Prefill / TTFT roadmap — GLM-5.3 on rome (opened 2026-09-06, rev 21, 2026-09-10 01:44)

A separate track, because it has a different goal, a different gate, and a
different bottleneck from everything in `ROADMAP-2026-09.md`. That roadmap
optimised **decode throughput** (tok/s on short prompts). This one is about
**time-to-first-token on real prompts** — the number a person actually waits on
in an interactive UI. Nothing in the decode roadmap moves it.

**Rev 21 (2026-09-10, 01:44): P9 — the conversation ledger, gated and in
service.** This week was three patches for three client behaviours, each found
by the owner as minutes of latency and nothing else. P9 replaces the pattern
with one mechanism: the gateway keeps, per conversation, the exact pieces THE
RENDERER produced for the prompt the engine ground plus the raw text it
generated, renders every later turn from that record — the client's copy is used
only to IDENTIFY each turn — and **asserts `new_prompt.startswith(ledger.prompt
+ ledger.generated)` before it submits**. `COLI_LEDGER` (default 1); 0 is today's
path with P7's pin and P8's reply pin, which are OFF rather than merely unused
when the ledger is on. The engine gains ONE trailing field on `DONE … STAT`
(`reused`, append-only), so the gateway can print
`[ledger] … expect_reuse=4746 engine_reuse=4746 ok` per request — the alarm this
project has been missing, and `accept_live.sh`'s new check 2b carries it into
the daily canary. **The three known behaviours, three seeds each, three arms:**

| behaviour | ledger ON | ledger OFF (today) | no mechanism |
|---|---|---|---|
| re-ranked memory block | **3/3** | 3/3 | **0/3** |
| a reasoning reply | **3/3** | 3/3 | **0/3** |
| a client that trims whitespace | **3/3** | **0/3** | 0/3 |

The third row is the item's claim: P8's pin tolerates a trimmed REPLY, nothing
tolerated a trimmed QUESTION, and today's gateway reuses **0 of 46** there (and
sends turn 2 to a different KV slot, because `conversation_cache_slot` hashes the
first user message). Divergence is byte-exact: regenerate, edit-an-earlier-
message and branch each render text **IDENTICAL to a cold gateway's** rendering
of the same transcript, each logging `ledger=reset`. Engine bit-identical
(`teacher_forcing IDENTICAL (782 positions)`, `cosine=1.0000000 max_abs=0`, TTFT
0.99×/1.00×/1.00×), `tworeq` IDENTICAL at both KDA knobs with 0 forcing lines,
`accept_live.sh` PASS, 267 gateway tests green, and the renderer still
byte-identical to the checkpoint's `chat_template.jinja` after gaining its
`parts` hook. **Still owed: the ten-turn browser conversation, `accept_ui.sh` and
`ui_matrix.sh`** — both attempts collided with a second session driving the rig
(`cancel2_chain.sh`), and a browser turn queued behind another session's chain
measures that chain. Gate `p9_gate.sh`, chain `p9_chain.sh`, cases
`p9_memory_case.sh` / `p8_reasoning_case.sh` / `p9_strip_case.sh`, divergence
`p9_divergence.sh`, §P9 in the record. Binary in service **34357e49…**, serving
script unchanged.


**Rev 20 (2026-09-09, 16:05): P8 — the reply pin. Built, the bug reproduced on
the served gateway, and UNGATED: the pristine stays in service.** When GLM-5.3
emits a `<think>` block, Open WebUI gives it back as visible `content` only
(confirmed here by reading `webui.db`, not by inference), the rendered prompt
stops matching the tokens the KV slot holds, the KDA state cannot rewind past
prompt + reply, and the turn falls back to the checkpoint. Measured through Open
WebUI's own backend against the binary in service, before any change: turn 1
**prompt 4 718, gen 29, 6 characters of visible answer**; turn 2 **REUSE 4 419
and 54.07 s** where it should be **4 747 and ~2.7 s**. The fix is gateway-side
(`c/openai_server.py`, `COLI_REPLY_PIN`, default 1): remember the RAW generated
text per conversation and render it back verbatim, so the prompt reproduces the
engine's own tokens. `c/glm53.c` is untouched and the built binary is
**byte-identical** to the one in service. Offline oracle passes — `teacher_forcing
IDENTICAL (782 positions)`, `cosine=1.0000000 max_abs=0`, TTFT 0.97×/1.00×/1.00×
— and 16 new offline tests assert the invariant that matters,
`render(turn 2) startswith render(turn 1) + RAW`. **Two things stopped it.**
(1) The gate's speed check exited 3 at the 27-token row **comparing a file with
itself**, sha256 equal: the same 2.7-second noise floor that made P5b's first run
exit 3. `p8_gate.sh` now downgrades the speed verdict to informational when the
two binaries are byte-identical — categorical, not a threshold move — and the
oracle half stays fatal. (2) The chain reverted on that rc, **verified** — binary
`f68d1cac…`, shader `b7d57f05…`, tree `356cdd3`, `gateway: 1 engine: 1`,
`/v1/models` 200, 99 s from verdict to serving — and the Mac then left the rig's
network, so steps 3 and 4, `accept_live.sh`, `accept_ui.sh` and the browser
matrix were never run. `tworeq` at 4 slots is **IDENTICAL at both KDA knobs with
0 forcing lines**. A second instrument defect came out of the same run: the
gate's verdict printed `rc=0` for steps 3 and 4, which had never started — fixed,
skipped steps now read `-- (not run)`. **The item is not done.** One instrument finding
belongs on the previous rev's matrix: its own 500-token filler plus "Reply with
the single word OK." answers `gen=2` with an EMPTY `<think>` block at
temperature 0 and at 0.8 — the browser rows that reasoned did so by SAMPLING,
3 of 5 — so the regression case uses a small system of equations, which reasons
every time (gen 29, 34 B reasoning, 6 B visible). Gate `p8_gate.sh`, chain
`p8_chain.sh`, case `p8_reasoning_case.sh`, §P8 in the record. Binary in service
unchanged, **f68d1cac…**, serving script unchanged.

**Rev 18 (2026-09-09, 09:00): P7b — a partial checkpoint was terminal, and a
new chat in the owner's browser now starts warm in 20.98 s instead of
362.94 s.** `serve_one` planned a capture only when `shared == 0`, so after a
`CKPT hit` it never planned again: this morning the engine held a 2 163-token
checkpoint (the LCP of the UI shape and an API-shaped warm-up, chosen over a
longer hint) and every UI-shaped request restored those 2 163 and re-prefilled
~2 300 more — **REUSE 6 2163 4424, ttft 380.14 s at 07:24**. Plan and hint now
also run after a RESTORE (never after slot reuse, whose session is continuing a
conversation), only strictly beyond `shared`, and the LONGER of the two wins —
the hint is the boundary the gateway knows is stable, and it was the longer one
in the case that hurt. Live: `CKPT hit prefix=2163` → **`CKPT plan prefix=4308
src=hint after-restore`** → the next new chat **REUSE 4 308/4 428 in 20.98 s**
(18.1× against 07:24). Bit-identical with checkpoints off (`max_abs=0` at 782
positions, TTFT 1.00×/1.01×/1.01×), `tworeq` IDENTICAL at both KDA knobs with
zero forcing lines, and the serve-path oracle on the new case is text-identical
at `cosine=0.9999986 max_abs=0.04251` — **2 000× P7's 2.301e-05, because that
state carries two extra chunk boundaries instead of one**; same class, same
knob, quantified in the record. New harness case
`ttft_serve.py --prefix-ckpt-partial`, gate `p7b_gate.sh`, chain
`p7b_chain.sh`. Binary **f68d1cac…**, serving script unchanged. One cost
recorded and deliberately not tuned away: after a restore, a prompt that shares
one more token with the previous fresh prompt captures again (~300 MB of copy
and disk write to gain five tokens) — a minimum *gain* rather than a minimum
*length* is the follow-up.

**Rev 18 (2026-09-10, 02:32): the CANCEL flake is closed too.** Rev 17 closed
the engine half; what remained was a gateway race, worth 1 abandoned request in
3 holding the engine for its whole prefill with no `CANCEL` in the log at all.
The cancel could reach the engine before it had dequeued that SUBMIT, was
answered `NOT_FOUND`, and was lost. Fixed by retrying on that ack (`3d80c07`)
and by not stranding a request whose cancel failed — the dispatcher popped a
request's routing entry on every error frame (`07fa39a`). Gate `check4.sh 6`:
**6/6 cancelled during prefill, next request 6.6–8.7 s**, plus `accept_live`
in full; it ran through `run_chain.sh`, the rig lock's first production run.
The first attempt at this fix was wrong and its own chain reverted it —
deferring the cancel to the ACCEPT frame measured 5/5 at ~155 s, worse than the
bug. §"CANCEL retry" in the record.

**Rev 17 (2026-09-09, 01:00): the CANCEL dependency is CLOSED — a
1 230-token prompt cancelled at 5 s is confirmed in 12.9 s instead of 251.3 s,
and the live path is where the design turned out to be wrong.** `glm53` now
polls stdin between prefill chunks and between decode steps (`serve_poll.h`,
#1332, the pattern `qwen36` already used), stops at the chunk or token boundary,
and answers `PROF` + `DONE … STAT` + `ERROR <id> CANCELLED`. Bit-identical
(`max_abs=0` at 782 positions) and the poll is free: TTFT **1.02× / 0.99× /
0.99×** at 27/390/1 236 tokens with decode unmoved, against a pristine that has
no poll at all. A cancelled prefill is **resumable**: the slot records exactly
`session->filled`, so the identical prompt resubmitted reuses **384 of 1 236**
— the position the cancel line reported, exactly — and returns **the same greedy
text** as an uncancelled run (89 bytes, 24 tokens), in 99.73 s instead of
140.01 s. Live, an abandoned `curl` goes from **132.32 s to 13.18 s** and the
request waiting behind it from **126.35 s to 7.13 s**. **Two things the plan had
wrong, both found by measuring and not by reading:** (1) a SUBMIT *can* arrive
mid-turn — `GenerationScheduler` is built with `capacity = kv_slots`, four in
service — and the first design's "push the header back and stop draining" left
the CANCEL stranded behind it, which the live check showed is the NORMAL case,
not a corner: two runs out of two at 132 s, and an earlier round at 7 s only
because the race went the other way. The drain now reads the queued frame in
full onto a FIFO and keeps looking. (2) A cancelled request had never appeared in
`owui_report.sh` at all: `generate()` raises `ClientCancelled` before its
`[req]` write. Fixed in `openai_server.py`, and that fix is what made (1)
visible. Three instrument findings are in the record's §CANCEL, all caught by
the new `--cancel-phase` check: a decode-phase cancel timed from SUBMIT is a
guess about two clocks; `--cancel-phase prefill` is unreachable over `--url`;
and `--cancel` used to print its verdict and leave the exit code at 0. Gate
`cancel_gate.sh` (b, c, a, d — the two steps a CANCEL change can break run
before the four-hour half), chain `cancel_chain.sh` with the live step now a
gate of its own, regression test `c/tests/test_glm53_cancel_frames.c` (no model,
seconds, and it fails on the exact line the live run failed on). Binary
**f41fc5cc…**, serving script unchanged. **This table is still empty.**

**Rev 16 (2026-09-08, 21:30): P5b is in service and the roadmap row that
predicted it was wrong about the cause — the big term was FALSE SHARING on the
per-thread scratch, not the 64 walks of the latent.** `mla_layer` sized its
per-thread slices as `malloc(nthreads * L * sizeof(float))` (L = 512 → 2 048 B)
and `malloc(nthreads * width * sizeof(float))` (width = 2 051 → 8 204 B): a
16-byte-aligned base and a stride that is not a multiple of 64, so every pair of
neighbouring threads shares the boundary cache lines of their `pooled` and
`score` slices — and the pool writes them once per selected slot per head,
1 444 times per head per layer at 3 462 tokens. **Rounding those two
allocations up to a cache line takes `mla.attn` from 51.6 to 22.5 ms/token at
3 462 tokens and from 23.0 to 13.7 at 781; the prefill token goes 147.3 → 118.1
(1.25×) and 108.7 → 99.2 (1.10×) against the same binary's knobs-off row, 1.30×
and 1.11× against the pristine binary in service.** The mechanism the row was
named after — one blocked walk of the latent for all 64 heads instead of 64
walks — is built, is bit-identical, is 3.6× on the pool in isolation, and is
**neutral in the engine**: +0.2 ms/token at 781 tokens and −0.3 at 3 462, because
once the false sharing is gone the per-head pool runs at L3 bandwidth and the
layer's latent (7.1 MB at 3 462 tokens, 78 MB for all eleven) is L3-resident
while it runs. So `COLI_MLA_POOL` defaults to **1** (padded scratch), not 2
(padded + blocked pool), by this table's own rule; mode 2 keeps its number and
its knob because it is the shape that wins if the latent stops fitting in L3.
Everything is bit-identical at every setting: `max_abs=0` at 782 positions at
BOTH KDA knobs, `tworeq` 4 slots IDENTICAL with 0 forcing lines, P7's capture
still fires (REUSE 1 527/1 544 in 2.25 s, t1/t2 = 80×), serve-path TTFT
**1.01× / 1.07× / 1.15×** at 27/390/1 236 tokens. One finding lands on an older
row: the artefact is an allocator lottery — the pristine binary and the
candidate's own knobs-off build differ by 5.8 ms/token in `mla.attn` on
identical source — and that is the explanation of the contradiction rev 15
recorded for the dev merge ("the CLI oracle run put the candidate 7 % faster
with all of it in `mla.attn`, which nothing in the merge touches"). The merge
moved the allocations, not the arithmetic. The gate ran twice: the first run
(`MIN_SPEEDUP=1.0`) exited 3 on the 27-token row alone — 2.82/2.61 s against
2.81/2.62 s, 0.4 % of a 2.7-second measurement — and the bound is now 0.97 in
`p5b_chain.sh`, the bound P6b, RP4 and the dev merge used, committed with the
reasoning rather than passed by hand. Binary `65d7d70d…`, serving script
unchanged. **This table is now empty.**

**Rev 15 (2026-09-08, 01:30): the upstream `dev` merge is done and in
service — bit-identical, neutral on every row.** `JustVugg/colibri` `dev`
`1ccee43` merged into `hot-expert-tier` (merge base `12a5c46`, 149 ahead / 125
behind), binary `5ed096a6…`, serving script unchanged. Three files conflicted —
the file list the 2026-09-07 dry run predicted, with **eight hunks instead of
five**: three of the new ones are not about mapping at all but about `dev`'s new
per-turn `PROF`/`HITS` dashboard clocks landing in the same lines as the fork's
G10/P2.3 parallel hyper-connections and P2.3's last-row head, and were resolved
as a **union**. The mapping hunks took `dev`'s `st_map_shard_range` (#1325 —
the fork's own PR, upstreamed as `eabeb9a`) and kept three things `dev`'s hunk
drops, because measurements here depend on them: the mapping stays **on by
default** for these two engines (`st.h` gained `COLI_MAP_EXPERTS_DEFAULT`;
upstream ships it opt-in), `expert_map_init` stays so `g_map_all` can still size
the LRU at one slot per expert and so `st.h`'s unsynchronised per-fd table is
never filled from inside an OpenMP region, and both prefaults stay
(`GLM53_MMAP_POPULATE` is §G1/§G1b's knob; qwen38's is worth 4–6× and is on).
`~/bench/devmerge_chain.sh` exits 0 on all six steps: 790 python tests and the
four qwen38 C tests green, the chat-template pin **25 renders byte-identical**
to the checkpoint's `chat_template.jinja`; `[MAP]`, `mmap serve=7446 copy=0`,
the bind counters and the 44.3 GB budget **identical to the pristine's**, with
`majflt` 0 and MemAvailable 12 GB *higher*, not lower; `prefill_gate.sh` at
`MIN_SPEEDUP=0.97` **`max_abs=0` at 782 positions** and TTFT **0.99×/1.00×/0.99×**;
`tworeq` at 4 slots IDENTICAL at both KDA knobs with **0 forcing lines**;
`--prefix-ckpt` t1/t2 = **91.8×** and `--pin-block` REUSE 994/1 010; and
`rome_bench.sh` paired against the P5 binary in the same session at
**1.004× / 1.003× / 1.013×** (rotating 2.60 vs the 2026-09-05 row's 2.59). One
number is recorded as a contradiction rather than a win: the CLI oracle run put
the candidate 7 % faster with all of it in `mla.attn`, which nothing in the merge
touches — the serving regime says neutral, and that is what the record keeps.
The eleven upstream changes that matter on this box (the GLM-5.3 prompt now
always opens `<think>`, `RssAnon`, `compat_mem_available_gb`, the Brain/Profile
lines, `max_kv_slots` 16 reached upstream too, the Makefile header deps, the
`image_url` hardening) are listed in §"Upstream dev merge" of the record. Next:
**P5b** — the only open item on this table. (P5b landed the same day — see rev 16.)

**Rev 14 (2026-09-07, 23:50): P5 in service — 174.4 → 146.4 ms/token at
3 462 tokens (1.19×), bit-identical, and none of the three sub-items needed the
knob the roadmap budgeted for them.** One cause in three places: `dot +=
a[d]*b[d]` compiles to vectorised PRODUCTS and an in-order scalar add chain, so
it runs at **0.4 MAC/cycle/thread** — latency-bound on its own accumulator, not
bandwidth-bound. Each site has a free axis for the SIMD lanes, and using it
keeps every summation order intact: **P5.1** puts the 64 MLA heads in the lanes
over a transposed query slab (`mla.attn` **71.4 → 48.2** ms/token at 3 462),
**P5.3** puts the chunk's tokens in the lanes over a transposed `x` (`router`
**5.2 → 1.4**), and **P5.2** records a chunk's S KDA recurrence dispatches into
ONE command buffer with barriers instead of S submits (`kda.step` **6.0 →
3.8**). `~/bench/p5_chain.sh` exits 0 on all four gate steps, including the one
the earlier gates did not have: **`prefill_gate.sh` at `ORACLE_KDA_GPU=2`,
which is P5.2's only oracle** — the GPU recurrence never executes at knob 0, so
step (a) alone would have passed a broken batched submit. `max_abs=0` at 782
positions at BOTH knobs; `tworeq` at 4 slots IDENTICAL with 0 forcing lines;
P7's checkpoint still captures (REUSE 1 520/1 538 in 2.97 s, 70× on t1/t2).
Serve-path TTFT **1.06× / 1.15× / 1.25×** at 21/384/1 230 tokens. Two things
the harness said and the record keeps: the FMA barrier (gcc contracts an
intrinsic mul+add back into `vfmadd231ps` and ignores `#pragma STDC
FP_CONTRACT OFF`; without an empty `asm` the oracle sees 78 355 of 92 416
scores 1 ulp off), and the **21.5× score pass buys only a 1.48× bucket**,
because the weighted pool that follows it lost the locality the old loop nest
had by accident — that is P5b, sized below. Next: **P5b**, then the
upstream-`dev` merge.

**Rev 13 (2026-09-07, 20:10): RP4 answered, and it closes the expert shader.**
`COLI_VK_TIMESTAMPS` (off by default, bit-identical on and off, TTFT 1.00× with
it on) puts `VK_QUERY_TYPE_TIMESTAMP` queries inside the expert group's command
buffer on all three devices and, via `VK_EXT_calibrated_timestamps`, splits the
CPU-side `eg` wait into queue latency, GPU busy per phase and fence tail. The
answer, at 781 and 3 462 tokens with three repeats each: **of `eg`'s 58.1
ms/token, 53.6 is the CPU experts running concurrently inside the same timer
(92 %), GPU busy on the busiest device is 1.97, and the time the GPU actually
makes the engine wait is 0.068 ms/token — 0.12 %.** P4b and P4c did not fail to
move the expert group; the bucket was never GPU work, and both of §P4c's
remaining suspects (per-dispatch fixed cost, occupancy at 8 output rows) are
refuted by their own numbers. The per-expert table does find two real kernel
facts — a one-row expert costs 2.7–3.4× a two-row one (the per-row shader's LDS
bank conflict, `vk_tile_ok4` requires `S > 1`), and the R = 8 tile is not flat
in rows because it re-reads 67.1 MB of activations per row against a 14.16 MB
weight stream — but both are capped by `gpu_late`: **≤ 0.07 ms of a 120.6 ms
prefill token, and ≤ 1.96 ms of a 167.5 ms decode token (1.2 %)**. Neither was
built. The larger residual is host-side: `eg − cpu` is 4.5 ms/token in prefill
and ~6.0 in decode for 126 issue/take pairs per token, which is `VK_PROF`'s
question and §G13's shape of fix. **The prefill levers are unchanged and are
not the GPU:** CPU experts 53.6 ms/token at 781 tokens, MLA attention 69.6
ms/token at 3 462 (40 % of the token) — that is P5. Next: P5, then the
upstream-`dev` merge. (P5 landed the same night — see rev 14.)

**Rev 12 (2026-09-07, 17:00): P6b in service — the CPU recurrence P6 forced is
gone.** One KDA state+window set per KV slot on dev0, allocated in `model_load`
**before** the expert preload (after it, the 624 MB would come out of dev0's
3.0 GB reserve); `coli_vk_kda_*` and `GSession` carry the slot; `slots_init`
forces `COLI_KDA_GPU=0` only when the pool is short, loudly. `p6b_gate.sh`
exits 0 on all six steps: `tworeq` at 4 slots IDENTICAL at `COLI_KDA_GPU=2`
**with zero `forcing` lines** (the check that makes the oracle mean anything)
and at `=0`; bit-identical against the P7 binary at one slot (782 teacher-forced
positions, last-logit `max_abs=0`, TTFT 1.00/0.98/0.98×); decode at four slots
**5.702 tok/s vs 5.728 at one slot on the GPU (0.996×) and 4.990–5.089 with the
CPU recurrence (+13.2 %)**; the live side-request case through the gateway
reuses **1 339/1 355 in 3.08 s** with no `forcing` line; and the device→host
state sync moved off write-combined memory — **370 MB checkpoint captured with
a 69 ms sync (156 MB of KDA spans, ~2 262 MB/s) where the old path was
~14 MB/s** — which is what makes a P7 checkpoint possible at the GPU knob at
all (conv 2 with the 34-tool block: 5.15 s, t1/t2 = 279×). Price: the preload
drops **1 296 → 1 248 heat-ranked experts** (48, 3.7 % of dev0's tier), hit
rate unchanged. Serving since 16:50 at `--kv-slots 4` with `COLI_KDA_GPU=2`.
Next: RP4, P5, then the upstream-`dev` merge. (RP4 landed the same day — see
rev 13.)

**Rev 11 (2026-09-07, 12:30): P7 in service.** Prefix checkpoints in the
engine (the `deepseek_v4.c` `v4_ckpt_*` pattern over the segment adapter's own
span list, persisted under `<SNAP>/.coli_ckpt`) plus the gateway pin. `p7_gate.sh`
exits 0 on all five steps: with `GLM53_PREFIX_CKPT=0` the candidate is
bit-identical to the pristine engine (teacher forcing identical at 782
positions, last-logit `max_abs=0`, serve-path TTFT 0.98–0.99×); with
checkpoints on, a **new** conversation carrying the 34-tool Open WebUI block
goes **1 452 s → 5.30 s** (6 328 of 6 346 tokens restored) and **9.72 s** after
the engine is killed and respawned; the restored answer is byte-identical over
56 greedy tokens with first-token logits `cosine=1.0000000, max_abs=2.146e-05`;
`tworeq` at 4 slots is identical at both KDA knobs; a re-ranked
`<memory_context>` block costs 17 prefilled tokens instead of 1 005
(126.82 s → 3.00 s). Through the live gateway, restarted with the gate's
checkpoints on disk and nothing in memory, turn 1 of a conversation the process
has never seen answers in **8.93 s**. Serving since 12:25 with
`GLM53_PREFIX_CKPT=1 COLI_PREFIX_PIN=1`. Next: P6b, RP4, P5, then the
upstream-`dev` merge. (P6b landed the same day — see rev 12.)

**Rev 10 (2026-09-07, 06:40): P7 scoped by measurement; spec written.** The
first thing P7 did was the two-turn chat with memory context on, through
Open WebUI's own backend: with the owner's single memory turn 2 reused
162/177 tokens (2.1 s); with four memories the block's ranking changed
between turns, the system message with it, and turn 2 reused **0/195**
(21.6 s) on a different slot (§P7 step 0 in the record). So P7 is a
checkpoint **and** a gateway-side prefix pin; the design is
`P7-PREFIX-CKPT-SPEC-2026-09.md` (engine checkpoints ported from
`deepseek_v4.c`, disk-persistent; `COLI_PREFIX_PIN` on the gateway; the
executable gate `p7_gate.sh`). Build: Opus, on `perf/p7-prefix-ckpt`.

**Rev 9 (2026-09-07, 01:30): P6 in service.** The gateway's 1-slot cap for
glm53 was unjustified and is now 16; the engine forces the CPU recurrence
for more than one slot; identity holds across slots; and through the live
gateway a follow-up turn with a title request in between reuses 1 339/1 355
tokens: **171 s → 3.1 s**. P4c landed at 1.03×. Day total on a fresh 384-token
prompt: 65.6 → 44.9 s (1.46×); on a follow-up turn: minutes → seconds.
Next: P6b (per-slot device state, recovers ~10 % decode), P7 (checkpoint the
system+tools prefix so a *new* conversation starts warm), RP4 (GPU
timestamps on the expert group), P5.

**Rev 8 (2026-09-07, night): P6 step 1 measured, P4b declined, P4c gating.**
The single KV slot is what loses Open WebUI its prefix: a title request
between two turns → turn 2 re-prefills (191.6 s, REUSE 0); with
`--kv-slots 4` and the gateway's own routing → 3.2 s (REUSE 1 339/1 355).
Serving switched to 4 slots + `COLI_KDA_GPU=0` (the device holds one KDA
state); P6b (per-slot device state) recovers the ~10 % decode. P4b (LDS
staging) measured 0.91× and was refused by the gate; P4c (vec4 loads) is
gating.

**Rev 7 (2026-09-06, 21:20): P4 landed — bit-identical (no knob), 1.13× at
384 tokens, in service. Day total: 1.44× at 384 tokens, 1.39× at 1 230,
three items, every one bit-identical and gated on the serve path.** The
remaining ms/token at the CPU-recurrence knob: MoE 66 (CPU experts bound),
MLA 34 (attention 27), KDA 23 (recurrence 13). P5 next, then P6/P7 for Open
WebUI; the per-bucket split at the *serving* knob is being taken now.

**Rev 6 (2026-09-06, 20:48): P3 landed — 1.23× at 384 tokens, 1.19× at
1 230, bit-identical, in service.** The CPU half of the MoE bucket was the
binding one (99 → 68 ms/token in the profile). P4 (tiled shaders) is
implemented and gating now. Cumulative since the morning's pristine: 1.28× at
384 tokens.

**Rev 5 (2026-09-06, late): P2 landed — honestly 1.05–1.10× on the serve
path, not the 1.5× the spec projected.** Bit-identical, decode unchanged, in
service. The projection was made from a profile at `COLI_KDA_GPU=0`; at the
serving knob the pristine's fused KDA chain was already cheap and the S-row
GPU matmul re-reads weights per row, so the batching only removed submit
overhead. The structure is right; the gain belongs to P4 (S-tiled shaders,
dense and expert), which is now the top item. Rule: **time profiles at the
serving knob; use `COLI_KDA_GPU=0` for identity only.**

**Rev 4 (2026-09-06, night): P2 implemented and gated twice.** The oracle
is bit-identical and the profile shows 1.18×, but the first serve-path gate
caught two regressions the profile could not see — a decode tax from OpenMP
`if`-clauses (5.4 → 2.1 tok/s) and a 0.65× TTFT from reading the KDA state
back from device memory per chunk — and the chain auto-served the second one
because the gate treated TTFT as informational. Both fixed; the gate now exits
non-zero below `MIN_SPEEDUP` or on a decode drop; p2d pending (§P2 in the
record). Lesson written into P0: *a gate that can pass a regression is not a
gate.*

**Rev 3 (2026-09-06, evening): P0 and P1 are done** — the serve-path
baseline is measured (§P0/P1 in the record), prefix reuse turns out to *work*
on a real turn 2 (408 of 424 tokens reused, 2.7 s instead of 65 s), and the
CANCEL gate is executable and fails as predicted. P6/P7 are re-scoped from
"build prefix caching" to "make the reuse that exists survive Open WebUI".

**Rev 2 (2026-09-06, a few hours after rev 1):** rev 1's diagnosis — "prefill
is streaming-bound, batching cannot help, prefix caching is the only lever" —
was **wrong**, and it was wrong because it was asserted from one sentence in
the record instead of from the profile that already existed. The corrected
diagnosis is below; it changes the item list and the ordering. Rev 1's items
survive as P1 (reuse diagnosis) and P6/P7 (prefix caching); the real ceiling
is materially higher than rev 1 said.

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

Prefill runs at **decode speed**. Consequences, all measured or derived:
- a plain short chat prompt (~30–100 tokens): **10–30 s** to first token
- a prompt carrying Open WebUI's 34 builtin tools (~6 000 tokens):
  **~30 minutes** to first token, every turn

The 34 tools are ~5 600 tokens of JSON schema attached to *every* message. That
is what turned "hi" into a 30-minute request.

## The diagnosis (rev 2) — prefill is decode in a loop

SPEC-PROBE (`ROME-3x7900XTX-2026-09-04.md`) prefilled the same 330 tokens at
chunk sizes 1→16 with the full `[OPTIME]` split. Nobody read it as a prefill
profile. Read that way, in ms per prompt token:

| bucket | K=1 | K=16 | K1/K16 | batches on any normal engine? | why it does not here |
|---|---:|---:|---:|---|---|
| **total (layers)** | 183.6 | 169.8 | 1.08× | | |
| cpu experts | 56.1 | 54.9 | 1.02× | yes, per expert across the rows that chose it | `mlp3_cpu` is called **once per token at S=1** (`glm53.c` ffn_layer, "per ogni token che lo ha scelto") even though the expert's weights were already read once per chunk |
| eg — GPU expert group | 65.6 | 62.5 | 1.05× | yes | the group *does* gather rows per expert (`vrows[]`), but `qmatmul_gate_up.comp` dispatches `(O/8, rows, 1)`: **every row re-reads the whole expert from VRAM** — S rows cost S× one row |
| shared expert | 14.3 | 15.2 | 0.94× | yes, dense | `mlp3_shared` per token, S=1 |
| router | 5.7 | 5.5 | 1.05× | yes | per token, one OpenMP team per token |
| kda.proj | 22.1 | 19.1 | 1.16× | yes, 8 plain matmuls | `kda_layer` loops `for t < tokens`, one fused GPU submit **per token** (`coli_vk_kda_layer(row)` is 1-row by signature) |
| kda.ko | 13.4 | 13.7 | 0.97× | yes, plain matmul | same loop, `KMV` per token |
| kda.step | 13.2 | 11.6 | 1.14× | **no** — sequential recurrence | (legitimate) |
| mla.proj | 6.9 | 5.8 | 1.18× | yes | `mla_layer` loops per token, `vk_batch_mv` at M=1 |
| mla.attn | 17.1 | 17.1 | 1.00× | mostly (sparse indexer selects per token) | scalar C per (token, head) |
| hc+norm | 9.0 | 8.9 | 1.01× | yes | per-token loops |

Decomposition at K=1: **cpu-expert stream 31 %, sequential recurrence 7 %,
everything else 62 % — and that 62 % (114 ms/token) is ordinary matmul and
attention work that batches on every other engine.** Perfect batching at K=16
would take it to ~7 ms/token; it stayed at 103.

The serve path runs at the default `GLM53_PREFILL_CHUNK=128` (not set in
`~/start_glm53.sh`; SPEC-PROBE never measured above 16). The chunk only
dedupes the *disk/RAM read* of expert weights — real (GPU expert-calls fell
75 628 → 47 921 from K=1 to K=16 in the `[PROF]` line) but irrelevant, because
**every stage costs per row what it costs per token in decode**, and the rows
are the same at every chunk size.

Three things rev 1 got wrong, stated so they are not repeated:
1. "GEMM batching buys almost nothing here" — it buys nothing *because the code
   does not batch*. The GPU dense matmul already takes `S` rows
   (`coli_vk_matmul(..., S, I, O, gs)`); `mv()` hard-codes 1. The CPU int4
   kernels already take `S`. `quant.h:1317` already carries a 1×4 row-tiled
   int4 kernel written for "the prefill union delivers nr=2..16 rows per
   expert" — on the Qwen engine. None of it is wired into glm53's prefill.
2. "Cold experts do not dedup across tokens" rested on `[PROF] cpu n=35252`,
   which counts (token, expert) *pairs* — constant by construction. It proved
   nothing. The `eg experts=` counter does show dedup.
3. "Faster kernels cannot fix TTFT" — the kernels are not the problem; the
   call shape is.

**What the corrected ceiling looks like** (derived, to be measured by P0):
batching the dense stages (P2) removes most of ~70 ms/token; row-batching
the CPU expert compute (P3) and S-tiling the GPU expert shader (P4) attack
the 56 + 65 (overlapped) MoE ms/token; the recurrence (13) and the sparse
attention (17) remain. A floor around 60–80 ms/token is plausible → **2.5–3×
on every prefill, including the first one of a new prefix**, on top of which
prefix reuse (P6/P7) makes repeated prefixes ~free. Rev 1 offered only the
latter.

## The prerequisite nobody built (this comes first, always)

**P0 — a prefill oracle, a serve-path TTFT harness, and an executable gate.**
The root cause of 2026-09-06 is that the engine was validated only as a
decode-throughput benchmark, never as an interactive server. The gates in
`ROADMAP-2026-09.md` were prose; "incomplete gates" cost this project days.
Here a gate is a **script that exits non-zero**, and an item is done when the
script passes, with its output pasted in the commit body. P0 delivers three
instruments in `tools/hot-expert/`:

1. `prefill_profile.sh` — the **diagnostic**: teacher-forcing CLI run
   (`--ids`, no `--greedy`, so the whole run *is* prefill) with
   `COLI_TIMERS=1` at the serve chunk size, on a fixed 600-token and a fixed
   3 000-token id list, printing the table above in ms/token. Every P item is
   measured against its row.
2. `ttft_serve.py` — the **gate instrument**: drives the real serve protocol
   through `openai_server.Engine` (the class the gateway uses — the pattern
   `tworeq.py` already proves works), measures first-token latency separated
   from decode at 30 / 300 / 3 000 / 6 000 prompt tokens, and runs the
   multi-turn check (submit A, then A+B: was the prefix reused?). Residency is
   asserted before every run and the engine is verified idle (`pgrep`) before
   and after. Refuses to run otherwise.
3. `prefill_gate.sh <pristine-binary> <candidate-binary>` — the **gate**
   (exit 1 on an oracle failure, **exit 3 when the candidate's median TTFT
   speedup is below `MIN_SPEEDUP` (default 1.0) at any size or its decode
   sanity window drops more than 10 %** — added after p2c passed the oracle
   at 0.65× and a chain served it):
   (a) teacher-forcing argmax **identical at every position** over the
   600-token list and (b) last-position logits cosine ≥ 1−1e-4 / argmax equal
   over the 3 000-token list, both vs pristine; (c) `ttft_serve.py` on both
   binaries, twice each, printing the deltas. Exit 0 only if (a)(b) pass; the
   deltas are the item's number.

A prefill item validated on the `--greedy` CLI path is not validated (the G16
CANCEL fix passed a `--greedy 64` oracle that never executed the serve loop it
changed, and shipped broken). Half a day to a day. Opus spec, Haiku/Sonnet
build.

## The items

Effort and effect are honest, not optimistic. Ordering is by
(expected effect × confidence) / effort, with the instrument first and the
cheapest possibly-decisive diagnosis second.

**Order after rev 16:** nothing. P5b landed 2026-09-08 21:19 and was the last
open item; the upstream-`dev` merge landed the same morning. The next prefill
item, if there is one, has to come from a fresh profile at the serving knob —
the standing ranking at 3 462 tokens is now CPU experts 51.6, `eg` 54.2,
`mla.attn` 22.3, `kda` 10.5, `router` 1.4, i.e. the MoE path, which §RP4 has
already shown is 92 % CPU inside a GPU-named timer.

**Order after rev 15 (kept for the history):** P5b (the MLA weighted pool: one
walk of the latent instead of 64 — the item P5's own microbenchmark uncovered).
The upstream-`dev` merge landed 2026-09-08; everything else on this table is
done.

**Order after rev 7 (kept for the history):** P5 (the recurrence's S per-token
submits into one command buffer per layer per chunk; the sparse attention,
27 ms/token and growing with context; a vectorised router behind a knob), then
P6/P7 for Open WebUI — the 24.5-minute tool prompt is now ~17 minutes and only reuse turns
it into seconds. Every projection is checked against a profile taken at the
serving knob (rev 5's rule).

| id | item | mechanism | expected effect on TTFT | effort | tier | gate (executable) |
|---|---|---|---|---|---|---|
| **P0** ✅ | Oracle + serve-path TTFT harness + gate script | `ttft_serve.py`, `prefill_profile.sh`, `prefill_gate.sh` (commits `65f0c8a`…`3777078`) | **baseline (pristine): 21 tok 3.1 s · 384 tok 66.7/65.6 s (171 ms/tok) · 1 230 tok 241.5/241.2 s (196 ms/tok)**, reproducible within 1 %; long rows on the P2 binary: **3 562 tok 845 s · "hi" + 34 tools (5 439 tok) 1 470 s = 24.5 min** | done | — | met: ±1 % at three sizes; pristine-vs-pristine gate run in the record |
| **P1** ✅ | Why prefix reuse "does not fire" | measured with the engine's own `REUSE` line: a real turn 2 `[A, reply, B]` reused 408/424 tokens, **TTFT 2.74 s vs 65.45 s**; the rev-1 "same prompt twice" test was the *regenerate* case, which cannot reuse by design (the KDA state cannot rewind past prompt + reply) | reuse already works for a well-formed conversation; what Open WebUI loses it to is the single slot (side requests evict the session) and per-turn prefix changes | done | — | met; the Open WebUI-specific cause is P6's first step, with `GLM53_VERBOSE=1` now in `~/start_glm53.sh` so the server log carries REUSE lines |
| **P2** ✅ | Batch the dense stages (`P2-BATCH-DENSE-SPEC-2026-09.md`; `b9c193e`…`0b75c42`; **landed 2026-09-06 18:58**: oracle bit-identical, serve-path TTFT 1.10× / 1.05× / 1.06× at 21 / 384 / 1 230 tokens, decode unchanged — §P2 in the record) | in `kda_layer` / `mla_layer` / `ffn_layer`: gather the chunk's rows and call the existing S-row kernels once per weight per chunk (`coli_vk_matmul` S>1 for kq/kk/kv/kfa/kb/kga/kfb/kgb/ko, qa/kva/iwk/ikpg/iwp, qb/iwq, shared gate/up/down; router as one matmul); the KDA recurrence stays a per-token loop between the batched projections and the batched ko. The GPU shader is per-row independent (`s = WorkGroupID.y`), so rows are bit-identical to today | ~70 ms/token → ~10; **~1.5×** on prefill of any prefix | 2–4 days | Opus | `prefill_gate.sh` (a) bit-identical argmax; TTFT delta at 600 and 3 000 tokens ≥ 1.3×, both runs |
| **P3** ✅ | Row-batched CPU expert compute (`c695a48`; **landed 2026-09-06 20:48**: bit-identical, serve-path TTFT 1.05× / 1.23× / 1.19× at 21 / 384 / 1 230 tokens vs P2, decode unchanged; profile ffn_moe 99 → 68 ms/token — §P3 in the record) | `coli_i4_rows4` decodes a weight row once per four activation rows; `cpu_expert_rows` runs each CPU expert once on the chunk rows that chose it | done | — | — | met |
| **P4** ✅ | S-tiled GPU matmul shaders — experts **and** dense (`f0197d0`; **landed 2026-09-06 21:20**: teacher-forcing identical and logits max-abs 0 — bit-identical, no knob; serve-path TTFT 1.11× / 1.13× / 1.11× at 21 / 384 / 1 230 vs P3; kda 31.7 → 22.7 ms/token in the profile — §P4 in the record) | `qmatmul_gate_up.comp` / down / `qmatmul.comp`: one weight read per output row applied to all S rows (register tile over rows), instead of `(O/8, rows, 1)` re-reading the matrix per row. P2 showed the dense path needs it too: batched kda.proj+ko still cost 18 ms/token for ~2 of arithmetic | eg per token drops with the chunk's dedup (×~1.6 at K=16, ×~3–4 at K=128) — the MoE bucket's floor moves | 2–4 days | Opus (shader) | `prefill_gate.sh` (b) logits within tolerance — the reduction order changes, so this ships behind a knob; TTFT delta |
| **P5** ✅ | The sequential remainder (`perf/p5-sequential`, **in service 2026-09-07 23:33**, binary `5dd26f65…`, serving script unchanged) | three sub-items, three knobs, all default ON and all bit-identical: **P5.1** `COLI_MLA_HEADVEC` — the MLA attention score pass with the 64 heads in the SIMD lanes over a transposed query slab `qT[d][h]`, each lane summing over d in the reference's order (an empty `asm` on the product stops gcc contracting mul+add into an FMA); **P5.2** `COLI_KDA_ROWS` — `kda_step.comp` gains a `tok` push constant and `coli_vk_kda_step_rows` records the chunk's S dispatches into ONE command buffer with write→read/write→write barriers, so the recurrence stays sequential and the S−1 round trips go; **P5.3** `COLI_ROUTER_LANES` — the router dot with the chunk's TOKENS in the lanes over `xT[d][t]`, which needs no reordering, so the knob this row used to budget for is not needed | measured at the serving knob, 3 462 tokens: **174.4 → 146.4 ms/token (1.19×)**, `mla.attn` 69.8 → 48.2, `router` 5.2 → 1.4, `kda.step` 5.8 → 3.8; at 781 tokens 120.5 → 109.2 (1.10×); serve path **1.06× / 1.15× / 1.25×** at 21/384/1 230; `max_abs=0` at 782 positions at both KDA knobs | done | Opus | `p5_gate.sh` exits 0 on all four steps (§P5 in the record) |
| **P5b** ✅ | The weighted pool — and the answer is the per-thread scratch, not the 64 walks (`perf/p5b-mla-pool`, **in service 2026-09-08 21:19**, binary `65d7d70d…`, serving script unchanged) | one knob, `COLI_MLA_POOL`, three settings, all bit-identical: **0** the code P5 left behind; **1 (default)** `pooled`/`score` per-thread slices 64-byte aligned with their stride rounded to a cache line — they were 2 048 B and 8 204 B from a 16-byte-aligned `malloc`, so neighbouring threads shared the boundary lines and the pool wrote them `used` times per head; **2** that plus the blocked pool (one walk of the latent for all 64 heads, d-tiles as the parallel axis, `acc[dt][H]` per tile in L1, eight slots folded at a time) | measured at the serving knob: `mla.attn` **51.6 → 22.5** ms/token at 3 462 tokens and **23.0 → 13.7** at 781; TOTAL 147.3 → 118.1 (**1.25×**) and 108.7 → 99.2 (**1.10×**) vs the same binary knobs-off, 1.30× and 1.11× vs the pristine binary; serve path **1.01× / 1.07× / 1.15×** at 27/390/1 236 tokens; `max_abs=0` at 782 positions at both KDA knobs. **Mode 2 is neutral in the engine (+0.2 ms/token at 781, −0.3 at 3 462) although it is 3.6× on the pool in isolation** — once the false sharing is gone the per-head pool runs at L3 bandwidth and the layer's latent is L3-resident — so it is not the default | done | Opus | `p5b_gate.sh` exits 0 on all four steps at `MIN_SPEEDUP=0.97` (§P5b in the record); the first run at 1.0 exited 3 on the 27-token row alone |
| **P6** ✅ | Make reuse survive Open WebUI (`cb9c11e`; **in service 2026-09-06 23:30**: `--kv-slots 4`, `COLI_KDA_GPU=0`; `tworeq` identical across slots at both knobs; live-gateway side-request case turn 2 **171 s → 3.1 s**, REUSE 1 339/1 355 — §P6 in the record). Confirmed on the owner's own Open WebUI on 2026-09-07 with tools and side tasks off: follow-up turn REUSE 32/41, ttft 5.3 s, 180 tokens at 4.4 tok/s. Stability *with* memory context and the tool block on is P7's question | | | | | met |
| **P6b** ✅ | Per-slot KDA device state — spec `P6B-KDA-SLOT-STATE-SPEC-2026-09.md`, built on `perf/p6b-kda-slots`, **in service 2026-09-07 16:50** (binary 53ccbb42…, `--kv-slots 4`, `COLI_KDA_GPU=2`) | `G.kda_slot[layer][slot]` state/window on dev0 (34 × 4.58 MB = 156 MB per slot), `coli_vk_kda_pool_init` allocates every slot's set BEFORE `vk_preload_tier` (after it, the 624 MB would eat dev0's 3.0 GB reserve); `coli_vk_kda_init/step/layer/sync/upload` and `GSession` take the slot; `slots_init` forces the CPU recurrence only when the pool is short. `coli_vk_kda_sync` goes through a `vkCmdCopyBuffer` into HOST_CACHED staging instead of a memcpy from write-combined memory | measured: decode at 4 slots **5.039 → 5.702 tok/s (+13.2 %)**, within 0.996× of the 1-slot GPU number; bit-identical at one slot (`max_abs=0`); live gateway turn 2 REUSE **1 339/1 355 in 3.08 s** with no `forcing` line; the state sync **~14 MB/s → ~2 262 MB/s** (370 MB checkpoint, 69 ms), which is what lets P7 capture at the GPU knob; price 1 296 → **1 248** preloaded experts (48, 3.7 %) | done | Opus (spec: Fable) | `p6b_gate.sh` exits 0 on all six steps (§P6b in the record) |
| **P7** ✅ | Checkpoint the stable prefix (tools + base system) **and pin the per-turn context block** — spec `P7-PREFIX-CKPT-SPEC-2026-09.md`, built on `perf/p7-prefix-ckpt`, **in service 2026-09-07 12:25** (binary 19e28e72…, `GLM53_PREFIX_CKPT=1 COLI_PREFIX_PIN=1`, `--kv-slots 8`, `COLI_KDA_GPU=0`) | engine: the session's span list (11 DSA layers' latent/ikeys/igates rows + 34 KDA states/windows; **370 MB at 6 328 tokens**) copied at the prefix boundary the gateway sends in the 8th SUBMIT field (`src=hint`, verified against the prompt's own ids) or at the LCP of successive fresh prompts, persisted under `<SNAP>/.coli_ckpt`. Gateway: `COLI_PREFIX_PIN=1` keeps the `<memory_context>` block byte-identical across a conversation's turns | measured: new conversation with the 34-tool block **1 452 s → 5.30 s** (engine) / **8.93 s** (live gateway, restored from disk after a restart); re-ranked memory block **126.82 s → 3.00 s**, 17 tokens prefilled instead of 1 005; off (`GLM53_PREFIX_CKPT=0`) the engine is bit-identical to the pristine one | done | Opus (spec: Fable) | `p7_gate.sh` exits 0 on all five steps (§P7 in the record); live gateway `--prefix-ckpt` and `--pin-block` PASS with `CKPT hit` and `[pin] … hit` in the server log |
| **P7b** ✅ | Capture beyond a RESTORED checkpoint — `perf/p7b-capture-after-restore`, **in service 2026-09-09 08:49** (binary f68d1cac…, serving script unchanged) | `serve_one` planned a capture only when `shared == 0`, so a partial checkpoint was terminal: the engine restored it and re-prefilled the same tail on every new conversation, with no `CKPT plan` line ever again. Plan and hint now also run when `shared` came from a RESTORE (not from slot reuse — that session is continuing a conversation and must not be split), only strictly beyond `shared` and never for a prefix already stored, and the LONGER of plan and hint wins (the hint is the gateway's known-stable boundary; the LCP was the shorter one in the case that hurt). Verbose line gains `src=lcp|hint` and `after-restore` | measured: live UI-shaped new chat **380.14 s → 20.98 s** (REUSE 2 163/4 424 → **4 308/4 428**, 18.1×) with `CKPT hit prefix=2163` → `CKPT plan prefix=4308 src=hint after-restore`; engine case 181.98 s → **2.92 s** (62.3×) with the second capture at 2 023 after a 1 527 restore; off (`GLM53_PREFIX_CKPT=0`) bit-identical to the pristine binary (`max_abs=0`, 782 positions, TTFT 1.00×/1.01×/1.01×); serve-path oracle text-identical, `cosine=0.9999986 max_abs=0.04251` (P7's one-split control on the same fixture: 2.301e-05) | done | Opus | `p7b_gate.sh` exits 0 on all five steps and the live UI turn is a gate in `p7b_chain.sh` (§P7b in the record) |
| PMLOCK (parked; this row used to be called P9, which is now the conversation ledger) | Pin mapped expert slots (`compat_mlock` at fill, unlock at evict) | upstream's datapoint on PR #1324 (2026-09-06): below ~1:1 model:RAM the kernel page LRU and the engine slot LRU disagree and a hit re-faults inside the matmul; rome sits near 1:1 once the engine's anon memory is counted, but shows ~0 major faults per request at 99 % residency — a robustness item (deterministic residency), not speed | none measured here; `ttft_serve.py` now prints majflt per request — build this only if that number stops being ~0 | 1 day | Sonnet | majflt per request stays 0 across a full gate with no `--warm` |
| — ✅ | Housekeeping (not this track): merge upstream `dev` — **done 2026-09-08 01:11**, `dev` `1ccee43` in service as `5ed096a6…` | merge base `12a5c46`, 149 ahead / 125 behind. Three files conflicted, **eight hunks**: the two mapping hunks took `dev`'s #1325 `st_map_shard_range` (`eabeb9a`, the fork's own PR upstreamed) while keeping the mapping ON by default (`st.h::COLI_MAP_EXPERTS_DEFAULT`), `g_map_all`'s LRU sizing, the one-thread `expert_map_init` (`st.h` maps lazily and its per-fd table is unsynchronised) and both prefaults; the three new hunks are `dev`'s per-turn `PROF`/`HITS` clocks against the fork's G10/P2.3 parallel hyper-connections and were unioned; `family_registry.py` kept ours. #1350 (`RssAnon`) is in and is upstream's precondition for that default; #1321 is still not in `dev`. `DEV-MERGE-NOTE-2026-09-07.md` has the full table | four qwen38 C tests + chat-template pin + 790 python tests green; `max_abs=0` at 782 positions; TTFT 0.99×/1.00×/0.99×; `tworeq` 4 slots IDENTICAL at both knobs, 0 forcing lines; `--prefix-ckpt` 91.8×, `--pin-block` 994/1 010; `rome_bench` paired 1.004×/1.003×/1.013×; `[MAP]` `copy=0`, `majflt` 0, MemAvailable unmoved | done | Opus | met — `~/bench/devmerge_chain.sh` exits 0 on all six steps (§"Upstream dev merge" in the record) |
| **P4b** ✗ / **P4c** ✅ | Tile shader variants: LDS staging measured 0.91× and was refused by the gate; vec4 x loads passed at 1.03× (neutral profile) and are in service. **RP4 settled why neither moved the expert group: the 58 ms/token bucket is 92 % CPU experts overlapping inside the same timer, GPU busy on the busiest device is 1.97 ms/token, and the GPU makes the engine wait 0.068 ms/token.** There was nothing there to move; a shader that ran in zero time would have saved 0.06 % of the token. P4b's premise (get the activations out of global memory) was right — the tile re-reads 67.1 MB of activations per row, 4.7× its whole weight stream — and its implementation wrong; P4c's premise (too many loads, not too many bytes) was wrong. **No further expert-shader work on this track** | | | | Opus | `prefill_gate.sh` |
| **RP4** ✅ | GPU-side timestamps on the expert group (`COLI_VK_TIMESTAMPS`, off by default; `perf/rp4-vk-timestamps`, **in service 2026-09-07 20:00**, binary `325c9c7a…`, serving script unchanged) | per-device `VK_QUERY_TYPE_TIMESTAMP` pool written at the top of the expert group's command buffer, between the gate+up and down phases and after the down phase (`=1`), plus per expert (`=2`, serializing and honest about it); `VK_EXT_calibrated_timestamps` (asked for only when the knob is set) maps the device clock onto `CLOCK_MONOTONIC`, which splits the CPU-side `eg` wait into queue latency / GPU busy per phase / fence tail, and yields `gpu_late` — the only part of the wait the GPU owns. `prefill_profile.sh` prints it as a row group under `eg` | measured at 781 and 3 462 tokens, three repeats each: GPU busy **1.97 / 0.93 / 0.89** ms/token on dev0/dev2/dev3, `gpu_late` **0.068** ms/token (0.12 % of `eg`), `queue_lat` 0.07–0.25; per-expert 232–235 µs (dev0, 9.3 rows) vs 84–86 µs (dev2/dev3, ~5 rows); a **one-row** expert costs 2.7–3.4× a two-row one because `vk_tile_ok4` requires `S > 1` and the per-row shader hits an LDS bank conflict. Prefill wall identical off / `=1` / `=2` | done | Opus | `rp4_chain.sh`: `prefill_gate.sh` OFF **rc 0** (bit-identical, TTFT 0.99×/1.00×/1.00×), `tworeq` 4 slots at `COLI_KDA_GPU=2` IDENTICAL with 0 forcing lines, `prefill_gate.sh` ON **rc 0** (bit-identical, TTFT 1.00× at every size) — §RP4 in the record |
| **P8** ✅ | **The reply pin** — `perf/p8-reply-pin`, **gated and in service 2026-09-09 21:24**, superseded by P9's ledger (it is now the `COLI_LEDGER=0` fallback path) | Open WebUI stores an assistant turn as its VISIBLE `content` only (`webui.db`: `content`, `done`, `model`, `output`, `usage`), so the `<think>` block the engine generated never comes back; the rendered prompt stops matching the tokens the KV slot holds and the KDA state cannot rewind past prompt + reply, so the turn re-prefills from the checkpoint. Amplified by the `dev` merge (#1327/#1278): `render_chat_glm53` now always ends `<|assistant|><think>`. Fix, gateway-side only, `COLI_REPLY_PIN` (default 1): remember the RAW generated text per conversation — before the reasoning splitter, before any strip — and render it back VERBATIM when a later request carries that turn's visible content, so the prompt reproduces the engine's own tokens. Reassembling `<think>{reasoning}</think>{content.strip()}` is what breaks it, so the pinned path never does. A reply truncated inside `<think>` has no visible part, so each remembered turn carries its INDEX and an empty content is restored only against an empty remembered one at the same index | measured on the SERVED pristine gateway, through Open WebUI's own backend: turn 1 **prompt 4 718 gen 29** (6 chars visible), turn 2 **REUSE 4 419, ttft 54.07 s** against an expected **4 747 and ~2.7 s** — the bug, reproduced on demand. Engine untouched: built binary byte-identical, `teacher_forcing IDENTICAL (782 positions)`, `cosine=1.0000000 max_abs=0`, TTFT 0.97×/1.00×/1.00× (the 0.97× is a file compared with itself). 16 offline tests + 207 existing gateway tests green. Live: with the pin the follow-up reuses `prompt + gen` **3/3** (2.77 / 2.81 / 2.77 s), without it **0/3** (3.51 / 52.22 / 53.28 s); the browser matrix's turn-2 rows all reuse `prompt + gen` | done | Opus | `p8_gate.sh` rc=0, `accept_live.sh` PASS, `accept_ui.sh` first token 3.82 s / 3.57 s — §P8 and §"P8 on the user's path" in the record |
| **P9** ✅ | **The conversation ledger** — `perf/p9-ledger`, **gated and in service 2026-09-10 01:44**, binary `34357e49…` | Prefix reuse rests on a contract nobody enforced: the transcript the client re-sends must re-render to exactly the token sequence the engine holds. Open WebUI broke it three times in three days (P7 21.6 s, P7b 362.94 s, P8 54.07 s) and every break was invisible except as latency. The gateway now keeps the renderer's OWN pieces per conversation plus the raw generation, renders later turns from that record, and checks `new_prompt.startswith(ledger.prompt + ledger.generated)` before submitting; on failure it logs `ledger=broken` and falls back to the client's rendering. The conversation also OWNS its KV slot for its life, replacing `conversation_cache_slot`'s hash-modulo routing. `COLI_LEDGER` (1), `COLI_LEDGER_MAX_CONV` (64), `COLI_LEDGER_STRICT` (0); at 1 P7's pin and P8's reply pin are off. Engine: one trailing `reused` field on `DONE … STAT`, append-only. Deliberate departures from the spec, both recorded: a changed tool list RESETS rather than pinning the old block (rendering a declaration the client withdrew would change the answer the user reads), and a `shorter`/`diverged` turn replays NOTHING (it buys no reuse — the slot holds more tokens than the shorter prompt — and it would cost the byte-identity that makes divergence checkable) | three behaviours x three seeds x three arms: ledger ON **9/9**, today's path 6/9, no mechanism 0/9; divergence byte-IDENTICAL to a cold render 3/3 with `ledger=reset` each; `expect_reuse == engine_reuse` on every continuation of the whole chain, 0 MISMATCH, 0 broken; engine bit-identical, TTFT 0.99x/1.00x/1.00x; `tworeq` IDENTICAL at both knobs | done (steps 5-6 owed) | Opus | `p9_gate.sh` rc=0, `accept_live.sh` PASS — §P9 in the record. **Owed:** `p9_ui_multiturn.sh` (ten browser turns), `accept_ui.sh`, `ui_matrix.sh --sizes 0,500,2000` |
| PCAP | Capacity (cross-ref; this row used to be called P8) | int3 experts (`ROADMAP-2026-09.md` 4h / G15) | partial; tracked on the main roadmap | weeks | Opus | its own gate |

Rev 1's "P4 — 4-accumulator BF16 prefill kernel (Q5)" is dropped from this
track: it is a Qwen-engine BF16 kernel, and the finding above is that the call
shape, not the kernel, is the problem.

## The honest ceiling, stated up front

- P2–P5 speed **every** prefill, including the first of a new prefix, by an
  estimated 2.5–3× (to be measured). A 6 000-token tool prompt goes from ~30 min
  to ~10 min: better, still not interactive. **P6/P7 remain necessary for
  Open WebUI with tools**; P2–P5 make everything P6 cannot cache tolerable.
- A 32k-token document summarised cold is ~2.7 h today and ~1 h after P2–P5.
  Only capacity (PCAP / more VRAM) changes that, and only partly.

## The alternative this track must not bury

Recorded so a future session weighs it honestly: **this box may not be the
right home for low-latency interactive use with large prompts.** Colibri's
strength is running a model too big to fit in VRAM at all. A model that
*fits* gives interactive tools that respond in seconds today, with none of
P0–P7. If the requirement is "tools that work," that is the faster answer; if
it is specifically "GLM-5.3 with tools on this hardware," this track is the
way.

## Discipline this track inherits from 2026-09-06 (non-negotiable)

1. **Validate on the path the user uses.** A prefill/serve item proven on the
   `--greedy` CLI path is not proven. `ttft_serve.py` or nothing.
2. **Measure the user's request, not a proxy.** Traffic from inside the Open
   WebUI container shares its source IP with the app — read the app's own
   record (its DB, its logs).
3. **Assert residency before every measurement.** A cold cache alone turned a
   2.8-second "hi" into 12 minutes. `ttft_serve.py` prints residency before
   every request and refuses below 96 % — 100 % is unreachable with an engine
   up (a fresh process evicts ~3 %; the gateway's sits at 91.6 %), so a gate
   compares both binaries at the *same* printed number, not at a fiction.
4. **One in-flight request while diagnosing.** An orphaned generation (a
   timed-out probe with the CANCEL bug unfixed) blocks the queue and poisons
   every later number. Kill hard, confirm `pgrep -x glm53` empty, then measure.
5. **No claim of "it works" without an end-to-end pass on the real path**,
   shown to the user, reproducible.
6. **Read the profile before asserting the bottleneck.** Rev 1 of this file is
   the counter-example.
7. **Measure the request after the one under test, on the user's client.**
   P7's gates proved every restore and never looked at what the restoring
   request left behind; the owner paid 380 s per new chat until he said so
   (P7b). `accept_live.sh` is that check, `serve_candidate.sh` makes it the
   last step of every chain, and a daily canary keeps the tool-block
   checkpoint current. Added at rev 19.

## Closed dependency: CANCEL (fixed 2026-09-09, binary f41fc5cc…)

For two days this section read "still unfixed": `glm53` ignored CANCEL, and a
1 230-token prompt cancelled at 5 s held the engine **251.3 s** — the whole
prefill and the whole generation. Prefill work made it worse, because a
17-minute tool prompt a user gives up on held the engine for 17 minutes. The
G16 attempt failed for lack of a serve-path oracle.

It is fixed. Confirmed in **12.9 s** in the prefill phase and **13.9 s** in the
decode phase, twice each; the next request is served in 2.6 s; a cancelled
prefill **resumes** at exactly the position it stopped, with the same greedy
text as an uncancelled run; live, an abandoned `curl` costs 13.18 s instead of
132.32 s and the request behind it 7.13 s instead of 126.35 s. Bit-identical,
`tworeq` IDENTICAL at both KDA knobs, P7 capture/restore unmoved, and the poll
itself measured free (1.02×/0.99×/0.99× TTFT against a binary that has no poll).
The whole table is in §CANCEL of the record.

Three things this leaves for whoever works here next:

1. **The engine's concurrency is the pipe.** The gateway admits up to
   `kv_slots` requests at once and writes each SUBMIT as soon as it is admitted;
   the engine serves one and the others wait in the pipe (now in a small FIFO
   inside the engine, read by the same parser). Anything that reads stdin
   mid-turn has to consume whole frames — the frames are byte-counted, so
   skipping a header desyncs everything behind it.
2. **`--cancel-phase` is the check that keeps this honest.** It caught three
   instrument bugs before it caught anything in the engine, including a
   decode-phase run whose turn had ended before the cancel was due. Ask for the
   phase; do not infer it from arithmetic about prefill speed.
3. **A live step that only prints a number is not a gate.** The gate passed
   (rc=0) on a binary whose live disconnect behaviour was still broken, and the
   chain served it. `cancel_chain.sh` now reverts on the live bound too.
