# Project handoff, 2026-09-20 18:30 CEST — the whole history, every bug, what is left

Written by the Fable session of 2026-09-20 (afternoon), at the owner's request,
for whoever continues: a person or a session on any model. It replaces nothing:
the numbers' home is the record (`ROME-3x7900XTX-2026-09-04.md`), the
decisions' home is the three plans, the rules' home is `CLAUDE.md`. Where this
file and those disagree, they win. `HANDOFF-2026-09-20.md` (written the same
morning) has the blow-by-blow of the Franken track F0–F10 and is still
accurate for that; this file is wider and later.

Every number here is copied from a gated run recorded in the record or a
commit body. Anything that is arithmetic and not a measurement says so.

---

## 0. Read this first: what the last session got wrong, and what it cost the owner

The owner's verdict on the session that wrote this file (Fable 5.1, the most
expensive tier, 2026-09-20 afternoon): too expensive for what it delivered.
Every mistake below was made or left unchecked by the orchestrator -- the model
that wrote the plans, the briefs and this file -- not by the cheaper agents it
directed. They did what they were told. He hit his plan's usage limit
during it. The facts behind that, so the next agent does not repeat them:

| mistake | what it cost | what to do instead |
|---|---|---|
| **Broke the owner's daily service five times in one afternoon.** Every chain launched that day restarted the gateway, passed `accept_live`, and then its exit trap killed the engine it had just served. The pattern was copied from an older chain without reading its trap | ~1 hour of a gateway that answered 500 to every chat, in five windows, the longest 30 minutes. Found only at the very end, by `accept_ui.sh` | Read a chain's `on_exit` before launching it. **After any chain has fully exited, send one real chat request.** `/v1/models` = 200 and `pgrep -x glm53` both lie (a killed engine is a zombie and still matches) |
| **The same outage had already happened on 2026-09-16** and was written down in `run_chain.sh` and the watchdog. The session did not check that the guard added then actually worked | the guard was blind both times | When a guard exists for a failure, test the guard against the failure, do not trust its comment |
| **Commissioned F3 step 2a on a projection instead of a measurement.** Plan rev 36 said "~27 tok/s by arithmetic"; the arithmetic assumed a GPU call gets cheaper when the work is smaller. It measured 22.4 tok/s, +2.0 % against a +20 % gate. The number that refutes the projection (0.25 ms per GPU round trip) was already in the session's own step-1 profile | 224k Sonnet tokens, two gateway stops (~20 min), for a negative result | **Step 0 is a measurement, never arithmetic.** A ten-minute microbenchmark of one `coli_vk_matmul` call at the two real shapes would have killed the item for free. `CLAUDE.md` and the older handoff both said this; it was skipped |
| Ran two `accept_ui.sh` checks on top of each other (launched the second while the first hung on the dead engine) | both results void, ten more minutes | one acceptance run at a time, and look at why the first one hangs before starting another |
| Plan revs 33 and 34 (previous session, same day) asserted two things nobody had checked: that 8-bit needed "a new int8 tier kernel" (the shaders already had it) and that "the per-op submit model" was what separated Colibri from hipFire (V1's own record said tier submits were 2 % of the token) | would have sent F3 to Opus for a kernel that existed | grep the code and the record before writing a claim into a plan |
| Left build trees behind on the rig. **This is the orchestrator's failure, not the subagents':** every Fable session since 09-09 wrote briefs that said "work in a rig checkout" without saying how or when to remove it, Fable's own Franken plan told agents to use "its own clone", and no Fable session ever looked at what its agents left on the owner's machine. Every item made a `~/src/colibri-<item>` clone or worktree per item and none removed them: 37 in `~/src` (3 GB) and three in `~`, three of them from that afternoon | the owner found his `~/src` buried in them | a worktree, not a clone; remove it and its sync branch when the item lands (`CLAUDE.md`). All 40 were merged work and were removed 2026-09-20 evening; the one uncommitted diff was saved to `~/bench/colibri-g5.uncommitted.diff` |
| Long status messages and option lists when the owner had asked for one decision and the work | his time, and his patience | decide, do, report the number |

**Spend that afternoon:** three Sonnet subagents, 777k tokens (step 0 208k,
step 1 345k, step 2a 224k), plus the Fable orchestrator for about four hours
(its own token count is not visible from inside the session; it was the larger
part of the bill, and it is what ran the plan into its limit). **What it
bought:** the finding that int4 costs real quality on Qwen3.6 (KL 0.0316, top-1
92.96 %), the 8-bit model 100 % resident on two cards at int4's speed, a
measured proof that piecemeal GPU offload of the trunk is worthless on this
backend, and the fix for an outage pattern that had been live since 09-16.
**What it did not buy:** any speed the owner can feel. GLM-5.3 is exactly as
fast tonight as it was at noon.

## 1. What this project is

A fork (`rcklomp/colibri`, canonical on Gitea, mirrored to GitHub) of
JustVugg's **colibrì**, a pure-C inference engine for very large
mixture-of-experts models that do not fit in VRAM: experts live in host RAM
(or on disk) and only the hot ones sit on the GPUs. The fork exists to make
that design useful on one machine, "rome":

- EPYC 7F32 (8 cores, Zen 2, AVX2, no AVX-512), 247 GiB RAM (= 265.6 GB),
  three RX 7900 XTX (24 GiB each, RADV/Vulkan, no hardware FP8), one NVMe.
- Two of the cards share an upstream PCIe link (3-card host→GPU stream
  61 GB/s, one card alone 28).

The owner is not a developer. He uses the box daily through Open WebUI
(port 3000) against the fork's OpenAI-compatible gateway (port 8081), which
serves **GLM-5.3-Flash, int4-g64, 182 GiB**, with `glm53`. He is on a fixed
Claude plan and hits its limits; he wants one decision and the work, not
options.

The fork is 579 commits ahead of `upstream/dev` and deliberately no longer merges
upstream wholesale (`UPSTREAM-POLICY-2026-09-15.md`): fixes come in by
cherry-pick, ours go out as PRs through `upstream_lint.sh` (#1521, #1524 sent).

## 2. State right now (verified 2026-09-20 16:10 UTC)

| | |
|---|---|
| Served model / binary | GLM-5.3-Flash, `glm53` sha256 `189fd9451c8e1c2e` (`~/bench/glm53.f9a`), in service since 12:13 UTC; **unchanged by today's F3 work** |
| Gateway env (`~/start_glm53.sh`) | `GLM53_VK_SWIGLU_CLAMP=1 COLI_PREFILL_STREAM=1 GLM53_PREFILL_CHUNK=512 GLM53_MLA_ATTN_GPU=1 GLM53_MOE_ONE_TEAM=1 GLM53_I4_FAST=2 COLI_KDA_GPU=2`, prefix checkpoints + pin, `--max-tokens 4096 --kv-slots 4`, tier caps 1695/1695 |
| Acceptance | `accept_ui.sh` PASS 18:10 CEST (first token in the browser 2.07 s), `accept_live.sh` PASS 16:10 UTC, engine alive, rig lock free |
| Branch | `hot-expert-tier`; Mac = Gitea (`origin`) = rig; `doc_currency.sh` PASS |
| Plan revs | Franken plan rev 37, prefill roadmap rev 38, decode roadmap closed (as of 09-20; on 2026-09-22: plan rev 64, prefill roadmap rev 46, design rev 12) |
| Rollback | `serve_candidate.sh` with `glm53.f8` / `.f7` / `.f2` / `.f2base` and the matching `~/bench/start_glm53.sh.pre-*` |

What the owner gets from GLM-5.3 today, against where each track started:

| | start | now |
|---|---:|---:|
| decode, rotating prompts, shallow (track G yardstick, 2026-09-04) | 1.65 tok/s | 2.60–2.75 (before F8's further +6–10 %) |
| decode at 18k context (2026-09-17) | 2.27 tok/s | **5.04** |
| cold prefill, 1 230 tokens (2026-09-06) | 241 s | ~50 s |
| the 18 439-token ladder turn, time to first token (2026-09-17) | 2 607 s | **356 s** |
| new Open WebUI chat (24-tool block, ~4 400 tokens) | ~10 min cold, every time | **2–4 s** (checkpoint restore); one cold prefill ~4.5 min when the tool list changes |
| follow-up turn | 65–171 s (reuse lost) | ~2 s |

## 3. History, in order

### 3.1 Before 2026-09-04: bring-up
Upstream's engines (`qwen38`, `glm53`) brought up on the rig with the Vulkan
expert tier. Only fresh-process numbers existed, and one session
(early September) produced three wrong baselines and took the box offline for
25 minutes by running a cache-sizing branch that ate MemAvailable. That is why
`CLAUDE.md` exists and why it is long.

### 3.2 Track G — GLM-5.3 decode (2026-09-04 → 09-11), CLOSED
`ROADMAP-2026-09.md`. Method established here and used ever since: a
persistent-engine yardstick (`datapoint.py`, rotating median), a per-op profile
(`[OPTIME]`), bit-identical oracles, one item per branch.

- G0 yardstick 1.65 tok/s. G3's profile: **69 % of the token ran on one core.**
- G2 (issue-all/take-all across three cards), G4 (KDA on all cores), G7
  (router 7.3×), G8 (MLA heads 2.4×), G9 (CPU experts inside the GPU wait),
  G10 (mHC), G11 (one OMP region per expert), G13 (shared expert one submit):
  all bit-identical, rotating 1.65 → **2.60–2.75 tok/s**, fresh-process token
  373 → 157 ms (135 with G12).
- G12: KDA recurrence on dev0 (`COLI_KDA_GPU=2`), not bit-identical, opt-in —
  the gateway uses it.
- G14: four int4 kernels tried, only one kept the output (+9.5 % on the expert
  path, `GLM53_I4_FAST`). G5: pooled DSA-indexer keys cached, 2.4× on the
  indexer constant. G6: dev2/dev3 preload stops on the VRAM budget.
- **Killed by measurement:** G1/G1b (mmap populate: a regression on this box),
  G15 (int3 experts: 16 of 1 232 predictions change), SPEC-PROBE.

### 3.3 Prefill / interactive track (2026-09-06 → 09-13), CLOSED
`PREFILL-ROADMAP-2026-09.md`. Opened the day Open WebUI showed that nothing
had ever measured time-to-first-token: "hi" with 34 tools took **24.5 min**.

- P0 harness + `prefill_gate.sh`. P2 batched dense stages, P3 row-batched CPU
  experts, P4/P4c tiled shaders, P5/P5b the sequential remainder (heads in
  SIMD lanes, per-thread scratch): all bit-identical; P2+P3+P4 alone 1.40× at 1 230 tokens (1.06 × 1.19 × 1.11), P5/P5b on top.
- The bigger win was **not recomputing**: P6 (4 KV slots), P6b (per-slot KDA
  state on the GPU), P7 (prefix checkpoints on disk + pinned memory block),
  P7b (capture after a restore), P8 (reply pin), **P9 (the conversation
  ledger: the gateway keeps the exact token sequence per conversation, because
  Open WebUI re-sends a transcript that does not re-render to it)**, P10
  (value-based checkpoint eviction), P11 (checkpoint-touch guard), P12
  (deterministic tool-call ids).
- RP4: GPU timestamps proved the GPU made the engine wait 0.068 ms/token —
  which killed the shader-tuning line of work (P4b refused by its gate).
- Acceptance tooling: `accept_live.sh` (rig, daily canary 05:00 UTC),
  `accept_ui.sh` (real Chromium from the Mac), `serve_candidate.sh` (serve,
  test, revert on failure), `gateway_watchdog.sh` (cron, 5 min), the rig lock.

### 3.4 Track Q — Qwen3.8-Flash-Next decode (2026-09-11 → 09-15), CLOSED
Rewritten from a profile after three of the original six items turned out to
be wrong before anything was built. Landed: Q1, Q2, Q3, Q5, Q10 (bit-identical
ports of track-G patterns), **Q7 (dense stream on the GPU, 84.7 → 38.9
ms/token)**. Rotating 4.04–4.09 tok/s. **Killed by measurement:** Q0, old-Q2,
Q8 (int4 experts: the model does not survive), Q9 (MTP: the verify block costs
more than it saves), Q11, Q14; Q4 stays opt-in (2 of 3 flips were not
near-ties); Q12 parked by the owner; Q13 (GPU power state) handed back as a
standing trade. The step-0 rule was written here: **measure the bucket and
re-derive the gate before building.**

### 3.5 Upstream (2026-09-08 → 09-15)
One whole-`dev` merge (2026-09-08) that, taken literally, would have shipped
four traps; policy since 09-15 is cherry-pick only. Seven qwen38 C tests pass
(two had failed since the merge for reasons in the tests, not the engine).

### 3.6 Franken track (2026-09-15 → today), ACTIVE
`FRANKEN-ENGINE-PLAN-2026-09-15.md`, rev 37. The owner's question: should the
backend move to HIP, and should the box serve a VRAM-resident 27–35B instead
of dragging 180 GB through RAM.

- **Night of 09-16, the regime map, measured on this box:** a resident 3-card
  MoE (gpt-oss-120b, llama.cpp) 75 tok/s at 18k; half its experts in RAM 28;
  all in RAM 17.6; GLM-5.3-Flash under *both* offload designs (Colibri and
  the owner's llama.cpp `-ngl 18`) under 3 tok/s at 18k, llama.cpp 24× faster
  at prefill; hipFire (HIP) on Qwen3.6-35B-A3B 131 → 79 tok/s down the context ladder on **one** card (164 on a short prompt).
  hipEngine cannot serve the 35B on 24 GB. RCCL tensor parallelism loses on
  arithmetic (25 µs × 80 hops). V1: a Vulkan expert tier for `qwen36`, all
  experts on one card, 14.6 → 21.6 tok/s.
- **F0** Vulkan copies at HIP's speed on the right queue family → no HIP port.
- **F4** per-request `[OPTIME]` → the DSA indexer was 54 % of the token at 18k.
- **F6a** indexer: OpenMP + exact heap top-k, bit-identical. 18k decode
  2.27 → 4.33, 18k turn 2 607 → 1 179 s. (Sonnet.)
- **F2** prefill streams non-resident experts to the cards through a host-RAM
  ring; chunk 512. 18k turn → 673 s. Found and fixed the swiglu-clamp bug on
  the way (§4). (Opus.)
- **F1** streamed decode misses: rejected on its own probe. **F5** (MTP):
  deferred, never run.
- **F7** prefill attention core on dev0, four shaders. 18k turn → 455 s.
  Numerics judged with a jitter arm (§5). (Opus.)
- **F8** the batch-1 CPU expert: vector group sums + one OMP team. Decode
  +6.4 to +9.6 %. (Sonnet.)
- **F9a** indexer score pass with heads in SIMD lanes, bit-identical. 18k turn
  → 358 s, decode +8.3 %. The planned GPU indexer (Opus) was closed unbuilt.
- **F10** step 0: KDA in prefill is dispatch latency; the shader fix is parked.

### 3.7 Today, 2026-09-20 afternoon — F3 (the VRAM-resident model)
The owner delegated the model choice; decided rev 34: **Qwen3.6-35B-A3B**.

| step | what | result | tier / tokens |
|---|---|---|---|
| 0 | int4-gs64 vs row-wise int8, CPU engine, same binary, containers differ only in routed experts; int8 container converted on the rig (`~/models/qwen36_i8_row`, 35 GiB, `~/venvs/conv`) | **mean KL 0.0316, top-1 92.96 %** over 625 positions — the size this track called a defect. → F3 uses int8. Plus the 3-card budget note `F3-STEP0-2026-09-20.md` | Sonnet, 208k |
| 1 | `qwen36-vk` tier over dev2+dev3, int8 accepted (no new shader: `fmt 1` existed) | 10 240/10 240 resident, 16.17 GB per card; tier on/off KL −1.7e-10, 100 %; int4-on-dev3 bit-identical to V1; **decode 21.5 tok/s = int4's speed**, TTFT −2.2 % | Sonnet, 345k |
| 2a | lm_head + DeltaNet projections on dev0, `Q36_VK_TRUNK=1` | numerics at the floor (0 flips in 625); **decode +2.0 % against a +20 % gate — NOT MET.** Merged behind its knob, off | Sonnet, 224k |

The token today (47 ms): DeltaNet 19.1 (CPU), MoE 13.6 (GPU tier round trips
9.9 + shared expert on CPU), attention 8.0 (CPU), lm_head 6.3 (CPU). 71 % is
CPU trunk. **A blocking GPU call costs ~0.3 ms on this backend whatever the
matrix**, which is what the CPU already took per layer — so moving the trunk
piece by piece is closed, by measurement.

**Mistakes of this session, stated plainly.** (1) Plan rev 36 projected
"~27 tok/s" for step 2a by arithmetic that assumed the call cost would fall
with the work; it measured 22.4. The Sonnet spend on 2a bought a negative
result and a hook, not speed. (2) The chains written today took the owner's
gateway out of service for about an hour in total (§4, bug 1). (3) Two
browser checks were run on top of each other and had to be discarded.

## 4. Bugs found and fixed, whole project

Ordered by what they cost the owner. "Found by" matters: most were not found
by the test that should have found them.

| # | bug | effect | found by | fix |
|---|---|---|---|---|
| 1 | **A chain's exit trap killed the engine it had just put in service** (pattern from `v1_step2a_chain.sh`, 11 chains). Server up, `[glm53] <defunct>`, `/v1/models` 200, every chat a 500 | 2026-09-16: 28 min. 2026-09-20: ~1 h over five windows | `accept_ui.sh` (09-20) | trap kills `glm53` only when about to restart (merge `c0acaaf`) |
| 2 | Both liveness guards (`run_chain.sh`, `gateway_watchdog.sh`) used `pgrep -x glm53`, **which matches a zombie** — the guard added after 09-16 was blind | bug 1 went unnoticed twice | reading the process table | `ps -C glm53 -o stat= \| grep -qv '^Z'`; watchdog redeployed to `~/bench` |
| 3 | **`--max-tokens 256`** in `~/start_glm53.sh` from day one; the server clamps every request down to it and Open WebUI sends none | every reply the box ever gave was cut at 256 tokens; a tool-calling turn returned nothing | the owner (2026-09-14) | 4096; `accept_live` check 5 |
| 4 | P12: tool-call ids re-parsed non-deterministically → the ledger diverged on every tool turn | **63–65 min of re-prefill per turn**, repeatedly | the owner's live chat | deterministic ids recorded in the ledger |
| 5 | P7b: a restored partial checkpoint was terminal — never captured again | a 380 s new chat, every time | the owner | capture after restore; rule "a gate must measure the request AFTER the one it tests" |
| 6 | P8/P9: Open WebUI stores only the visible reply, so the re-sent transcript never matched the KV (three breaks in three days: 21.6 s, 363 s, 54 s) | reuse silently lost | latency only | the conversation ledger |
| 7 | **The routed-expert GPU kernel never applied GLM-5.3's swiglu clamp** (since the tier existed) | output depended on which experts were tier-resident; top-1 vs the CPU reference 91.95 % | F2's KL bar, which the agent had not run | `GLM53_VK_SWIGLU_CLAMP=1`, second shader pair; 98.84 %; old checkpoints set aside |
| 8 | P11: `ckpt_disk_touch` located a checkpoint file by size alone | a counter could be written inside another capture; KV silently wrong | a code review the owner asked for | header re-read + guard |
| 9 | P10: checkpoint eviction by recency | a 170-token chat evicted the 4 307-token tool block (~24 min lost, nightly) | the canary | value-based eviction |
| 10 | G6: dev2/dev3 preload ignored the VRAM budget; HOST_VISIBLE allocations spill over ReBAR | 91 GB "in VRAM" was in host RAM | the record | budget stop |
| 11 | `fix/expert-cache-vs-page-cache` sized a cache from MemAvailable | the box thrashed until the OOM killer; SSH dead 25 min | it happened | fixed at `d9d38c5`; rule in `CLAUDE.md` |
| 12 | F2 step 0b: a 96 GiB `external_memory_host` import killed mid-run | a thread stuck in amdgpu in D state, 96 GiB pinned, **needed a reboot** | it happened | rule: no large anonymous imports into Vulkan here |
| 13 | G16: a prefill change validated on the CLI path only broke the served path; the Open WebUI rollback blamed the UI | a broken engine commit in service | the owner | serve-path oracle (`ttft_serve.py`), `accept_live` |
| 14 | Two sessions on the rig at once (2026-09-10); `pgrep` and good intentions did not stop them | a gateway restart under the other's acceptance run | a MISMATCH that was pure collision | the rig lock; watchdog honours it |
| 15 | `tee /dev/stderr` truncated chain logs; `pkill -f` over ssh kills its own session; `scp` over a running script | lost logs, dead sessions | agents stalling | fixed / bracket patterns / rules |
| 16 | P13: a "KDA garbage" regression that four gates seemed to share | two days of suspicion | bisect | it was a **stale shader** in a reused build tree; rule: a binary's own shaders, always |
| 17 | F8: thread-local scale table NULL in worker threads | would have been wrong output | its unit test | fixed before the gate |
| 18 | `test_qwen38_prefix` SIGFPE (our own G5 divide), `test_qwen38_native_weights` failing since the dev merge | red tests blamed on a "partial merge" that never happened | building the test at the merge base | guard; tolerance instead of `memcmp`; the false claim retracted |
| 19 | One `cat` of a 35 GB container under a full page cache leaves it ~85 % resident (pages read once are reclaimed first) | today's first chain refused itself | the chain's own residency check | warm in two passes |
| 20 | `flip_margin.py` crashed on zero flips | cosmetic | today's KL report | fixed |

**Measurement bugs** (each produced a confident wrong number): two empty
`grep`s compared as IDENTICAL (→ `gate_compare` refuses); uninterleaved A/B
reading +20 % and +11 % for real effects of +3.9 % and zero (→ A,B,B,A,
`gate_ab_verdict`); the decode track's tool used to re-baseline the prefill
track (→ `MEASURING.md`); a "decode slowdown" that was `datapoint.py`
evicting the model before every run on a 723 MB/s NVMe; timers that overlap
(F1 step 0); `webui.db` used as an oracle for tool calling; PR #1321's budget
formula subtracting true GB from GiB.

## 5. What was learned that a successor should not re-learn

1. **Profile first, then step 0, then build.** Step 0 (a microbenchmark or a
   reading of the hot loop) killed F1, shrank F9 from an Opus shader to a
   Sonnet CPU change, corrected F10, and deleted half of track Q. Today's 2a is
   what happens when the step 0 is arithmetic instead of a measurement.
2. **On this backend a synchronous GPU call costs 0.25–0.4 ms.** Anything that
   offloads one op per layer competes with a CPU that does the same op in that
   time. Wins came from fusing submits (G2, G12, G13), batching rows (P2–P4,
   F7), or not computing (P6–P9).
3. **A reordered float kernel is judged against a jitter arm**
   (`GLM53_MLA_ATTN_JITTER`), not a float64 reference: GLM moves ~2 % of
   argmaxes under fp32-rounding-sized noise.
4. **A gate that was not run is not a pass; read the chain's output, not the
   agent's summary** (F2). Agents build and write chains; the orchestrator
   launches, polls and reads.
5. **Every quantization shortcut on these models was measured and most died:**
   int3 experts (GLM), int4 experts and int8 dense (Qwen3.8), int4 vs int8
   (Qwen3.6, today). int4-g64 on GLM-5.3 is what ships because it is what
   exists.
6. **After any chain exits, send one real chat request.** `/v1/models` = 200
   and `pgrep` both lie.
7. Tier by cost: Sonnet did F1's probe, F6a, F8, F9a, F10 step 0 and all of F3
   so far; Opus was needed only for F2's ring and F7's shaders.

## 6. What still needs to be done

### 6.1 The one open decision — the owner's
**DECIDED 2026-09-21 (Franken plan rev 38): the owner closed F3. Qwen3.6 is
too far behind; only the recent Chinese models (Qwen3.8, DeepSeek V4.x,
GLM-5.3) are acceptable, on whatever engine fits the rig, smart quantizations
included. Step 2b is not started. The open item is F11 (rev 38). What follows
in this section is history.**

[2026-09-22: the F3 step-2b text and the Qwen3.6 trial recommendation that stood here were deleted; both are dead with F3 (rev 38). The live program is `FRANKEN-ENGINE-DESIGN-2026-09-22.md`.]

### 6.2 Candidates on GLM-5.3, none opened (evidence in `HANDOFF-2026-09-20.md` §4)
| candidate | evidence | size |
|---|---|---|
| ~~KDA `proj` in decode: 19 % of the decode token at 18k, eight batch-1 GPU matmuls per layer at ~0.4 ms each~~ **CLOSED by reading the code, 2026-09-21, nothing built:** under the served `COLI_KDA_GPU=2` `kda_layer` already runs the whole layer (eight projections, decay, recurrence, head norm, `ko`) in ONE submit, `coli_vk_kda_layer` (G12), and books all of it under `proj`. 0.911 ms is one round trip plus the layer's GPU work; there is nothing left to fuse | `c/glm53.c` `kda_layer`, the `gpu == 2` branch | none |
| `hc+norm` in prefill: 13 % of the 18k turn. **Read 2026-09-21:** not unexamined -- `run_layers` already runs `coli_hc_pre`+`rms` and `coli_hc_post` row-parallel over the chunk (P2.3) on G10's kernels; what is left is two passes over `n*H*D` floats per site. Fusing a site's post with the next site's pre saves one pass at best, a few % of the long turn | `c/glm53.c` `run_layers` | not worth an item |
| F10's shader: token loop inside `kda_step`, ≤ 8 % of the 18k turn | §F10-STEP0 | Opus, 200–300k tokens, parked |
| CPU expert kernel redesign (several rows per nibble decode) | §F8-STEP0 | Opus, not opened |
| F5 (MTP) re-pricing with F8's cheaper miss | arithmetic, no rig | free |

No single prefill bucket exceeds 36 %, so no single item is worth more than
~1.3× on the long turn. The decode side has one item of size (KDA `proj`).

### 6.3 Known gaps and housekeeping
- F8-style decode-only changes have no KL oracle (streamed prefill runs no CPU
  expert). Chunk size alone moves numerics (128 vs 512: KL 3.7e-04); cause not
  pinned. `f6a_gate_chain.sh`'s oracle arms ran with an empty tier — do not
  copy it. The engine's dev0 = PCI 83:00.0 is inferred, not confirmed.
- `~/bench` on the rig is ~135 GB, mostly logit dumps of merged items
  (`f2_out`, `f7_out`, `f9a_out`, `f6a_out`, today's `f3s*_out` ~3 GB);
  deletable, keep the `*_engine.log` and `.jsonl`. Pre-F2 checkpoints
  `<model>/.coli_ckpt.pre-f2.*` (1.3 GB) deletable.
- The rig's per-item trees `~/src/colibri-*` were all removed on 2026-09-20
  (see §0). Unmerged branches `perf/franken-d3`, `-h2b`, `-h4` are
  superseded; leave them.
- `~/bench/gateway_watchdog.sh` is a **copy** of the repo file; redeploy by
  `cp` + `mv` after editing it.
- Upstream: #1519 blocks any re-port; FP8 vectorisation and the two test fixes
  are upstream-worthy and unsent.

## 7. How to work here (the short version; `CLAUDE.md` is the law)

1. `tools/hot-expert/doc_currency.sh` must PASS before anything.
2. `ListAgents`; on the rig: lock free, `sha256sum ~/src/colibri/c/glm53`,
   `tail ~/bench/accept_live.log`.
3. One item = one `perf/...` branch = one chain through `run_chain.sh`
   (copy **today's `f3s2a_*` chains**, which carry the trap fix): lock, stop
   gateway, oracle, A,B,B,A, `gate_ab_verdict`, restart on every exit path,
   `accept_live`. Launch detached, poll the log.
4. After the chain exits: one real request. Before telling the owner anything
   works: `accept_ui.sh` from the Mac, **one at a time**.
5. Land: merge, record §, plan rev (look up the top rev first), §8.3 row,
   `CLAUDE.md` if a knob or binary changed, `doc_currency.sh`, push `origin`,
   sync the rig through `p0-sync`, update memory.

## 8. Where everything is

| what | where |
|---|---|
| Rules and traps | `CLAUDE.md` |
| Every measured number | `tools/hot-expert/ROME-3x7900XTX-2026-09-04.md` (15k lines; grep `^## §`) |
| **The design (current truth, 2026-09-22)** | `tools/hot-expert/FRANKEN-ENGINE-DESIGN-2026-09-22.md` (M0-M5, L0-L5) |
| Decision log | `tools/hot-expert/FRANKEN-ENGINE-PLAN-2026-09-15.md` (top rev; its F items are all closed) |
| Closed plans | `ROADMAP-2026-09.md` (decode G/Q), `PREFILL-ROADMAP-2026-09.md` |
| Which tool measures what | `tools/hot-expert/MEASURING.md` |
| Upstream policy | `tools/hot-expert/UPSTREAM-POLICY-2026-09-15.md` |
| F3 budget + step 0 | `tools/hot-expert/F3-STEP0-2026-09-20.md` |
| Rig | ssh `rome`; tree `~/src/colibri`; logs, binaries, chains' output `~/bench`; models `~/models`; gateway log `~/glm53_server.log`; `~/bench/owui_report.sh N` |
| Git | `origin` = Gitea (canonical), `fork` = GitHub mirror, `rome` = rig over ssh, `upstream` = JustVugg |
