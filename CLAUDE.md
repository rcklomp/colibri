# Working on this fork (rcklomp/colibri) on the rome box

This file is loaded into every Claude Code session on any model. It exists
because one session in September 2026 produced three wrong baselines and
took the machine offline for 25 minutes by ignoring what is written here.
Follow it before forming any plan.

## Read first, in this order

**Step 0, before any of the four below: run `tools/hot-expert/doc_currency.sh`.**
It exits non-zero when the docs this section tells you to trust are behind the
tree, and it takes seconds — no rig, no engine. It exists because on 2026-09-15
three separate stale pointers were found in one day and every one of them was
found late, by the owner pushing, not by the session checking: the prefill
roadmap sat four days behind while a live-incident fix (P12) went unrecorded,
its item table had no rows for P11 or P12, and Track Q's "what is next" still
told the reader to run a probe that had already been completed. Run against the
tree as it stood at the start of that session the script fails immediately on
the first of those. **This section calls the roadmaps "current by construction";
that is a claim to verify, not to assume.**


0. `tools/hot-expert/PREFILL-ROADMAP-2026-09.md` — the prefill /
   interactive-use track, opened 2026-09-06 when Open WebUI exposed that
   nothing had ever measured time-to-first-token. **Read the top of the
   file (the highest `Rev N` entry) for what is actually next — do not
   trust a prior session's summary of this, including an old copy of this
   paragraph: a stale hardcoded "the open items are ..." list here once sent
   a session chasing an item (RP4) that had already landed a rev earlier
   (2026-09-10). This file changes every session; the roadmap doc is the
   only thing here that is current by construction.** **Its gate is a
   script:** `tools/hot-expert/prefill_gate.sh <pristine> <candidate>` exits
   1 on an oracle miss and 3 on a TTFT regression or a decode drop; an item
   is done when it exits 0 with its table in the commit body. The chain
   scripts in `~/bench/p*_chain.sh` on the rig are the pattern for running
   it unattended (stop gateway → merge → build → gate → tworeq → serve only
   on rc 0 → restart).
1. `tools/hot-expert/ROADMAP-2026-09.md` — the two tracks (GLM-5.3, Qwen3.8),
   each item with evidence, expected delta, gate, and the model tier that
   may execute it. Work only on a named item.
2. `tools/hot-expert/ROME-3x7900XTX-2026-09-04.md` — the measurement record:
   hardware, baselines in both regimes, the per-op placement matrix, what was
   tried and rejected. Do not re-derive anything that is in it.
3. `tools/hot-expert/MEASURING.md` — **which tool measures which track.**
   The decode-throughput roadmap and the prefill roadmap have different
   headline metrics and different harnesses (`tools/rome_bench.sh` vs
   `tools/hot-expert/prefill_snapshot.sh`); using one track's tool to
   re-baseline the other produces a number that looks like a regression and
   is not (2026-09-10 — read this before running any benchmark, not after).
4. `git log` on the branch you are on. Commit bodies carry the numbers
   behind every landed change; the history is the benchmark archive.

## Where sessions run

Claude Code sessions run on the owner's Mac in a checkout of this repo
(`~/Projects/colibri`, tracking Gitea). The engines, models and benchmarks
live on the rig, reached over SSH (host alias `rome`; access details are in
the owner's notes, not in this repo). The rig's working tree is
`/home/ronald/src/colibri`; keep the two in sync through Gitea, not by
editing on both sides. Bench scripts and logs on the rig are in `~/bench`.

## The machine and its traps

- EPYC 7F32 (8 cores / 16 threads, Zen 2, AVX2+FMA+F16C, no AVX-512),
  247 GiB RAM (MemTotal; **that is 265.6 GB** — this box's RAM is habitually quoted in GiB and labelled GB, and that exact confusion produced a real bug: PR #1321's budget formula subtracted a true-GB model size from a GiB total), three RX 7900 XTX on RADV/Vulkan (no hardware FP8), one NVMe.
- **Threads:** the `coli` launcher and `c/tools/datapoint.py` pin engines to
  physical cores (8). All recorded numbers are 8-thread numbers. Direct
  engine runs must set `OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close`
  or say explicitly that they did not.
- **Page cache:** Qwen3.8 (173 GiB) and GLM-5.3 (182 GiB) do not fit in RAM
  together. Warm one model at a time (drop caches, `cat` its shards) and
  verify with `fincore` before any number is recorded. A cold run pays 100k+
  major faults and is not a baseline.
- **GLM tier caps:** **G6 landed 2026-09-06** — the dev2/dev3 preload now
  stops on the VRAM budget as well as the count cap (1.0 GB reserve on those
  expert-only devices, `COLI_VK_TIER_RESERVE_GB` to change it), so a large cap
  can no longer spill into host RAM. Verified: cap 2200 stops at 1752 with
  MemAvailable unmoved; the old behaviour is what put 91 GB in host RAM.
  **Still always pass `COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695`** — now for
  COMPARABILITY, not safety: every recorded number was taken at 1695, the guard
  deliberately does not fire there, and changing the tier changes routing.
  Preload from the histogram with the most history (`~/.glm53_explain.bin`).
  Note an *unset* cap does not fill the tier, it skips dev2/dev3 entirely.
- **Never run a `glm53` built from the branch `fix/expert-cache-vs-page-cache*`
  at any commit before its fix `d9d38c5`:** that cache sizing took
  MemAvailable and thrashed the box until the OOM killer acted. (An older
  version of this rule named commit `eabeb9a`; that sha is #1325's harmless
  `st.h` mapping primitive and has been an ancestor of `hot-expert-tier`
  since the upstream `dev` merge on 2026-09-08 — the branch is the warning,
  not the sha.) If SSH stops answering, that is why; it recovers on its own.
- **Expert mapping** lives in `c/st.h` since the dev merge
  (`st_map_shard_range`, `COLI_MAP_EXPERTS_DEFAULT`): on by default for
  `glm53` and `qwen38`, off for every other engine; `COLI_MAP_EXPERTS=0` is
  the A/B next to `GLM53_NO_MMAP` / `Q38_NO_MMAP`. The startup line reads
  `[MAP] 59 file mappati (st_map_shard_range), esperti mappabili 12096/12096`;
  `majflt` per request stays 0 and `[MAP] … copy=0` — if either moves, the
  mapping is the first suspect.
- **GLM-5.3's routed-expert GPU kernel skipped the swiglu clamp until
  2026-09-19; the fix is a KNOB and the gateway sets it.**
  `c/shaders/qmatmul_gate_up.comp` computes `silu(gate)*up` with no bound;
  the CPU path applies `swiglu_limit=10.0`. `GLM53_VK_SWIGLU_CLAMP=1` binds
  a second pair of SPIR-V modules (`qmatmul_gate_up_clamp.spv`,
  `qmatmul_gate_up_tile_clamp.spv`, same source with `-DSWIGLU_CLAMP`) on
  all three devices; unset, the engine is bit-identical to the old one and
  the unclamped modules are byte-identical (so `qwen38-vk` is untouched).
  Measured (record §F2 step 2): against the clamped all-CPU reference
  (`GLM53_EXPERTS_CPU=1`) top-1 agreement 91.95 % → 98.84 %, mean KL
  0.0233 → 0.00143; cost +0.3 %. **A direct engine run without the knob is
  the OLD numerics** — output then depends on which experts are
  tier-resident (6 of 42 and 8 of 1232 `teacher_forcing` predictions,
  §G15). Every gate from here on runs both arms with the clamp on, as
  `f2_gate_chain.sh` does, or says that it did not. The remaining 1.2 % vs
  the all-CPU reference is placement (summation order), not the clamp.
- **Prefill streams non-resident experts to the cards (F2, in service
  2026-09-19).** `COLI_PREFILL_STREAM=1`: per MoE layer the chunk's
  non-resident experts are copied into a HOST-RAM ring (two banks of 32
  slots per card, 2.72 GB, zero VRAM — a ring in ReBAR VRAM filled at 16
  GB/s against 58.6) and run on the GPU tile kernels; the CPU int4 path is
  the knob-off default. It does not touch `GLM53_PREFILL_CHUNK` (default
  128; the gateway serves 512, worth 1.12×). Gate (record §F2-LADDER):
  turn 1 136.8 → 67.6 s (2.03×), the 18 439-token turn 1 174.8 → 672.6 s
  (1.75×), stream on vs off under the clamp mean KL 3.5e-05 / top-1
  99.89 %. After it the 18k prefill token is 79.2 ms and **attention
  (`mla.attn`, 27.6 ms, 35 %) is the largest prefill bucket**, the indexer
  second (13.4 ms, 17 %). Chunk size alone changes numerics (128 vs 512
  under the clamp: KL 3.7e-04, top-1 99.68 %); the cause is not pinned
  (indexer and tile threshold ruled out, record §F2k).
- **Prefill attention runs on dev0 (F7, in service 2026-09-20).**
  `GLM53_MLA_ATTN_GPU=1`: the MLA score/softmax/pool core of a prefill chunk
  runs batched over its rows in four `mla_attn_*.spv` shaders, from one shared
  scratch (140 MB at 18k, 236 MB at 64k; nothing per layer or per slot);
  decode and the knob-off default stay on the CPU and are bit-identical to
  before. Gate (record §F7-VERDICT): the 18 439-token turn 683.3 -> 454.6 s
  (1.50x), turn 1 1.35x. **How its numerics were judged, and how the next
  reordered attention kernel must be:** against the CPU path it moves ~2 % of
  argmaxes, and so does the CPU path itself under
  `GLM53_MLA_ATTN_JITTER=6e-7` (a random fp32-rounding-sized perturbation,
  oracle only) -- same KL, same first position. A float64 arm
  (`GLM53_MLA_ATTN_REF64=1`) stays close to the CPU path only because it keeps
  the summation order; it is not the yardstick for a reordered kernel, the
  jitter arm is. A synthetic probe with a near-uniform softmax cannot validate
  an attention kernel; replay real dumped inputs.
- **One benchmark at a time, across sessions as well as inside one.** The rig
  serialises measurements. Parallel sessions or subagents may edit and build
  concurrently; only one may run an engine. Check `pgrep -x glm53`,
  `pgrep -x qwen38`, `pgrep -x qwen38-vk` first (one pattern per call) — and
  take the rig lock, because on 2026-09-10 two Claude sessions worked this rig
  at once and neither `pgrep` nor good intentions stopped them from
  interleaving: one restarted the gateway underneath the other's acceptance
  run, truncating `~/glm53_server.log` and producing a MISMATCH that was pure
  collision. **`ListAgents` shows peer sessions; if one is busy on this repo,
  say so and coordinate over `SendMessage` before touching the rig.**
  **Take the rig lock (`~/bench/.rig.lock`, `tools/hot-expert/rig_lock.sh`)
  before stopping the gateway for ANY reason, not only inside a chain
  launched through `run_chain.sh`.** `gateway_watchdog.sh` runs from cron
  every 5 minutes and restarts the gateway whenever it is down and no lock
  is held — stop the gateway by hand without the lock and the watchdog can
  race back in mid-measurement (happened 2026-09-10, corrupted a
  `datapoint.py` run with a second `glm53` starting underneath it).
  `tools/rome_bench.sh` now takes the lock and handles the stop/restart
  itself; if you ever stop the gateway some other way, take the lock first.
- **The gateway is the owner's daily service.** `~/start_glm53.sh` runs
  `openai_server.py` on 8081 with **`--max-tokens 4096`** (2026-09-14: it was
  `256` from the first day of Open WebUI service, and `openai_server.py` clamps
  every request DOWN to the server cap while Open WebUI sends no `max_tokens` of
  its own — so 256 was the ceiling on every reply the box had ever produced, and
  a tool-calling turn spent it opening a `<tool_call>` box it could never close,
  which is why the owner's chat returned nothing at all. `accept_live.sh` check 5
  exists to catch a recurrence and runs in the daily canary too), `--kv-slots 4`,
  **`GLM53_VK_SWIGLU_CLAMP=1 COLI_PREFILL_STREAM=1 GLM53_PREFILL_CHUNK=512`,
  `GLM53_MLA_ATTN_GPU=1` (F7, 2026-09-20),
  `GLM53_MOE_ONE_TEAM=1 GLM53_I4_FAST=2` (F8, 2026-09-20: the batch-1 CPU
  expert path, decode +6.4 to +9.6 %; `I4_FAST=2` is a reassociating kernel,
  so a direct engine run without it is NOT the served numerics)
  (F2, 2026-09-19; prefix checkpoints from before that date were set aside,
  they carry the unclamped numerics)** and `COLI_KDA_GPU=2`
  (P6b, 2026-09-07: each slot has its own KDA device state, pool allocated
  before the expert preload — dev0 holds 1 248 experts instead of 1 296),
  `GLM53_PREFIX_CKPT=1` and `COLI_PREFIX_PIN=1` (P7: prefix checkpoints under
  `<SNAP>/.coli_ckpt`, the Open WebUI memory block pinned per conversation),
  `COLI_REQ_LOG=1` and `GLM53_VERBOSE=1` into a timestamped
  `~/glm53_server.log`; `~/bench/owui_report.sh N` shows the last N
  requests with prompt tokens, reused tokens and ttft; `CKPT hit/store`
  lines say when a checkpoint fired. Stop it only inside a chain script
  that restarts it on every exit path (`pkill -f "openai_[s]erver.py"`, then
  `pkill -9 -x glm53`; restart with
  `SKIP_WARM=1 setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &`).
  A gate step must wait for its engine to die before the next one starts
  (`wait_no_engine()` in `p7_gate.sh`), or the chain's `cp` of the pristine
  fails with ETXTBSY and a failed candidate stays in service.
- **`pkill -f` over ssh matches the ssh command itself** if the pattern
  appears in it, and kills the session: always use the bracket form
  (`"openai_[s]erver.py"`, `"p7_[c]hain.sh"`). Never `scp` over a bash
  script that is running on the rig; git merges are safe (new inode).
- **Open WebUI** is the container `open-webui-new` (image
  `ghcr.io/open-webui/open-webui:main` = 0.11.3, port 3000) since 2026-09-09;
  the old `open-webui` (0.11.0) is stopped with restart off — the 09-06
  rollback blamed the UI for the broken G16 engine commit. The preset's
  `builtin_tools` and `memory` capabilities are ON again; the title/tags/
  follow-up tasks stay off. Its 24-tool block (~4 000 tokens) is
  checkpointed (`CKPT hit prefix=4011` in the server log): a new
  conversation answers in ~20 s, a follow-up turn in ~2 s; a changed tool
  set pays one cold prefill (~4.5 min since F2, 274 s measured 2026-09-19; ~10 min before) and is captured again. The server
  caches presets: after a `webui.db` edit call `GET /api/models`. Drive its
  backend from inside the container (PyJWT token from `WEBUI_SECRET_KEY`;
  never `import open_webui` in a side process, it runs the migrations); the
  builtin tools attach only to requests carrying a `session_id`.
- Dropping caches needs sudo; ask the owner or run it yourself in an
  interactive shell. Never write the password into a file or a script.

## How a change is measured (no exceptions)

- Before and after, same prompt, same regime. The serving regime is
  `c/tools/datapoint.py` (persistent engine, cold / warm-identical / rotating);
  the rotating median is the headline, the warm-identical row is an upper
  bound, and a fresh-process run is a diagnosis, not a result.
- **Oracle:** greedy output text identical, and last-token logits compared
  (`DUMP=` + a cosine/max-abs/argmax diff) against the pristine binary.
  For `glm53` the `teacher_forcing` line is the oracle. `tok/s` and hit rate
  have both moved the right way on a corrupted model before.
- A change that alters numerics ships behind an env knob, off by default,
  with the logit diff in the commit body.
- Repeat the headline number at least twice; report both.
- **Source `tools/hot-expert/gate_lib.sh` and let it decide, rather than reading
  the numbers yourself.** `gate_compare` REFUSES a comparison whose pattern is
  absent from either side — two empty `grep`s diffed against each other print
  IDENTICAL, and that is how a gate once reported a passing numerics oracle from
  two runs that had produced no output at all. `gate_ab_verdict` refuses a
  verdict from a single unpaired sample, and reports NO VERDICT when the arms
  overlap. **Run arms interleaved A,B,B,A.** On this box an uninterleaved pair
  read **+20%** and **+11%** on two changes whose real effects were +3.9% and
  zero: the second arm inherits a warm page cache, and that is worth more here
  than most optimisations. Validated against all three of those real cases.
- The numbers go in the commit body and one row goes into the record. A
  change without a measured delta does not merge.
- **A binary is in service only when `tools/hot-expert/accept_live.sh`
  exits 0 on it**, run on the rig against the live gateway: two UI-shaped
  new chats through Open WebUI's own backend (the second must restore the
  prefix and answer in seconds), a follow-up turn with memory on, and a
  short request behind an abandoned one. Every chain ends with
  `tools/hot-expert/serve_candidate.sh <candidate> <pristine> [shaders]`,
  which restarts, runs it, and reverts on failure. It exists because P7b
  (2026-09-09): every P7 gate proved the request under test and none looked
  at the request after it, and the owner found the 380-second new chat
  himself. **A gate must measure the request after the one it tests** — the
  state a request leaves behind is what the next user turn pays for.
- A daily canary (`accept_live.sh --canary`, cron 05:00 UTC, log
  `~/bench/accept_live.log`) re-checks the user's path and captures the tool
  block again whenever Open WebUI's tool list drifts.
- **Before telling the owner that anything about serving works, run
  `tools/hot-expert/accept_ui.sh` from the Mac.** It drives Open WebUI in a
  real Chromium at the rig's IPv4 (mDNS gives only a link-local IPv6, which
  Chrome refuses): a new chat, a typed question, and the first token as it
  appears ON SCREEN — then the rig-side `accept_live.sh` over ssh. Measured
  2026-09-09: first token 3.58 s and 3.82 s in the browser against 2.2–2.6 s
  at the gateway; the difference is front end and network, and it is the
  number the owner actually waits for. The probe signs in with a token minted
  inside the container (no password), deletes the chats it creates, and
  labels `browser=system-chrome` when it had to fall back — those numbers are
  ~10 s slower and not comparable. The rig has no browser and no Node, so
  this gate cannot run there and is not the canary: it is what a session owes
  the owner before saying "it works".

## Engines

- `qwen38` (CPU) and `qwen38-vk` (VK=1 target, `Q38_VULKAN=1` plus
  `COLI_VK_DEV2=auto COLI_VK_DEV3=auto`) — `c/qwen38.c`, `c/qwen38_core.h`.
- `glm53` (`COLI_VULKAN=1`, three devices) — `c/glm53.c`, `c/sparse_index.h`.
- Shared: `c/quant.h` (CPU kernels; `matmul_fp8` is Qwen-only in practice),
  `c/backend_vulkan.c`, `c/shaders/*.comp`. A shared-file change must
  rebuild and re-measure both engines.
- Build: `make -C c qwen38 qwen38-vk glm53 VK=1` (header prerequisites are in
  the Makefile since the dev merge, so a `st.h`/`quant.h` change rebuilds both
  engines by itself; the tiled shaders
  `shaders/qmatmul_tile.spv` / `qmatmul_gate_up_tile.spv` are built by the
  same Makefile and loaded from `c/shaders` — a binary copied elsewhere must
  still point `COLI_VK_SHADERS` at the repo's `c/shaders`; the harness
  refuses to run without them).
- Binary copies for gates live in `~/bench/` (`glm53.pristine` = pre-prefill
  track, `glm53.p2`, `glm53.p4base`, `glm53.p4c_base`, `glm53.p6base`,
  `glm53.p7base` = pre-P7, `glm53.p6bbase` = P7, `glm53.p6b`, `glm53.rp4`,
  `glm53.p5` (+ `shaders_p5`), `glm53.devmerge` (+ `shaders_devmerge`) = the
  binary in service from 2026-09-08 01:00, `glm53.f6abase`, `glm53.f2base` =
  F6a `15462dc2`, `glm53.f2` = `5c01246c`, `glm53.f7` = `e87ae939`, `glm53.f8` = `93f0f681`, in service since 2026-09-20 08:54 UTC); each gate's pristine is the
  binary in service before the item. **Every gate runs with
  `GLM53_PREFIX_CKPT=0` and a private `COLI_CKPT_DIR`**: two gates nearly
  passed for the wrong reason because the candidate restored a checkpoint
  the pristine had just written. **qwen38 C tests: there are SEVEN, not four, and as of 2026-09-15 evening
  all seven pass.** `test_qwen38_native_weights` had failed since at least the
  dev merge `132d177` (2026-09-08) for TWO independent reasons, both in the
  TEST's assumptions and neither in the engine; both are fixed in the test and
  both fixes are upstream-worthy. (1) Line 587 did a bit-exact `memcmp`
  between the test's scalar `reference_fp8_matmul` and `q38_weight_matmul`;
  this fork VECTORISED `matmul_fp8` (AVX2/FMA eight-lane accumulation,
  `9f2cce2`, then the F16C decode `b3b20fb`) and a different summation order
  cannot pass a `memcmp` — proved by `-mno-avx2 -mno-fma`, which made 587
  pass. It is now a tolerance of 1e-6 of the sum of absolute terms of each dot
  product (measured 8.3e-8 on AVX2+FMA; a wrong decode, scale block or layout
  is off by O(1) of that sum). If the FP8 vectorisation is ever offered
  upstream, upstream's copy of this test fails the same way. (2) Line 635,
  `check_fixture_mode(adjacent_directory,1,1)`, asserted that a fast
  native-FP8 expert lives in a slot-owned slab; this fork serves experts from
  the shard mapping by default (`COLI_MAP_EXPERTS_DEFAULT 1`), which releases
  the slab by design — bytes, read counters and gate/up/down adjacency all
  correct, only the owner of the memory differs. Proved 2026-09-15 by building
  the test at the merge base `1ccee43` and at `upstream/dev` on the Mac:
  both PASS by default and both FAIL at 635 with `COLI_MAP_EXPERTS=1`.
  **A hand-off had blamed a "partial merge" that dropped `qt_ready` /
  `qt_note` / `q38_tier_note` from upstream `a5d177b`; that commit is not an
  ancestor of the merge base, nothing from it was ever taken, and the merge
  base passes without those symbols. The claim was wrong — do not re-inherit
  it.** Full table in `tools/hot-expert/UPSTREAM-POLICY-2026-09-15.md`. `tests/test_qwen38_prefix` used to SIGFPE -- **on OUR tree, not
  "on every tree" as this line used to say**: upstream's `ensure_kv` has no
  `IK_pooled` and no `idx_ratio` divide at all, so the crash arrived with our own
  G5 pooled-index cache and could never have happened upstream. It is fixed
  (unguarded `m->max_t / c->idx_ratio` in `ensure_kv`, which the test's fabricated
  Model leaves at 0). If it crashes again, that guard is the first place to look.

## Sending anything upstream

**Run `tools/hot-expert/upstream_lint.sh upstream/dev <branch>` before opening or
updating a PR against another project, and re-run it after every edit.** It fails
on added lines carrying references that mean nothing outside this fork: roadmap
row labels (`G4`, `Q10`, `P7b`), `tools/hot-expert/...` paths, `CLAUDE.md`, the
rig's name, `~/bench/...`, "see the record". Say what the thing *describes*
instead; the substance — especially a negative result — is worth keeping, the
label is not.

It exists because a reviewer had to ask for this on PR #1521 (2026-09-15), and
because when asked whether the sibling PR had the same problem the answer was
"no" from memory and was wrong: #1524 carried two more labels, a link to a path
that does not exist upstream, the rig's name and "see the record". **Do not
answer that question from memory — run the script.**

## Git

- Canonical remote is Gitea (`gitea`); GitHub is a mirror of it. The rig
  has no push credentials; push from the owner's Mac, which relays the branch.
- One item, one branch or one commit series; commit message body = what was
  measured, in numbers, plus which oracle passed.
- Do not push to `hot-expert-tier` directly; land on a `perf/...` branch and
  merge after the gate is met.
- **No whole-`upstream/dev` merges (decided 2026-09-15,
  `tools/hot-expert/UPSTREAM-POLICY-2026-09-15.md`).** Take upstream fixes by
  `git cherry-pick -x`, one at a time, each through its own gate; send ours up
  as PRs. A full re-port is re-opened only when #1519 is closed upstream AND
  upstream lands something this box needs that a cherry-pick cannot carry.

## Model tiers (from the roadmap)

Haiku 4.5: run campaigns, fill the record, apply a fully specified patch.
Sonnet 5: port a pattern that already exists in the repo, knobs, harness,
tests. Opus 5: kernels, shaders, profiling and its interpretation. Fable:
the KDA decision after G3, the Q3 and Q4 designs, arbitration. Subagent
definitions for the first three live in `.claude/agents/`.
