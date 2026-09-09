# Working on this fork (rcklomp/colibri) on the rome box

This file is loaded into every Claude Code session on any model. It exists
because one session in September 2026 produced three wrong baselines and
took the machine offline for 25 minutes by ignoring what is written here.
Follow it before forming any plan.

## Read first, in this order

0. `tools/hot-expert/PREFILL-ROADMAP-2026-09.md` (rev 9) — the prefill /
   interactive-use track, opened 2026-09-06 when Open WebUI exposed that
   nothing had ever measured time-to-first-token. P0–P6 landed the same
   day; the open items are P7 (checkpoint the system+tools prefix), P6b
   (per-slot KDA device state), RP4 (GPU timestamps on the expert group),
   P5, then the upstream-`dev` merge. **Its gate is a script:**
   `tools/hot-expert/prefill_gate.sh <pristine> <candidate>` exits 1 on an
   oracle miss and 3 on a TTFT regression or a decode drop; an item is done
   when it exits 0 with its table in the commit body. The chain scripts in
   `~/bench/p*_chain.sh` on the rig are the pattern for running it unattended
   (stop gateway → merge → build → gate → tworeq → serve only on rc 0 → restart).
1. `tools/hot-expert/ROADMAP-2026-09.md` — the two tracks (GLM-5.3, Qwen3.8),
   each item with evidence, expected delta, gate, and the model tier that
   may execute it. Work only on a named item.
2. `tools/hot-expert/ROME-3x7900XTX-2026-09-04.md` — the measurement record:
   hardware, baselines in both regimes, the per-op placement matrix, what was
   tried and rejected. Do not re-derive anything that is in it.
3. `git log` on the branch you are on. Commit bodies carry the numbers
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
  247 GB RAM, three RX 7900 XTX on RADV/Vulkan (no hardware FP8), one NVMe.
- **Threads:** the `coli` launcher and `tools/datapoint.py` pin engines to
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
- **One benchmark at a time.** The rig serialises measurements. Parallel
  sessions or subagents may edit and build concurrently; only one may run an
  engine. Check `pgrep -x glm53`, `pgrep -x qwen38`, `pgrep -x qwen38-vk`
  first (one pattern per call).
- **The gateway is the owner's daily service.** `~/start_glm53.sh` runs
  `openai_server.py` on 8081 with `--kv-slots 4` and `COLI_KDA_GPU=2`
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
  set pays one cold prefill (~10 min) and is captured again. The server
  caches presets: after a `webui.db` edit call `GET /api/models`. Drive its
  backend from inside the container (PyJWT token from `WEBUI_SECRET_KEY`;
  never `import open_webui` in a side process, it runs the migrations); the
  builtin tools attach only to requests carrying a `session_id`.
- Dropping caches needs sudo; ask the owner or run it yourself in an
  interactive shell. Never write the password into a file or a script.

## How a change is measured (no exceptions)

- Before and after, same prompt, same regime. The serving regime is
  `tools/datapoint.py` (persistent engine, cold / warm-identical / rotating);
  the rotating median is the headline, the warm-identical row is an upper
  bound, and a fresh-process run is a diagnosis, not a result.
- **Oracle:** greedy output text identical, and last-token logits compared
  (`DUMP=` + a cosine/max-abs/argmax diff) against the pristine binary.
  For `glm53` the `teacher_forcing` line is the oracle. `tok/s` and hit rate
  have both moved the right way on a corrupted model before.
- A change that alters numerics ships behind an env knob, off by default,
  with the logit diff in the commit body.
- Repeat the headline number at least twice; report both.
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
  binary in service since 2026-09-08 01:00); each gate's pristine is the
  binary in service before the item. **Every gate runs with
  `GLM53_PREFIX_CKPT=0` and a private `COLI_CKPT_DIR`**: two gates nearly
  passed for the wrong reason because the candidate restored a checkpoint
  the pristine had just written. **All four qwen38 C tests
  pass** as of 2026-09-06. `tests/test_qwen38_prefix` used to SIGFPE on every
  tree; that was C1's merge blocker and it is fixed (unguarded
  `m->max_t / c->idx_ratio` in `ensure_kv`, which the test's fabricated Model
  leaves at 0). If it crashes again, that guard is the first place to look.

## Git

- Canonical remote is Gitea (`gitea`); GitHub is a mirror of it. The rig
  has no push credentials; push from the owner's Mac, which relays the branch.
- One item, one branch or one commit series; commit message body = what was
  measured, in numbers, plus which oracle passed.
- Do not push to `hot-expert-tier` directly; land on a `perf/...` branch and
  merge after the gate is met.

## Model tiers (from the roadmap)

Haiku 4.5: run campaigns, fill the record, apply a fully specified patch.
Sonnet 5: port a pattern that already exists in the repo, knobs, harness,
tests. Opus 5: kernels, shaders, profiling and its interpretation. Fable:
the KDA decision after G3, the Q3 and Q4 designs, arbitration. Subagent
definitions for the first three live in `.claude/agents/`.
