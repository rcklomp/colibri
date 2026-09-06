# Working on this fork (rcklomp/colibri) on the rome box

This file is loaded into every Claude Code session on any model. It exists
because one session in September 2026 produced three wrong baselines and
took the machine offline for 25 minutes by ignoring what is written here.
Follow it before forming any plan.

## Read first, in this order

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
- **GLM tier caps:** `glm53`'s dev2/dev3 preload stops on a count cap only.
  Always pass `COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695` (until roadmap
  item G6 lands) and preload from the histogram with the most history
  (`~/.glm53_explain.bin`). An unlimited cap spilled 91 GB into host RAM.
- **Never run a `glm53` built from `fix/expert-cache-vs-page-cache*` or
  commit `eabeb9a` here:** its cache sizing takes MemAvailable and thrashes
  the box until the OOM killer acts. If SSH stops answering, that is why;
  it recovers on its own.
- **One benchmark at a time.** The rig serialises measurements. Parallel
  sessions or subagents may edit and build concurrently; only one may run an
  engine. Check `pgrep -x qwen38 -x qwen38-vk -x glm53` first.
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

## Engines

- `qwen38` (CPU) and `qwen38-vk` (VK=1 target, `Q38_VULKAN=1` plus
  `COLI_VK_DEV2=auto COLI_VK_DEV3=auto`) — `c/qwen38.c`, `c/qwen38_core.h`.
- `glm53` (`COLI_VULKAN=1`, three devices) — `c/glm53.c`, `c/sparse_index.h`.
- Shared: `c/quant.h` (CPU kernels; `matmul_fp8` is Qwen-only in practice),
  `c/backend_vulkan.c`, `c/shaders/*.comp`. A shared-file change must
  rebuild and re-measure both engines.
- Build: `make -C c qwen38 qwen38-vk glm53 VK=1`. **All four qwen38 C tests
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
