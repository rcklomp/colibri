# Which tool answers which question (read before running any benchmark)

Written 2026-09-10 after a re-baseline session used the wrong harness for the
wrong track and reported what looked like a regression that was not one — a
comparison between two numbers that were never measuring the same thing. The
two tracks in this repo (`ROADMAP-2026-09.md` and `PREFILL-ROADMAP-2026-09.md`)
have different headline metrics and different tools. Using one track's tool
to "re-baseline" the other produces numbers that are internally consistent
and still meaningless next to that track's own history.

| Track | Headline metric | Tool | Touches the live gateway? |
|---|---|---|---|
| `ROADMAP-2026-09.md` (decode throughput, G/C/Q items) | rotating-prompt tok/s on short prompts (30-43 tok), persistent engine, no checkpoint/ledger | `tools/rome_bench.sh <engine> <config>` | No — spawns its own engine, stops the gateway for the duration (does this itself since 2026-09-10; do not hand-roll the stop/restart) |
| `PREFILL-ROADMAP-2026-09.md` (TTFT / interactive use, P/RP items) | TTFT at 27/384/1230/3462 tokens, checkpoint reuse, ledger correctness | `tools/hot-expert/prefill_snapshot.sh [tag]` for a read-only snapshot of the SERVED binary; `ttft_serve.py --engine <bin>` (via `prefill_gate.sh`) to compare two binaries before one goes into service | Yes, snapshot mode — no stop needed |

## `rome_bench.sh`'s headline number is half a disk benchmark (2026-09-10)

`datapoint.py` evicts the model from the page cache — `posix_fadvise(DONTNEED)`
over all 62 shards — before **every** run unless `--no-evict` is passed, and
`rome_bench.sh` does not pass it. So the harness warms 182 GiB, asserts 100%
residency with `fincore`, and datapoint throws all of it out again one second
later; the engine then re-reads the model from NVMe *inside* the measurement.
The cold row says so in its own label ("cache evicted before engine load").
Same box, same binary, same hour:

| | rotating | cold TTFT |
|---|---:|---:|
| with eviction (the default, and every historical row) | 1.74–1.91 | 31–35 s |
| `ROME_BENCH_NO_EVICT=1` | **3.91** | **5.3 s** |

That is a factor of two on the headline and a factor of six on cold, and it
means **an apparent movement in this number can be the disk rather than the
engine**. It is why 2026-09-10 spent an afternoon "bisecting" a 28% decode
regression that did not exist: the machine was reading at 723 MB/s steady-state
(measured directly, 20.5 GB, cache evicted, no encryption layer), against a
cold TTFT of 17.5–17.9 s in the 2026-09-04 rows — the same disk-dominated
metric, roughly 1.9× faster then.

The default is unchanged, because every row in the record was taken with
eviction and comparability matters more than the number being flattering. Use
`ROME_BENCH_NO_EVICT=1` when the question is about the ENGINE, and say which
one a row is in its config-name. If a rotating number moves and nothing in the
engine explains it, measure the disk before bisecting the code.

Rule of thumb: if the question is "is decode on short prompts still fast",
use `rome_bench.sh`. If the question is "does a real conversation still
start fast, and does the tool-block checkpoint still survive", use
`prefill_snapshot.sh`. **Before trusting or writing down ANY fresh number,
compare it to the nearest historical number produced by the SAME tool at the
SAME regime** — a number that implies a whole track's landed, gated work
regressed below where it started is almost never a real regression; it is
almost always a different regime being compared as if it were the same one,
or a config knob (`COLI_VK_EXPERTS2/3`, `COLI_KDA_GPU`, GPU-tier preload)
that silently didn't come up. Both tools now assert their own tier/config
came up and refuse to report a number if it didn't; neither tool used to
require that, which is how the 2026-09-10 bad numbers got as far as a commit.

## `rome_bench.sh` freezes GLM's routing history and not Qwen's (2026-09-11)

Found while taking §Q-PROFILE. `glm53` reads `COLI_USAGE_PATH` and the harness
points it at a per-run **copy**, so a GLM run cannot mutate the histogram the
next run preloads its tier from (§C0, §RP1 — 23.5 ms/token of apparent gain
once turned out to be exactly that). `qwen38` reads a **different variable**,
`COLI_USAGE`, defaults it to `<snap>/.coli_usage`, and **rewrites it at exit**;
the harness sets nothing for qwen38. So every Qwen row ever recorded mutated
what the next one preloaded from.

It has not bitten, and §Q-REBASE says why: Qwen's tier is VRAM-limited, not
history-limited, and stops at `14673 of 21858` in every run regardless. That is
luck, not design. If you are taking Qwen numbers where the tier composition
could matter — anything touching expert placement, formats or VRAM — freeze it
yourself (`COLI_USAGE=/tmp/copy.bin`), the way
`tools/hot-expert/q_profile_run.sh` does. The harness default is left alone
deliberately: changing it would make new rows incomparable to every existing
one for no measured benefit.

Both tools are self-contained: `rome_bench.sh` takes `~/bench/.rig.lock`
itself and stops/restarts the owner's gateway around the measurement;
`prefill_snapshot.sh` never touches the gateway at all. Do not call
`python3 c/tools/datapoint.py` or `ttft_serve.py` directly for a number that
is going in the record — go through the wrapper, or the same class of silent
misconfiguration that produced the bad 2026-09-10 numbers will happen again.
