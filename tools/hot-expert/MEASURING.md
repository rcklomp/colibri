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

Both tools are self-contained: `rome_bench.sh` takes `~/bench/.rig.lock`
itself and stops/restarts the owner's gateway around the measurement;
`prefill_snapshot.sh` never touches the gateway at all. Do not call
`python3 c/tools/datapoint.py` or `ttft_serve.py` directly for a number that
is going in the record — go through the wrapper, or the same class of silent
misconfiguration that produced the bad 2026-09-10 numbers will happen again.
