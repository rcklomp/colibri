# tools/hot-expert

Instrumentation and one-shot source patches from the investigation that produced
the hot-expert tier. **These are a lab record, not a supported build path.**

Each `patch_*.py` is a small script that does exact-string replacements on
`c/glm53.c` (a couple also touch `c/backend_vulkan.c`). They hardcode absolute
paths and assert on every anchor, so they fail loudly rather than half-applying.
The changes they make are already committed to the source — you do not need to
run them to build. They are kept because they document *how* each measurement
was taken, and because the instrumentation ones are useful again the next time
someone profiles this path.

## Instrumentation (apply, measure, `git checkout --` to revert)

| script | what it adds |
| --- | --- |
| `patch_io.py` | Times `expert_read`, prints `[IO] expert_read wall=… n=… avg=…ms`. This is how the page-cache problem was found. |
| `patch_other.py` | Wall-clock timers around the attention call sites and `ffn_layer`; prints `[OTHER] attn=… ffn=…`. Independent of Vulkan. |
| `patch_score.py` | Dumps the top-16 `(expert, score)` per `(layer, token)` to `$COLI_TRACE` as `S <layer> <token> <rank> <eid> <score>`. |
| `patch_trace.py` | Routing trace + resident-set dump (`$COLI_TRACE`, `$COLI_DUMP_RESIDENT`). |
| `patch_histgpu.py` | Collects an expert-usage histogram for the *current* config into `$COLI_USAGE_OUT`, separate from the ranking map read from `$COLI_USAGE_PATH`. |
| `patch_hist.py` | Older histogram collector; writes back to `$COLI_USAGE_PATH`. |

`analyze_scores.py` compares two `S`-format traces positionally and reports
per-rank differences, score-delta buckets, top-8 set churn, layer concentration,
and how well a heat map collected under one config transfers to another:

```
python3 analyze_scores.py trace_a.txt trace_b.txt LABEL_A LABEL_B
```

Compare **positionally, in file order** — the token index resets each forward
call, so `(layer, token)` is not a unique key across decode steps.

## Feature patches (already committed; kept for provenance)

`patch_hybrid.py`, `patch_dev2.py`, `patch_dev2fix.py`, `patch_dev3_backend.py`,
`patch_dev3_glm.py`, `patch_cachefix.py`, plus earlier exploratory ones
(`patch_fused.py`, `patch_hotexpert.py`, `patch_lastmile.py`, `patch_perlayer.py`,
`patch_poc*.py`, `patch_scores.py`).

Apply order if you ever need to rebuild the stack from a clean tree:
`patch_hybrid` → `patch_dev2` → `patch_dev2fix` → `patch_dev3_backend` →
`patch_dev3_glm` → `patch_cachefix`.

## The one lesson worth carrying forward

The engine prints a `teacher_forcing` line: the model's `argmax` at every prompt
position. **Diff it against a CPU-only run before believing any speedup.**

The two-device expert dispatch once produced literal `!!!!!!!` output while
reporting *better* tokens/sec *and* a *higher* cache hit rate than the
single-device path. Both headline metrics moved the right way while the model
was computing garbage, because corrupted hidden state collapses routing onto a
small set of experts — which looks exactly like good cache locality. The
`teacher_forcing` diff caught it in a single run.

Caveat: that line is only a valid oracle across runs with the **same resident
set**. Changing which experts are GPU-resident changes which of them take the
GPU vs CPU numeric path, which can legitimately shift the argmax at
near-tied positions.
