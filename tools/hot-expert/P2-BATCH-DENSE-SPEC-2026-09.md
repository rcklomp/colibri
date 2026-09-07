# P2 — batch the dense stages of glm53 prefill (spec, 2026-09-06)

Roadmap: `PREFILL-ROADMAP-2026-09.md` item P2. Tier: Opus implements from this
spec; the spec is Fable's. Gate: `prefill_gate.sh` exits 0 (teacher-forcing
identical, logits within tolerance) and the printed TTFT deltas go in the
commit body. Nothing here changes decode: every change is inside
`if (tokens > 1)` or is a per-row-independent reformulation that is identical
at `tokens == 1`.

## What is being fixed

`forward_prefill` chunks the prompt (`GLM53_PREFILL_CHUNK`, default 128) and
hands `forward_span` S rows at a time, but every stage below `run_layers`
consumes them one row at a time (`for (t = 0; t < tokens; t++)`) and issues
M=1 work: one GPU submit per token per matrix, one OpenMP team per token.
Measured (§PREFILL-PROFILE in the record): 114 ms/token of ordinary matmul
and elementwise work that does not batch. The kernels already accept S rows:

| kernel | S support today | used at |
|---|---|---|
| `coli_vk_matmul(tensor, y, x, w, s, fmt, S, I, O, gs)` | yes — dispatch `(O/8, S, 1)`, per-row independent, **bit-identical per row for any S** | `mv()` passes 1 |
| `coli_vk_matmul_pair(..., x, S, I, grp)` | yes | `shared_gate_up_gpu` passes 1 |
| `coli_vk_matmul_multi` / `coli_vk_kda_layer` | **no** — 1 row by construction | KDA per-token chain |
| `matmul_i4_grouped` / `matmul_q` / `matmul` (quant.h) | yes | `mv_cpu` passes 1 |

## Design, stage by stage

Order of implementation = order of expected payoff / risk. Each sub-step is
separately gateable; land them as one series, gate after each.

### P2.1 — `kda_layer` for S > 1 (≈ 35 of 54 ms/token)

Today, `COLI_KDA_GPU=2` (the serving default) runs the whole per-token chain
in one submit per token: 8 projections → decay → recurrence → head-norm →
`ko`. For S > 1 restructure into five stages; the recurrence is the only
sequential one.

```
A  (GPU, batched)   qkv[S,3P], low[S,D], beta[S,H], lowg[S,D] = 6 matmuls over all S rows
                    decay[S,P] = kfb · low ; gate[S,P] = kgb · lowg   (2 more, chained on A's outputs)
B  (CPU, per row, parallel over t)   decay = gate_lb·σ(exp(alog[h])·(decay+dt)); beta = σ(beta)
C  (CPU, SEQUENTIAL over t)          coli_kda_step(core_t, state, window, qkv_t, conv, decay_t, beta_t, ...)
D  (CPU, per row, parallel over t)   normed_t = headnorm(core_t)·onorm·σ(gate_t)
E  (GPU, batched)                    out[S,hidden] = ko · normed[S,P]
```

- **A / E** call `coli_vk_matmul` once per matrix with `S` rows (8 + 1 calls
  per layer per chunk instead of 8 + 1 per token). x is the S×hidden slab
  `x` already is; outputs are S×O slabs. On the CPU fallback (`COLI_KDA_CPU`
  or no VK) the same slab call goes to `matmul_i4_grouped(..., S, ...)`.
  Optional later: extend `coli_vk_matmul_multi` with an S parameter so A is
  one submit — not needed for the gate.
- **C** is `coli_kda_step` exactly as the CPU path runs it today, token by
  token, on the **CPU copy** of the state. When the state lives on the
  device (`gpu == 2 || gpu == 3`): before stage C call
  `coli_vk_kda_sync(layer, state, window)` (device → host), run C for the S
  rows, then `coli_vk_kda_upload(layer, state, window)` (host → device) so
  decode resumes on the device from the chunk's final state. Two 4 MB copies
  per layer per chunk (state = H·D·D floats). Both functions exist (G12).
- **Numerics.** With `COLI_KDA_GPU=0` every stage is the pristine CPU code
  in the pristine order → **bit-identical**; the gate's (a) must pass
  unchanged. With `COLI_KDA_GPU=2` prefill moves from the GPU chain (GLSL
  exp, tree-reduced norms) to the CPU recurrence — i.e. *closer* to the
  pristine CPU numerics, and decode is unchanged. The gate's (b) tolerance
  covers it; state the cosine in the commit body. Do not try to keep the
  per-token GPU chain for prefill "for consistency": it is the thing being
  removed.
- **tokens == 1** must take the existing per-token path untouched (decode).
- Scratch: `qkv/low/beta/lowg/decay/gate/core/normed` become S-row slabs;
  allocate once per call (S ≤ chunk ≤ 128; P = H·D).

Expected: kda.proj 22 → ~1–2 ms/token, kda.ko 13 → <1, step unchanged 13,
B+D ~1. **KDA 54 → ~17 ms/token.**

### P2.2 — shared expert for S > 1 (≈ 13 of 14 ms/token)

`mlp3_shared(out_t, x_t, l, ...)` per token → one call on the S×hidden slab:
`shared_gate_up_gpu` gets an `S` parameter and passes it to
`coli_vk_matmul_pair` (already takes S); `swiglu_clamped` over S·rows; `rd`
via `coli_vk_matmul` with S rows. CPU fallback: the three `matmul_*` with S.
Bit-identical per row. **14 → ~1–2 ms/token.**

### P2.3 — router and hyper-connection loops (≈ 12 of 15 ms/token)

- Router: keep the exact scalar dot per (token, expert) — the reduction
  order must not change or a near-tie flips an expert and (a) fails — but
  move the `#pragma omp parallel for` from the inner `e` loop to the outer
  `t` loop with a per-thread `score[288]`. One team per chunk instead of one
  per token. Bit-identical. 5.5 → <1 ms/token.
- `run_layers`: the three `for (t) coli_hc_pre/rms/coli_hc_post` loops get
  `#pragma omp parallel for` over `t` (per-row outputs are disjoint slices;
  `coli_hc_pre` writes only its `t` slices of collapsed/post/comb). Bit-
  identical. 9 → ~2 ms/token.
- `forward_span` tail: collapse + final rms + head per row — parallel over t
  for the first two; for the head, when the caller only keeps the last row
  (`keep_all == 0`, the serve path) compute logits for the last row only
  (add a `last_only` flag; the teacher-forcing oracle passes `keep_all=1`
  and is unaffected). Head is 3 ms/token on the GPU — small but free.

### P2.4 — `mla_layer` projections for S > 1 (≈ 5 of 7 ms/token)

Stage the 5 independent projections (`qa, kva, iwk, ikpg, iwp`) and then the
2 chained ones (`qb, iwq`) as S-row `coli_vk_matmul` calls; the per-row
`rms`/`layer_norm` in between run parallel over t. The absorb loop
(`mv_rows` per head per token) and the sparse attention stay per token in
P2 — they are P5's. Bit-identical. mla.proj 7 → ~2 ms/token.

## Expected total

| bucket | before | after P2 | note |
|---|---:|---:|---|
| kda | 54 | ~17 | recurrence remains |
| shared | 14 | ~2 | |
| router + hc | 15 | ~3 | |
| mla.proj | 7 | ~2 | |
| unchanged (moe cpu/eg, mla.attn, step) | ~94 | ~94 | P3/P4/P5 |
| **total** | **~184** | **~118** | **≈1.55×** on every prefill |

The gate asks ≥ 1.3× TTFT at 300 and 1 000 tokens, both runs.

## Oracle and gate procedure (no exceptions)

1. Build the candidate as `c/glm53`; keep the pristine binary as
   `~/bench/glm53.pristine` (copy it before the first edit — `md5sum`
   4e39b6b0c0d34565d1185e2cc2702454 is the 2026-09-06 11:21 build).
2. Stop the gateway; `pgrep -x glm53` must be empty.
3. `tools/hot-expert/prefill_gate.sh ~/bench/glm53.pristine ~/src/colibri/c/glm53 p2.N`
   — (a) teacher-forcing identical with `COLI_KDA_GPU=0` (export it for the
   oracle half; the gate's TTFT half runs the serving default), (b) logits,
   (c) TTFT twice per size.
4. `tworeq.py` (three identical requests in one persistent engine) for
   P2.1 — it gives per-conversation state a new home (the CPU/device
   sync), which is exactly the class of bug it was written for.
5. Commit body: the gate's printed table, the cosine, the residency line.
6. Restart the gateway (`~/start_glm53.sh`) and confirm one request from
   Open WebUI completes — rule 5 of the roadmap's discipline.

## Non-goals

- The expert paths (`mlp3_cpu` rows → P3; expert shader S-tile → P4).
- The recurrence and the attention (P5).
- Any change to `forward_prefill`'s chunking or to decode.
