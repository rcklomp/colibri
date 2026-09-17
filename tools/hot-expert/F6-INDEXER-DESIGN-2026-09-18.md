# F6 design: GLM-5.3-Flash's DSA indexer at depth (Opus, 2026-09-18)

> Design step of item **F6** (`tools/hot-expert/FRANKEN-ENGINE-PLAN-2026-09-15.md`
> §8.3, promoted from §4 row X3 by F4's verdict, rev 14). **No rig time was
> spent**: everything below is read out of `c/glm53.c`, `c/sparse_index.h`,
> `c/backend_vulkan.c` and the numbers already in
> `tools/hot-expert/ROME-3x7900XTX-2026-09-04.md`. One read-only `sed` of
> `~/bench/ctx_ladder_out/f409171830_F4_engine.log` was used to confirm the
> record's verbatim tables and to read the `REUSE` lines beside them.
>
> Every figure is labelled **measured** (with its record section) or
> **projected** (with the derivation shown inline). A projection is a thing to
> be beaten or refuted by the item's own A,B,B,A row, never a result.
>
> **The design's three headline conclusions, up front:**
>
> 1. The indexer is **54 % of the decode token at 18.6k** — 226 ms of a 452 ms
>    token — and it is **compute/latency-bound on ONE core at ~1 GMAC/s**,
>    streaming 116 MB/s where this box reads DRAM at 91.6 GB/s. It is not a
>    bandwidth problem and it is not an algorithm problem yet: seven of eight
>    cores are idle inside it.
> 2. The recommended first step is therefore **not** the stricter selection X3
>    was written as. It is **§G7's pattern applied to the score loop** —
>    `#pragma omp parallel for` over the pools, plus an exact partial top-512 —
>    **bit-identical, no numerics change, no KL bar needed, Sonnet not Opus**.
>    Projected **25.0 → 3.3 ms per MLA call at ctx 18 630**, decode indexer
>    226 → 30 ms/token, token 452.6 → 256 ms, **2.21 → ~3.9 tok/s**.
> 3. **X3's named graft is already in the engine.** "pool 4 → 1 key, select 512
>    blocks" is exactly what `coli_sparse_index_select_range_cached` does today
>    (`pool = 4`, `wanted = topk/pool = 512`). The remaining algorithmic lever
>    is a *second* level, not the first, and it is worth less than the free
>    parallelism. See §3(ii).

---

## 1. What the indexer does per call today

### 1.1 The call site, and what `index` times

`mla_layer` (`c/glm53.c:1798`) runs on the 11 full-attention layers of 45
(layer 3, 7, … 43; `c->is_full[]`, `c/glm53.c:109`). The indexer is one call
per layer per forward, at `c/glm53.c:1896-1905`:

```c
1896    const int width = coli_sparse_index_width(c->index_topk, c->index_kpool, c->index_kpool_tail);
1897    int *selected = malloc((size_t)tokens * width * sizeof(int));
1898    const int index_rc = index_cache_on()
1899        ? coli_sparse_index_select_range_cached(selected, iq, ik, gates, head_w, l->ikpa,
1900                                       valid, seen, IH, ID, c->index_kpool, c->index_topk,
1901                                       c->index_kpool_tail, base, seen,
1902                                       st->pool_cache, &st->pool_cache_count)
1903        : coli_sparse_index_select_range(selected, iq, ik, gates, head_w, l->ikpa, valid,
1904                                       seen, IH, ID, c->index_kpool, c->index_topk,
1905                                       c->index_kpool_tail, base, seen);
```

The `index` bucket of `[OPTIME] mla split` is bracketed at `c/glm53.c:1894-1895`
and `c/glm53.c:1920` (`g_mt_index`, declared `c/glm53.c:1022`). It therefore
covers **exactly** the `malloc` of `selected`, the call above, and nothing else.
It does **not** include the five projections that produce `iq`/`ik`/`gates`/
`head_w` (those are `proj`, `_tm0`→`1894`) and it does not include the
attention core (`attn`, `_tm2`→`c/glm53.c:2112`). The selection is the whole of
`index`.

Note for §1.7: the counters at `c/glm53.c:2112` are `g_mn_calls++` and
`g_mn_seen += seen` — **once per call, not once per query row** — and
`index/ctx` is printed as `1e6 * g_mt_index / g_mn_seen`
(`c/glm53.c:4036-4042`). `mean ctx` is a mean over *calls*. This is the whole of
the 83-vs-25 puzzle.

### 1.2 Inputs and shapes

From the signature (`c/sparse_index.h:196-203`) and its parameter table
(`c/sparse_index.h:51-57`), with GLM's names from `c/glm53.c:104-105, 301-305`:

| parameter | shape | source | lives in |
|---|---|---|---|
| `queries` = `iq` | `[tokens][IH][ID]` f32 | `l->iwq` on the normalised q_a, this call | host `malloc`, this call |
| `keys` = `ik` | `[cap][ID]` f32, whole prefix | `st->ikeys`, appended per token | **host RAM** (`c/glm53.c:3915`) |
| `gates` = `igates` | `[cap][ID]` f32, whole prefix | `st->igates`, appended per token | **host RAM** (`c/glm53.c:3916`) |
| `head_w` | `[tokens][IH]` f32, already `IH^-0.5`-scaled (`c/glm53.c:1891`) | `l->iwp` | host, this call |
| `ape` = `l->ikpa` | `[pool][ID]` f32 — **a trained tensor**, `index_kpool_compress_ape` | checkpoint (`c/glm53.c:414, 500`) | host, weights |
| `pool_cache` | `[pools_cap][ID]` f32 (G5) | built here, per session | **host RAM** (`c/glm53.c:3926-3928`) |
| `valid` | `[seen]` u8, `memset` to 1 every call (`c/glm53.c:1812-1813`) | — | host, this call |

Constants for this model: `pool = index_kpool = 4`, `topk = index_topk = 2048`,
`wanted = topk/pool = **512**` (`c/sparse_index.h:215`),
`width = topk + pool − 1 = **2051**` (`c/sparse_index.h:38-40`, tail always
selected), `ID = index_hd = **128**`, 11 DSA layers.

`ID = 128` is **derived exactly**, not assumed: the `cache:` line prints
`full × (kv_lora + 2 × index_hd) × 4` bytes (`c/glm53.c:3957`), the run's own
value is 33.0 KB/token over 11 DSA layers (§X3 step 0), and
33.0 × 1024 / (11 × 4) = 768 = 512 + 2 × 128. `topk = 2048` and `pool = 4` are
§P5b's own shapes (record line 3793: "width = index_topk 2048 + kpool 4 − 1 =
2051").

**`IH = index_n_heads` is the one shape this design could not read from the
repo.** It is printed by the engine's own banner
(`indexer  : top-%d, %d heads x %d, kpool %d`, `c/glm53.c:349-358`) and is in
the first ~30 lines of any `GLM53_VERBOSE=1` log — step 0 reads it for free.
The arithmetic below is given for both plausible values; **the recommended
design's factor does not depend on it** (it is a ÷7.3 on whatever the scoring
term is). §1.7 shows that `IH = 32` is the value that makes the measured rate
equal §G3's measured rate for the identically-shaped router loop.

### 1.3 The pooled-key structure it scans

`pool_cache` is the G5 cache (§G5): one **f32** mixed key per pool of 4 context
tokens per DSA layer, `[pools_cap][ID]`, host `malloc`, one array per layer per
session (`c/glm53.c:659, 3926-3928`). Its content is a *per-channel softmax
mixture* of the pool's 4 member keys, weighted by `gates + ape`
(`c/sparse_index.h:246-264`) — not an average, so a pool is not forced to
describe itself by its mean (`c/sparse_index.h:8-13`).

**Bytes per context token per layer = `ID × 4 / pool` = 128 × 4 / 4 = 128 B.**
So:

| ctx | pooled keys / layer | bytes / layer | bytes across 11 layers |
|---:|---:|---:|---:|
| 2 051 (the cap) | 513 | 262 KB | 2.9 MB |
| 18 630 | 4 658 | **2.385 MB** | **26.2 MB** |
| 128 000 (model cap) | 32 000 | 16.4 MB | 180 MB |

At 18.6k the whole scanned structure is **26.2 MB against this box's 128 MB
L3** — it is not a DRAM working set. For comparison the latent+indexer KV it
is derived from is 33.0 KB/token = 608 MB at 18.4k (§X3 step 0).

Since G5 the mixing is amortised: a pool's mix is written once, when the pool
closes (`c/sparse_index.h:238-267`, `frontier`), and never recomputed
(`GLM53_NO_INDEX_CACHE=1` restores the old always-recompute path,
`c/glm53.c:1789-1796`). In decode that is 1 new pool per 4 tokens per layer.
§G5 measured the *whole* recompute at ctx 786 as 1.549 − 0.615 = 0.934 ms for
197 pools = **4.77 µs per pool**, so the amortised decode cost is
4.77/4 ≈ **1.2 µs per token per layer, ≤ 0.6 % of the call** at depth. The
mixing half is done; it is not where F6's ms are.

### 1.4 The score computation (the hot loop)

`c/sparse_index.h:269-288`, for each query row `q` in `[q_from, q_to)`:

```c
275        for (int p = 0; p < pools; p++) {
276            const int last = first + (p + 1) * pool - 1;
277            if (!complete[p] || last > q) { scores[p] = -FLT_MAX; continue; }
278            const float *pv = pool_cache + (size_t)p * dim;
279            float score = 0.0f;
280            for (int h = 0; h < heads; h++) {
281                const float *query = queries + ((size_t)(q - q_from) * heads + h) * dim;
282                float dot = 0.0f;
283                for (int d = 0; d < dim; d++) dot += query[d] * pv[d];
284                if (dot > 0.0f)                                  /* ReLU */
285                    score += head_w[(size_t)(q - q_from) * heads + h] * dot * scale;
286            }
287            scores[p] = score;
288        }
```

`pools = (sequence + pool − 1) / pool` (`c/sparse_index.h:214`), computed from
`seen` — **the loop runs over every pool in the sequence, for every query row,
on every call.** Per query row per layer: `pools × IH` dots of length `ID`, i.e.
**`pools × IH × 128` MAC** and `pools × 128 × 4` bytes of `pool_cache` re-read.
This is the half G5 explicitly could not cache ("genuinely query-dependent and
stays O(pools) per token", §G5) and it is single-threaded: **there is no
`#pragma omp` anywhere in `c/sparse_index.h`** — the file has none, at any line.

### 1.5 The selection

`c/sparse_index.h:290-299`, per query row: a greedy repeated-maximum, `wanted`
= 512 passes over all `pools`:

```c
290        for (int rank = 0; rank < wanted; rank++) {
291            int best = -1;
292            for (int p = 0; p < pools; p++)
293                if (!taken[p] && scores[p] > -FLT_MAX &&
294                    (best < 0 || scores[p] > scores[best])) best = p;
295            if (best < 0) break;
296            taken[best] = 1;
297            for (int j = 0; j < pool; j++) row[rank * pool + j] = first + best * pool + j;
298        }
299        memset(taken, 0, (size_t)pools);
```

**`wanted × pools` iterations per query row** = 512 × 4 658 = **2.385 M** at
ctx 18 630. Also single-threaded, also scalar (the `best` carry makes it a
sequential reduction). Then the incomplete tail is appended unconditionally
(`c/sparse_index.h:301-309`), which is why the row is `topk + pool − 1` wide.
Ties go to the lower pool index because the scan takes the **first** strict
maximum — selection is deterministic (`c/sparse_index.h:22`).

Two properties of this loop matter for every candidate design:

- the row is written **rank-ordered** (`row[rank * pool + j]`, line 297): slot 0
  holds the best pool's 4 tokens, slot 4 the second best, and so on;
- the filter is `scores[p] > -FLT_MAX` and the comparison is `>` only — NaN and
  exact `-FLT_MAX` semantics are load-bearing and any replacement must
  reproduce them.

### 1.6 What it hands the attention core, and why the order is load-bearing

`selected[tokens][2051]` of absolute token indices, `-1` where unused. GLM does
**not** consume it through `coli_sparse_attention_range`
(`c/sparse_index.h:336-383`, which `c/glm53.c` never calls); it compacts the
row itself at `c/glm53.c:1969-1982`:

```c
1970        const int *chosen = selected + (size_t)t * width;
...
1977            for (int i = 0; i < width; i++) {
1978                const int at = chosen[i];
1979                if (at < 0 || at >= seen) continue;
1980                slot_at[used_all++] = at;
1981            }
```

`slot_at` therefore inherits the row's **rank order**, and that order is the
order in which the softmax's `expf` terms are summed into `total` and in which
`glm_pool_blocked` walks the latent (`c/glm53.c:1996-2020`). **A design that
preserves the selected *set* but changes its *order* is not bit-identical** —
not even below the dense threshold of §2. This is the first thing F6's gate
must check rather than assume.

### 1.7 Which step the 4.46 µs/token/call slope pays for — and the 83 vs 25 ms reconciliation

**The reconciliation, first, because it is the reason the naive figure is
wrong.** `index/ctx` is `g_mt_index / Σ_calls(seen)` (`c/glm53.c:4042`), so it
is µs per *call*-context-token and is blind to how many query rows a call
carries. The cost, from §1.4, is `c × rows × ctx` per layer. Writing `R̄` for
rows per call, `ctx̄r` for the row-mean context and `ctx̄c` for the printed
call-mean:

    index/ctx  =  c × R̄ × (ctx̄r / ctx̄c)
    ms/call    =  c × R̄ × ctx̄r

Both windows, from the same engine log (`f409171830_F4_engine.log`; the `REUSE
N <reused> <total>` line immediately above each table gives the new-token count,
and `forwards` divided into `n/11` gives the call count):

| | req 2 (turn 1) | req 7 (follow-up) |
|---|---:|---:|
| ctx | 1 287 | 18 586 |
| `REUSE` | (after the 13-token warm-up) | `18566 18586` → **20** new tokens |
| new prompt rows | 1 274 | 20 |
| generated rows | 128 | 88 |
| **query rows** | **1 402** | **108** |
| forwards (= n/11) | 138 | 89 |
| **R̄ rows/call** | **10.16** | **1.213** |
| ctx̄r (row-mean ctx) | 708 | 18 584 |
| ctx̄c (`mean ctx`, printed) | 1 304 | 18 630 |
| `index` ms/call (**measured**, §X3 step 0) | **5.810** | **24.977** |
| `index/ctx` µs (**measured**) | **4.4551** | **1.3407** |
| **c, µs per row × layer × ctx-token** (derived) | **0.808** | **1.108** |

Three factors, and their product is exact:

| factor | value | why |
|---|---:|---|
| rows per call | **÷ 8.38** | req 2's window is a 1 274-token fresh prefill in ~10 chunks of 128 rows plus 128 one-row decode steps; req 7's is 20 new tokens in one forward plus 88 one-row decode steps |
| printed `mean ctx` vs the row-mean that drives the cost | **× 1.84** | in req 2 the 128 deep single-row decode calls dominate the *call*-count mean (1 304) while carrying 9 % of the rows, so the row-mean is 708. In req 7 the two coincide (18 584 vs 18 630) and the printed slope is honest |
| the constant c | **× 1.371** | 0.808 → 1.108 µs as the scanned `pool_cache` grows from 167 KB (L2-resident, 512 KB/core on Zen 2) to 2.385 MB (L3). §G5's own decode-only pair moves the same way and by a similar amount: 0.783 µs at ctx 786 → 0.867 at 1 514 |

    83.1 ms/call (= 18 630 × 4.4551 µs)  ×  (1/8.38)  ×  1.84  ×  1.371  =  25.0 ms/call

against **24.977 measured**. There is no discrepancy in the instrument and no
sub-linearity in the algorithm: the slope is per *row*, the ms/call figure folds
in rows-per-call, and `mean ctx` is a call-mean.

**A correction to §X3 step 0's verdict, from the same log.** The verdict's
lower bound ("if the indexer's cost were linear in rows, the decode-only share
would be as low as 24.45 s / (25·6 + 64) ≈ 114 ms/token = 26 %") took the
follow-up prompt as **147 new tokens in ~25 forwards**. 147 is
18 586 − 18 439, the growth against the *previous turn's* context; the prefix
the engine actually restored was 18 566 tokens (`REUSE 7 18566 18586`), because
turn 6's own 128 generated tokens extended it. The real window is **20 new
prompt rows and 88 decode rows in 89 forwards**. The cost *is* linear in rows,
and per row it lands on the verdict's **upper** figure, not its lower one:

- 24.453 s / 108 rows = **226.4 ms per row** of indexer;
- the window's whole `layers` bucket is 45.282 s / 108 = **419.2 ms per row**,
  against a **measured** decode token of **452.6 / 452.5 ms** at ctx 18 586
  (§X3 step 0, A-arm turn 6) — 8 % apart, so **in this window a row is a
  token**, and the per-forward table in the record is the same table × 1.215.

So the decode-token decomposition at ~18.6k, per query row, from `req=7`
(every figure = the record's own bucket total ÷ 108):

| bucket | ms/row ≈ ms/token | share of the 419 ms |
|---|---:|---:|
| **mla `index` (the DSA selection)** | **226.4** | **54.0 %** |
| `ffn_moe` (router + shared + experts, GPU and CPU) | 102.8 | 24.6 % |
| mla `attn` (the core) | 34.9 | 8.3 % |
| `kda` (34 layers) | 34.8 | 8.3 % |
| `hc+norm` | 9.8 | 2.3 % |
| mla `proj` | 7.0 | 1.7 % |
| `ffn_dense` | 3.5 | 0.8 % |
| `head` (per forward, outside `layers`) | 3.4 | — |

**The figure the design must beat: 25.0 ms per MLA call at ctx 18 630
(measured, §X3 step 0 req 7; 24.75 at req 8) = 20.6 ms per query row per DSA
layer = 226 ms/token of decode = 54 % of the token.**

**Which step pays it.** Of the 20.6 ms per row per layer:

- the **score pass** (§1.4): `pools × IH × 128` MAC = 19.1 M (`IH = 32`) or
  38.1 M (`IH = 64`);
- the **greedy top-512** (§1.5): 2.385 M scalar iterations. At 1–3 cycles per
  iteration on a 3.7 GHz core that is **0.64–1.93 ms, i.e. 3–9 %**;
- the **tail + row init + `memset(taken)`**: 2 051 int stores, 4 658 bytes, and
  the `visible` count loop over `ctx` (`c/sparse_index.h:302-303`, an integer
  reduction GCC does vectorise) — together **≤ 0.03 ms, < 0.2 %**;
- the **pool mixing**: ≤ 0.6 % since G5 (§1.3).

So **91–97 % of the 20.6 ms is the score pass**, and it runs at
19.1 MMAC / 19.3 ms = **0.99 GMAC/s** at `IH = 32`, or 1.98 at `IH = 64`.
§G3 measured the MoE router — the *same* shape, "a hand-rolled `for e<288: for
d<4096: sum+=` in f32; a scalar reduction GCC will not vectorize" — at
**1.2 GMAC/s** single-threaded on this box. The indexer's dot is that loop with
`d < 128`, and the build carries `-O3 -march=native -fopenmp` and **no
`-ffast-math`** (`c/Makefile:106`), so GCC may not reassociate the f32
reduction and cannot vectorise it. `IH = 32` reproduces §G3's measured rate to
18 %; `IH = 64` would mean the dot is running ~1.7× faster than the router's.
Step 0 settles it from the banner; it changes only §3(iii)'s projected factor.

### 1.8 Bandwidth-bound or compute-bound: compute, by two orders of magnitude

| quantity at ctx 18 630 | value | derivation |
|---|---:|---|
| bytes of `pool_cache` scanned per query row per layer | 2.385 MB | `pools × ID × 4` = 4 658 × 128 × 4 |
| bytes per **call** (R̄ = 1.213 rows) | 2.893 MB | × 1.213 |
| time per call (**measured**) | 24.977 ms | §X3 step 0 req 7 |
| **achieved byte rate** | **116 MB/s** | 2.893 MB / 24.977 ms |
| this box's 8-thread DRAM read (**measured**, §PCIE-STREAM) | 91.6 GB/s | — |
| the CPU int4 expert path's achieved rate (**measured**, §G11/§G3) | 21.95 GB/s | — |
| **fraction of the int4 path's byte rate** | **0.53 %** | — |
| MAC rate, single thread | 0.99–1.98 GMAC/s | §1.7 |
| one Zen 2 core's AVX2 FMA peak | 16 MAC/cycle ≈ 59 GMAC/s | 2 × 256-bit FMA at 3.7 GHz |
| **fraction of one core's vector peak** | **1.7–3.4 %** | — |
| **fraction of eight cores' vector peak** | **0.2–0.4 %** | — |

And the working set is 26.2 MB across the 11 layers, inside the 128 MB L3, so
it is not even a DRAM problem. **The indexer is compute/latency-bound on a
single core, at a few percent of that core's vector throughput, with seven
cores idle.** Two orthogonal factors of ~8 and ~4 are available before any
algorithmic change: threads, and the reduction's vector width. §G3 said the
same thing about this engine as a whole — "the single largest fact in this
profile is that 69 % of the decode token runs on one core" — and the indexer is
the last large serial term left after G4/G7/G8/G9/G10.

---

## 2. Why it grows linearly, and what the dense threshold implies

**Linear growth.** `pools = ceil(sequence / 4)` (`c/sparse_index.h:214`) and the
score pass runs over all of them for every query row
(`c/sparse_index.h:275`). The score of pool `p` depends on the querying token
(`queries` is indexed by `q` at line 281), so no part of it can be hoisted out
of the query loop — that is G5's own stated limit. Per token the work is
Θ(ctx/4 × IH × 128); per generation it is Θ(ctx²). Measured: c is flat within
37 % over a 14× context range (§1.7), i.e. the growth is linear in ctx with a
mild cache-tier term, exactly as §"Long context" first measured at ctx 52/502.

**The attention core, by contrast, is capped — and this design pass confirms it
measured, which §"Long context" asked for and never got.** `width = topk + pool
− 1 = 2051` is a constant (`c/sparse_index.h:38-40`), `used_all ≤ width`
(`c/glm53.c:1977-1981`), so `attn` cannot grow past the cap. §"Long context"
left the flattening "derived from the code, not measured", with the warning
"if `attn` does *not* flatten near 2k, this whole ranking inverts". Row-
normalising §X3 step 0's four deep tables (bucket total ÷ rows, rows from the
`REUSE` lines as in §1.7):

| window | row-mean ctx | `attn` ms/row |
|---|---:|---:|
| req 3 | 1 971 | 29.2 |
| req 5 | 7 377 | 29.5 |
| req 6 | 14 029 | 34.8 |
| req 7 | 18 584 | 34.9 |

**Flat within 20 % over a 9.4× context range**, where linear growth would have
been 9.4×. The cap holds. §"Long context"'s caveat is closed, at zero rig cost,
and the ranking it worried about stands: above the cap the indexer is the only
term that still grows.

**The dense-identical threshold.** For query `q` the selectable pools are those
whose last member is causally visible (`last > q` → `-FLT_MAX`, line 277), i.e.
`floor((q+1)/4)` of them. That is `≤ wanted = 512` exactly while `q ≤ 2050`,
i.e. **while the sequence is ≤ 2 051 tokens** (the plan's "below 2 052"). Below
it every visible complete pool is selected and the selection is a no-op **as a
set, by construction**.

What that implies for a stricter scheme, and it is sharper than the plan's gate
line assumes:

1. **Below 2 052 the item is worth nothing and above it there is no guarantee
   at all.** The whole prize and the whole numerics risk live in the same place.
   The "dense-identical below 2 052" leg is a *cheap sanity check that the
   implementation did not break the easy case*, not evidence about the regime
   the item exists for. It must be run (the plan says "checked, not assumed" —
   correct) and it must not be reported as if it bounded the risk.
2. **It is a set property, not an output property.** The row is rank-ordered
   (§1.6), so even below 2 052 the *order* of `slot_at` is score-descending. A
   scheme that changes scores (a coarser summary, int8 keys, a GPU reduction) or
   changes the order (emitting in index order) is **not** dense-identical below
   2 052 either. Only a scheme that reproduces the exact scores *and* the exact
   rank order gets that leg for free — which is precisely the recommended
   design and precisely not the alternatives.
3. **X2's packet cannot exercise any of this.** `tools/hot-expert/x2_packet_450.txt`
   tokenises to 564 positions (§X2) — 3.6× below the threshold, where every
   candidate is a no-op. **A KL number over that packet would be a guaranteed
   0.0 and would prove nothing.** Any numerics-changing stage of F6 needs a new
   deep packet (≥ 20k positions) and its KL taken over the *decode* positions
   of a restored-prefix follow-up, not a fresh prefill — a fresh 20k prefill
   dump costs ~70 min per arm (§X3 step 0's own estimate). This is a second,
   independent reason to take the bit-identical win first.

---

## 3. Candidate designs

All projections are per **query row per DSA layer** from the measured
**20.6 ms** at ctx 18 630 (§1.7), then converted to ms/call in the `req=7`
window shape (× 1.213 rows/call) and to ms/token (× 11 layers). Decode token
base: **452.6 ms measured** at ctx 18 586 (§X3 step 0, A-arm turn 6, 2.21 tok/s);
the indexer is 226.4 ms of it.

| # | design | bytes/row/layer before → after | proj. ms/row/layer | proj. ms/call @18.6k | proj. ms/token | proj. tok/s @18k | numerics | text identical? |
|---|---|---|---:|---:|---:|---:|---|---|
| **(vi)** | **threads + exact partial top-512** (recommended) | 2.385 MB → 2.385 MB | **2.71** | **3.3** | **29.8** | **3.91** | none — bit-identical | **yes, by construction** |
| (i) | coarser pooling, pool 4 → 8/16 | → 1.19 / 0.60 MB | 10.3 / 5.2 | 12.5 / 6.3 | 113 / 57 | 2.94 / 3.30 | large | **REJECTED at design time — see below** |
| (ii) | two-level select (super-blocks of 16 pools, coarse pass with one aggregated query, fine pass over the top 64) | 2.385 MB → 0.56 MB | 4.5 | 5.5 | 50 | 3.61 | changes the selected set | no |
| (iii) | int8 / fp16 pooled keys | → 0.60 / 1.19 MB | ~20.6 alone | ~25 alone | ~226 alone | 2.21 alone | score quantisation | no |
| (iv) | the scan on dev0 (new shader + device-resident `pool_cache`) | 2.385 MB device-side | 0.3–1.0 | 0.4–1.2 | 3–11 | 4.28–4.36 | GPU reduction order | no |
| (v) | re-select every N decode tokens (N = 4) | 2.385 MB / 4 amortised | 5.6 | 6.8 | 62 | 3.52 | stale selection | no |
| **(vi)+(ii)** | both | → 0.56 MB | 0.75 | 0.9 | 8.3 | **4.34** | (ii)'s | no |
| **(vi)+(v)** | both, N = 4 | amortised | 0.86 | 1.0 | 9.5 | **4.32** | (v)'s | no |

tok/s column = 1000 / (452.6 − 226.4 + projected ms/token). The absolute
ceiling, indexer cost zero, is 1000/226.2 = **4.42 tok/s**.

### (vi) Threads on the score pass + an exact partial top-512 — **recommended**

Two changes inside `c/sparse_index.h`, both bit-identical:

**(a) `#pragma omp parallel for schedule(static)` on the pool loop**
(`c/sparse_index.h:275`). Each iteration writes only `scores[p]` and reads only
`pool_cache[p]`, `queries`, `head_w`, `complete[p]` — disjoint writes,
read-only sharing, and **no individual score's summation order changes at all**.
This is §G7 verbatim ("each row is independent … own `score[e]`, no shared
accumulator, so `#pragma omp parallel for` on the expert loop changes nothing
about any row's own summation, only the order rows are computed in"), which
measured **0.998 → 0.137 ms/call, 7.3× = 91 % of the theoretical 8×,
bit-identical**. For multi-row calls (prefill) the outer `q` loop is the better
axis (fewer barriers), which needs per-thread `scores`/`taken` slices — the
G4/G8 per-thread-scratch pattern, 64-byte aligned and cache-line-strided per
§P5b.1's false-sharing lesson. Pick the axis on `q_to − q_from ≥ threads`.

**(b) replace the O(`wanted × pools`) greedy scan** (`c/sparse_index.h:290-298`)
with a bounded min-heap of 512 keyed by `(score, −index)`, then a descending
sort of the 512 survivors. That is *exactly* "top 512 by score, ties to the
lower index, emitted in rank order", so the row — and therefore `slot_at`'s
order (§1.6) — is byte-identical. Cost: 4 658 compares + rare heap pushes +
512·log 512 ≈ 30 k ops, **≈ 0.05 ms** against 0.64–1.93. The `>`-only
comparison and the `> -FLT_MAX` filter must be carried over verbatim.

**Projection.** Score pass 19.3 ms / 7.3 = 2.64; selection 1.29 → 0.05; misc
0.02; one OpenMP region per layer per forward at §Q11's measured **8.9 µs per
site** (0.85 ms/token over 96 sites) = 0.009. Total **2.71 ms per row per
layer**, a **7.6×** cut: **25.0 → 3.3 ms/call**, **226.4 → 29.8 ms/token**,
token **452.6 → 256.0 ms**, **2.21 → 3.91 tok/s (+77 %)**.
Pessimistic arm (threads deliver only 4×, heap not done): 4.8 + 1.3 = 6.1
ms/row → 67 ms/token → 293 ms → **3.41 tok/s**. So **3.4–3.9 tok/s**.

**Upside not in the projection:** §G4, §G7 and §G8 each beat their own ms
arithmetic in the serving regime (G7: −36 ms/token predicted, +9–12 % serving
measured) because freeing single-threaded work also relieves the barrier spin
§G3 measured at 60.8 % of all cycles. The indexer is the largest serial term
left, so this effect should be at its strongest here.

**Falsifier:** the isolated microbenchmark (step 0.3) measures < 3× from 8
threads on the score pass — then the loop is not what §1.8 says it is and the
whole design is re-derived from that number. Second falsifier: the A,B,B,A
ladder's `index` ms/call at 18.6k does not fall by ≥ 4×, through
`gate_ab_verdict`.

**X2's KL bar: not applicable, and that is the point.** The gate is
`diff`-level bit-identity on `teacher_forcing` and `last_logits`, plus
`GLM53_DUMP_INDEX=1` row equality (§4).

### (i) Coarser pooling (pool 4 → 8 or 16) — REJECTED at design time

The arithmetic is attractive (`pools` halves, so both the score pass and the
selection halve: 20.6 → 10.3 ms/row at pool 8, and `wanted = topk/pool` keeps
the selected budget at 2 048 tokens, with `topk % pool == 0` still satisfied,
`c/sparse_index.h:69`). The accuracy cost is also arguable: the pooled key is a
*learned per-channel softmax mixture*, not a mean (`c/sparse_index.h:8-13`), so
it degrades more gracefully than an average would — a pool can still be
dominated by its most distinctive member.

**It cannot be built.** `ape` is `index_kpool_compress_ape`, a **trained tensor
of shape `[index_kpool][index_hd]` = [4][128]** loaded from the checkpoint only
when `index_kpool > 1` (`c/glm53.c:414, 499-501`) and consumed as
`ape[j * dim + d]` for `j < pool` (`c/sparse_index.h:250, 255, 259`). Pooling 8
members requires 8 rows of intra-pool positional bias that do not exist in the
weights. Tiling or interpolating the 4 trained rows is inventing weights, which
is outside anything this fork has shipped and is a far larger numerics change
than the 2× it buys. The honest version of "coarser" is a second level over the
*existing* pooled keys, which is (ii).

**Falsifier if anyone revisits it:** a checkpoint that ships an 8-row
`index_kpool_compress_ape`. Until then the row is closed.

### (ii) Two-level select (the hipEngine-QSA shape, one level up)

**First, the correction the plan's row needs:** X3 was written as "pool 4 → 1
key, select 512 blocks". That is what the engine already does — `pool = 4`,
one mixed key per pool, `wanted = topk/pool = 512` (`c/sparse_index.h:215`).
The graft named in §4 row X3 is *already in `c/sparse_index.h`*, and G5 already
cached its expensive half. What is left is a level above it.

**Design.** Group the `pools` pooled keys into super-blocks of `B = 16` (64
context tokens). Score each super-block once with a single aggregated query
(the `head_w`-weighted sum of the `IH` query vectors, so one dot of length 128
per super-block instead of `IH`), take the top `M = 64` super-blocks, then run
the exact score pass of §1.4 over only those `M × B = 1 024` candidate pools and
the exact top-512 over them.

**Bytes and ops per row per layer at ctx 18 630:** coarse pass reads the
super-block summaries (291 × 128 × 4 = 149 KB) and costs 291 × 128 = 37 k MAC;
fine pass reads 1 024 × 128 × 4 = **0.56 MB** (−77 %) and costs 1 024 × IH ×
128 = 22 % of baseline; selection 512 × 1 024 = 0.52 M iterations (−78 %).
Projected **4.5 ms/row/layer** (4.5×), **5.5 ms/call**, **50 ms/token**,
**3.61 tok/s**; stacked on (vi)'s threads, **0.75 ms/row → 0.9 ms/call →
8.3 ms/token → 4.34 tok/s**, i.e. within 2 % of the absolute ceiling.

**Numerics.** The selected set changes whenever a true top-512 pool sits in a
super-block that the coarse pass ranks below `M`. Text will change above
2 052 tokens. Below 2 052 it is a no-op **only if** the fallback "if the
selectable pools are ≤ wanted, skip both passes" is written explicitly — and
even then the *order* is preserved only because the fine pass recomputes exact
scores, which it does. That fallback is one line and must be there.

**Falsifier:** on a dumped 18.6k row (`GLM53_DUMP_INDEX=1`, free, §1.5's
existing instrument), top-512 **set recall below 95 %**, or the KL bar of §4
missed. Second falsifier: `M` large enough for 99 % recall turns out to cost
more than (vi) already costs, i.e. the design collapses into the exact scan.

**Not recommended now** because (vi) delivers 7.6× with no numerics change and
no deep-packet KL campaign, and (ii) delivers 4.5× with both. (ii) is the right
*second* item if the owner wants the last 0.4 tok/s.

### (iii) int8 / fp16 pooled keys

They are **f32 today** (`float *pool_cache`, `c/glm53.c:659`). fp16 halves the
bytes to 64 B per ctx token per layer, int8 quarters them to 32 B.

**On its own it is worth nothing**, and §1.8 is the reason: the scan achieves
116 MB/s, 0.53 % of the byte rate the CPU int4 expert path already sustains, on
a 26.2 MB working set that fits L3. Cutting the bytes 4× cuts nothing that is
on the critical path. Its only value is **more MAC per cycle after the
reduction is vectorised** — and this box has **AVX2+FMA+F16C, no AVX-512 and no
VNNI** (CLAUDE.md, §the machine): F16C is convert-only, so fp16 costs an extra
unpack per 8 lanes, and int8 must go through `VPMADDUBSW`/`VPMADDWD` rather
than a dot-product instruction. Realistic gain **1.5–2× on the score term,
stacked on top of a vectorising change that is itself a numerics change**.

**Precedent is discouraging:** §G14's int8 expert kernel measured cosine
0.98964 with the **text changed** and was rejected; §QP(a) killed int4-g64
experts outright; §X2 put G14 at mean KL 0.0296 / top-1 93.44 %. One
asymmetry is in the indexer's favour and should be stated: its output is a
*selection*, not a value, so a score error only matters when it reorders the
512-boundary — the failure mode is discrete and is measured directly by set
recall rather than by KL.

**Falsifier:** top-512 set recall below 99 % against the f32 scan on a dumped
18.6k row. **Not scheduled**; the derivation is this paragraph.

### (iv) The scan on dev0

**Where the data lives today, read from the code, not assumed:** `st->latent`,
`st->ikeys`, `st->igates` and `st->pool_cache` are all host `malloc`
(`c/glm53.c:3914-3928`). dev0 holds the MLA/dense weights and, since G12, the
KDA recurrence state — **not the KV and not the pooled keys**. There is no
indexer shader: `c/shaders/` contains `attention_absorb.comp`, `kda_decay`,
`kda_headnorm`, `kda_step`, `qmatmul{,_tile}`, `qmatmul_gate_up{,_tile}`,
`rmsnorm` and nothing matching `*index*`.

The plumbing a GPU indexer would need **exists and is unused**:
`coli_vk_kv_ensure` (`c/backend_vulkan.c:2090`), `coli_vk_kv_row`
(`c/backend_vulkan.c:2108`) and `coli_vk_attention_absorb`
(`c/backend_vulkan.c:2135`) implement a device-side KV ring with a per-row
append — and `c/glm53.c` has **zero call sites** for any of them; they are
exercised only by that file's own probe `main()` (`c/backend_vulkan.c:3537`).
That is the concrete starting point: `pool_cache` becomes a device buffer with
one 512-byte append per 4 tokens per layer, and the scan becomes one
`pools × IH` dispatch.

**Arithmetic.** Compute is trivial: 19–38 MMAC = 38–76 MFLOP, ~1–2 µs on a
7900 XTX. It is entirely round-trip-bound, which this record has measured
repeatedly: §X1 puts a driver submit at **22–25 µs** and the GPU wait at
**135–1 568 µs**; §G12 got the KDA step to **0.93 ms/call** through the same
path. Projected **0.3–1.0 ms/call** — the best ceiling of any candidate,
**4.28–4.36 tok/s** — for a new shader plus residency plumbing.
VRAM: 2.385 MB × 11 layers × 4 KV slots = **105 MB at 18.6k**, 722 MB at the
128k cap; against §Q-ARB's per-GB rule and dev0's current 1 248 experts (§P6b)
that is affordable but not free.

**"Or batching the 11 layers' scans": not possible.** Layer `L`'s indexer query
is `l->iwq` applied to *that layer's* normalised `q_a`, which comes from layer
`L−1`'s output (`c/glm53.c:1817-1841`). The 11 calls are strictly sequential in
the forward. The parallelism available is *within* a call (4 658 pools × `IH`
heads), which is exactly what (vi) and (iv) both exploit.

**Numerics.** A GPU tree reduction changes the score rounding, so the 512
boundary can flip; and §G15's standing warning applies with extra force — GLM's
output already depends on which experts are tier-resident, and a second
CPU/GPU numerics split on the *selection* would make the selected set depend on
placement too. Knob-gated, off by default, KL bar, no exception.

**Falsifier:** a measured per-call round trip ≥ 3.3 ms at 18.6k — then (vi) has
already won and (iv) is dead. Cheap to test before writing the shader: time 11
empty dispatches per forward.

### (v) Cache the selection across decode tokens

Re-run the selection every `N` decode tokens (`GLM53_INDEX_REUSE=N`, 1 =
today); in between, reuse the previous row. The tail must stay per-token
(`c/sparse_index.h:301-309`, O(pool), free) and the pools that closed since the
last selection must be force-included or the scheme systematically ignores the
most recent context — which is where attention usually wants to look.

Cost divides by `N` on both the score pass and the selection: `N = 4` →
**5.6 ms/row/layer, 6.8 ms/call, 62 ms/token, 3.52 tok/s**; stacked on (vi),
**1.0 ms/call, 9.5 ms/token, 4.32 tok/s**. Cheapest of all to implement (one
counter and a row buffer per layer per slot, Sonnet, days).

**Drift risk, stated.** The selection is stale by up to `N−1` tokens, and the
staleness is not symmetric: the rows most likely to be wrong are the newest
pools, i.e. the model's own just-generated text. It is **not dense-identical
below 2 052** either (a stale set omits pools that closed in between), so that
gate leg is lost as well. §G15's finding — this model's argmax already moves
when 6 of 42 and 8 of 1 232 predictions change under a placement-only change —
says GLM-5.3 is not a forgiving subject for a quality gamble.

**Falsifier:** greedy text diverges within the first 32 generated tokens at
`N = 2`, or the KL bar of §4 is missed at `N = 2`. **Not recommended before
(vi)**: it trades text for ms where (vi) does not.

---

## 4. The recommended design and its step list

**F6a — the DSA indexer's score pass parallelised and its top-512 made exact,
bit-identical.** (§3(vi).) The numerics-changing candidates become F6b and are
not scheduled by this design.

### Step 0 — what must still be measured (0 rig-lock minutes + one 15-minute window)

**0.1 `IH = index_n_heads`, free.** Read the banner off any existing
`GLM53_VERBOSE=1` log: `indexer  : top-%d, %d heads x %d, kpool %d`
(`c/glm53.c:349-358`), e.g. the first ~30 lines of
`~/bench/ctx_ladder_out/f409171830_F4_engine.log`. Needed only to close §1.7's
`IH = 32` vs `64` question, which sizes F6b, not F6a.

**0.2 A decode-only window at depth, free (rides the next ladder arm).** §X3
step 0's verdict asks for it and this design's §1.7 row model is its
replacement. Run one follow-up turn on a restored ~18.6k prefix with a
**1-token prompt** and `--gen 256`, `COLI_TIMERS=1`, everything else as
`tools/hot-expert/franken_chain.sh`'s `run_A`. Then rows = forwards = generated
tokens, R̄ = 1, and no row model is needed at all. **The exact lines to read,
in order:**

```
REUSE <n> <reused> <total>
[OPTIME req=<n> ctx=~18600] forwards=<F> layers=<L>s head=<H>s (n=<F>)
[OPTIME req=<n> ctx=~18600] mla split (n=<11F>, mean ctx=~18600): proj=… index=…s (<X> ms) attn=… | index/ctx=… us
```

With `total − reused = 1`, `index` ms/call × 11 **is** the indexer's ms/token
and `layers`/`F` **is** the decode token. That single window replaces every
range in §1.7 with one number and costs nothing extra.

**0.3 The scoring/selection split and the 8-thread factor — one 15-minute rig
window, no engine, no model, no gateway stop.** A microbenchmark beside
`tools/hot-expert/rome_mlaattn.c` (same shape: rig lock, 8 threads pinned,
pools > L3-resident sizes swept), calling
`coli_sparse_index_select_range_cached` directly at `pools = 4 658`, the
measured `IH`, `ID = 128`, `pool = 4`, `topk = 2048`, in four arms: serial;
threaded; threaded + heap top-k; and the `GLM53_NO_INDEX_CACHE` path for the
control. It reports the split §1.7 could only bound (91–97 %), the achieved
thread factor against §G7's 7.3×, and — because it can run the exact scan
beside the candidate on the same inputs — the bit-identity proof offline,
before any engine build. This is F6's only new rig time, and the gateway keeps
running through it.

### Step 1 — the change, and its knob

`c/sparse_index.h` only (the file is included by `c/glm53.c` and its tests
alone — `c/tests/test_sparse_index.c`, `tests/test_glm53_cancel_frames`,
`tests/test_glm53_ckpt_touch` and the segment adapter — **not** by `qwen38`,
so this is not a shared-file change and the other engine needs a rebuild, not a
re-measurement):

1. the `q`-or-`p` parallel axis on the score pass, per-thread `scores`/`taken`
   slices, 64-byte aligned, cache-line strided (§P5b.1);
2. the exact bounded-heap top-512 replacing `c/sparse_index.h:290-298`,
   preserving rank order, `>`-only comparison and the `> -FLT_MAX` filter;
3. **`GLM53_NO_INDEX_MT=1`** restores the serial, pre-item path byte-for-byte —
   the `GLM53_NO_MMAP` / `GLM53_NO_CANCEL_POLL` / `GLM53_NO_INDEX_CACHE` naming
   convention (`c/glm53.c:1789-1796`), and the A/B arm for the ladder. The new
   path is the **default**: it changes no arithmetic, which is the same footing
   G4, G7, G8 and G5 shipped on.

Checks before building, the §G7/§G8 discipline (both of those items caught a
real defect by inspection first):

- `mla_layer`'s indexer call sits outside every `#pragma omp` in that function
  (`c/glm53.c:1894-1920` vs the pragmas at 1829, 1886, 1985, 1996, 2021) and the layer
  loop is sequential, so the new region does **not** nest. Assert it
  (`omp_in_parallel()`), because a nested region collapses to one thread on this
  box's libgomp (max-active-levels 1, verified §G8) — silently buying nothing;
- `scores`/`taken` must not be shared across `q` when the parallel axis is `q` —
  this is exactly the bug §G7 caught in its own first draft (one buffer, no
  per-token dimension, silent corruption during prefill, not a crash);
- `pool_cache_count` handling is untouched, so G5's two restore-site resets
  (`c/glm53.c:5247`, `c/glm53.c:6448`) keep their meaning.

### The gate

Bit-identity is the gate; a numerics leg would be a defect, not a trade.
Everything through `tools/hot-expert/gate_lib.sh` — `gate_compare` refuses a
comparison whose pattern is absent from either side, which is how a gate once
reported a passing oracle from two empty runs.

1. **Oracle, `diff`, no tolerance**, candidate vs the binary in service:
   `teacher_forcing` **and** `last_logits`, at §G5's own pair (781 and 1 509
   tokens) **and** at one prompt above the threshold (≥ 3 000 tokens), because
   below 2 051 the selection code path barely runs. `GLM53_PREFIX_CKPT=0` and a
   private `COLI_CKPT_DIR` on every leg; the CLI oracle at `COLI_KDA_GPU=0`,
   per §X2/§P13's finding about the single-slot CLI path.
2. **The selection itself, which is free and exact:** `GLM53_DUMP_INDEX=1`
   (`c/glm53.c:1909-1918`) already prints every chosen row. Diff the dumps,
   candidate vs pristine, at a ≥ 3 000-token prompt: byte-identical rows prove
   set *and* order (§1.6), which is stronger than any logit oracle for this
   item. **"Dense-identical below 2 052 by construction" is then checked and not
   assumed** by the same diff at a ≤ 2 000-token prompt.
3. **`GLM53_NO_INDEX_MT=1` reproduces the serial path byte-for-byte** — §G5's
   knob-routing check, which is what proves the A/B arm is really the old code
   and not the new code coincidentally agreeing.
4. **The ladder, interleaved A,B,B,A** (`~/bench/f6_chain.sh` around
   `tools/hot-expert/franken_chain.sh`'s `run_A` shape), `COLI_TIMERS=1` so the
   per-request tables land in the engine log. Headline row: `index` ms/call at
   ctx ~18.6k from `[OPTIME req=] mla split`, before/after, plus `decode_tps` at
   turns 5–7. Verdict via **`gate_ab_verdict`** — never by reading the numbers.
   Uninterleaved arms read +20 % and +11 % on this box for real effects of
   +3.9 % and zero.
5. **Serving, in this order:** `tools/hot-expert/prefill_gate.sh <pristine>
   <candidate>` rc 0 → `tools/hot-expert/tworeq.py` at 4 slots →
   `tools/hot-expert/serve_candidate.sh` → `tools/hot-expert/accept_live.sh`
   exit 0 on the live gateway → `tools/hot-expert/accept_ui.sh` from the Mac
   **before** anything is said to the owner about serving.
6. Rig lock (`tools/hot-expert/rig_lock.sh`) held before the gateway is stopped
   for any reason; the chain launched through `tools/hot-expert/run_chain.sh`
   so every exit path restores the gateway; `ListAgents` checked for a peer
   session on this rig first.

**Numbers in the commit body**, per CLAUDE.md: the four `index` ms/call figures
(A,B,B,A) at 18.6k, the decode tok/s at turns 5–7, the oracle lines, and the
`GLM53_DUMP_INDEX` diff result. One row into
`tools/hot-expert/ROME-3x7900XTX-2026-09-04.md`.

### If F6b (a numerics-changing stage) is ever opened

Its KL bar, and the two things §2 says about it: **X2's own packet cannot be
used** (564 positions, 3.6× below the threshold — the KL would be 0.0 by
construction), so a ≥ 20k-position packet is needed and the KL must be taken
over a restored-prefix follow-up's decode positions. The bar itself, read off
§X2's four measured rows: G15 (killed) mean KL 0.251 / top-1 78.0 %; G14
(rejected) 0.0296 / 93.4 %; the swiglu clamp (the live, unfixed gap the owner
already runs with) 0.0284 / 92.7 %. Since the mildest row X2 has is a rejected
change, "milder than G14" is not a bar. Proposed: **mean KL ≤ 0.01, max KL
≤ 0.5, top-1 ≥ 97 %, last-position cosine ≥ 0.9995**, through `gate_kl`
(`tools/hot-expert/gate_lib.sh:78` → `tools/hot-expert/kl_compare.py`), **plus**
a top-512 set-recall ≥ 99 % from the `GLM53_DUMP_INDEX` dumps — the direct
measure of the discrete failure mode a KL averages away.

### Tier and effort

| step | tier | effort |
|---|---|---|
| 0.1, 0.2 | Haiku | minutes; 0.2 rides the next ladder arm |
| 0.3 microbenchmark | Sonnet | ½ day, one 15-minute rig window, no gateway stop |
| 1 (F6a) | **Sonnet** — "port a pattern that already exists in the repo" (§G7 verbatim, §G4/§G8's per-thread scratch) | 1–2 days + one chain window |
| F6b (ii)/(iv), if opened | Opus | 1–2 weeks each, and a deep-packet KL campaign first |

**This is a downgrade from the plan's "Opus, design 2 days, implementation 1–2
weeks."** The design cost Opus a day; the implementation is a Sonnet item
because the mechanism is already in the repo three times and carries no
numerics.

### The ceiling, plainly

- **Absolute**, indexer cost zero: 452.6 − 226.4 = 226.2 ms → **4.42 tok/s** at
  18k. The record's stated 4.3 is the same number.
- **F6a, projected: ~3.9 tok/s** (3.4 pessimistic). The record's "3.3 tok/s if
  the indexer holds its shallow cost" is the same statement with a weaker
  mechanism; F6a beats it because it cuts the slope, not just the constant.
- **F6a + (ii) or + (v): ~4.3 tok/s**, within 2 % of the absolute ceiling, at
  the price of a numerics change, a deep-packet KL campaign and changed text.
- **Decode at 18k does not reach 5 tok/s through F6 by any route.** Even a free
  indexer leaves 226 ms/token.

### What remains after it, measured (§1.7, per query row at ctx 18 630)

| bucket | ms/token | share of the 256 ms post-F6a token | owner |
|---|---:|---:|---|
| `ffn_moe` | 102.8 | 40 % | F1 (the ≤ 45 ms of CPU misses) and F2's ring buffer |
| `mla attn` core | 34.9 | 14 % | capped at 2 051 and already parallel (§G8, §P5b) — no item |
| `kda`, 34 layers | 34.8 | 14 % | G4/G12 already; nothing cheap left (§RP3) |
| **`mla index` after F6a** | **29.8** | **12 %** | F6b, if the owner wants the last 0.4 tok/s |
| `hc+norm` | 9.8 | 4 % | G10 |
| `mla proj` | 7.0 | 3 % | G8 |
| `ffn_dense` | 3.5 | 1 % | — |

The record's verdict states these per *forward* (MoE 125 ms, KDA 42 ms, attn
42 ms); those are the same figures × 1.215 (89 forwards for 108 rows, §1.7).

---

## 5. What this design does not touch, and why

**The attention core.** `attn` is 34.9 ms/row at 18.6k, 8 % of the token, and
§2 now shows it **measured flat within 20 % from ctx 1 971 to 18 584** — the
`width = 2051` cap holds, so it does not grow and it is not a depth item. It is
also already parallel over heads and slots (§G8 3.08×, §P5b's blocked pool), so
the cheap factor is spent. F6a must nonetheless **not disturb it**: the core
consumes `selected` in rank order (§1.6), which is why the recommended design
reproduces that order exactly and why the `GLM53_DUMP_INDEX` diff is a gate leg
rather than a nicety.

**The MoE path.** 102.8 ms/row, 24.6 % of the token, and the second-largest
bucket — but it is a *different* bucket with two items already pointed at it
(F1's streamed decode misses, F2's chunk ring buffer) and a measured floor: the
CPU int4 path already runs at 21.95 GB/s, 27 % of this box's DRAM bandwidth
(§G3/§G11), which is a bandwidth problem, not a parallelism one. Nothing in
F6 changes routing, residency or expert arithmetic, so §G6's tier caps and the
`COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695` comparability rule are unaffected.

**Prefill.** F2 owns it, with its own headline metric and its own harness
(`tools/hot-expert/prefill_gate.sh`, `tools/hot-expert/MEASURING.md` — using
one track's tool on the other produces a number that looks like a regression
and is not). F6a's threads do land on prefill for free, and the size of that
should be recorded rather than claimed as F6's: in §X3 step 0's `req=6` window
(18.4k, 97 % prefill rows) `index` is **1 516.461 s of 2 662.345 s = 57.0 %**,
so F6a projects that 18.4k ladder-turn TTFT from **2 561 s measured
(§FRANKEN-H2) to ≈ 1 300 s** — real, but half an order of magnitude short of
llama.cpp's 108 s, which is why F2's expert streaming remains the prefill item
and F6 is not sequenced against it.

**One cross-item flag while the same table is open.** F2's projection prices
today's non-resident experts at "113.9 ms/token (fit, §1)". §X3 step 0's
`req=6` table measures the **whole** `ffn_moe` bucket — router, shared expert,
GPU groups, CPU misses, bind/dispatch/accumulate — at
612.465 s / 9 108 rows = **67.2 ms/row** at 18.4k. The CPU-miss part alone
cannot be 1.7× the bucket that contains it, so **F2's chunk arithmetic should
be re-based on §X3 step 0's table before F2 is built.** Not this item's scope;
recorded here because this is where the two numbers met.

---

## Provenance

**Measured, with section:** every `[OPTIME req=]` figure, the decode ms/token
ladder and the `REUSE` lines — §X3 step 0 (`ROME-3x7900XTX-2026-09-04.md`,
engine log `~/bench/ctx_ladder_out/f409171830_F4_engine.log`). G5's
`index/ctx`, the 4.77 µs/pool mixing cost and the bit-identical oracle — §G5.
The 7.3× on an identically-shaped scalar f32 reduction — §G7. 3.08× on `attn`
and the nesting hazard — §G8. 1.2 GMAC/s for a scalar f32 reduction, 60.8 %
barrier spin, 69 % of the token on one core — §G3. 8.9 µs per OpenMP site —
§Q11. 21.95 GB/s CPU int4 path — §G11/§G3. 91.6 GB/s 8-thread DRAM read,
61.4–62.0 GB/s three-card stream — §PCIE-STREAM. 22–25 µs submit, 135–1 568 µs
GPU wait — §X1. 0.93 ms/call KDA step on dev0 — §G12. int8/int4 expert
numerics — §G14, §QP(a), §X2. The four KL rows and the 564-position packet —
§X2. 2 561 s ladder TTFT at 18k — §FRANKEN-H2 / §GLM53FLASH-LADDER. Cap
2 051 / topk 2048 / kpool 4 — §P5b, `c/sparse_index.h`.

**Projected, with the derivation inline:** every ms/row, ms/call, ms/token and
tok/s figure in §3 and §4; `ID = 128` is *derived* exactly from the `cache:`
line, not projected; `IH` is the one shape read from neither and is step 0.1.

**Read, not guessed:** `c/sparse_index.h` in full; `c/glm53.c` lines 104-109,
301-358, 409-424, 494-501, 594, 651-659, 1018-1022, 1487, 1639-1796,
1798-2112, 3900-3975, 4030-4042, 5238-5247, 6235-6236, 6442-6448;
`c/backend_vulkan.c` 2090-2135, 2774, 3353-3441, 3537; `c/delta_attention.h`
46-58; `c/Makefile` 72-106, 1136, 1204-1207, 1477; `c/shaders/` listing;
`docs/glm53-flash.md`.
