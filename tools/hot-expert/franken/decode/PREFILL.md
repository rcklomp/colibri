# Batched prefill for `franken_decode` (design §9.4) — design and causal rule

Written 2026-09-22 against §L0-STEP3's measured decode (22.70 ms a token over
three cards, host at §M5's launch floor). The engine ingests a prompt one
token at a time, so a 256k prompt is 97 minutes and the depth ladder is out of
reach. This is the specification the implementation follows; it is committed
before the code because the causal rule is the part that is expensive to get
wrong and cheap to review.

## 1. The causal rule, from the reference

Sources, read on the rig:

- `qwen4exp.cpp:477-623` — `build_qsa_top_k`
- `qwen4exp.cpp:594-614` — the block bias, the per-cell expansion, the mask
  add, and `width = min(n_kv, indexer_top_k + r - 1)`
- `llama-memory-hybrid-idx.cpp:430-458` — `set_input_qsa`, which computes the
  bias for **every row of the ubatch**
- `qwen4exp.cpp:1026` — "apply_ubatch() already stored this ubatch, so its own
  tokens count too"

For a query at position `p`, with compress ratio `r`:

```
tail_start = ((p + 1) / r) * r                       // integer division
bias(b)    = +1e9        if b*r >= tail_start        // the incomplete tail
           = -INFINITY   else if filled[b] < r       // a block that cannot be pooled
           = 0           otherwise
score(cell j) = blk_score[j/r] + bias(j/r) + mask(j)
mask(j)    = 0 if cell j is non-empty, same sequence, and pos(j) <= p
           = -INFINITY otherwise
select      = top_k(score, min(n_kv, indexer_top_k + r - 1))
```

`filled[b]` counts the cells of block `b` **present in the cache**, and during
a batched prefill the cache already holds *every* row of the chunk — including
rows after `p`.

## 2. Why that is still causal, which is not obvious

The worry is that a block pooled from tokens later than `p` could leak into
`p`'s selection. It cannot, and the reason is arithmetic:

> `tail_start` is a multiple of `r`, so a block with `b*r < tail_start` has
> `b*r + r <= tail_start <= p + 1`, i.e. **every position in it is `<= p`**.

So a block that is *scored* (bias 0 or -inf) can only contain tokens at or
before `p`, and a block that could contain a later token always satisfies
`b*r >= tail_start` and is therefore in the +1e9 set, where its score is
irrelevant — and its future cells are then removed by the per-cell mask
anyway. The pooled key of a scored block is causally clean by construction.

**Therefore a chunk of T rows must produce exactly what T single-token steps
produce.** That is the oracle, not an approximation of one:

```
franken_decode --tokens <6 ids> --chunk 1 --dump  A
franken_decode --tokens <6 ids> --chunk 6 --oracle A      # must be cos 1.0
```
plus the greedy ids against `~/bench/franken/oracle/greedy.txt`.

## 3. Order inside a layer for a chunk

The order matters because of §2: the caches must be complete before anything
is scored.

1. project all T rows (one batched GEMM per projection group)
2. write the indexer keys for all T cells, then re-pool every block the chunk
   touched
3. write K and V (q8_0) for all T cells
4. **then** per query row `p`: indexer scan, bias, expand, top-k, attention

The reference does the same: `cpy_k` for the whole ubatch precedes the
pooling, and `build_attn_qsa`'s `cpy_k`/`cpy_v` precede `build_attn_mha`.

## 4. What is T-wide, what loops, and why

| part | shape | treatment |
|---|---|---|
| trunk projections | `[T,K] x [K,rows]` | **token-tiled GEMM**: a wave loads a weight block once and dots it against TILE activation columns. This is the point of prefill — the trunk is ~5 GB a token and a tile of 8 cuts it 8x. |
| routed experts | 10 of 512 a row | sort the `T x 10` assignments by expert on the device, then amortise the weight read over the rows sharing an expert |
| shared expert | `[T,2560] x [2560,640]` | the same tiled GEMM |
| elementwise, hc, norms | row-wise | already row-wise: pass `n * T`, or a grid of `n_groups * T` |
| GDN conv | window over the ring **and** the chunk | one kernel, the window straddling the two |
| GDN recurrence | sequential in `t` | one kernel looping `t` internally, state in registers/LDS per head. Correctness over speed, as the brief says: it is ~0.6 ms a token of a 22.7 ms token. |
| QSA scan/top-k/attention | per query | a loop of T over the existing kernels. At T=256 that is ~1 000 launches a QSA layer, ~3 ms, against the chunk's weight time — small, and it reuses kernels already proven. |
| lm_head | last row only | the prompt needs one logit vector |

## 5. Memory at T = 512

Activations are token-major (`buf[t*dim + i]`), so a row is contiguous and a
tiled GEMM's loads are coalesced. The T-wide scratch is dominated by the MoE
intermediates: `T x 10 x 640` for gate/up/h and `T x 10 x 2560` for the
per-expert outputs = 512 x 10 x 3840 x 4 B = **78 MB**, which is why the
default chunk is 256 and the cap is 512. Everything else is under 20 MB. The
QSA per-query buffers do not grow with T at all, because that stage loops.

## 6. Not in the first implementation

- the expert row-gather batching (item 2 above): correct but unamortised
  first, so the oracle gate is met on a smaller change, then the sort lands on
  a proven base
- prefix checkpoints, and any reuse of a cache across requests

## 7. What landed, and what it measured (2026-09-22)

All of the above is implemented, in three commits, and **the row-gather is no
longer pending**: section 6's first bullet is history.

| commit | what |
|---|---|
| `5baa3f3` | items 1-3: token-major T-wide scratch and graph, one GEMM path (`k_gemv_batch` deleted), the GDN conv window + `conv_slide` + the in-order recurrence, QSA after the chunk's cache writes |
| `94d9522` | item 4: `--chunk C` (default 256, cap 512), `--time-prefill N`, the oracle |
| `53c1644` | item 5: `k_moe_sort` + the `_gather` expert kernels, on the `T > 1` path only |

**The oracle, run on the CPU arm** (`franken_decode_cpu`, no GPU touched),
6 tokens, layers 0-47, ctx 512, against a `--chunk 1 --dump` of the same
binary:

```
compared=1604  missing=0  incomparable=60  refused=0   STEP2 PASS
1 592 float taps  cos=1.000000  maxabs=0     <- bit-for-bit, not a tolerance
12 indexer_top_k  contained=1.000000
```

with the same verdict at `--chunk 2`, `4` (= 4+2) and `5` (= 5+1), which is
where the cross-chunk carries are exercised rather than the single-chunk
case, and `--chunk 6 --greedy 16` giving the same 16 ids and the same
`residual l1=4136.68` as `--chunk 1`. The 60 incomparable are
`indexer_k` / `indexer_k_pooled`, which the comparator refuses at any cache
depth by design.

**Two things that keep it bit-for-bit, and that a later change must not
undo.** The split-K count of a GEMM is chosen from the MATRIX alone and never
from T -- it sets the summation order, so a chunk whose split differed from a
decode token's would not be comparable to it. And there is only ONE
GEMV/GEMM kernel: `k_gemm_batch` runs the decode token as a chunk of one,
carrying the old one-column kernel's accumulator structure, because two
kernels would mean the decode path and the prefill path could disagree in
the last bit and this gate would be comparing two different things.

## 8. The GPU arm, and why item 5 is off by default (2026-09-22, three cards)

The device oracle at first FAILED where the CPU one passed: `--chunk 6`
against `--chunk 1` diverged from layer 1 onwards, `Kcur-3` at maxabs 1.2e-6
growing to `Kcur-27` at 1.0e-3, and greedy id 7 flipping. The bisection that
localised it, in the order it was run:

| arm | result |
|---|---|
| `--chunk 1` against its own dump | identical — the engine is deterministic |
| `--chunk 6` against `--chunk 1` | differs from layer 1, then every layer |
| `--gemv-lds 0` on **both** arms | unchanged — the LDS staging flip is NOT it |
| `--chunk 6` with the expert gather OFF | **zero differing taps: bit-identical** |

So the whole divergence is design 9.4 item 5, the device-side expert sort and
the row-gather kernels — the only code in the engine that runs at `T > 1` and
not at `T = 1`. **It is therefore a knob, `--expert-gather MASK`, and the
default is 0.** Bit 0 is gate/up and bit 1 is down, so one run each says which
stage is at fault; they write different tensors and the first differing tap
distinguishes them.

A second pass with the mask per stage narrowed it further:

| arm | result |
|---|---|
| `--expert-gather 1` (gate/up only) | **bit-identical** — so gate/up is exact and is ON by default |
| `--expert-gather 2` (down only) | diverges from layer 1 on, `Kcur-11` 1.7e-6 … `Kcur-27` 1.0e-3 |

So **the default mask is 1**: the gate/up gather is measured exact and runs,
the down gather is measured wrong and does not. That is most of the
amortisation — gate and up are two `K = 2 560` tensors an assignment against
down's one `K = 640`, about 60 % of the expert bytes.

**Why the down gather is wrong is still not known, and what that now rules
out is the interesting part.** `moe_gather_test.cpp` transcribes both down
paths on the host — emulating a 32-lane wave, over the SAME shared primitives
from `decode_quant.h` / `m1_native_decode.h` that the kernels call, so the
decode is not re-derived and a disagreement could only be the loop, the
indexing, the accumulators or the reduction. They agree **bit for bit**, as
do the counting sort's invariants (every column covered exactly once, by a
tile whose `tile_exp` is its expert). The sort is independently proved on the
device by the mask-1 arm, which shares it. And `moe_finish` sums a token's ten
contributions in rank order either way, because the column index is
`t*K_TOP + k` in both paths.

So the fault is not the algorithm, not the indexing, not the accumulator
structure and not the reduction: it is something the device's code generation
does to one of these two kernels and not to its per-assignment twin. The
Q8_0 gather's one textual difference — a hoisted `d * qs[tid]`, algebraically
identical and bit-identical on the host — has been removed, so its inner
statement is now literally the per-assignment kernel's. That leaves IQ4_NL's
array-of-accumulators against the per-assignment kernel's two scalars as the
last textual difference standing, and the next device run is the one that
says whether removing the hoist was enough.

**What the down stage being off costs:** it reads a chosen expert's row once
per assignment rather than once per group — at `T = 256`, 2 560 reads a layer
instead of ~512, for ~40 % of the expert bytes. The 7.48 ms/token first
prefill number was measured with the whole gather ON, so it needs re-taking
either way.

**Not measured here:** anything on a GPU by the agent that wrote this.

## 9. What a later change must not undo

Two invariants, and a third the hard way:

- the split-K count of a GEMM is chosen from the MATRIX alone, never from T;
- there is ONE GEMV/GEMM kernel, `k_gemm_batch<TILE>`, and `T == 1`
  instantiates `TILE = 1` — same source, so decode and prefill cannot drift;
- **every method of `GpuBackend` that allocates, copies, creates an event or
  launches calls `dev_ensure(dev_)` first.** `set_profile`'s has now been lost
  three times and the symptom is always "invalid resource handle" from
  `mark()`, because `hipEventCreate` acts on the process-wide current device.
