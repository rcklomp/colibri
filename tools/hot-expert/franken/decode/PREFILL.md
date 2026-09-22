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
