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

That put the default mask at 1 for a day: the gate/up gather was measured
exact and ran, the down gather was measured wrong and did not. That is most of
the amortisation — gate and up are two `K = 2 560` tensors an assignment
against down's one `K = 640`, about 60 % of the expert bytes.

### 8.1 What the down gather was doing wrong (2026-09-23, found in the ISA)

`moe_gather_test.cpp` had already cleared everything a reader would suspect.
It transcribes both down paths on the host — emulating a 32-lane wave, over
the SAME shared primitives from `decode_quant.h` / `m1_native_decode.h` that
the kernels call, so the decode is not re-derived and a disagreement could
only be the loop, the indexing, the accumulators or the reduction. They agree
**bit for bit**, as do the counting sort's invariants (every column covered
exactly once, by a tile whose `tile_exp` is its expert). The sort is
independently proved on the device by the mask-1 arm, which shares it. And
`moe_finish` sums a token's ten contributions in rank order either way,
because the column index is `t*K_TOP + k` in both paths.

So the fault had to be code generation, and it is. Compile the file for the
device alone and read the two kernels — no GPU, no `--device` flags, four
seconds:

```
hipcc --offload-arch=gfx1100 -O3 -std=c++17 --offload-device-only -S \
      decode_gpu.hip -o /tmp/decode_gpu.s
```

| kernel | `v_fma_mix_f32` | `v_fma_f32` | `v_fmac_f32` | `v_mul_f32` | `v_add_f32` |
|---|---|---|---|---|---|
| `k_moe_down_iq4nl` (per assignment) | 8 | 0 | 1 | 7 | 13 |
| `k_moe_down_iq4nl_gather` (tiled, WRONG) | 8 | 16 | 48 | 0 | 48 |
| `k_moe_down_iq4nl_gather` (m-outer, now) | 8 | 0 | 1 | 7 | 13 |

The accumulate that matters lives inside `m1n_iq4nl_chunk_dot`:

```c
lo += (d * (float)kvals[q & 0xf]) * hb[c];
```

Under `-ffp-contract=fast` — hipcc's default — the backend MAY fuse that into
an FMA and MAY leave it as a multiply and an add. Both are legal, they round
differently, and it chooses per kernel on scheduling grounds: with the `w`
loop outside and eight independent accumulator pairs inside, the tiled gather
had ILP to spare and every term stayed fused; the per-assignment kernel has
ONE accumulator pair, so the fused chain IS its critical path and the machine
combiner splits all but one term into `v_mul_f32` + `v_add_f32` to shorten it.
One rounding apart, which is 1.7e-6 at `Kcur-11` growing to 1.0e-3 by
`Kcur-27` and greedy id 7 flipping — exactly the reported divergence.

**The fix is to give the gather one accumulator pair too**: its `m` loop
outer, its `w` loop inner, `lo`/`hi` as scalars, and `#pragma clang loop
unroll(disable)` on the `m` loop so the backend does not rebuild the
scheduling problem it has just lost. The body is then literally
`k_moe_down_iq4nl`'s, and the table above is the device agreeing. Note that a
`#pragma clang fp contract(off)` around the kernel would NOT have worked: FP
options are lexical at the point the expression is WRITTEN, which is inside
the shared primitive in `m1_native_decode.h`, and setting it there would
change the CPU backend and the decode path with it.

**The amortisation survives**, which is the point of the whole item. The
expert row is `row_bytes` = 360 B at `K = 640`; the second and later passes
over it are L1 hits, so the VRAM read is still once a TILE rather than once an
assignment — only the order of the reads changes, not their count. VGPRs
43 → 31, occupancy 16 either way, no spills.

`k_moe_down_q8_0_gather` keeps its tile and needs none of this: against its
twin it is the per-assignment body times eight, op for op (8 `v_fma_f32`
against 1 `v_fmac_f32`, 40 `v_add_f32` against the wave reduction's 5, and one
`v_fma_mix_f32` either way, because `d * qs[lane]` does not depend on `m` and
is CSEd). Every term stays fused there, so there is no rounding to disagree
about. This model's down projection is IQ4_NL; Q8_0 is for a container that is
not.

**So the default mask is 3.** `make moe-gather-test` passes on the new shape
(`compared=60 iq4nl_mismatch=0 q8_0_mismatch=0`) and the ISA table is the
other half of the claim. The device gate that closes it is the usual one, and
it has NOT been run by the agent that wrote this section:

```
franken_decode --tokens <6 ids> --chunk 1 --dump A
franken_decode --tokens <6 ids> --chunk 6 --oracle A     # must be cos 1.0
```

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

## 10. The pipeline and the LDS GEMM (2026-09-22, the two profile items)

`g.txt` said two things about the 7.6 ms prompt token and this section is
what was done about each.

**(a) The cards were idle two thirds of the time.** Chunk n ran on card 0,
then card 1, then card 2: card 1 waited 2.5 ms and card 2 4.9 ms per token
for the upstream chunk, and 2.56 + 2.43 + 2.39 = 7.4 of the 7.6 ms. Design
§3.5 allows micro-batching for prefill, so chunk n+1 now starts on card 0 as
soon as card 0 has handed chunk n to card 1 (`--prefill-pipeline 1`, the
default). Three parts:

- the wide residual — the ONLY buffer that crosses a card — has **two banks
  a device**, picked by chunk parity, and `boundary_recv` gained the other
  half of its handshake: the destination records a `drained_[bank]` event
  after its peer copy and the source waits on it before overwriting that
  bank. Stream events, no host call. Two banks also bound the run-ahead at
  two chunks, so there is no depth counter. Every other buffer a chunk
  touches is written only by its own card in chunk order, and a stream is
  in-order;
- the pinned staging ring **grows to the largest upload it is handed**. It
  was 64 KB, a chunk uploads 2.6 MB at T = 256, and the overflow path was a
  blocking `hipMemcpy` on the default stream — which drains a blocking
  stream, so every chunk started by emptying card 0's queue;
- `step(..., flush=false)` enqueues and returns; only the last chunk is
  awaited. It overrides `flush` whenever the recorder is capturing (a tap is
  a download), so no caller can lose a tap by asking for the pipeline, and
  `last_flushed()` is what a caller reads back rather than re-deriving it.

`--profile` had to stop synchronising per chunk or it would have serialised
what it measures: it defers, and the gap between a chunk's closing event and
the next chunk's first op is charged to `PC_GAP`, which is the card's idle
time. `prof_busy_us` / `prof_busy_frac` are the overlap.

**Numerics unchanged by construction** — no kernel moved and no summation
order changed, only an address and the host's waiting. Gated on the CPU arm
at layers 0-11: `--chunk 1` against the pre-change binary, and `--chunk 2`,
`4`, `5`, `6` against `--chunk 1`, all 404 float taps `cos=1.000000
maxabs=0`, and the same 16 greedy ids at chunk 6 and chunk 1.

**(b) The trunk GEMM is 6x off its bytes.** `k_gemm_batch<8>` reads its
~1.7 GB of weights once per TILE of 8 token columns — 32 passes at T = 256,
54 GB a chunk, 0.27 ms a token at the 800 GB/s bound — and takes 1.69-1.76.
At 135 VGPR it gets 10 waves/SIMD and every wave re-decodes the same Q8_0
block for each group of 8 columns. So there is now a second kernel, and
section 9's "there is ONE GEMV/GEMM kernel" holds in the form that matters:
**the default is still that one kernel**, and `--gemm-lds` is off unless a
gate turns it on.

`k_gemm_lds`: a workgroup owns a 64 x 64 (rows x tokens) tile and stages K in
steps of 32 through LDS, so a weight element is read and decoded ONCE per 64
columns — 4 passes at T = 256, 6.8 GB a chunk, 0.033 ms a token at the
bound. The activation tile is re-read once per 64 rows, which is 160 passes
over 2.6 MB and costs nothing because 2.6 MB lives in the 96 MB Infinity
Cache. 60 VGPR, 17 408 B of LDS, no spills, occupancy 14 waves/SIMD before
the LDS limit.

`k_gemm_lds_i8` (`--gemm-lds 2`): the same tiling with the activation tile
quantised to int8 per 32-block (ggml's Q8_1 recipe) and RDNA3's
`v_dot4_i32_iu8`. **The instruction IS reachable on this toolchain**, which
this directory's Makefile used to say it was not:
`__builtin_amdgcn_sdot4` wants `dot1-insts` and is rejected, but
`__builtin_amdgcn_sudot4(true, a, true, b, acc, false)` compiles and emits
`v_dot4_i32_iu8 ... neg_lo:[1,1,0]`. The WEIGHTS are exact — Q8_0 is
already int8 and the kernel's K step is exactly its 32-element block — so
only the activations are requantised. 95 VGPR, 5 120 B of LDS, 128 dot4s,
no spills.

**Both modes accumulate K linearly per thread** where `k_gemm_batch`
accumulates across a wave's lanes and closes with a butterfly. That is a
different summation order, so neither can be bit-identical to decode; mode 2
additionally changes the activations. Hence the knob, the default of 0, and
a gate that measures the divergence rather than asserting there is none.
Small matrices (< 256 rows) and anything that is not Q8_0/BF16/F32 keep the
old path whatever the knob says — all-or-nothing per batch, because
`nsplit` is a function of the concatenation's row count and splitting a
batch would change the untouched half's summation order for no reason.

## 11. The query rows of a chunk are a grid axis (2026-09-23, item 3)

Section 4 budgeted the QSA selection stage as "a loop of T over the existing
kernels ... ~1 000 launches a QSA layer, ~3 ms". The three-card profile of
2026-09-22 charged it **0.19 ms a prompt token a card** of the 1.63 total, and
all of it is issue cost: at T = 256 and 12 QSA layers a chunk enqueues ~12 000
launches for that stage, which at this card's ~5 us launch floor IS the 48 ms
a chunk it costs. There is no kernel to make faster.

Row t differs from row t+1 only in its POSITION, and everything the host loop
computed from it -- `n_kv`, `n_blocks`, `tail_start`, the budget, whether the
top-k is the identity -- is a closed form of that position. So the five
kernels (`k_idx_scan`, `k_qsa_expand`, `k_topk_radix_select`,
`k_attn_flash_split`, `k_attn_combine`) take a **row axis** and the host
enqueues one launch a stage for a whole block of rows: 5 a QSA layer instead
of ~1 000, 60 a chunk instead of ~12 000.

**Nothing about the numerics moves, and the reason is structural rather than
argued.** A row reads and writes its own slice of every buffer, no reduction
crosses rows, and the per-row arithmetic is the code that was already there --
only `blockIdx` and a base pointer are new. The two shapes the kernels have to
handle are the widest row's (which sizes the grid) and everything narrower,
and a workgroup past its own row's extent returns before any shared-memory
barrier.

**A block is bounded by SCRATCH, not by T**, which is why section 4 said the
stage loops in the first place: `cell_scores` is O(n_kv) a row, so 256 rows of
a 256k cache would be 268 MB of them plus 537 MB of radix-select candidates
and 208 MB of attention partials. `Backend::reserve_qsa_rows(rows_max, ctx)`
answers how many rows the backend has room for under `--qsa-row-mb` (default
256 MB a device) and the runner walks the chunk in blocks of the answer. At
ctx 8 192 a row is ~0.9 MB and the whole chunk is one block; at 262 144 it is
~4.2 MB and the block is ~60 rows, which still cuts the launches sixtyfold.
A backend that returns 1 gets the old loop back, exactly.

`Backend::qsa_rows()` has a DEFAULT implementation which is that old loop over
the four single-row calls, so the CPU backend is unchanged by construction and
the chunk-vs-decode oracle keeps its meaning; the GPU override is the only
thing under test. The GPU's own single-row entry points survive as blocks of
one over the same kernels -- section 9's rule, applied to this stage: one
kernel, so the decode path and the prefill path cannot disagree in the last
bit.

**The gate, run on the CPU arm** (`franken_decode_cpu`, no GPU touched),
6 tokens, layers 0-3 (which is QSA layer 3), ctx 512, `--chunk 1 --dump`
against `--chunk 2`, `4` (= 4+2), `5` (= 5+1) and `6` -- the row-block loop's
partial last block is exercised at 4 and 5, and at this ctx the CPU backend
batches the whole chunk, so the strides and the tap indices are under test:

```
compared=141  missing=0  incomparable=5  refused=0     STEP2 PASS   (x4)
140 float taps  cos=1.000000  maxabs=0    <- bit-for-bit, not a tolerance
indexer_top_k-3 contained=1.000000        residual l1=110.396 at every chunk
```

Separately, the five per-row closed forms the device helpers compute
(`qsa_n_kv`, `qsa_n_blocks`, `qsa_n_sel`, `tail_start`, the identity test)
were checked against the old host loop's values over 7 477 470 combinations of
(ratio 2-8, pos0 0-3000, rows 1-256, t): **0 mismatches**.

Resources, gfx1100, no spills anywhere: `k_idx_scan` 34 VGPR occ 16,
`k_qsa_expand` 6 VGPR occ 16, `k_topk_radix_select` 21 VGPR 12 316 B LDS
occ 16, `k_attn_flash_split` 49 VGPR 3 328 B LDS occ 15, `k_attn_combine`
10 VGPR occ 16.

`--profile` gained two classes for exactly the two things these items change:
`prof_qsa_rows_us` is the batched row stage (scan, expand, select, attention)
apart from `prof_qsa_us`, which keeps the per-chunk work that is not per query
row (the cache writes, the pooling, `qk_post`, rope); and
`prof_expert_gather_us` is the counting sort plus the row-gather kernels,
apart from `prof_expert_gate_up_us` / `prof_expert_down_us`, which now hold
only the per-assignment path. Together they say how much of the expert time
the amortisation is carrying and how much of the QSA time was the loop.

**Not measured here:** anything on a GPU by the agent that wrote this section.
