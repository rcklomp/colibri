# `franken_decode` — L0 step 2: one card, layers 0–15, a decode loop with a per-layer oracle

Deliverable B of `tools/hot-expert/L0-STEP2-BRIEF-2026-09-22.md`
(design rev 8 §9.5 step 2). Built 2026-09-22 on branch `m5-harness`.

The math is ported op for op from `~/src/llama-glm53/src/models/qwen4exp.cpp`
@ `39931761a`, plus `src/models/delta-net-base.cpp` and
`ggml/src/ggml-cpu/ops.cpp` for the two ops whose graph form and fused form
differ (`ggml_gated_delta_net`, `ggml_rope_multi`).

## The two binaries

| binary | what it is |
|---|---|
| `franken_decode_cpu` | the same graph on the host, in plain C++. Links **no HIP at all** (`make ldd-check` proves it), so it cannot touch a card even by accident. |
| `franken_decode` | the same graph on device 0 through the HIP kernels. Uploads ~22 GB; **run it only under the rig lock with the gateway stopped.** |

```
franken_decode[_cpu] --model <any shard.gguf> --tokens <id> [<id> ...]
                     [--layers 0-15] [--oracle DIR] [--ctx N] [--threads N]
                     [--time N] [--min-cos X] [--quant-act] [--verbose]
                     [--dump DIR] [--jitter X] [--profile]
```

`make` builds both and runs nothing. `make resources` prints the per-kernel
VGPR/SGPR/LDS/spill report (compile-only, no GPU).

## Why `--cpu` is not optional

The agent that wrote this could not run a GPU. `decode_graph.cpp` holds the
math and never touches a float: it calls `Backend` (`decode_backend.h`), which
`decode_cpu.cpp` and `decode_gpu.hip` implement. Both are checked against the
**same** oracle, so a disagreement localises itself — `--cpu` failing is a
port bug in the graph, `--cpu` passing and the GPU run failing is a kernel bug.

## What the file actually contains (and three ways it bites)

Read off the GGUF header, not assumed:

- **A UD quant is not uniform.** `ffn_{gate,up}_exps` are IQ3_S on every layer
  of 0–15 **except layer 2**, which is IQ4_XS; `ffn_down_exps` is IQ4_NL
  except on **layers 2 and 4**, which are Q8_0. Four expert kernels, not two.
- **The indexer has ONE key head.** `blk.N.indexer.k_proj` is `[2560, 128]` and
  `q_proj` is `[2560, 512]` — four *query* heads against one key. The
  reference rectifies each query head's dot **before** summing
  (`qwen4exp.cpp:576-591`). `m3_attn.hip`'s scan kernel assumed four key heads
  and no relu, so it is a cost model for this pass, not a port of it — and the
  real pass reads a **quarter** of the pooled-key bytes §M3 charged the depth
  term.
- **`m3_attn.hip`'s GDN step is not the model's.** It predicts from the
  *undecayed* state; `ops.cpp:11012` decays first. It is also transposed:
  ggml stores the state with the KEY axis contiguous (`s_out[j*S_v + i]`).

## Reading the oracle dump

Deliverable A writes `<dir>/<key>.f32` with `<dir>/index.txt` lines
`<key> <il> <ne0> <ne1> <ne2> <type>`. The payload is the last slice along
the slowest **non-unit** axis, which is *usually* the token axis and three
times is not:

- `hc_norm-3` is the **attention** module's value; `hc_norm-3.2` is the FFN
  module's (`build_hc_mix` runs twice a layer).
- `hc_combine-3` is the **post-attention** residual; `l_last-3` is the
  post-FFN one — `ggml_set_name` overwrites, so the second `hc_combine` never
  reaches the dump under that name. (Same reason `kqv_out` and `v_conv` are
  absent: the dump itself reported them "NOT FOUND in this build's graph".)
- `state_predelta-0` has ne2 = 48 **heads**, so the file is head 47's
  128×128 state, not a token.

Two points cannot be compared at all at a different cache depth, and the
comparator says so instead of failing or skipping silently:

- `indexer_k_pooled` / `indexer_k` — the file is the last *block's* column,
  and the last block is llama.cpp's **padded** `n_kv/ratio` (block 63 for a
  6-token prompt in a 256-cell cache). The selection these keys drive **is**
  checked: `indexer_score` compares every block this engine has.
- `state_predelta` — llama.cpp evaluated the prompt as **one ubatch**, so the
  state it read is the pre-batch (zero) state, while a decode loop reads the
  state after five tokens. The recurrence is checked by `attn_output`, which
  is its output for the same token.

## `--quant-act`: the arm that settles the numerics

llama.cpp's CPU `mul_mat` does **not** dot f32 activations against quantised
weights: it converts the activation to the weight type's `vec_dot_type` first
(Q8_0 for a Q8_0 weight, BF16 for a BF16 one). `--quant-act` makes that
rounding ours. It is a diagnostic arm, never the engine — this engine is the
more accurate side and stays that way.

It is the reason the residual gap is *explained* rather than *tolerated*:
under it, layer 0's whole body matches llama.cpp **exactly**.

## The routing coin-flip

`build_moe_ffn` selects 10 of 512 experts by a softmax top-k, which is
**discontinuous** in its input: at layer 1 the tenth expert's probability is
0.0606 against the eleventh's 0.0600. Measured directly: the plain and
`--quant-act` arms, which differ only by activation rounding, pick the *same*
ten experts at layer 0 and a *different* tenth at layer 1.

It is **not** ulp-chaotic, which is worth stating because the obvious guess
is wrong and the jitter arm refuted it: a 6e-7 perturbation of every block
input flips nothing at all, at any of the 16 layers. It takes a perturbation
of about 4e-4 — the scale of the reference's own activation quantisation, not
of fp32 rounding — before the first flip appears (layer 7).

So a cosine bar on `ffn_moe_out` tests the routing, not the arithmetic — the
same lesson as §F7's jitter arm (CLAUDE.md). `--verbose` prints the chosen
ids and weights so a flip is visible instead of showing up as an unexplained
divergence.

## What is NOT here

No `lm_head` (Q6_K, 521 MB), no final hyper-connection mixer, no sampling, no
gateway, no prefix checkpoints, no three-card boundaries. Those are step 3
and step 4.

## What `--cpu` measured, 2026-09-22

Ids `[248044, 785, 10945, 315, 1495, 374]` (the dump's own), six tokens fed
one at a time, `--ctx 32`, against `~/bench/franken/oracle`. Logs in
`~/bench/franken/cpu_l0*.log`, `jitter*.log`.

**Layer 0 matches to the printed precision.** With `--quant-act` every tap of
layer 0 up to and including the Gated DeltaNet output reads `cos=1.000000`:
`hc_norm-0`, `hc_gate-0`, `hc_mixed-0`, `hc_inject-0`,
`linear_attn_qkv_mixed-0`, `z-0`, `alpha-0`, `beta-0`, `a_softplus-0`,
`beta_sigmoid-0`, `conv_output_silu-0`, `q_conv-0`, `k_conv-0`,
`attn_output-0`, `shared_expert_gate_sigmoid-0`. Only `hc_norm-0` is
bit-identical (`maxabs=0`) — it is the one tap with no matmul in it; the
others carry a `maxabs` of 1e-5 to 1e-2 against vectors whose elements are
O(1)–O(10), i.e. the last-ulp disagreement of two f32 reductions.
The rest of the layer is at the floor: `linear_attn_out-0` 0.999996,
`ffn_shexp-0` 0.999995, `ffn_moe_out-0` 0.999965, `hc_combine-0` 0.999995,
`l_last-0` 0.999993.
Without the arm (the engine's real numerics, f32 activations) the same points
are 0.99988–0.99999 — that difference is llama.cpp's activation quantisation
and **this engine is the more accurate side**.

**Layer 1 (the PLE layer)** `ple_embd` 1.000000, `ple_gated_value-1`
0.999998, `hc_*-1` ≥ 0.99997, `attn_output-1` 0.999995, `l_last-1` 0.999657.

**Layer 3 (QSA)** `indexer_q-3` 0.999905, `indexer_score-3` 0.999999,
`indexer_top_k-3` containment 1.000000 (6 of 6 cells, against a 256-cell
reference selection), `attn_pregate-3` 0.999744, `attn_gated-3` 0.999570,
`attn_output-3` 0.998914, `l_last-3` 0.999810. Its input is already
`hc_mixed-3.2` = 0.999438, so these are upper bounds on layer 3's own error.

**Depth: the MoE router, not the arithmetic.** `l_last` falls smoothly
0.999959 → 0.989552 over layers 0→15 while `ffn_moe_out` collapses from
0.9999 (layers 0–6) to 0.90–0.97 (layers 7–15). `--verbose` shows why: the
plain and `--quant-act` arms, differing only by activation rounding, pick the
*same* ten experts at layer 0 and a *different* tenth at layer 1 — the
contested pair's router probabilities are 0.0606 and 0.0600.

**The jitter arm sizes it** (CLAUDE.md §F7 discipline; `--dump` makes one run
the oracle of another, so this needs no llama.cpp):

| perturbation of every block input | first routing flip | `l_last-15` |
|---|---|---|
| 6e-7 (fp32 rounding) | none — `ffn_moe_out` 1.000000 at every layer | **1.000000** |
| 4e-4 | layer 7 | 0.999287 |
| 4e-3 (q8_0 activation scale) | layer 7 | 0.999271 |
| llama.cpp, plain arm | ~layer 7 | **0.989552** |

So the architecture is **not** ulp-chaotic, and one routing flip costs
`l_last` about 7e-4. The gap to llama.cpp at layer 15 is an order of
magnitude larger than a single-point perturbation explains, because its
activation quantisation happens at *every* matmul rather than once per block.

**STEP2 is FAIL at the brief's bar, and the bar is not reachable as written**:
`cos ≥ 0.999` on `ffn_moe_out-15` asks a top-10-of-512 selection to agree with
a reference whose own selection is decided by its activation quantisation.
The clean fix is one line in deliverable A: **dump the selected expert ids
per layer** (`ffn_moe_topk_ids`). With those, routing can be compared as a
set and, if needed, forced identical — and the arithmetic bar becomes
meaningful at every depth instead of only at layers 0–6.

## The 12.9 ms token, and what was done about it (2026-09-22, second cut)

The first GPU run measured `layers0_15_ms_median=12.9059` (n=32) — about 5x
the bytes (~1.8 GB of weights for these 16 layers is ~2.2 ms at the 800 GB/s
of §M5). Three things were wrong, and `--profile` now measures which of them
mattered.

**1. Host-blocking calls inside the layer body.** Three are gone:

- the QSA per-block bias was built on the host and uploaded **once per QSA
  layer** — a blocking `hipMemcpy` in the middle of the loop. It is a closed
  form of the position, so `k_qsa_expand` computes it.
- the PLE row upload happened inside `layer_ple`. It is now done by `step()`
  before the body, with everything else the token needs.
- `--verbose` downloaded the routed ids **per layer**, which is 16 device
  syncs a token — and `--verbose` was on in the run that measured 12.9 ms.
  The ids are now logged device-to-device and read back after the body.

Per-token uploads also go through a pinned staging ring and `hipMemcpyAsync`,
so they are stream-ordered rather than blocking. `prof_host_syncs_per_token`
must read **0**; anything else is a bug.

**2. Launch geometry: most of the trunk could not fill the card.** One wave
per output row is right for the two wide projections and starves a 96-CU /
192-SIMD card on everything else:

| tensor | rows | K | bytes | old WGs | waves/SIMD | new WGs |
|---|---:|---:|---:|---:|---:|---:|
| `attn_qkv` | 10240 | 2560 | 27.9M | 10240 | 53 | 1280 × 8 waves |
| `ssm_out` | 2560 | 6144 | 16.7M | 2560 | 13 | 2240 |
| `hc_*_down` (×4/layer) | 320 | 10240 | 3.5M | **320** | **1.7** | 2080 |
| `ffn_gate_inp` | 512 | 2560 | 5.2M | **512** | **2.7** | 2048 |
| `ssm_alpha`/`beta` (×2) | 48 | 2560 | 0.5M | **48** | **0.25** | 480 |
| `hc_*_inject` (×4/layer) | 4 | 10240 | 0.16M | **4** | **0.02** | 320 |
| `ffn_gate_inp_shexp` | 1 | 2560 | 0.01M | **1** | **0.005** | 80 |

A workgroup is now 256 threads = 8 waves, one row each, and K is split across
`nsplit` workgroups chosen so `row_groups × nsplit` reaches 2048 — but only
when `row_groups` is under 512, so the two already-full GEMVs are untouched.
Split partials are summed by `k_reduce_splits` in a **fixed order**, so the
result stays reproducible (an `atomicAdd` would not, and the oracle depends
on it). The Q8_0 inner loop keeps decode_quant.h's lane mapping — a wave
reads one block's 32 contiguous `qs` bytes, the widest request ggml's
34-byte AoS stride allows without an unaligned dword load — but unrolls four
blocks instead of two, so four requests are in flight per wave.

The reductions (`rms_norm`, `l2_norm`, `gated_rms_norm`, `ple_gate`) have a
grid fixed by the model (HC=4 groups for the hc modules), so the only lever
is width: the block size is now chosen by the host from `ne0` up to 1024,
with dynamic LDS.

**3. Launch count.** A dependent chain of trivial kernels pays a kernel
latency per link whatever its size, and §M5's "240 launches a token are free"
was measured on five substantial kernels a layer, not on the mostly-trivial
ones a layer here (`prof_launches_per_token` reports the exact figure; it was
never counted before this flag existed). Two fusions that
keep every oracle tap are in: `scale`+`silu` in `build_hc_mix` (32 launches a
token) and the whole GDN gate chain — `sigmoid(beta)`, `softplus(alpha+dt)`,
`*ssm_a`, `exp` — as one launch over 48 elements instead of five (48 a
token). `prof_launches_per_token` reports the rest; if `prof_elem_hc_us` is
still large against its bytes, the remaining elementwise chains
(`sigmoid`+`hc_collapse`, the shared-expert gate and add, the QSA output
gate) are the next candidates and are the same pattern.

**None of this has been run.** The CPU path is unchanged and was re-checked
against its own pre-change dump: 538 of 538 points `cos=1.000000`, STEP2
PASS — so the GPU-side rework cannot have moved the math, and
`--dump`/`--oracle` will say whether it moved the kernels.

## Step 2b (2026-09-22): the launch count, and per-projection bandwidth

The profiled run (`be26445`, box idle) gave `layers0_15_ms_median=11.7251`,
`prof_host_syncs_per_token=0` — the sync work of the first cut was right —
and `prof_launches_per_token=931`, i.e. **58 launches a layer**, with
`trunk_gemv 5890 µs` and `elem_hc 2004 µs` of an 11.7 ms token.

### Why ≤ 15 launches a layer is not reachable, and what is

Seventeen of those 58 were projections, one kernel each however small:
`hc_down`×2, `hc_inject`×2, `hc_up`×2, `ssm_qkv`, `ssm_gate`, `ssm_beta`,
`ssm_alpha`, `ssm_out`, `ffn_gate_inp`, `sh_up`, `sh_gate`, `sh_down`,
`sh_gate_inp`, expert gate_up, expert down. That is already above 15 before a
single elementwise op. The only thing that collapses them is that most
**share an activation vector**, so they can share a kernel:

| batch | matrices | rows | launches |
|---|---|---:|---:|
| `xn` → hc | `hc_down` + `hc_inject` | 324 | 2 → 1 (+1 reduce) |
| `mixed` → GDN | `ssm_qkv` + `ssm_gate` + `ssm_beta` + `ssm_alpha` | 16480 | 4 → 1 |
| `mixed` → QSA | `wq` + `wk` + `wv` + `idx_q` + `idx_k` | 13952 | 5 → 1 |
| `mixed` → FFN | `ffn_gate_inp` + `sh_up` + `sh_gate` + `sh_gate_inp` | 1793 | 4 → 1 |
| `ple_emb` → PLE | `ple_key` + `ple_value` | 12800 | 2 → 1 |

`k_gemv_batch` runs one kernel over the concatenated row space, with a
per-matrix format (the QSA batch mixes Q8_0 and BF16), output pointer and
epilogue. **Predicted** (not measured — `prof_launches_per_token` is the
measurement): **23 a GDN layer, 26 a QSA layer, ~389 a token**, against 931.

The rest of the reduction is fusions that all keep every oracle tap:

- the scale+silu and the sigmoid became GEMV **epilogues** (`GE_SCALE_SILU`,
  `GE_SIGMOID`), riding on the split reduce a 320-row matrix needs anyway;
- `ssm_qkv` writes **straight into the conv ring slot** and the PLE norm into
  its own ring slot, so neither conv needs a copy in front of it;
- the GDN conv and the whole gate chain are one launch (`k_gdn_conv_gate`),
  the gate's 48 elements riding on threads past the channel count;
- `q_conv` and `k_conv` are adjacent, so one `l2_norm` of 32 groups does both;
- `hc_combine` also produces the **next** hc_mix's norm (`k_hc_combine_norm`)
  — both reduce over the same 2 560 elements — including across the layer
  boundary, except before the PLE layer, which rewrites the residual;
- the shared expert's down-projection forms its own `silu(gate)*up`
  activation inside the GEMV (`GX_SILU_MUL`);
- `k_moe_finish` does the weighted expert sum, the shared-expert gate and the
  add in one launch, writing `ffn_moe_out` and `ffn_shexp_gated`; the sigmoid
  lives here rather than in the GEMV epilogue so `shared_expert_gate` stays a
  tap too;
- `k_qsa_qk_post` does the `[q|gate]` split, the three QK-norms, the three
  IMRoPEs and the output gate's sigmoid — nine launches with grids as small as
  four workgroups of one wave — in one kernel of 30 workgroups;
- the attention combine applies the gate and writes both `kqv_out` and
  `attn_gated`;
- **the top-k is not launched at all** when the budget covers the cache. At
  ctx 512 against a 2 051-cell budget the reference's own `ggml_top_k` is the
  identity, so `k_qsa_expand` writes the selection. That was a 1 024-thread
  workgroup doing four passes of LDS histograms and scans over ~70 scores,
  every QSA layer, and it is the largest single piece of the 292 µs a QSA
  layer cost.

### Per-projection bandwidth, and two knobs instead of two guesses

`--profile` now reports, per batch group,
`prof_gemv_<group>_us`, `_mb` and `_gbs` — the weight bytes those launches
actually read over their device time — for `hc_down_inject`, `hc_up`,
`gdn_qkv_gate_beta_alpha`, `ssm_out`, `ffn_router_shexp`, `shexp_down`,
`qsa_q_k_v_indexer`, `attn_output` and `ple`. That is the number to hold
against §M5's 800 GB/s, not the lumped 5 890 µs.

Two things the brief asked about cannot be settled without running, so both
are **runtime knobs** rather than assumptions:

- `--gemv-lds 0|1` stages the activation slice in LDS instead of reading it
  from L1. The kernel is weight-bandwidth-bound and x is small enough to sit
  in L1, so this may well be neutral; sweep it.
- `--gemv-min-rows N` (default 1024) is the row count below which K is split.
  **Total waves == total rows**, so wider workgroups never add parallelism —
  only splitting K does, at the price of a reduce launch. Sweep it against
  `prof_gemv_*_gbs`.

On 16-byte loads: two consecutive Q8_0 blocks are 68 bytes and `qs` sits at
offset 34b+2, so a block's payload is 4-byte aligned only for even b and
**never** 16-byte aligned. A `dwordx4` from such an address is not something
this file will do on the strength of a guess. The widest safe request stays
the 32 contiguous bytes a wave pulls per block; what is tunable is how many
are in flight, so the unroll depth is the compile-time `GEMV_UNROLL`
(default 4).

**Correctness:** the CPU graph after this rewrite is **bit-identical** to
before it — 538 of 538 taps `cos=1.000000` against the pre-rewrite dump,
`missing=0` — and the llama.cpp comparison is unchanged (`l_last-0`
0.999993, `l_last-3` 0.999407 under `--quant-act`). 35 kernels, still zero
scratch and zero spills, occupancy 16 except `attn_flash_split` at 15;
`k_gemv_batch` is 46 VGPR. Not run: no GPU was touched.

## Step 3 (2026-09-22): the whole model on three cards, and a real oracle

Step 2b measured `layers0_15_ms_median=7.6204` (from 11.73) at 389 launches
and 0 host syncs, with the GPU kernels matching the CPU graph 534/534.
Step 3 is design §9.5 step 3: all 48 layers, three cards, `lm_head`, and an
oracle that tests the MODEL rather than the arithmetic.

### What crosses a card boundary

Exactly one thing: the wide residual, `hc_count × n_embd` f32 = **40 KB**.
Nothing else survives a layer. `Backend::boundary_recv` is a method of the
*destination*: it records an event on the source's stream, makes its own
stream wait on it, and issues `hipMemcpyPeerAsync` on its own stream. No host
call, so a boundary is M5's ~30 µs of P2P and nothing else.
`enable_peer_access()` runs before anything is placed.

Each device has its own stream, its own scratch set (`bind()` swaps the
active pointers at a layer, so the body code never knows which card it is
on), and owns the QSA caches and GDN/conv state of its own layers.

The token time is taken on the **last** device, whose stream is idle when the
start event is recorded, so its start timestamp is the token's start. Each
token fully drains before the next begins — which costs nothing, because a
decode token depends on the previous one's argmax.

### The head

`output_hc_*` is the final hyper-connection mixer and there is no separate
output norm — "the final mixer IS the output norm" (qwen4exp.cpp:380). It
takes no inject, so only down/up are placed. `lm_head` is Q6_K, the one
tensor in that format; `fk_q6k_block_dot` ports ggml's
`dequantize_row_q6_K`. Its super-blocks are 256 weights, so a batch
containing it splits K on 256-element boundaries rather than 32.

Greedy sampling is a **device-side** two-stage argmax over the 248 320
logits, so the only thing crossing to the host per token is one int. The
full logits stay downloadable for the last token.

### The three oracles

- `--oracle` still compares every tap, now on any device.
- `--expect-ids F` compares the greedy ids with llama.cpp at temperature 0
  and prints the first mismatch.
- `--routing DIR` compares the routed expert set per layer per position
  against `DIR/moe_ids.txt` and reports the mean `|A ∩ B| / 10`.

Two things the comparator learned doing this, both of which would otherwise
have read as engine failures:

- **The dump writes the last ne1 column when ne2 == 1.** After llama.cpp
  applies `inp_out_ids` the final layer's `l_last-47` is `[2560, 4, 1]` and
  its file holds hyper-connection stream **3** alone. Comparing it against
  stream 0 read `cos=0.579`; comparing like with like reads **0.9787**.
- **`result_norm` and `result_output` are 4 bytes each** in this dump — one
  float for a 2 560- and a 248 320-long tensor. A one-element cosine is ±1
  whatever the values, so those are now REFUSED rather than reported as a
  spectacular failure. Worth fixing in the dump tool; until then the head is
  checked by the greedy ids, not by a tap.

And one the routing oracle needed: beyond the first greedy mismatch the two
engines are decoding **different text**, so a routing difference there is a
consequence of the sequence having diverged, not evidence about the router.
The report gives both numbers and the same-input one is the headline.

### What `--cpu` shows over all 48 layers

Six prompt ids then 16 greedy, `--devices 3`, against
`~/bench/franken/oracle`:

- **greedy: 7 of 16 ids identical to llama.cpp** — `271 2064 10054 1040 488
  2493 381` — then position 7 diverges (mine 8008, ref 3322).
- **routing: mean set overlap 0.909** across all 48 layers over the 13
  positions whose input token is the reference's own. Layer 0 is **0.992**,
  falling to ~0.85 by layer 47. (Over all 22 positions it reads 0.713, but
  that number mixes in positions where the sequences had already parted.)
- taps: `l_last-0` 0.99996, `l_last-15` 0.98955, `l_last-16` 0.98579,
  `l_last-31` 0.94168, `l_last-32` 0.93344, `l_last-47` 0.97872. 1 184
  compared, 410 with no dump entry, 60 incomparable, 10 refused.

So the compounding of step 2's routing divergence is now quantified at both
ends: ~9 of 10 experts agree per layer per position, and the model says the
same thing for seven tokens.

### What the GPU run needs

```
franken_decode --model <shard> --tokens 248044 785 10945 315 1495 374 \
               --devices 3 --layers 0-47 --ctx <N> \
               --greedy 16 --expect-ids ~/bench/franken/oracle/greedy.txt \
               --routing ~/bench/franken/oracle --oracle ~/bench/franken/oracle \
               --dump <dir> --time 32 --profile
```
under the rig lock with the gateway stopped. `--ctx` sizes the QSA caches
**per device**; each card holds 4 QSA layers, so 256k costs
4 × 256k × (2 heads × 256 dims × 2 tensors q8_0 + indexer) ≈ 1.5 GB — design
§9.1's figure. `vram_report` prints used/free/total per card after placement
and again after the scratch, so the fullest card's headroom is a measurement
rather than an estimate. Nothing here has been run on a GPU.

### Step 3 fix (2026-09-22): the placement bug, and the cache budget

The first GPU run died at `--ctx 262144` **and again at `--ctx 4096`** with
`out of VRAM placing blk.18.ffn_down_exps.weight` on empty cards. It was not
the caches.

**`hipMalloc`, `hipMemcpy` and a kernel launch all act on the CURRENT device,
and the current device is process-wide state — not something a stream
carries.** `GpuBackend` set it once in its constructor, so after
`make_gpu_backend(0..2)` the current device was 2 and every later allocation
went there whichever backend was asked. Placement filled one card with all 48
layers and died at layer 18: 24 GB / ~1.34 GB a layer is 17.9. Step 2 never
saw it because with a single backend the constructor's `hipSetDevice` was also
the right one for everything after it.

Every entry point that allocates, copies or launches now ensures its own
device first (`dev_ensure`, a compare against a process-global, so the cost is
one predictable branch on the launches that do not switch). And the failure is
caught where it happens: `verify_placement` fails early and says so when a card
holds under 80 % of what it was asked to place, instead of a message about the
wrong tensor a dozen layers later. `vram_dev<i>_used_gb=` is printed **before**
placement, after it, and after the caches.

**The cache sizing was also wrong, and is now the design's.** The indexer kept
every token's RAW key in f32 to re-pool from — 512 B a token a layer, *eight
times* the 64 B design §9.1 budgets for the pooled bf16 key, and 134 MB a layer
at 256k. It does not need them: a block's members arrive in order, so

```
pooled = (sum_of_members_so_far + (r - n_filled) * key_of_cell_0) / r
```

is the reference's formula exactly — unset slots read cell 0
(`llama-memory-hybrid-idx.cpp:395`) — from a running sum and one saved key,
`IDX_DIM` floats each. Two more buffers stopped scaling with `ctx`: the f32
shadows of the pooled cache (67 MB a device, for two taps the comparator marks
INCOMPARABLE at any other cache depth anyway) and the selection array, which is
capped by the *budget*, not the cache. The attention split scratch was already
independent of `ctx` (33 chunks × 24 heads × 256 dims = 811 KB).

Measured by the new `kv_bytes_dev<i>=` line at `--ctx 262144`:

```
kv_bytes_dev0=1.208 GB  (qsa_layers=4 x 302.0 MB, 1152 B/token/layer)
gdn_state_dev0=0.040 GB (gdn_layers=12)
```

1152 B/token/layer is 1088 (K and V, 2 heads × 256 dims in q8_0) + 64 (one
pooled bf16 indexer key per 4 cells) — design §9.1's number, and 1.208 GB a
device against its ~1.2 GB. With ~21.5 GB of weights the fullest card sits at
~22.8 GB of 25.77.

The CPU graph is **bit-identical** after the rework: 538 of 538 taps
`cos=1.000000` against the step-2 dump.

### Step 3 fix 2 (2026-09-22): the device-2 page fault

`Memory access fault by GPU node-3 ... Page not present` was **`lm_head`
reading past its own allocation**, and it was one line:

```c
b.d[i].row_stride = (W.type == FK_Q_Q8_0) ? W.row_bytes : (size_t) W.K;
```

`k_gemv_batch` addresses a **quantised** row in BYTES and an f32/bf16 row in
ELEMENTS — the two pointer casts in the kernel. The test above was
accidentally right while Q8_0, BF16 and F32 were the only formats a batch
ever saw. Q6_K is the first quantised format other than Q8_0 to go through
it: its rows are **2 100 B** but `W.K` is **2 560**, so

```
row 203 700 of 248 320 first addresses past the 521.5 MB tensor
the last row addresses 635.7 MB — 114 MB past the end
```

on exactly the card that holds `lm_head`. `place_mat` had already validated
`row_bytes × rows == nbytes` at load; the wrong field was passed at launch.

A second bug in the same launcher, found looking for the first: the LDS slice
was `ceil_div(units, nsplit) * WAVE`, but `units` counts *granules*, which are
256 for a Q6_K batch and 32 otherwise. That under-sized the staging buffer 8×
and let the staging loop write past it. It is `* gran` now.

**`--sync-debug`** drains the stream and checks `hipGetLastError` after every
launch and copy. The check runs at the *next* op's `mark()`, so the op named
in the message is the one that faulted rather than whichever launch was in
flight when the queue drained; it prints the op, its class, the phase and
layer, and for a boundary the pointer ranges and both device ids.

**stdout is line-buffered** (`setvbuf` at the top of `main`) because a GPU
memory fault kills the process and takes a full stdout buffer with it — which
is why the first three-card run produced no output at all. Progress lines now
mark placement, cache allocation, the first token, each layer range as it
finishes on its device, and the head.

**Audited for buffers used off their owner**, all clean: the boundary
destination is the destination device's `res_hc` and the event is now recorded
with the *source* device current; the cross-layer norm fusion is off whenever
`layer(il+1).dev != layer(il).dev`, so layers 16 and 32 normalise the local
copy they just received; all 61 `Scratch` members are rebound by `bind()`;
the embedding and PLE uploads target device 0's buffers through device 0's
backend; the argmax scratch, logits and head scratch are the last device's;
the `__constant__` decode tables are filled per device in each constructor
after its `hipSetDevice`; and every `Mat` and `LayerState` buffer is placed
through `dev_for(il)`.

### Step 3b (2026-09-22): the host issue rate

The three-card run gave `layers0_47_ms_median=28.38` with dev0 at 12.04 ms for
389 launches and dev1 at 7.6 ms of compute for the same layer shapes. dev1 and
dev2 have their whole token pre-queued while they wait on a boundary; **dev0
does not**, so dev0's device-timeline intervals are the host's issue period —
~31 µs an op against §M5's 3 µs launch floor.

`issue_ms` measures it directly: the host wall time to **enqueue** one token's
whole body and head, with nothing waited on. `prof_cpu_per_launch_us` is that
over the launches all devices made. Printed by `--time`. Run it with and
without `--profile` and the difference is what the instrumentation costs.

Per-launch host work removed:

- **The routed-expert log was two device-to-device copies a layer** — 32 a
  token — and they stalled the queue badly enough to make a `--routing` run
  1.8× slower on *every* card. `k_router` now writes the log itself, so
  capture costs **no extra op at all**.
- **`--profile` recorded an event per op.** It now records only on a **class
  change**: the interval from event i to event i+1 is charged to `ev_cls_[i]`,
  so a run of ops in one class needs one event. A GEMV always takes its own,
  because the per-projection bandwidths are per launch group.
  `prof_events_merged_per_token` says how many were saved.
- **The batch descriptor is a by-value kernel argument**, so its size is host
  work on every GEMV launch. `GEMV_BATCH_MAX` is 5 — the largest batch is the
  QSA projection's five — not 8: ~260 bytes a launch instead of ~404.
- **A `std::vector<float>` was allocated per token** for the embedding row.
  Hoisted to a member.
- `--time` turns the `--verbose`/`--routing` capture off, so a timing loop
  measures the engine and not the instrumentation; `--dump` already recorded
  only the last prompt token.

`debug_check` is a single `if (!sync_debug_) return;`, and nothing else in the
issue path calls a blocking HIP entry point — `prof_host_syncs_per_token`
reads 0 and is the check that it stays that way.

The CPU graph is still bit-identical (538 of 538 taps `cos=1.000000`), and the
routing capture through the router kernel reproduces the same overlap.

## Step 4 (2026-09-22): the engine behind the gateway

`franken_decode --serve` (or `SERVE=1`, which is what `c/openai_server.py`
sets) is the same binary speaking the gateway's line protocol on stdin/stdout
instead of running a CLI turn. The protocol is specified, with a `file:line`
for every claim, in `GATEWAY-PROTOCOL.md`; the implementation is
`franken_serve.cpp`, and §7 of that document is the map from one to the other.

```
franken_decode --serve          # every setting comes from the environment
franken_decode_cpu --serve-test # the same loop, CPU backend, short span
```

Six decisions in it that are not obvious:

- **stdout is the wire, so fd 1 is pointed at stderr.** `wire_open()` dups the
  real stdout to a private unbuffered `FILE *` and then `dup2(2, 1)`. Every
  `printf` already in this directory — the runner's progress lines, the VRAM
  report, libllama's loader chatter — lands in the log, and no future one can
  corrupt a frame.
- **The tokenizer is `dlopen`ed, not linked.** `ldd libggml.so.0` lists
  `libggml-hip.so.0`, so `-lllama` would put a HIP runtime in
  `franken_decode_cpu`'s dependency list and break `make ldd-check`. It is
  also loaded **after** the weights are placed: the conformance log shows
  `ggml_cuda_init` running during a *vocab-only* load, so on a box with cards
  visible ggml does enumerate them, and doing it after this engine's own
  `hipSetDevice` means it finds contexts rather than creating them.
  (`libhipblas.so.3`/`librocblas.so.5` are not on this host's path — /opt/rocm
  here is 6.2 and the build is 7.14 — so the two are preloaded by absolute
  path from the rocm SDK wheels. `LD_LIBRARY_PATH` is deliberately *not* used:
  that directory also holds a different `libamdhip64`.)
- **Prefix reuse is a rollback, not a checkpoint cache.** The K/V cells and the
  pooled indexer keys are positional and stay valid up to any earlier
  position; what is *not* is the recurrent state (GDN state + conv windows,
  the indexer's running block sum, the PLE conv window). `DecodeRunner::
  save_state/load_state/reset_state` move exactly that set — 118 MB a snapshot
  at 48 layers, host side, one download a card — taken at every prefill chunk
  boundary and once at the end of every turn, eight kept per slot with
  decimation on eviction (the useful snapshot is the one just below the shared
  prefix's end, and that end is anywhere).
- **`reused` is the position rolled back to**, i.e. the exact number of tokens
  the turn did not run through the model. The end-of-turn snapshot is what
  makes a continuation report *exactly* `prompt + completion (- 1 if the turn
  hit its budget)`, which is the gateway's own independent prediction —
  `accept_live.sh` check 2b fails the gate on any disagreement.
- **Feed, then sample.** An emitted token is fed back before the next one is
  sampled, which is what makes the two cases above land on the ledger's
  arithmetic instead of one token away from it.
- **The PLE gather is windowed.** The hash reads a token and its two
  predecessors and nothing else, so a chunk gathers over
  `[start-2, start+T)` rather than the whole prefix — the CLI's per-token
  whole-prefix gather is O(pos) a token and unusable at 256k.
  `FRANKEN_PLE_SELFCHECK=1` checks the window against the full gather
  (`--serve-test` has it on: every row `IDENTICAL`).

### The CPU conformance run (2026-09-22)

`serve_conformance.py` plays the gateway's half by hand — the exact SUBMIT
bytes `Engine.generate` writes, the exact parse `Engine._dispatch_stdout`
does — against `franken_decode_cpu --serve-test` with `--layers 0-3`,
`--chunk 8`, `--ctx 512` and `HIP_VISIBLE_DEVICES=` (the second guard; the
first is the binary). 14 checks, 0 failures: boot handshake, CONTEXT_EXCEEDED,
EMPTY_PROMPT, a stray CANCEL answered NOT_FOUND, a full round trip
(ACCEPT/DATA×n/DONE with all seven STAT fields), a continuation reusing
`prompt + completion - limited` exactly, a **rollback** to the snapshot below
the LCP (24 of 26 tokens at chunk 8), a CANCEL in flight ending `DONE` then
`ERROR CANCELLED`, and a normal request served after it. The output is
garbage text, as it must be with 4 of 48 layers — this run tests the
protocol, not the model.

### What the GPU run needs (nothing here claims a GPU measurement)

```
tools/hot-expert/serve_alt.sh franken     # stops GLM, takes the rig lock,
                                          # warms the GGUF, starts the gateway
                                          # with franken_decode behind it,
                                          # sends one real chat, then runs
                                          # accept_live.sh
tools/hot-expert/serve_alt.sh glm         # the way back (accept_live must PASS)
```

`start_franken.sh` is what it launches: `openai_server.py --arch qwen38`
(Qwen3.8's own chat template — the model's) `--model-id glm-5.3-flash` (the
name Open WebUI already has), `--max-tokens 4096`, `--kv-slots 1`, with
`FRANKEN_CTX=262144 FRANKEN_CHUNK=256 FRANKEN_GEMM_LDS=0 FRANKEN_DEVICES=3`.
From the Mac, `tools/hot-expert/accept_ui.sh` is the browser-side check the
owner is owed before anyone says serving works.

Known gaps, all of them unmeasured rather than unknown:

1. **One slot at 262144 cells**, because that is what the cards have room for
   (1.21 GB a card a slot). Slots are allocated lazily and a slot that does not
   fit answers `ERROR <id> SLOT_UNAVAILABLE` rather than killing the engine.
   The qwen38 family caps `--kv-slots` at 1 anyway.
2. **No prefix checkpoints.** `accept_live.sh` check 3 (the API two-turn with
   memory) is the checkpoint path on GLM; here it has to pass on plain slot
   reuse or not at all. `GATEWAY-PROTOCOL.md` §5 argues it should; nothing has
   run it.
3. **Sampling drops the tail below rank `FRANKEN_TOPK_CAND` (4096)** before
   applying top_p. Greedy (`temperature 0`) uses the device argmax and
   downloads no logits at all; every other temperature costs one ~1 MB logit
   download a token.
4. **The snapshot cost at 256k is not measured**: 118 MB a chunk boundary is
   ~2.6 % of a 256-token chunk's compute at the recorded 1.63 ms/token, on
   paper. `FRANKEN_SNAP_EVERY` (tokens) is the knob; raising it trades reuse
   granularity for prefill speed, and taking a snapshot forces the chunk to be
   awaited, so it also costs the pipeline.
5. **Tool calling** is parsed gateway-side out of the DATA text for this family
   as for every other; nothing about it has been exercised here.

### Step 4 fix (2026-09-23): what the first served run broke, and why

The engine served the owner's gateway on 2026-09-22 (commit `3ae6bdb` plus a
docker wrapper). It answered — "I am Qwen, and 12 plus 30 is 42", 4916-token
prefill 6.5–6.7 s, decode 22–27 tok/s, first token on screen 6.84 s through
`accept_ui.sh`, CANCEL mid-prefill and the request-behind-an-abandoned-one both
fine. Every failure was prefix reuse, and they had three distinct causes.

**1. The number the gates read was not on the wire this engine wrote to.**
`accept_live.sh` check 3 and `owui_ui_turn.sh` both take `reused` from
`grep " REUSE <id> " | awk '{print $(NF-1)}'` over the gateway log — glm53's
own stderr line (`c/glm53.c:6717`), not the DONE frame's 8th field. This
engine's DONE was correct and the line did not exist, so check 3 read an empty
string ("integer expected") and check 2 read `owui_ui_turn.sh`'s `'0'`
default. It now prints `[serve] REUSE <id> <reused> <prompt_tokens>`, and the
conformance driver checks that line against the frame for every request.

**2. A checkpoint that carries only the recurrent state is worthless the
moment another request touches the slot.** With `--kv-slots 1` the gateway
routes *every* conversation to slot 0. A 58-token request between UI chat A
and UI chat B re-prefilled from its own zero and overwrote the cells the
4.6k-token tool block lived in; restoring A's recurrent state into those cells
would have been reuse over another conversation's keys, so the engine (which
dropped its snapshots on any reuse=0 turn) reported `reused=0` and re-prefilled
everything. A checkpoint is now **both halves**: the recurrent state, always,
and the positional cells [0, pos) — K/V q8_0 plus the pooled indexer keys,
~13.8 kB a token — **copy-on-write**. Nothing is copied while the slot's own
cells still hold that prefix (the continuation case, free); the copy happens
once, for the deepest threatened checkpoint, at the instant something is about
to overwrite them. Checkpoints are never dropped for belonging to another
conversation — with one slot, the prefix they share is the whole point.
`turn D` in the conformance suite is that exact sequence and it is now a PASS.

**3. Nothing under one chunk could be reused** — answered by the end-of-request
checkpoint, which was already taken and is now proven by phase 2 of the driver
(chunk checkpoints switched off entirely, a 19-token follow-up still reuses
exactly the 10 tokens the ledger predicts).

Two more things came out of the env diff against `~/start_glm53.sh`:
`COLI_PREFIX_PIN=1` (Open WebUI rebuilds its `memory_context` block every turn;
without the pin the shared prefix changes near its head and reuse is lost
before the engine ever sees it) and `COLI_THINK=0`, plus the one-line gateway
change that makes the second one work: `ARCH == "qwen38"` forced xhigh thinking
on clients that asked for nothing (`c/openai_server.py:5041`), which is why
every UI answer arrived behind a `reasoning_content` block. The default is
unchanged; only `COLI_THINK=0` opts out. The ledger (`accept_live.sh` check 2b)
stays SKIP by code, not by configuration: `ledger_enabled()` is
`ARCH == "glm53" and COLI_LEDGER != "0"`, and widening it would have
`_ledger_record` build part-less entries, which risks `ledger=broken` — a FAIL
where there is a SKIP today.

**And the instrument the next GPU run needs.** The one thing that could not be
answered from the served log was whether a rollback re-prefilled only the tail
("reused=4096, prefill 6.72 s" looked like a full prefill). Every request now
prints where it started, how many chunks it ran, and where the wall clock went:

```
[serve] req=3 slot=0 prompt=93 reused=0 from=0 chunks=12 emitted=4 limited=1 \
  cancelled=0 prefill_s=38.64 [restore=6 protect=0 ple=1249 step=37397 snap=76 ms] \
  decode_s=1.94 tok/s=2.06 ckpts=8 ckpt_mb=81
```

`from`/`chunks` make "only the tail" a checked fact (the driver asserts
`chunks == ceil((prompt - reused) / chunk)` on every rollback), and `ple` is
there because the PLE gather reads scattered rows of a 28.8 GB table: the CPU
run above shows 1249 ms of a 38.6 s cold prefill and **90 ms for a single
token** on a cold row against 2 ms warm. At the served scale that term, not the
forward pass, is the first suspect for a slow tail — record §PLE-GATHER puts a
cold row at 4.2 ms against 9.5 µs warm, and design 9.1's pinned PLE table is
the fix if it is.

### Step 4 fix 2 (2026-09-23): the checkpoints were eating the prefill pipeline

The second serve reported the reuse fixes working and a prefill that had gone
from 6.5 s to 42.2 s for a 4 881-token UI prompt:

```
prefill_s=42.18 [restore=47 protect=2 ple=3784 step=36252 snap=2300 ms] ckpt_mb=944
```

`step` is 36.3 s for 4 881 tokens = **7.4 ms a token, which is the UNPIPELINED
rate** (record §L0-PREFILL-2: 7.7 unpipelined, 3.0 pipelined, 1.63 with
`--gemm-lds 1`). The instrumentation added the day before is what made that
readable, and the cause was in this file: a checkpoint per chunk, each one a
`sync()` of all three cards plus a blocking 118 MB `download()`. Every chunk
therefore ended in a full device drain, and the three-card overlap — the thing
`08fc15f` exists for — could never happen. Three changes:

- **A checkpoint copy is stream-ordered and never awaited.**
  `Backend::download_async` issues the copy on the engine's own stream, which
  puts it behind the chunk that has just run and in front of the one that comes
  next: exactly the ordering a checkpoint needs, with no host wait and no
  drain. The destination must be pinned or `hipMemcpyAsync` silently becomes
  synchronous (the same trap the upload staging ring exists for), so the blobs
  come from a **pool** of `hipHostMalloc`'d buffers — pinning 118 MB costs real
  time and a checkpoint taken mid-prefill must not pay it. The one wait a
  request owes its checkpoints is a single `sync_devices()` after the last
  token, where the host is about to write DONE anyway. The prefill loop no
  longer forces a flush for anything but its final chunk.
- **The schedule is the chat template's turn boundaries, not an interval.**
  `accept_live.sh` check 2 wants reuse within 256 tokens of where two UI chats
  diverge, and that point is not a multiple of anything — it is the start of
  the last user turn, which begins with `<|im_start|>`. So `snap_points()`
  takes a checkpoint at the last `FRANKEN_SNAP_TURNS` (3) boundary tokens, with
  `FRANKEN_SNAP_EVERY` (512) as a floor under them, and the prefill chunks stop
  on those positions. This is what glm53 gets from the gateway's `prefix_bytes`
  hint, which the gateway computes for glm53 alone (GATEWAY-PROTOCOL.md §2);
  finding it in the token stream needs no hint and no gateway change. A
  4 881-token prompt now takes ~6 checkpoints instead of 19, and each costs a
  copy the host does not see.
- **The copy-on-write image is asynchronous and bounded too**, and if the
  deepest threatened checkpoint's cells do not fit `FRANKEN_SNAP_BUDGET_MB`,
  the ones that do are protected and the rest are dropped — a request is never
  stalled behind a multi-GB copy.

**The PLE gather's 3.8 s** is design 9.1's other half: the table's rows are
scattered over 28.8 GB and a row whose page is not cached costs 4.2 ms against
9.5 µs (record §PLE-GATHER). `FRANKEN_PLE_RESIDENT=1` (default in
`start_franken.sh`) reads the whole table once at boot into memory this process
owns, after which no gather can fault; the load time and rate are printed.
`FRANKEN_PLE_PINNED=1` uses `hipHostMalloc` instead and is **not** the default:
nothing DMAs from this table — the gather dequantises on the CPU and uploads
the result — so pinning 28.8 GB would cost registration time for a transfer
that never happens.

Also: `FRANKEN_GEMM_LDS=1` is now the serving default (the LDS-tiled trunk
GEMM, T > 1 only; decode is untouched and stays bit-identical), because it is
llama.cpp's prefill rate against half of it. Its divergence from the CPU
reference has not been measured — that is the open item, and this is the one
line to change if the quality harness ever rules against it.

### Verifying the pipeline claim (GPU, under the rig lock)

`--snap-every N` makes `--time-prefill` take serving checkpoints on exactly the
serving path (pinned host memory, `download_async`, never awaited), so the cost
is one number against another:

```
# gateway stopped, rig lock held, cards empty
M=~/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
./franken_decode --model $M --time-prefill 4096 --chunk 256 --ctx 8192 \
                 --gemm-lds 1 --snap-every 0     # the baseline
./franken_decode --model $M --time-prefill 4096 --chunk 256 --ctx 8192 \
                 --gemm-lds 1 --snap-every 512   # with checkpoints
```

`prefill_ms_per_token` must agree within 5 %; `prefill_snapshots` says how many
were taken. **The CPU binary cannot answer this** — it has one backend, no
streams and no overlap to lose, so its `snap=` is a memcpy and its
`prefill_ms_per_token` measures nothing about the pipeline. Nothing here claims
a GPU measurement.

## Decode step 1 (2026-09-23): what the 7.6 ms a card is made of, and four changes to it

§L0-256K left decode at **26.1 ms a token at 256k depth (38.4 tok/s)**, ~7.6 ms
of GPU time a card, and named the owner: "the small-matrix trunk GEMVs and the
~390-launch floor". This section is what those two actually are, measured off
the numbers already in the record rather than guessed, and the four changes
that follow from it. **Nothing here has been run on a GPU.**

### The diagnosis: achieved bandwidth tracks the CONTIGUOUS RUN A WAVE READS

Step 2b reported per-projection GB/s and they look scattered — 128 to 497 — until
they are put next to the bytes ONE WAVE reads in a row. A wave owns one output
row of one split, so that run is `row_bytes / nsplit`, and with
`--gemv-min-rows 1024` the only decode batch that splits at all is
`hc_down`+`hc_inject` (324 rows):

| group | rows | K | nsplit | **contiguous bytes a wave** | measured GB/s |
|---|---:|---:|---:|---:|---:|
| `shexp_down` | 2560 | 640 | 1 | 680 | 141 |
| `hc_down_inject` | 324 | 10240 | **50** | **238** | **128** |
| `hc_up` | 10240 | 320 | 1 | 340 | 199 |
| `ffn_router_shexp` | 1793 | 2560 | 1 | 2720 | 245 |
| `qsa_q_k_v_indexer` | 7808 | 2560 | 1 | 2720 | 434 |
| `gdn_qkv_gate_beta_alpha` | 16480 | 2560 | 1 | 2720 | 461 |
| `ple` | 20480 | 2560 | 1 | 2720 | 484 |
| `attn_output` | 2560 | 6144 | 1 | 6528 | 493 |
| `ssm_out` | 2560 | 6144 | 1 | 6528 | 497 |

Two things fall out and neither is what "small matrices do not fill the card"
suggests. **Wave count is not the problem** — `hc_down_inject` runs 16 400 waves
and `ffn_router_shexp` 1 793, and the 1 793-wave one is the faster of the two.
**Bytes per launch are not the problem either** — `ffn_router_shexp` moves
4.9 MB at 245 GB/s and `attn_output` moves 16.7 MB at 493.

What does fit is a straight line through the whole table in *launch* terms:
`t ≈ 7 µs + MB / rate`, with the marginal rate rising from ~340 GB/s where a
wave reads a few hundred bytes to ~630 GB/s where it reads kilobytes. The 7 µs
is the launch floor — and it is the same number step 2b measured the hard way,
931 → 389 launches for 11.73 → 7.62 ms, i.e. **7.6 µs a launch removed**.

`ffn_router_shexp` at 245 with a 2 720-byte run is the one row that does not fit
the run-length story, and it is the one row where 1 793 rows is 224 workgroups —
9.3 waves a SIMD, under half the card. So both effects are real and they are
separable: run length sets the rate, wave count sets whether the card is full,
and the launch floor is a flat ~7 µs on top of both.

`--profile` now reports the sub-10 MB set as a class of its own,
**`prof_gemv_small_us` / `_mb` / `_gbs` / `_launches`**, because that set —
`hc_down_inject`, `hc_up`, `ffn_router_shexp`, `shexp_down` — is 1.99 ms of the
4.40 ms trunk for 278 MB of the 1 295 MB, and it is what the four changes aim at.

### 1. The split rides on grid.x, so the card reads the tensor in order

`hc_down`'s 50 splits were `blockIdx.y` and its 41 row groups `blockIdx.x`.
Workgroups dispatch with **x fastest**, so the card ran all 41 row groups of
split 0, then all 41 of split 1: it touched the tensor at 50 scattered offsets
in 238-byte pieces, 41 rows (446 KB) apart. Swap the axes and 50 consecutive
workgroups sweep the SAME eight rows along K, in order.

This is an **address** change and not an arithmetic one — split `s` still owns
`split_range`'s block range `s`, still writes partial `s`, and the reduce still
sums `s = 0 .. nsplit-1`. It is the cheapest line in this section and, on the
table above, the one with the most to gain: `hc_down_inject` is 913 µs a card a
token at 128 GB/s against 3.65 MB of bytes.

It is done **only at TILE == 1**, from the template parameter, so the prompt
chunk's kernel is byte-for-byte the one it was (checked: the `<8,1>` ISA is
op-for-op `<8>`'s, 135 VGPR, occupancy 10, unchanged).

### 2. The split reduce happens inside the GEMV

`k_reduce_splits_gemm` was a second launch for a kernel that reads 50 floats a
row. `cnt` is now one counter per output row: each wave stores its partial,
fences, and increments; the wave that comes back `nsplit-1` has every partial
visible and sums them **in s order — k_reduce_splits_gemm's loop, copied, not
approximated** — applies the epilogue and writes the destination, then puts the
counter back to 0 so the next launch finds it as this one did.

32 launches a card a token (`hc_down_inject` is the only split GEMV on the
decode path). `--gemv-fused-reduce 0` restores the two-launch form, and the two
must agree bit for bit; the knob exists because the fence/counter pairing is new
here, not because a bit can move. The ISA confirms the pairing:
`s_waitcnt_vscnt 0` + `buffer_gl1_inv` + `buffer_gl0_inv` before
`global_atomic_add_u32 ... glc`, and the invalidate pair again after it.

### 3. `build_hc_mix`'s collapse rides on the gate GEMV's workgroup

```
mixed[i] = mean over the HC streams of xn[c][i] * sigmoid(gate[c][i])
```

`hc_up`'s 10 240 rows are HC streams of `n_embd`, and the collapse wants all HC
of them for ONE embedding index. A workgroup is `GEMV_WAVES` = 8 = 2*HC waves,
so instead of eight consecutive rows it now takes **two indices and every stream
of each** — same rows, same per-row dot, different wave → row map — and the
gate values it has just produced are exactly what the collapse reads. Same
expression, same order (`c = 0` first, then 1..HC-1), same scale.

32 launches a card a token, plus one at the head. `gate` is still written, so
`hc_gate` is still an oracle tap; `--gemv-fuse-collapse 0` is the old pair.
It is expressed as ONE backend call, `gemv_hc_gate_collapse`, whose DEFAULT
implementation is the two ops — which is what the CPU backend runs, unchanged.

### 4. More loads in flight, at constant summation order

The baseline ISA waits on `vmcnt(6)` of eight outstanding loads: the compiler
pipelines about one iteration ahead, so a wave whose row is ten blocks long
(`hc_up` at K = 320) never gets more than two rounds of memory latency in
flight. `--gemv-burst N` issues `N * GEMV_UNROLL` weight loads before the first
multiply.

**It is not a deeper unroll.** The accumulator count stays `GEMV_UNROLL`, block
`g` still lands in `acc[g % GEMV_UNROLL]`, and the order within each accumulator
is still increasing `g` — which is exactly what the unroll-4 loop does across
its iterations. It consumes a multiple of `GEMV_UNROLL` blocks, so the loops
after it see the same alignment and the same tail.

**And that claim is checked the way PREFILL.md section 8 had to check the down
gather**, because this file has already been bitten once by a reassociation that
was not one: `-ffp-contract=fast` lets `acc += w*x` compile to either
`v_fma` or `v_mul` + `v_add`, the backend picks by how much ILP it has, and
*more ILP is exactly what this change adds*. So the ISA was read, not assumed:

| kernel | Q8_0 loop body | closing |
|---|---|---|
| `<1>` (before) | 4 cvt, 4 `v_fma_mix`, 4 fmac | 1 `v_fma_f32` |
| `<1,1>` | 4 cvt, 4 `v_fma_mix`, 4 fmac | 1 `v_fma_f32` |
| `<1,2>` | 8 cvt, 8 `v_fma_mix`, 8 fmac | 4 fused |
| `<1,4>` | 16 cvt, 16 `v_fma_mix`, 16 fmac | 4 fused |
| `<8>` (before) / `<8,1>` | 4 cvt, 4 `v_fma_mix` | — |

`<1,1>` is op-for-op the kernel it replaced and `<8,1>` is op-for-op the prompt
chunk's, so the burst is the only thing that moves. **Every multiply-accumulate
is fused at every burst** — not one `v_mul_f32`/`v_add_f32` pair appears — so
the contraction decision did not change with the ILP.

`make resources`: all three TILE = 1 instantiations are **88 VGPR (from 58),
occupancy 16 waves/SIMD, 32 B of LDS, 0 spills, 0 scratch**. Occupancy is the
number that matters and it is at the ceiling — the TILE = 8 kernel's problem was
135 VGPR buying 10 waves, and 88 is comfortably under the 96 that 16 waves
allows. `<8,1>` is unchanged at 135/10.

### What this is expected to be worth, and what it is not

Against the 4.40 ms of trunk GEMV and 0.90 ms of elementwise a card:

| change | launches saved | what it attacks | expected |
|---|---:|---|---|
| 1. split on grid.x | 0 | `hc_down_inject` 913 µs at 128 GB/s | ~0.5 ms |
| 2. reduce in kernel | 32 | the launch floor | ~0.2 ms |
| 3. collapse in the GEMV | 33 | the launch floor | ~0.25 ms |
| 4. burst | 0 | `hc_up`, `shexp_down`, `gdn_proj` latency | 0.2-0.4 ms |

so ~7.6 → ~6.4-6.6 ms a card, launches ~389 → ~325. **That is short of the
6.0 ms and the 250 launches the brief asks for, and the honest reason is that
the remaining launches are not fusions that were missed.** What is left per
layer is `hc_combine_norm` (×2), `hc_collapse`'s partner ops, `conv_slide`,
`l2_norm`, `gated_rms_norm`, `silu_mul`, and each one is either a reduction
whose consumer is on another workgroup or a buffer hazard between workgroups;
folding them needs either grid-wide synchronisation (which costs what the launch
costs) or recomputing a reduction per consumer. Two levers that WOULD close the
rest, both bigger than this change:

- **HIP graph capture of the token body.** 389 dependent dispatches at a ~7 µs
  floor is ~2.7 ms of the 7.6; a captured graph replays them without the
  per-dispatch round trip and fuses nothing, so it cannot move a bit. This is
  the single largest remaining item and it is not a kernel change.
- **`silu_mul` into `moe_down`** (16 launches) and the MoE tail into
  `hc_combine_norm` (32) are both reachable, but both move a
  multiply-accumulate into a different ILP context, which is exactly what
  `-ffp-contract=fast` decides on — so each needs its own ISA check, and
  neither should be taken on the strength of a launch count.

### The knobs, and what each is for

| knob | env | default | off means |
|---|---|---|---|
| `--gemv-fused-reduce 0\|1` | `FRANKEN_GEMV_FUSED_REDUCE` | 1 | the second launch is back |
| `--gemv-fuse-collapse 0\|1` | `FRANKEN_GEMV_FUSE_COLLAPSE` | 1 | `hc_collapse` is its own launch |
| `--gemv-burst 1\|2\|4` | `FRANKEN_GEMV_BURST` | 4 | 1 is the pre-change loop exactly |

All three default ON and all three are bit-identical either way, so they are
A/B switches for attribution rather than a numerics decision. The grid-axis swap
(change 1) has no knob: it moves no arithmetic at all, only an address.

### The gates

**CPU, run (no GPU):** the graph rewiring — `hc_mix` and the head now call
`gemv_hc_gate_collapse` — must not move the CPU path, which takes the default
two-op implementation. `franken_decode_cpu --layers 0-47 --chunk 1` against the
pre-change dump: see the commit body.

**GPU, NOT run here.** Under the rig lock with the gateway stopped:

```
cd ~/src/colibri-m1/tools/hot-expert/franken/decode
M=~/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
T="248044 785 10945 315 1495 374"

# (a) the pre-change path, out of the SAME binary: all three knobs off
./franken_decode --model $M --tokens $T --devices 3 --layers 0-47 --ctx 512 \
  --chunk 1 --greedy 16 --gemv-fused-reduce 0 --gemv-fuse-collapse 0 --gemv-burst 1 \
  --dump ~/bench/franken/dec1_off

# (b) the defaults against it -- must be every tap cos=1.000000, maxabs=0
./franken_decode --model $M --tokens $T --devices 3 --layers 0-47 --ctx 512 \
  --chunk 1 --greedy 16 --oracle ~/bench/franken/dec1_off

# (c) and against the CPU graph, which is the oracle of record
./franken_decode --model $M --tokens $T --devices 3 --layers 0-47 --ctx 512 \
  --chunk 1 --greedy 16 --dump ~/bench/franken/dec1_on
./franken_decode_cpu --model $M --tokens $T --devices 3 --layers 0-47 --ctx 512 \
  --chunk 1 --threads 8 --oracle ~/bench/franken/dec1_on

# (d) the prompt chunk is untouched, and this says so
./franken_decode --model $M --tokens $T --devices 3 --layers 0-47 --ctx 512 \
  --chunk 6 --greedy 16 --oracle ~/bench/franken/dec1_off
```

Timing, all three cards, 256k allocated — the headline is (f):

```
# (e) short depth, and the attribution
./franken_decode --model $M --tokens $T --devices 3 --layers 0-47 \
  --ctx 262144 --time 32
./franken_decode --model $M --tokens $T --devices 3 --layers 0-47 \
  --ctx 262144 --time 32 --profile      # prof_gemv_small_*, prof_launches_per_token

# (f) 256k depth, the §L0-256K point: 26.06 ms/token is what this must beat
./franken_decode --model $M --tokens $T --devices 3 --layers 0-47 \
  --ctx 262144 --chunk 256 --gemm-lds 1 --time-prefill 262000 --time 32

# (g) the A/B, interleaved A,B,B,A as CLAUDE.md requires, at short depth
for k in "" "--gemv-burst 1" "--gemv-fused-reduce 0" "--gemv-fuse-collapse 0"; do
  ./franken_decode --model $M --tokens $T --devices 3 --layers 0-47 \
    --ctx 262144 --time 32 $k
done
```

`--time 32` prints `layers0_47_ms_median` and `issue_ms`; `--profile` costs the
first card and must not be in the headline run (§L0-STEP3).

### Step 4, check 3 (2026-09-23): the renderer was not the problem

The third serve passed everything but `accept_live.sh` check 3, with
`prompt=203 reused=161 (need >=171)`, and the reading offered was that the
gateway re-renders a previous assistant turn without the empty thinking block
the prompt had at generation time, so the sequences diverge at the assistant
header. **They do not.** Rendered and tokenized against the served tokenizer:

```
p1 = <|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
p2 = p1 + "Four" + <|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
p2.startswith(p1 + answer) -> True;  tokens(p2)[:24] == tokens(p1)  -> True
```

`render_chat_qwen38` renders an assistant turn as
`<think>\n{reasoning.strip()}\n</think>\n\n{text}`, which with no
`reasoning_content` **is** `<think>\n\n</think>\n\n` + text — the same bytes the
generation prompt ended with. The invariant is now a test
(`test_qwen38_follow_up_prompt_extends_the_generation_prompt`, 172/172 pass).

What the numbers actually say: turn 1 was `prompt=160 emitted=1 limited=0`, so
the engine held 161 tokens, and it reused **161** — every token it had, which
is exactly `ledger_expect_reuse`'s prediction of `prompt + completion`. The
remaining 42 are the tail the second prompt adds *after* the previous answer,
and none of it has ever been through the model. That tail is the template's:
for the same two turns, GLM's costs 21 tokens and Qwen3.8's costs 38, against a
check that allows 32. The full table and the reasoning are in
`GATEWAY-PROTOCOL.md` §"Two things §5 cannot tell you".

So check 3 as written cannot pass on this template with any engine, and the
honest options are the owner's to pick, not this code's: make the check's
tolerance template-aware (Qwen needs ~44 where GLM needs 32), or accept a
documented SKIP for the qwen38 family the way check 2b already is. Nothing here
moves a gate's bar to make a number look better.

One engine-side option was considered and rejected: after generating, feed the
turn-closing tokens the next prompt is certain to contain (`<|im_end|>\n`) so
the checkpoint covers them. It buys 2 tokens of the 42, and it would make
`reused` come out at `prompt + completion + 2` — which is a MISMATCH against
`ledger_expect_reuse` for any family whose ledger is on. Two tokens is not
worth teaching this engine to disagree with the gateway's arithmetic.

## Decode step 2 (2026-09-23): `--hip-graph`, the token body as one graph a card

§L0-PERF-1 closed step 1 with the reading that fewer launches at the same
~7 µs floor buy nothing, and that the lever left is the ~335 dependent
dispatches a card a token. `--hip-graph 1` captures each card's share of a
decode token -- its layer range, plus `output_hc_*`, `lm_head` and the argmax
on the last card -- into a hipGraph once, and replays it. It fuses nothing and
reorders nothing: **the same kernels in the same order with the same
integers**, so the replay is bit-identical to the eager token by construction,
and the gate below checks that rather than trusting it. Off by default;
`FRANKEN_HIP_GRAPH=1` does the same for `--serve`. **Nothing here has been run
on a GPU.**

### What is parameterised, and how

A graph bakes its kernel arguments, so everything that moves per token had to
stop being an argument:

| per-token quantity | how the graph gets it |
|---|---|
| the position (RoPE in `qsa_qk_post`, the cell `kv_store_q8_0` writes, the block `idx_pool_chunk` pools, `n_kv`/`n_blocks`/`n_sel`/the tail/the identity test of the five QSA-row kernels) | **device-resident**: one `int` a card, read at the top of those 8 kernels when the launch passes its pointer (eager launches pass nullptr and take the argument as before). The graph's LAST node is `k_pos_advance` (`*pos += 1`), so the next replay reads the next position with no host write. The host uploads it (4 bytes, staging ring, stream-ordered) only when its shadow disagrees with `pos_`: after an eager chunk, a serving rollback, a reset. |
| the QSA-row grids (`idx_scan` ~n_blocks/8 WGs, `qsa_expand` n_kv/256, `attn_flash_split` n_sel/64) | sized for the **position class**'s largest cache: class = `pos / --hip-graph-bucket` (default 1024), bound = `min((class+1)*bucket, ctx)`. Every one of those kernels already returned or grid-strode past its own row's extent (the row-block guards of PREFILL.md section 11), so the idle workgroups do nothing; a scanned block's score is one wave's work whatever the grid, so the grid cannot move a bit. A new class re-captures and `hipGraphExecUpdate`s the existing executable in place (a fresh instantiate only when the topology changed). |
| the top-k skip (budget covers the cache -> identity, no radix select) | a class-level decision, as the row block already made it: the radix-select node is captured iff the class bound exceeds the budget, and the kernel returns for any row whose budget still covers its cache (`qsa_expand` wrote the identity for it). So classes 0-1 carry no radix node, class 2 (n_kv 2049-3072) does and it returns for n_kv <= 2051. |
| the residual bank (`res_hc[chunk & 1]` is a different pointer) | **two graphs a card**, one a bank. |
| the embedding and PLE rows, and their staging slot | uploaded to the SAME fixed buffers as before, through the staging ring, BEFORE card 0's graph -- the graph never sees a slot. |
| the routed-id log pointer (`--verbose` / `--routing`) | never in a graph: a capturing token runs eagerly. |

A backend `epoch` moves whenever a buffer a T = 1 kernel is handed is
reallocated (split-K partials and counters, attention partials, radix
candidates -- e.g. a later, larger prefill chunk growing `gemv_part_`); a graph
captured under another epoch would hold a freed pointer, so it is recaptured.
A reallocation DURING a capture aborts with a message.

### What the graph does not capture

- **The boundaries.** A graph is one card's stream only; nothing spans a
  device. The event record on the source, the wait and the
  `hipMemcpyPeerAsync` on the destination, the `drained_[bank]` record and the
  bank-free wait stay host-issued stream operations BETWEEN the cards' graph
  launches, exactly where they were -- `seg_end(src)` runs before
  `boundary_recv` records its event, `seg_begin(dst)` after the copy. A graph
  launch is stream-ordered like any op, so the pipeline's handshake is
  unchanged. Per token the host now issues ~15 calls (3 graph launches, 2
  uploads, the boundary ops, the timer events) against ~1 004 launches.
- **Taps, `--jitter`, `--sync-debug`, the routing capture, and every T > 1
  chunk**: those tokens run eagerly. So `--sync-debug` cannot debug a graph;
  for the first graph run use `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3`
  instead.
- **The per-op profile.** Inside a graph there are no per-op events; `--profile`
  charges a card's whole replay to one class, **`prof_graph_replay_us`**. The
  per-class/per-projection numbers need `--hip-graph 0`.
- **The first token of each (card, bank, class)** pays the capture (plus an
  update or instantiate); `--time` now runs two warm-ups (one a bank) so the
  counted tokens are all replays at short depth, and at depth the capture
  recurs once per bucket. `hip_graph_dev<i> nodes= captures= instantiates=
  updates= replays= capture_ms_total=` is printed after `--time`.

### The oracle, and why taps alone would not have been one

The recorder's taps are downloads in the middle of the body, so the recorded
token (the prompt's last) always runs eagerly. With `--chunk 1` the prompt's
first five tokens run through the graphs and the taps see the state they left
(K/V cells, pooled keys, GDN state, both conv windows) -- but not a graph's own
arithmetic. So `--greedy N` now also records **`greedy_ids`** (the continuation
plus the final argmax, compared EXACTLY -- a cosine over ids would pass one
wrong id in seventeen) and **`greedy_logits`** (all 248 320 logits of the last
step). Every greedy token runs through the graphs. Both keys read MISSING
against llama.cpp's dump, which does not carry them.

**ISA:** the device-resident position is an integer load at the top of 8
kernels. Compiled both ways (`hipcc -O3 --cuda-device-only -S`,
`~/bench/franken/hipgraph_isa/`): 55 kernels have an identical instruction
stream; the 8 touched ones have an identical FP-op multiset (VOPD dual ops
split and counted) -- no multiply-add changed between fused and split, the
thing `-ffp-contract=fast` decides by scheduling -- and an identical FP-op
sequence in 7 of them; `k_idx_pool_chunk` issues the same ops in a different
schedule order. The GPU gates (c) below are what settle it.

**CPU:** `franken_decode_cpu --layers 0-3 --no-head --chunk 1` against a dump of
the pre-change binary: 138 of 138 taps `cos=1.000000 maxabs=0`; with the head
and `--greedy 3`, `--chunk 6 --hip-graph 1` against `--chunk 1`:
`greedy_ids exact=1`, `greedy_logits maxabs=0`, STEP2 PASS (the CPU backend has
no graphs, so this checks the runner's rewiring and the new taps only).

### Card 0's slow small GEMVs (97 vs ~160 GB/s): a hypothesis and the run that decides it

What card 0 does that cards 1-2 do not: (1) it receives the token's two host
uploads (embedding + PLE, ~10 KB each, staged H2D on its stream -- BEFORE the
body, so they cannot slow a GEMV in the middle of it); (2) it holds the PLE
layer (one extra 20 480-row GEMV, not in the small set); (3) **its token is not
pre-queued.** Cards 1-2 sit on a boundary wait while the host enqueues their
whole body; card 0 starts on the first op and is then fed at the host's issue
rate (step 3b). The per-op profile charges event-to-event intervals, so on card
0 an op shorter than the host's per-op issue time (a launch plus, under
`--profile`, an event record) is charged the ISSUE time. The small set is
exactly the ops short enough for that: ~96 launches of 3-5 MB, ~20-40 µs of
kernel each. Two runs tell it apart from a card-intrinsic cause (clocks,
power state):

- **`--prequeue-gate 1`** (new, diagnostic): card 0's stream is held by a
  one-thread spin kernel on a host-mapped flag until the host has enqueued the
  whole token, then released -- card 0 becomes pre-queued like 1-2 (a 2 s wall
  clock bound means a full hardware queue cannot hang it; `prequeue_timed_out`
  says so if it happened). **If card 0's `prof_gemv_small_gbs` rises to ~160
  under the gate, the gap was host issue and `--hip-graph` removes it by
  construction.** The token time under the gate includes the issue time; read
  the per-class numbers, not the median.
- **`HIP_VISIBLE_DEVICES=1,0,2`** swaps which physical card is logical card 0.
  If the slowness follows the logical position it is the feed; if it follows
  the physical card it is the card (then `amd-smi metric -c` during the run).

With `--hip-graph 1 --profile`, `prof_graph_replay_us` per card is the third
reading: equal replay times on three cards with the same layer shapes (card 0
+1 PLE GEMV) would say the same thing from the other side.

### The GPU commands (NOT run here; rig lock, gateway stopped, cards empty)

```
cd ~/src/colibri-m1/tools/hot-expert/franken/decode
M=~/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
T="248044 785 10945 315 1495 374"
C="--model $M --tokens $T --devices 3 --layers 0-47"

# (a) eager, the reference of the pair
./franken_decode $C --ctx 512 --chunk 1 --greedy 16 --hip-graph 0 --dump ~/bench/franken/hg_off
# (b) the graphs against it -- every tap cos=1.000000 maxabs=0, greedy_ids exact=1,
#     greedy_logits maxabs=0, STEP2 PASS; first time under the serialisers
AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 \
./franken_decode $C --ctx 512 --chunk 1 --greedy 16 --hip-graph 1 --oracle ~/bench/franken/hg_off
./franken_decode $C --ctx 512 --chunk 1 --greedy 16 --hip-graph 1 --oracle ~/bench/franken/hg_off
# (c) and against the pre-change binary's dump (c0c6545, knobs off == defaults):
#     every tap maxabs=0 (the two greedy keys read MISSING there)
./franken_decode $C --ctx 512 --chunk 1 --greedy 16 --hip-graph 1 --oracle ~/bench/franken/dec1_off

# (d) short depth, A,B,B,A -- layers0_47_ms_median, issue_ms, hip_graph_dev<i>
for g in 0 1 1 0; do ./franken_decode $C --ctx 262144 --time 32 --hip-graph $g; done
# (e) 256k depth, A,B,B,A (26.06 -> 25.82 ms is the number to beat)
for g in 0 1 1 0; do ./franken_decode $C --ctx 262144 --chunk 256 --gemm-lds 1 \
    --time-prefill 262000 --time 32 --hip-graph $g; done
# (f) attribution and card 0
./franken_decode $C --ctx 262144 --time 32 --hip-graph 1 --profile     # prof_graph_replay_us
./franken_decode $C --ctx 262144 --time 32 --profile --prequeue-gate 0
./franken_decode $C --ctx 262144 --time 32 --profile --prequeue-gate 1
HIP_VISIBLE_DEVICES=1,0,2 ./franken_decode $C --ctx 262144 --time 32 --profile
```

## DeepSeek-V4: adaptive expert placement (2026-09-24, L5 step 4)

`--adapt 1` on the DeepSeek path learns the hot experts from the engine's own
routing (the router kernel counts them; the host keeps a decayed average) and
swaps experts host -> VRAM on a low-priority stream between tokens. The
qwen4exp path is unchanged: its 65 kernels are ISA-identical. **The ordering
proof lives in `DEEPSEEK4.md` section 12.** In short:
- evictions and installs are table-edit kernels on each card's main stream
  at a token boundary, after every reader of the old entry (the side-stream
  staging copies are joined within their own token);
- a slot's copy is fenced behind the eviction;
- an install waits on the copy's event on the device.

So no kernel sees a half-copied slot or a torn table entry, and since an
expert's bytes are identical everywhere, every tap stays bit-identical to
`--adapt 0`. That held on the CPU arm with the real routing: 173/173, 74 swaps,
`--adapt-verify` PASS. GPU gate: `ds4_gpu_gate.sh adapt`.
