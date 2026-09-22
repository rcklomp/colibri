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
