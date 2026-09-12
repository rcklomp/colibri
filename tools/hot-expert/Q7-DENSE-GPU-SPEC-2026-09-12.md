# Q7 spec: the dense BF16 stream onto dev0 (decided at the third Fable gate, 2026-09-12)

> Written after QP landed (`796faf6` … `00a7713`). The *arm* was decided by
> measurement in record §QP (b): int8 dense is dead in its weights-only,
> f32-activation best case (cos 0.980 / 0.982 against §G14's rejected
> 0.98964), so Q7-cpu and the `int8 fmt` GPU variant are both withdrawn and
> this is **Q7-gpu at BF16, summation-order numerics only**. The *tensor set*
> and its *VRAM price* were decided by §QP (d): the whole 6.8 GB dense set
> costs **4.06 ms/token** of warm-identical time (0.60 ms/token per GB, ~0.27
> ms per evicted expert, linear and monotonic), and §Q-ARB's condition for the
> gated-residual pair (6.8 GB within 10 ms of 5.4) is met by an order of
> magnitude (+1.16). Evidence is §Q-PROFILE, §Q-ARB, §Q1, §Q2, §QP and the
> 09-04 placement matrix of `ROME-3x7900XTX-2026-09-04.md`, plus the source as
> read today; nothing below is inferred from anything else. The Q6 arbitration
> that puts the LM head in this set at BF16 is record §QP `### Q6 arbitration`.

## Decision

**Build Q7-gpu now, in four measured steps, each behind one bit of a single
knob (`Q38_DENSE_GPU`, off by default): DeltaNet projections → QSA projections
→ the LM head → the gated-residual pair. Two submits per DeltaNet/QSA layer,
one for the head, and the gated-residual pair as ONE submit per site or not at
all. Do not wait for Q12. The router stays on the CPU.** Expected −42 to −59
ms/token on the 174 ms fresh-process token for the first three steps, −5 to
−10 more for the fourth; the VRAM price is measured, not bounded, and is under
10 % of the gain at every step.

Three judgement calls this spec had to make, each with its arithmetic below:

1. **The gated-residual pair does not record into the neighbouring layer's
   submit, and the 6.56 → 1.43 shrink of the sigmoid mix (§Q2) does not
   change that.** The mix is CPU work *between* the pair's `up` and the
   block's projections; its cost is paid on the CPU whether it is 6.56 or
   1.43, and it can only stop separating the two submits by moving onto the
   device — which needs three elementwise dispatches per site (rms, silu,
   mix) that §G12 measured at ~60–100 µs each inside an open command buffer,
   i.e. the same order as the round trips they would save. What *does* pay is
   the fold that adds **no dispatch**: chain `up` behind `down` with the
   activation applied while the shader stages its input. One submit per site
   instead of two: 97 round trips saved, ~5.8 ms/token, and the pair's net
   goes from ~−1 (not worth 1.28 GB) to ~−6 to −10.
2. **Q12 does not gate this.** QP(c) showed that ONE int8 tensor already
   fails the greedy leg; any Q12 subset large enough to matter (≥ 36
   tensors, ≥ 1 GB) is expected to fail harder, and the only tensor set where
   a surviving int8 CPU arm would beat the GPU BF16 arm on the token is the
   gated-residual pair, which is step 4 and last. Q12 may run in the gap
   before step 4 if rig time allows; it cannot change steps 1–3.
3. **The router never moves, for a third reason on top of the two in §Q-ARB.**
   48 round trips (≥ 2.9 ms) exceed its 1.7 ms of matmul; its output is
   consumed by a CPU top-k that has to be on the host before the expert
   groups are issued, so it chains into nothing; and a reassociation on the
   router flips *routing* near ties, not a logit wobble — the one place on
   this engine where order-only numerics are not harmless. Keep the routing
   decision on a bit-identical path.

## What is already settled (do not re-derive)

| fact | value | source |
|---|---|---|
| dense bytes per decode token | DeltaNet **4.16 GB** (115.4 MB/layer × 36: qkv 2560×10240, z 2560×6144, b/a 2560×48, out 6144×2560), QSA **1.24 GB** (103 MB/layer × 12: q 2560×12288, k, v, idx_qk, o 6144×2560), gated residual **1.28 GB** (13.2 MB/site × 97: down 10240×320, up 320×10240, inject 10240×4), router 0.13 GB, LM head **1.27 GB** (2560×248 320) | §Q-PROFILE bandwidth table |
| what those bytes cost on the CPU today | `dn-proj` **52.8**, `gr-down` 9.05 + `gr-up` 8.48 + `gr-lowsilu` 0.24 + `gr-inject` 0.68 = **18.45**, `lm-head` **15.8**, QSA projections **~16.4 (est.)** — the one figure with no sub-timer; `dense-matmul` 90.0 in total | §Q1, §Q2 step 0, §Q-PROFILE |
| the token today | **174.18 ms** fresh-process after Q2; warm-identical 5.655 tok/s; rotating 4.41 | §Q2 |
| GPU dense GEMV, real backend, this box | int8 2560→10240 **0.198 ms** incl. round trip (26 MB: ~190 GB/s compute-only), LM head int8 **1.58 ms** (636 MB: 403 GB/s), fp8-emul 2560→10240 0.342 ms; **no BF16 shader existed** | 09-04 placement matrix |
| submit + fence round trip | **53–66 µs** (tiny matmul); marginal submit **44 µs** (four tensors in one submit 0.247 vs four submits 0.378); `vk_fence_wait` spins 300 µs before blocking, so this is the spun figure | placement matrix; `backend_vulkan.c` |
| VRAM price of a dense reservation on dev0 | warm-identical **+1.77 / +2.91 / +4.06 ms/token** at 3.4 / 5.4 / 6.8 GB; **0.60 ms/token per GB**; ~0.26–0.29 ms per evicted expert; tier 14 673 → 13 966 / 13 586 / 13 260; placement 91.25 → 89.85 / 88.93 / 88.35 %; +6.7 / +11.1 / +13.9 CPU-served experts per token | §QP (d) |
| where the reservation goes | after the three devices exist and **before `q38vk_preload`** (`qwen38_core.h` ≈1298, where `coli_vk_ballast_gb` already sits); the hottest-first fill then sees a smaller dev0 and drops its coldest experts by itself | §QP (d) step 0 |
| the dispatch primitive | `coli_vk_matmul_multi` (`backend_vulkan.c:1048`): up to `VK_MM_MAX = 8` tensors against one input in one submit; **one level of `src` chaining** (a chained item reads a previous item's output from the device-side `G.y`, no round trip); `out = NULL` keeps an intermediate on the device; **S = 1 by construction** (`struct PC pc = {fmt, 1, …}`) | source |
| the shader | `qmatmul.comp`: one subgroup per output row, lanes stride the row's uint32 words, `subgroupAdd` tree reduction; x staged into `xsh[6144]` when `I ≤ 6144`, else read straight from the storage buffer; `fmt` 1/2/4/5/7/8 in use, **3, 6, 9 free** | source |
| the dense tensors on the host | private BF16 copies (`q38_weight_reserve(…,Q38_WEIGHT_BF16,…)`, `owns_data = 1`), **not** mmap'd shards; 6.68 + 1.27 GB of process memory | `qwen38_core.h:877`, `:316` |
| the numerics class of a device move on this engine | placement-only (all routed experts CPU vs GPU, 1 440 reassociated GEMVs/token): **cos 1.000000000, relL2 4.9e−7 long / 1.9e−6 last-token, greedy identical**; 09-04: max-abs 7.6e−6, same argmax | §QP R3 vs R1; 09-04 rows |
| the numerics class this item is NOT | §G12's 0.99992: GLSL `exp` vs libm `expf` in a *recurrent* state, 1.1 M transcendentals/token compounding across a conversation | §G12 "the drift is `exp`" |

## Tensor contract

**Format.** A new `fmt = 9` in `qmatmul.comp`, `upload_tensor`, `rowwords`
and `scale_floats`: two BF16 values per uint32 word, low half = even column
(the same packing order every other `fmt` uses), `rowWords = ceil(I / 2)`,
host row bytes `2·I`. Dequant is a shift — `uintBitsToFloat(w << 16)` for the
even lane, `uintBitsToFloat(w & 0xffff0000u)` for the odd — which is exactly
`bf16_to_f32` (QP check A proved the CPU side exhaustively; the GPU side is
the same bit operation and needs a one-line self-test in `rome_vkbench.c`,
not an argument). **No per-row scale**: add 9 to the shader's no-scale list
(`fmt == 5 || 4 || 7 || 8`), and give the tensor a one-float `sbuf` so the
descriptor set is unchanged. Rows of `I = 10240` (gr `down`) exceed the
6144-float staging array and take the unstaged path, which the shader already
supports; every other tensor here stages (`I ∈ {320, 2560, 6144}`).

**Placement and priority.** dev0 only. Upload with `coli_vk_tensor_ensure`
at the ballast point (before the preload), bracketed by
`coli_vk_alloc_priority(1.0f)` so an oversubscribed heap evicts experts and
never a dense tensor (the tier fills at 0.4). Print one line the record can
copy: `[qwen38] Q7 dense on dev0: <n> tensors, <x.xx> GB (deltanet a.aa,
qsa b.bb, head c.cc, gr d.dd)`, immediately before the existing preload line.

**Host copy: kept.** The knob-off path, the S > 1 prefill path (below) and
any `coli_vk_matmul_multi` failure return all read it, and this box has 247
GB against Qwen's 173 GiB of page cache — 6.7 GB twice is not the constraint
here. Freeing it is a v2 option only when prefill also moves.

**Scope of the dispatch.** Decode only, `S = 1`. Prefill (`S > 1`) keeps the
CPU kernels (`q38_matmul_bf16`, `q38_dense_matmul_multi`, and Q4's
four-accumulator variant if that lands first). TTFT is unchanged by this
item. A conversation therefore prefills on the CPU and decodes on dev0; the
two compute the same products in a different order, which is the oracle's
whole subject.

**Knob.** `Q38_DENSE_GPU` is a bitmask: 1 = DeltaNet, 2 = QSA, 4 = LM head,
8 = gated-residual pair. Unset or 0 = today's binary, bit-identical. The
bits are what make each step its own A/B and let the record carry a
per-set sub-timer, as the row asks. `Q38_DENSE_GPU` with `Q38_I8_DENSE` or
`Q38_I8_HEAD` set is refused at startup (the QP simulations attach to the
same `Q38Weight`s; a sim-carrying weight must never reach the device path).

**Accounting.** Time spent in a Q7 submit is added to `dense-matmul` (so the
counter keeps meaning "the dense stream, wherever it runs") **and** to a new
`vk-dense` sub-counter; `dn-proj`, `gr-down`, `gr-up`, `lm-head` wrap the
calls and keep working unchanged. Step 0 adds the one sub-timer that is
missing today: `qsa-proj` around the two projection calls in
`q38_attention`.

## Per-layer submit plan

The dependency chain inside a layer, decode, from the source:

```
gr_read(attn):  rms(CPU) → down ──┐→ silu(CPU) → up → mix(CPU) → inject(CPU sigmoid)
                                  └ inject (independent of down)
DeltaNet:       {qkv, z, b, a}(mixed) → conv/qknorm/recurrence/gnorm (CPU, Q1) → out
QSA:            {q, k, v, idx_qk}(mixed) → index + attention (CPU) → o
gr_apply(CPU) → gr_read(mlp) → router(CPU) → expert groups (async, all 3 devices) → gr_apply(CPU)
```

| site | submit | items | reads | returns to host | per token |
|---|---|---|---|---|---:|
| DeltaNet | A | `dn_qkv`, `dn_z`, `dn_b`, `dn_a` (all `src = −1`) | `mixed` (10 KB up) | 10240 + 6144 + 48 + 48 floats (66 KB) | 36 |
| DeltaNet | B | `dn_out` | `norm` (24 KB up) | 2560 floats | 36 |
| QSA | A | `q`, `k`, `v`, `idx_qk` | `mixed` | q/k/v/idx rows | 12 |
| QSA | B | `o` | `heads` (24 KB up) | 2560 floats | 12 |
| LM head | — | `lm_head` | final `mixed` | 248 320 floats (993 KB, cached readback) | 1 |
| gated residual | — | `down` (`src −1`, `out = NULL`), `inject` (`src −1`), `up` (`src = 0`, **`act = silu(x / C)`**) | `norm` (40 KB up) | `mix` 10240 floats + `inject` 4 floats | 97 |

**Submits per token: 97 for steps 1–3, 194 with step 4.** At 60 µs that is
5.8 ms and 11.6 ms of round trips — the item's own concern, written before
Q2, priced the pair at "97 × 2 round trips would eat two thirds of it"
(11.6 of ~17); the single-submit form halves that and is the only form worth
building.

**Why two submits per DeltaNet/QSA layer and not one.** Between A and B
sits the CPU recurrence (DeltaNet) or the CPU attention (QSA), whose input
is A's output and whose output is B's input. One submit per layer needs that
CPU stage on the device — a G12-class kernel, explicitly out of scope for
v1 in §Q-ARB — for a saving of 36 + 12 round trips ≈ 2.9 ms/token. The
row's "one submit per layer or two" range therefore resolves to **two**, and
the low end of its −35 to −50 is the one this plan claims.

**Why the pair does not fold into the neighbouring submit, in numbers.** The
fold needs `hyper` device-resident across the site and three elementwise
stages recorded between the GEMVs: the four-row `rms` (1.06 ms/token of CPU
today), the 320-wide `silu` (0.24), and the 10 240-wide sigmoid `mix` (1.43
after Q2). The CPU work removed is 2.7 ms/token. The dispatches added are 3
× 97 = 291 per token; §G12 stage 2b measured its chain — eight projections,
three elementwise stages, `ko` — at 0.93 ms/call with ~0.5 ms of matvec in
it, i.e. **~60–100 µs per latency-bound elementwise dispatch inside an open
command buffer**. 291 × 0.06–0.10 = **17–29 ms/token**, more than the pair's
whole CPU cost. So the fold loses even at zero round trips, and it loses at
every value of the mix — 6.56 or 1.43 — because the mix's size was never the
term that mattered. The activation-on-input fold is different in kind: it is
a flag on a dispatch that exists anyway (`xsh[i] = silu(x[i] / C)` while
staging `up`'s 320-float input), so it costs 320 GLSL `exp` per site and
no dispatch. That is the fold this spec builds.

**What the activation flag touches.** `struct PC` gains `int act; float
act_c;` (0 = none, 1 = silu(x / act_c)). `struct PC` is shared with every
pipeline in `backend_vulkan.c` and with glm53's shaders' push-constant
range, so step 4 is a shared-file change in the full CLAUDE.md sense: glm53
rebuilt, its `teacher_forcing` / `last_logits` / greedy continuation
identical, and the serving path re-accepted at the merge (§QP's precedent).
Steps 1–3 also touch shared files (`qmatmul.comp`, `upload_tensor`) and
carry the same obligation; the glm53 check is per step, not per item.

**Queue contention: none.** Q7's submits run in the attention/DeltaNet site
and the gated-residual sites; the expert groups run in the MoE site, and the
engine calls `take` before the next `gr_apply`. They are sequential within a
layer, so dev0's queue never holds a dense submit and an expert group at the
same time. `coli_vk_matmul_multi` clobbers `G.cmd_ready`/`G.bound_tensor`
like every other chain; qwen38 does not use the singular cached path on
dev0, so nothing is lost.

## VRAM ledger per device (predicted; the preload line is the check)

Budget per device 25.7 GB; `q38vk_expert_ensure` stops a device at
`budget − used < 1.5 GB`; an FP8 expert is 4.92 MB; the fill is hottest-first
dev0 → dev2 → dev3, so a smaller dev0 pushes its coldest residents down a
device and the tier's coldest fall off dev3.

| after step | dense on dev0 | dev0 experts (≈) | dev2 / dev3 (≈, unchanged) | tier total | evicted vs 14 673 | warm-identical price |
|---|---:|---:|---|---:|---:|---:|
| today | 0 | 4 920 | 4 880 / 4 880 | 14 673 | — | — |
| 1 (DeltaNet) | 4.16 GB | ~4 070 | same | ~13 830 | ~840 | +2.5 (rule: 0.60 × 4.16) |
| 2 (+ QSA) | 5.40 GB | ~3 820 | same | **13 586** | **1 087** | **+2.91 (measured)** |
| 3 (+ head) | 6.67 GB | ~3 560 | same | ~13 330 | ~1 340 | +4.0 (rule) — **measure with `Q38_VK_BALLAST_GB=6.7` once, one `rome_bench.sh` run, before step 3's A/B** |
| 4 (+ gr pair) | 7.95 GB | ~3 300 | same | ~13 070 | ~1 600 | +4.8 (rule; 6.8 GB measured at +4.06, the pair adds 1.28) |

dev0 stays the hottest device: every expert it loses is colder than every
expert it keeps. The step-0 ballast table in §QP (d) matched this arithmetic
to 0.2 % at 5.4 and 6.8 GB, so a preload count more than ~50 experts off the
prediction at any step is a sign the dense allocation did not land where it
was meant to (wrong device, wrong priority, or an arena that could not grow),
and is investigated before anything is timed. **The record gets, per step,
the printed dense line, the preload line, and the `[OPTIME] placement`
percentage**, as the row demands.

## In numbers — what each step should measure

Per-submit time = bytes / effective rate + one round trip. Two rates bracket
the unknown, because no BF16 `fmt` has ever run on this box: **200 GB/s**
(what the int8 proxy achieved compute-only at the 2560→10240 shape) and
**400 GB/s** (what it achieved on the LM head). Step 0 replaces the bracket
with the measured figure per shape before anything in the engine is built.

| set | CPU today | GPU at 400 / 200 GB/s (+ 60 µs per submit) | Δ | VRAM | net |
|---|---:|---:|---:|---:|---:|
| DeltaNet, 36 × (A 84.4 MB + B 31.5 MB) | 52.8 | 14.8 / 25.2 | −38.0 / −27.6 | +2.5 | **−35.5 / −25.1** |
| QSA, 12 × (A 71.5 MB + B 31.5 MB) | ~16.4 (est.) | 4.6 / 7.7 | −11.8 / −8.7 | +0.4 | **−11.4 / −8.3** |
| LM head, 1 × 1.27 GB | 15.8 | 3.3 / 6.4 | −12.5 / −9.4 | +0.8 | **−11.7 / −8.6** |
| **steps 1–3** | **85.0** | **22.7 / 39.3** | **−62.3 / −45.7** | **+3.7** | **−58.6 / −42.0** |
| gr pair, 97 × 13.2 MB, one submit | 18.45 | 9.0 / 12.3 | −9.5 / −6.2 | +1.2 | **−8.3 / −5.0** |
| gr pair, two submits (rejected form) | 18.45 | 14.8 / 18.1 | −3.7 / −0.4 | +1.2 | −2.5 / +0.8 |

Steps 1–3 take the 174.18 ms token to **~116–132 ms (7.6–8.6 tok/s
warm-identical)**; the rotating median carries its 56 ms of miss service on
top and moves less in percentage terms, exactly as §Q-ARB says. Nothing is
claimed for the eight idle CPU cores during a submit: a layer is a
sequential dependency, there is nothing to overlap (§G12's reasoning, and
§Q3's 54 % absorption is the warning against counting idle as free).

**Not moved, and why, one line each.** *Shared expert* (0.47 GB, 7.2 ms):
54 % of it is already absorbed in the GPU gap after Q3; moving it gains at
most the unabsorbed 3.4 ms less 48 round trips (2.9) — and it would sit in
dev0's queue in the MoE site, where the expert groups are. *PLE
projections* (~0.09 GB, ~1.1 ms, one site): one round trip against one
millisecond, ≤ 0.5 ms either way. *Router*: the decision above.

## Oracle, by name — and why the bar is NOT §G12's

This item changes **summation order and possibly FMA contraction** over
products that are exact in f32 (a bf16 value times an f32 activation). It
introduces **no transcendental on the device** in steps 1–3 and **no state on
the device** at any step. That is the class this engine has already measured
twice: the 09-04 rows (max-abs 7.6e−6, cosine 1.0, same argmax) and §QP's
R3-vs-R1 (1 440 reassociated GEMVs per token: relL2 **4.9e−7** over 1200
positions, **1.9e−6** on the short prompt's last position, greedy identical).
§G12's 0.99992 belongs to a different class — GLSL `exp` against libm inside
a recurrence — and is two to three orders of magnitude looser than a
reassociation ever measured here. Inheriting it would let a layout bug
(a truncated row, a mis-packed odd column, a wrong `rowWords`) pass as
"drift". So:

- **Knob off** (every step, every commit): last-token logits **byte-identical**
  to the pristine (`DUMP=`, `cmp` on the 993 280-byte row) and greedy text
  identical, on the 30-token prompt and the 1200-position one. This is the
  same claim §QP made and met; it is the first thing measured and the only
  part of a step that is unambiguously good.
- **Knob on, per bit and with all landed bits together**, against the
  knob-off binary, same frozen histogram copy, same tier:
  - greedy text **identical** over 128 tokens on both prompts;
  - `teacher_forcing` (`Q38_TF=1`, both sides) **identical** over the 30
    positions and over the 1200 positions;
  - last-token logits: **relL2 ≤ 1e−5** on both prompts, cosine printed to
    nine decimals reading `1.000000000`, max-abs reported (not gated). The
    calibration point is 1.9e−6 for 1 440 reassociated GEMVs per token; this
    item reassociates ~340 larger ones plus the head. 1e−5 is headroom of
    5×, not a licence: **a relL2 above it is a stop, not a tolerance
    discussion** — something other than reassociation is happening.
  - **Any `teacher_forcing` change at any position** is reported with the
    top-2 logit margin at that position (`Q38_TF_DUMP=` gives the row) and
    the step goes to arbitration; it is not rounded (§QP (c)'s rule, and the
    Q6 arbitration's stopping rule: one flip with a nonzero long-prompt rate
    is a real rate; one flip with 0 of 1200 is a near tie, and the margin
    decides).
- **Step 4's extra leg.** The activation flag puts 320 GLSL `silu` per site
  on the device — the first transcendental in this item. Same bar. If step 4
  misses the bar and steps 1–3 met it, build the two-submit form once as a
  *diagnosis* (silu on the CPU, everything else identical): if that form
  passes, the miss is the `exp`, the record says so, and the pair ships as
  two submits only if its net is still ≥ 4 ms/token — by the table above it
  will not be, and the pair is then out of scope with the reason recorded.
- **Session state.** `tworeq.py` is written for glm53 (`SNAP`, `EXE`
  hard-coded). Give it `TWOREQ_ARCH=qwen38` / `TWOREQ_SNAP` in step 1's
  commit (Sonnet, hours) and run three identical requests in one persistent
  engine, text byte-identical. Q7 keeps no per-conversation state on the
  device, so this is cheap insurance rather than G12's load-bearing check;
  it is still required, because §Q-ARB asked for it and because "the state a
  request leaves behind is what the next user turn pays for".
- **The other engine.** Every step touches a shared file. glm53 rebuilt at
  the candidate: `teacher_forcing`, `last_logits`, the 128-token greedy
  continuation and the `[MAP]`/hit counters identical (§QP's table is the
  template, including the note about the absent `greedy` prefix).

## Ordered build, with the gate per step

Every step: **step 0 first**, then the candidate, then the oracle, then the
`[OPTIME]` A/B, then `rome_bench.sh qwen38-vk` interleaved two-and-two with
its own pristine rows inside the campaign (never against another campaign's
— §Q1's rule). Gates are stated as **half of the step-0 bucket**, the
judgment floor §Q3's and §Q1's arbitrations set, and re-derived from step 0
before the candidate exists per the roadmap's "(est.)" rule. The pristine is
`hot-expert-tier` at branch time; if Q4 has merged by then, Q4's knob is in
the same state on both sides of every A/B and the step-0 buckets are taken
on that binary.

**Step 0 — the shader and the numbers, no engine change.**
`fmt = 9` in `qmatmul.comp` / `upload_tensor` / `rowwords` /
`scale_floats`; a `rome_vkbench.c` case that (a) checks the GPU BF16 GEMV
against `q38_matmul_bf16`'s output at each shape (relL2 ≤ 1e−6 on random
data, exact on data with ≤ 8 significant bits), (b) times one submit at the
five shapes this item dispatches — 4-tensor DeltaNet A (84.4 MB), 31.5 MB
`out`, 4-tensor QSA A (71.5 MB), the 1.27 GB head, and the chained
13.2 MB pair with the activation flag — distinct-tensor pools larger than
Infinity Cache, ten repeats, medians. Also the `qsa-proj` sub-timer in the
engine (bit-identical, timers off and on). **Output:** the effective GB/s
per shape, the per-submit ms, and the four gates below re-derived from
them, written into the record before step 1 is built. Stop condition: if
the measured per-submit times predict `dn-proj` above **26.4** (half of
52.8), stop and report — the shader is not fast enough and the fix is in
the kernel (wider loads, `uvec4` per lane), not in the engine.

**Step 1 — DeltaNet (`Q38_DENSE_GPU=1`).** The two submits above; A via
`coli_vk_matmul_multi` with four `src = −1` items, B with one. Oracle in
full. Gate: `[OPTIME] dn-proj` **≤ 26.4 ms/token** and within 1.25× of
step 0's prediction; `dn-conv`/`dn-qknorm`/`dn-recur`/`dn-gnorm` flat
within spread (the saving is work moved off the CPU, not work moved between
buckets); the token down by `Δdn-proj` less the ballast price within spread;
`dense-matmul` reported (expected 90 → ~52–63). The preload line within ~50
of 13 830. **The row's old "stop if `dense-matmul` is not under 55" is
replaced**: 55 implied `dn-proj ≤ 17.8`, which is the 400 GB/s end of the
bracket with nothing to spare — a threshold at the mechanism's ceiling, the
shape §Q1 and §Q2 taught this track to re-derive at step 0. Stop and
re-profile before step 2 if `dn-proj` is above half.

**Step 2 — QSA (`|= 2`).** Same shape. Gate: `qsa-proj` ≤ half its step-0
figure (≈ 8.2 if it measures ~16.4); `qsa-attn`/`qsa-index` flat; preload
within ~50 of 13 586 (§QP (d)'s measured 5.4 GB row, which this step
reproduces with real tensors instead of ballast — the two must agree, and
that agreement is the proof the ledger transfers).

**Step 3 — the LM head (`|= 4`).** One submit, `O = 248 320`, the
grid-stride path; the 993 KB readback from the cached memtype. Before the
A/B, one `rome_bench.sh` run at `Q38_VK_BALLAST_GB=6.7` to price 1.27 GB
more of eviction as a measurement (the linear rule predicts +0.76 over the
5.4 GB row). Gate: `lm-head` ≤ 7.9; token down by `Δlm-head` less the
measured price within spread; the head's argmax at the last position
identical on both prompts (it is, by the TF leg, but the head is the one
tensor whose reassociation lands directly on the logits, so say it
separately).

**Step 4 — the gated-residual pair (`|= 8`), only in the one-submit form.**
The `act` push constant; the chain `down` → `up` with `inject` alongside.
Step 0's measured per-submit figure for this shape × 97 must be **≤ 9.2
ms/token** (half of 18.45) or the step is not built and the reason is the
measurement. Gate: `gr-down + gr-up + gr-lowsilu + gr-inject` ≤ 9.2;
`gr-rms` and `gr-mix` flat (§Q2's neighbouring-bucket rule — the pair's
buffers `norm`, `low`, `mix` are read and written by those loops); token
down by the delta less +1.15 within spread; the extra oracle leg above.

**Merge.** Each step is its own commit with its table; the branch merges
when steps 1–3 have met their gates (step 4 may follow as its own
commit or be recorded as out of scope with its step-0 number). Then
`serve_candidate.sh` / `accept_live.sh` / `accept_ui.sh`, because glm53's
binary changed with the shared files — even though nothing in glm53's
default path did.

## Q12 sequencing — decided: after, not before

Q12 asks which subset of the 676 dense tensors survives int8. Three
numbers say it cannot change what this spec builds:

- QP(c): **one** int8 tensor (the head, g64, the tightest arm anyone has
  measured here) fails the greedy leg per-row and the short-TF leg g64. QP's
  own depth arithmetic is a random-walk: 676 tensors moved the logits 26×
  one tensor (0.198 / 7.7e−3), i.e. ~√676. A subset worth anything on the
  CPU is ≥ 36 tensors (one tensor type across the DeltaNet layers, ≥ 1 GB),
  predicted at ~6× one tensor's perturbation — squarely inside the class
  that already failed.
- Even where a subset survived, the CPU int8 arm beats the GPU BF16 arm
  only if 2.35× on the CPU (no VRAM, no submits) exceeds the GPU's ~3–4×
  less the VRAM price. For DeltaNet: CPU int8 ~22 ms against GPU 15–25 +
  2.5 — a coin flip at the pessimistic rate and a loss at the optimistic
  one, *and* it changes numerics where the GPU arm does not. For the head:
  CPU int8 ~6.7 ms against GPU BF16 ~3.3–6.4 + 0.8 — the same. Only the
  gated-residual pair (CPU int8 ~7.9 against GPU ~9–12 + 1.2) is a case
  where a surviving subset would flip the placement, and that is step 4.
- Q12 costs a Sonnet half-day for the filter knob plus ~25 min of rig time
  per subset, on a rig that Q4 is using now and steps 1–3 need next.

So: **Q7-gpu steps 1–3 are built now; Q12, if anyone runs it, runs in the
gap before step 4 and can only ever move the pair.** Its row in the roadmap
says so.

## Environment and configuration

- Build: `make -C c qwen38 qwen38-vk glm53 VK=1` (the shader is in the
  Makefile; `COLI_VK_SHADERS` must point at the repo's `c/shaders` for any
  copied binary).
- Run: `Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto`, 8 threads pinned
  (`OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close`), a **frozen copy
  of `.coli_usage` per run** (`COLI_USAGE=<copy>`; §RP1's confound, and
  `rome_bench.sh` does not freeze it for qwen38), Qwen asserted 100.0000 %
  resident over all 131 shards by `fincore` before every phase, rig lock
  held, gateway down, one engine at a time.
- Knobs: `Q38_DENSE_GPU=<mask>`; `COLI_TIMERS=1` for `[OPTIME]`; `Q38_TF=1`
  and `Q38_TF_DUMP=` for the oracle; `DUMP=` for the last-token row;
  `Q38_VK_BALLAST_GB` for the step-3 price; `VK_PROF=1` if a per-submit
  breakdown is wanted in-engine.
- Harness: `qp_probe.sh` is the pattern for the oracle runs (two prompts,
  the R1/R2 knob-off pair first, then per-bit); `rome_bench.sh qwen38-vk`
  for the serving A/B; `q1_chain.sh`/`q2_chain.sh` for the fresh-process
  `[OPTIME]` triples. Read `MEASURING.md` before the first benchmark.

## Effort and tier

**Opus**, 3–5 days, in five measured commits (step 0 + shader; DeltaNet;
QSA; head; pair or its refusal). This is a new shader format with a
cross-device numerics oracle, which is Opus's tier by CLAUDE.md. Sonnet may
take the knob plumbing, the `qsa-proj` sub-timer, the `tworeq.py`
parametrisation and the ballast run once the shader passes step 0. Fable is
not asked again unless a step misses its numeric leg by a margin the step-0
bracket did not contain, or a `teacher_forcing` position changes — and for
the second, the answer to "may we loosen the leg" should be no unless the
margin at that position is a demonstrable near tie.

## Measurement hygiene, inherited

Residency asserted before every run, not once per campaign; netdata
stopped; `top` not `ps`; A/B paired and alternating, twice; the histogram
copied per run; the rotating row read as a pair inside one campaign; the
warm-identical row is the one that resolves effects of this size (§QP (d):
the rotating row's own N=0 pair spanned 5 ms); one engine on the rig at a
time, and the rig lock before the gateway is stopped for any reason.
