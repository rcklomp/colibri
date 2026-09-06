# G12 spec: the KDA recurrence on the GPU (decided at the second Fable gate, 2026-09-05)

> This reverses `G4-KDA-SPEC-2026-09-04.md` §"Why not the shader", on the
> condition that spec itself set: *"revisit only after items 1–6 of the G3
> ranked list are done, if KDA is then the largest remaining bucket."* On
> 2026-09-05 both halves hold for the first time — items 1–6 (G11, G4, G8,
> G7, G10, G9) are done or deliberately closed, and KDA is the largest
> bucket at 53.3 ms/token (record §RP1-CORRECTION). Evidence is §RP1,
> §RP1-CORRECTION, §G11 and §G12 of `ROME-3x7900XTX-2026-09-04.md`; nothing
> below is inferred from anything else.

## Decision

**GPU. Move the KDA recurrence — decay, step, per-head norm — onto dev0 so
that one layer's eight projections, the recurrence, and `ko` record into
ONE command buffer and pay ONE submit+fence where the engine pays two today.
Ship behind `COLI_KDA_GPU=1`, off by default. Sized at −15 to −22 ms/token,
not the −35 the roadmap carried.**

Three things changed since the first gate, and one number in the current
roadmap is wrong:

1. **"From scratch" is no longer true.** `coli_vk_attn_qprep`
   (`backend_vulkan.c:1773`) already records *batched matmuls → an
   elementwise norm stage → a matmul* into one command buffer with
   `VkMemoryBarrier` compute barriers between stages, one `vkQueueSubmit`,
   one fence, and has a CPU-reference self-test (`run_qprep`). A KDA chain
   is the same shape with a different middle. Only the recurrence shader
   itself is new.
2. **"A second numerics path" is weaker than it looked.** The expert tier
   already *is* a GPU/CPU fork decided per-expert by residency, and the
   engine tolerates it. What is genuinely new is that KDA state is
   *recurrent* — divergence compounds across a conversation — and that is
   an oracle-design problem (below), not a veto.
3. **The recurrence is shader-shaped.** `coli_kda_step`
   (`delta_attention.h:85`): 64 independent heads, each a 128×128 fp32
   state read twice and written twice, plus a 24,576-lane elementwise conv.
   One workgroup per head; the state tile is 64 KB.
4. **The roadmap's "up to −35 ms" is wrong** and this spec does not inherit
   it. `proj` (21.7) + `ko` (13.3) is mostly real matvec compute that stays
   on the GPU either way. What actually goes away is one round trip and the
   CPU-side recurrence — see "in numbers".

## What changes

Shapes, from the record: 64 heads × head_dim 128 → `kda_proj` 8192; conv
kernel 4 (24,576 channels at a 12-byte shift = 3·8192 channels, kernel−1 = 3
floats). State 64·128·128·4 B = **4 MB/layer, 136 MB over 34 layers**;
window 3·8192·4·4 B = 393 KB/layer, 13.4 MB total. Decode is S=1 per call.

1. **State becomes GPU-resident for the session** (`backend_vulkan.c`).
   Two per-layer device buffers, `kda_state[34]` and `kda_window[34]`,
   allocated at session open from the CPU buffers (upload once), same
   `lnbuf` resident pattern as the q-prep norm weights. Per-layer small
   weights the recurrence needs — `alog[64]`, `dt[8192]`, `onorm[128]`,
   `conv_w[3·8192·4]`, and the scalar `gate_lb` — upload once the same way.
   **Nothing is read back per call.** 136 MB/token over PCIe would be
   ~5 ms — a third of the win — so the CPU spans go stale while the knob is
   on, and are re-synced only at the two points that read them (item 5).
2. **Three shaders, one of them real** (`c/shaders/`), all on the
   `rmsnorm.comp` template (std430 buffers, push constants, a 256-thread
   workgroup per row, `shared float red[256]` tree reduction):
   - `kda_decay.comp` — elementwise, S·8192 lanes: `decay[i] = gate_lb ·
     sigmoid(exp(alog[h]) · (decay_raw[i] + dt[i]))` with `alog` hoisted
     once per head exactly as G4 did on the CPU; and `beta[h] =
     sigmoid(beta_raw[h])`. Trivial.
   - `kda_step.comp` — **the recurrence**, one workgroup per head. First
     the conv for that head's 3·128 channels (shift the 4-tap window, dot,
     SiLU — the window lives on the GPU so the shift is a register rename,
     not a memmove); then q/k norms (tree-reduce); then the two passes over
     the 128×128 tile exactly as `coli_kda_step` orders them — pass one
     decays each row by `exp(decay[k])` and accumulates `memory[v]`, pass
     two applies the beta-scaled correction and reads `result[v]` out.
     The 64 KB tile fits RDNA3's 64 KB LDS exactly; if the compiler
     spills, stream it from VRAM — at ~900 GB/s the 16 MB/call is ~18 µs,
     which is not the cost. The cost is dispatch latency, and there are
     three dispatches.
   - `kda_headnorm.comp` — per-head RMS over D=128, times `onorm[d]`,
     times `sigmoid(gate[h·D+d])`. `rmsnorm.comp` with a 64-row grid and
     one extra multiply.
3. **One chain per layer per token** — `coli_vk_kda_layer()` in
   `backend_vulkan.c`, the `coli_vk_attn_qprep` body with a longer middle:
   the existing eight-projection recording from `coli_vk_matmul_multi`
   (its two-stage `src` chaining for `kfb←kfa`, `kgb←kga` stays as is) →
   barrier → `kda_decay` → barrier → `kda_step` → barrier →
   `kda_headnorm` → barrier → `ko` on the plain matmul pipeline using the
   spare `dset_qp3` set (so `VK_MM_MAX` need not grow). One submit, one
   fence, one readback: `out[hidden]`. It clobbers `G.cmd_ready` like every
   other chain does. Returns 0 → caller runs today's path unchanged.
4. **`kda_layer` (`glm53.c:1021`)** gains a branch at the top of the token
   loop: if `COLI_KDA_GPU` is on and `coli_vk_kda_layer()` returns 1, skip
   to the next token; otherwise fall through to the existing code, which
   is not touched. `COLI_KDA_CPU=1` (forces CPU projections) takes
   precedence and disables the chain.
5. **Two sync points, and only two** (`glm53.c`): the segment-state
   exporter at `glm53.c:3755`, which builds `ColiSegmentStateSpan`s from
   `st->kda_state`/`st->kda_window` (3768–3771) — before it builds them,
   if the knob is on, read all 34 layers back (one-off, ~150 MB, ~6 ms at
   PCIe rates, fine for an export); and the importer's inverse, upload
   after the spans are written. These are the only places that read the
   CPU copies; `session_open` (2676/2678) initialises them to zero and the
   upload takes that zero state.

## What does not change

`coli_kda_step` and the CPU path stay exactly as they are — they *are* the
reference. `delta_attention.h` is still included by `glm53.c` only. The
projection weights and `ko` stay in their existing resident tensors; the
chain binds them, it does not re-upload them. The MoE path, the three
GPUs' expert groups and the histogram are untouched. The KDA chain runs on
dev0 in the attention site and the expert groups run in the FFN site; they
are sequential per layer, so they do not contend for the queue.

## In numbers — why −15 to −22, not −35

Per call today (record, 100 % resident, quiet box): proj 0.638, decay
0.060, step 0.386, norm 0.101, `ko` 0.392 → 1.577 ms; ×34 = 53.6 ms/token.

- **One round trip removed.** RP1's `VK_PROF` puts a dense submit+fence at
  23.1 + 220.7 ≈ 244 µs. Two per call become one: −0.244 ms/call,
  **−8.3 ms/token**.
- **CPU recurrence moves off.** step + norm + decay = 0.547 ms/call,
  **−18.6 ms/token** of CPU time.
- **GPU cost of what moved.** Three latency-bound dispatches inside an
  already-open command buffer, ~50–100 µs each: **+5 to +10 ms/token**.
  The first gate's floor (0.15–0.2 ms/call) sits inside this range.
- **Net: −15 to −22 ms/token**, 9–14 % of 162.1. Nothing else on the road
  is above 14 ms. `proj` and `ko`'s matvec compute (~0.5 ms/call) is not
  counted as saving because it is not saved.

No credit is taken for the eight idle CPU cores during the chain: the
layer is a sequential dependency, there is nothing to overlap.

## Oracle, by name

This is **not bit-identical and does not claim to be**: GLSL `exp` differs
from libm `expf` by ULPs, and the state is recurrent, so the two paths
diverge and the divergence compounds. The oracle is therefore staged and
long-horizon:

- **Stage 1 — the shader alone.** A `run_kda` self-test in
  `backend_vulkan.c`'s test harness (the `run_qprep` pattern): random
  state, window, qkv, weights; 1000 consecutive steps through `kda_step`
  on the GPU and `coli_kda_step` on the CPU from the same initial state;
  after every step, **max relative error on `out` and on the state ≤ 1e-5,
  no NaN/Inf.** Fails → stop; it is the shader that is wrong, not the
  integration.
- **Stage 2 — integrated, knob on.** Against the pristine binary, 3 GPUs,
  same frozen histogram copy: **greedy text identical on every
  `rome_bench` prompt through 128 tokens** (the record's own oracle for
  glm53 is the `teacher_forcing` line); **last-logit cosine ≥ 0.99999 and
  argmax identical at tokens 8, 64 and 128**; and the ~700-token prompt
  (`~/bench/prompt_glm.txt` ×30) with `teacher_forcing` identical across
  all 1260 positions. Max-abs is reported, not gated — it is the drift
  curve at 8/64/128 that says whether the recurrence is stable.
- **Segment migration.** Export with the knob on, import into a fresh
  session with the knob off, continue generating: greedy text identical
  to a never-migrated run. This is what proves the sync points.

## Gate

`[OPTIME] kda` ms/call, fresh process, `COLI_TIMERS=1`, 128 greedy tokens,
paired and alternating knob-off/knob-on, twice each, model **100 %
resident on every shard** (`fincore`, asserted, not spot-checked), netdata
stopped: **1.577 → < 1.0 ms/call** is go. The internal timer is the
instrument; fresh-process tok/s scatters ±10 % and cannot resolve it.
Then `rome_bench.sh glm53` rotating median against G11's 2.67/2.71, two
runs, both reported — the serving number is the result, but at this
effect size it confirms "no regression", it does not adjudicate.

## Stop conditions

- Stage 1 relative error > 1e-5 at any step, or any NaN/Inf → **stop**,
  do not integrate.
- Stage 2 greedy text diverges on any prompt within 128 tokens → **stop**,
  knob stays off, record the token index and the logit cosine at
  divergence.
- Chain lands and `[OPTIME] kda` is **> 1.2 ms/call** → **stop**, ship
  off by default, record why (the likely cause is dispatch latency, and
  the fix is not more parallelism).
- dev0 cannot allocate the 150 MB alongside the tier (it should: ~22.1 GB
  of 24 GiB used at 1296/1695/1695 — derived from the preload figures, not
  re-measured; confirm with `coli_vk_mem_budget` before uploading) →
  reduce `COLI_VK_EXPERTS` on dev0 by the deficit and note it.

## Effort and tier

**Opus.** 2–4 days, in three measured commits (shader + self-test; chain +
knob + sync points; measurement + record). This is a new kernel with a
non-identical numerics oracle, which is Opus's tier by CLAUDE.md, and the
first gate's reason for saying no — the second numerics path — is exactly
the part that needs judgement rather than typing. Sonnet may take the
sync points and the knob plumbing once the shader passes Stage 1. Fable
does not execute this; it is asked again only if Stage 2 diverges and the
question becomes whether a looser oracle is acceptable — the answer to
that should be no.

## Measurement hygiene, inherited from today

Assert 100 % residency across all 62 shards; stop netdata first and read
`top`, not `ps`, for what the box is doing; A/B paired and alternating,
twice; `COLI_USAGE_PATH` on a *copy* of the histogram; one engine on the
rig at a time. RP1's headline was 1.86× wrong on one bucket for skipping
the first of these.
