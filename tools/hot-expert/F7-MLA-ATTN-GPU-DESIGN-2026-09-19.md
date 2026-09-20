# F7 — batched MLA attention core on the GPU in prefill (design, 2026-09-19)

`FRANKEN-ENGINE-PLAN-2026-09-15.md` §8.3 item **F7**, re-based by record
§F2-LADDER. Engine `glm53`. Knob `GLM53_MLA_ATTN_GPU=1`, off by default,
prefill only (`tokens > 1`), CPU path unchanged and the fallback on any
Vulkan failure. Decode untouched.

## 1. What is being moved, and the shapes

Served config (`text_config`): hidden 4096, H = 64, L (`kv_lora`) 512,
QK (`qk_nope`) 256, V (`v_head`) 256, `qk_rope` 0, width = 2048/4 pools
expanded + tail = **2051**, 11 DSA layers, prefill chunk S = 512.

The `_tm2` window of `mla_layer` (glm53.c ~1935–2130), per row t, per layer:

| stage | MAC | note |
|---|---:|---|
| score `glm_lane_dots` | 2051×64×512 = 67.2 M | CPU, serial over t |
| softmax | 2051×64 | float max, `double` total |
| pool (mode 1, default) | 2051×64×512 = 67.2 M | per-head walk of the latent |
| `kvb_v` `mv_rows` ×64 | 16384×512 = 8.4 M | int4-g64 |
| `o` `mv` | 4096×16384 = 67.1 M | already GPU, **one submit per row** |

Measured (§F2-LADDER): 2.51 ms per row-layer, i.e. **1.29 s per 512-row
layer-chunk**, 27.62 ms/token, 34.9 % of the 18k prefill token.

## 2. Where the latents live, and the VRAM bill

`G.memtype` on dev0 is HOST_VISIBLE|DEVICE_LOCAL (ReBAR, "memtype 3" in the
startup line) — scratch allocated through `scratch_reserve` is **in VRAM**,
written by the CPU at ~12 GB/s (§F2a) and read by the GPU at VRAM bandwidth
with the MALL behind it. That is the right home for the latent.

**Rejected: a persistent per-layer device mirror** (the decode path's
`coli_vk_kv_ensure`/`coli_vk_kv_row`, which `colibri.c` uses and which is
explicitly skipped when DSA is on). It costs 11 layers × seen × L × 4 =
0.41 GB at 18k and 1.47 GB at 65 536 — and it is per sequence, so ×4 KV
slots it is 5.9 GB. dev0 has **1 447 MiB free while serving** (measured
2026-09-19 18:30 UTC, card0). It does not fit and it would need
invalidation logic for checkpoint restore and cancelled prefills.

**Chosen: one shared set of scratches, re-filled per (layer, chunk) call.**
Nothing is per-slot and nothing is per-layer, because the prefix is copied
in on every call. Rows are sub-batched at SB (`GLM53_MLA_ATTN_SB`, default
128) to bound the score scratch.

| buffer | memory | size formula | at 18 439 | at 65 536 |
|---|---|---|---:|---:|
| LAT latents `[seen][L]` | VRAM | seen·512·4 | 37.8 MB | 134.2 MB |
| Q absorbed `[SB][H][L]` | VRAM | SB·64·512·4 | 16.8 MB | 16.8 MB |
| SC scores `[SB][H][width]` | VRAM | SB·64·2051·4 | 67.2 MB | 67.2 MB |
| POOL `[SB][H][L]` | VRAM | SB·64·512·4 | 16.8 MB | 16.8 MB |
| SLOT `[SB][width]`+counts | VRAM | SB·2051·4 | 1.1 MB | 1.1 MB |
| CTX `[SB][H][V]` readback | host (memtype_cached) | SB·64·256·4 | 8.4 MB | 8.4 MB |
| **dev0 VRAM total** | | | **140 MB** | **236 MB** |

**The expert tier caps do not move** (1695/1695, dev0 1 248). 236 MB comes
out of dev0's existing ~1.4 GB of free VRAM; allocation failure falls back
to the CPU path per call, so a tighter box degrades instead of dying.
dev2/dev3 are expert-tier-only contexts in this backend and get nothing.

## 3. The shaders (new files, additive; nothing existing is touched)

`c/shaders/mla_attn_score.comp`, `mla_attn_softmax.comp`,
`mla_attn_pool.comp`, `mla_attn_vproj.comp`. Score layout `sc[s][h][u]`
(h-major) so the softmax reads coalesced.

1. **score** — for each row s, `C[u][h] = scale · Σ_d LAT[slot[s][u]][d]·Q[s][h][d]`.
   64×64 output tile per workgroup, 256 threads × (4 u × 4 h) registers,
   K-loop in 32-steps with a 64×32 LAT tile and a 64×32 Q tile in LDS
   (16 KB). Grid `ceil(2051/64)=33 × SB`.
2. **softmax** — one workgroup (256) per (s, h): max over u (the same set,
   and max is exact, so this part is bit-identical), `exp`, a **float
   workgroup tree sum**, then in-place divide. fp64 was the first choice and
   is not available: the engine passes no `pEnabledFeatures`, so
   `shaderFloat64` is off. A tree over ≤ 2 051 non-negative terms has a
   relative error bound of ≈ log2(2051)·eps = 6.6e-7, which is orders of
   magnitude under the score pass's own reordering error — it is not what
   the KL bar will be measuring. Grid `SB × H`.
3. **pool** — for each row s, `P[h][d] = Σ_u W[s][h][u]·LAT[slot[s][u]][d]`.
   64×64 tile, same register blocking, K-loop over the 2 051 slots in
   32-steps. Grid `(512/64=8) × SB`. The latent is read once per (row,
   d-tile), not once per head — the shape §P5b's `COLI_MLA_POOL=2` has.
4. **vproj** — `CTX[s][h][v] = Σ_d dequant(kvb_v[h·V+v][d])·P[s][h][d]`,
   int4-g64 (`fmt 4`), dequant lifted from `attention_absorb.comp` pass 5.
   Grid `SB × H`, 256 threads = one v each. Keeps the 33.5 MB `pooled`
   readback off the bus (CTX is 8.4 MB per sub-batch instead).

One command buffer per sub-batch: 4 dispatches + 3 memory barriers + one
fence. Four submits per 512-row layer-chunk.

**Stays on the CPU:** the indexer, the absorb (`kvb_kt`), all of `_tm0`,
the compaction of `selected` into `slot[s][i]`+`used[s]` (same order as
the CPU's `slot_at`), and the `o` projection — which becomes
`mv_rows_s(out, &l->o, context_all, S)`, one submit per layer-chunk
instead of 512. `mv_rows_s` is documented and already gated as per-row
bit-identical to S `mv()` calls, so that part adds no numerics risk.

## 4. Projected cost per 512-row layer-chunk (seen ≈ 14k, the 18k turn's mean)

| item | GFLOP / MB | assumed rate | ms |
|---|---:|---|---:|
| LAT copy | 28.7 MB | 12 GB/s (§F2a) | 2.4 |
| Q copy | 67.1 MB | 12 GB/s | 5.6 |
| SLOT copy | 4.2 MB | 12 GB/s | 0.4 |
| score | 68.8 GFLOP | 8 TFLOP/s | 8.6 |
| softmax | 268 MB ×3 + 67 M exp | — | 1.5 |
| pool | 68.8 GFLOP | 8 TFLOP/s | 8.6 |
| vproj | 8.6 GFLOP int4 | — | 2.0 |
| `o` batched | 68.7 GFLOP int4 | — | 8.0 |
| CTX readback + submits | 33.5 MB | 22 GB/s + CPU read | 6.0 |
| **total** | | | **43.1** |

8 TFLOP/s is 27 % of the 7900 XTX's 29.5 TFLOP/s fp32 non-dual-issue peak —
deliberately conservative for a hand-tiled kernel.

**1 290 → 43 ms per layer-chunk = 30×; 27.62 → 0.92 ms/token.** With a 4×
pessimism factor on every line, 3.7 ms/token. The 18k prefill token goes
79.16 → 55.2 ms (**1.43×**) at the projection, 79.16 → 59.5 ms (**1.33×**)
at F7's own ≤ 8 ms/token target. **Gate stays at ≥ 1.25× on the 18k
ladder-turn TTFT** as the plan sets it; the projection clears it with room,
and the microbenchmark is what decides whether the 8 TFLOP/s holds.

## 5. Numerics

Knob-off is bit-identical to `~/bench/glm53.f2` (`5c01246c`) — O1 proves it
with `teacher_forcing` plus a full `GLM53_LOGIT_DUMP_ALL` dump.

Knob-on changes summation order in score (K-tiled fma vs the CPU's
`MLA_MULADD` rounded-product chain in ascending d), in pool (K-tiled vs
ascending u), in the softmax total (float tree vs sequential `double`) and
in `exp` (RADV vs glibc `expf`). The max-subtraction is kept and the max is
over the same set, so that part is bit-identical. Judged by the KL bar against
knob-off, same binary, clamp on in both: **mean KL < 0.0284 AND top-1
≥ 99.0 %** over every position (`kl_compare.py`), on the deep ~6.3k packet
AND the shallow ~2k one (shallow exercises `used = seen`, the dense
regime below 2 052 tokens).

## 6. Decode

Untouched: the knob is checked only on the `tokens > 1` path. Decode
*could* use these kernels later, but at S = 1 it would pay the full
per-call latent copy (2.4 ms at 14k) for one row of work — it needs the
persistent per-layer device mirror rejected in §2, which is a different
item with a different VRAM answer.
