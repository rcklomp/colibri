# P6b — per-slot KDA device state (spec and decision, Fable, 2026-09-07)

Item P6b of `PREFILL-ROADMAP-2026-09.md`; the roadmap reserves "the
per-slot device-state decision" for Fable. This is that decision, with the
numbers it rests on and the executable gate. Build: Opus, after P7 lands
(P7 touches the same sync hooks; §5 says what to change in it).

## 1. Decision

**Per-slot device buffers, one set per KV slot, allocated up front — before
the expert preload — and never swapped. All or nothing: if the pool cannot
be allocated for every slot, the engine keeps forcing the CPU recurrence,
loudly, as it does today.** Serve at `--kv-slots 4` with `COLI_KDA_GPU=2`.

## 2. The facts it rests on

- Today one state/window per layer lives on dev0 (`backend_vulkan.c:106`,
  `G.kda[VK_KDA_LAYERS]`); `coli_vk_kda_init` re-seeds it on every
  `session_open` (`:1943-1955`), so two live slots on the GPU path would
  share one recurrence. `slots_init` therefore forces `COLI_KDA_GPU=0` for
  `KV_SLOTS > 1` (`glm53.c:3898`). Measured cost of that (§P6 step 1 in the
  record, G12): **~10 % decode, ~5 % prefill** (132 vs 125 ms/token at 781
  tokens).
- Size per slot: 34 KDA layers × (64·128·128·4 B state = 4.19 MB + 3·8192·4·4 B
  window = 0.39 MB) = **156 MB**.
- dev0 is full: 23.2 GB of 24 GB in service (3.78 GB dense, 1 296
  heat-ranked experts × 14.2 MB, and the preload stops when free VRAM drops
  under the 3.0 GB reserve, `glm53.c` `vk_preload_tier`). A pool allocated
  *after* the preload would eat that reserve; allocated *before* it, it
  costs experts: **4 slots = 625 MB = 44 experts** (the coldest 44 of
  dev0's 1 296, 3.4 % of its tier); 8 slots = 88. Against a 10 % decode
  gain, 44 cold experts is the right trade; 88 is still fine but 4 slots
  is what Open WebUI needs now that its side tasks are off (P6).
- Reading device state back through the host-visible arena is a memcpy from
  write-combined memory at ~14 MB/s (p2c, 2026-09-06: 150 MB → 11 s). The
  arena has a `memtype_cached` type for buffers the CPU reads
  (`backend_vulkan.c:52`, `scratch_reserve_mt`). Any device→host sync — the
  P7 checkpoint capture at the GPU knob, the segment snapshot — must go
  through a `vkCmdCopyBuffer` into cached staging, not the WC memcpy.

## 3. Design

### 3.1 `c/backend_vulkan.c`

- `G.kda[layer]` keeps the per-layer weights (`conv`, `alog`, `dt`,
  `onorm`) once; state/window become `G.kda[layer].slot[s].{state, window,
  state_p, window_p}` for `s < G.kda_slots` (≤ `GLM53_MAX_SLOTS` = 16).
- `int coli_vk_kda_pool_init(int nslots, int heads, int k, int v, int
  kernel)`: allocates every slot's state+window for all KDA layers at
  `G.prio = 1.0` (never evict), returns the number of slots it managed;
  anything short of `nslots` frees what it took and returns 0. Prints
  `[VK] KDA slot pool: N × 156 MB`.
- `coli_vk_kda_init(layer, slot, …)`: re-seed that slot (the memcpy at
  `:1952`), weights uploaded once on the first call per layer as today.
- `coli_vk_kda_step(layer, slot, …)`, `coli_vk_kda_layer(layer, slot, …)`,
  `coli_vk_kda_upload(layer, slot, …)`: bind that slot's buffers in the
  descriptor writes (`:2087`, `:2211`) — the only change inside the chain.
- `coli_vk_kda_sync(layer, slot, state, window)`: record a copy of the two
  buffers into a cached staging scratch (`scratch_reserve_mt(&G.kda_stage,
  …, G.memtype_cached)`), submit, wait, memcpy out. ≥ 1 GB/s expected;
  the gate measures it (§4, step 5).
- `int coli_vk_kda_pool_slots(void)`: what the pool holds; 0 = none.

### 3.2 `c/glm53.c`

- The pool is sized from `KV_SLOTS` **in `model_load`'s Vulkan init, right
  after `coli_vk_init` and before `vk_preload_tier`** (`:3144-3156`), when
  `COLI_KDA_GPU > 0` and the model has KDA layers: `KV_SLOTS` is an env var
  the gateway sets, so it is known there; `slots_init` (`:3888`) runs too
  late for the VRAM budget. `slots_init` then forces `g_kda_gpu = 0` only
  when `coli_vk_kda_pool_slots() < g_n_slots`, with the existing message
  plus the pool size it found.
- `GSession` gets `int slot`; `session_open(m, cap, slot)` passes it to
  `coli_vk_kda_init`; every `coli_vk_kda_step/layer/sync/upload` call site
  (`:1225`, `:1308`, `:1367`, `:4439`, `:4451`, and P7's capture/restore)
  passes `s->slot`. The CLI (`--ids`, `--prompt`) and the segment adapter
  use slot 0.
- `slot_reset`/`session_close` leave the device buffers; the next
  `session_open` on that slot re-seeds zeros — exactly today's rule, per
  slot.

### 3.3 Numerics

Unchanged per slot: the same shader on the same slot-private state. The
oracle for "state has a new home" is `tworeq` across slots at the GPU knob
(the G12 lesson), not the teacher-forcing CLI, which never opens two
sessions.

## 4. Gate (executable; `p6b_gate.sh`, exit 1 oracle, 3 speed, 2 refusal)

1. `tworeq.py` with `TWOREQ_SLOTS=4` at `COLI_KDA_GPU=2`: IDENTICAL, **and
   `grep -c "forcing COLI_KDA_GPU=0"` on its stderr is 0**; again at `=0`:
   IDENTICAL.
2. `prefill_gate.sh <p7-served binary> <candidate>` with `MIN_SPEEDUP=0.97`
   (the gate runs one slot: the pool has one set, the numbers must be
   neutral).
3. Decode at four slots, `ttft_serve.py --engine … --kv-slots 4 --sizes 30
   --repeat 2 --gen 64`: the candidate's `decode_tps` within 1 % of the
   pristine at `--kv-slots 1` (GPU knob) and ≥ 8 % above the pristine at
   `--kv-slots 4` (CPU forced). Print all three rows; the roadmap's number is
   3.4 → 3.8 tok/s.
4. The live side-request case through the gateway after serving
   (`ttft_serve.py --url http://127.0.0.1:8081 --multiturn --side-request
   --system ~/bench/p6_system.txt`): turn 2 REUSE ≫ 0 with the server log
   showing no `forcing` line.
5. Sync speed at the GPU knob: P7's `CKPT store prefix=<len> <MB> MB in <ms>
   ms` line for the 5 400-token tool-block prefix under 1 000 ms (the WC
   path would print ~11 000).
6. The startup line `preload: N heat-ranked experts` shows N ≈ 1 296 − 44 at
   4 slots, and the `DONE … STAT` hit-rate over the gate's requests is
   within 0.5 points of the pristine's.

## 5. Interplay with P7 (in flight)

P7 captures a checkpoint with `coli_vk_kda_sync(layer, …)` per KDA layer
and restores with `coli_vk_kda_upload(layer, …)`; P6b adds the slot
argument at those two call sites and makes the sync fast. Until P6b lands,
P7 runs at the serving knob (`COLI_KDA_GPU=0`, host state), where neither
call fires. Land P6b as one commit series on `perf/p6b-kda-slots` after
`perf/p7-prefix-ckpt` merges.

## 6. Rejected

- **One device set, swapped on slot switch (LRU).** Saves 470 MB (33
  experts) for a 150 MB round trip per conversation switch (~0.1 s with
  staging, 11 s without) and a state-ownership protocol of the kind that
  produced the G12 cross-session leak. 33 cold experts are not worth it.
- **Mixed slots (GPU for the first k, CPU for the rest).** Numerics would
  depend on which slot a conversation hashed to, and `tworeq` across slots
  could not be identical by construction. All or nothing.
- **Allocating the pool after the preload.** It would come out of dev0's
  3 GB reserve, which the attention/scratch buffers and the KV cache need;
  the reserve is not slack.
