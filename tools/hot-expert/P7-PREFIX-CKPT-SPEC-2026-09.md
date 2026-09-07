# P7 — prefix checkpoints for `glm53`, and a gateway-side prefix pin (spec, 2026-09-07)

Design pass (Fable) for the Opus build. Item P7 of
`PREFILL-ROADMAP-2026-09.md`. Everything here is either measured on rome on
2026-09-07 or read from the code named by file and line. The gate is a
script (§6); the item is done when it exits 0 and its tables are in the
commit body.

## 0. The measurement that fixed the scope

The roadmap row for P7 said "checkpoint the system+tools prefix so a *new*
conversation starts warm" and left one question open: is Open WebUI's
system prompt byte-stable across turns once memory context is on? Measured
first, through Open WebUI 0.11.0's own backend (`/api/chat/completions`
with `features.memory=true`, token minted inside the container, preset
capability `memory` switched on for the run and restored after), gateway
`--kv-slots 8`, served binary 4dd110f2… (P6):

| run | memories in the account | turn 1 | turn 2 | REUSE (turn 2) | slot 1 → 2 |
|---|---|---:|---:|---:|---|
| A | 1 (`context`, `setup/openwebui`) | 127 tok, 18.7 s ttft | 177 tok, **2.1 s** | **162 / 177** | 5 → 5 |
| B | 4 (three temporary `context` rows added and deleted through the API) | 176 tok, 20.6 s | 195 tok, **21.6 s** | **0 / 195** | 3 → 6 |

Why B breaks (`open_webui/utils/memory.py:290-406`, `misc.py:580`):
`add_memory_context` builds the block from a vector search (`k=8`) over the
**last seven user messages**, so its ranking changes every turn — the query
endpoint returned `[travel, cooking, work, setup]` for turn 1 and
`[cooking, travel, work, setup]` for turn 2 — and appends it to the **end of
the leading system message** (`add_or_update_system_message(..., append=True)`).
That message is inside both the engine's cached prefix and the gateway's
slot hash (`conversation_cache_slot`, `openai_server.py:2174`), so turn 2
re-prefills everything and lands on another slot. With one memory the
block is trivially stable (run A): the owner's account today is run A, any
account that uses memory is run B.

Decision: **P7 is a checkpoint AND a gateway-side prefix pin.** Neither
alone reaches the target ("turn 1 of a new conversation with the tool
block: seconds; every later turn: seconds"): the checkpoint without the pin
leaves turn 2+ re-prefilling the history whenever the block moves; the pin
without the checkpoint leaves the 5 400-token tool block on every new
conversation's turn 1 (876 s measured on 2026-09-07 with 24 tools).

## 1. Pieces

- **E** — engine (`c/glm53.c`): prefix checkpoints, a port of the
  DeepSeek-V4 pattern (`c/deepseek_v4.c:12470-12760` — `V4PrefixCkpt`,
  `v4_ckpt_restore`, `v4_ckpt_plan`, `v4_ckpt_store`, the disk slots — and
  its use at `12838-12990`, the prefill split at `ckpt_at`). The state to
  copy is exactly the span list the segment adapter already writes
  (`glm53_segment_spans`, `c/glm53.c:4461`).
- **G** — gateway (`c/openai_server.py`): the 8th SUBMIT field (prefix
  hint) for `glm53`, and the pin knob that keeps a per-turn context block
  byte-identical for the life of a conversation.
- **H** — harness: `ttft_serve.py --prefix-ckpt`, `--pin-block`, an
  engine-side first-token logit dump for the serve-path oracle,
  `p7_gate.sh`, and the chain script.

Tier: Opus for E and G (state with a new home: it needs `tworeq` at both
KDA knobs); H is Sonnet-shaped but lands in the same series.

## 2. Engine (E)

### 2.1 What a checkpoint is

```c
typedef struct {
    int *ids; int len;              /* the prefix, exactly the tokens fed */
    float **latent, **ikeys, **igates;   /* per layer, DSA layers only: len rows each */
    float **kda_state, **kda_window;     /* per layer, KDA layers only: whole */
    uint64_t used; int kind;        /* LRU clock; 0 = prefix (plan/hint), 1 = prompt end */
} PrefixCkpt;
```

Sizes on this model (config.json: 45 layers, 11 DSA (`is_full`), 34 KDA;
`kv_lora` 512, `index_hd` 128; KDA 64 heads × 128, `conv_k` 4):
DSA rows 33.0 KB/token (the `cache:` line), KDA 34 × (4.19 MB state +
0.39 MB window) = 156 MB. A 5 400-token tool-block prefix is **334 MB**; the
default 4 slots are ≤ 1.4 GB. `free -g` on 2026-09-07 shows 186 GB
available with the model resident; the gate's `majflt` column must stay ~0
(P9's counter) so this cannot silently evict model pages.

Knobs (all read once, like `slots_init`):

| env | default | meaning |
|---|---|---|
| `GLM53_PREFIX_CKPT` | 1 | 0 disables every path below (the pristine behaviour) |
| `GLM53_PREFIX_CKPT_SLOTS` | 4 | in-memory slots, max 8 |
| `GLM53_PREFIX_CKPT_MIN` | 128 | shortest prefix worth a capture (tokens) |
| `GLM53_PREFIX_CKPT_DISK` | 1 | persist kind-0 captures under `COLI_CKPT_DIR` (default `<SNAP>/.coli_ckpt`); 0 off |
| `GLM53_PREFIX_CKPT_END` | 0 | also capture at prompt end after every prefill (DSV4 kind 1); off: a full copy per request is not free here |
| `GLM53_LOGIT_DUMP` | unset | dir: write the first generated position's logits of every request as `req_<id>.f32` (vocab floats) — the serve-path oracle |

### 2.2 Where it hooks in (`serve_one`, `c/glm53.c:4088-4136`)

Today: `cached = session->filled`; `shared = cached` iff the new prompt
agrees on every cached position and is longer; else `slot_reset` +
`session_open`. Insert, in this order:

1. **Slot reuse first** (unchanged). It always covers at least as much as
   any checkpoint could.
2. **Checkpoint restore** when `shared == 0` and no image is pending:
   the longest checkpoint whose `ids` are a *strict* prefix of `sequence`
   (`len < total`, `len >= GLM53_PREFIX_CKPT_MIN`). Restore = fresh
   `session_open(m, room)`, memcpy each span (`len` rows for the DSA
   arrays, whole for KDA), `session->filled = len`, `slot_remember(slot,
   ids, len)`, `shared = len`. On the GPU-recurrence path (`st->kda_gpu`,
   only reachable at `KV_SLOTS=1`) follow the restore with
   `coli_vk_kda_upload(i, st->kda_state, st->kda_window)` per KDA layer,
   exactly as `glm53_kda_sync_in` does (`c/glm53.c:4445`). Lazy disk load
   before the search (once per process).
3. **Plan a capture** when nothing was reused: `ckpt_at = ckpt_plan(sequence,
   total)` — the DSV4 rule verbatim: the longest common prefix with the
   *previous fresh prompt* (`prev_ids`), never the whole prompt, tail ≥ 8
   tokens, ≥ MIN, not already stored. Then the **hint** (§3.1): if
   `ckpt_at == 0` and the SUBMIT carried `prefix_bytes` with `0 <
   prefix_bytes < plen`: tokenize `payload[0..prefix_bytes)` and accept
   its count `pn` only if `pn >= MIN`, `pn > shared`, `pn <= total - 8`, the
   ids equal `sequence[0..pn)`, and no slot already holds them. A hint that
   does not land on a token boundary is simply ignored (the plan finds the
   real boundary on the next fresh prompt).
4. **Prefill in two calls** when `ckpt_at > shared`:
   `forward_prefill(m, session, sequence + shared, ckpt_at - shared, ...)`,
   free those logits, **sync** (`coli_vk_kda_sync` per layer with
   `st->kda_gpu`, as `glm53_kda_sync_out` at `:4433`), `ckpt_store(session,
   sequence, ckpt_at, 0)`, then `forward_prefill(...)` for the rest. The
   split changes the 128-token chunk boundaries of the tail; P2/P3/P4 are
   per-row batch-independent, so this is expected bit-identical — the
   gate says whether it is (§6, step 2).
5. Vision: an announced image cancels restore and capture for that request
   (same rule as today's reuse, `:4118`).

`REUSE <id> <reused> <prompt_tokens>` keeps its meaning (`reused` includes a
checkpoint restore); add, behind `GLM53_VERBOSE`, `CKPT hit prefix=<len>
slot=<i>` / `CKPT store prefix=<len> <MB> MB` / `CKPT disk load|write
prefix=<len>` so the server log and `owui_report.sh` can tell a slot hit
from a checkpoint hit.

### 2.3 Eviction and disk

Victim order as DSV4: empty slot, else LRU kind-1, else LRU overall — a
prefix capture is the last to go. Disk: `<dir>/ckpt_<fingerprint>_<n>.bin`,
header `{magic "G53CKPT1", fingerprint, len, kind, payload_bytes}`, then
`ids`, then the spans in `glm53_segment_spans` order; `fingerprint` = FNV-1a
of (`n_layers`, `kv_lora`, `index_hd`, `kda_heads`, `kda_hd`, `conv_k`,
`glm53_dense_bits()`, the `is_full` bitmap, the SNAP path). A mismatch
deletes the file. Write synchronously after a kind-0 store (334 MB, ~0.3 s
into the page cache; once per distinct prefix). Load lazily on the first
SUBMIT. This is what makes every gate chain's gateway restart warm.

### 2.4 SUBMIT parsing (`:4041`)

`sscanf(header, "SUBMIT %llu %d %d %d %f %f %d %d", ..., &xlen,
&prefix_bytes)`: 6 fields as today; 8 fields → use `prefix_bytes` only when
`xlen == 0` (the gateway never sends an extra payload to glm53; if it ever
does, the frame would have to be consumed first). Older binaries keep
parsing 6 and ignore the rest — the gateway change is safe against the
served binary.

### 2.5 Must not change

The CLI path (`--ids`, `--prompt`, the teacher-forcing oracle), the segment
adapter, `forward_span`, decode. `GLM53_PREFIX_CKPT=0` must be the pristine
code path (the gate runs `prefill_gate.sh` with it).

## 3. Gateway (G)

### 3.1 Prefix hint for glm53 (`openai_server.py:2879-2896`)

Extend the `ARCH == "deepseek_v4"` branch to glm53. The stable boundary is
**the byte offset where the per-conversation part begins**: the start of
the pinned block when §3.2 found one (it sits at the end of the leading
system message, so everything before it — `[gMASK]<sop>`, the tool block,
the base system text — is shared by every conversation), else the first
`<|user|>`. Field format as DSV4: `" 0 <prefix_bytes>"`. No hint when the
offset is 0.

### 3.2 The pin (`COLI_PREFIX_PIN`, default 0; `~/start_glm53.sh` sets 1)

`pin_context_blocks(messages) -> messages` runs at the top of
`chat_completion` (`:3872`), on `body["messages"]` in place, so the render
and `conversation_cache_slot` both see the pinned version:

1. Only when `messages[0]` is a system message containing
   `<TAG>…</TAG>` for a TAG in `COLI_PREFIX_PIN_TAGS` (default
   `memory_context`). Split it into `base` (block removed, stripped) and
   `block`.
2. `key = sha1(base + "\0" + content of the first user message)`.
3. If `key` is in the pin cache: replace `block` with the cached one; else
   cache this block. LRU, 64 entries, process lifetime (the KV slots are
   process lifetime too).
4. Re-render the system message as `base + "\n\n" + pinned block` (the
   shape Open WebUI produced on turn 1) and return the byte offset of the
   block for §3.1.

Effect: every turn of a conversation carries turn 1's memory block, so the
prompt is byte-stable, the slot hash is stable, and the engine's slot
reuse covers the whole history. The per-turn retrieval Open WebUI intended
is reduced to per-conversation retrieval — that is the trade, it is behind
a knob, and the owner's client is known. A new conversation whose first
user message equals an old one's inherits that pin until restart;
document it, do not special-case it. Under `COLI_REQ_LOG=1` print
`[pin] key=<8 hex> <hit|new> block=<bytes>` per request.

Sorting the block's items instead of pinning was considered and rejected:
it fixes the order but not membership (`[Memory Neighborhood]` depends on
path hints in the query; `[Relevant Context]` is top-8 of N; the 2 000-char
limits truncate by rank).

## 4. Harness (H)

`tools/hot-expert/ttft_serve.py` (engine mode through `openai_server.Engine`,
HTTP mode against the live gateway; the pattern is `--multiturn`):

- `--prefix-ckpt`: (1) fresh conversation 1 `[S, A]`, `--gen 48` → REUSE 0,
  `t1`; (2) fresh conversation 2 `[S, B]` on another slot (`--kv-slots ≥ 2`)
  → `t2`, `r2`; (3) engine mode only: kill and respawn the engine, `[S, C]`
  → `r3` (disk persistence). Print `prefix-ckpt: len(S)≈…, turn 2 REUSE
  r2/…, t1/t2 = …×, after restart REUSE r3`. PASS iff `r2 ≥ len(S) − 32`,
  `t2 < 0.25·t1`, and (engine mode) `r3 ≥ len(S) − 32`. `S` for the gate is
  the real Open WebUI shape: `--tools ~/bench/owui_tools.json` (the 24-tool
  dump, copied from `/tmp/owui_tools.json`) plus `--system
  ~/bench/p6_system.txt`; `len(S)` is read from the engine (`prompt_tokens`
  of a `[S]`-only probe is not possible — use the REUSE line of turn 2 and
  the harness's own token estimate, and print both).
- `--prefix-ckpt --oracle`: serve `[S, B]` a second time on an engine
  spawned with `GLM53_PREFIX_CKPT=0`, `--gen 128`, greedy: the text must be
  byte-identical, and the two `GLM53_LOGIT_DUMP` vectors must have cosine
  ≥ 1 − 1e-4 and equal argmax (gate (b)'s bound). Print max|diff|.
- `--pin-block`: turn 1 `[system S+"\n\n<memory_context>X</memory_context>",
  A]`, turn 2 `[system S+block Y, A, reply, B]` with `pin_context_blocks`
  applied in the harness's `render()` (engine mode) or by the live gateway
  (HTTP mode). PASS iff turn 2's REUSE ≥ `prompt_tokens(turn 1) + gen(turn 1)
  − 1` and its ttft < 0.25 × turn 1's.

`tools/hot-expert/p7_gate.sh <pristine> <candidate>` (exit 1 oracle, 3
speed, 2 refusal), in order:

1. `GLM53_PREFIX_CKPT=0 MIN_SPEEDUP=0.97 prefill_gate.sh` — P7 is off on
   the CLI path and must be neutral on the serve rows; 0.97 because the
   claim is "no regression" and the rows repeat within ±1 %. Its tables go
   in the commit body.
2. `ttft_serve.py --engine <cand> --prefix-ckpt --oracle --tools --system
   --kv-slots 4` → PASS, with the logit numbers.
3. `tworeq.py` with `TWOREQ_SLOTS=4` at `COLI_KDA_GPU=0` and `=2` →
   IDENTICAL (a checkpoint is per-conversation state with a new home).
4. `ttft_serve.py --engine <cand> --pin-block --kv-slots 4` → PASS.
5. `ttft_serve.py --engine <cand> --sizes 300 --repeat 2` with checkpoints
   ON: the second run of an identical prompt must not capture or restore
   (REUSE 0 both, ttft within 3 % of step 1's candidate row).

`~/bench/p7_chain.sh` = the `p6c_chain.sh` pattern: stop gateway → merge
`p0-sync` → build → `p7_gate.sh ~/bench/glm53.p6base c/glm53` → on rc 0
patch `~/start_glm53.sh` (`export COLI_PREFIX_PIN=1`, `export
GLM53_PREFIX_CKPT=1` explicit, comment with the date) and restart → live
check `ttft_serve.py --url http://127.0.0.1:8081 --prefix-ckpt --tools
--system` and `--pin-block` → `CKPT hit` lines in `~/glm53_server.log` →
else revert the start script and restart the previous binary.

## 5. Expected effect (to be measured, in the commit body)

| case | today (measured) | after P7 |
|---|---:|---:|
| turn 1, new conversation, 24 tools + memory (4 053 tok) | 876 s | restore 334 MB (~0.3 s) + the user turn (~50 tok × 0.2 s) ≈ **10 s** |
| turn 2, memory on, ≥ 2 memories | full re-prefill (21.6 s at 195 tok; minutes with tools) | slot reuse: **~2 s** (run A's number) |
| first request after a gateway restart | full prefill of the prefix | disk restore: seconds |
| regenerate `[S, A]` | full prefill (182.6 s at 1 292 tok) | restore S, prefill A |

## 6. Risks, stated

- The hint may not land on a token boundary ("\n\n<memory_context>" vs
  `<|user|>`); the engine verifies and the plan covers it from the second
  fresh prompt on. Print which one fired.
- The split prefill may not be bit-identical (a chunked KDA scan would
  associate differently); the oracle in §4 bounds it exactly as gate (b)
  does, and the number goes in the commit body either way.
- 1.4 GB of host RAM at 4 slots; the `majflt` column is the guard.
- The pin makes memory context per-conversation. Knob, documented.
- Open WebUI's base system prompt may carry the date: one checkpoint per
  day per prefix; 4 slots cover it, the LRU handles the rest.
