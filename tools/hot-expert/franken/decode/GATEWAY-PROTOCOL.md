# Gateway <-> engine line protocol

What `c/openai_server.py` requires of a child engine process, read out of the
code that actually enforces it (not out of any prior description of it).
Every claim below carries a `file:line`. Where the code leaves something
unstated, that is said explicitly rather than filled in by assumption.

Today the only family that speaks this protocol in production is `glm53`
(`c/glm53.c`, SERVE mode). `c/qwen38.c` and `c/qwen36.c` speak a related but
NOT identical dialect through a shared parser, `c/serve_codec.h` (ACCEPT
frame, `CONTEXT_EXCEEDED <used> <limit>` on the wire as
`prompt_tokens=N requested=M capacity=C`, ID as a **string** not a `%llu`).
This document specs the `glm53` dialect, because that is the binary in
service and the one `tools/hot-expert/FRANKEN-ENGINE-DESIGN-2026-09-22.md`
targets replacing. Where qwen38/qwen36 differ in a way a new engine should
prefer, it is called out as "the newer dialect".

`tools/hot-expert/franken/decode/franken_decode.cpp` was, when this document
was written (L0 step 4, first half of 2026-09-22), a `--tokens <id> [<id> ...]`
CLI oracle/measurement harness with **no tokenizer, no chat template, no SERVE
loop, no SUBMIT/CANCEL/STOP handling at all**. **That changed the same day:
`franken_decode --serve` implements what follows, in
`tools/hot-expert/franken/decode/franken_serve.cpp`, and §7 at the end of this
document maps every requirement here onto the code that answers it plus the
CPU conformance run that exercised it.** The body below is unchanged: it
describes what `openai_server.py` and `glm53.c` do, which is the same
whoever is behind the pipe.

## 1. Process contract

### argv

`Engine.__init__` launches the child as exactly `[str(executable),
str(resolved_cap)]` — one positional integer, nothing else
(`c/openai_server.py:3664-3667`). `resolved_cap` comes from `cap_for_arch`
(`c/openai_server.py:3482-3521`): an explicit `--cap` passes through
verbatim (0 included); with no explicit cap, a `glm5_next`/`glm5_next_text`
model (family id `glm53`) gets the sentinel `0` (platform-auto), everything
else gets the legacy `8`. `glm53.c`'s `main()` reads this as a bare numeric
argv token, "capacity per layer", `> 0` overrides `g_cap_override`
(`c/glm53.c:6900-6906`); `0` or absent means "you decide" (RAM-derived).
This positional argument is **not** the KV context length — that is
`GLM53_MAXT` below.

The executable path itself: with no `--engine`/`COLI_ENGINE` override, the
launcher resolves it via `default_engine()` — `<directory containing
openai_server.py>/<family.engine_artifact>` (`c/openai_server.py:34-43`,
`glm53` for the `glm53` family per `c/family_registry.py:926`). A new engine
that wants to stand in for `glm53` either replaces that binary or is wired
in through `--engine`/`COLI_ENGINE`; nothing in the wire protocol itself
names the binary.

### Environment the engine must honour

Set unconditionally by `Engine.__init__` (`c/openai_server.py:3658-3667`):

| var | value | meaning |
|---|---|---|
| `SNAP` | `str(model)` | model directory. `glm53.c` requires `tokenizer.json` at `<SNAP>/tokenizer.json` (`c/glm53.c:6913,6919-6921`) and refuses to start without `SERVE` *and* `SNAP` both set (`c/glm53.c:6912-6914`). |
| `SERVE` | `"1"` | switches `main()` from CLI mode into the SERVE loop (`c/glm53.c:6912`). |
| `SERVE_BATCH` | `"1"` | **always** set by the gateway alongside `SERVE` (`c/openai_server.py:3658-3659`) — the gateway never runs batch=0. It selects the stop-token policy: batch mode arms only the true end-of-text stops; non-batch (`coli chat` interactive use, not the gateway) also arms every special token as a stop (`c/glm53.c:6288-6300,6922-6923`). A new engine only needs the batch=1 behaviour to serve the gateway. |
| `NGEN` | `str(max_tokens)` | **`glm53.c` never reads this var** (confirmed: no `getenv("NGEN")` anywhere in the file; only `c/colibri.c` and `c/deepseek_v4.c` use it as a CLI-mode default cap). The gateway sets it out of habit across all engine families; `glm53` gets its per-request cap exclusively from the SUBMIT header's `max_tokens` field (§2). A new engine can ignore `NGEN` entirely as long as it honours the per-request field. |
| `KV_SLOTS` | `str(kv_slots)` | number of independent KV slots. `glm53.c` reads it twice — once in `model_load` (sizes the KDA device pool before the expert preload) and again in `slots_init` (`c/glm53.c:1143-1153,5615-5648`); the comment there is explicit that the two reads must agree or the pool is short and the engine aborts. Capped at `GLM53_MAX_SLOTS = 16` (`c/glm53.c:1143`), matching `family.limits.max_kv_slots` for `glm53` (`c/family_registry.py:950-955`). |

Read directly by `glm53.c`, not set by the gateway, operator-facing:

- `GLM53_MAXT` — per-slot context length in tokens (`g_slot_context`,
  `c/glm53.c:5644-5646`), default 8192, floor 64. This is the real context
  cap (`family.limits.context_env = "GLM53_MAXT"`,
  `c/family_registry.py:955`), not the argv cap.
- `COLI_REQ_LOG` — **gateway-side only.** Read in `c/openai_server.py` at
  `2420,2636,2722,3878` to print the `[req]` accounting line (§3); `glm53.c`
  has no knowledge of it.
- Prefix-checkpoint knobs, all optional and all `glm53`-specific (§5):
  `GLM53_PREFIX_CKPT` (on by default, `c/glm53.c:5717-5724`),
  `GLM53_PREFIX_CKPT_SLOTS` (default 4, cap 8, `c/glm53.c:5726-5735`),
  `GLM53_PREFIX_CKPT_MIN` (default 128 tokens, floor 8,
  `c/glm53.c:5737-5745`), `GLM53_CKPT_VALUE_EVICT` (on by default,
  `c/glm53.c:5769-5776`), `GLM53_PREFIX_CKPT_DISK` (on by default,
  `c/glm53.c:5795-5802`), `GLM53_PREFIX_CKPT_END` (**off** by default,
  `c/glm53.c:5807-5814`), `COLI_CKPT_DIR` (defaults to
  `<SNAP>/.coli_ckpt`, `c/glm53.c:5926-5931`). None of these affect the
  wire protocol's correctness, only its prefix-reuse performance; a new
  engine may implement none of them and still be conformant (§6).
- `GLM53_VERBOSE` — stderr diagnostics only (`REUSE`, `CKPT plan/store/hit`,
  `CANCEL ... at prefill pos=`/`at decode tok=` lines), never read by the
  Python side.

`tune_child_env` (`c/openai_server.py:3524-3555`) applies OMP/speculation
defaults, but **only when `arch == "deepseek_v4"`**; for `glm53` (or any
other family) it is a no-op that returns `env` unchanged. A new engine
under a different family id gets nothing from it unless it opts in there.

### stdio framing

`subprocess.Popen(..., stdin=subprocess.PIPE, stdout=subprocess.PIPE,
bufsize=0)` (`c/openai_server.py:3664-3667`) — fully unbuffered pipes, text
is written/read as raw bytes (`.encode()`/`.decode("utf-8","replace")`
throughout). `glm53.c` matches on its side: `setvbuf(stdin, NULL, _IONBF,
0)` and `coli_serve_binary_mode()` before the loop starts
(`c/glm53.c:6834-6835`), and every `serve_line`/`serve_data` write ends in
an explicit `fflush(stdout)` (`c/glm53.c:6307-6320`). **A new engine must
flush stdout after every frame it writes** — nothing else guarantees the
gateway sees it in time; libc's own default stdout buffering (block-buffered
when stdout is a pipe) would otherwise stall every reply.

### What must happen before `/v1/models` answers

`serve()` binds and activates the TCP listener **before** constructing the
`Engine` (`c/openai_server.py:5425-5429`, comment: "Bind before starting the
... engine. A stale/occupied port must fail in milliseconds..."), but
`server.serve_forever()` — the call that actually dequeues connections — is
not reached until *after* `Engine.__init__` returns
(`c/openai_server.py:5441-5446`). So: the port is bound, but no HTTP request
of any kind, `/v1/models` included, gets a response until the engine has
completed its boot handshake, which is exactly two things, **in order**:

1. **The READY sentinel**, byte string `\x01\x01READY\x01\x01\n`
   (`c/openai_server.py:45`, emitted by `glm53.c:6837`). `read_engine_turn`
   (`c/openai_server.py:3448-3463`) reads stdout byte-by-byte, discarding
   everything up to and including the first occurrence of this exact byte
   sequence — so arbitrary startup chatter on stdout before it is silently
   swallowed as long as it never accidentally contains the sentinel bytes.
   (Startup logging should go to stderr regardless.)
2. **One `STAT` line immediately after**, read via a plain `readline()`
   (`c/openai_server.py:3464` inside `read_engine_turn`, called again as
   `Engine.__init__`'s own follow-up read at `c/openai_server.py:3690`).
   Must split into `>= 5` whitespace-separated fields with `fields[0] ==
   "STAT"` or `Engine.__init__` raises `RuntimeError("invalid engine
   status: ...")` and the whole server fails to start
   (`c/openai_server.py:3465-3466`, matching the boot-time use inside
   `read_engine_turn` itself). `glm53.c` emits
   `serve_line("STAT 0 0.00 0.0 %.1f\n", rss_gb())` (`c/glm53.c:6838`) —
   `STAT` + exactly 4 numbers, the minimum that passes. Its fields are the
   normal DONE-STAT shape (§3) with zero completion tokens; the gateway
   parses but discards this specific line (the call site does not capture
   `read_engine_turn`'s return value).

After that, `Engine.__init__` spawns the stdout dispatcher thread
(`c/openai_server.py:3691-3693`) and returns; `serve()` proceeds to
`server.serve_forever()`. Anything the engine writes after the boot STAT
(e.g. `glm53.c`'s `EMAP` grid, `c/glm53.c:6841`) is consumed by the
dispatcher's normal frame loop (§3), not by the boot handshake — the
comment at `c/glm53.c:6839-6840` is explicit that EMAP must come *after*
READY+STAT for this reason.

**A new engine's minimum boot sequence: write the READY sentinel, then a
line `STAT 0 0.00 0.0 <rss_gb>\n`, flush both, then enter the SUBMIT loop.**

### Shutdown

`Engine.close()` (`c/openai_server.py:4103-4126`): marks itself closed,
fails every pending request with a synthetic error, then if the process is
still alive sends `SIGTERM` (`process.terminate()`), waits up to 5 s,
escalates to `SIGKILL` (`process.kill()`) and waits up to 5 s more, silently
giving up after that (comment: "Teardown is best-effort: never raise from
here"). A new engine does not need a graceful SUBMIT-drain path on SIGTERM;
it is not given one today — `glm53.c` installs no SIGTERM handler for this
path. Plain-EOF shutdown (closing stdin) also works: `serve_read_req`
returns 0 on EOF and `serve_loop` exits its `for(;;)` cleanly
(`c/glm53.c:6400-6414` `fgets` returning NULL, `c/glm53.c:6844`).

## 2. Request frames (gateway -> engine, on stdin)

### SUBMIT

Gateway writer: `c/openai_server.py:3929-3932,3947-3948`. Format (6 or 8
space-separated fields, glm53 never sends the 7-field grammar/audio
variant — see below):

```
SUBMIT <request_id> <cache_slot> <len> <max_tokens> <temperature> <top_p> [<xlen> [<prefix_bytes>]]\n
<len bytes of UTF-8 prompt text><xlen bytes of extension, if any>\n
```

Engine parser: `c/glm53.c:6355-6382` (`serve_read_body`), via `sscanf(header,
"SUBMIT %llu %d %d %d %f %f %d %d", &q->id, &q->slot, &q->plen,
&q->max_tokens, &q->temp, &q->top_p, &xlen, &hint_bytes)`
(`c/glm53.c:6363-6365`); `fields < 6` is `BAD_FRAME`
(`c/glm53.c:6366-6369`); the payload is then read as exactly `q->plen` raw
bytes plus the trailing `\n` (`c/glm53.c:6372-6381`).

Field by field:

- **`request_id`** — `%llu` on the `glm53` wire (an unsigned 64-bit decimal
  the gateway assigns itself, monotonically, starting at 1:
  `self.next_request_id`, `c/openai_server.py:3679,3904-3905`). Every reply
  frame for this request echoes it verbatim. **Note the newer dialect
  differs here**: `c/serve_codec.h` types the id as a string (`char
  id[COLI_SERVE_ID_CAP]`), and `qwen38.c`/`qwen36.c` print it with `%s`
  (`c/qwen38.c:1561`, `c/qwen36.c:2725`) — today the gateway always sends a
  plain decimal integer either way, so both parse it fine, but a new engine
  should not assume the id is numeric if it ever wants to be reused for
  those families too.
- **`cache_slot`** — an integer in `[0, kv_slots)`, chosen entirely by the
  **gateway**, not negotiated with the engine. For chat requests with the
  conversation ledger enabled (`ledger_enabled()`,
  `c/openai_server.py:2706`, on by default for `glm53`), the slot is
  assigned once per conversation and kept for its life
  (`ledger_pick_slot`, `c/openai_server.py:2860-2874`: first free slot, LRU
  eviction of another conversation's slot only when none is free). With the
  ledger off, `conversation_cache_slot` hashes the leading
  system+first-user messages to a slot (`c/openai_server.py:2284-2312`).
  The engine's only contract for this field: `KVSlot` state at that index is
  **per-slot, retained across requests**, reused when (and only when) the
  new prompt's tokens agree with what the slot already holds, position for
  position (§5). The engine validates `0 <= slot < g_n_slots`; anything
  else is `ERROR <id> BAD_REQUEST` (`c/glm53.c:6478-6481`).
- **`len`** — byte length (not token count) of the prompt payload that
  immediately follows the header line, UTF-8 (`c/openai_server.py:3860`,
  NUL bytes rejected client-side at `3861-3862`). Bounds-checked engine-side
  to `[0, 1<<24]` (`c/glm53.c:6371`).
- **`max_tokens`** — the per-request completion budget, already clamped by
  the gateway to the server's `--max-tokens` cap (`generation_options`,
  `c/openai_server.py:3434-3438`: values above the cap are silently clamped
  DOWN to it rather than rejected, specifically so OpenAI-client SDKs that
  send a large default `max_tokens` still work). `glm53.c` treats `<= 0` as
  "no explicit budget" and substitutes 256 (`const int budget = q->max_tokens
  > 0 ? q->max_tokens : 256`, `c/glm53.c:6510`) — a new engine should match
  that fallback or the "5. a reply may finish" acceptance check (§5,
  `accept_live.sh`) can behave unexpectedly for any caller that sends 0.
- **`temperature`, `top_p`** — `%.8g`-formatted floats
  (`c/openai_server.py:3930`), validated gateway-side to `0 <= temperature
  <= 2` and `0 < top_p <= 1` (`c/openai_server.py:3439-3444`). Sampling
  itself is entirely the **engine's** responsibility — the gateway never
  samples; it only forwards these two numbers and decodes whatever text
  bytes the engine emits. `q->topp = q->top_p > 0.0f ? q->top_p : 1.0f`
  (`c/glm53.c:6476`) is the engine-side floor.
- **`xlen`** (7th field, optional) — byte length of a same-line **extension
  payload** appended after the prompt payload, before the trailing `\n`.
  Two mutually exclusive uses depending on family
  (`c/openai_server.py:3866-3871,3907`): grammar-constraint bytes for
  families with `capabilities.grammar_payload` true, or DMel audio bytes
  for `inkling`. **`glm53`'s `FamilyCapabilities` has `grammar_payload =
  False`** (`c/family_registry.py:960`: `FamilyCapabilities(True, False,
  False, True)` = tools, grammar, audio, thinking), and the gateway raises
  a 400 before ever sending a grammar SUBMIT to it
  (`c/openai_server.py:4649-4652`, comment: "sibling engines speak the
  6-field SUBMIT header only; sending the grammar payload extension would
  desync its stdin framing"). **In practice the gateway never sends `glm53`
  a non-zero `xlen`.** `glm53.c`'s parser accepts the field syntactically
  (for forward compatibility) but only acts on the 8th field when `xlen ==
  0` (`c/glm53.c:6370`: `if (fields == 8 && xlen == 0 && hint_bytes > 0)
  q->prefix_bytes = hint_bytes;`) — i.e. `glm53` supports the 6-field and
  the 6+2-field (hint, no extension) forms, never the extension form. A new
  `glm53`-family engine can ignore `xlen`/extension payloads entirely.
- **`prefix_bytes`** (8th field, only sent when `xlen == 0`) — **a hint, not
  a command.** The byte offset (into the *prompt* payload) where the
  gateway believes the stable shared prefix ends — for `glm53`,
  `glm53_prefix_cut(prompt)` (`c/openai_server.py:3925-3928`), which
  prefers the pinned Open-WebUI context block's start, falling back to the
  first `<|user|>` marker. The engine is free to ignore it. `glm53.c`'s use
  (`ckpt_hint`, `c/glm53.c:6215-6229`): re-tokenize the payload's first
  `prefix_bytes` bytes, accept the hint as a checkpoint boundary only if the
  resulting token count is `>= ckpt_min_tokens()` (default 128), strictly
  greater than what the KV slot already has in flight (`shared`), at most
  `total - 8`, and — critically — **its token ids are memcmp-identical to
  the corresponding prefix of the fully re-tokenized prompt**
  (`c/glm53.c:6223-6225`). A hint that does not land on a real token
  boundary, or is wrong, is silently dropped and the engine falls back to
  its own longest-common-prefix plan (`ckpt_plan`, over the previous fresh
  prompt) — **never trust the byte offset without re-verifying it
  token-for-token**, per the comment at `c/glm53.c:6210-6214`.

### IMAGE

`COLI_SERVE_COMMAND_IMAGE` header (`c/serve_codec.h:24-27`), gateway writer
`c/openai_server.py:3941-3946`:

```
IMAGE <request_id> <bytes> <grid_h> <grid_w>\n
<bytes of raw float32 patch data>\n
```

Announced, under the same `write_lock`, **immediately before** the SUBMIT
frame it belongs to (`c/openai_server.py:3934-3948`) so no other request can
interleave between the two. `glm53.c` buffers it in a single-slot
`g_pending` (one image in flight at a time; a second IMAGE before its
SUBMIT arrives discards the first, `c/glm53.c:6259-6265,6340-6351`) and
consumes it inside the matching `serve_one()` by request id
(`c/glm53.c:6541-6556`) — an image announcement forces `shared = 0` (no
prefix reuse for that turn) because the vision tower's embeddings are tied
to exact token positions (`c/glm53.c:6533-6552`).

**Not needed for a text-only engine.** A new text engine can reject IMAGE
frames outright (as any pure-text family does today per the comment at
`c/serve_codec.h:24-27`: "un motore di solo testo che non lo gestisce lo
vede come un comando che non sa trattare e lo rifiuta, che e' la risposta
giusta"). Since `glm53`'s own `FamilyCapabilities` for the model family
Franken targets is text-only, this frame kind is out of scope entirely.

### CANCEL

`CANCEL <request_id>\n` (`c/openai_server.py:4019,4041,...`). Sent by the
gateway when its client disconnects, polled both while idle between frames
(`c/openai_server.py:3989-4021`) and after each DATA/TOOL frame
(`c/openai_server.py:4028-4042,4045-4056`) — **never sent before the engine
has ACCEPTed/started the request** (comment at `c/openai_server.py:4007-4015`
explains why: `glm53` has no dequeue-time ack, so the gateway effectively
treats the first DATA or DONE as the ack; see `_accept`,
`c/openai_server.py:3978-3987`, and note `glm53` never sends `ACCEPT`, so
`on_accept` only ever fires there). Two engine-side reactions, and the
gateway must handle both:

1. **In flight** (matches the request currently generating): a drain
   function, `serve_cancel_pending` (`c/glm53.c:6421-6471`), is polled
   between decode steps and at the boundary of every prefill chunk (see the
   long comment block at `c/glm53.c:5288-5348` for exactly when) — it never
   blocks waiting for one. On seeing `CANCEL <matching id>` it sets
   `cancel.cancelled = 1`; generation stops at the next checked point
   (`c/glm53.c:6665,6679`), the engine still emits **DONE** for the request
   (with the partial counts) followed by **`ERROR <id> CANCELLED`**
   (`c/glm53.c:6747-6766` — DONE first, then the ERROR ack, in that
   specific order; see the comment there for why: the gateway's dispatcher
   `pending.pop()`s on ERROR, so a DONE that followed it would be dropped
   and the `[req]` accounting line for that request would be lost).
   `Engine.generate`'s reader treats a DONE that arrives after it sent
   CANCEL as the terminal event and raises `ClientCancelled`
   (`c/openai_server.py:4059-4067`); it treats a subsequent bare `ERROR ...
   CANCELLED` (no preceding DONE) the same way (`c/openai_server.py:4095-4099`).
2. **Not the in-flight request** (the request already finished, or was
   never dequeued): `ERROR <id> NOT_FOUND\n`
   (`c/glm53.c:6439,6856`). `NOT_FOUND` is the one ERROR message the gateway
   treats as **transient, not terminal** — it does not pop the pending
   entry (`c/openai_server.py:3842-3847`, `transient = (message ==
   "NOT_FOUND")`) because it can mean the SUBMIT for that id had not been
   dequeued yet when the CANCEL raced ahead of it in the same pipe. The
   gateway retries the CANCEL (up to 20 times, 0.25 s apart,
   `c/openai_server.py:4076-4094`) rather than giving up on the first
   `NOT_FOUND`.

While a request is generating, the engine also queues any SUBMIT frames
that arrive mid-turn instead of dropping or misreading them —
`serve_cancel_pending` reads a full frame off stdin and buffers it in a
ring (`g_queue`, capacity `GLM53_MAX_SLOTS`,
`c/glm53.c:6388-6390,6445-6462`) precisely so a CANCEL for one of *those*
still-queued requests, arriving later in the same pipe, is not missed
either. **A new engine must replicate this queuing** if it wants to support
`KV_SLOTS > 1` under this same single-stdin-pipe design — the gateway
serialises all of a slot pool's requests down one pipe and relies on the
engine to read ahead rather than block.

### STOP

`STOP <request_id>\n` (`c/openai_server.py:4033,4050`). Sent when the
gateway's own stop-sequence matcher (`stop_filter`) declares the turn over
client-side, so the engine should end generation but the reply is a normal
successful completion (`finish_reason="stop"`), not a cancellation.
**`glm53.c` does not special-case STOP inside an in-flight turn at all** —
`serve_cancel_pending`'s drain explicitly ignores it (`c/glm53.c:6442-6444`,
"STOP non porta corpo e serve_loop lo ignora gia' ... scartarlo qui e' lo
stesso comportamento"), and the idle-loop handler for a *stray* STOP (no
turn in flight) is likewise absent from `serve_loop`'s verb dispatch
(`c/glm53.c:6845-6860` handles `SUBMIT`, `IMAGE`, `BAD_FRAME`, `CANCEL` —
no `STOP` branch, falling through to the "unrecognised verbs are ignored"
rule at `c/glm53.c:6858-6860`). **This means STOP is currently a no-op on
`glm53`**: the gateway relies entirely on its own `stop_filter` to stop
*consuming* tokens and simply keeps reading (and discarding) whatever the
engine still generates until its own DONE/budget ends the turn, rather than
on the engine actually curtailing generation early. A new engine is free to
either match this (ignore STOP, let the gateway's client-side filter do the
work) or implement it as an early, ungenerous CANCEL-like stop that still
produces a normal DONE — either is compatible with the gateway as written,
since the gateway never blocks on a STOP being acknowledged.

## 3. Reply frames (engine -> gateway, on stdout)

All parsed by `Engine._dispatch_stdout` (`c/openai_server.py:3730-3849`), one
dedicated thread reading `readline()` in a loop for as long as the process
lives; an unrecognised line kills the dispatcher and fails every pending
request (`c/openai_server.py:3848-3853` — the `else: raise RuntimeError(...)`
branch, deliberate: see the comment at `c/glm53.c:6706-6711`, "a server
ignores lines it doesn't know" was rejected on purpose "so a motore non
puo' parlare a un server che non lo capisce").

- **`DATA <id> <n>\n<n bytes>\n`** — one generated token's UTF-8 bytes
  (`c/glm53.c:6314-6320` writer, `c/openai_server.py:3740-3758` reader). `n`
  is bounds-checked to `[0, 65536]`
  (`c/openai_server.py:3750-3751`); the payload is read as exactly `n`
  bytes followed by a mandatory `\n` terminator
  (`c/openai_server.py:3752-3754`, `RuntimeError("invalid engine DATA
  terminator")` if it isn't there). Routed to the request's queue as
  `("data", bytes)`; the caller side runs it through an incremental UTF-8
  decoder so a multi-byte codepoint split across two DATA frames still
  decodes correctly (`decoder = codecs.getincrementaldecoder("utf-8")
  ("replace")`, `c/openai_server.py:3872,3882-3887`). **Required, one per
  emitted token, in order** — this is the only channel that carries the
  actual model output.
  (A 3-or-more-numeric-field extended `DATA` variant exists in the reader,
  `c/openai_server.py:3740-3747`, for a per-token logprob channel that no
  current request path opts into; `glm53` never emits it. Not required.)
- **`ACCEPT <id> <prompt_tokens>\n`** — **`glm53` never sends this frame.**
  It exists in the protocol (`colibri.c:8815`, `qwen38.c:1561`,
  `qwen36.c:2725`, `serve_codec.h:332`) as an early, pre-prefill
  acknowledgement that lets `#597`'s CONTEXT_EXCEEDED-before-commit
  behaviour work cleanly (`c/openai_server.py:3978-3987`: "An older engine
  that never sends ACCEPT still commits on its first DATA/DONE" — `glm53`
  is that "older engine"). The gateway's `_accept()` is idempotent-guarded
  (a second ACCEPT for the same request is a protocol error,
  `c/openai_server.py:4022-4025`) but otherwise optional: without it, the
  gateway simply treats the first `DATA` or `DONE` as commit time
  (`c/openai_server.py:4026-4027,4043-4044,4057-4058`, each passing
  `{"prompt_tokens": None}`). **Sending ACCEPT is optional but recommended**
  for a new engine — it is the only way a context-length rejection reaches
  the client as a clean early 400 rather than only being detectable by
  parsing the ERROR text after the fact (glm53 does not do this either; see
  the CONTEXT_EXCEEDED note under ERROR below).
- **`TOOL <id> <n>\n<n bytes>\n`** — same shape as DATA (bounds, terminator,
  incremental UTF-8 decode) but routed to a parallel "structured output"
  sideband channel (`c/openai_server.py:3759-3773,3889-3894`). `glm53.c`
  never emits `TOOL` (grep found no emitter in the file); tool-call parsing
  for `glm53` is done gateway-side by parsing the DATA text stream, not via
  this frame. Not required.
- **`ECHO <id> <n> <pos> <lp> <k> [tid tlp]*k\n<n bytes>\n`** — a prefill
  read-out channel for an opt-in feature no current request path uses
  (`c/openai_server.py:3774-3786`). Read and discarded to keep frame sync;
  `glm53.c` never emits it. Not required.
- **`DONE <id> STAT <emitted> <tok/s> <hit%> <rss_gb> [<prompt_tokens>
  [<limited> [<reused>]]]\n`** — terminal frame for a request, exactly one,
  always. Reader: `Engine._dispatch_stdout`'s `DONE` branch requires `len(fields)
  >= 7` (i.e. `DONE <id> STAT` plus at least 4 numeric STAT fields —
  `fields[2:]` is what gets parsed) (`c/openai_server.py:3796-3802`), and
  `Engine._stats` (`c/openai_server.py:3695-3710`) reads positionally:
  `fields[1]`=`completion_tokens` (int), `fields[2]`=`tokens_per_second`
  (float), `fields[3]`=`cache_hit_percent` (float), `fields[4]`=`rss_gb`
  (float), then **optional, trailing, append-only**: `fields[5]`
  =`prompt_tokens` (default 0 if absent), `fields[6]`=`length_limited` as
  `0`/`1` (default `False`), `fields[7]`=`reused` token count (default
  `None`, and `None` is what silently disables the ledger's MISMATCH check
  — §5). `glm53.c`'s writer supplies all seven: `serve_line("DONE %llu STAT
  %d %.2f %.1f %.1f %d %d %d\n", q->id, emitted, tok_per_s, hit_pct,
  rss_gb, prompt_tokens, limited, reused)` (`c/glm53.c:6747-6750`). **Extra
  trailing fields are safe to add** (a newer gateway reading an older
  engine that omits them gets the documented defaults; the reverse — an
  engine adding a field an older gateway doesn't parse — is exactly what
  `reused` itself was, added append-only per the comment at
  `c/openai_server.py:3706-3709` and `c/glm53.c:6733-6746`). This is the
  channel the ledger's `expect_reuse`/`engine_reuse` comparison and the
  `[req]` log's token counts both depend on — see §5.
- **`ERROR <id> <message...>\n`** — `len(fields) >= 2`
  (`c/openai_server.py:3833-3847`). One special-cased message string,
  `NOT_FOUND` (see CANCEL, above): does **not** pop the pending-request
  table entry (the request may still be coming). Every other message pops
  it and delivers a terminal error to the request. Messages `glm53.c`
  actually emits: `EMPTY_PROMPT` (`c/glm53.c:6475,6490`), `BAD_REQUEST`
  (`c/glm53.c:6479,6495,6545`, including — notably — the case where the
  prompt does not fit context: `total >= room` at `c/glm53.c:6493-6497` is
  a plain `BAD_REQUEST`, **not** the `CONTEXT_EXCEEDED <used> <limit>`
  spelling the gateway specifically pattern-matches for a clean 400 message
  — `c/openai_server.py:79-106`, `_engine_error`; a `glm53` context
  overflow therefore surfaces to the client as a generic 500 "engine
  failed", not the friendlier 400 that `colibri`/`deepseek_v4`/`qwen38`
  give. **A new engine should emit the `CONTEXT_EXCEEDED` spelling instead**
  — either the original `CONTEXT_EXCEEDED <used> <limit>` or the
  `qwen38`/`qwen36` `prompt_tokens=N requested=M capacity=C` form, both of
  which `_engine_error` understands), `BAD_FRAME` (malformed SUBMIT header,
  `c/glm53.c:6367,6851`), `NOT_FOUND` (§CANCEL), `CANCELLED` (§CANCEL, the
  ack that follows a cancelled DONE).
- **`HWINFO`, `EMAP`, `HITS`, `PROF`** — dashboard telemetry, parsed and
  stashed on the `Engine` object (`c/openai_server.py:3803-3832`) for an
  operator dashboard/profile endpoint, never required for a chat completion
  to succeed. `glm53.c` emits `EMAP` once at boot (`c/glm53.c:6841`), `HITS`
  and `PROF` once per completed request (`c/glm53.c:6718-6725`, note `PROF`
  is written **before** `DONE`, so a client reading strictly for DONE is
  unaffected either way). All optional for conformance.

Ordering that matters, all confirmed from `glm53.c`'s writer: for a normal
(non-cancelled) request, `PROF` then `DONE`
(`c/glm53.c:6718-6750`); for a cancelled one, `DONE` (with the partial
counts) then `ERROR ... CANCELLED`, that order specifically
(`c/glm53.c:6747-6766`, with the reasoning spelled out at
`c/glm53.c:6751-6765`). `DATA`/`TOOL` frames for a request always precede
its terminal `DONE`/`ERROR`.

## 4. Who does what

- **Payload is UTF-8 prompt text, already chat-templated by the gateway —
  confirmed.** `generate()` encodes the already-rendered `prompt` string as
  UTF-8 (`payload = prompt.encode("utf-8")`, `c/openai_server.py:3860`); the
  chat template itself is applied gateway-side by `render_chat_for_arch`
  before `generate()` is ever called (per-family Jinja-equivalent renderers
  in `c/openai_server.py`, selected by `ARCH`). **The engine tokenizes.**
  `glm53.c`'s `serve_one` calls `tok_encode(tokenizer, q->payload, q->plen,
  sequence, room)` (`c/glm53.c:6486`) using the tokenizer loaded from
  `<SNAP>/tokenizer.json` at boot (`c/glm53.c:6919-6921`). **A new engine
  must own its own tokenizer and its own token-id-to-UTF-8 decode** (for
  `DATA` frame text) — nothing upstream of the SUBMIT frame does this for
  it. This was the single biggest gap for `franken_decode`; it is closed by
  the GGUF's own tokenizer through libllama with `vocab_only=true` (§7).
- **Stop sequences / EOS**: two layers. The gateway maintains its own
  client-facing `stop_filter` over the decoded text stream (matching
  user-supplied `stop` strings and the family's default role markers,
  `DEFAULT_CHAT_STOP_SEQUENCES = ("<|user|>", "<|observation|>")`,
  `c/openai_server.py:2260`) and trims/ends the HTTP response there,
  independent of what the engine does. The engine separately maintains its
  own hard stop-token set from a `stops` file plus, in non-batch mode, every
  special token (`arm_stops`, `c/glm53.c:6280-6300`) and truncates
  generation itself the moment `is_stop(next)` is true, **before** emitting
  that token as a DATA frame (`c/glm53.c:6670`: `if (is_stop(next)) break;`
  — the stop token itself is never sent to the gateway). In `SERVE_BATCH=1`
  mode (what the gateway always sets) only true end-of-text stops are armed
  server-side; role markers are left for the gateway's own filter
  (`c/glm53.c:6280-6287`). A new engine should do the same split: stop
  cleanly at true EOS/whatever the model's own stop tokens are, and not try
  to reproduce the gateway's `stop` string matching — that is the gateway's
  job and it runs regardless.
- **`prompt_tokens`/`reused` for the ledger** are computed **by the
  engine**, not the gateway: `prompt_tokens` is the tokenizer's own count of
  the SUBMIT payload (`const int prompt_tokens = total;`,
  `c/glm53.c:6486-6487`), and `reused` is `shared` — however many of the
  slot's already-cached tokens matched, position-for-position, the new
  request's re-tokenized sequence (`const int reused = shared;`,
  `c/glm53.c:6626`, computed via `slot_shared`,
  `c/glm53.c:5658-5664`). The gateway's own **expectation** of that number
  (`ledger_expect_reuse`, `c/openai_server.py:2842-2857`) is a completely
  independent, purely arithmetic prediction — previous turn's
  `prompt_tokens + completion_tokens`, minus one if that turn hit its token
  budget — computed gateway-side with no knowledge of the engine's actual
  cache state. Both numbers are logged and compared
  (`ledger_report`, `c/openai_server.py:3179-3202`); disagreement is the
  `MISMATCH` `accept_live.sh` check 2b fails the daily canary on. **A new
  engine's `reused` value must be an honest, exact count of tokens it did
  not need to re-run through the model for this turn** — the gateway takes
  it at face value and alarms on any mismatch with its own prediction.
- **Sampling (temperature/top_p, greedy)** is entirely the engine's job; the
  gateway never samples. `temperature=0` from a client is passed straight
  through as `0.0` (not specially converted to a "greedy" flag anywhere
  gateway-side) — whatever `sample_token` does with `g_temp = 0` is what the
  engine defines as greedy (`c/glm53.c:6476` sets the globals, the sampler
  itself is elsewhere in `glm53.c`, not reviewed here as out of scope for
  the wire protocol).
- **`max_tokens` cap**: three layers, outermost to innermost. (1) the
  server's own `--max-tokens` flag (`~/start_glm53.sh` sets `4096`, default
  in code is `1024`, `c/openai_server.py:5473`) — every request's requested
  `max_tokens` is clamped DOWN to this, never up, and never rejected for
  exceeding it (`c/openai_server.py:3434-3438`); this is what CLAUDE.md's
  incident narrative (the old `256` value truncating every tool-calling
  reply) is about. (2) the per-request SUBMIT field, clamped value from (1).
  (3) `glm53.c`'s own `<= 0` fallback to 256 (`c/glm53.c:6510`), which only
  matters if a caller manages to send 0 (the gateway's own validation
  already rejects `maximum < 1` client-side at `c/openai_server.py:3434`,
  so in practice this fallback is defence in depth, not a live path).

## 5. KV slots and prefix reuse

**What the gateway assumes a slot retains between requests:** exactly and
only what the previous request that used that slot actually ground through
the model — `prompt_tokens + completion_tokens` of that prior turn (minus
one if it was length-limited, because the final generated token is never
fed back through another forward pass in that case) — per
`ledger_expect_reuse` (`c/openai_server.py:2842-2857`). The gateway does
**not** track token ids itself; it tracks byte/text history per
conversation (`_ledger_cache`, `entry["prompt"]`/`entry["generated"]`,
`c/openai_server.py:2877-2882`) and re-renders the full prompt text for
every request, relying on the **engine** to notice how much of the
re-tokenized sequence matches what its own slot state remembers
(`slot_shared`, `c/glm53.c:5658-5664`) and reuse the KV cache for that
shared prefix instead of recomputing it. This is a strict, ordered,
token-for-token longest-common-prefix match against the slot's *own last
remembered sequence* — not a text diff, not a hash, not a "fuzzy" match; a
single differing token anywhere before the end kills all reuse from that
point on for that slot (there is no KDA-recurrence rewind — see the long
comment at `c/glm53.c:6512-6523`).

**The minimum an engine must implement for `accept_live.sh` to pass**
(reading `tools/hot-expert/accept_live.sh` check by check):

1. **Check 1** (`accept_live.sh:46`, UI-shaped new chat, may be cold) —
   just needs a normal end-to-end SUBMIT -> DATA* -> DONE turn through Open
   WebUI's own backend. No slot reuse required.
2. **Check 2** (`accept_live.sh:47-48`, a second UI-shaped new chat) —
   requires the reported `reused` (DONE STAT's 8th field) to be `>=
   prompt_tokens - 256` and `ttft <= 60s`. This is the load-bearing one: the
   engine **must** actually reuse the KV state of a shared system
   prompt/tool block across two different conversations that share a slot
   (or across the same conversation's own multi-turn continuation, per
   check 3) — a from-scratch engine that never implements prefix reuse at
   all will fail this check on `reused`, not on correctness.
3. **Check 2b** (`accept_live.sh:49-80`, the ledger invariant, function
   `ledger_check`) — requires the engine's own `reused` field to *agree*
   with the gateway's independent prediction, for every continuation turn
   the run drove. Any `MISMATCH` fails the gate. An engine that reports
   `reused=0` always (never reuses anything) will not fail check 2b by
   itself (the gateway's prediction would then also have to be 0, which it
   never is on a continuation) — so this check punishes reporting the
   *wrong* number more than reporting a *conservative* one, but check 2
   above still requires genuine reuse regardless.
4. **Check 5** (`accept_live.sh:81-99`) — a reply sent with no client
   `max_tokens` must end with `finish_reason=stop`, never `length`. This is
   purely about not truncating early on the server's own `--max-tokens` cap
   applying correctly, and is a gateway-side clamp, not something the
   engine itself needs to implement beyond honouring the `max_tokens` field
   it is given.
5. **Check 3** (`accept_live.sh:101-121`, API two-turn with `memory: true`)
   — same reuse requirement as check 2, `reused >= prompt_tokens - 32`,
   `<= 15s` total, but via the ledger's own slot-pinning path rather than
   the hash-based fallback. Requires the prefix-checkpoint / slot-pinning
   machinery to actually persist and match across the two calls.
6. **Check 4** (`accept_live.sh:128-136`, abandoned request behind a live
   one) — requires CANCEL to actually free the engine to serve the next
   queued request promptly (`<= 45s` for a short reply behind an abandoned
   ~1000-token one with an 8 s client timeout). This is the CANCEL
   drain-and-requeue behaviour from §2 — an engine that blocks on
   in-flight generation and ignores CANCEL entirely will fail this check by
   starving every request behind the abandoned one.

None of the prefix-**checkpoint** machinery (§1's `GLM53_PREFIX_CKPT*`
knobs — restoring a prefix into a *fresh* slot from a side cache, not the
in-slot reuse above) is required for `accept_live.sh` to pass; it is a
`glm53`-specific optimisation on top of plain slot reuse, and checks 2/3
above are satisfiable by slot reuse alone (a request landing on the *same*
slot its conversation always used). A minimal new engine can skip
checkpointing entirely and still pass every `accept_live.sh` check as long
as it does true per-slot KV reuse.

## 6. Minimal conformance checklist

1. **Boot**: emit `\x01\x01READY\x01\x01\n`, then a line `STAT 0 0.00 0.0
   <rss_gb>\n`, flush both, before touching stdin further (§1).
2. **SUBMIT**: parse `SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>`
   (6 fields minimum), read exactly `len` bytes of UTF-8 prompt text plus a
   trailing `\n`; tolerate (but for a text-only `glm53`-family engine, may
   ignore) an optional 7th (`xlen`) and 8th (`prefix_bytes`) field, and an
   `IMAGE` frame preceding it (reject IMAGE cleanly if unsupported) (§2).
3. **Own tokenizer**: encode the payload text to token ids and decode
   generated ids back to UTF-8 text (§4).
4. **Per-slot KV state**: `KV_SLOTS` independent, persistent contexts;
   given a new prompt's token sequence, compute the longest exact prefix
   match against the slot's last remembered sequence, reuse that much KV
   state, and report the match length honestly as the `reused` field (§3,
   §5) — required for `accept_live.sh` checks 2, 2b and 3.
5. **DATA per token**: `DATA <id> <n>\n<n bytes>\n`, flushed, in generation
   order, never including the stop token itself (§3, §4).
6. **DONE**: `DONE <id> STAT <emitted> <tok/s> <hit%> <rss_gb>
   <prompt_tokens> <limited 0|1> <reused>\n`, exactly once per request, as
   the terminal frame for a successful or budget-limited turn (§3).
7. **CANCEL**: while generating, poll stdin between tokens for `CANCEL
   <this id>`; on match, stop generating, still emit `DONE` (partial
   counts) then `ERROR <id> CANCELLED`, in that order; for any other id
   seen while nothing matching is in flight, answer `ERROR <id>
   NOT_FOUND`; queue (don't drop or block on) any SUBMIT frames that arrive
   while a turn is in flight (§2) — required for `accept_live.sh` check 4.
8. **ERROR** for malformed frames (`BAD_FRAME`), unencodable/empty prompt
   (`EMPTY_PROMPT`), and prefer emitting `CONTEXT_EXCEEDED <used> <limit>`
   over a generic `BAD_REQUEST` when the prompt does not fit context, so
   the gateway can surface a clean 400 (§3) — `glm53` itself gets this
   wrong today; a new engine doesn't have to copy that gap.
9. **STOP** may be a no-op (§2) — safe to ignore entirely.
10. **ACCEPT, TOOL, ECHO, HWINFO, EMAP, HITS, PROF** are all optional (§3).

### Smallest local test: run `openai_server.py` against a fake engine process

`c/tests/test_openai_server.py` drives the `Engine` class end-to-end against
a synthetic child process with no real binary involved — this is the
fastest way to validate a candidate wire implementation by hand-writing the
exact bytes it should produce and replaying them:

- `FakeProcess` (`c/tests/test_openai_server.py:632-659`) stands in for
  `subprocess.Popen`'s return value: a `BlockingStream` for `stdout`
  pre-seeded with `READY + b"STAT 0 0 0 0\n"`
  (`c/tests/test_openai_server.py:634`), and a `stdin` whose `.write()`
  captures every frame the gateway sends and calls back into the test's
  `on_write(process, data)` hook so the test can assert on the exact SUBMIT
  bytes and then `process.stdout.feed(...)` the scripted reply frames
  in response.
- `DispatcherTest` (`c/tests/test_openai_server.py:662+`) is the harness
  class; each `test_*_request_and_response_transcript_is_byte_exact` method
  (e.g. `test_v4_request_and_response_transcript_is_byte_exact`,
  `c/tests/test_openai_server.py:690-716`, which is the one that exercises
  the `ACCEPT` frame and the 8-field prefix-hint SUBMIT) patches
  `openai_server.subprocess.Popen` to return a `FakeProcess`, constructs a
  real `Engine(family_id, "model")`, calls `engine.generate(...)`, and
  asserts on both the exact bytes written to the fake stdin and the
  decoded text/stats the gateway produced from the scripted reply.
- Run it with: `cd c && python3 -m pytest tests/test_openai_server.py -k
  <name>` (or `python3 -m unittest tests.test_openai_server -v` from `c/`) —
  no rig, no GPU, no real model directory; this is exactly the right place
  to add a `test_franken_...` case once a `franken` family id exists in
  `family_registry.py`, feeding it the exact byte sequences this document
  specifies and asserting the gateway parses them as expected. It is also
  the right place to write the *inverse* test — feed the gateway's own
  SUBMIT bytes into a hand-written reference decoder and confirm they match
  what this document says field-for-field — before ever pointing a real
  `franken_decode` process at `openai_server.py`.

## Top 5 things a new engine most easily gets wrong

1. **Assuming the SUBMIT payload is already tokenized.** It is UTF-8 prompt
   text, already chat-templated by the gateway — the largest actual gap for a
   harness that took `--tokens <id>...` on argv, not a subtlety (§4).
2. **Treating `reused` as advisory or skippable.** It is a specific,
   checked number the gateway's ledger cross-validates against its own
   independent prediction every single continuation turn
   (`ledger_report`, `c/openai_server.py:3179-3202`) and `accept_live.sh`'s
   check 2b fails the whole gate on any `MISMATCH` — reporting it wrong is
   worse than not implementing reuse at all (§5).
3. **Forgetting to flush stdout after every frame**, or block-buffering
   stdout because it's a pipe. The gateway's reader blocks on exactly the
   bytes it expects, in order; a buffered write that sits in libc's stdio
   buffer until the process exits or the buffer fills will hang every
   request (§1).
4. **Not queuing SUBMIT/CANCEL frames that arrive while a turn is
   in flight.** With `KV_SLOTS > 1` the gateway pushes multiple requests
   down the same single stdin pipe and expects the engine to read ahead
   during generation (`glm53.c`'s `g_queue` ring, §2) — an engine that only
   reads stdin between requests will desync the frame stream the first time
   two slots are both busy, or will simply never see a CANCEL for a request
   still sitting unread in the pipe.
5. **Getting the CANCEL/DONE/ERROR ordering backwards.** DONE must precede
   the `ERROR ... CANCELLED` ack, never the reverse — the gateway's
   dispatcher pops the pending-request table entry on ERROR, so a DONE
   arriving after would be silently discarded along with its token counts
   and accounting (§3). This is exactly the kind of ordering bug that
   passes every "does it generate text" smoke test and only shows up as a
   missing `[req]` log line or a `accept_live.sh` check 2b false pass from
   an unchecked continuation.

(Runner-up, worth a line of its own: sending `CONTEXT_EXCEEDED` as a plain
`BAD_REQUEST` — as `glm53` itself does today — silently downgrades a clean
400 "shorten your conversation" into an opaque 500 for every client that
overflows context. Free to copy `glm53`'s other gaps for parity, but not
this one, since `_engine_error` (`c/openai_server.py:79-106`) already knows
how to produce the better message if the engine gives it the chance.)

## 7. What answers each of these, in `franken_serve.cpp` (L0 step 4, 2026-09-22)

The checklist of §6 against the code, so the two can be diffed rather than
believed. Everything below was exercised by `serve_conformance.py` on the CPU
binary (14 checks, 0 failures, `--layers 0-3`); **no GPU run has happened.**

| §6 item | where | conformance check |
|---|---|---|
| 1 boot: READY then `STAT 0 0.00 0.0 <rss>` | `Server::run()` | "boot: READY + STAT with >= 5 fields" |
| 2 SUBMIT parse, 6/7/8 fields, counted-byte body | `FrameIO::read_body` | every turn |
| 3 own tokenizer | `Vocab`, libllama `vocab_only=true`, `dlopen`ed | every turn (the prompt is text) |
| 4 per-slot KV + honest `reused` | `Slot`, `plan_reuse`, `DecodeRunner::save_state/load_state` | "turn B: reused == ledger_expect_reuse", "turn C: rollback to the snapshot below the LCP" |
| 5 DATA per token, flushed, stop token never sent | `wire_data`, the decode loop's `is_eog` break | "one DATA per emitted token" |
| 6 DONE with all seven STAT fields | end of `serve_one` | every turn |
| 7 CANCEL: drain, queue SUBMITs, DONE **then** ERROR CANCELLED | `FrameIO::drain`, `serve_one` | "turn D", "turn E" |
| 8 BAD_FRAME / EMPTY_PROMPT / CONTEXT_EXCEEDED | `serve_one`, `read_body` | "context overflow", "empty prompt" |
| 9 STOP is a no-op | `FrameIO::drain` ignores it | — |
| 10 ACCEPT sent (optional but recommended); TOOL/ECHO/HWINFO/EMAP/HITS/PROF absent | `serve_one` | "turn A: ACCEPT carries prompt_tokens" |

Two things this engine does that `glm53` does not, both called for above:
it sends `ACCEPT <id> <prompt_tokens>` before prefilling, and it spells a
context overflow `CONTEXT_EXCEEDED prompt_tokens=N requested=M capacity=C`
(the qwen38 form `_engine_error` understands) instead of `BAD_REQUEST`, so the
client gets a 400 that says what to do.

One thing it does not do: prefix **checkpoints**. Per §5 that is allowed —
plain per-slot reuse is what `accept_live.sh` checks 2/2b/3 need — but it is
also the one place where a real serving run could still differ from `glm53`'s
behaviour, and no serving run has been made.
