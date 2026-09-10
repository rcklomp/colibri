# P10 — value-based checkpoint eviction: worth, not recency (spec, 2026-09-10)

Design pass for the build that follows. It replaces the only policy in the
checkpoint cache that was never measured — *which slot dies* — and lets
`GLM53_PREFIX_CKPT_MIN` go back to being a sanity floor instead of a shield.

## 0. Why, from the measurements

The engine keeps four checkpoint slots (`GLM53_CKPT_MAX_SLOTS = 8`,
`ckpt_slot_count()` default 4) and evicts, inside each `kind` pass, **the
slot with the oldest `used` clock** (`c/glm53.c`, `ckpt_store`). That is
LRU, and LRU is the wrong question here. The cache does not hold pages of a
working set; it holds *prefixes*, and a prefix is worth what it saves times
how often it saves it.

The live gateway says so directly. Taken from `~/glm53_server.log` at
2026-09-10 06:40, one restart's worth of history:

| slot | prefix | size | `CKPT hit` since 02:36 |
|---:|---:|---:|---:|
| 0 | 4 707 | 315 MB | **0** |
| 1 | 4 307 | 302 MB | **0** |
| 2 | 4 712 | 315 MB | **0** |
| 3 | **4 418** | 305 MB | **9** |

Slot 3 is Open WebUI's tool block — the prefix every new conversation in the
owner's browser starts with, the one thing in this cache that has ever paid
for itself. Slots 0, 1 and 2 are captures from gate runs and API-shaped
warm-ups that no request has matched since they were written. **LRU cannot
tell these four apart**: after a disk load all four `used` clocks are set
within one loop, and from then on the only slot whose clock advances is the
one that is being hit — so the tool block survives today by the accident that
it is popular, and dies the moment a burst of captures lands between two of
its hits.

That is not hypothetical. It was measured twice:

| when | what happened | cost |
|---|---|---:|
| P9 gate, 2026-09-09 | a **170-token** API chat stored ~160 MB and evicted a **4 307-token** capture | ~24 min of re-prefill inside the gate |
| 05:00 canary, nightly | the same shape, reproduced every morning | one cold new chat per day |

**And the capture that wins is not even cheap.** Fitting the two sizes we
have — 170 tokens → 160 MB, 4 418 tokens → 305 MB — the blob is about
**154 MB fixed plus 34 KB per token**: the KDA recurrence and convolution
window do not depend on length, only the DSA rows do. A 170-token checkpoint
therefore costs **52 % of the memory** of the tool block and returns
**3.8 % of the tokens**. Recency ranked it first anyway.

`GLM53_PREFIX_CKPT_MIN=1024` (in service since 2026-09-10 01:35, commit
`995cd48`) is the threshold half of the fix. It works — the short-chat probe
in `ckptmin_verify.sh` stores nothing — and it is blunt in exactly the way a
threshold is: it also refuses a legitimate 500-token prefix, and it does not
help at all against the case the roadmap names, **a long one-off paste**,
because that capture is *longer* than the tool block. A length-only rule
hands it the slot.

## 1. What P10 changes

One field, one comparison.

```c
typedef struct {
    int *ids;
    int len;
    unsigned char *blob;
    size_t bytes;
    unsigned long long used;   /* orologio LRU — resta, come spareggio */
    unsigned hits;             /* P10: quante volte ckpt_restore l'ha scelto */
    int kind;
} PrefixCkpt;
```

**The score is `len × hits`.** Eviction picks the *lowest* score inside the
same `kind` pass the code already walks, with `used` as the tie-break so the
order stays total and deterministic.

- **A new capture is born with `hits = 1`**, not 0. Its score is therefore
  its length, which is exactly the length-only rule — the right starting
  guess for a prefix nothing has voted on yet, and it means a fresh capture
  is never the automatic next victim.
- **`ckpt_restore` increments the winning slot's `hits`.** Only the winner:
  a shorter checkpoint that also matched did not save anything this time.
- **A re-capture would reset the counter, and cannot happen.** `ckpt_plan`
  and `ckpt_hint` both consult `ckpt_have()` and refuse a prefix already in
  a slot, so the tool block accumulates hits instead of being rewritten at
  `hits = 1` every morning. This is load-bearing for the whole design and it
  is already true.

Applied to the table above, the next capture evicts **slot 1** (score 4 307,
the stale one nobody has matched) instead of choosing between four slots it
cannot rank. The tool block scores 4 418 × 10 = **44 180** and is now the
*last* thing out, not the first — and it beats the roadmap's long one-off
paste (score ≈ its length, ~6 000) by seven times.

## 2. The threshold goes back down

Once eviction can rank, the minimum no longer has to shield. `~/start_glm53.sh`
drops `GLM53_PREFIX_CKPT_MIN=1024` and returns to the code default, **128** —
the 500-token prefix is cacheable again, which is the second half of P10's
expected outcome.

This is not a free ride on the new policy, it is the *test* of it: at
MIN=1024 a flood of short chats stores nothing and the gate below would prove
nothing. **The flood gate is only meaningful with the minimum lowered**, so
the gate runs at 128 and the serving change ships with it.

## 3. Persistence

`hits` is what the slot has earned; a gateway restart should not confiscate
it. The on-disk record gains a trailing `uint32` after the blob, and the
magic stays **`G53CKPT1`**:

- an **old file** read by the new binary ends after the blob, `fread` returns
  short, and `hits` defaults to 1 — the length-only rule, which is the
  correct fallback;
- a **new file** read by an old binary is byte-for-byte what it expects up to
  the end of the blob, and it never reads further.

Both directions work, so **no existing checkpoint is invalidated** and nobody
pays the ~10-minute cold prefill that a magic bump would have cost.

The counter is updated in place on a hit — a 4-byte write at a computed
offset (`28 + len*4 + bytes`), best-effort, guarded on the file already being
at least that long, and silently skipped on any failure. Rewriting 305 MB per
hit is obviously not on the table; leaving the disk copy stale would mean the
tool block came back from every restart at `hits = 1`.

## 4. Instrument

The two verbose lines gain the number the policy now turns on, so the gate
and the canary can read it instead of inferring it:

```
CKPT store prefix=4418 305 MB kind=0 slot=3 hits=1 score=4418 evicted=4307/1 sync=… copy=…
CKPT hit prefix=4418 slot=3 hits=10
```

`evicted=<len>/<hits>` says what was thrown away and what it was worth — the
line that would have made the P9-gate incident obvious the morning it
happened, instead of ~24 minutes of unexplained re-prefill.

## 5. Knob and blast radius

| env | default | meaning |
|---|---|---|
| `GLM53_CKPT_VALUE_EVICT` | **1** | 0 = today's pure LRU inside each `kind` pass, for the gate's control arm and for a one-line rollback |
| `GLM53_PREFIX_CKPT_MIN` | 128 (code) | serving drops from 1 024 back to the default |

Default-on, following `COLI_LEDGER` (P9) and `COLI_REPLY_PIN` (P8): a policy
change that is off by default measures nothing. The rollback is the knob.

**No numerics change, and the oracle is exact.** Eviction decides *which*
prefill is reused, never *what* is computed: `ckpt_restore` accepts a slot
only after `memcmp` of the full `len` ids, so a restored prefix is by
construction the tokens the engine would have ground itself. With
`GLM53_PREFIX_CKPT=0` none of this code runs at all, which is why step 1
below must come back bit-identical rather than merely close. (The known
~1e-5 chunk-boundary drift recorded under P7b is a property of restoring at a
*different* boundary; P10 does not move any boundary, it only chooses which
boundary survives.)

`c/glm53.c` only. No shader, no gateway, no header — but the build still
makes both engines, per CLAUDE.md, because the Makefile's header
prerequisites make that cheap and the rule has no exceptions.

## 6. Gate (executable; `p10_gate.sh`)

1. **`prefill_gate.sh <served> <candidate>`**, `GLM53_PREFIX_CKPT=0`,
   private `COLI_CKPT_DIR`, `MIN_SPEEDUP=0.97`, `PROFILE_MIN_RESIDENT=97`:
   teacher forcing IDENTICAL, `max_abs=0`, and the serve rows neutral. A
   policy that only fires with checkpoints ON must be invisible with them off.
2. **`tworeq.py`** at `TWOREQ_SLOTS=4`, `COLI_KDA_GPU=2` and `=0`:
   IDENTICAL, no forcing line.
3. **THE FLOOD, in two arms.** 20 short chats, then one UI-shaped new chat.
   The flood is 10 distinct ~200-token system prefixes, each sent twice
   (question A then question B): the second of a pair gives `ckpt_plan` an
   LCP equal to that unique prefix, so each pair *does* store — a flood of
   captures, which is the thing being defended against, and not merely a
   flood of requests.

   | arm | | expected |
   |---|---|---|
   | **on** | `GLM53_CKPT_VALUE_EVICT=1` | `CKPT hit prefix>=4300` on the chat after the flood |
   | **off** | `GLM53_CKPT_VALUE_EVICT=0` | the tool block is gone: no such hit, one cold chat |

   Both arms run at `GLM53_PREFIX_CKPT_MIN=128`. The `off` arm is the control
   that proves the new policy is what saved the capture, in the shape P9's
   gate established — an item that passes without its control passes for an
   unknown reason.
4. **`accept_live.sh`** PASS on the rig against the served gateway: the four
   checks plus the ledger's alarm, i.e. the request *after* the one under
   test, per P7b.
5. **Mac-side, and owed before the owner is told anything works:**
   `accept_ui.sh` (real Chromium, first token on screen) and
   `ui/ui_matrix.sh --sizes 0,500,2000` with **every turn-2 row unchanged**
   against `matrix-p9-2026-09-10.tsv` — P10 must not cost a follow-up turn
   anything, and turn 2 is where a checkpoint mistake would show.

## 7. What this does not fix, stated plainly

- `hits` is a lifetime count with no decay. A prefix that was popular last
  month and is dead today keeps its score. With four slots and a workload of
  one dominant prefix that is the behaviour we want; if the workload ever
  grows several competing prefixes, an aging term (halve every N stores) is
  the follow-up, and the field is already there to age.
- Nothing here makes a *new* prefix cheaper. A 3 500-token paste is still
  minutes; P10 only stops it from taking the tool block down with it.
- Four slots is still four. P10 makes the choice among them defensible, it
  does not add capacity — `GLM53_PREFIX_CKPT_SLOTS` already goes to 8 and the
  cost is ~300 MB each.

## 8. Effort

One day: the field, the comparison, the trailing disk word, the two log
lines, the flood gate and its chain. Opus.
