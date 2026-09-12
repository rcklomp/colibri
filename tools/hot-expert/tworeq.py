"""Session-state oracle: N identical requests in ONE persistent engine.

Written for G12, after the chain shipped a `coli_vk_kda_init` that kept the
previous conversation's KDA recurrence on the GPU across sessions. Every other
oracle in this tree runs one request per process and is blind to that class of
bug by construction; this one is not. Run it for ANY change that gives
per-conversation state a new home.

Greedy decode, so the text must be byte-identical across requests; if request 2
differs from request 1, session state leaked across requests.

ARCHES
------
`TWOREQ_ARCH` picks which engine is driven; it defaults to `glm53`, which is
what every recorded run before 2026-09-12 used, so no existing caller changes
behaviour. `qwen38` was added for Q7-gpu (the dense BF16 stream on dev0), whose
spec asked for this check and whose arbitration made it a merge condition: that
item uploads 6.7 GB of dense weights to dev0 once, before the expert preload,
and dispatches them per decode row -- no per-conversation state by design, which
is exactly the claim a three-request run in one process can falsify and a
one-request-per-process oracle cannot.

`TWOREQ_SNAP` and `TWOREQ_EXE` override the snapshot and the binary for either
arch (TWOREQ_EXE since P7: gate a CANDIDATE without copying it over the served
one -- the served tree's binary cannot even be relinked while the gateway runs).

The engine's family is resolved from the snapshot's own config.json, not from
TWOREQ_ARCH, so a mismatched pair is caught by `resolve_model` rather than by a
confusing decode.

ENV
---
glm53 : COLI_VULKAN=1, COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695, a COPY of
        the histogram, OMP 8/cores/close, plus whichever knob is under test.
qwen38: Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto, a COPY of
        `.coli_usage` (qwen38 REWRITES it at exit -- §RP1's confound; without a
        copy the three requests here are fine but the NEXT run is not), OMP
        8/cores/close, plus the knob (e.g. Q38_DENSE_GPU=7).
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c"))
import openai_server as rt

ARCH = os.environ.get("TWOREQ_ARCH", "glm53")
DEFAULTS = {
    "glm53":  (os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64"),
               os.path.expanduser("~/src/colibri/c/glm53"),
               "Explain in three sentences why a gated delta rule needs a decay term."),
    "qwen38": (os.path.expanduser("~/models/Qwen3.8-Flash-Next-FP8"),
               os.path.expanduser("~/src/colibri/c/qwen38-vk"),
               "Explain in three sentences why a gated delta rule needs a decay term."),
}
if ARCH not in DEFAULTS:
    sys.exit(f"TWOREQ_ARCH={ARCH!r} is not one of {sorted(DEFAULTS)}")
snap_default, exe_default, question = DEFAULTS[ARCH]
SNAP = os.environ.get("TWOREQ_SNAP") or snap_default
EXE  = os.environ.get("TWOREQ_EXE") or exe_default

res  = rt.resolve_model(SNAP)
rt.ARCH = res.descriptor.id
print(f"[tworeq] arch={ARCH} resolved={rt.ARCH} exe={EXE} snap={SNAP}", flush=True)
prompt = rt.render_chat_for_arch([{"role": "user", "content": question}],
                                 enable_thinking=False)

# TWOREQ_SLOTS=N (P6b): N KV slots, request i on slot i % N -- every slot must
# produce the same text as slot 0, or per-slot state (CPU or device) leaks.
NSLOTS = int(os.environ.get("TWOREQ_SLOTS", "1"))
NGEN = int(os.environ.get("TWOREQ_NGEN", "64"))
eng = rt.Engine(EXE, SNAP, cap=512, max_tokens=NGEN, kv_slots=NSLOTS)
outs = []
for i in range(max(3, NSLOTS)):
    buf = []
    eng.generate(prompt, NGEN, 0.0, 1.0, lambda t: buf.append(t), cache_slot=i % NSLOTS)
    outs.append("".join(buf))
    print(f"--- request {i+1} ---\n{outs[-1]}\n", flush=True)
ok = all(o == outs[0] for o in outs)
# Close the engine and WAIT for it: leaving the child alive made the next step
# of p7_gate.sh refuse ("another glm53 is running") and, worse, made a revert's
# `cp` over the served binary fail with ETXTBSY, which left the candidate in
# service after a failed gate (2026-09-07).
try:
    eng.close()
    eng.process.wait(timeout=60)
except Exception:
    try:
        eng.process.kill()
    except Exception:
        pass
print("RESULT:", f"IDENTICAL across {len(outs)} requests"
      if ok else f"DIVERGED across {len(outs)} requests")
sys.exit(0 if ok else 1)
