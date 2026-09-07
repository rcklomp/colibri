"""Session-state oracle: N identical requests in ONE persistent engine.

Written for G12, after the chain shipped a `coli_vk_kda_init` that kept the
previous conversation's KDA recurrence on the GPU across sessions. Every other
oracle in this tree runs one request per process and is blind to that class of
bug by construction; this one is not. Run it for ANY change that gives
per-conversation state a new home.

Greedy decode, so the text must be byte-identical across requests; if request 2
differs from request 1, session state leaked across requests.

Needs the usual GLM env (COLI_VULKAN=1, the 1695 tier caps, a COPY of the
histogram, OMP 8/cores/close) plus whichever knob is under test."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c"))
import openai_server as rt

SNAP = os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")
# TWOREQ_EXE (P7): gate a CANDIDATE binary without copying it over the served
# one -- the served tree's glm53 cannot even be relinked while the gateway runs.
EXE  = os.environ.get("TWOREQ_EXE") or os.path.expanduser("~/src/colibri/c/glm53")
res  = rt.resolve_model(SNAP)
rt.ARCH = res.descriptor.id
prompt = rt.render_chat_for_arch([{"role":"user","content":
    "Explain in three sentences why a gated delta rule needs a decay term."}],
    enable_thinking=False)

# TWOREQ_SLOTS=N (P6b): N KV slots, request i on slot i % N -- every slot must
# produce the same text as slot 0, or per-slot state (CPU or device) leaks.
NSLOTS = int(os.environ.get("TWOREQ_SLOTS", "1"))
eng = rt.Engine(EXE, SNAP, cap=512, max_tokens=64, kv_slots=NSLOTS)
outs = []
for i in range(max(3, NSLOTS)):
    buf = []
    eng.generate(prompt, 64, 0.0, 1.0, lambda t: buf.append(t), cache_slot=i % NSLOTS)
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
print("RESULT:", "IDENTICAL across 3 requests" if ok else "DIVERGED")
sys.exit(0 if ok else 1)
