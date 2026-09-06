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
EXE  = os.path.expanduser("~/src/colibri/c/glm53")
res  = rt.resolve_model(SNAP)
rt.ARCH = res.descriptor.id
prompt = rt.render_chat_for_arch([{"role":"user","content":
    "Explain in three sentences why a gated delta rule needs a decay term."}],
    enable_thinking=False)

eng = rt.Engine(EXE, SNAP, cap=512, max_tokens=64)
outs = []
for i in range(3):
    buf = []
    eng.generate(prompt, 64, 0.0, 1.0, lambda t: buf.append(t), cache_slot=0)
    outs.append("".join(buf))
    print(f"--- request {i+1} ---\n{outs[-1]}\n", flush=True)
ok = all(o == outs[0] for o in outs)
print("RESULT:", "IDENTICAL across 3 requests" if ok else "DIVERGED")
sys.exit(0 if ok else 1)
