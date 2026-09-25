#!/usr/bin/env python3
# tools/hot-expert/greedy_ids.py -- record one MMLU-Pro item's rendered prompt
# token ids, its greedy completion under llama-server, and the top-5
# logprobs at every generated step, so franken_decode_ds4's --teacher-force
# can replay the exact same ids and report the first position where the two
# engines' greedy choices diverge.
#
# Why this exists (record §L5-DS4-SERVE, ROME-3x7900XTX-2026-09-04.md):
# DeepSeek-V4-Flash on the Franken engine falls into greedy repetition loops
# roughly 3x as often as llama.cpp on the same MMLU-Pro prompts (temperature
# 0, reasoning xhigh): question_id 570 loops to the 16000-token budget on
# Franken, llama.cpp stops at 613 tokens. This script captures llama.cpp's
# own greedy trace so the engine can be teacher-forced along it and the
# first disagreement -- and its logit margin -- can be found.
#
# The prompt is rendered with c/openai_server.py's OWN render_chat_v4 (import,
# not a reimplementation, so this can never drift from what the gateway
# would actually send), because that is the chat-template rendering the bug
# report is about. It is then tokenized and run through llama-server's own
# /tokenize and /completion endpoints -- NOT llama-server's baked-in jinja
# chat template (which /v1/chat/completions would use instead, and which
# might legitimately render the same conversation differently) -- so the
# token ids driving generation are unambiguously llama-server's own vocab,
# and the exact ids used for the completion are recorded, not re-derived.
#
# Usage (against `serve_alt.sh deepseek`, port 8081, orchestrator only --
# this script performs live inference against a GPU-backed engine and must
# never be launched by a subagent; see CLAUDE.md "How a session drives the
# rig"):
#   python3 tools/hot-expert/greedy_ids.py --question-id 570 \
#       --out ~/bench/franken/ds4/greedy_570.json
#
# Output JSON: prompt_ids, gen_ids, and per-step top-5 (id, logprob), plus
# enough of the request/response to audit the run (rendered prompt, finish
# reason, the decoded text). Capped at --max-tokens generated tokens
# (default 1024, matching franken_decode_ds4's --teacher-force scope).

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent          # tools/hot-expert
REPO = HERE.parents[1]                           # colibri repo root
C_DIR = REPO / "c"

sys.path.insert(0, str(HERE))                    # quality_eval
sys.path.insert(0, str(C_DIR))                   # openai_server, v4_dsml, family_registry

import quality_eval                              # noqa: E402  (mmlu_prompt, mmlu_load_sample)
import openai_server                              # noqa: E402  (render_chat_v4)


DEFAULT_MMLU = os.path.expanduser("~/bench/f11_quality/mmlu_pro_sample.json")
DEFAULT_KEY_FILE = os.path.expanduser("~/.colibri_api_key")


def http_post(url, key, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def load_item(mmlu_path, question_id):
    items = quality_eval.mmlu_load_sample(mmlu_path)
    for it in items:
        if it["question_id"] == question_id:
            return it
    raise SystemExit(f"question_id {question_id} not found in {mmlu_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question-id", type=int, default=570)
    ap.add_argument("--mmlu", default=DEFAULT_MMLU)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--reasoning-effort", default="xhigh",
                     help="rendered into the prompt via render_chat_v4 (DSV4_REASONING_EFFORT)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024,
                     help="cap on generated tokens (--teacher-force's scope)")
    ap.add_argument("--n-probs", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    key = open(args.key_file).read().strip()
    base = f"http://{args.host}:{args.port}"

    item = load_item(args.mmlu, args.question_id)
    prompt_text = quality_eval.mmlu_prompt(item)
    messages = [{"role": "user", "content": prompt_text}]

    # The gateway's OWN renderer (c/openai_server.py render_chat_v4), imported
    # directly: enable_thinking=True + reasoning_effort="xhigh" is exactly what
    # chat_completion() computes for a client that sends reasoning_effort=
    # "xhigh" (reasoning_effort not in (None, "none") => enable_thinking=True;
    # openai_server.py:5054).
    rendered = openai_server.render_chat_v4(
        messages, enable_thinking=True, reasoning_effort=args.reasoning_effort,
        tools=None, tool_choice=None,
    )

    # llama-server's OWN tokenizer (not a local guess): parse_special=True so
    # the literal special-token strings render_chat_v4 wrote (bos/user/
    # assistant markers) become their real ids instead of being split into
    # subword pieces; add_special=False because render_chat_v4 already wrote
    # the literal BOS marker itself -- add_special=True here would double it.
    tok_resp = http_post(f"{base}/tokenize", key,
                          {"content": rendered, "add_special": False, "parse_special": True},
                          timeout=120)
    prompt_ids = tok_resp["tokens"]
    if not prompt_ids or not isinstance(prompt_ids[0], int):
        raise SystemExit(f"/tokenize returned an unexpected shape: {tok_resp}")

    # Submit the prompt as an EXPLICIT id array, not a string: server-common.cpp's
    # tokenize_input_prompts() treats a JSON array of numbers as already-
    # tokenized and passes it straight through (no re-tokenization, no
    # add_special applied a second time) -- so the ids recorded above are
    # bit-for-bit the ids the completion actually ran on, not a
    # reconstruction of them.
    t0 = time.time()
    comp = http_post(
        f"{base}/completion", key,
        {
            "prompt": prompt_ids,
            "temperature": args.temperature,
            "n_predict": args.max_tokens,
            "n_probs": args.n_probs,
            "cache_prompt": False,
            "stream": False,
        },
        timeout=args.timeout,
    )
    wall_s = time.time() - t0

    probs = comp.get("completion_probabilities") or []
    gen_ids = [p["id"] for p in probs]
    steps = []
    for p in probs:
        top5 = [{"id": t["id"], "logprob": t["logprob"]} for t in p.get("top_logprobs", [])]
        steps.append({"chosen_id": p["id"], "top5": top5})

    out = {
        "question_id": item["question_id"],
        "category": item["category"],
        "correct_letter": item["answer"],
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "max_generated_tokens": args.max_tokens,
        "n_probs": args.n_probs,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "rendered_prompt": rendered,
        "prompt_ids": prompt_ids,
        "gen_ids": gen_ids,
        "n_gen": len(gen_ids),
        "finish_reason": comp.get("stop_type") or comp.get("stopped_eos") or comp.get("truncated"),
        "content": comp.get("content"),
        "wall_s": wall_s,
        "steps": steps,
        "source": {"host": args.host, "port": args.port, "engine": "llama-server (serve_alt.sh deepseek)"},
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: prompt_ids={len(prompt_ids)} gen_ids={len(gen_ids)} "
          f"finish={out['finish_reason']} wall_s={wall_s:.1f}")


if __name__ == "__main__":
    main()
