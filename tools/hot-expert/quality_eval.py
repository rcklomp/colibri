#!/usr/bin/env python3
"""quality_eval.py -- answer-QUALITY harness for whichever model serve_alt.sh has put behind
port 8081 (Franken plan item F11, "what nobody has measured is answer quality of these
quantizations"). Python 3 stdlib only -- the rig has no numpy/pandas/pyarrow and nothing is
pip-installed for this (CLAUDE.md).

Two parts, one item id namespace, one jsonl per model:
  Part A -- MMLU-Pro (TIGER-Lab/MMLU-Pro, test split, MIT), 70 questions (5 per each of 14
    categories, frozen by `fetch` into a sample file), zero-shot CoT, greedy (temperature 0).
  Part B -- a needle-in-haystack test built from this repo's own measurement record
    (tools/hot-expert/ROME-3x7900XTX-2026-09-04.md), 8 made-up 6-digit codes inserted at
    evenly spread depths, asked back as JSON.

Subcommands:
  fetch      -- write the frozen 70-question MMLU-Pro sample (network, HF datasets-server).
  run        -- send every not-yet-answered item to one server, append results as they land.
  summarize  -- per-model accuracy (Wilson 95% CI), per-category table, needle table, and a
                paired McNemar comparison between every two models on their SHARED items.

Identity check (requirement f): `run` refuses to talk to a server that is not the one
--expect names, verified read-only. For qwen38/deepseek (llama-server, HIP docker) this reads
GET /props and checks the `model_path` field -- the on-disk gguf path, NOT the OpenAI-style
model id, which is always the gateway's own alias "glm-5.3-flash" regardless of backend
(serve_alt.sh's whole point). Read in ~/src/llama-glm53/tools/server on the rig:
  - server.cpp:135,142 `is_router_server = params.model.path.empty() && ...` -- serve_alt.sh
    always passes `-m <path>`, so this instance takes the NON-router branch and its
    routes.get_props/get_models come from server-context.cpp's own handlers (server.cpp
    assigns the router versions only `if (is_router_server)`, server.cpp:236-237).
  - server-context.cpp:4777 `this->get_props = ...` calls `get_res_props(*meta, params, false)`
    (server-context.cpp:4784); get_res_props (server-context.cpp:4572) puts
    `{"model_path", meta.model_path}` in the response (line 4590).
  - server-context.cpp:4167 sets that same field from `impl->params_base.model.path` at
    construction time -- the exact path given to `-m`, independent of `--alias`.
  - server.cpp:252 `ctx_http.get("/v1/models", ...)` and server-context.cpp:5045's
    `get_models` return `meta.model_name` (server-context.cpp:1374, `= *params_base
    .model_alias.begin()` when `--alias` is set) -- this is ALWAYS "glm-5.3-flash" here and
    cannot distinguish the three servers. /props's `model_path` is therefore the only
    read-only field the identity check can use.
For glm, no HTTP call is made for the identity check at all (CLAUDE.md's own description of
what serve_alt.sh's Colibri-gateway path exposes read-only): `pgrep -f "openai_[s]erver.py"`
must find the gateway, and `ps -C glm53 -o stat=` must show a non-zombie row (the same
zombie-safe check every chain on this rig uses since the 2026-09-20 incident CLAUDE.md
documents -- a killed engine child is a ZOMBIE and still matches a bare `pgrep -x glm53`).

This script never prints or copies the API key; it is read from --key-file at request time.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(HERE, "ROME-3x7900XTX-2026-09-04.md")

# ----------------------------------------------------------------------------------- Part A --
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_SIZE_URL = "https://datasets-server.huggingface.co/size"
MMLU_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_CONFIG = "default"
MMLU_SPLIT = "test"
MMLU_PAGE = 100                 # datasets-server refuses length > 100 (verified 2026-09-21)
MMLU_PER_CATEGORY = int(os.environ.get("MMLU_PER_CATEGORY", "5"))   # 15 for the 210-item run of 2026-09-22
MMLU_EXPECTED_CATEGORIES = 14
MMLU_SAMPLE_SEED = 20260921     # today's date, fixed -- documented, not re-derived per run
LETTERS = "ABCDEFGHIJ"

EXPECTED_MMLU_FEATURES = {
    "question_id", "question", "options", "answer", "answer_index", "cot_content",
    "category", "src",
}


def _http_get_json(url, timeout=60, retries=10, retry_sleep_base=2.0):
    """GET url, parse JSON, retry on connect/HTTP-5xx/429/timeout errors. Raises on anything
    else (a 4xx that isn't 429, or a body that isn't JSON) -- those are shape problems to
    report, not to paper over with a retry."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "colibri-quality-eval/1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(retry_sleep_base * (attempt + 1))
                last_err = e
                continue
            raise
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(retry_sleep_base * (attempt + 1))
                continue
            raise
    raise last_err  # pragma: no cover -- loop always returns or raises above


def _seed_for(base_seed, *parts):
    """A reproducible sub-seed derived from base_seed and arbitrary string parts, via sha256
    (NOT python's built-in hash(), which is salted per-process by default and would make a
    'fixed seed' not actually fixed across runs/machines)."""
    key = ":".join([str(base_seed)] + [str(p) for p in parts]).encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % (2 ** 31)


def mmlu_scan_rows(page=MMLU_PAGE, sleep_s=0.3, on_progress=None):
    """Full read-only scan of the MMLU-Pro test split via the HF datasets-server rows API.
    Returns (rows_by_category: {category: [ (row_idx, row_dict), ... ]}, total_rows: int).
    Raises RuntimeError with the exact shape seen if the API does not look like what this
    script was written against (requirement a: 'do not substitute another dataset silently')."""
    size = _http_get_json(
        f"{HF_SIZE_URL}?dataset={urllib.parse.quote(MMLU_DATASET, safe='')}"
    )
    splits = size.get("size", {}).get("splits", [])
    split_row = next(
        (s for s in splits if s.get("config") == MMLU_CONFIG and s.get("split") == MMLU_SPLIT),
        None,
    )
    if split_row is None:
        raise RuntimeError(
            f"/size did not list a {MMLU_CONFIG}/{MMLU_SPLIT} split for {MMLU_DATASET}; "
            f"full response: {json.dumps(size)[:2000]}"
        )
    total = split_row["num_rows"]

    # Verify the feature shape on the first page before committing to a full scan.
    first = _http_get_json(
        f"{HF_ROWS_URL}?dataset={urllib.parse.quote(MMLU_DATASET, safe='')}"
        f"&config={MMLU_CONFIG}&split={MMLU_SPLIT}&offset=0&length=1"
    )
    feature_names = {f["name"] for f in first.get("features", [])}
    if not EXPECTED_MMLU_FEATURES.issubset(feature_names):
        raise RuntimeError(
            f"MMLU-Pro features are not what this script expects.\n"
            f"  expected (subset of): {sorted(EXPECTED_MMLU_FEATURES)}\n"
            f"  got:                  {sorted(feature_names)}\n"
            f"  full first-page response: {json.dumps(first)[:3000]}"
        )

    rows_by_cat = {}
    offset = 0
    while offset < total:
        length = min(page, total - offset)
        data = _http_get_json(
            f"{HF_ROWS_URL}?dataset={urllib.parse.quote(MMLU_DATASET, safe='')}"
            f"&config={MMLU_CONFIG}&split={MMLU_SPLIT}&offset={offset}&length={length}"
        )
        for r in data["rows"]:
            row = r["row"]
            cat = row["category"]
            idx = offset + r["row_idx"]
            rows_by_cat.setdefault(cat, []).append((idx, row))
        offset += length
        if on_progress:
            on_progress(offset, total)
        time.sleep(sleep_s)
    return rows_by_cat, total


def mmlu_build_sample(rows_by_cat, seed=MMLU_SAMPLE_SEED, per_category=MMLU_PER_CATEGORY):
    """Deterministic selection: 5 rows per category via a seed derived from (seed, category),
    sorted ascending by row_idx for a stable file. Raises RuntimeError (not a silent fallback)
    if the category count or per-category count is not what was expected."""
    cats = sorted(rows_by_cat.keys())
    if len(cats) != MMLU_EXPECTED_CATEGORIES:
        raise RuntimeError(
            f"expected {MMLU_EXPECTED_CATEGORIES} categories, found {len(cats)}: {cats}"
        )
    items = []
    for cat in cats:
        candidates = rows_by_cat[cat]
        if len(candidates) < per_category:
            raise RuntimeError(
                f"category {cat!r} has only {len(candidates)} rows, need {per_category}"
            )
        rng = random.Random(_seed_for(seed, cat))
        chosen = rng.sample(candidates, per_category)
        chosen.sort(key=lambda t: t[0])
        for idx, row in chosen:
            answer = row["answer"]
            options = row["options"]
            if not (isinstance(answer, str) and len(answer) == 1 and answer in LETTERS):
                raise RuntimeError(f"row {idx}: answer {answer!r} is not a single A-J letter")
            letter_pos = LETTERS.index(answer)
            if letter_pos >= len(options):
                raise RuntimeError(
                    f"row {idx}: answer {answer!r} is out of range for {len(options)} options"
                )
            items.append({
                "question_id": row["question_id"],
                "row_idx": idx,
                "category": cat,
                "question": row["question"],
                "options": options,
                "answer": answer,
            })
    if len(items) != MMLU_EXPECTED_CATEGORIES * per_category:
        raise RuntimeError(f"expected {MMLU_EXPECTED_CATEGORIES * per_category} items, "
                            f"built {len(items)}")
    return items


def cmd_fetch(args):
    print(f"=== fetch: scanning {MMLU_DATASET} ({MMLU_CONFIG}/{MMLU_SPLIT}) via "
          f"datasets-server, page={MMLU_PAGE}")

    def progress(done, total):
        if done % 1000 == 0 or done == total:
            print(f"  scanned {done}/{total} rows", file=sys.stderr)

    try:
        rows_by_cat, total = mmlu_scan_rows(on_progress=progress)
    except Exception as e:
        print(f"FATAL: could not scan {MMLU_DATASET}: {e}", file=sys.stderr)
        return 1
    print(f"  total rows: {total}, categories seen: {sorted(rows_by_cat.keys())} "
          f"({len(rows_by_cat)})")
    try:
        items = mmlu_build_sample(rows_by_cat, seed=args.seed)
    except Exception as e:
        print(f"FATAL: sample selection did not match expectations: {e}", file=sys.stderr)
        return 1

    out = {
        "dataset": MMLU_DATASET,
        "config": MMLU_CONFIG,
        "split": MMLU_SPLIT,
        "total_rows_in_split": total,
        "seed": args.seed,
        "per_category": MMLU_PER_CATEGORY,
        "categories": sorted(rows_by_cat.keys()),
        "items": items,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")

    by_cat = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it["question_id"])
    print(f"=== wrote {args.out}: {len(items)} items, {len(by_cat)} categories x "
          f"{MMLU_PER_CATEGORY}")
    for cat in sorted(by_cat):
        print(f"  {cat}: question_ids={by_cat[cat]}")
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"=== sha256({args.out}) = {digest}")
    return 0


def mmlu_load_sample(path):
    with open(path) as f:
        data = json.load(f)
    items = data["items"]
    if len(items) != MMLU_EXPECTED_CATEGORIES * MMLU_PER_CATEGORY:
        raise RuntimeError(f"{path}: expected "
                            f"{MMLU_EXPECTED_CATEGORIES * MMLU_PER_CATEGORY} items, "
                            f"found {len(items)}")
    return items


# ----------------------------------------------------------------------------------- prompts --
def mmlu_prompt(item):
    """Zero-shot CoT, lettered A-J, identical bytes for every model (requirement b)."""
    lines = [f"Question: {item['question']}", "Options:"]
    for i, opt in enumerate(item["options"]):
        lines.append(f"{LETTERS[i]}. {opt}")
    lines.append("")
    lines.append(
        "Think step by step, then finish your answer with \"Answer: X\" where X is the "
        "letter of the correct option."
    )
    return "\n".join(lines)


ANSWER_RE = re.compile(r"Answer\s*:?\s*[\*\(\[\s]*([A-J])\b", re.IGNORECASE)


def parse_mmlu_answer(content):
    """Last 'Answer: X' in content, tolerant of bold (**) and parentheses/brackets around the
    letter. Returns the upper-case letter, or None if nothing matched."""
    if not content:
        return None
    matches = ANSWER_RE.findall(content)
    if not matches:
        return None
    return matches[-1].upper()


# ----------------------------------------------------------------------------------- Part B --
NEEDLE_OBJECTS = [
    "Valdris pump", "Kestrel turbine", "Halcyon reactor", "Meridian valve",
    "Osprey compressor", "Tamsin coil", "Brackwater manifold", "Lyric injector",
]
NEEDLE_SEED = 20260921
NEEDLE_DEPTHS_TOKENS = [30000, 60000, 120000, 200000]
NEEDLE_CHARS_PER_TOKEN = 2.9    # brief's own figure, "Qwen's tokenizer" -- one constant used
                                 # for every arm so the depths are comparable, not a claim that
                                 # it is exact for every model's own tokenizer
NEEDLE_MAX_TOKENS = 1500


def needle_codes_for_depth(depth_tokens, seed=NEEDLE_SEED):
    """8 made-up 6-digit codes, one per NEEDLE_OBJECTS, deterministic per (seed, depth) so
    every depth gets distinct codes (a model cannot carry a code over from a shallower run of
    the same document prefix) while a re-run reproduces the same file."""
    rng = random.Random(_seed_for(seed, "needle", depth_tokens))
    return {name: f"{rng.randint(0, 999999):06d}" for name in NEEDLE_OBJECTS}


def build_haystack(corpus_text, depth_tokens, chars_per_token=NEEDLE_CHARS_PER_TOKEN,
                    seed=NEEDLE_SEED):
    """Deterministic haystack: the corpus's own first depth_chars characters (capped at what
    the corpus has), with 8 needle sentences spliced in at evenly spread positions (snapped to
    the next newline so a needle never lands mid-table-row). Returns (haystack, codes, depth_chars).
    """
    depth_chars = min(int(round(depth_tokens * chars_per_token)), len(corpus_text))
    base = corpus_text[:depth_chars]
    codes = needle_codes_for_depth(depth_tokens, seed=seed)
    n = len(NEEDLE_OBJECTS)
    positions = [int(round((i + 1) * depth_chars / (n + 1))) for i in range(n)]
    pairs = sorted(zip(positions, NEEDLE_OBJECTS), key=lambda p: p[0])

    parts = []
    prev = 0
    for pos, name in pairs:
        pos = min(max(pos, prev), len(base))
        nl = base.find("\n", pos)
        cut = nl + 1 if nl != -1 else pos
        parts.append(base[prev:cut])
        parts.append(f"\n\nThe calibration code of the {name} is {codes[name]}.\n\n")
        prev = cut
    parts.append(base[prev:])
    return "".join(parts), codes, depth_chars


def needle_prompt(haystack):
    names_list = "\n".join(f"- {n}" for n in NEEDLE_OBJECTS)
    return (
        "You will read a long technical document. Hidden within it are short sentences, each "
        "giving a 6-digit code for a named object, in exactly this form: \"The calibration "
        "code of the X is 123456.\"\n\n"
        f"<document>\n{haystack}\n</document>\n\n"
        "List the 6-digit code for each of the following objects, based only on what you read "
        "above. Respond with ONLY a JSON object whose keys are exactly the object names below "
        "and whose values are the 6-digit code strings, and nothing else:\n"
        f"{names_list}"
    )


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_needle_response(content):
    """Extract the first {...} block and parse it as JSON. Returns a dict, or None if nothing
    parses (counts as 0 correct, not a crash)."""
    if not content:
        return None
    m = JSON_OBJ_RE.search(content)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def score_needles(parsed, truth):
    """(num_correct, detail_dict). parsed=None scores 0 with every detail unmatched."""
    detail = {}
    correct = 0
    for name, code in truth.items():
        got = None
        if isinstance(parsed, dict) and name in parsed:
            got = str(parsed[name])
        ok = got == code
        detail[name] = {"expected": code, "got": got, "ok": ok}
        if ok:
            correct += 1
    return correct, detail


# --------------------------------------------------------------------------------- HTTP send --
CONTEXT_ERROR_HINTS = (
    "context", "n_ctx", "too many tokens", "exceed", "maximum context",
)


def send_chat(url, key, model_id, prompt, max_tokens, timeout=10800, retries=2,
              temperature=0.0):
    """POST one chat completion, no tools, no system prompt. Retries (up to `retries` extra
    attempts) only on connection errors or 5xx; never retries a request that got a real
    response (including a 4xx, which is a definitive answer -- e.g. context-length-exceeded).
    Returns a dict: ok, wall_s, and either data (parsed JSON body) or error/http_status."""
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    last_err = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions",
            data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                wall = time.time() - t0
                return {"ok": True, "data": json.loads(raw), "wall_s": wall,
                        "http_status": resp.status}
        except urllib.error.HTTPError as e:
            wall = time.time() - t0
            raw = e.read()
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                data = {"raw": raw.decode("utf-8", "replace")[:2000]}
            if e.code >= 500 and attempt < retries:
                last_err = f"HTTP {e.code}"
                time.sleep(min(30, 2 ** (attempt + 1)))
                continue
            return {"ok": False, "http_status": e.code, "data": data, "wall_s": wall,
                     "error": f"HTTP {e.code}"}
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            wall = time.time() - t0
            if attempt < retries:
                last_err = str(e)
                time.sleep(min(30, 2 ** (attempt + 1)))
                continue
            return {"ok": False, "error": f"connection error: {e}", "wall_s": wall}
    return {"ok": False, "error": f"exhausted retries: {last_err}"}


def looks_like_context_error(result):
    if result.get("ok"):
        return False
    if result.get("http_status") not in (400, 413, 422):
        return False
    blob = json.dumps(result.get("data", {})).lower()
    return any(h in blob for h in CONTEXT_ERROR_HINTS)


def extract_choice(result):
    """(content, reasoning_content, finish_reason, completion_tokens, prompt_tokens) from a
    successful chat-completion response. `content` only -- CLAUDE.md/the brief: never parse the
    answer out of reasoning_content."""
    data = result["data"]
    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content")
    if isinstance(content, list):  # some servers return content parts; join the text ones
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = msg.get("reasoning_content")
    finish_reason = choice.get("finish_reason")
    usage = data.get("usage", {}) or {}
    return content, reasoning, finish_reason, usage.get("completion_tokens"), \
        usage.get("prompt_tokens")


# ---------------------------------------------------------------------------- identity check --
EXPECT_SUBSTR = {
    "qwen38": "Qwen3.8-Flash-Next",
    "deepseek": "DeepSeek-V4-Flash",
    "glm-llama": "GLM-5.3-Flash-UD-IQ4_XS",
}


def verify_expect(expect, url, key):
    """Refuse (raise SystemExit) if the server at `url` is not plausibly the model named by
    `expect`. See the module docstring for exactly what each branch reads and why."""
    if expect in EXPECT_SUBSTR:
        req = urllib.request.Request(
            url.rstrip("/") + "/props",
            headers={"Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                props = json.loads(r.read())
        except Exception as e:
            raise SystemExit(f"REFUSED: --expect {expect} but GET /props failed: {e}")
        model_path = props.get("model_path", "")
        want = EXPECT_SUBSTR[expect]
        if want not in model_path:
            raise SystemExit(
                f"REFUSED: --expect {expect} but /props model_path={model_path!r} does not "
                f"contain {want!r}"
            )
        print(f"[identity] --expect {expect}: /props model_path={model_path!r} OK")
    elif expect == "glm":
        gw = subprocess.run(["pgrep", "-f", "openai_[s]erver.py"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if gw.returncode != 0:
            raise SystemExit("REFUSED: --expect glm but openai_[s]erver.py is not running")
        eng = subprocess.run(["ps", "-C", "glm53", "-o", "stat="],
                              capture_output=True, text=True)
        stats = [s.strip() for s in eng.stdout.splitlines() if s.strip()]
        if not any(not s.startswith("Z") for s in stats):
            raise SystemExit(
                f"REFUSED: --expect glm but no non-zombie glm53 process (ps stat rows: {stats})"
            )
        print(f"[identity] --expect glm: gateway process present, glm53 stat rows={stats} OK")
    else:
        raise SystemExit(f"unknown --expect {expect!r}")


# --------------------------------------------------------------------------------- jsonl I/O --
def load_done_ids(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in rec:
                done.add(rec["id"])
    return done


def append_record(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------------- run cmd --
def run_mmlu_item(url, key, model_id, item, timeout, retries, max_tokens):
    prompt = mmlu_prompt(item)
    result = send_chat(url, key, model_id, prompt, max_tokens, timeout=timeout,
                        retries=retries)
    rec = {
        "id": f"mmlu:{item['question_id']}",
        "type": "mmlu",
        "question_id": item["question_id"],
        "category": item["category"],
        "correct_letter": item["answer"],
        "wall_s": result.get("wall_s"),
    }
    if not result.get("ok"):
        rec["completed"] = False
        rec["error"] = result.get("error")
        rec["http_status"] = result.get("http_status")
        return rec, looks_like_context_error(result) or result.get("http_status") is not None
    content, reasoning, finish_reason, completion_tokens, prompt_tokens = extract_choice(result)
    parsed = parse_mmlu_answer(content)
    truncated = finish_reason == "length"
    unparsed = parsed is None
    correct = (parsed == item["answer"]) and not unparsed
    rec.update({
        "completed": True,
        "parsed_letter": parsed,
        "correct": correct,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "truncated": truncated,
        "unparsed": unparsed,
        "content_len": len(content) if content else 0,
    })
    return rec, True


def run_needle_item(url, key, model_id, depth_tokens, corpus_text, timeout, retries):
    haystack, codes, depth_chars = build_haystack(corpus_text, depth_tokens)
    prompt = needle_prompt(haystack)
    result = send_chat(url, key, model_id, prompt, NEEDLE_MAX_TOKENS, timeout=timeout,
                        retries=retries)
    rec = {
        "id": f"needle:depth{depth_tokens}",
        "type": "needle",
        "depth_tokens": depth_tokens,
        "depth_chars": depth_chars,
        "wall_s": result.get("wall_s"),
    }
    if not result.get("ok"):
        rec["completed"] = False
        rec["error"] = result.get("error")
        rec["http_status"] = result.get("http_status")
        rec["context_error"] = looks_like_context_error(result)
        return rec, rec["context_error"] or result.get("http_status") is not None
    content, reasoning, finish_reason, completion_tokens, prompt_tokens = extract_choice(result)
    parsed = parse_needle_response(content)
    score, detail = score_needles(parsed, codes)
    rec.update({
        "completed": True,
        "score": score,
        "score_max": len(NEEDLE_OBJECTS),
        "detail": detail,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "truncated": finish_reason == "length",
        "unparsed": parsed is None,
        "context_error": False,
    })
    return rec, True


def cmd_run(args):
    key = open(args.key_file).read().strip()
    verify_expect(args.expect, args.url, key)

    parts = set(args.parts.split(","))
    out_path = args.out
    done = load_done_ids(out_path)
    print(f"=== run: expect={args.expect} url={args.url} model_id={args.model_id} "
          f"out={out_path} already-done={len(done)}")

    any_incomplete = False

    if "mmlu" in parts:
        items = mmlu_load_sample(args.mmlu)
        for item in items:
            item_id = f"mmlu:{item['question_id']}"
            if item_id in done:
                continue
            print(f"[mmlu] {item_id} ({item['category']}) ...", end="", flush=True)
            rec, final = run_mmlu_item(args.url, key, args.model_id, item, args.timeout,
                                        args.retries, args.mmlu_max_tokens)
            if final:
                append_record(out_path, rec)
                done.add(item_id)
                status = "OK" if rec.get("completed") else "FAILED(final)"
                extra = ""
                if rec.get("completed"):
                    extra = f" correct={rec['correct']} letter={rec['parsed_letter']}"
                print(f" {status}{extra} wall_s={rec.get('wall_s')}")
            else:
                any_incomplete = True
                print(f" NOT-FINISHED (will retry on resume): {rec.get('error')}")

    if "needle" in parts:
        corpus_text = open(args.corpus).read()
        corpus_sha = hashlib.sha256(corpus_text.encode()).hexdigest()
        print(f"[needle] corpus={args.corpus} sha256={corpus_sha} chars={len(corpus_text)}")
        depths = [d for d in NEEDLE_DEPTHS_TOKENS if d <= args.max_context]
        skipped = [d for d in NEEDLE_DEPTHS_TOKENS if d > args.max_context]
        if skipped:
            print(f"[needle] depths skipped (over --max-context {args.max_context}): "
                  f"{skipped}")
        for depth in depths:
            item_id = f"needle:depth{depth}"
            if item_id in done:
                continue
            print(f"[needle] {item_id} ...", end="", flush=True)
            rec, final = run_needle_item(args.url, key, args.model_id, depth, corpus_text,
                                          args.timeout, args.retries)
            if final:
                append_record(out_path, rec)
                done.add(item_id)
                status = "OK" if rec.get("completed") else "FAILED(final)"
                extra = ""
                if rec.get("completed"):
                    extra = f" score={rec['score']}/{rec['score_max']}"
                print(f" {status}{extra} wall_s={rec.get('wall_s')}")
            else:
                any_incomplete = True
                print(f" NOT-FINISHED (will retry on resume): {rec.get('error')}")

    print(f"=== run done: out={out_path}")
    return 1 if any_incomplete else 0


# ----------------------------------------------------------------------------- summarize cmd --
def wilson_ci(correct, n, z=1.959963985):
    if n == 0:
        return (0.0, 0.0, 0.0)
    phat = correct / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = (center - adj) / denom
    hi = (center + adj) / denom
    return (phat, max(0.0, lo), min(1.0, hi))


def mcnemar_exact_p(b, c):
    """Exact two-sided McNemar p-value (binomial, p=0.5), stdlib math only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p_tail)


def load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def summarize_model(label, recs):
    mmlu = [r for r in recs if r.get("type") == "mmlu"]
    needles = [r for r in recs if r.get("type") == "needle"]
    completed = [r for r in mmlu if r.get("completed")]
    n = len(completed)
    correct = sum(1 for r in completed if r.get("correct"))
    truncated = sum(1 for r in completed if r.get("truncated"))
    unparsed = sum(1 for r in completed if r.get("unparsed"))
    failed = len(mmlu) - n
    ctoks = [r["completion_tokens"] for r in completed if r.get("completion_tokens") is not None]
    mean_ctoks = sum(ctoks) / len(ctoks) if ctoks else 0.0
    total_wall = sum(r.get("wall_s") or 0 for r in mmlu) + sum(r.get("wall_s") or 0 for r in needles)
    phat, lo, hi = wilson_ci(correct, n)

    print(f"--- {label}: MMLU-Pro n={n}/{len(mmlu)} (n excludes {failed} not-finished items)")
    print(f"    accuracy={correct}/{n} = {phat*100:.1f}%  95% Wilson CI [{lo*100:.1f}, "
          f"{hi*100:.1f}]")
    print(f"    truncated(finish_reason=length)={truncated}  unparsed={unparsed}  "
          f"mean_completion_tokens={mean_ctoks:.0f}")
    print(f"    total wall time (mmlu+needle) = {total_wall:.1f} s")

    cats = sorted({r["category"] for r in completed} | {r["category"] for r in mmlu})
    print(f"    per-category:")
    for cat in cats:
        cn = [r for r in completed if r["category"] == cat]
        cc = sum(1 for r in cn if r["correct"])
        ctot = len([r for r in mmlu if r["category"] == cat])
        print(f"      {cat:22s} {cc}/{len(cn)} correct (of {ctot} attempted)")

    if needles:
        print(f"    needle depths:")
        for r in sorted(needles, key=lambda r: r.get("depth_tokens", 0)):
            if r.get("completed"):
                print(f"      depth={r['depth_tokens']:>7} tokens  score={r['score']}/"
                      f"{r['score_max']}  wall_s={r.get('wall_s'):.1f}")
            else:
                reason = "context_error" if r.get("context_error") else r.get("error")
                print(f"      depth={r['depth_tokens']:>7} tokens  NOT COMPLETED ({reason})")
    return {"n": n, "correct": correct, "phat": phat, "lo": lo, "hi": hi,
            "completed_by_id": {r["id"]: r for r in completed}}


def summarize_pair(label_a, sum_a, label_b, sum_b):
    shared = sorted(set(sum_a["completed_by_id"]) & set(sum_b["completed_by_id"]))
    if not shared:
        print(f"--- {label_a} vs {label_b}: no shared completed mmlu items -- SKIPPED")
        return
    both_right = only_a = only_b = both_wrong = 0
    for item_id in shared:
        ra = sum_a["completed_by_id"][item_id]["correct"]
        rb = sum_b["completed_by_id"][item_id]["correct"]
        if ra and rb:
            both_right += 1
        elif ra and not rb:
            only_a += 1
        elif rb and not ra:
            only_b += 1
        else:
            both_wrong += 1
    n = len(shared)
    acc_a = (both_right + only_a) / n
    acc_b = (both_right + only_b) / n
    p = mcnemar_exact_p(only_a, only_b)
    print(f"--- {label_a} vs {label_b}: n_shared={n}")
    print(f"    both_right={both_right} only_{label_a}={only_a} only_{label_b}={only_b} "
          f"both_wrong={both_wrong}")
    print(f"    accuracy: {label_a}={acc_a*100:.1f}%  {label_b}={acc_b*100:.1f}%  "
          f"diff={100*(acc_a-acc_b):+.1f} pts")
    print(f"    McNemar exact two-sided p = {p:.4f}")
    diff_pts = abs(100 * (acc_a - acc_b))
    if p >= 0.05 or diff_pts < 10:
        print(f"    VERDICT: NOT DISTINGUISHABLE at n={n} -- {diff_pts:.1f}-point apparent "
              f"difference, this sample size cannot reliably resolve differences under "
              f"roughly 10 points (McNemar p={p:.4f})")
    else:
        print(f"    VERDICT: SEPARATED (p={p:.4f} < 0.05, diff={diff_pts:.1f} pts >= 10)")


def cmd_summarize(args):
    models = {}
    for spec in args.model:
        if "=" not in spec:
            print(f"FATAL: --model must be NAME=PATH, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        models[name] = load_jsonl(path)

    summaries = {}
    for name, recs in models.items():
        summaries[name] = summarize_model(name, recs)

    print()
    print("=== paired comparisons (shared MMLU-Pro items only; McNemar exact, stdlib math) ===")
    print("NOTE: with 70 MMLU-Pro items total, differences under roughly 10 points are not "
          "reliably distinguishable from noise -- every verdict below says so explicitly "
          "rather than ranking models on it.")
    names = sorted(models.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            summarize_pair(names[i], summaries[names[i]], names[j], summaries[names[j]])
    return 0


# --------------------------------------------------------------------------------- argparse --
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="write the frozen MMLU-Pro sample")
    pf.add_argument("--out", default=os.path.expanduser("~/bench/f11_quality/mmlu_pro_sample.json"))
    pf.add_argument("--seed", type=int, default=MMLU_SAMPLE_SEED)
    pf.set_defaults(func=cmd_fetch)

    pr = sub.add_parser("run", help="evaluate one server, append to its jsonl")
    pr.add_argument("--expect", required=True, choices=["qwen38", "deepseek", "glm", "glm-llama"])
    pr.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8081")
    pr.add_argument("--key-file", required=True)
    pr.add_argument("--model-id", required=True, help="the `model` field to send, e.g. glm-5.3-flash")
    pr.add_argument("--out", required=True, help="jsonl output path (appended, resumable)")
    pr.add_argument("--mmlu", default=os.path.expanduser("~/bench/f11_quality/mmlu_pro_sample.json"))
    pr.add_argument("--corpus", default=DEFAULT_CORPUS)
    pr.add_argument("--parts", default="mmlu,needle", help="comma list: mmlu,needle")
    pr.add_argument("--max-context", type=int, required=True,
                     help="token budget for needle depths; only depths <= this run")
    pr.add_argument("--timeout", type=float, default=10800.0, help="seconds per HTTP request")
    pr.add_argument("--retries", type=int, default=2,
                     help="extra attempts on connection errors/5xx only")
    pr.add_argument("--mmlu-max-tokens", type=int, default=3500)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("summarize", help="summarize one or more models' jsonl files")
    ps.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ps.set_defaults(func=cmd_summarize)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
