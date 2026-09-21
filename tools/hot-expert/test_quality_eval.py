#!/usr/bin/env python3
"""test_quality_eval.py -- unit tests for quality_eval.py's parser/grader/needle-builder/
statistics, run entirely locally (no network, no rig). Plain stdlib unittest:

    python3 tools/hot-expert/test_quality_eval.py -v

Network-shaped behaviour (send_chat's retry logic, verify_expect's /props read) is exercised
against a real in-process http.server, not mocks, per CLAUDE.md/the brief's instruction to
unit-test "with a fake in-process HTTP server or canned responses".
"""
import http.server
import json
import math
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality_eval as q


class TestMmluPrompt(unittest.TestCase):
    def test_prompt_letters_and_shape(self):
        item = {
            "question": "What is 2+2?",
            "options": ["3", "4", "5"],
        }
        prompt = q.mmlu_prompt(item)
        self.assertIn("Question: What is 2+2?", prompt)
        self.assertIn("A. 3", prompt)
        self.assertIn("B. 4", prompt)
        self.assertIn("C. 5", prompt)
        self.assertIn('finish your answer with "Answer: X"', prompt)

    def test_prompt_identical_bytes_for_same_item(self):
        item = {"question": "Q", "options": ["a", "b"]}
        self.assertEqual(q.mmlu_prompt(item), q.mmlu_prompt(dict(item)))


class TestParseMmluAnswer(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(q.parse_mmlu_answer("blah blah Answer: C"), "C")

    def test_bold(self):
        self.assertEqual(q.parse_mmlu_answer("reasoning...\n**Answer: D**"), "D")

    def test_parens(self):
        self.assertEqual(q.parse_mmlu_answer("Answer: (H)"), "H")

    def test_brackets_and_lowercase(self):
        self.assertEqual(q.parse_mmlu_answer("answer: [b]"), "B")

    def test_last_wins(self):
        text = "First I thought Answer: B, but on reflection Answer: D."
        self.assertEqual(q.parse_mmlu_answer(text), "D")

    def test_no_match_returns_none(self):
        self.assertIsNone(q.parse_mmlu_answer("The answer is C, definitely."))
        self.assertIsNone(q.parse_mmlu_answer(""))
        self.assertIsNone(q.parse_mmlu_answer(None))

    def test_never_reads_out_of_range_letter(self):
        # K is outside A-J and must not match
        self.assertIsNone(q.parse_mmlu_answer("Answer: K"))


class TestSeedDerivation(unittest.TestCase):
    def test_deterministic(self):
        a = q._seed_for(123, "foo", 456)
        b = q._seed_for(123, "foo", 456)
        self.assertEqual(a, b)

    def test_sensitive_to_each_part(self):
        base = q._seed_for(123, "foo", 456)
        self.assertNotEqual(base, q._seed_for(124, "foo", 456))
        self.assertNotEqual(base, q._seed_for(123, "bar", 456))
        self.assertNotEqual(base, q._seed_for(123, "foo", 457))

    def test_not_pythons_builtin_hash(self):
        # built-in hash() of a str is salted per-process by default (PYTHONHASHSEED) --
        # this function must not reduce to it.
        h = q._seed_for(1, "x")
        self.assertIsInstance(h, int)
        self.assertGreaterEqual(h, 0)


class TestMmluSampleBuild(unittest.TestCase):
    def _fake_rows(self, n_categories=14, per_cat=8):
        rows_by_cat = {}
        qid = 0
        for c in range(n_categories):
            cat = f"cat{c}"
            rows = []
            for i in range(per_cat):
                letters = q.LETTERS[: 4 + (i % 3)]
                answer = letters[i % len(letters)]
                rows.append((qid, {
                    "question_id": qid,
                    "question": f"q{qid}",
                    "options": list(letters),
                    "answer": answer,
                    "category": cat,
                }))
                qid += 1
            rows_by_cat[cat] = rows
        return rows_by_cat

    def test_selects_5_per_category_deterministically(self):
        rows_by_cat = self._fake_rows()
        items1 = q.mmlu_build_sample(rows_by_cat, seed=999)
        items2 = q.mmlu_build_sample(rows_by_cat, seed=999)
        self.assertEqual(len(items1), 14 * 5)
        self.assertEqual([i["question_id"] for i in items1], [i["question_id"] for i in items2])
        by_cat = {}
        for it in items1:
            by_cat.setdefault(it["category"], []).append(it)
        for cat, its in by_cat.items():
            self.assertEqual(len(its), 5)
        # sorted ascending by row_idx (== question_id here) within each category
        for cat, its in by_cat.items():
            ids = [i["question_id"] for i in its]
            self.assertEqual(ids, sorted(ids))

    def test_different_seed_can_change_selection(self):
        rows_by_cat = self._fake_rows()
        items_a = q.mmlu_build_sample(rows_by_cat, seed=1)
        items_b = q.mmlu_build_sample(rows_by_cat, seed=2)
        ids_a = [i["question_id"] for i in items_a]
        ids_b = [i["question_id"] for i in items_b]
        self.assertNotEqual(ids_a, ids_b)

    def test_wrong_category_count_raises(self):
        rows_by_cat = self._fake_rows(n_categories=13)
        with self.assertRaises(RuntimeError):
            q.mmlu_build_sample(rows_by_cat)

    def test_too_few_rows_in_a_category_raises(self):
        rows_by_cat = self._fake_rows(per_cat=3)
        with self.assertRaises(RuntimeError):
            q.mmlu_build_sample(rows_by_cat)

    def test_every_answer_letter_within_its_options(self):
        rows_by_cat = self._fake_rows()
        items = q.mmlu_build_sample(rows_by_cat, seed=42)
        for it in items:
            pos = q.LETTERS.index(it["answer"])
            self.assertLess(pos, len(it["options"]))

    def test_bad_answer_letter_raises(self):
        rows_by_cat = self._fake_rows(n_categories=14, per_cat=8)
        # corrupt one row's answer to be out of range of its own options
        cat0 = rows_by_cat["cat0"]
        idx, row = cat0[0]
        row = dict(row)
        row["options"] = ["x", "y"]     # 2 options
        row["answer"] = "F"             # out of range
        cat0[0] = (idx, row)
        with self.assertRaises(RuntimeError):
            q.mmlu_build_sample(rows_by_cat, seed=1)


class TestNeedles(unittest.TestCase):
    CORPUS = ("Line %d of the corpus, nothing special here at all, just filler text.\n" * 2000)

    def test_haystack_deterministic(self):
        h1, c1, d1 = q.build_haystack(self.CORPUS, 100, chars_per_token=3.0)
        h2, c2, d2 = q.build_haystack(self.CORPUS, 100, chars_per_token=3.0)
        self.assertEqual(h1, h2)
        self.assertEqual(c1, c2)
        self.assertEqual(d1, d2)

    def test_haystack_contains_every_needle_sentence(self):
        h, codes, _ = q.build_haystack(self.CORPUS, 5000, chars_per_token=3.0)
        for name, code in codes.items():
            self.assertIn(f"The calibration code of the {name} is {code}.", h)

    def test_haystack_capped_at_corpus_length(self):
        short_corpus = "abc\n" * 10
        h, codes, depth_chars = q.build_haystack(short_corpus, 1_000_000, chars_per_token=3.0)
        self.assertLessEqual(depth_chars, len(short_corpus))
        # every needle sentence must still appear even when capped
        for name, code in codes.items():
            self.assertIn(f"The calibration code of the {name} is {code}.", h)

    def test_different_depths_get_different_codes(self):
        _, codes_30k, _ = q.build_haystack(self.CORPUS * 50, 30000, chars_per_token=2.9)
        _, codes_60k, _ = q.build_haystack(self.CORPUS * 50, 60000, chars_per_token=2.9)
        self.assertNotEqual(codes_30k, codes_60k)

    def test_needle_prompt_lists_every_object(self):
        h, codes, _ = q.build_haystack(self.CORPUS, 1000, chars_per_token=3.0)
        prompt = q.needle_prompt(h)
        for name in q.NEEDLE_OBJECTS:
            self.assertIn(name, prompt)


class TestParseNeedleResponse(unittest.TestCase):
    def test_plain_json(self):
        content = '{"Valdris pump": "123456", "Kestrel turbine": "654321"}'
        parsed = q.parse_needle_response(content)
        self.assertEqual(parsed["Valdris pump"], "123456")

    def test_json_in_prose(self):
        content = 'Here you go:\n```json\n{"Valdris pump": "123456"}\n```\nHope that helps.'
        parsed = q.parse_needle_response(content)
        self.assertEqual(parsed["Valdris pump"], "123456")

    def test_garbage_returns_none(self):
        self.assertIsNone(q.parse_needle_response("no json here at all"))
        self.assertIsNone(q.parse_needle_response(""))
        self.assertIsNone(q.parse_needle_response(None))

    def test_non_object_json_returns_none(self):
        self.assertIsNone(q.parse_needle_response("[1, 2, 3]"))


class TestScoreNeedles(unittest.TestCase):
    def test_all_correct(self):
        truth = {"a": "111111", "b": "222222"}
        score, detail = q.score_needles({"a": "111111", "b": "222222"}, truth)
        self.assertEqual(score, 2)
        self.assertTrue(all(d["ok"] for d in detail.values()))

    def test_partial_and_missing(self):
        truth = {"a": "111111", "b": "222222"}
        score, detail = q.score_needles({"a": "111111"}, truth)
        self.assertEqual(score, 1)
        self.assertFalse(detail["b"]["ok"])
        self.assertIsNone(detail["b"]["got"])

    def test_none_parsed_scores_zero(self):
        truth = {"a": "111111", "b": "222222"}
        score, detail = q.score_needles(None, truth)
        self.assertEqual(score, 0)

    def test_int_value_coerced_to_string(self):
        # a model might emit the code as a bare number instead of a zero-padded string
        truth = {"a": "000123"}
        score, detail = q.score_needles({"a": 123}, truth)
        # "123" != "000123": this is intentionally strict (see module docstring: exact match)
        self.assertEqual(score, 0)
        score2, _ = q.score_needles({"a": "000123"}, truth)
        self.assertEqual(score2, 1)


class TestWilsonCI(unittest.TestCase):
    def test_zero_n(self):
        self.assertEqual(q.wilson_ci(0, 0), (0.0, 0.0, 0.0))

    def test_known_value_half(self):
        # n=100, x=50 -> phat=0.5, a textbook 95% Wilson interval is about [0.404, 0.596]
        phat, lo, hi = q.wilson_ci(50, 100)
        self.assertAlmostEqual(phat, 0.5, places=6)
        self.assertAlmostEqual(lo, 0.404, places=2)
        self.assertAlmostEqual(hi, 0.596, places=2)

    def test_bounds_within_0_1(self):
        for correct, n in [(0, 10), (10, 10), (7, 70)]:
            phat, lo, hi = q.wilson_ci(correct, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, phat)
            self.assertLessEqual(phat, hi)


class TestMcNemar(unittest.TestCase):
    def test_zero_discordant_is_certain(self):
        self.assertEqual(q.mcnemar_exact_p(0, 0), 1.0)

    def test_symmetric(self):
        self.assertAlmostEqual(q.mcnemar_exact_p(3, 7), q.mcnemar_exact_p(7, 3), places=9)

    def test_matches_hand_computed_binomial(self):
        # b=1, c=9 (n=10): two-sided exact p = 2 * P(X<=1 | n=10,p=0.5)
        n, k = 10, 1
        p_tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        expected = min(1.0, 2 * p_tail)
        self.assertAlmostEqual(q.mcnemar_exact_p(1, 9), expected, places=9)

    def test_large_discordance_is_significant(self):
        # 0 vs 20 discordant pairs, all favouring one model: p should be tiny
        p = q.mcnemar_exact_p(0, 20)
        self.assertLess(p, 0.001)

    def test_balanced_discordance_is_not_significant(self):
        p = q.mcnemar_exact_p(10, 10)
        self.assertGreater(p, 0.5)


class TestSummarizePair(unittest.TestCase):
    """Exercises the 'not distinguishable' messaging path end to end (capturing stdout)."""

    def _fake_summary(self, ids_correct):
        by_id = {i: {"correct": ok} for i, ok in ids_correct.items()}
        n = len(by_id)
        correct = sum(1 for v in by_id.values() if v["correct"])
        phat, lo, hi = q.wilson_ci(correct, n)
        return {"n": n, "correct": correct, "phat": phat, "lo": lo, "hi": hi,
                "completed_by_id": by_id}

    def test_small_difference_says_not_distinguishable(self):
        import io
        import contextlib
        ids = {f"mmlu:{i}": (i % 10 != 0) for i in range(70)}       # 90% for both
        ids_b = dict(ids)
        ids_b["mmlu:5"] = not ids_b["mmlu:5"]                        # one flip: tiny diff
        sum_a = self._fake_summary(ids)
        sum_b = self._fake_summary(ids_b)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            q.summarize_pair("A", sum_a, "B", sum_b)
        out = buf.getvalue()
        self.assertIn("NOT DISTINGUISHABLE", out)

    def test_large_clean_difference_says_separated(self):
        import io
        import contextlib
        ids_a = {f"mmlu:{i}": True for i in range(70)}     # 100%
        ids_b = {f"mmlu:{i}": (i % 2 == 0) for i in range(70)}  # 50%
        sum_a = self._fake_summary(ids_a)
        sum_b = self._fake_summary(ids_b)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            q.summarize_pair("A", sum_a, "B", sum_b)
        out = buf.getvalue()
        self.assertIn("SEPARATED", out)


# ------------------------------------------------------------------- in-process HTTP server --
class _FakeHandler(http.server.BaseHTTPRequestHandler):
    """Canned responses, keyed on self.server.script: a list of (status, body_dict) popped in
    order for POST /v1/chat/completions; GET /props always returns self.server.props_body."""

    def log_message(self, fmt, *args):
        pass  # keep test output quiet

    def do_GET(self):
        if self.path == "/props":
            body = json.dumps(self.server.props_body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain the body
        self.server.requests_seen += 1
        if not self.server.script:
            self.send_response(500)
            self.end_headers()
            return
        status, body = self.server.script.pop(0)
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _FakeServer(http.server.HTTPServer):
    pass


class TestSendChatAgainstFakeServer(unittest.TestCase):
    def _start(self, script, props_body=None):
        server = _FakeServer(("127.0.0.1", 0), _FakeHandler)
        server.script = list(script)
        server.props_body = props_body or {}
        server.requests_seen = 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _stop(self, server, thread):
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    def test_successful_first_try(self):
        body = {
            "choices": [{"message": {"content": "Answer: C"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10, "prompt_tokens": 5},
        }
        server, thread = self._start([(200, body)])
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            result = q.send_chat(url, "k", "m", "hi", 16, timeout=5, retries=2)
            self.assertTrue(result["ok"])
            content, reasoning, finish, ctoks, ptoks = q.extract_choice(result)
            self.assertEqual(content, "Answer: C")
            self.assertEqual(finish, "stop")
            self.assertEqual(ctoks, 10)
            self.assertEqual(server.requests_seen, 1)
        finally:
            self._stop(server, thread)

    def test_retries_on_500_then_succeeds(self):
        ok_body = {
            "choices": [{"message": {"content": "Answer: A"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        }
        server, thread = self._start([
            (500, {"error": "boom"}),
            (200, ok_body),
        ])
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            result = q.send_chat(url, "k", "m", "hi", 16, timeout=5, retries=2)
            self.assertTrue(result["ok"])
            self.assertEqual(server.requests_seen, 2)
        finally:
            self._stop(server, thread)

    def test_no_retry_on_400(self):
        server, thread = self._start([
            (400, {"error": {"message": "context length exceeded (n_ctx=100)"}}),
        ])
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            result = q.send_chat(url, "k", "m", "hi", 16, timeout=5, retries=2)
            self.assertFalse(result["ok"])
            self.assertEqual(result["http_status"], 400)
            self.assertEqual(server.requests_seen, 1)   # no retry burned on a 4xx
            self.assertTrue(q.looks_like_context_error(result))
        finally:
            self._stop(server, thread)

    def test_exhausts_retries_on_repeated_500(self):
        server, thread = self._start([
            (500, {"error": "1"}), (500, {"error": "2"}), (500, {"error": "3"}),
        ])
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            result = q.send_chat(url, "k", "m", "hi", 16, timeout=5, retries=2)
            self.assertFalse(result["ok"])
            self.assertEqual(server.requests_seen, 3)   # 1 try + 2 retries
        finally:
            self._stop(server, thread)

    def test_verify_expect_qwen38_matches_model_path(self):
        server, thread = self._start(
            [], props_body={"model_path": "/home/ronald/models/Qwen3.8-Flash-Next/UD-IQ4_XS/x.gguf"},
        )
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            q.verify_expect("qwen38", url, "k")  # must not raise
        finally:
            self._stop(server, thread)

    def test_verify_expect_refuses_on_mismatch(self):
        server, thread = self._start(
            [], props_body={"model_path": "/home/ronald/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/x.gguf"},
        )
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with self.assertRaises(SystemExit):
                q.verify_expect("qwen38", url, "k")
        finally:
            self._stop(server, thread)


class TestJsonlIO(unittest.TestCase):
    def test_append_and_load_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            q.append_record(path, {"id": "mmlu:1", "correct": True})
            q.append_record(path, {"id": "mmlu:2", "correct": False})
            done = q.load_done_ids(path)
            self.assertEqual(done, {"mmlu:1", "mmlu:2"})
            recs = q.load_jsonl(path)
            self.assertEqual(len(recs), 2)

    def test_load_done_ids_on_missing_file(self):
        self.assertEqual(q.load_done_ids("/nonexistent/path/out.jsonl"), set())

    def test_load_done_ids_skips_corrupt_lines(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            with open(path, "w") as f:
                f.write("not json\n")
                f.write(json.dumps({"id": "mmlu:1"}) + "\n")
            self.assertEqual(q.load_done_ids(path), {"mmlu:1"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
