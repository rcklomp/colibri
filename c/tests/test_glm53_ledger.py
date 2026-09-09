#!/usr/bin/env python3
"""P9 -- the conversation ledger: the match/decide state machine, at its own level.

The spec says where the bugs will be: "the match/decide table is where the bugs
will be, so it is tested at that level, not through the engine". This file is
that level. No model, no engine, no gateway socket -- string algebra over the
renderer, which is the only thing the ledger manipulates.

Two properties are asserted over and over, because they are the ones that make
the mechanism safe rather than merely fast:

  1. THE PREFIX INVARIANT.  For a continuation,
         render(turn N+1)  startswith  prompt(turn N) + raw generation of turn N
     which is exactly the sequence the KV slot holds. When it does not hold, the
     ledger must say `ledger=broken` and fall back rather than submit a prompt
     that claims a prefix it does not have.

  2. THE SAFETY RULE.  The ledger decorates the client's transcript and never
     replaces it: one rendered piece per message the client sent, never one
     more, never one fewer, and never a piece from another conversation. It
     never injects a turn the client did not send, and it never changes what
     the user sees -- only what is sent BACK to the engine.

The cases the item was given: exact continuation, client-shorter (regenerate,
edit, branch), diverged mid-history, unknown key, and two conversations
colliding on the derived key (Open WebUI pops `metadata` before forwarding, so
the key IS derived and collisions are reachable by construction).
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import openai_server as S


SYSTEM = "You are a helpful assistant."
MEM_A = "<memory_context>\n- likes espresso\n- lives in Rome\n</memory_context>"
MEM_B = "<memory_context>\n- lives in Rome\n- likes espresso\n</memory_context>"
TOOLS = [{"type": "function", "function": {"name": "get_time", "description": "the time",
                                           "parameters": {"type": "object", "properties": {}}}}]
GEN = "<|assistant|><think>"


def raw_reply(thought, answer):
    """What the engine emits after the prompt's own `<think>`: reasoning, the
    closing marker, then the answer -- with the newline and the trailing space
    that `content.strip()` throws away and the KV slot does not."""
    return f"{thought}</think>\n{answer} "


RAW1 = raw_reply("The user wants one word. I will answer OK.", "OK")
RAW2 = raw_reply("A clear sky is blue.", "Blue")
RAW3 = raw_reply("Grass is green.", "Green")


class LedgerCase(unittest.TestCase):
    """A two-conversation fixture driven exactly as the gateway drives it."""

    def setUp(self):
        S._ledger_cache.clear()
        S._prefix_pin_cache.clear()
        S._reply_pin_cache.clear()
        self.arch = patch.object(S, "ARCH", "glm53")
        self.arch.start()
        self.addCleanup(self.arch.stop)
        self.env = patch.dict(S.os.environ, {"COLI_LEDGER": "1", "COLI_LEDGER_STRICT": "0",
                                             "COLI_REQ_LOG": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.kv_slots = 4

    # ---- the gateway's two calls, in one place --------------------------
    def render(self, messages, tools=None, effort=None, thinking=False):
        return S.ledger_render(messages, thinking, effort, tools, None, self.kv_slots)

    def record(self, plan, raw, prompt_tokens=100, gen=None, length_limited=False,
               cancelled=False):
        gen = len(raw.split()) if gen is None else gen
        stats = {"prompt_tokens": prompt_tokens, "completion_tokens": gen,
                 "length_limited": length_limited, "reused": 0}
        S.ledger_record(plan, raw, S.reply_pin_visible(raw, False), stats,
                        cancelled=cancelled)
        return stats

    def turn(self, messages, raw, **kw):
        """Render a request, then record its generation. Returns (prompt, plan)."""
        prompt, plan = self.render(messages, tools=kw.pop("tools", None),
                                   effort=kw.pop("effort", None),
                                   thinking=kw.pop("thinking", False))
        self.record(plan, raw, **kw)
        return prompt, plan

    # ---- transcripts ----------------------------------------------------
    def t1(self, system=SYSTEM, user="Say OK."):
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def visible(self, raw):
        return S.reply_pin_visible(raw, False).strip()

    def extend(self, messages, raw, user):
        """What the client sends next: OUR visible reply (stripped, as clients
        store it) followed by the new question."""
        return messages + [{"role": "assistant", "content": self.visible(raw)},
                           {"role": "user", "content": user}]

    # ================================================================== 1
    # exact continuation
    # ==================================================================
    def test_exact_continuation_reproduces_the_engines_tokens(self):
        m1 = self.t1()
        p1, plan1 = self.turn(m1, RAW1)
        self.assertTrue(p1.endswith(GEN))
        m2 = self.extend(m1, RAW1, "What colour is a clear sky?")
        p2, plan2 = self.render(m2)
        self.assertEqual(plan2.state, "continuation")
        self.assertTrue(p2.startswith(p1 + RAW1),
                        f"\n  want prefix ...{(p1 + RAW1)[-70:]!r}\n  got     ...{p2[len(p1) - 20:len(p1) + 60]!r}")
        self.assertEqual(p2, p1 + RAW1 + "<|user|>What colour is a clear sky?" + GEN)

    def test_a_third_turn_keeps_extending(self):
        m1 = self.t1()
        p1, _ = self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky colour?")
        p2, _ = self.turn(m2, RAW2)
        m3 = self.extend(m2, RAW2, "And grass?")
        p3, plan3 = self.render(m3)
        self.assertEqual(plan3.state, "continuation")
        self.assertTrue(p3.startswith(p1 + RAW1))
        self.assertTrue(p3.startswith(p2 + RAW2))

    def test_continuation_survives_a_reranked_memory_block(self):
        """P7's case. Open WebUI rebuilds <memory_context> from a vector search
        over the last seven user messages, so its item ORDER moves every turn."""
        m1 = self.t1(system=SYSTEM + "\n\n" + MEM_A)
        p1, _ = self.turn(m1, RAW1)
        m2 = self.extend(self.t1(system=SYSTEM + "\n\n" + MEM_B), RAW1, "Sky colour?")
        p2, plan2 = self.render(m2)
        self.assertEqual(plan2.state, "continuation")
        self.assertTrue(p2.startswith(p1 + RAW1))
        self.assertIn("likes espresso\n- lives in Rome", p2)      # turn 1's ORDER
        self.assertEqual(p2.count("<memory_context>"), 1)

    def test_continuation_survives_a_client_that_strips_the_reply(self):
        """The synthetic fourth behaviour: neither existing pin covers it.

        `RAW1` ends in a space and begins inside the reasoning block. A client
        that stores `content.strip()` -- or trims trailing whitespace of its own
        accord -- hands back neither."""
        m1 = self.t1()
        p1, _ = self.turn(m1, RAW1)
        for stored in ("OK", "OK  ", "\nOK\n", " OK\t"):
            with self.subTest(stored=stored):
                S._ledger_cache.clear()
                self.turn(m1, RAW1)
                m2 = self.t1() + [{"role": "assistant", "content": stored},
                                  {"role": "user", "content": "Sky?"}]
                p2, plan2 = self.render(m2)
                self.assertEqual(plan2.state, "continuation")
                self.assertTrue(p2.startswith(p1 + RAW1))

    def test_continuation_survives_a_client_that_strips_the_question(self):
        """The same class on the USER side, which no pin has ever covered."""
        m1 = self.t1(user="Say OK.  ")
        p1, _ = self.turn(m1, RAW1)
        m2 = [{"role": "system", "content": SYSTEM},
              {"role": "user", "content": "Say OK."},          # trimmed by the client
              {"role": "assistant", "content": self.visible(RAW1)},
              {"role": "user", "content": "Sky?"}]
        p2, plan2 = self.render(m2)
        self.assertEqual(plan2.state, "continuation")
        self.assertTrue(p2.startswith(p1 + RAW1))
        self.assertIn("<|user|>Say OK.  <|assistant|>", p2)     # turn 1's BYTES

    def test_a_reply_truncated_inside_think_has_no_visible_part(self):
        """The budget ran out mid-reasoning: the client stores an empty message,
        and that turn's tokens are the ones most worth replaying."""
        cut = "still working through the three boxes, no close marker at all"
        m1 = self.t1()
        p1, _ = self.turn(m1, cut, gen=128, length_limited=True)
        m2 = self.t1() + [{"role": "assistant", "content": ""},
                          {"role": "user", "content": "Sky?"}]
        p2, plan2 = self.render(m2)
        self.assertEqual(plan2.state, "continuation")
        self.assertTrue(p2.startswith(p1 + cut))

    def test_an_empty_stored_reply_does_not_match_a_visible_one(self):
        """The narrow rule: an empty message matches only an empty recording."""
        m1 = self.t1()
        self.turn(m1, RAW1)
        m2 = self.t1() + [{"role": "assistant", "content": ""},
                          {"role": "user", "content": "Sky?"}]
        _p2, plan2 = self.render(m2)
        self.assertEqual(plan2.state, "diverged")

    # ================================================================== 2
    # client-shorter: regenerate, edit, branch
    # ==================================================================
    def test_regenerate_truncates_the_ledger_and_resets(self):
        """The client re-sends the conversation WITHOUT the last assistant turn.

        That is `shorter`, not `continuation`: the record holds four turns and
        the client sends three. The engine cannot rewind, so the turn
        re-prefills -- correct, not a regression -- and the record is cut back
        to what the client still claims."""
        m1 = self.t1()
        p1, _ = self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky?")
        self.turn(m2, RAW2)
        again = m2                                   # regenerate turn 2
        p, plan = self.render(again)
        self.assertEqual(plan.state, "shorter")
        self.assertEqual(plan.matched, 3)
        self.assertEqual(len(S._ledger_cache[plan.key]["turns"]), 3)
        self.assertTrue(p.startswith(p1 + RAW1))     # the turns that DID match
        self.assertNotIn("A clear sky is blue.", p)  # the reply being regenerated is gone

    def test_regenerate_of_the_first_reply_is_shorter(self):
        m1 = self.t1()
        p1, _ = self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky?")
        self.turn(m2, RAW2)
        p, plan = self.render(m1)                    # back to just [S, U1]
        self.assertEqual(plan.state, "shorter")
        self.assertEqual(plan.matched, 1)
        self.assertEqual(p, p1)                      # byte-identical to the cold render
        self.assertEqual(len(S._ledger_cache[plan.key]["turns"]), 1)

    def test_edit_an_earlier_message_diverges_and_renders_the_client_version(self):
        m1 = self.t1()
        self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky?")
        self.turn(m2, RAW2)
        edited = list(m2)
        edited[3] = {"role": "user", "content": "Actually: what colour is grass?"}
        p, plan = self.render(edited)
        self.assertEqual(plan.state, "diverged")
        self.assertEqual(plan.matched, 2)            # S+U1 and the reply still matched
        self.assertIn("Actually: what colour is grass?", p)
        self.assertNotIn("Sky?", p)

    def test_branch_from_the_first_turn_is_shorter_then_diverges(self):
        m1 = self.t1()
        self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky?")
        self.turn(m2, RAW2)
        branch = self.extend(m1, RAW1, "A different second question")
        p, plan = self.render(branch)
        self.assertEqual(plan.state, "diverged")
        self.assertIn("A different second question", p)
        self.assertNotIn("Sky?", p)

    def test_a_truncated_render_equals_a_cold_render_of_that_transcript(self):
        """Slow is allowed, wrong is not: whatever the ledger decides, the bytes
        it renders for a diverged or shortened transcript are the bytes a fresh
        process would render for it."""
        m1 = self.t1()
        self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky?")
        self.turn(m2, RAW2)
        edited = list(m2)
        edited[3] = {"role": "user", "content": "Something else"}
        p, _plan = self.render(edited)
        S._ledger_cache.clear()
        cold, _ = self.render(edited)
        self.assertEqual(p, cold)

    # ================================================================== 3
    # unknown key, and two conversations colliding on the derived one
    # ==================================================================
    def test_unknown_key_renders_the_clients_version_verbatim(self):
        m = self.extend(self.t1(), RAW1, "Sky?")     # a history this process never saw
        p, plan = self.render(m)
        self.assertEqual(plan.state, "new")
        self.assertNotIn("I will answer OK.", p)      # nothing invented
        self.assertIn("<think></think>OK", p)         # exactly today's rendering
        S._ledger_cache.clear()
        cold, _ = self.render(m)
        self.assertEqual(p, cold)

    def test_two_conversations_that_open_alike_share_a_key_and_do_not_leak(self):
        """Open WebUI pops `metadata` (chat_id) before forwarding, so the key is
        derived and two chats with the same first message collide BY
        CONSTRUCTION. The match rule is what makes that harmless."""
        m1 = self.t1()
        self.turn(m1, RAW1)
        other = self.t1() + [{"role": "assistant", "content": "A completely different reply"},
                             {"role": "user", "content": "Second chat, second question"}]
        p, plan = self.render(other)
        self.assertEqual(S.conversation_pin_key(m1), S.conversation_pin_key(other))
        self.assertEqual(plan.state, "diverged")
        self.assertNotIn("I will answer OK.", p)
        self.assertIn("A completely different reply", p)

    def test_a_colliding_conversation_does_not_steal_the_first_ones_record(self):
        m1 = self.t1()
        p1, _ = self.turn(m1, RAW1)
        other = self.t1() + [{"role": "assistant", "content": "Different"},
                             {"role": "user", "content": "Q"}]
        self.turn(other, RAW3)                        # the collision overwrites the record
        # the FIRST conversation now diverges -- it re-prefills, which is correct,
        # and it does not receive the other chat's text
        m2 = self.extend(m1, RAW1, "Sky?")
        p2, plan2 = self.render(m2)
        self.assertIn(plan2.state, ("diverged", "shorter"))
        self.assertNotIn("Grass is green.", p2)
        self.assertNotIn("Different", p2)
        del p1

    # ================================================================== 4
    # the head: tools, effort, extra system messages
    # ==================================================================
    def test_a_changed_tool_list_resets_rather_than_pinning_the_old_block(self):
        m1 = self.t1()
        self.turn(m1, RAW1, tools=TOOLS)
        m2 = self.extend(m1, RAW1, "Sky?")
        p2, plan2 = self.render(m2, tools=None)
        self.assertEqual(plan2.state, "reset")
        self.assertNotIn("get_time", p2)              # never a declaration the client withdrew

    def test_a_changed_reasoning_effort_resets(self):
        m1 = self.t1()
        self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky?")
        _p2, plan2 = self.render(m2, effort="high")
        self.assertEqual(plan2.state, "reset")

    def test_a_second_system_message_mid_history_is_a_turn_not_the_head(self):
        m1 = self.t1()
        p1, _ = self.turn(m1, RAW1)
        m2 = m1 + [{"role": "assistant", "content": self.visible(RAW1)},
                   {"role": "system", "content": "Extra context arrived."},
                   {"role": "user", "content": "Sky?"}]
        p2, plan2 = self.render(m2)
        self.assertEqual(plan2.state, "continuation")
        self.assertTrue(p2.startswith(p1 + RAW1))
        self.assertIn("<|system|>Extra context arrived.", p2)

    def test_more_leading_system_messages_starts_a_new_record(self):
        m1 = self.t1()
        self.turn(m1, RAW1)
        m2 = ([{"role": "system", "content": SYSTEM},
               {"role": "system", "content": "and another"}]
              + m1[1:] + [{"role": "assistant", "content": self.visible(RAW1)},
                          {"role": "user", "content": "Sky?"}])
        p2, plan2 = self.render(m2)
        # A different head is a different conversation key, so this is a new
        # record rather than a reset of the old one. Either way nothing is
        # replayed, which is the property that matters.
        self.assertIn(plan2.state, ("new", "reset"))
        self.assertNotIn("I will answer OK.", p2)

    # ================================================================== 5
    # the invariant check (§3.3) -- the alarm this project was missing
    # ==================================================================
    def test_a_broken_invariant_falls_back_and_does_not_submit_the_claim(self):
        m1 = self.t1()
        p1, plan1 = self.turn(m1, RAW1)
        with S._ledger_lock:                          # corrupt the record by hand
            S._ledger_cache[plan1.key]["prompt"] = p1 + "AN IMPOSSIBLE EXTRA"
        m2 = self.extend(m1, RAW1, "Sky?")
        p2, plan2 = self.render(m2)
        self.assertEqual(plan2.state, "broken")
        S._ledger_cache.clear()
        cold, _ = self.render(m2)
        self.assertEqual(p2, cold)                    # the client's own rendering, exactly

    def test_strict_refuses_instead_of_falling_back(self):
        m1 = self.t1()
        p1, plan1 = self.turn(m1, RAW1)
        with S._ledger_lock:
            S._ledger_cache[plan1.key]["prompt"] = p1 + "AN IMPOSSIBLE EXTRA"
        m2 = self.extend(m1, RAW1, "Sky?")
        with patch.dict(S.os.environ, {"COLI_LEDGER_STRICT": "1"}):
            with self.assertRaises(S.APIError):
                self.render(m2)

    def test_an_exception_inside_the_ledger_falls_back_and_never_fails(self):
        m1 = self.t1()
        self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky?")
        with patch.object(S, "ledger_decide", side_effect=RuntimeError("boom")):
            p2, plan2 = self.render(m2)
        self.assertIsNone(plan2)
        S._ledger_cache.clear()
        cold, _ = self.render(m2)
        self.assertEqual(p2, cold)

    # ================================================================== 6
    # the safety rule, asserted directly
    # ==================================================================
    def test_the_ledger_never_adds_or_drops_a_turn(self):
        """One rendered piece per message the client sent, in every state."""
        m1 = self.t1()
        self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky?")
        self.turn(m2, RAW2)
        edited = list(m2); edited[3] = {"role": "user", "content": "Edited"}
        for name, messages in (("continuation", self.extend(m2, RAW2, "And grass?")),
                               ("shorter", m1),
                               ("diverged", edited),
                               ("new", self.t1(user="a brand new opening"))):
            with self.subTest(state=name):
                _p, plan = self.render(messages)
                owned = [o for o, _piece in (plan.parts or []) if o is not None]
                self.assertEqual(owned, list(range(len(messages))))
                self.assertEqual(len(plan.messages), len(messages))

    def test_the_annotation_never_leaks_into_the_callers_messages(self):
        m1 = self.t1()
        self.turn(m1, RAW1)
        m2 = self.extend(m1, RAW1, "Sky?")
        _p2, plan2 = self.render(m2)
        self.assertFalse(any(S.LEDGER_FIELD in m for m in m2))
        self.assertTrue(any(S.LEDGER_FIELD in m for m in plan2.messages))

    def test_the_parts_join_back_to_the_prompt(self):
        """The pieces the ledger records ARE the prompt, not a re-derivation."""
        p, plan = self.render(self.t1())
        self.assertEqual("".join(piece for _o, piece in plan.parts), p)

    def test_the_visible_reply_is_never_what_the_ledger_touches(self):
        """The ledger changes the prompt, never the response path: the visible
        text a client sends is rendered back only into the PROMPT, and the reply
        the user reads is whatever the engine generated for it."""
        m1 = self.t1()
        _p1, plan1 = self.turn(m1, RAW1)
        entry = S._ledger_cache[plan1.key]
        self.assertEqual(entry["generated"], RAW1)
        self.assertEqual(entry["turns"][-1]["rendered"], GEN + RAW1)

    # ================================================================== 7
    # slot ownership (§3.5)
    # ==================================================================
    def test_each_conversation_owns_a_slot_and_keeps_it(self):
        slots = []
        for i in range(4):
            _p, plan = self.turn(self.t1(user=f"conversation {i}"), RAW1)
            slots.append(plan.slot)
        self.assertEqual(sorted(slots), [0, 1, 2, 3])
        for i in range(4):
            m = self.extend(self.t1(user=f"conversation {i}"), RAW1, "next?")
            _p, plan = self.render(m)
            self.assertEqual(plan.slot, slots[i])

    def test_a_fifth_conversation_takes_the_least_recently_used_slot(self):
        for i in range(4):
            self.turn(self.t1(user=f"conversation {i}"), RAW1)
        first = S._ledger_cache[S.conversation_pin_key(self.t1(user="conversation 0"))]["slot"]
        _p, plan = self.render(self.t1(user="conversation 4"))
        self.assertEqual(plan.slot, first)

    def test_one_slot_is_slot_zero(self):
        self.kv_slots = 1
        _p, plan = self.render(self.t1())
        self.assertEqual(plan.slot, 0)

    # ================================================================== 8
    # expect_reuse arithmetic and the report line
    # ==================================================================
    def test_expect_reuse_is_the_previous_prompt_plus_its_generation(self):
        m1 = self.t1()
        _p1, plan1 = self.render(m1)
        self.record(plan1, RAW1, prompt_tokens=4718, gen=29)
        m2 = self.extend(m1, RAW1, "Sky?")
        _p2, plan2 = self.render(m2)
        self.assertEqual(plan2.expect_reuse, 4747)     # the P8 table's own number

    def test_a_length_limited_turn_is_one_position_short(self):
        m1 = self.t1()
        _p1, plan1 = self.render(m1)
        self.record(plan1, RAW1, prompt_tokens=4718, gen=128, length_limited=True)
        m2 = self.extend(m1, RAW1, "Sky?")
        _p2, plan2 = self.render(m2)
        self.assertEqual(plan2.expect_reuse, 4718 + 128 - 1)

    def test_a_cancelled_turn_predicts_nothing_and_keeps_the_transcript(self):
        m1 = self.t1()
        p1, plan1 = self.render(m1)
        self.record(plan1, "partial thought", prompt_tokens=900, cancelled=True)
        p_retry, plan_retry = self.render(m1)
        self.assertIsNone(plan_retry.expect_reuse)
        self.assertEqual(p_retry, p1)                  # an identical retry, byte for byte
        self.assertNotIn("partial thought", p_retry)   # and no invented assistant turn

    def test_the_report_line_says_ok_or_MISMATCH(self):
        m1 = self.t1()
        _p1, plan1 = self.render(m1)
        self.record(plan1, RAW1, prompt_tokens=100, gen=7)
        m2 = self.extend(m1, RAW1, "Sky?")
        _p2, plan2 = self.render(m2)
        lines = []
        with patch.object(S, "ledger_log", side_effect=lambda t, force=False: lines.append(t)):
            S.ledger_report(plan2, {"reused": 107})
            S.ledger_report(plan2, {"reused": 4419})
            S.ledger_report(plan2, {"reused": None})
        self.assertIn("expect_reuse=107 engine_reuse=107 ok", lines[0])
        self.assertIn("MISMATCH", lines[1])
        self.assertNotIn("MISMATCH", lines[2])         # an engine with no 8th field is quiet

    def test_the_engines_stat_line_carries_reuse_and_older_ones_do_not(self):
        base = "STAT 29 5.10 99.0 180.0 4718 0".split()
        self.assertIsNone(S.Engine._stats(base)["reused"])
        self.assertEqual(S.Engine._stats(base + ["4747"])["reused"], 4747)

    # ================================================================== 9
    # bookkeeping and the knob
    # ==================================================================
    def test_the_cache_is_bounded(self):
        with patch.dict(S.os.environ, {"COLI_LEDGER_MAX_CONV": "8"}):
            for i in range(20):
                self.turn(self.t1(user=f"conversation {i}"), RAW1)
            self.assertEqual(len(S._ledger_cache), 8)

    def test_the_knob_off_is_todays_path(self):
        with patch.dict(S.os.environ, {"COLI_LEDGER": "0"}):
            m1 = self.t1()
            p1, plan1 = self.render(m1)
            self.assertIsNone(plan1)
            self.record(plan1, RAW1)
            self.assertEqual(len(S._ledger_cache), 0)
            m2 = self.extend(m1, RAW1, "Sky?")
            p2, _ = self.render(m2)
            self.assertFalse(p2.startswith(p1 + RAW1))

    def test_the_ledger_does_not_run_for_another_engine(self):
        with patch.object(S, "ARCH", "qwen38"):
            self.assertFalse(S.ledger_enabled())

    # ================================================================== 10
    # the decision table itself, as a table
    # ==================================================================
    def test_decide_table(self):
        a, b, c = ("user", "a", ""), ("assistant", "b", ""), ("user", "c", "")
        for recorded, client, want in (
                ([], [a], ("continuation", 0)),
                ([a, b], [a, b, c], ("continuation", 2)),
                ([a, b], [a, b], ("continuation", 2)),
                ([a, b, c], [a, b], ("shorter", 2)),
                ([a, b, c], [a], ("shorter", 1)),
                ([a, b, c], [a, c, c], ("diverged", 1)),
                ([a, b], [c, b], ("diverged", 0)),
                ([a], [], ("shorter", 0)),
        ):
            with self.subTest(recorded=recorded, client=client):
                self.assertEqual(S.ledger_decide(recorded, client), want)


if __name__ == "__main__":
    unittest.main()
