#!/usr/bin/env python3
"""P8 -- the reply pin: the rendered prompt of turn 2 must reproduce turn 1's tokens.

The bug this guards (measured 2026-09-09, browser matrix in the record): Open
WebUI stores an assistant turn as its VISIBLE content only, so the `<think>`
block GLM-5.3 generated never comes back. render_chat_glm53 then rebuilds the
turn as `<think></think>{content.strip()}`, the rendered prompt stops matching
the token sequence the KV slot holds, and the conversation re-prefills from the
checkpoint -- 45-321 s instead of ~3 s.

The invariant the pin has to satisfy is one line of string algebra, and it is
what this file asserts:

    render(turn 2 messages)  startswith  render(turn 1 messages) + RAW

i.e. the prompt of the next turn is the previous prompt plus exactly the bytes
the engine generated. No model, no engine, seconds to run.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import openai_server as S


SYSTEM = "You are a helpful assistant."
MEM = "<memory_context>\n- likes espresso\n- lives in Rome\n</memory_context>"
U1 = "Here is some filler. Reply with the single word OK."
U2 = "In one word: what colour is a clear sky?"
# What the engine actually emits after the prompt's own `<think>`: reasoning,
# the closing marker, then the answer. Note the leading newline the model puts
# after </think> and the trailing space -- both are bytes the KV slot holds and
# both are what `content.strip()` used to throw away.
RAW = "The user wants one word. I will answer OK.</think>\nOK "


def visible_of(raw):
    with patch.object(S, "ARCH", "glm53"):
        return S.reply_pin_visible(raw, enable_thinking=False)


class ReplyPinRender(unittest.TestCase):
    def setUp(self):
        S._reply_pin_cache.clear()
        # P9 (2026-09-10): the conversation ledger subsumes this pin and turns it
        # OFF when COLI_LEDGER=1 (its default), so these tests -- which are the
        # fallback path's tests now -- state the knob they exercise. The ledger's
        # own state machine is tested in test_glm53_ledger.py.
        self.env = patch.dict(S.os.environ, {"COLI_REPLY_PIN": "1", "COLI_LEDGER": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    # ---- helpers ---------------------------------------------------------
    def turn1(self, system=SYSTEM):
        return [{"role": "system", "content": system},
                {"role": "user", "content": U1}]

    def turn2(self, content, system=SYSTEM, **extra):
        assistant = {"role": "assistant", "content": content}
        assistant.update(extra)
        return self.turn1(system) + [assistant, {"role": "user", "content": U2}]

    def render(self, messages):
        with patch.object(S, "ARCH", "glm53"):
            return S.render_chat_glm53(S.apply_reply_pin(messages))

    def remember(self, messages, raw):
        with patch.object(S, "ARCH", "glm53"):
            S.remember_reply(messages, raw,
                             S.reply_pin_visible(raw, enable_thinking=False))

    # ---- the invariant ---------------------------------------------------
    def test_turn2_prompt_extends_turn1_prompt_by_the_raw_generation(self):
        p1 = self.render(self.turn1())
        self.assertTrue(p1.endswith("<|assistant|><think>"), p1[-40:])
        self.remember(self.turn1(), RAW)
        # what Open WebUI stores: the visible part, stripped
        stored = visible_of(RAW).strip()
        self.assertEqual(stored, "OK")
        p2 = self.render(self.turn2(stored))
        self.assertTrue(p2.startswith(p1 + RAW),
                        "turn 2 does not reproduce turn 1's tokens:\n"
                        f"  expected prefix: {(p1 + RAW)[-80:]!r}\n"
                        f"  got:             {p2[len(p1) - 20:len(p1) + 60]!r}")
        self.assertEqual(p2, p1 + RAW + f"<|user|>{U2}<|assistant|><think>")

    def test_without_the_pin_the_reasoning_is_lost(self):
        """The regression itself: this is what the pre-P8 gateway renders."""
        p1 = self.render(self.turn1())
        self.remember(self.turn1(), RAW)
        with patch.dict(S.os.environ, {"COLI_REPLY_PIN": "0"}):
            p2 = self.render(self.turn2(visible_of(RAW).strip()))
        self.assertFalse(p2.startswith(p1 + RAW))
        self.assertNotIn("I will answer OK.", p2)
        self.assertIn("<think></think>OK", p2)

    def test_pin_is_off_when_the_knob_is_zero_even_for_remembering(self):
        with patch.dict(S.os.environ, {"COLI_REPLY_PIN": "0"}):
            self.remember(self.turn1(), RAW)
        self.assertEqual(len(S._reply_pin_cache), 0)

    def test_the_ledger_turns_this_pin_off(self):
        """P9: not "unused" -- off, so a measurement can only be about one of them."""
        with patch.object(S, "ARCH", "glm53"), patch.dict(S.os.environ, {"COLI_LEDGER": "1"}):
            self.assertFalse(S.reply_pin_enabled())
            self.remember(self.turn1(), RAW)
            self.assertEqual(len(S._reply_pin_cache), 0)
        with patch.object(S, "ARCH", "glm53"), patch.dict(S.os.environ, {"COLI_LEDGER": "0"}):
            self.assertTrue(S.reply_pin_enabled())

    # ---- matching --------------------------------------------------------
    def test_trailing_whitespace_tolerance(self):
        p1 = self.render(self.turn1())
        self.remember(self.turn1(), RAW)
        for stored in ("OK", "OK ", " OK\n", "\nOK\n\n"):
            with self.subTest(stored=stored):
                self.assertTrue(self.render(self.turn2(stored)).startswith(p1 + RAW))

    def test_client_that_returns_the_whole_block_still_matches(self):
        p1 = self.render(self.turn1())
        self.remember(self.turn1(), RAW)
        whole = "<think>The user wants one word. I will answer OK.</think>\nOK"
        self.assertTrue(self.render(self.turn2(whole)).startswith(p1 + RAW))

    def test_a_different_reply_is_not_pinned(self):
        p1 = self.render(self.turn1())
        self.remember(self.turn1(), RAW)
        p2 = self.render(self.turn2("Something else entirely"))
        self.assertFalse(p2.startswith(p1 + RAW))
        self.assertIn("<think></think>Something else entirely", p2)

    def test_reasoning_content_from_the_client_wins(self):
        """A client that kept the reasoning needs nothing from the pin."""
        self.remember(self.turn1(), RAW)
        p2 = self.render(self.turn2("OK", reasoning_content="client kept this"))
        self.assertIn("<think>client kept this</think>OK", p2)

    def test_tool_calls_are_not_pinned(self):
        self.remember(self.turn1(), RAW)
        p2 = self.render(self.turn2("OK", tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]))
        self.assertNotIn("I will answer OK.", p2)

    def test_reply_truncated_inside_think_is_restored_by_turn_index(self):
        """A reply that hit --max-tokens while thinking has NO visible part.

        Open WebUI stores an empty assistant message for it, so there is nothing
        to compare -- and it is exactly the turn whose tokens are most worth
        restoring. The remembered turn INDEX carries it.
        """
        cut = "still thinking about the boxes, no close marker at all"
        p1 = self.render(self.turn1())
        self.remember(self.turn1(), cut)
        p2 = self.render(self.turn2(""))
        self.assertTrue(p2.startswith(p1 + cut), p2[len(p1) - 10:len(p1) + 40])

    def test_empty_content_is_not_pinned_from_a_visible_reply(self):
        """The empty rule is narrow: it never restores a reply that HAD content."""
        self.remember(self.turn1(), RAW)
        p2 = self.render(self.turn2(""))
        self.assertNotIn("I will answer OK.", p2)

    def test_a_regenerated_turn_replaces_the_one_it_regenerates(self):
        again = "Second thoughts.</think>\nSure"
        p1 = self.render(self.turn1())
        self.remember(self.turn1(), RAW)
        self.remember(self.turn1(), again)      # same turn index: a regenerate
        key = S.conversation_pin_key(self.turn1())
        self.assertEqual(len(S._reply_pin_cache[key]), 1)
        self.assertTrue(self.render(self.turn2("Sure")).startswith(p1 + again))

    # ---- the conversation key -------------------------------------------
    def test_key_survives_a_reranked_memory_block(self):
        """Open WebUI re-ranks <memory_context> every turn; the key must not move."""
        a = SYSTEM + "\n\n" + MEM
        b = SYSTEM + "\n\n<memory_context>\n- lives in Rome\n- likes espresso\n</memory_context>"
        self.assertEqual(S.conversation_pin_key(self.turn1(a)),
                         S.conversation_pin_key(self.turn1(b)))
        p1 = self.render(self.turn1(a))
        self.remember(self.turn1(a), RAW)
        # the block re-ranks on turn 2, as it does in service with the pin off
        self.assertTrue(self.render(self.turn2("OK", system=b))
                        .endswith(RAW + f"<|user|>{U2}<|assistant|><think>"))
        self.assertNotEqual(p1, "")

    def test_key_separates_conversations(self):
        self.assertNotEqual(S.conversation_pin_key(self.turn1()),
                            S.conversation_pin_key(
                                [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": "a different opening"}]))

    # ---- bookkeeping -----------------------------------------------------
    def test_multi_turn_matches_in_order(self):
        raw2 = "Blue sky, easy.</think>\nBlue"
        p1 = self.render(self.turn1())
        self.remember(self.turn1(), RAW)
        two = self.turn2("OK")
        p2 = self.render(two)
        self.remember(two, raw2)
        three = two + [{"role": "assistant", "content": "Blue"},
                       {"role": "user", "content": "And grass?"}]
        p3 = self.render(three)
        self.assertTrue(p3.startswith(p1 + RAW))
        self.assertTrue(p3.startswith(p2 + raw2))

    def test_caches_are_bounded(self):
        for i in range(S.REPLY_PIN_MAX_CONV + 10):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"conversation {i}"}]
            self.remember(msgs, RAW)
        self.assertEqual(len(S._reply_pin_cache), S.REPLY_PIN_MAX_CONV)
        # a conversation that keeps growing: one remembered turn per assistant turn
        msgs = self.turn1()
        for i in range(S.REPLY_PIN_MAX_TURNS + 5):
            self.remember(msgs, f"thought {i}</think>\nanswer {i}")
            msgs = msgs + [{"role": "assistant", "content": f"answer {i}"},
                           {"role": "user", "content": f"and then? {i}"}]
        key = S.conversation_pin_key(self.turn1())
        self.assertEqual(len(S._reply_pin_cache[key]), S.REPLY_PIN_MAX_TURNS)

    def test_annotation_does_not_leak_into_the_caller_messages(self):
        self.remember(self.turn1(), RAW)
        messages = self.turn2("OK")
        out = S.apply_reply_pin(messages)
        self.assertNotIn(S.REPLY_PIN_FIELD, messages[2])
        self.assertIn(S.REPLY_PIN_FIELD, out[2])


if __name__ == "__main__":
    unittest.main()
