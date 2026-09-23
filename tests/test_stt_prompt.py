"""
tests/test_stt_prompt.py — Unit tests for app.stt_prompt.build_initial_prompt.

Tests the pure function in isolation — no models, no audio, no server imports.
"""
from __future__ import annotations

import unittest
from app.stt_prompt import build_initial_prompt


class TestBuildInitialPrompt(unittest.TestCase):

    def test_empty_history_returns_empty_string(self):
        """No history → no initial_prompt. Whisper ignores '' safely."""
        self.assertEqual(build_initial_prompt([]), "")

    def test_none_equivalent_empty(self):
        """Defensive: empty list is the canonical 'no history' signal."""
        result = build_initial_prompt([])
        self.assertIsInstance(result, str)
        self.assertEqual(result, "")

    def test_only_assistant_messages_returns_empty(self):
        """
        Kelly's responses are synthesized speech — the model has never 'heard' them.
        Including them adds noise; we exclude assistant messages by design.
        """
        history = [
            {"role": "assistant", "content": "Hola, soy Kelly."},
            {"role": "assistant", "content": "¿En qué puedo ayudarte?"},
        ]
        self.assertEqual(build_initial_prompt(history), "")

    def test_single_user_message(self):
        history = [{"role": "user", "content": "reglas del ajedrez"}]
        result = build_initial_prompt(history)
        self.assertIn("ajedrez", result)

    def test_uses_only_last_two_user_messages(self):
        """With last_n_user_messages=2 (default), only the 2 most recent user
        messages contribute to the prompt — older ones are excluded."""
        history = [
            {"role": "user", "content": "primera"},
            {"role": "assistant", "content": "respuesta uno"},
            {"role": "user", "content": "segunda"},
            {"role": "assistant", "content": "respuesta dos"},
            {"role": "user", "content": "tercera"},
        ]
        result = build_initial_prompt(history)
        self.assertIn("segunda", result)
        self.assertIn("tercera", result)
        self.assertNotIn("primera", result)

    def test_truncates_to_max_chars(self):
        """Prompt is truncated at max_chars from the right (keeps most recent)."""
        long_msg = "palabra " * 100  # ~700 chars
        history = [{"role": "user", "content": long_msg}]
        result = build_initial_prompt(history, max_chars=50)
        self.assertLessEqual(len(result), 50)

    def test_truncates_keeping_end_of_string(self):
        """When truncating, we keep the tail (most recent words), not the head."""
        history = [{"role": "user", "content": "AAAAAA BBBBBB CCCCCC"}]
        result = build_initial_prompt(history, max_chars=13)
        # tail of "AAAAAA BBBBBB CCCCCC" truncated to 13 chars is "BBBB CCCCCC" → ends with CCCCCC
        self.assertTrue(result.endswith("CCCCCC"), f"Got: {repr(result)}")

    def test_mixed_history_extracts_user_only(self):
        history = [
            {"role": "user", "content": "color favorito"},
            {"role": "assistant", "content": "No sé tu color favorito."},
            {"role": "user", "content": "Ken Carson"},
        ]
        result = build_initial_prompt(history)
        self.assertIn("color favorito", result)
        self.assertIn("Ken Carson", result)
        self.assertNotIn("No sé", result)

    def test_custom_last_n(self):
        """last_n_user_messages parameter controls how many to include."""
        history = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = build_initial_prompt(history, last_n_user_messages=1)
        self.assertEqual(result.strip(), "c")

    def test_missing_role_key_ignored(self):
        """Malformed entries (no 'role') don't crash and are excluded."""
        history = [
            {"content": "sin rol"},
            {"role": "user", "content": "con rol"},
        ]
        result = build_initial_prompt(history)
        self.assertIn("con rol", result)
        self.assertNotIn("sin rol", result)

    def test_missing_content_key_ignored(self):
        """Entries with no 'content' don't crash and are excluded."""
        history = [
            {"role": "user"},  # no content
            {"role": "user", "content": "válido"},
        ]
        result = build_initial_prompt(history)
        self.assertEqual(result.strip(), "válido")

    def test_result_is_always_string(self):
        """Return type is always str, never None."""
        for h in [[], [{"role": "assistant", "content": "x"}]]:
            result = build_initial_prompt(h)
            self.assertIsInstance(result, str)

    def test_does_not_include_hardcoded_vocabulary(self):
        """
        Smoke test confirming there's no static vocabulary in the output
        when history is empty. This is a design invariant: we never hardcode words.
        """
        result = build_initial_prompt([])
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()

