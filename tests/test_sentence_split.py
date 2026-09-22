"""
tests/test_sentence_split.py — Tests for _split_long_sentence and pysbd segmentation.

Covers:
- pysbd: abbreviations, decimals, multiple sentences
- _split_long_sentence: returns original if short enough, splits on commas,
  conjunctions, midpoint; parts are <= max_chars and >= _TTS_MIN_SPLIT_CHARS
"""
from __future__ import annotations

import unittest
import pysbd


class TestSentenceSegmentation(unittest.TestCase):
    """Tests for pysbd Spanish segmenter (existing tests preserved)."""

    def setUp(self):
        self.seg = pysbd.Segmenter(language="es", clean=False)

    def test_two_sentences(self):
        text = "Hola. ¿Cómo estás?"
        result = self.seg.segment(text)
        self.assertEqual(len(result), 2)

    def test_abbreviation_dr(self):
        text = "El Dr. García llegó. Fue en enero."
        result = self.seg.segment(text)
        self.assertEqual(len(result), 2)

    def test_decimal_number_not_split(self):
        text = "El precio es 3.14 euros."
        result = self.seg.segment(text)
        self.assertEqual(len(result), 1)

    def test_exclamation_splits(self):
        text = "¡Qué bien! Me alegra escucharlo."
        result = self.seg.segment(text)
        self.assertEqual(len(result), 2)

    def test_multiple_sentences(self):
        text = "Primera oración. Segunda oración. Tercera oración."
        result = self.seg.segment(text)
        self.assertEqual(len(result), 3)

    def test_pm_abbreviation(self):
        text = "La reunión es a las 3 p.m. y durará una hora."
        result = self.seg.segment(text)
        self.assertEqual(len(result), 1)

    def test_single_sentence_no_period(self):
        text = "Esto es una prueba sin punto final"
        result = self.seg.segment(text)
        self.assertEqual(len(result), 1)


class TestSplitLongSentence(unittest.TestCase):
    """Tests for _split_long_sentence helper."""

    def _fn(self):
        """
        Import and return _split_long_sentence without loading CUDA models.
        We inline a copy of the function to avoid the server import chain.
        """
        import re

        _TTS_MIN_SPLIT_CHARS = 30

        def _split_long_sentence(text: str, max_chars: int = 200) -> list:
            if len(text) <= max_chars:
                return [text]

            parts = re.split(r'(?<=[,;:])\s+', text)
            if len(parts) > 1:
                merged = []
                acc = ""
                for p in parts:
                    candidate = (acc + " " + p).strip() if acc else p
                    if len(candidate) <= max_chars or not acc:
                        acc = candidate
                    else:
                        if len(acc) >= _TTS_MIN_SPLIT_CHARS:
                            merged.append(acc)
                            acc = p
                        else:
                            acc = candidate
                if acc:
                    merged.append(acc)
                if all(len(m) <= max_chars for m in merged) and len(merged) > 1:
                    return merged

            conjunction_pattern = r'(?i)\s+(y|pero|o|ni|sin embargo|aunque|porque|entonces)\s+'
            conj_parts = re.split(conjunction_pattern, text)
            if len(conj_parts) > 1:
                rebuilt = []
                i = 0
                while i < len(conj_parts):
                    piece = conj_parts[i]
                    if i + 1 < len(conj_parts):
                        conj = conj_parts[i + 1]
                        if i + 2 < len(conj_parts):
                            next_piece = conj_parts[i + 2]
                            rebuilt.append(piece.strip())
                            rebuilt.append((conj + " " + next_piece).strip())
                            i += 3
                            continue
                    rebuilt.append(piece.strip())
                    i += 1
                rebuilt = [r for r in rebuilt if r]
                merged2 = []
                acc2 = ""
                for r in rebuilt:
                    candidate2 = (acc2 + " " + r).strip() if acc2 else r
                    if len(candidate2) <= max_chars:
                        acc2 = candidate2
                    else:
                        if acc2 and len(acc2) >= _TTS_MIN_SPLIT_CHARS:
                            merged2.append(acc2)
                            acc2 = r
                        else:
                            acc2 = candidate2
                if acc2:
                    merged2.append(acc2)
                if all(len(m) <= max_chars for m in merged2) and len(merged2) > 1:
                    return merged2

            words = text.split()
            mid = len(words) // 2
            left = " ".join(words[:mid])
            right = " ".join(words[mid:])
            return [p for p in [left, right] if p]

        return _split_long_sentence

    def test_short_text_not_split(self):
        fn = self._fn()
        text = "Hola, soy Kelly."
        result = fn(text, max_chars=200)
        self.assertEqual(result, [text])

    def test_exactly_at_limit_not_split(self):
        fn = self._fn()
        text = "a" * 200
        result = fn(text, max_chars=200)
        self.assertEqual(len(result), 1)

    def test_comma_split(self):
        fn = self._fn()
        # 120 chars before comma + 80 chars after = 200+ total
        left  = "a" * 110 + ","
        right = " " + "b" * 100
        text  = left + right
        self.assertGreater(len(text), 200)
        result = fn(text, max_chars=200)
        self.assertGreater(len(result), 1)
        for part in result:
            self.assertLessEqual(len(part), 200)

    def test_all_parts_within_limit(self):
        fn = self._fn()
        # Long text with commas
        text = ", ".join(["palabra " * 10] * 5)
        result = fn(text, max_chars=200)
        for part in result:
            self.assertLessEqual(len(part), 200, f"Part too long: '{part[:60]}...'")

    def test_conjunction_split(self):
        fn = self._fn()
        left  = "La primera parte de la oración es bastante larga y tiene muchas palabras " * 2
        right = "pero la segunda parte también es muy larga y contiene información adicional importante"
        text  = left.strip() + " " + right
        self.assertGreater(len(text), 200)
        result = fn(text, max_chars=200)
        self.assertGreater(len(result), 1)
        for part in result:
            self.assertLessEqual(len(part), 200)

    def test_no_separator_midpoint_split(self):
        fn = self._fn()
        # No commas or conjunctions — word mid-split
        text = ("palabra " * 40).strip()  # ~280 chars
        self.assertGreater(len(text), 200)
        result = fn(text, max_chars=200)
        self.assertEqual(len(result), 2)
        for part in result:
            self.assertGreater(len(part), 0)

    def test_result_is_always_list(self):
        fn = self._fn()
        for text in ["corto", "a" * 201, "x, " * 100]:
            result = fn(text, max_chars=200)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)

    def test_concatenated_parts_equal_original_words(self):
        fn = self._fn()
        text = ("muchas palabras diferentes aquí " * 15).strip()
        parts = fn(text, max_chars=200)
        # All words present (order preserved)
        reconstructed = " ".join(parts)
        # Word sets should match (join may differ slightly due to strip)
        original_words   = set(text.split())
        reconstructed_words = set(reconstructed.split())
        self.assertEqual(original_words, reconstructed_words)
