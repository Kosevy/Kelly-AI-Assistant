"""
tests/test_sentence_split.py
Tests para la segmentación de oraciones con pysbd.
"""
import pytest
import pysbd


@pytest.fixture(scope="module")
def segmenter():
    return pysbd.Segmenter(language="es", clean=False)


class TestSentenceSegmentation:
    def test_abbreviation_dr(self, segmenter):
        """'Dr.' no debe separar la oración."""
        text = "El Dr. García llegó a las 3.30 p.m."
        sentences = segmenter.segment(text)
        assert len(sentences) == 1, (
            f"Se esperaba 1 oración, se obtuvieron {len(sentences)}: {sentences}"
        )

    def test_two_sentences(self, segmenter):
        """Dos oraciones distintas con '?' y '.'"""
        text = "¿Cómo estás? Bien, gracias."
        sentences = segmenter.segment(text)
        assert len(sentences) == 2, (
            f"Se esperaba 2 oraciones, se obtuvieron {len(sentences)}: {sentences}"
        )

    def test_decimal_number_not_split(self, segmenter):
        """Un número decimal (3.14) no debe separar la oración."""
        text = "El valor de pi es aproximadamente 3.14 y es irracional."
        sentences = segmenter.segment(text)
        assert len(sentences) == 1, (
            f"Se esperaba 1 oración, se obtuvieron {len(sentences)}: {sentences}"
        )

    def test_exclamation_splits(self, segmenter):
        """'!' debe separar oraciones."""
        text = "¡Qué sorpresa! No lo esperaba."
        sentences = segmenter.segment(text)
        assert len(sentences) == 2

    def test_multiple_sentences(self, segmenter):
        """Párrafo con múltiples oraciones."""
        text = "El sol salió temprano. Las aves cantaban. Era un día perfecto."
        sentences = segmenter.segment(text)
        assert len(sentences) == 3

    def test_pm_abbreviation(self, segmenter):
        """'p.m.' no debe separar la oración."""
        text = "La reunión es a las 3 p.m. en la sala principal."
        sentences = segmenter.segment(text)
        assert len(sentences) == 1, (
            f"Se esperaba 1 oración, se obtuvieron {len(sentences)}: {sentences}"
        )

    def test_single_sentence_no_period(self, segmenter):
        """Texto sin punto final debe ser una sola oración."""
        text = "Hola, ¿cómo te llamas"
        sentences = segmenter.segment(text)
        assert len(sentences) == 1

