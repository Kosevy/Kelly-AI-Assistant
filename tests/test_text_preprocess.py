"""
tests/test_text_preprocess.py
Tests unitarios para la normalización de texto TTS.
"""
import pytest
from app.text_preprocess import normalize_for_tts


class TestLetterExpansion:
    def test_single_letter_sequence_uppercase(self):
        result = normalize_for_tts("A, B, C")
        assert "a" in result.lower()
        assert "be" in result.lower()
        assert "ce" in result.lower()

    def test_letter_sequence_no_commas(self):
        result = normalize_for_tts("Las letras son A B C")
        # Debe expandir las letras sueltas
        assert "be" in result.lower() or "ce" in result.lower()

    def test_single_letter_not_expanded(self):
        # Una sola letra suelta no debe expandirse (requiere al menos 2)
        result = normalize_for_tts("El nombre es Ana")
        # "Ana" no debe expandirse, pero tampoco romperse
        assert "Ana" in result or "ana" in result.lower()

    def test_mixed_case_sequence(self):
        result = normalize_for_tts("Opciones: a, b, c")
        assert "be" in result.lower()
        assert "ce" in result.lower()


class TestNumberExpansion:
    def test_integer(self):
        result = normalize_for_tts("Son 3 resultados")
        assert "tres" in result.lower()

    def test_year(self):
        result = normalize_for_tts("El año 2025")
        assert "dos mil veinticinco" in result.lower()

    def test_large_number(self):
        result = normalize_for_tts("Hay 1000 opciones")
        assert "mil" in result.lower()

    def test_decimal(self):
        result = normalize_for_tts("Son 3.5 kilómetros")
        # num2words en es para float: "tres coma cinco" o similar
        assert "tres" in result.lower()


class TestAcronymExpansion:
    def test_ia(self):
        result = normalize_for_tts("La IA es increíble")
        assert "inteligencia artificial" in result.lower()

    def test_pc(self):
        result = normalize_for_tts("Abre el PC")
        assert "computadora" in result.lower()

    def test_url(self):
        result = normalize_for_tts("Visita esta URL")
        assert "u erre ele" in result.lower()

    def test_api(self):
        result = normalize_for_tts("Llama a la API")
        assert "a pe i" in result.lower()


class TestPunctuationNormalization:
    def test_ellipsis_collapsed(self):
        result = normalize_for_tts("Espera...")
        assert "..." not in result
        assert "." in result

    def test_emojis_removed(self):
        result = normalize_for_tts("Hola 😀 mundo 🌍")
        assert "😀" not in result
        assert "🌍" not in result
        assert "Hola" in result
        assert "mundo" in result

    def test_multiple_spaces_collapsed(self):
        result = normalize_for_tts("Hola   mundo")
        assert "  " not in result

    def test_markdown_asterisks_removed(self):
        result = normalize_for_tts("Texto **importante** aquí")
        assert "**" not in result
        assert "importante" in result

    def test_spanish_chars_preserved(self):
        result = normalize_for_tts("¿Cómo estás? ¡Muy bien, señor!")
        assert "¿" in result or "Cómo" in result
        assert "ñ" in result or "señor" in result


class TestFullPipeline:
    def test_combined(self):
        """Test completo del pipeline."""
        text = "Las letras son A, B, C y el año 2025"
        result = normalize_for_tts(text)
        assert "be" in result.lower()
        assert "ce" in result.lower()
        assert "dos mil veinticinco" in result.lower()

    def test_empty_string(self):
        result = normalize_for_tts("")
        assert result == ""

    def test_whitespace_only(self):
        result = normalize_for_tts("   ")
        assert result.strip() == ""

