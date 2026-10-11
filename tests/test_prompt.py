"""
tests/test_prompt.py
Unit tests for app.system_prompt (pure prompt construction functions).
"""
import pytest
from app.system_prompt import build_system_prompt, format_input_mode_instruction


class TestInputModeInstruction:
    def test_voice_mode_instruction(self):
        desc = format_input_mode_instruction("voice")
        assert "Voz" in desc
        assert "TTS" in desc

    def test_text_mode_instruction(self):
        desc = format_input_mode_instruction("text")
        assert "Texto" in desc
        assert "Markdown" in desc

    def test_case_and_whitespace_insensitivity(self):
        desc_upper = format_input_mode_instruction("  TEXT  ")
        assert "Texto" in desc_upper
        desc_none = format_input_mode_instruction("")
        assert "Voz" in desc_none


class TestBuildSystemPrompt:
    def test_injects_current_datetime(self):
        template = "Hola. Hoy es {current_datetime}."
        result = build_system_prompt(template, current_datetime="Lunes 10 de Octubre")
        assert "Lunes 10 de Octubre" in result
        assert "{current_datetime}" not in result

    def test_injects_explicit_input_mode_placeholder(self):
        template = "Contexto: {current_datetime}. Modo: {input_mode}."
        result = build_system_prompt(
            template,
            current_datetime="2026-10-10",
            input_mode="text",
        )
        assert "2026-10-10" in result
        assert "{input_mode}" not in result
        assert "Texto" in result

    def test_appends_mode_when_placeholder_missing(self):
        template = "Eres Kelly. Hoy es {current_datetime}."
        result = build_system_prompt(
            template,
            current_datetime="2026-10-10",
            input_mode="voice",
        )
        assert "Eres Kelly. Hoy es 2026-10-10." in result
        assert "Modo de interacción actual:" in result
        assert "Voz" in result
