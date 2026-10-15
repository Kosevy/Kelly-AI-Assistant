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


class TestDatetimeContext:
    def test_datetime_context_includes_resolved_ranges(self):
        from datetime import datetime
        from app.system_prompt import build_datetime_context

        fixed_dt = datetime(2026, 10, 15, 14, 30)
        ctx = build_datetime_context(now=fixed_dt)
        assert "Rangos resueltos:" in ctx
        assert '"esta semana"' in ctx
        assert '"próxima semana"' in ctx
        assert '"este mes"' in ctx
        assert "2026-10-12" in ctx  # start of this week
        assert "2026-10-18" in ctx  # end of this week
        assert "2026-10-19" in ctx  # start of next week
        assert "2026-10-25" in ctx  # end of next week
        assert "2026-10-01 a 2026-10-31" in ctx  # this month

    def test_datetime_context_uses_spanish_day_names(self):
        from datetime import datetime
        from app.system_prompt import build_datetime_context

        fixed_dt = datetime(2026, 10, 15, 10, 0)
        ctx = build_datetime_context(now=fixed_dt)
        assert "jueves" in ctx.lower()
        assert "octubre" in ctx.lower()
        for english_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            assert english_day not in ctx

    def test_next_week_starts_on_monday(self):
        from datetime import datetime
        from app.system_prompt import build_datetime_context

        # 2026-10-15 is Thursday
        fixed_dt = datetime(2026, 10, 15, 9, 0)
        ctx = build_datetime_context(now=fixed_dt)
        assert '"próxima semana" = 2026-10-19 (lunes) a 2026-10-25 (domingo)' in ctx

    def test_this_week_includes_today(self):
        from datetime import datetime
        from app.system_prompt import build_datetime_context

        # 2026-10-15 is Thursday
        fixed_dt = datetime(2026, 10, 15, 12, 0)
        ctx = build_datetime_context(now=fixed_dt)
        assert '"esta semana" = 2026-10-12 (lunes) a 2026-10-18 (domingo)' in ctx
        assert "Hoy: 2026-10-15 (jueves)" in ctx

    def test_default_build_system_prompt_injects_dynamic_context(self):
        from datetime import datetime
        from app.system_prompt import build_system_prompt

        template = "Prompt:\n{current_datetime}"
        fixed_dt = datetime(2026, 10, 15, 12, 0)
        res = build_system_prompt(template, now=fixed_dt)
        assert "Contexto temporal:" in res
        assert "2026-10-15" in res
        assert "{current_datetime}" not in res
