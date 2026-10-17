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


class TestCalendarSystemPromptRules:
    """Verifica que prompts/system.txt incorpore las reglas anti-alucinación y UX de calendario."""

    def test_system_prompt_contains_calendar_presentation_rules(self):
        from pathlib import Path
        prompt_path = Path("prompts/system.txt")
        assert prompt_path.exists()
        content = prompt_path.read_text(encoding="utf-8")

        assert "Presentación de resultados del calendario:" in content
        assert "TEXTO PLANO con viñetas simples" in content
        assert "NO uses bloques de código" in content
        assert "NO uses tablas Markdown" in content
        assert "agrúpalos por día con un encabezado en negrita" in content

    def test_system_prompt_contains_calendar_tools_and_temporal_rules(self):
        from pathlib import Path
        content = Path("prompts/system.txt").read_text(encoding="utf-8")

        assert "`calendar_next_deadline`" in content
        assert "`calendar_read_range`" in content
        assert "`calendar_free_slots`" in content
        assert "Reglas temporales:" in content
        assert "NUNCA pidas al usuario la fecha actual" in content
        assert "NO pidas confirmación para operaciones de lectura" in content
        assert "Uso de datos de tools:" in content
        assert "Usa EXACTAMENTE los horarios, fechas y nombres que devolvió el tool" in content
        assert "Nomenclatura del calendario del usuario:" in content


class TestSystemPromptTypeMapping:
    """Verifica que prompts/system.txt contenga el mapeo explícito de argumentos
    para calendar_next_deadline. Regresión contra Bug: LLM no pasaba 'types'
    en tool calls al no encontrar el mapeo en el prompt."""

    def _load_prompt(self) -> str:
        from pathlib import Path
        return Path("prompts/system.txt").read_text(encoding="utf-8")

    def test_prompt_has_explicit_mapping_section(self):
        content = self._load_prompt()
        assert "Mapeo explícito de argumentos para `calendar_next_deadline`:" in content

    def test_prompt_maps_examen_to_exam_type(self):
        content = self._load_prompt()
        # "examen" → types=["exam"]
        assert '"examen"' in content
        assert '{"types": ["exam"]}' in content

    def test_prompt_maps_practica_to_practica_exam_types(self):
        content = self._load_prompt()
        # "PC" → types=["practica", "exam"]
        assert '"PC"' in content
        assert '{"types": ["practica", "exam"]}' in content

    def test_prompt_maps_entrega_to_entregable_type(self):
        content = self._load_prompt()
        # "entrega" → types=["entregable"]
        assert '"entrega"' in content
        assert '{"types": ["entregable"]}' in content

    def test_prompt_has_mandatory_types_rule(self):
        content = self._load_prompt()
        # Regla obligatoria: SIEMPRE pasa types cuando aplique
        assert "SIEMPRE pasa el parámetro `types`" in content



class TestFrontendCalendarRenderer:
    """Verifica que index.html configure marked y CSS para renderizar bloques sin lenguaje como plain-output."""

    def test_index_html_contains_plain_output_css_and_renderer(self):
        from pathlib import Path
        index_path = Path("index.html")
        assert index_path.exists()
        html = index_path.read_text(encoding="utf-8")

        # CSS
        assert ".plain-output" in html
        assert "white-space: pre-wrap;" in html
        assert "font-family: inherit;" in html

        # JS escapeHtml & renderer hook
        assert "function escapeHtml" in html
        assert "renderer.code" in html
        assert "plain-output" in html

    def test_index_html_contains_calendar_card_accordion_components(self):
        from pathlib import Path
        index_path = Path("index.html")
        assert index_path.exists()
        html = index_path.read_text(encoding="utf-8")

        # CSS para acordeón y tarjetas de calendario
        assert ".calendar-card" in html
        assert ".calendar-accordion" in html
        assert ".calendar-day" in html
        assert ".calendar-day-header" in html
        assert ".calendar-day-content" in html
        assert ".calendar-event-item" in html
        assert ".type-class" in html
        assert ".type-exam" in html
        assert ".type-entregable" in html
        assert ".type-meal" in html

        # JS función de renderizado seguro de tarjeta
        assert "function renderCalendarCard" in html
        assert "msg.type === 'calendar_card'" in html


class TestToolCallingTemperature:
    """Verifica que llm_client aplique tool_calling_temperature condicionalmente."""

    @pytest.mark.asyncio
    async def test_stream_chat_with_tools_temperature_selection(self, monkeypatch):
        from unittest.mock import MagicMock
        from app.llm_client import stream_chat_with_tools
        from app.config import settings

        captured_payloads = []

        class DummyResponse:
            status_code = 200
            async def aiter_lines(self):
                yield 'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}'
                yield 'data: [DONE]'

        class DummyStreamContext:
            def __init__(self, method, url, json=None, **kwargs):
                captured_payloads.append(json)
            async def __aenter__(self):
                return DummyResponse()
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        dummy_client = MagicMock()
        dummy_client.stream = DummyStreamContext

        class DummyAsyncClient:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return dummy_client
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        monkeypatch.setattr("httpx.AsyncClient", DummyAsyncClient)

        messages = [{"role": "user", "content": "hola"}]

        # 1. Con tools y sin temp explícita -> tool_calling_temperature (0.5)
        async for _ in stream_chat_with_tools(messages, tools=[{"type": "function", "function": {"name": "test"}}]):
            pass
        assert captured_payloads[0]["temperature"] == settings.tool_calling_temperature
        assert captured_payloads[0]["temperature"] == 0.5

        # 2. Sin tools y sin temp explícita -> llm_temperature (0.4)
        captured_payloads.clear()
        async for _ in stream_chat_with_tools(messages, tools=None):
            pass
        assert captured_payloads[0]["temperature"] == settings.llm_temperature

        # 3. Temp explícita siempre tiene prioridad
        captured_payloads.clear()
        async for _ in stream_chat_with_tools(messages, tools=[{"type": "function"}], temperature=0.85):
            pass
        assert captured_payloads[0]["temperature"] == 0.85

    @pytest.mark.asyncio
    async def test_stream_chat_with_tools_injection_modes(self, monkeypatch):
        from unittest.mock import MagicMock
        from app.llm_client import stream_chat_with_tools
        from app.config import settings

        async def fake_execute_tool(name, args, **kwargs):
            return "resultado de prueba"

        monkeypatch.setattr("app.llm_client.execute_tool", fake_execute_tool)

        # 1. Modo user_message
        captured_messages = []
        iter_count = [0]

        class DummyResponse:
            status_code = 200
            async def aiter_lines(self):
                if iter_count[0] == 1:
                    tc_json = '{"index": 0, "id": "call_1", "type": "function", "function": {"name": "test_fn", "arguments": "{}"}}'
                    yield f'data: {{"choices": [{{"delta": {{"tool_calls": [{tc_json}]}}, "finish_reason": "tool_calls"}}]}}'
                    yield 'data: [DONE]'
                else:
                    yield 'data: {"choices": [{"delta": {"content": "fin"}, "finish_reason": "stop"}]}'
                    yield 'data: [DONE]'

        class DummyStreamContext:
            def __init__(self, method, url, json=None, **kwargs):
                iter_count[0] += 1
                captured_messages.append(list(json["messages"]))
            async def __aenter__(self):
                return DummyResponse()
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        class DummyAsyncClient:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                d = MagicMock()
                d.stream = DummyStreamContext
                return d
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        monkeypatch.setattr("httpx.AsyncClient", DummyAsyncClient)
        monkeypatch.setattr(settings, "tool_result_injection_mode", "user_message")

        messages = [{"role": "user", "content": "pregunta"}]
        async for _ in stream_chat_with_tools(messages, tools=[{"type": "function"}]):
            pass

        assert len(captured_messages) == 2
        iter2_msgs = captured_messages[1]
        assert iter2_msgs[-1]["role"] == "user"
        assert "[RESULTADO DEL SISTEMA" in iter2_msgs[-1]["content"]
        assert iter2_msgs[-2]["role"] == "assistant"

        # 2. Modo tool_role
        captured_messages.clear()
        iter_count[0] = 0
        monkeypatch.setattr(settings, "tool_result_injection_mode", "tool_role")

        async for _ in stream_chat_with_tools(messages, tools=[{"type": "function"}]):
            pass

        assert len(captured_messages) == 2
        iter2_msgs_tool = captured_messages[1]
        assert iter2_msgs_tool[-1]["role"] == "tool"
        assert iter2_msgs_tool[-1]["content"] == "resultado de prueba"


