"""
tests/test_e2e_calendar.py
Pruebas End-to-End para el bypass del LLM y el modo híbrido de calendario (Bloque E).
"""
from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Mock STT and TTS before importing server to allow running fast tests without GPU/model loading
mock_tts_instance = MagicMock()
mock_tts_instance.to.return_value = mock_tts_instance
mock_tts_instance.synthesizer.tts_model.get_conditioning_latents.return_value = ("fake_latent", "fake_speaker")

with patch("faster_whisper.WhisperModel"), \
     patch("TTS.api.TTS", return_value=mock_tts_instance), \
     patch("os.path.exists", return_value=True):
    import server


def _make_mock_websocket():
    ws = AsyncMock()
    ws.sent_json = []

    async def _send_json(data):
        ws.sent_json.append(data)

    ws.send_json = _send_json
    return ws


class TestCalendarBypassE2E:
    @pytest.mark.asyncio
    async def test_calendar_bypass_read_week(self, monkeypatch):
        """Verifica que '¿Qué tengo la próxima semana?' envía calendar_card y NO llama al LLM."""
        ws = _make_mock_websocket()
        ctx = server._PipelineContext()

        llm_called = []

        async def fake_stream_chat(*args, **kwargs):
            llm_called.append(True)
            yield "token inventado"

        monkeypatch.setattr("server.stream_chat_with_tools", fake_stream_chat)

        await server._run_llm_and_tts(
            websocket=ws,
            user_text="¿Qué tengo la próxima semana?",
            history=[],
            ctx=ctx,
            input_mode="text",
            voice_enabled=False,
        )

        # 1. El LLM NO fue invocado para transcribir
        assert len(llm_called) == 0, "El LLM NO debió ser llamado para una consulta directa de calendario"

        # 2. Se envió un mensaje tipo calendar_card con datos estructurados
        card_msgs = [m for m in ws.sent_json if m.get("type") == "calendar_card"]
        assert len(card_msgs) == 1, "Debió enviarse exactamente 1 calendar_card"
        card = card_msgs[0]
        assert card["view_type"] in ("week", "range")
        assert "days" in card
        assert len(card["days"]) == 7

        # 3. Se envió un mensaje assistant_text con la frase introductoria
        assistant_msgs = [m for m in ws.sent_json if m.get("type") == "assistant_text"]
        assert len(assistant_msgs) == 1
        assert "agenda" in assistant_msgs[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_calendar_bypass_next_deadline(self, monkeypatch):
        """Verifica que '¿Cuál es mi próximo examen?' envía calendar_card tipo deadline sin llamar al LLM."""
        ws = _make_mock_websocket()
        ctx = server._PipelineContext()

        llm_called = []

        async def fake_stream_chat(*args, **kwargs):
            llm_called.append(True)
            yield "token"

        monkeypatch.setattr("server.stream_chat_with_tools", fake_stream_chat)

        await server._run_llm_and_tts(
            websocket=ws,
            user_text="¿Cuál es mi próximo examen?",
            history=[],
            ctx=ctx,
            input_mode="text",
            voice_enabled=False,
        )

        # 1. Sin llamada al LLM
        assert len(llm_called) == 0

        # 2. Se envió calendar_card de tipo deadline
        card_msgs = [m for m in ws.sent_json if m.get("type") == "calendar_card"]
        assert len(card_msgs) == 1
        assert card_msgs[0]["view_type"] == "deadline"

        # 3. Frase introductoria enviada
        assistant_msgs = [m for m in ws.sent_json if m.get("type") == "assistant_text"]
        assert len(assistant_msgs) == 1

    @pytest.mark.asyncio
    async def test_calendar_bypass_free_slots(self, monkeypatch):
        """Verifica que '¿Tengo tiempo para estudiar hoy?' envía calendar_card tipo free_slots sin llamar al LLM."""
        ws = _make_mock_websocket()
        ctx = server._PipelineContext()

        llm_called = []

        async def fake_stream_chat(*args, **kwargs):
            llm_called.append(True)
            yield "token"

        monkeypatch.setattr("server.stream_chat_with_tools", fake_stream_chat)

        await server._run_llm_and_tts(
            websocket=ws,
            user_text="¿Tengo tiempo para estudiar hoy?",
            history=[],
            ctx=ctx,
            input_mode="text",
            voice_enabled=False,
        )

        assert len(llm_called) == 0
        card_msgs = [m for m in ws.sent_json if m.get("type") == "calendar_card"]
        assert len(card_msgs) == 1
        assert card_msgs[0]["view_type"] == "free_slots"

    @pytest.mark.asyncio
    async def test_non_calendar_query_uses_llm_normally(self, monkeypatch):
        """Verifica que consultas no-calendario fluyen al LLM con streaming y NO envían calendar_card directa."""
        ws = _make_mock_websocket()
        ctx = server._PipelineContext()

        llm_called = []

        async def fake_stream_chat(*args, **kwargs):
            llm_called.append(True)
            yield "Respuesta "
            yield "del "
            yield "LLM."

        monkeypatch.setattr("server.stream_chat_with_tools", fake_stream_chat)

        await server._run_llm_and_tts(
            websocket=ws,
            user_text="Explícame qué es una base de datos relacional",
            history=[],
            ctx=ctx,
            input_mode="text",
            voice_enabled=False,
        )

        # 1. El LLM SÍ fue llamado
        assert len(llm_called) == 1

        # 2. NO se envió calendar_card
        card_msgs = [m for m in ws.sent_json if m.get("type") == "calendar_card"]
        assert len(card_msgs) == 0

        # 3. Se recibieron tokens de streaming y texto final
        chunks = [m for m in ws.sent_json if m.get("type") == "llm_chunk"]
        assert len(chunks) == 3
        assistant_msgs = [m for m in ws.sent_json if m.get("type") == "assistant_text"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["text"] == "Respuesta del LLM."

    @pytest.mark.asyncio
    async def test_hybrid_mode_sends_card_on_tool_execution(self, monkeypatch):
        """
        En consultas de razonamiento ('Resume mi semana y dime qué priorizar'), el LLM
        se ejecuta normalmente; si llama a calendar_read_range, la callback on_tool_execute
        emite el calendar_card al frontend de forma transparente.
        """
        ws = _make_mock_websocket()
        ctx = server._PipelineContext()

        async def fake_stream_chat_with_tool_call(*args, **kwargs):
            on_tool_execute = kwargs.get("on_tool_execute")
            if on_tool_execute:
                # Simular que el LLM ejecutó calendar_read_range
                await on_tool_execute("calendar_read_range", {"start_date": "2026-10-19", "end_date": "2026-10-25"}, "resultado")
            yield "Resumen de la semana: te recomiendo priorizar..."

        monkeypatch.setattr("server.stream_chat_with_tools", fake_stream_chat_with_tool_call)

        await server._run_llm_and_tts(
            websocket=ws,
            user_text="Resume mi semana y dime qué priorizar",
            history=[],
            ctx=ctx,
            input_mode="text",
            voice_enabled=False,
        )

        # Se emitió la tarjeta al frontend mediante la callback híbrida
        card_msgs = [m for m in ws.sent_json if m.get("type") == "calendar_card"]
        assert len(card_msgs) == 1
        assert card_msgs[0]["view_type"] in ("week", "range")

        # Y el LLM generó su respuesta conversacional de resumen
        assistant_msgs = [m for m in ws.sent_json if m.get("type") == "assistant_text"]
        assert len(assistant_msgs) == 1
        assert "priorizar" in assistant_msgs[0]["text"]

    @pytest.mark.asyncio
    async def test_followup_gracias_after_calendar_does_not_repeat_events(self, monkeypatch):
        """
        Garantiza que si el historial contiene una interacción previa de calendario
        y el usuario dice 'gracias Kelly', el historial enviado al LLM se sanea sin datos
        crudos y el LLM responde con cortesía breve sin volcar listas de eventos.
        """
        ws = _make_mock_websocket()
        ctx = server._PipelineContext()

        # Simulamos que en el historial el cliente envió texto que contenía eventos de calendario
        raw_calendar_history = [
            {"role": "user", "content": "¿Qué tengo la próxima semana?"},
            {"role": "assistant", "content": "Lunes 19 de octubre: 08:00 Transporte | 10:00 FAS\nMartes 20: 08:00 Transporte..."},
        ]

        received_messages_in_llm = []

        async def fake_stream_chat(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            received_messages_in_llm.extend(msgs)
            yield "¡De nada! Si necesitas algo más, avísame."

        monkeypatch.setattr("server.stream_chat_with_tools", fake_stream_chat)

        await server._run_llm_and_tts(
            websocket=ws,
            user_text="gracias Kelly",
            history=raw_calendar_history,
            ctx=ctx,
            input_mode="text",
            voice_enabled=False,
        )

        # 1. Verificar que el LLM NO recibió el texto de eventos crudos en su historial
        assert len(received_messages_in_llm) > 0
        for m in received_messages_in_llm:
            content = m.get("content", "")
            if m.get("role") == "assistant":
                assert "08:00 Transporte" not in content
                assert "[Sistema: Se mostró al usuario la agenda visual en pantalla. Los datos exactos están en la UI.]" in content

        # 2. Verificar que la respuesta enviada al usuario es corta y NO incluye eventos
        assistant_msgs = [m for m in ws.sent_json if m.get("type") == "assistant_text"]
        assert len(assistant_msgs) == 1
        resp_text = assistant_msgs[0]["text"]
        assert "De nada" in resp_text
        for forbidden in ["Lunes", "08:00", "Transporte", "FAS"]:
            assert forbidden not in resp_text

        # 3. No se emitió ningún calendar_card en respuesta a un simple agradecimiento
        card_msgs = [m for m in ws.sent_json if m.get("type") == "calendar_card"]
        assert len(card_msgs) == 0

