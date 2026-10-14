"""
tests/test_protocol.py
Unit tests for app.protocol (WebSocket message parsing and sanitization).
"""
import pytest
from app.protocol import parse_history_payload, parse_text_message_payload


class TestParseHistoryPayload:
    def test_valid_history(self):
        raw = [
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Hola, ¿en qué te ayudo?"},
        ]
        parsed = parse_history_payload(raw)
        assert len(parsed) == 2
        assert parsed[0] == {"role": "user", "content": "Hola"}
        assert parsed[1] == {"role": "assistant", "content": "Hola, ¿en qué te ayudo?"}

    def test_filters_invalid_roles(self):
        raw = [
            {"role": "system", "content": "Ignorar"},
            {"role": "admin", "content": "Ignorar"},
            {"role": "user", "content": "Válido"},
        ]
        parsed = parse_history_payload(raw)
        assert len(parsed) == 1
        assert parsed[0]["role"] == "user"

    def test_filters_malformed_entries(self):
        raw = [
            "no es dict",
            {"role": "user"},  # falta content
            {"content": "sin role"},
            {"role": "user", "content": 12345},  # content no es string
            None,
        ]
        parsed = parse_history_payload(raw)
        assert parsed == []

    def test_non_list_returns_empty(self):
        assert parse_history_payload(None) == []
        assert parse_history_payload("historial") == []
        assert parse_history_payload({"role": "user"}) == []


class TestParseTextMessagePayload:
    def test_valid_text_message(self):
        msg = {
            "type": "text_message",
            "text": "  Explícame qué es un puntero en C++  ",
            "history": [{"role": "user", "content": "Hola"}],
            "voice_enabled": False,
            "attachment_ids": ["uuid-123", "uuid-456"],
        }
        text, history, voice_enabled, attachment_ids = parse_text_message_payload(msg)
        assert text == "Explícame qué es un puntero en C++"
        assert len(history) == 1
        assert voice_enabled is False
        assert attachment_ids == ["uuid-123", "uuid-456"]

    def test_default_voice_enabled_is_false(self):
        msg = {"type": "text_message", "text": "Hola"}
        text, history, voice_enabled, attachment_ids = parse_text_message_payload(msg)
        assert text == "Hola"
        assert history == []
        assert voice_enabled is False
        assert attachment_ids == []

    def test_voice_enabled_true(self):
        msg = {"type": "text_message", "text": "Hola", "voice_enabled": True}
        text, history, voice_enabled, attachment_ids = parse_text_message_payload(msg)
        assert voice_enabled is True
        assert attachment_ids == []

    def test_attachment_ids_sanitization(self):
        msg = {
            "type": "text_message",
            "text": "Hola",
            "attachment_ids": ["uuid-1", 1234, "", "  ", "uuid-2", None],
        }
        _, _, _, attachment_ids = parse_text_message_payload(msg)
        assert attachment_ids == ["uuid-1", "uuid-2"]

    def test_empty_or_whitespace_text_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_text_message_payload({"type": "text_message", "text": ""})

        with pytest.raises(ValueError):
            parse_text_message_payload({"type": "text_message", "text": "   \n\t  "})

    def test_missing_text_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_text_message_payload({"type": "text_message"})
