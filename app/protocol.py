"""
app/protocol.py
Pure functions to validate and parse client WebSocket messages.

Rationale:
  Decouples WebSocket payload sanitization and validation from networking logic.
  Allows comprehensive unit testing of text_message and history parsing,
  guaranteeing that malformed or malicious client payloads are filtered safely.
"""
from __future__ import annotations

from typing import Any


def parse_history_payload(history_raw: Any) -> list[dict[str, str]]:
    """
    Sanitize and validate conversational history from client payload.

    Filters out entries that lack required keys, invalid roles, or non-string contents.
    Returns a list of dicts with keys 'role' and 'content'.
    """
    if not isinstance(history_raw, list):
        return []

    valid: list[dict[str, str]] = []
    for item in history_raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            valid.append({"role": role, "content": content})
    return valid


def parse_text_message_payload(msg: dict[str, Any]) -> tuple[str, list[dict[str, str]], bool]:
    """
    Parse a text_message payload received over WebSocket.

    Returns:
        (clean_text, sanitized_history, voice_enabled)

    Raises:
        ValueError: If text is missing, empty, or whitespace-only.
    """
    raw_text = msg.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Field 'text' must be a non-empty string.")

    clean_text = raw_text.strip()
    history = parse_history_payload(msg.get("history", []))
    voice_enabled = bool(msg.get("voice_enabled", False))

    return clean_text, history, voice_enabled
