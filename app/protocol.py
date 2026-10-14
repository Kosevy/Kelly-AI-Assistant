"""
app/protocol.py — Funciones puras para validar y sanitizar mensajes WebSocket del cliente.

Rationale:
- Desacopla la validación de payloads de la lógica de red y del framework FastAPI/WebSockets.
- Permite testing unitario exhaustivo de text_message, historial y adjuntos,
  garantizando que entradas maliciosas o malformadas sean filtradas sin excepciones no controladas.
"""
from __future__ import annotations

from typing import Any


def parse_history_payload(history_raw: Any) -> list[dict[str, str]]:
    """
    Sanitiza y valida el historial conversacional recibido del cliente.

    Filtra entradas sin claves obligatorias, con roles inválidos o con contenidos no textuales.
    Retorna una lista de diccionarios con 'role' ('user' o 'assistant') y 'content'.
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


def parse_attachment_ids_payload(raw: Any) -> list[str]:
    """
    Sanitiza y valida la lista de IDs de adjuntos (attachment_ids).

    Filtra elementos no-strings o strings vacíos.
    """
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if isinstance(item, str) and item.strip()]


def parse_text_message_payload(
    msg: dict[str, Any],
) -> tuple[str, list[dict[str, str]], bool, list[str]]:
    """
    Parsea un payload 'text_message' recibido por WebSocket.

    Returns:
        (clean_text, sanitized_history, voice_enabled, attachment_ids)

    Raises:
        ValueError: Si el campo 'text' falta, es nulo o está compuesto solo por espacios.
    """
    raw_text = msg.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("El campo 'text' debe ser una cadena de texto no vacía.")

    clean_text = raw_text.strip()
    history = parse_history_payload(msg.get("history", []))
    voice_enabled = bool(msg.get("voice_enabled", False))
    attachment_ids = parse_attachment_ids_payload(msg.get("attachment_ids", []))

    return clean_text, history, voice_enabled, attachment_ids
