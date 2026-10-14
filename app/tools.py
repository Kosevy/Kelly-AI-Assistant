"""
app/tools.py — Definición y ejecución de herramientas locales para Function Calling con LLM.

Rationale:
- Expone herramientas bajo el esquema estándar de OpenAI / JSON Schema compatible con Qwen2.5.
- La ejecución siempre aplica validación estricta de rutas mediante app.file_sandbox antes
  de cualquier operación de I/O, previniendo acceso a archivos del sistema operativo.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.calendar.tools import CALENDAR_TOOLS, execute_calendar_tool
from app.config import settings
from app.file_sandbox import resolve_safe_path
from app.file_extractors import extract_by_extension
from app.file_chunker import truncate_to_token_budget
from app.logging_config import logger

FILE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Lee el contenido textual de un archivo local (PDF, Python, Markdown, TXT, código). "
                "Solo funciona dentro de los directorios permitidos por la política de seguridad."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Ruta absoluta del archivo a leer (ej: D:\\Git projects\\notas.md)",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "Lista los archivos y subcarpetas contenidos en un directorio local. "
                "Solo funciona dentro de los directorios permitidos por la política de seguridad."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Ruta absoluta del directorio a inspeccionar",
                    }
                },
                "required": ["path"],
            },
        },
    },
]

TOOLS: list[dict[str, Any]] = FILE_TOOLS + CALENDAR_TOOLS


def _format_size(size_bytes: int) -> str:
    """Formatea bytes a KB o MB legibles."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


async def execute_tool(
    name: str,
    args: dict[str, Any],
    allowed_roots: list[str] | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Ejecuta una herramienta solicitada por el LLM y retorna el resultado como string.

    Captura excepciones de seguridad o I/O y retorna mensajes informativos
    para que el LLM pueda explicar la situación al usuario sin colapsar.
    """
    roots = allowed_roots if allowed_roots is not None else settings.allowed_paths
    token_budget = max_tokens if max_tokens is not None else settings.max_file_tokens

    if name == "read_file":
        raw_path = args.get("path", "")
        if not raw_path:
            return "Error: Falta el parámetro requerido 'path'."

        try:
            safe_path = resolve_safe_path(raw_path, allowed_roots=roots)
        except PermissionError as perm_err:
            logger.warning(f"[tool:read_file] Sandbox blocked: {raw_path}")
            return f"Acceso denegado: {perm_err}"

        if not safe_path.is_file():
            return f"Error: El archivo '{raw_path}' no existe o no es un archivo regular."

        try:
            raw_text = extract_by_extension(safe_path)
            truncated_text, was_truncated = truncate_to_token_budget(raw_text, max_tokens=token_budget)
            
            prefix = ""
            if was_truncated:
                prefix = f"[Aviso: El archivo '{safe_path.name}' fue truncado a los primeros {token_budget} tokens por límite de contexto]\n\n"

            result = prefix + truncated_text
            logger.info(f"[tool] read_file(path='{safe_path}') -> {len(result)} chars (truncated={was_truncated})")
            return result
        except Exception as exc:
            logger.error(f"[tool:read_file] Error procesando '{safe_path}': {exc}")
            return f"Error al leer el archivo '{safe_path.name}': {exc}"

    elif name == "list_directory":
        raw_path = args.get("path", "")
        if not raw_path:
            return "Error: Falta el parámetro requerido 'path'."

        try:
            safe_path = resolve_safe_path(raw_path, allowed_roots=roots)
        except PermissionError as perm_err:
            logger.warning(f"[tool:list_directory] Sandbox blocked: {raw_path}")
            return f"Acceso denegado: {perm_err}"

        if not safe_path.is_dir():
            return f"Error: La ruta '{raw_path}' no existe o no es un directorio."

        try:
            entries = sorted(safe_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            lines = [f"Contenido del directorio: {safe_path}"]
            dir_count = 0
            file_count = 0

            for entry in entries:
                if entry.is_dir():
                    lines.append(f"  [DIR]  {entry.name}/")
                    dir_count += 1
                else:
                    try:
                        sz = _format_size(entry.stat().st_size)
                    except Exception:
                        sz = "desconocido"
                    lines.append(f"  [FILE] {entry.name} ({sz})")
                    file_count += 1

            lines.append(f"\nTotal: {dir_count} carpetas, {file_count} archivos.")
            result = "\n".join(lines)
            logger.info(f"[tool] list_directory(path='{safe_path}') -> {len(entries)} items")
            return result
        except Exception as exc:
            logger.error(f"[tool:list_directory] Error listando '{safe_path}': {exc}")
            return f"Error al listar directorio '{safe_path}': {exc}"

    elif name.startswith("calendar_"):
        return await execute_calendar_tool(name, args)

    return f"Error: Herramienta desconocida '{name}'."
