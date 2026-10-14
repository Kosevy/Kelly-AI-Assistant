"""
app/file_chunker.py — Control de presupuesto de tokens para archivos adjuntos y leídos.

Rationale:
- Utiliza la heurística estándar de 1 token ≈ 4 caracteres (tokens_per_char = 0.25)
  para estimar y truncar archivos de texto/código sin requerir tokenización pesada en CPU.
- Preserva el inicio del archivo (encabezados, importaciones, primeras páginas de PDFs)
  cuando el contenido supera el presupuesto máximo permitido.
"""
from __future__ import annotations


def estimate_tokens(text: str, tokens_per_char: float = 0.25) -> int:
    """Calcula una estimación rápida de tokens a partir de la longitud en caracteres."""
    return int(len(text) * tokens_per_char)


def truncate_to_token_budget(
    text: str,
    max_tokens: int,
    tokens_per_char: float = 0.25,
) -> tuple[str, bool]:
    """
    Trunca texto al presupuesto de tokens estimado.

    Retorna una tupla (texto_procesado, fue_truncado).
    Si el texto excede el presupuesto, se conservan los primeros N caracteres
    correspondientes a max_tokens y se indica fue_truncado=True.
    """
    if max_tokens <= 0:
        return "", bool(text)

    max_chars = int(max_tokens / tokens_per_char)
    if len(text) > max_chars:
        return text[:max_chars], True

    return text, False
