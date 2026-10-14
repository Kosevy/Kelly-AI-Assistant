"""
app/file_extractors.py — Funciones puras para extracción de texto estructurado.

Rationale:
- PyMuPDF (fitz) proporciona extracción veloz de PDFs conservando bloques y layout básico.
- Archivos de código y texto se leen con decodificación UTF-8 estricta y fallback a latin-1.
- Mantiene funciones puras independientes de estado global o framework web para testeo directo.
"""
from __future__ import annotations

from pathlib import Path
import fitz  # PyMuPDF

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".py",
    ".md",
    ".txt",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".ts",
    ".js",
    ".java",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".sql",
}


def extract_pdf(path: Path) -> str:
    """
    Extrae texto de un PDF utilizando PyMuPDF (fitz).

    Retorna las páginas separadas por encabezados para dar estructura al LLM.
    Lanza FileNotFoundError si el archivo no existe o RuntimeError si está corrupto.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Archivo PDF no encontrado: {path}")

    pages_text: list[str] = []
    try:
        with fitz.open(path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if text:
                    pages_text.append(f"--- Página {page_num + 1} ---\n{text}")
    except Exception as exc:
        raise RuntimeError(f"Error extrayendo texto del PDF '{path.name}': {exc}") from exc

    return "\n\n".join(pages_text)


def extract_text(path: Path) -> str:
    """
    Lee archivos de texto o código en disco.

    Intenta decodificar como UTF-8; si encuentra errores de codificación,
    utiliza fallback transparente a latin-1 para evitar excepciones de decodificación.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Archivo de texto no encontrado: {path}")

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def extract_by_extension(path: Path) -> str:
    """
    Despacha la extracción según la extensión del archivo.

    Lanza ValueError si el formato no está soportado.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extensión '{ext}' no soportada. Extensiones válidas: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        return extract_pdf(path)
    return extract_text(path)
