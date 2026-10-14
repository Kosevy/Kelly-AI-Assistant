"""
tests/test_file_extractors.py
Pruebas unitarias para app.file_extractors (extracción de texto de PDFs y archivos de código/texto).
"""
from pathlib import Path
import fitz
import pytest

from app.file_extractors import extract_by_extension, extract_pdf, extract_text, SUPPORTED_EXTENSIONS


def test_extract_python_file(tmp_path: Path):
    py_file = tmp_path / "sample.py"
    py_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    content = extract_by_extension(py_file)
    assert "def hello():" in content
    assert "return 'world'" in content


def test_extract_markdown_file(tmp_path: Path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Título\nContenido en español con tildes: árbol, canción.", encoding="utf-8")

    content = extract_by_extension(md_file)
    assert "# Título" in content
    assert "canción" in content


def test_extract_latin1_fallback(tmp_path: Path):
    latin1_file = tmp_path / "latin1.txt"
    # Escribir bytes que no son UTF-8 válidos pero sí latin-1 (ej: byte 0xe9 = é)
    latin1_file.write_bytes(b"Texto en latin1: \xe9\xe1\xed")

    content = extract_text(latin1_file)
    assert "Texto en latin1:" in content


def test_extract_pdf_file(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Página uno: Introducción a Kelly.")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Página dos: Arquitectura del sistema.")
    doc.save(str(pdf_path))
    doc.close()

    content = extract_pdf(pdf_path)
    assert "--- Página 1 ---" in content
    assert "Introducción a Kelly" in content
    assert "--- Página 2 ---" in content
    assert "Arquitectura del sistema" in content


def test_unsupported_extension_raises_value_error(tmp_path: Path):
    exe_file = tmp_path / "danger.exe"
    exe_file.write_bytes(b"binary data")

    with pytest.raises(ValueError, match="no soportada"):
        extract_by_extension(exe_file)


def test_missing_file_raises_file_not_found(tmp_path: Path):
    missing_file = tmp_path / "non_existent.py"
    with pytest.raises(FileNotFoundError):
        extract_text(missing_file)
