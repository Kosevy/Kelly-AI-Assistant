"""
tests/test_tools.py
Pruebas unitarias para app.tools (ejecución de herramientas y sandboxing integrado).
"""
from pathlib import Path
import pytest

from app.tools import execute_tool, TOOLS


@pytest.mark.asyncio
async def test_read_file_within_sandbox(tmp_path: Path):
    safe_file = tmp_path / "hello.py"
    safe_file.write_text("print('test tools')", encoding="utf-8")

    result = await execute_tool(
        "read_file",
        {"path": str(safe_file)},
        allowed_roots=[str(tmp_path)],
    )
    assert "print('test tools')" in result


@pytest.mark.asyncio
async def test_read_file_outside_sandbox(tmp_path: Path):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secreto", encoding="utf-8")

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()

    result = await execute_tool(
        "read_file",
        {"path": str(outside_file)},
        allowed_roots=[str(sandbox_dir)],
    )
    assert "Acceso denegado" in result


@pytest.mark.asyncio
async def test_read_file_nonexistent(tmp_path: Path):
    result = await execute_tool(
        "read_file",
        {"path": str(tmp_path / "missing.txt")},
        allowed_roots=[str(tmp_path)],
    )
    assert "no existe" in result


@pytest.mark.asyncio
async def test_list_directory_within_sandbox(tmp_path: Path):
    (tmp_path / "sub_folder").mkdir()
    (tmp_path / "doc.txt").write_text("contenido", encoding="utf-8")

    result = await execute_tool(
        "list_directory",
        {"path": str(tmp_path)},
        allowed_roots=[str(tmp_path)],
    )
    assert "[DIR]  sub_folder/" in result
    assert "[FILE] doc.txt" in result
    assert "Total: 1 carpetas, 1 archivos." in result


@pytest.mark.asyncio
async def test_list_directory_outside_sandbox(tmp_path: Path):
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()

    result = await execute_tool(
        "list_directory",
        {"path": r"C:\Windows\System32"},
        allowed_roots=[str(sandbox_dir)],
    )
    assert "Acceso denegado" in result


@pytest.mark.asyncio
async def test_execute_unknown_tool():
    result = await execute_tool("launch_missile", {"target": "moon"})
    assert "desconocida" in result
