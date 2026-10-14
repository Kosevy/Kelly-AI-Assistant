"""
tests/test_file_sandbox.py
Pruebas exhaustivas para app.file_sandbox (validación de rutas y prevención de Path Traversal).
"""
import os
from pathlib import Path
import pytest

from app.file_sandbox import is_allowed, resolve_safe_path


def test_sandbox_allowed_path():
    roots = [r"D:\Git projects", r"D:\Ciclo 6"]
    assert is_allowed(Path(r"D:\Git projects\foo.txt"), roots) is True
    assert is_allowed(Path(r"D:\Ciclo 6\notas.pdf"), roots) is True


def test_sandbox_case_insensitivity_windows():
    roots = [r"D:\Git projects"]
    # En Windows, 'd:\git projects\sub\file.py' debe coincidir sin distinción de mayúsculas
    assert is_allowed(Path(r"d:\git projects\sub\file.py"), roots) is True


def test_sandbox_blocks_path_traversal():
    roots = [r"D:\Git projects"]
    dangerous_path = r"D:\Git projects\..\Windows\system32\config\SAM"
    assert is_allowed(Path(dangerous_path), roots) is False

    with pytest.raises(PermissionError, match="fuera de los directorios permitidos"):
        resolve_safe_path(dangerous_path, allowed_roots=roots)


def test_sandbox_blocks_external_drives_and_folders():
    roots = [r"D:\Git projects", r"D:\Ciclo 6"]
    unauthorized = r"C:\Users\admin\Documents\notas.pdf"
    assert is_allowed(Path(unauthorized), roots) is False

    with pytest.raises(PermissionError):
        resolve_safe_path(unauthorized, allowed_roots=roots)


def test_sandbox_similar_prefix_not_allowed():
    # 'D:\Git projects_evil' NO debe considerarse dentro de 'D:\Git projects'
    roots = [r"D:\Git projects"]
    similar_path = r"D:\Git projects_evil\secret.txt"
    assert is_allowed(Path(similar_path), roots) is False

    with pytest.raises(PermissionError):
        resolve_safe_path(similar_path, allowed_roots=roots)


def test_sandbox_blocks_symlink_pointing_outside(tmp_path: Path):
    # Crear carpeta sandbox simulada y archivo exterior
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret_file = outside_dir / "secret.txt"
    secret_file.write_text("sensible", encoding="utf-8")

    link_path = sandbox_dir / "link_to_secret.txt"
    try:
        os.symlink(secret_file, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("Creación de symlinks no permitida en este entorno sin privilegios de desarrollador")

    roots = [str(sandbox_dir)]
    # El symlink está físicamente dentro de sandbox_dir, pero apunta afuera -> debe rechazarse
    assert is_allowed(link_path, roots) is False
    with pytest.raises(PermissionError):
        resolve_safe_path(link_path, allowed_roots=roots)
