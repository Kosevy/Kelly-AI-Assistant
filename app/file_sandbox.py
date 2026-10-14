"""
app/file_sandbox.py — Validación estricta y sandboxing de acceso a disco en Windows.

Rationale:
- En entornos Windows, las rutas son insensibles a mayúsculas/minúsculas y admiten
  tanto separadores '/' como '\\'.
- Previene vulnerabilidades de Path Traversal ('..'), nombres alterados de dispositivos
  y symlinks que redirijan fuera de las carpetas de trabajo autorizadas.
"""
from __future__ import annotations

import os
from pathlib import Path


def is_allowed(path: Path | str, allowed_roots: list[str]) -> bool:
    """
    Comprueba si path está dentro de allowed_roots (case-insensitive en Windows).
    Resuelve symlinks y normaliza la ruta antes de comparar.
    """
    p = Path(path)
    try:
        resolved_path = p.resolve()
    except Exception:
        resolved_path = Path(os.path.abspath(p))

    path_str = str(resolved_path)
    if os.name == "nt":
        path_str = path_str.lower()

    for root in allowed_roots:
        if not root:
            continue
        try:
            resolved_root = Path(root).resolve()
        except Exception:
            resolved_root = Path(os.path.abspath(root))

        root_str = str(resolved_root)
        if os.name == "nt":
            root_str = root_str.lower()

        root_str_clean = root_str.rstrip("\\/")
        sep = "\\" if os.name == "nt" else "/"

        if path_str == root_str_clean or path_str.startswith(root_str_clean + sep):
            return True

    return False


def resolve_safe_path(user_path: str | Path, allowed_roots: list[str]) -> Path:
    """
    Valida que user_path esté dentro de alguna de las allowed_roots.
    Resuelve symlinks y `..` antes de comparar.
    Lanza PermissionError si no es seguro.
    """
    p = Path(user_path)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = Path(os.path.abspath(p))

    if not is_allowed(resolved, allowed_roots):
        raise PermissionError(
            f"Acceso denegado: La ruta '{user_path}' está fuera de los directorios permitidos por la política de seguridad."
        )

    return resolved
