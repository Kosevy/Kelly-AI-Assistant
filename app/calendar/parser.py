"""
app/calendar/parser.py — Funciones puras e I/O para cargar, guardar y validar el calendario YAML.

Rationale:
- Carga estructurada con PyYAML y Pydantic, garantizando mensajes de error legibles para el LLM.
- Escritura atómica vía archivo temporal + os.replace para evitar corrupción ante caídas del sistema.
- Backup rotativo automático antes de sobreescribir para proteger la fuente de verdad.
"""
from __future__ import annotations

import os
import shutil
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
import yaml
from pydantic import ValidationError

from app.calendar.exceptions import CalendarValidationError
from app.calendar.models import (
    BlockType,
    Calendar,
    Course,
    DayOfWeek,
    Event,
    EventType,
    Metadata,
    RecurringBlock,
)
from app.config import settings
from app.logging_config import logger


def validate_calendar(cal: Calendar) -> list[str]:
    """
    Realiza una validación exhaustiva de integridad sobre la instancia de Calendar.
    Retorna una lista de mensajes de error encontrados (vacía si es totalmente válido).
    """
    errors: list[str] = []

    # Validar fechas de metadata
    if cal.metadata.semester_start >= cal.metadata.semester_end:
        errors.append(
            f"semester_start ({cal.metadata.semester_start}) debe ser anterior a semester_end ({cal.metadata.semester_end})"
        )

    # Validar cursos únicos
    course_ids: set[str] = set()
    for c in cal.courses:
        if c.id in course_ids:
            errors.append(f"Curso duplicado con id '{c.id}'")
        course_ids.add(c.id)

    # Validar bloques recurrentes
    for b in cal.recurring_blocks:
        if b.course_id and b.course_id not in course_ids:
            errors.append(f"Bloque '{b.name}' referencia course_id inexistente: '{b.course_id}'")
        if b.from_date >= b.to_date:
            errors.append(f"Bloque '{b.name}': from_date ({b.from_date}) no es anterior a to_date ({b.to_date})")
        if b.end_time <= b.start_time:
            errors.append(f"Bloque '{b.name}': end_time ({b.end_time}) no es posterior a start_time ({b.start_time})")

    # Validar eventos
    seen_event_ids: set[str] = set()
    for ev in cal.events:
        if ev.id in seen_event_ids:
            errors.append(f"ID de evento duplicado: '{ev.id}'")
        seen_event_ids.add(ev.id)

        if ev.course_id and ev.course_id not in course_ids:
            errors.append(f"Evento '{ev.id}' referencia course_id inexistente: '{ev.course_id}'")

        if ev.duration_minutes is not None and ev.duration_minutes <= 0:
            errors.append(f"Evento '{ev.id}': duration_minutes ({ev.duration_minutes}) debe ser mayor a 0")

        if ev.weight is not None and not (0.0 <= ev.weight <= 1.0):
            errors.append(f"Evento '{ev.id}': weight ({ev.weight}) debe estar entre 0.0 y 1.0")

    return errors


def _calendar_to_clean_dict(cal: Calendar) -> dict[str, Any]:
    """
    Convierte el modelo Calendar a una estructura de diccionarios limpia para YAML,
    omitiendo campos opcionales que sean None y formateando tipos nativos.
    """
    res: dict[str, Any] = {
        "metadata": {
            "student_name": cal.metadata.student_name,
            "semester": cal.metadata.semester,
            "semester_start": cal.metadata.semester_start.isoformat(),
            "semester_end": cal.metadata.semester_end.isoformat(),
            "exam_weeks": dict(cal.metadata.exam_weeks),
        },
        "courses": [],
        "recurring_blocks": [],
        "events": [],
    }

    for c in cal.courses:
        c_dict: dict[str, Any] = {
            "id": c.id,
            "name": c.name,
            "difficulty": c.difficulty,
            "sessions_per_week": c.sessions_per_week,
        }
        if c.topics:
            c_dict["topics"] = list(c.topics)
        res["courses"].append(c_dict)

    for b in cal.recurring_blocks:
        b_dict: dict[str, Any] = {
            "name": b.name,
        }
        if b.course_id:
            b_dict["course_id"] = b.course_id
        b_dict["type"] = b.type.value if hasattr(b.type, "value") else str(b.type)
        b_dict["day_of_week"] = [
            d.value if hasattr(d, "value") else str(d) for d in b.day_of_week
        ]
        b_dict["start_time"] = b.start_time.strftime("%H:%M")
        b_dict["end_time"] = b.end_time.strftime("%H:%M")
        b_dict["from_date"] = b.from_date.isoformat()
        b_dict["to_date"] = b.to_date.isoformat()
        res["recurring_blocks"].append(b_dict)

    for ev in cal.events:
        ev_dict: dict[str, Any] = {
            "id": ev.id,
            "name": ev.name,
            "type": ev.type.value if hasattr(ev.type, "value") else str(ev.type),
        }
        if ev.course_id:
            ev_dict["course_id"] = ev.course_id
        ev_dict["date"] = ev.date.isoformat()
        if ev.start_time:
            ev_dict["start_time"] = ev.start_time.strftime("%H:%M")
        if ev.duration_minutes is not None:
            ev_dict["duration_minutes"] = ev.duration_minutes
        if ev.weight is not None:
            ev_dict["weight"] = ev.weight
        if ev.topics_covered:
            ev_dict["topics_covered"] = list(ev.topics_covered)
        if ev.estimated_hours is not None:
            ev_dict["estimated_hours"] = ev.estimated_hours
        if ev.notes is not None:
            ev_dict["notes"] = ev.notes
        res["events"].append(ev_dict)

    return res


def load_calendar(path: Path) -> Calendar:
    """
    Carga y valida el archivo de calendario YAML.
    Lanza CalendarValidationError con explicaciones precisas si falla.
    """
    if not path.exists():
        raise CalendarValidationError(f"El archivo de calendario no existe: '{path}'")

    if not path.is_file():
        raise CalendarValidationError(f"La ruta del calendario no es un archivo regular: '{path}'")

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise CalendarValidationError(f"Error al leer el archivo de calendario '{path}': {exc}")

    if not content.strip():
        raise CalendarValidationError(f"El archivo de calendario '{path}' está vacío.")

    try:
        raw_data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise CalendarValidationError(f"Error de sintaxis YAML en '{path}': {exc}")

    if not isinstance(raw_data, dict):
        raise CalendarValidationError(
            f"El archivo YAML '{path}' debe contener un mapa (diccionario) en la raíz."
        )

    try:
        cal = Calendar(**raw_data)
    except ValidationError as val_err:
        err_msgs = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in val_err.errors()]
        raise CalendarValidationError("Error de validación del esquema de calendario", errors=err_msgs)

    semantic_errors = validate_calendar(cal)
    if semantic_errors:
        raise CalendarValidationError("Errores de validación semántica del calendario", errors=semantic_errors)

    logger.debug(f"[calendar:parser] Calendario cargado exitosamente desde '{path}'")
    return cal


def _rotate_backups(backup_dir: Path, max_count: int) -> None:
    """Conserva únicamente las últimas max_count copias de respaldo."""
    if max_count <= 0:
        return

    backups = sorted(
        backup_dir.glob("calendar_*.yaml"),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    if len(backups) > max_count:
        excess = len(backups) - max_count
        for old_file in backups[:excess]:
            try:
                old_file.unlink()
                logger.debug(f"[calendar:backup] Respaldo antiguo rotado y eliminado: {old_file.name}")
            except Exception as exc:
                logger.warning(f"[calendar:backup] No se pudo eliminar respaldo antiguo {old_file}: {exc}")


def save_calendar(
    calendar: Calendar,
    path: Path,
    backup: bool = True,
    backup_dir: Path | None = None,
    backup_count: int | None = None,
) -> None:
    """
    Guarda el calendario en formato YAML aplicando validación previa, backup automático
    rotativo y escritura atómica (tmp + os.replace).
    """
    errors = validate_calendar(calendar)
    if errors:
        raise CalendarValidationError(
            "No se puede guardar un calendario con errores de validación",
            errors=errors,
        )

    # Backup automático si el archivo actual ya existe
    if backup and path.exists():
        target_bdir = backup_dir or settings.get_calendar_backup_dir()
        target_bcount = backup_count if backup_count is not None else settings.calendar_backup_count
        try:
            target_bdir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = target_bdir / f"calendar_{ts}.yaml"

            counter = 1
            while backup_file.exists():
                backup_file = target_bdir / f"calendar_{ts}_{counter}.yaml"
                counter += 1

            shutil.copy2(path, backup_file)
            logger.info(f"[calendar:backup] Backup generado: {backup_file.name}")
            _rotate_backups(target_bdir, target_bcount)
        except Exception as exc:
            logger.error(f"[calendar:backup] Error generando respaldo de {path}: {exc}")
            raise CalendarValidationError(f"Error al generar respaldo de seguridad del calendario: {exc}")

    # Escritura atómica
    clean_dict = _calendar_to_clean_dict(calendar)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f"{path.name}.tmp"

    try:
        yaml_str = yaml.dump(
            clean_dict,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        tmp_path.write_text(yaml_str, encoding="utf-8")
        os.replace(tmp_path, path)
        logger.info(f"[calendar:parser] Calendario guardado atómicamente en '{path}'")
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        logger.error(f"[calendar:parser] Fallo en la escritura de '{path}': {exc}")
        raise CalendarValidationError(f"Fallo durante la persistencia atómica del calendario: {exc}")


def init_calendar_if_missing(
    calendar_path: Path | None = None,
    template_path: Path | None = None,
) -> bool:
    """
    Inicializa el archivo de calendario copiando el template si no existe aún en disco.
    Registra en log INFO: '[calendar] Calendario inicializado desde template'.
    Retorna True si fue inicializado desde template, False si ya existía.
    """
    cal_path = calendar_path or settings.get_calendar_path()
    tpl_path = template_path or Path("data/calendar.yaml.example")

    if not cal_path.exists():
        if tpl_path.exists():
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tpl_path, cal_path)
            logger.info("[calendar] Calendario inicializado desde template")
            return True
        else:
            logger.warning(f"[calendar] Template '{tpl_path}' no encontrado para inicializar '{cal_path}'")
            return False

    return False
