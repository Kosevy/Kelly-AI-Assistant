"""
app/calendar/parser.py — Funciones puras e I/O para cargar, guardar y validar el calendario YAML.

Rationale:
- DuplicateKeyLoader: PyYAML sobreescribe claves duplicadas silenciosamente (e.g. dos bloques
  'events:'). El loader custom detecta esto y lanza un error ruidoso con número de línea.
- Errores contextuales: validate_calendar() retorna ValidationIssue con section[index] "name"
  y campo afectado, en lugar de strings genéricos. Esto permite mensajes accionables para
  el usuario tanto en logs como en la respuesta de Kelly.
- extra="ignore" + warning: campos desconocidos en el YAML no rompen la carga; se emite
  un warning INFO para cada campo ignorado, guiando al usuario sin detener el servidor.
- Escritura atómica vía archivo temporal + os.replace para evitar corrupción ante caídas.
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

from app.calendar.exceptions import CalendarValidationError, ValidationIssue
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


# ─── DuplicateKeyLoader ───────────────────────────────────────────────────────

class _DuplicateKeyLoader(yaml.SafeLoader):
    """
    Loader PyYAML que detecta claves duplicadas en el mismo nivel del YAML.

    Rationale: yaml.safe_load sobreescribe la primera clave con la segunda
    silenciosamente. El usuario que define 'events:' dos veces pierde todos
    los eventos del primer bloque sin ningún aviso. Este loader lanza
    yaml.constructor.ConstructorError con número de línea.
    """


def _construct_yaml_map_no_duplicates(loader: _DuplicateKeyLoader, node: yaml.MappingNode) -> dict:
    """Constructor de mapping que detecta claves duplicadas."""
    loader.flatten_mapping(node)
    pairs: list[tuple[Any, Any]] = loader.construct_pairs(node)
    seen: dict[str, int] = {}
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"clave duplicada encontrada: '{key}' "
                f"(primera aparición en línea {seen[key] + 1}, "
                f"duplicado en línea {node.start_mark.line + 1}). "
                "Fusiona ambos bloques en uno solo.",
                node.start_mark,
            )
        seen[key] = node.start_mark.line
        result[key] = value
    return result


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_yaml_map_no_duplicates,
)


# ─── Validación semántica contextual ─────────────────────────────────────────

def validate_calendar(cal: Calendar) -> list[ValidationIssue]:
    """
    Realiza una validación exhaustiva de integridad sobre la instancia de Calendar.

    Retorna una lista de ValidationIssue estructurados (vacía si es totalmente válido).
    Cada issue incluye section, index, name, field y message para que el LLM pueda
    transmitir errores accionables al usuario.
    """
    issues: list[ValidationIssue] = []
    course_ids: set[str] = {c.id for c in cal.courses}

    # ── Metadata ────────────────────────────────────────────────────────────
    if cal.metadata.semester_start >= cal.metadata.semester_end:
        issues.append(ValidationIssue(
            section="metadata",
            name="",
            field="semester_start/semester_end",
            message=(
                f"semester_start ({cal.metadata.semester_start}) debe ser "
                f"anterior a semester_end ({cal.metadata.semester_end})"
            ),
        ))

    # ── Courses ──────────────────────────────────────────────────────────────
    seen_course_ids: set[str] = set()
    for i, c in enumerate(cal.courses):
        if c.id in seen_course_ids:
            issues.append(ValidationIssue(
                section="courses",
                index=i,
                name=c.name,
                field="id",
                message=f"id='{c.id}' duplicado. Cada curso debe tener un id único.",
            ))
        seen_course_ids.add(c.id)

    # ── Recurring blocks ─────────────────────────────────────────────────────
    for i, b in enumerate(cal.recurring_blocks):
        if b.course_id and b.course_id not in course_ids:
            available = ", ".join(sorted(course_ids)) if course_ids else "ninguno"
            issues.append(ValidationIssue(
                section="recurring_blocks",
                index=i,
                name=b.name,
                field="course_id",
                message=(
                    f"course_id='{b.course_id}' no existe en courses. "
                    f"IDs disponibles: {available}."
                ),
            ))
        if b.from_date >= b.to_date:
            issues.append(ValidationIssue(
                section="recurring_blocks",
                index=i,
                name=b.name,
                field="from_date/to_date",
                message=(
                    f"from_date ({b.from_date}) no es anterior a to_date ({b.to_date})."
                ),
            ))
        if b.end_time <= b.start_time:
            issues.append(ValidationIssue(
                section="recurring_blocks",
                index=i,
                name=b.name,
                field="start_time/end_time",
                message=(
                    f"end_time ({b.end_time}) no es posterior a start_time ({b.start_time})."
                ),
            ))

    # ── Events ───────────────────────────────────────────────────────────────
    seen_event_ids: set[str] = set()
    for i, ev in enumerate(cal.events):
        if ev.id in seen_event_ids:
            issues.append(ValidationIssue(
                section="events",
                index=i,
                name=ev.name,
                field="id",
                message=f"id='{ev.id}' duplicado. Cada evento debe tener un id único.",
            ))
        seen_event_ids.add(ev.id)

        if ev.course_id and ev.course_id not in course_ids:
            available = ", ".join(sorted(course_ids)) if course_ids else "ninguno"
            issues.append(ValidationIssue(
                section="events",
                index=i,
                name=ev.name,
                field="course_id",
                message=(
                    f"course_id='{ev.course_id}' no existe en courses. "
                    f"IDs disponibles: {available}."
                ),
            ))

        if ev.duration_minutes is not None and ev.duration_minutes <= 0:
            issues.append(ValidationIssue(
                section="events",
                index=i,
                name=ev.name,
                field="duration_minutes",
                message=f"duration_minutes ({ev.duration_minutes}) debe ser mayor a 0.",
            ))

        if ev.weight is not None and not (0.0 <= ev.weight <= 1.0):
            issues.append(ValidationIssue(
                section="events",
                index=i,
                name=ev.name,
                field="weight",
                message=f"weight ({ev.weight}) debe estar entre 0.0 y 1.0.",
            ))

    return issues


def _warn_extra_fields(raw_data: dict[str, Any]) -> None:
    """
    Emite warnings INFO para campos de nivel raíz no reconocidos por el schema.

    Rationale: extra="ignore" en Pydantic tolera campos extra silenciosamente.
    Este helper los detecta antes de construir el modelo para que el usuario sepa
    qué campos están siendo ignorados, sin romper la carga.
    """
    known_root = {"metadata", "courses", "recurring_blocks", "events"}
    for key in raw_data:
        if key not in known_root:
            logger.info(f"[calendar] Campo '{key}' en la raíz del YAML no está en el schema y se ignorará.")

    # Emitir warnings por campos extra en items de listas
    section_models = {
        "courses": {"id", "name", "difficulty", "sessions_per_week", "topics"},
        "recurring_blocks": {"name", "course_id", "type", "day_of_week", "start_time", "end_time", "from_date", "to_date"},
        "events": {"id", "name", "type", "course_id", "date", "start_time", "duration_minutes", "weight", "topics_covered", "estimated_hours", "notes"},
        "metadata": {"student_name", "semester", "semester_start", "semester_end", "exam_weeks"},
    }
    for section, known_fields in section_models.items():
        items = raw_data.get(section)
        if isinstance(items, list):
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    for key in item:
                        if key not in known_fields:
                            name = item.get("name") or item.get("id") or f"#{i}"
                            logger.info(
                                f"[calendar] Campo '{key}' en {section}[{i}] \"{name}\" "
                                "no está en el schema y se ignorará."
                            )
        elif isinstance(items, dict):
            for key in items:
                if key not in known_fields:
                    logger.info(f"[calendar] Campo '{key}' en {section} no está en el schema y se ignorará.")


def _format_pydantic_errors_as_issues(val_err: ValidationError) -> list[ValidationIssue]:
    """
    Convierte errores de ValidationError de Pydantic a ValidationIssue estructurados.

    Rationale: los errores de Pydantic usan tuplas de loc (e.g. ('recurring_blocks', 2, 'type'))
    que se mapean directamente a section, index y field de ValidationIssue.
    """
    issues: list[ValidationIssue] = []
    for e in val_err.errors():
        loc = e.get("loc", ())
        msg = e.get("msg", "error desconocido")
        # Extraer sección, índice y campo del loc
        section = str(loc[0]) if loc else "unknown"
        index: int | None = None
        field = ""
        name = ""
        if len(loc) >= 2 and isinstance(loc[1], int):
            index = loc[1]
            field = ".".join(str(l) for l in loc[2:]) if len(loc) > 2 else ""
        elif len(loc) >= 2:
            field = ".".join(str(l) for l in loc[1:])

        # Limpiar prefijos de Pydantic del mensaje
        clean_msg = msg.removeprefix("Value error, ")

        issues.append(ValidationIssue(
            section=section,
            index=index,
            name=name,
            field=field,
            message=clean_msg,
        ))
    return issues


def load_calendar(path: Path) -> Calendar:
    """
    Carga y valida el archivo de calendario YAML.

    Lanza CalendarValidationError con ValidationIssue estructurados si falla.
    Emite warnings INFO para campos extra desconocidos (no rompe la carga).
    Detecta claves YAML duplicadas al nivel de mapa y falla ruidosamente.
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
        raw_data = yaml.load(content, Loader=_DuplicateKeyLoader)  # noqa: S506
    except yaml.YAMLError as exc:
        raise CalendarValidationError(f"Error de sintaxis YAML en '{path}': {exc}")

    if not isinstance(raw_data, dict):
        raise CalendarValidationError(
            f"El archivo YAML '{path}' debe contener un mapa (diccionario) en la raíz."
        )

    # Emitir warnings para campos extra antes de construir el modelo
    _warn_extra_fields(raw_data)

    try:
        cal = Calendar(**raw_data)
    except ValidationError as val_err:
        pydantic_issues = _format_pydantic_errors_as_issues(val_err)
        raise CalendarValidationError(
            "Error de validación del esquema de calendario",
            issues=pydantic_issues,
        )

    semantic_issues = validate_calendar(cal)
    if semantic_issues:
        raise CalendarValidationError(
            "Errores de validación semántica del calendario",
            issues=semantic_issues,
        )

    logger.debug(f"[calendar:parser] Calendario cargado exitosamente desde '{path}'")
    return cal


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
    issues = validate_calendar(calendar)
    if issues:
        raise CalendarValidationError(
            "No se puede guardar un calendario con errores de validación",
            issues=issues,
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


def check_calendar_health(cal_path: Path | None = None) -> dict[str, Any]:
    """
    Evalúa la salud del calendario YAML para el arranque del servidor y el endpoint /health.

    Rationale:
    - Centraliza la lógica de validación de inicio.
    - Si el calendario es válido, loguea INFO y retorna status='ok'.
    - Si tiene errores de validación, loguea ERROR detallado y retorna status='error' con la lista de errores.
    - No lanza excepciones ni detiene el servidor.
    """
    target_path = cal_path or settings.get_calendar_path()
    try:
        load_calendar(target_path)
        logger.info("[calendar] calendar.yaml válido — listo.")
        return {"status": "ok", "errors": [], "path": str(target_path)}
    except CalendarValidationError as cal_err:
        err_list = cal_err.to_list()
        error_lines = "\n".join(f"  {i + 1}. {e}" for i, e in enumerate(err_list))
        logger.error(
            f"[calendar] ERROR — {target_path.name} tiene {len(err_list)} error(es) de validación:\n"
            f"{error_lines}\n"
            "[calendar] Kelly no podrá usar el calendario hasta que se corrijan."
        )
        return {"status": "error", "errors": err_list, "path": str(target_path)}
    except Exception as exc:
        logger.error(f"[calendar] Error inesperado al validar '{target_path}': {exc}")
        return {"status": "error", "errors": [str(exc)], "path": str(target_path)}
