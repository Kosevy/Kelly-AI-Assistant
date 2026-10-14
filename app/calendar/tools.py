"""
app/calendar/tools.py — Definición y despacho de herramientas locales de calendario para el LLM.

Rationale:
- Expone esquemas JSON Schema compatibles con function calling nativo en Qwen2.5.
- Conecta consultas de disponibilidad y mutaciones del calendario directamente con la
  fuente de verdad YAML, ejecutando backups automáticos en cada modificación.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from unidecode import unidecode

from app.calendar.exceptions import CalendarValidationError
from app.calendar.expander import expand_all
from app.calendar.models import Calendar, Event, EventType
from app.calendar.parser import load_calendar, save_calendar
from app.calendar.query import events_in_range, free_slots
from app.config import settings
from app.logging_config import logger

CALENDAR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calendar_read_range",
            "description": (
                "Lee los eventos y bloques del calendario en un rango de fechas. "
                "Útil para responder '¿qué tengo esta semana?', "
                "'¿qué hay el 12 de mayo?', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Fecha inicio en formato YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "Fecha fin en formato YYYY-MM-DD"},
                    "types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Opcional. Filtra por tipo: exam, entregable, class, meal, transport, etc.",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_free_slots",
            "description": (
                "Calcula bloques de tiempo libre en un rango de fechas. "
                "Útil para '¿tengo tiempo para estudiar mañana?', "
                "'¿cuándo puedo hacer 90 min seguidos?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Fecha inicio en formato YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "Fecha fin en formato YYYY-MM-DD"},
                    "min_block_minutes": {
                        "type": "integer",
                        "default": 30,
                        "description": "Duración mínima del hueco libre en minutos",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_add_event",
            "description": (
                "Añade un evento puntual (parcial, entrega, presentación, práctica, personal) "
                "al calendario. Escribe directo al YAML con backup automático."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nombre descriptivo del evento"},
                    "date": {"type": "string", "description": "Fecha del evento en formato YYYY-MM-DD"},
                    "type": {
                        "type": "string",
                        "enum": ["exam", "entregable", "presentacion", "practica", "personal"],
                        "description": "Tipo de evento",
                    },
                    "course_id": {"type": "string", "description": "ID del curso asociado (opcional)"},
                    "start_time": {"type": "string", "description": "Hora de inicio HH:MM (opcional)"},
                    "duration_minutes": {"type": "integer", "description": "Duración en minutos (opcional)"},
                    "weight": {"type": "number", "description": "Peso en nota final 0.0-1.0 (opcional)"},
                    "topics_covered": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Temas cubiertos en la evaluación (opcional)",
                    },
                    "notes": {"type": "string", "description": "Notas o comentarios adicionales (opcional)"},
                },
                "required": ["name", "date", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_list_courses",
            "description": "Lista los cursos del semestre con su información (nombre, dificultad, temas).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_delete_event",
            "description": "Elimina un evento por su ID. Usar solo cuando el usuario lo pida explícitamente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "Identificador del evento a eliminar"}
                },
                "required": ["event_id"],
            },
        },
    },
]


def _generate_event_id(name: str, event_date: date, existing_ids: set[str]) -> str:
    """Genera un slug único para el evento a partir de su nombre y fecha."""
    clean_name = unidecode(name).lower()
    slug_base = re.sub(r"[^a-z0-9]+", "_", clean_name).strip("_")
    date_str = event_date.strftime("%Y%m%d")
    candidate = f"{slug_base}_{date_str}" if slug_base else f"event_{date_str}"

    if candidate not in existing_ids:
        return candidate

    counter = 1
    while f"{candidate}_{counter}" in existing_ids:
        counter += 1
    return f"{candidate}_{counter}"


async def execute_calendar_tool(
    name: str,
    args: dict[str, Any],
    calendar_path: Path | None = None,
) -> str:
    """
    Despacha y ejecuta herramientas locales de calendario.
    Retorna un string formateado para el contexto del LLM.
    """
    target_path = calendar_path or settings.get_calendar_path()

    if not target_path.exists():
        logger.warning(f"[calendar_tool] Archivo de calendario no encontrado en: {target_path}")
        return (
            f"Error: El archivo de calendario no existe en '{target_path}'. "
            "Copia 'data/calendar.yaml.example' a 'data/calendar.yaml' para inicializarlo."
        )

    try:
        cal = load_calendar(target_path)
    except CalendarValidationError as exc:
        logger.error(f"[calendar_tool] Error cargando calendario '{target_path}': {exc}")
        return f"Error en el calendario: {exc}"

    try:
        if name == "calendar_read_range":
            start_str = args.get("start_date")
            end_str = args.get("end_date")
            if not start_str or not end_str:
                return "Error: Se requieren 'start_date' y 'end_date' (formato YYYY-MM-DD)."

            try:
                start_d = date.fromisoformat(str(start_str).strip())
                end_d = date.fromisoformat(str(end_str).strip())
            except ValueError as ve:
                return f"Error de formato de fecha: {ve}. Usa YYYY-MM-DD."

            types_filter = args.get("types")
            expanded = expand_all(cal, start_d, end_d)
            filtered = events_in_range(expanded, start_d, end_d, types=types_filter)

            if not filtered:
                return f"No hay eventos ni actividades programadas entre {start_d} y {end_d}."

            lines = [f"Actividades y eventos del {start_d} al {end_d}:"]
            for b in filtered:
                time_info = (
                    f"{b.start.strftime('%H:%M')}-{b.end.strftime('%H:%M')}"
                    if b.start != b.end
                    else "[Todo el día]"
                )
                course_info = f" (Curso: {b.course_id})" if b.course_id else ""
                id_info = f" [ID: {b.event_id}]" if b.event_id else ""
                lines.append(f"- {b.date.isoformat()} {time_info} [{b.type}] {b.name}{course_info}{id_info}")

            return "\n".join(lines)

        elif name == "calendar_free_slots":
            start_str = args.get("start_date")
            end_str = args.get("end_date")
            if not start_str or not end_str:
                return "Error: Se requieren 'start_date' y 'end_date' (formato YYYY-MM-DD)."

            try:
                start_d = date.fromisoformat(str(start_str).strip())
                end_d = date.fromisoformat(str(end_str).strip())
            except ValueError as ve:
                return f"Error de formato de fecha: {ve}. Usa YYYY-MM-DD."

            min_minutes = int(args.get("min_block_minutes", 30))
            expanded = expand_all(cal, start_d, end_d)
            slots = free_slots(expanded, start_d, end_d, min_block_minutes=min_minutes)

            if not slots:
                return f"No se encontraron bloques libres de al menos {min_minutes} minutos entre {start_d} y {end_d}."

            lines = [f"Bloques libres (mínimo {min_minutes} min) del {start_d} al {end_d}:"]
            for s, e in slots:
                duration_m = int((e - s).total_seconds() / 60.0)
                lines.append(f"- {s.strftime('%Y-%m-%d %H:%M')}-{e.strftime('%H:%M')} ({duration_m} min)")

            return "\n".join(lines)

        elif name == "calendar_add_event":
            ev_name = args.get("name")
            date_str = args.get("date")
            type_str = args.get("type")

            if not ev_name or not date_str or not type_str:
                return "Error: 'name', 'date' y 'type' son requeridos para añadir un evento."

            try:
                ev_date = date.fromisoformat(str(date_str).strip())
            except ValueError as ve:
                return f"Error en la fecha: {ve}. Usa YYYY-MM-DD."

            try:
                ev_type = EventType(str(type_str).lower().strip())
            except ValueError:
                valid_types = [t.value for t in EventType]
                return f"Error: tipo '{type_str}' inválido. Tipos permitidos: {', '.join(valid_types)}."

            course_id = args.get("course_id")
            if course_id:
                course_id = str(course_id).strip()
                known_courses = {c.id for c in cal.courses}
                if course_id not in known_courses:
                    return (
                        f"Error: course_id '{course_id}' no existe en el calendario. "
                        f"Cursos disponibles: {', '.join(sorted(known_courses)) if known_courses else 'ninguno'}."
                    )

            start_t: time | None = None
            if args.get("start_time"):
                try:
                    raw_time = str(args["start_time"]).strip()
                    parts = raw_time.split(":")
                    start_t = time(int(parts[0]), int(parts[1]))
                except Exception:
                    return f"Error: formato de hora inválido en '{args.get('start_time')}'. Usa HH:MM."

            duration = int(args["duration_minutes"]) if args.get("duration_minutes") is not None else None
            weight = float(args["weight"]) if args.get("weight") is not None else None
            topics = [str(t) for t in args.get("topics_covered", [])]
            est_hours = int(args["estimated_hours"]) if args.get("estimated_hours") is not None else None
            notes = str(args["notes"]) if args.get("notes") is not None else None

            existing_ids = {e.id for e in cal.events}
            assigned_id = _generate_event_id(ev_name, ev_date, existing_ids)

            new_event = Event(
                id=assigned_id,
                name=str(ev_name).strip(),
                type=ev_type,
                course_id=course_id,
                date=ev_date,
                start_time=start_t,
                duration_minutes=duration,
                weight=weight,
                topics_covered=topics,
                estimated_hours=est_hours,
                notes=notes,
            )

            cal.events.append(new_event)
            save_calendar(cal, target_path, backup=True)
            logger.info(f"[calendar_tool] Evento añadido: {assigned_id} ('{ev_name}')")

            details = [f"Evento '{new_event.name}' añadido exitosamente (ID: {assigned_id})."]
            details.append(f"Fecha: {new_event.date.isoformat()}")
            if new_event.start_time:
                details.append(f"Hora: {new_event.start_time.strftime('%H:%M')}")
            if new_event.course_id:
                details.append(f"Curso: {new_event.course_id}")
            return " ".join(details)

        elif name == "calendar_list_courses":
            if not cal.courses:
                return "No hay cursos registrados en el calendario para este semestre."

            lines = [f"Cursos del semestre ({cal.metadata.semester}):"]
            for c in cal.courses:
                topics_str = f" | Temas: {', '.join(c.topics)}" if c.topics else ""
                lines.append(
                    f"- [{c.id}] {c.name} (Dificultad: {c.difficulty}/5, {c.sessions_per_week} sesiones/sem){topics_str}"
                )
            return "\n".join(lines)

        elif name == "calendar_delete_event":
            event_id = args.get("event_id")
            if not event_id:
                return "Error: Se requiere 'event_id' para eliminar un evento."

            target_id = str(event_id).strip()
            found_idx: int | None = None
            found_event: Event | None = None

            for idx, ev in enumerate(cal.events):
                if ev.id == target_id:
                    found_idx = idx
                    found_event = ev
                    break

            if found_idx is None or found_event is None:
                return f"Error: No se encontró ningún evento con ID '{target_id}' en el calendario."

            cal.events.pop(found_idx)
            save_calendar(cal, target_path, backup=True)
            logger.info(f"[calendar_tool] Evento eliminado: {target_id} ('{found_event.name}')")

            return f"Evento '{found_event.name}' (ID: {target_id}) eliminado exitosamente del calendario."

        return f"Error: Herramienta de calendario desconocida '{name}'."

    except Exception as exc:
        logger.error(f"[calendar_tool] Error ejecutando '{name}': {exc}")
        return f"Error al ejecutar {name}: {exc}"
