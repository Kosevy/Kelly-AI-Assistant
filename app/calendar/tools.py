"""
app/calendar/tools.py — Definición y despacho de herramientas locales de calendario para el LLM.

Rationale:
- Expone esquemas JSON Schema compatibles con function calling nativo en Qwen2.5.
- Conecta consultas de disponibilidad y mutaciones del calendario directamente con la
  fuente de verdad YAML, ejecutando backups automáticos en cada modificación.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from unidecode import unidecode

from app.calendar.exceptions import CalendarValidationError
from app.calendar.expander import expand_all
from app.calendar.models import Calendar, Event, EventType
from app.calendar.parser import init_calendar_if_missing, load_calendar, save_calendar
from app.calendar.query import events_in_range, free_slots, next_deadline
from app.config import settings
from app.logging_config import logger

CALENDAR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calendar_next_deadline",
            "description": (
                "Encuentra el PRÓXIMO evento evaluativo desde hoy hacia adelante "
                "SIN LÍMITE de tiempo. Busca entre todos los eventos del calendario "
                "hacia el futuro y devuelve el más cercano. "
                "USAR CUANDO: el usuario pregunta '¿cuál es mi próximo examen?', "
                "'¿qué entrega viene?', '¿cuándo es lo próximo?'. "
                "NO usar para '¿qué tengo esta semana?' (usar calendar_read_range)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Opcional. Filtra por tipo: exam, entregable, presentacion, practica. "
                            "Si no se especifica, busca todos."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_read_range",
            "description": (
                "Muestra los eventos y bloques en un rango de fechas. "
                "USAR CUANDO: el usuario quiere ver el horario completo o preguntar "
                "qué eventos hay en un rango. NO usar para calcular tiempo libre. "
                "Ej: '¿qué tengo esta semana?', '¿qué hay el 12 de mayo?'."
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
                "Calcula bloques de tiempo LIBRE disponibles. "
                "USAR CUANDO: el usuario pregunta si tiene tiempo disponible para algo, "
                "o quiere saber huecos sin compromisos. "
                "NO usar para ver el horario completo. "
                "Ej: '¿tengo tiempo para estudiar?', '¿cuándo puedo hacer 90 min?'."
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
    return_structured: bool = False,
) -> str | dict[str, Any]:
    """
    Despacha y ejecuta herramientas locales de calendario.
    Si return_structured=True, retorna un dict JSON estructurado (tipo calendar_card)
    para renderizado directo de interfaz de usuario.
    Si return_structured=False, retorna un string formateado para el contexto del LLM.
    """
    target_path = calendar_path or settings.get_calendar_path()

    if not target_path.exists():
        init_calendar_if_missing(calendar_path=target_path)

    if not target_path.exists():
        logger.warning(f"[calendar_tool] Archivo de calendario no encontrado en: {target_path}")
        err_msg = (
            f"Error: El archivo de calendario no existe en '{target_path}'. "
            "Copia 'data/calendar.yaml.example' a 'data/calendar.yaml' para inicializarlo."
        )
        if return_structured:
            return {"type": "calendar_card", "error": err_msg, "days": [], "data": []}
        return err_msg

    try:
        cal = load_calendar(target_path)
    except CalendarValidationError as exc:
        logger.error(f"[calendar_tool] Error cargando calendario '{target_path}': {exc}")
        err_msg = f"Error en el calendario: {exc}"
        if return_structured:
            return {"type": "calendar_card", "error": err_msg, "days": [], "data": []}
        return err_msg

    course_lookup = {c.id.lower(): c.name for c in cal.courses}
    DAY_NAMES = [
        "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"
    ]
    MONTH_NAMES = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]

    try:
        if name == "calendar_next_deadline":
            types_arg = args.get("types")
            if types_arg is not None and not isinstance(types_arg, list):
                types_arg = [str(types_arg)]

            from_d = date.today()
            if args.get("from_date"):
                try:
                    from_d = date.fromisoformat(str(args["from_date"]).strip())
                except ValueError:
                    pass

            end_candidates = [cal.metadata.semester_end]
            if cal.events:
                end_candidates.extend(ev.date for ev in cal.events)
            max_end = max(end_candidates)

            if from_d > max_end:
                types_desc = f" del tipo {types_arg}" if types_arg else ""
                no_ev_msg = f"No hay eventos próximos{types_desc} en el calendario."
                if return_structured:
                    return {
                        "type": "calendar_card",
                        "view_type": "deadline",
                        "title": "Próximo evento evaluativo",
                        "deadline": None,
                        "message": no_ev_msg,
                        "days": [],
                        "data": [],
                    }
                return no_ev_msg

            expanded = expand_all(cal, from_d, max_end)
            deadline = next_deadline(expanded, from_date=from_d, types=types_arg)

            if not deadline:
                types_desc = f" del tipo {types_arg}" if types_arg else ""
                no_ev_msg = f"No hay eventos próximos{types_desc} en el calendario."
                if return_structured:
                    return {
                        "type": "calendar_card",
                        "view_type": "deadline",
                        "title": "Próximo evento evaluativo",
                        "deadline": None,
                        "message": no_ev_msg,
                        "days": [],
                        "data": [],
                    }
                return no_ev_msg

            days_left = (deadline.date - from_d).days
            if days_left == 0:
                days_msg = "faltan 0 días (hoy)"
            elif days_left == 1:
                days_msg = "falta 1 día"
            else:
                days_msg = f"faltan {days_left} días"

            time_str = f" {deadline.start.strftime('%H:%M')}" if deadline.start != deadline.end else ""
            course_str = f" (Curso: {deadline.course_id})" if deadline.course_id else ""

            if return_structured:
                c_name = course_lookup.get(deadline.course_id.lower()) if deadline.course_id else None
                d_name = DAY_NAMES[deadline.date.weekday()]
                m_name = MONTH_NAMES[deadline.date.month - 1]
                deadline_event = {
                    "start": deadline.start.strftime("%H:%M"),
                    "end": deadline.end.strftime("%H:%M") if deadline.start != deadline.end else deadline.start.strftime("%H:%M"),
                    "name": deadline.name,
                    "type": deadline.type,
                    "course": c_name or deadline.course_id,
                }
                days_list = [
                    {
                        "date": deadline.date.isoformat(),
                        "day_name": f"{d_name} {deadline.date.day} de {m_name}",
                        "is_today": (deadline.date == date.today()),
                        "events": [deadline_event],
                    }
                ]
                return {
                    "type": "calendar_card",
                    "view_type": "deadline",
                    "title": "Próximo evento evaluativo",
                    "deadline": {
                        "name": deadline.name,
                        "date": deadline.date.isoformat(),
                        "day_name": f"{d_name} {deadline.date.day} de {m_name}",
                        "start": deadline_event["start"],
                        "end": deadline_event["end"],
                        "type": deadline.type,
                        "course": c_name or deadline.course_id,
                        "days_left": days_left,
                        "days_left_text": days_msg,
                    },
                    "days": days_list,
                    "data": days_list,
                }

            return f"Próximo: {deadline.name} — {deadline.date.isoformat()}{time_str}{course_str} — {days_msg}."

        elif name == "calendar_read_range":
            start_str = args.get("start_date")
            end_str = args.get("end_date")
            if not start_str or not end_str:
                return "Error: Se requieren 'start_date' y 'end_date' (formato YYYY-MM-DD)."

            try:
                start_d = date.fromisoformat(str(start_str).strip())
                end_d = date.fromisoformat(str(end_str).strip())
            except ValueError as ve:
                return f"Error de formato de fecha: {ve}. Usa YYYY-MM-DD."

            if start_d > end_d:
                return f"Error: start_date ({start_d}) no puede ser posterior a end_date ({end_d})."

            types_filter = args.get("types")
            total_days = (end_d - start_d).days + 1
            max_days = settings.calendar_read_max_days

            was_truncated = False
            effective_end_d = end_d
            if total_days > max_days:
                was_truncated = True
                effective_end_d = start_d + timedelta(days=max_days - 1)

            expanded = expand_all(cal, start_d, effective_end_d)
            filtered = events_in_range(expanded, start_d, effective_end_d, types=types_filter)

            DAY_NAMES = [
                "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"
            ]
            MONTH_NAMES = [
                "enero", "febrero", "marzo", "abril", "mayo", "junio",
                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
            ]

            course_lookup = {c.id.lower(): c.name for c in cal.courses}

            days_map: dict[date, list[ExpandedBlock]] = {}
            for b in filtered:
                days_map.setdefault(b.date, []).append(b)

            day_sections: list[str] = []
            cur_date = start_d
            while cur_date <= effective_end_d:
                d_name = DAY_NAMES[cur_date.weekday()]
                m_name = MONTH_NAMES[cur_date.month - 1]
                header = f"**{d_name} {cur_date.day} de {m_name}**"

                day_blocks = days_map.get(cur_date, [])
                if not day_blocks:
                    day_sections.append(f"{header}\n- (sin eventos programados)")
                else:
                    bullets: list[str] = []
                    for b in day_blocks:
                        if b.start != b.end:
                            t_str = f"{b.start.strftime('%H:%M')}-{b.end.strftime('%H:%M')}"
                        else:
                            t_str = b.start.strftime("%H:%M")

                        course_suffix = ""
                        if b.course_id:
                            c_name = course_lookup.get(b.course_id.lower())
                            if c_name and b.name.strip().lower() != c_name.strip().lower():
                                course_suffix = f" ({c_name})"
                            elif not c_name and b.course_id.lower() != b.name.strip().lower():
                                course_suffix = f" ({b.course_id})"

                        bullets.append(f"- {t_str} {b.name}{course_suffix}")
                    day_sections.append(f"{header}\n" + "\n".join(bullets))

                cur_date += timedelta(days=1)

            if return_structured:
                days_payload: list[dict[str, Any]] = []
                cur_date = start_d
                while cur_date <= effective_end_d:
                    d_name = DAY_NAMES[cur_date.weekday()]
                    m_name = MONTH_NAMES[cur_date.month - 1]
                    day_blocks = days_map.get(cur_date, [])
                    events_list: list[dict[str, Any]] = []
                    for b in day_blocks:
                        c_name = course_lookup.get(b.course_id.lower()) if b.course_id else None
                        events_list.append({
                            "start": b.start.strftime("%H:%M"),
                            "end": b.end.strftime("%H:%M") if b.start != b.end else b.start.strftime("%H:%M"),
                            "name": b.name,
                            "type": b.type,
                            "course": c_name or b.course_id,
                        })
                    days_payload.append({
                        "date": cur_date.isoformat(),
                        "day_name": f"{d_name} {cur_date.day} de {m_name}",
                        "is_today": (cur_date == date.today()),
                        "events": events_list,
                    })
                    cur_date += timedelta(days=1)

                view_type = "week" if total_days >= 5 else ("day" if total_days == 1 else "range")
                if view_type == "week":
                    card_title = f"Semana del {start_d.day} al {effective_end_d.day} de {MONTH_NAMES[effective_end_d.month - 1]}"
                elif view_type == "day":
                    d_name = DAY_NAMES[start_d.weekday()]
                    m_name = MONTH_NAMES[start_d.month - 1]
                    card_title = f"Agenda de {d_name} {start_d.day} de {m_name}"
                else:
                    card_title = f"Agenda del {start_d.day} de {MONTH_NAMES[start_d.month - 1]} al {effective_end_d.day} de {MONTH_NAMES[effective_end_d.month - 1]}"

                return {
                    "type": "calendar_card",
                    "view_type": view_type,
                    "title": card_title,
                    "days": days_payload,
                    "data": days_payload,
                }

            res = "\n\n".join(day_sections)
            if was_truncated:
                res += (
                    f"\n\n[Rango completo: {total_days} días. Mostrando primeros {max_days}. "
                    "Pedir rangos más específicos para ver más.]"
                )

            return res

        elif name == "calendar_free_slots":
            start_str = args.get("start_date")
            end_str = args.get("end_date")
            if not start_str or not end_str:
                err_msg = "Error: Se requieren 'start_date' y 'end_date' (formato YYYY-MM-DD)."
                if return_structured:
                    return {"type": "calendar_card", "error": err_msg, "days": [], "data": []}
                return err_msg

            try:
                start_d = date.fromisoformat(str(start_str).strip())
                end_d = date.fromisoformat(str(end_str).strip())
            except ValueError as ve:
                err_msg = f"Error de formato de fecha: {ve}. Usa YYYY-MM-DD."
                if return_structured:
                    return {"type": "calendar_card", "error": err_msg, "days": [], "data": []}
                return err_msg

            min_minutes = int(args.get("min_block_minutes", 30))
            expanded = expand_all(cal, start_d, end_d)
            slots = free_slots(expanded, start_d, end_d, min_block_minutes=min_minutes)

            if return_structured:
                slots_payload: list[dict[str, Any]] = []
                days_slots_map: dict[date, list[dict[str, Any]]] = {}
                for s, e in slots:
                    duration_m = int((e - s).total_seconds() / 60.0)
                    slot_data = {
                        "start": s.strftime("%H:%M"),
                        "end": e.strftime("%H:%M"),
                        "name": f"Tiempo libre ({duration_m} min)",
                        "type": "free",
                        "course": None,
                        "duration_minutes": duration_m,
                    }
                    slots_payload.append(slot_data)
                    days_slots_map.setdefault(s.date(), []).append(slot_data)

                days_payload_free: list[dict[str, Any]] = []
                cur_date = start_d
                while cur_date <= end_d:
                    d_name = DAY_NAMES[cur_date.weekday()]
                    m_name = MONTH_NAMES[cur_date.month - 1]
                    day_slots = days_slots_map.get(cur_date, [])
                    days_payload_free.append({
                        "date": cur_date.isoformat(),
                        "day_name": f"{d_name} {cur_date.day} de {m_name}",
                        "is_today": (cur_date == date.today()),
                        "events": day_slots,
                    })
                    cur_date += timedelta(days=1)

                return {
                    "type": "calendar_card",
                    "view_type": "free_slots",
                    "title": f"Tiempo libre disponible ({len(slots)} bloques)",
                    "slots": slots_payload,
                    "days": days_payload_free,
                    "data": days_payload_free,
                }

            if not slots:
                return f"No se encontraron bloques libres de al menos {min_minutes} minutos entre {start_d} y {end_d}."

            header = (
                f"{len(slots)} bloques disponibles:"
                if start_d == end_d
                else f"{len(slots)} bloques disponibles del {start_d} al {end_d}:"
            )
            lines = [header]
            for s, e in slots:
                duration_m = int((e - s).total_seconds() / 60.0)
                cur_dt = s.date()
                day_blocks = [b for b in expanded if b.date == cur_dt and b.end > b.start]

                prev_candidates = [b for b in day_blocks if b.end <= s]
                prev_b = max(prev_candidates, key=lambda b: b.end) if prev_candidates else None

                next_candidates = [b for b in day_blocks if b.start >= e]
                next_b = min(next_candidates, key=lambda b: b.start) if next_candidates else None

                if prev_b and next_b:
                    context = f"entre {prev_b.name} y {next_b.name}"
                elif next_b:
                    context = f"antes de {next_b.name}"
                elif prev_b:
                    context = f"después de {prev_b.name}"
                else:
                    context = "todo el día libre"

                time_label = (
                    f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}"
                    if start_d == end_d
                    else f"{s.strftime('%Y-%m-%d %H:%M')}-{e.strftime('%H:%M')}"
                )
                lines.append(f" - {time_label} ({duration_m} min) — {context}")

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
