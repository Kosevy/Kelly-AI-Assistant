"""
app/calendar/expander.py — Expansión cronológica de bloques recurrentes y eventos puntuales.

Rationale:
- Convierte reglas de recurrencia semanales y eventos puntuales en instancias concretas
  (ExpandedBlock) situadas en una fecha y rango horario específico.
- Al separar la expansión de recurrentes y eventos en funciones puras, cada algoritmo
  es testeable de forma aislada sin efectos colaterales.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from app.calendar.models import (
    Calendar,
    Event,
    RecurringBlock,
)


@dataclass(frozen=True)
class ExpandedBlock:
    """Un bloque recurrente o evento expandido a una fecha concreta."""
    name: str
    type: str
    course_id: str | None
    date: date
    start: datetime
    end: datetime
    source: str  # 'recurring' | 'event'
    event_id: str | None = None  # solo si source == 'event'


def expand_recurring_blocks(
    blocks: list[RecurringBlock],
    start_date: date,
    end_date: date,
) -> list[ExpandedBlock]:
    """
    Expande bloques recurrentes semanales a fechas concretas dentro del rango [start_date, end_date] (inclusivo).
    """
    if start_date > end_date:
        return []

    result: list[ExpandedBlock] = []
    delta_days = (end_date - start_date).days

    for i in range(delta_days + 1):
        cur_date = start_date + timedelta(days=i)
        day_name = cur_date.strftime("%A").lower()

        for b in blocks:
            # Comprobar vigencia temporal del bloque
            if not (b.from_date <= cur_date <= b.to_date):
                continue

            # Comprobar si aplica en el día de la semana actual
            block_days = [d.value if hasattr(d, "value") else str(d).lower() for d in b.day_of_week]
            if day_name in block_days:
                start_dt = datetime.combine(cur_date, b.start_time)
                end_dt = datetime.combine(cur_date, b.end_time)
                b_type = b.type.value if hasattr(b.type, "value") else str(b.type)

                result.append(
                    ExpandedBlock(
                        name=b.name,
                        type=b_type,
                        course_id=b.course_id,
                        date=cur_date,
                        start=start_dt,
                        end=end_dt,
                        source="recurring",
                        event_id=None,
                    )
                )

    return result


def expand_events(
    events: list[Event],
    start_date: date,
    end_date: date,
) -> list[ExpandedBlock]:
    """
    Convierte eventos puntuales dentro del rango [start_date, end_date] en ExpandedBlocks.
    Si el evento no define start_time o duration, se expande con duración 0 como marcador/deadline.
    """
    if start_date > end_date:
        return []

    result: list[ExpandedBlock] = []

    for ev in events:
        if not (start_date <= ev.date <= end_date):
            continue

        if ev.start_time is not None:
            start_dt = datetime.combine(ev.date, ev.start_time)
            duration = ev.duration_minutes if ev.duration_minutes is not None else 60
            end_dt = start_dt + timedelta(minutes=duration)
        else:
            # Marcador de fecha sin hora fija (ej. entregables o deadlines sin hora fijada)
            start_dt = datetime.combine(ev.date, time(0, 0))
            end_dt = start_dt

        ev_type = ev.type.value if hasattr(ev.type, "value") else str(ev.type)

        result.append(
            ExpandedBlock(
                name=ev.name,
                type=ev_type,
                course_id=ev.course_id,
                date=ev.date,
                start=start_dt,
                end=end_dt,
                source="event",
                event_id=ev.id,
            )
        )

    return result


def expand_all(
    calendar: Calendar,
    start_date: date,
    end_date: date,
) -> list[ExpandedBlock]:
    """
    Expande tanto los bloques recurrentes como los eventos puntuales en el rango indicado.
    Retorna la lista unificada ordenada cronológicamente por (date, start, end, name).
    """
    recurring = expand_recurring_blocks(calendar.recurring_blocks, start_date, end_date)
    events = expand_events(calendar.events, start_date, end_date)

    combined = recurring + events
    combined.sort(key=lambda b: (b.date, b.start, b.end, b.name))
    return combined
