"""
app/calendar/query.py — Consultas y cálculo de disponibilidad sobre bloques expandidos.

Rationale:
- Implementa algoritmos puros para cálculo de huecos libres (free slots) e inspección de eventos.
- Los tipos no bloqueantes (meal, transport, exercise) se ignoran en el cálculo de disponibilidad
  permitiendo identificar momentos reales para concentración y estudio.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Sequence

from app.calendar.expander import ExpandedBlock
from app.config import settings


def free_slots(
    expanded: list[ExpandedBlock],
    start_date: date,
    end_date: date,
    min_block_minutes: int = 30,
    day_start: time = time(6, 0),
    day_end: time = time(23, 0),
    busy_types: Sequence[str] | None = None,
) -> list[tuple[datetime, datetime]]:
    """
    Calcula los intervalos de tiempo libre dentro del rango [start_date, end_date] (inclusivo).

    Solo considera ocupados los bloques cuyo tipo figure en busy_types
    (por defecto: class, study, exam, entregable, presentacion, practica).
    Bloques como transport, meal o exercise no bloquean disponibilidad para estudio.
    """
    if start_date > end_date:
        return []

    if busy_types is None:
        raw_types = settings.free_slots_busy_types
        active_busy = [t.lower().strip() for t in raw_types]
    else:
        active_busy = [t.lower().strip() for t in busy_types]

    result: list[tuple[datetime, datetime]] = []
    delta_days = (end_date - start_date).days

    for i in range(delta_days + 1):
        cur_date = start_date + timedelta(days=i)
        win_start = datetime.combine(cur_date, day_start)
        win_end = datetime.combine(cur_date, day_end)

        if win_start >= win_end:
            continue

        # Filtrar bloques ocupados del día
        busy_intervals: list[tuple[datetime, datetime]] = []
        for b in expanded:
            if b.date != cur_date:
                continue

            # Solo los tipos busy ocupan el horario
            if b.type.lower() not in active_busy:
                continue

            # Bloques sin duración (ej. recordatorios de entrega sin hora) no ocupan franjas
            if b.end <= b.start:
                continue

            # Acotar el bloque dentro de la ventana del día
            seg_start = max(b.start, win_start)
            seg_end = min(b.end, win_end)

            if seg_end > seg_start:
                busy_intervals.append((seg_start, seg_end))

        # Ordenar y fusionar intervalos contiguos o solapados
        busy_intervals.sort(key=lambda x: (x[0], x[1]))
        merged: list[tuple[datetime, datetime]] = []
        for s, e in busy_intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Encontrar huecos libres entre los bloques ocupados
        curr_ptr = win_start
        for b_s, b_e in merged:
            if b_s > curr_ptr:
                duration_m = (b_s - curr_ptr).total_seconds() / 60.0
                if duration_m >= min_block_minutes:
                    result.append((curr_ptr, b_s))
            curr_ptr = max(curr_ptr, b_e)

        if win_end > curr_ptr:
            duration_m = (win_end - curr_ptr).total_seconds() / 60.0
            if duration_m >= min_block_minutes:
                result.append((curr_ptr, win_end))

    return result


def events_in_range(
    expanded: list[ExpandedBlock],
    start_date: date,
    end_date: date,
    types: list[str] | None = None,
) -> list[ExpandedBlock]:
    """
    Filtra eventos y bloques dentro de un rango de fechas, opcionalmente filtrando por tipos.
    """
    if start_date > end_date:
        return []

    target_types = [t.lower().strip() for t in types] if types is not None else None

    result: list[ExpandedBlock] = []
    for b in expanded:
        if not (start_date <= b.date <= end_date):
            continue

        if target_types is not None and b.type.lower() not in target_types:
            continue

        result.append(b)

    result.sort(key=lambda b: (b.date, b.start, b.end, b.name))
    return result


def next_deadline(
    expanded: list[ExpandedBlock],
    from_date: date,
    types: list[str] | None = None,
) -> ExpandedBlock | None:
    """
    Retorna el siguiente evento relevante (examen, entrega, práctica) a partir de from_date (inclusivo).
    """
    default_types = ["exam", "entregable", "presentacion", "practica"]
    target_types = [t.lower().strip() for t in (types or default_types)]

    candidates = [
        b for b in expanded
        if b.date >= from_date and b.source == "event" and b.type.lower() in target_types
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda b: (b.date, b.start, b.end, b.name))
    return candidates[0]
