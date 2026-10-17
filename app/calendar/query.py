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
    break_types: Sequence[str] | None = None,
    mode: str | None = None,
) -> list[tuple[datetime, datetime]]:
    """
    Calcula los intervalos de tiempo libre dentro del rango [start_date, end_date] (inclusivo).

    Rationale:
    - En modo 'literal' (settings.free_slots_mode == 'literal'), solo los tipos en busy_types
      (class, study, exam, entregable, presentacion, practica) bloquean el tiempo.
    - En modo 'realistic' (settings.free_slots_mode == 'realistic', por defecto), los tipos en
      break_types (meal, transport, exercise) parten los huecos libres en ventanas útiles y realistas,
      evitando devolver franjas irreales de 17 horas continuas.
    """
    if start_date > end_date:
        return []

    effective_mode = (mode or settings.free_slots_mode).lower().strip()

    if busy_types is None:
        raw_busy = settings.free_slots_busy_types
        active_busy = [t.lower().strip() for t in raw_busy]
    else:
        active_busy = [t.lower().strip() for t in busy_types]

    if effective_mode == "literal":
        active_breaks: list[str] = []
    else:
        if break_types is None:
            raw_breaks = settings.free_slots_realistic_break_types
            active_breaks = [t.lower().strip() for t in raw_breaks if t.lower().strip() not in active_busy]
        else:
            active_breaks = [t.lower().strip() for t in break_types if t.lower().strip() not in active_busy]

    result: list[tuple[datetime, datetime]] = []
    delta_days = (end_date - start_date).days

    for i in range(delta_days + 1):
        cur_date = start_date + timedelta(days=i)
        win_start = datetime.combine(cur_date, day_start)
        win_end = datetime.combine(cur_date, day_end)

        if win_start >= win_end:
            continue

        # 1. Filtrar bloques hard busy del día
        busy_intervals: list[tuple[datetime, datetime]] = []
        for b in expanded:
            if b.date != cur_date:
                continue
            if b.type.lower() not in active_busy:
                continue
            if b.end <= b.start:
                continue

            seg_start = max(b.start, win_start)
            seg_end = min(b.end, win_end)
            if seg_end > seg_start:
                busy_intervals.append((seg_start, seg_end))

        # Ordenar y fusionar hard busy
        busy_intervals.sort(key=lambda x: (x[0], x[1]))
        merged_busy: list[tuple[datetime, datetime]] = []
        for s, e in busy_intervals:
            if merged_busy and s <= merged_busy[-1][1]:
                merged_busy[-1] = (merged_busy[-1][0], max(merged_busy[-1][1], e))
            else:
                merged_busy.append((s, e))

        # 2. Encontrar huecos libres iniciales fuera de hard busy
        initial_slots: list[tuple[datetime, datetime]] = []
        curr_ptr = win_start
        for b_s, b_e in merged_busy:
            if b_s > curr_ptr:
                initial_slots.append((curr_ptr, b_s))
            curr_ptr = max(curr_ptr, b_e)

        if win_end > curr_ptr:
            initial_slots.append((curr_ptr, win_end))

        # 3. Si no hay breaks activos (ej. modo literal), filtrar por duración mínima
        if not active_breaks:
            for s, e in initial_slots:
                duration_m = (e - s).total_seconds() / 60.0
                if duration_m >= min_block_minutes:
                    result.append((s, e))
            continue

        # 4. Modo realista: recopilar breaks del día y partir los huecos libres
        day_breaks: list[tuple[datetime, datetime]] = []
        for b in expanded:
            if b.date != cur_date:
                continue
            if b.type.lower() not in active_breaks:
                continue
            if b.end <= b.start:
                continue

            seg_start = max(b.start, win_start)
            seg_end = min(b.end, win_end)
            if seg_end > seg_start:
                day_breaks.append((seg_start, seg_end))

        for slot_start, slot_end in initial_slots:
            intersecting_breaks: list[tuple[datetime, datetime]] = []
            for brk_s, brk_e in day_breaks:
                clamped_s = max(brk_s, slot_start)
                clamped_e = min(brk_e, slot_end)
                if clamped_e > clamped_s:
                    intersecting_breaks.append((clamped_s, clamped_e))

            if not intersecting_breaks:
                duration_m = (slot_end - slot_start).total_seconds() / 60.0
                if duration_m >= min_block_minutes:
                    result.append((slot_start, slot_end))
                continue

            intersecting_breaks.sort(key=lambda x: (x[0], x[1]))
            merged_breaks: list[tuple[datetime, datetime]] = []
            for s, e in intersecting_breaks:
                if merged_breaks and s <= merged_breaks[-1][1]:
                    merged_breaks[-1] = (merged_breaks[-1][0], max(merged_breaks[-1][1], e))
                else:
                    merged_breaks.append((s, e))

            sub_ptr = slot_start
            for brk_s, brk_e in merged_breaks:
                if brk_s > sub_ptr:
                    duration_m = (brk_s - sub_ptr).total_seconds() / 60.0
                    if duration_m >= min_block_minutes:
                        result.append((sub_ptr, brk_s))
                sub_ptr = max(sub_ptr, brk_e)

            if slot_end > sub_ptr:
                duration_m = (slot_end - sub_ptr).total_seconds() / 60.0
                if duration_m >= min_block_minutes:
                    result.append((sub_ptr, slot_end))

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
    Si types es None o una lista vacía, busca entre todos los eventos del calendario hacia el futuro.
    """
    target_types = [t.lower().strip() for t in types] if types else None

    candidates = [
        b for b in expanded
        if b.date >= from_date and b.source == "event" and (target_types is None or b.type.lower() in target_types)
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda b: (b.date, b.start, b.end, b.name))
    return candidates[0]
