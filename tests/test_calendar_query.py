"""
tests/test_calendar_query.py
Pruebas unitarias para las consultas de calendario y cálculo de disponibilidad (Bloque B2).
"""
from __future__ import annotations

from datetime import date, datetime, time
import pytest

from app.calendar import (
    BlockType,
    Calendar,
    Course,
    DayOfWeek,
    Event,
    EventType,
    ExpandedBlock,
    Metadata,
    RecurringBlock,
    events_in_range,
    expand_all,
    free_slots,
    next_deadline,
)


def test_free_slots_empty_calendar():
    """Un calendario sin bloques debe tener libre toda la ventana diaria (6:00 a 23:00)."""
    target_date = date(2026, 4, 10)
    slots = free_slots(
        expanded=[],
        start_date=target_date,
        end_date=target_date,
        min_block_minutes=30,
        day_start=time(6, 0),
        day_end=time(23, 0),
    )

    assert len(slots) == 1
    start_dt, end_dt = slots[0]
    assert start_dt == datetime(2026, 4, 10, 6, 0)
    assert end_dt == datetime(2026, 4, 10, 23, 0)


def test_free_slots_ignores_meal_and_transport():
    """Bloques tipo meal y transport no deben bloquear la disponibilidad."""
    target_date = date(2026, 4, 10)
    blocks = [
        ExpandedBlock(
            name="Desayuno",
            type="meal",
            course_id=None,
            date=target_date,
            start=datetime(2026, 4, 10, 7, 0),
            end=datetime(2026, 4, 10, 8, 0),
            source="recurring",
        ),
        ExpandedBlock(
            name="Transporte",
            type="transport",
            course_id=None,
            date=target_date,
            start=datetime(2026, 4, 10, 8, 0),
            end=datetime(2026, 4, 10, 9, 0),
            source="recurring",
        ),
    ]

    slots = free_slots(
        expanded=blocks,
        start_date=target_date,
        end_date=target_date,
        min_block_minutes=30,
        day_start=time(6, 0),
        day_end=time(23, 0),
    )

    # Debe seguir libre todo el día de 6:00 a 23:00 porque meal y transport no son busy
    assert len(slots) == 1
    assert slots[0] == (datetime(2026, 4, 10, 6, 0), datetime(2026, 4, 10, 23, 0))


def test_free_slots_with_busy_blocks():
    """Con una clase de 8:00 a 10:00 y un examen de 14:00 a 16:00, calcula 3 huecos libres."""
    target_date = date(2026, 4, 10)
    blocks = [
        ExpandedBlock(
            name="Clase Cálculo",
            type="class",
            course_id="calc3",
            date=target_date,
            start=datetime(2026, 4, 10, 8, 0),
            end=datetime(2026, 4, 10, 10, 0),
            source="recurring",
        ),
        ExpandedBlock(
            name="Parcial BD",
            type="exam",
            course_id="bd",
            date=target_date,
            start=datetime(2026, 4, 10, 14, 0),
            end=datetime(2026, 4, 10, 16, 0),
            source="event",
            event_id="parcial_bd",
        ),
    ]

    slots = free_slots(
        expanded=blocks,
        start_date=target_date,
        end_date=target_date,
        min_block_minutes=30,
        day_start=time(6, 0),
        day_end=time(23, 0),
    )

    # Hueco 1: 06:00 a 08:00 (120 min)
    # Hueco 2: 10:00 a 14:00 (240 min)
    # Hueco 3: 16:00 a 23:00 (420 min)
    assert len(slots) == 3
    assert slots[0] == (datetime(2026, 4, 10, 6, 0), datetime(2026, 4, 10, 8, 0))
    assert slots[1] == (datetime(2026, 4, 10, 10, 0), datetime(2026, 4, 10, 14, 0))
    assert slots[2] == (datetime(2026, 4, 10, 16, 0), datetime(2026, 4, 10, 23, 0))


def test_free_slots_min_block_minutes_filters_short_gaps():
    """min_block_minutes=60 debe descartar huecos menores a 60 minutos."""
    target_date = date(2026, 4, 10)
    blocks = [
        # Ocupado 6:00 a 8:00
        ExpandedBlock(
            name="Clase 1",
            type="class",
            course_id="calc3",
            date=target_date,
            start=datetime(2026, 4, 10, 6, 0),
            end=datetime(2026, 4, 10, 8, 0),
            source="recurring",
        ),
        # Ocupado 8:30 a 23:00 (deja un hueco de solo 30 min: 8:00 a 8:30)
        ExpandedBlock(
            name="Clase 2",
            type="class",
            course_id="bd",
            date=target_date,
            start=datetime(2026, 4, 10, 8, 30),
            end=datetime(2026, 4, 10, 23, 0),
            source="recurring",
        ),
    ]

    slots_30 = free_slots(blocks, target_date, target_date, min_block_minutes=30)
    assert len(slots_30) == 1
    assert slots_30[0] == (datetime(2026, 4, 10, 8, 0), datetime(2026, 4, 10, 8, 30))

    slots_60 = free_slots(blocks, target_date, target_date, min_block_minutes=60)
    assert len(slots_60) == 0


def test_next_deadline_returns_correct_event():
    """next_deadline debe retornar el primer evento evaluativo desde la fecha consultada."""
    expanded = [
        ExpandedBlock(
            name="Clase recurrente",
            type="class",
            course_id="calc3",
            date=date(2026, 5, 1),
            start=datetime(2026, 5, 1, 8, 0),
            end=datetime(2026, 5, 1, 10, 0),
            source="recurring",
        ),
        ExpandedBlock(
            name="Entrega BD - Avance 1",
            type="entregable",
            course_id="bd",
            date=date(2026, 5, 5),
            start=datetime(2026, 5, 5, 0, 0),
            end=datetime(2026, 5, 5, 0, 0),
            source="event",
            event_id="entrega_1",
        ),
        ExpandedBlock(
            name="Parcial Cálculo III",
            type="exam",
            course_id="calc3",
            date=date(2026, 5, 12),
            start=datetime(2026, 5, 12, 10, 0),
            end=datetime(2026, 5, 12, 12, 0),
            source="event",
            event_id="parcial_1",
        ),
    ]

    # Consultando desde el 1 de mayo -> el siguiente es la Entrega BD (5 de mayo)
    deadline_1 = next_deadline(expanded, from_date=date(2026, 5, 1))
    assert deadline_1 is not None
    assert deadline_1.name == "Entrega BD - Avance 1"
    assert deadline_1.event_id == "entrega_1"

    # Consultando desde el 6 de mayo -> el siguiente es el Parcial (12 de mayo)
    deadline_2 = next_deadline(expanded, from_date=date(2026, 5, 6))
    assert deadline_2 is not None
    assert deadline_2.name == "Parcial Cálculo III"
    assert deadline_2.event_id == "parcial_1"

    # Consultando solo exámenes
    deadline_exams = next_deadline(expanded, from_date=date(2026, 5, 1), types=["exam"])
    assert deadline_exams is not None
    assert deadline_exams.name == "Parcial Cálculo III"


def test_events_in_range_empty_when_no_match():
    """Rango sin eventos debe retornar lista vacía."""
    expanded = [
        ExpandedBlock(
            name="Parcial Cálculo III",
            type="exam",
            course_id="calc3",
            date=date(2026, 5, 12),
            start=datetime(2026, 5, 12, 10, 0),
            end=datetime(2026, 5, 12, 12, 0),
            source="event",
            event_id="parcial_1",
        ),
    ]

    # Rango en abril donde no hay eventos
    res = events_in_range(expanded, start_date=date(2026, 4, 1), end_date=date(2026, 4, 30))
    assert res == []

    # Filtrar por tipo inexistente en mayo
    res_type = events_in_range(
        expanded,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 31),
        types=["personal"],
    )
    assert res_type == []
