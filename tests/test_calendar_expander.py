"""
tests/test_calendar_expander.py
Pruebas unitarias para la expansión de recurrencias y eventos (Bloque B1).
"""
from __future__ import annotations

from datetime import date, time
import pytest

from app.calendar import (
    BlockType,
    Calendar,
    Course,
    DayOfWeek,
    Event,
    EventType,
    Metadata,
    RecurringBlock,
    expand_all,
    expand_events,
    expand_recurring_blocks,
)


def test_expand_weekly_block_over_4_weeks():
    """Un bloque semanal en un rango de 4 semanas debe expandirse exactamente a 4 instancias."""
    # Del lunes 2026-04-06 al domingo 2026-05-03 son exactamente 4 semanas completas (4 lunes)
    start_date = date(2026, 4, 6)  # Monday
    end_date = date(2026, 5, 3)    # Sunday

    block = RecurringBlock(
        name="Cálculo III - Teoría",
        course_id="calc3",
        type=BlockType.CLASS,
        day_of_week=[DayOfWeek.MONDAY],
        start_time=time(8, 0),
        end_time=time(10, 0),
        from_date=date(2026, 3, 1),
        to_date=date(2026, 7, 1),
    )

    expanded = expand_recurring_blocks([block], start_date, end_date)
    assert len(expanded) == 4
    for b in expanded:
        assert b.name == "Cálculo III - Teoría"
        assert b.start.time() == time(8, 0)
        assert b.end.time() == time(10, 0)
        assert b.date.weekday() == 0  # Lunes


def test_expand_multiple_days_of_week():
    """Un bloque con múltiples días (lunes y miércoles) en 2 semanas debe producir 4 instancias."""
    start_date = date(2026, 4, 6)   # Monday
    end_date = date(2026, 4, 19)   # Sunday (2 semanas: 2 lunes y 2 miércoles)

    block = RecurringBlock(
        name="Base de Datos",
        course_id="bd",
        type=BlockType.CLASS,
        day_of_week=[DayOfWeek.MONDAY, DayOfWeek.WEDNESDAY],
        start_time=time(10, 0),
        end_time=time(12, 0),
        from_date=date(2026, 3, 1),
        to_date=date(2026, 7, 1),
    )

    expanded = expand_recurring_blocks([block], start_date, end_date)
    assert len(expanded) == 4
    weekdays = [b.date.weekday() for b in expanded]
    assert weekdays.count(0) == 2  # 2 lunes
    assert weekdays.count(2) == 2  # 2 miércoles


def test_expand_events_with_and_without_start_time():
    """Verifica que eventos puntuales con y sin hora se expanden correctamente."""
    events = [
        Event(
            id="parcial_calc",
            name="Parcial Cálculo",
            type=EventType.EXAM,
            course_id="calc3",
            date=date(2026, 5, 12),
            start_time=time(9, 0),
            duration_minutes=120,
        ),
        Event(
            id="entrega_proyecto",
            name="Entrega Avance",
            type=EventType.ENTREGABLE,
            course_id="bd",
            date=date(2026, 5, 15),
            start_time=None,  # Todo el día / deadline
        ),
    ]

    expanded = expand_events(events, date(2026, 5, 1), date(2026, 5, 31))
    assert len(expanded) == 2

    exam_block = [b for b in expanded if b.event_id == "parcial_calc"][0]
    assert exam_block.start.time() == time(9, 0)
    assert exam_block.end.time() == time(11, 0)
    assert exam_block.source == "event"

    deadline_block = [b for b in expanded if b.event_id == "entrega_proyecto"][0]
    assert deadline_block.start.time() == time(0, 0)
    assert deadline_block.end.time() == time(0, 0)
    assert deadline_block.source == "event"


def test_expand_all_merges_and_sorts_chronologically():
    """expand_all combina recurrentes y eventos y los ordena cronológicamente."""
    cal = Calendar(
        metadata=Metadata(
            student_name="Test",
            semester="2026-1",
            semester_start=date(2026, 3, 1),
            semester_end=date(2026, 7, 1),
            exam_weeks={},
        ),
        courses=[Course(id="c1", name="C1", difficulty=3, sessions_per_week=1)],
        recurring_blocks=[
            RecurringBlock(
                name="Clase Tarde",
                course_id="c1",
                type=BlockType.CLASS,
                day_of_week=[DayOfWeek.MONDAY],
                start_time=time(14, 0),
                end_time=time(16, 0),
                from_date=date(2026, 4, 1),
                to_date=date(2026, 6, 1),
            )
        ],
        events=[
            Event(
                id="ev_manana",
                name="Práctica Mañana",
                type=EventType.PRACTICA,
                course_id="c1",
                date=date(2026, 4, 6),  # Lunes 6 de abril
                start_time=time(8, 0),
                duration_minutes=60,
            )
        ],
    )

    expanded = expand_all(cal, date(2026, 4, 6), date(2026, 4, 6))
    assert len(expanded) == 2
    # El de las 8am debe ser el primero
    assert expanded[0].name == "Práctica Mañana"
    assert expanded[0].start.time() == time(8, 0)
    # El de las 14pm debe ser el segundo
    assert expanded[1].name == "Clase Tarde"
    assert expanded[1].start.time() == time(14, 0)
