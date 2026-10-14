"""
tests/test_calendar_e2e.py
Pruebas de integración E2E para los 5 flujos conversacionales de calendario (D5).
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
import pytest

from app.calendar import (
    load_calendar,
    next_deadline,
    expand_all,
)
from app.config import settings
from app.tools import execute_tool


@pytest.fixture
def e2e_calendar(tmp_path: Path, monkeypatch) -> Path:
    """Configura un calendar.yaml a partir del template de ejemplo y aísla los backups."""
    source_example = Path("data/calendar.yaml.example")
    cal_file = tmp_path / "calendar.yaml"
    shutil.copy2(source_example, cal_file)

    backup_dir = tmp_path / "backups"
    monkeypatch.setattr(settings, "calendar_path", str(cal_file))
    monkeypatch.setattr(settings, "calendar_backup_dir", str(backup_dir))
    return cal_file


@pytest.mark.asyncio
async def test_e2e_flow_1_read_week(e2e_calendar: Path):
    """Flujo 1: '¿Qué tengo esta semana?' (ej: semana del 2026-05-11 al 2026-05-17)."""
    result = await execute_tool(
        "calendar_read_range",
        {"start_date": "2026-05-11", "end_date": "2026-05-17"},
    )

    # Debe contener clases recurrentes y el parcial de Cálculo III
    assert "Parcial Cálculo III" in result
    assert "Cálculo III - Teoría" in result
    assert "Transporte ida" in result
    assert "Almuerzo" in result


@pytest.mark.asyncio
async def test_e2e_flow_2_next_exam_deadline(e2e_calendar: Path):
    """Flujo 2: '¿Cuándo es mi próximo examen?' (consultando desde el inicio de semestre 2026-03-15)."""
    cal = load_calendar(e2e_calendar)
    expanded = expand_all(cal, date(2026, 3, 15), date(2026, 7, 20))
    deadline = next_deadline(expanded, from_date=date(2026, 3, 15), types=["exam"])

    assert deadline is not None
    assert deadline.name == "Parcial Cálculo III"
    assert deadline.date == date(2026, 5, 12)
    assert deadline.start.time().strftime("%H:%M") == "10:00"


@pytest.mark.asyncio
async def test_e2e_flow_3_add_event(e2e_calendar: Path):
    """Flujo 3: 'Kelly, añade una práctica de Cálculo el 20 de abril a las 8am'."""
    result = await execute_tool(
        "calendar_add_event",
        {
            "name": "Práctica de Cálculo",
            "date": "2026-04-20",
            "type": "practica",
            "course_id": "calc3",
            "start_time": "08:00",
            "duration_minutes": 90,
        },
    )

    assert "añadido exitosamente" in result.lower()
    assert "practica_de_calculo_20260420" in result

    # Verificar actualización en el YAML
    cal = load_calendar(e2e_calendar)
    assert any(e.id == "practica_de_calculo_20260420" for e in cal.events)

    # Verificar creación de backup previo
    backup_files = list(Path(settings.calendar_backup_dir).glob("calendar_*.yaml"))
    assert len(backup_files) == 1


@pytest.mark.asyncio
async def test_e2e_flow_4_free_slots_availability(e2e_calendar: Path):
    """Flujo 4: '¿Tengo 90 minutos libres el miércoles?' (Miércoles 2026-04-08)."""
    # Miércoles tiene clase de Cálculo de 08:00 a 10:00.
    # Transporte y almuerzo NO bloquean disponibilidad.
    result = await execute_tool(
        "calendar_free_slots",
        {"start_date": "2026-04-08", "end_date": "2026-04-08", "min_block_minutes": 90},
    )

    # Huecos esperados: 06:00 a 08:00 (120 min) y 10:00 a 23:00 (780 min)
    assert "06:00-08:00 (120 min)" in result
    assert "10:00-23:00 (780 min)" in result


@pytest.mark.asyncio
async def test_e2e_flow_5_list_courses(e2e_calendar: Path):
    """Flujo 5: '¿Cuáles son mis cursos de este ciclo y qué dificultad tienen?'."""
    result = await execute_tool("calendar_list_courses", {})

    assert "Cálculo III" in result
    assert "Dificultad: 4/5" in result
    assert "Base de Datos" in result
    assert "Dificultad: 3/5" in result
