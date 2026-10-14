"""
tests/test_calendar_tools.py
Pruebas unitarias para las herramientas de calendario ejecutables por el LLM (Bloque C).
"""
from __future__ import annotations

import shutil
from datetime import date, time
from pathlib import Path
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
    load_calendar,
    save_calendar,
)
from app.calendar.tools import execute_calendar_tool
from app.config import settings
from app.tools import execute_tool, TOOLS


@pytest.fixture
def test_calendar_file(tmp_path: Path) -> Path:
    cal_file = tmp_path / "test_calendar.yaml"
    cal = Calendar(
        metadata=Metadata(
            student_name="Estudiante UPC",
            semester="2026-1",
            semester_start=date(2026, 3, 15),
            semester_end=date(2026, 7, 20),
            exam_weeks={"parciales": 8, "finales": 15},
        ),
        courses=[
            Course(
                id="calc3",
                name="Cálculo III",
                difficulty=4,
                sessions_per_week=2,
                topics=["Series", "Integrales"],
            ),
            Course(
                id="bd",
                name="Base de Datos",
                difficulty=3,
                sessions_per_week=2,
                topics=["SQL", "Normalización"],
            ),
        ],
        recurring_blocks=[
            RecurringBlock(
                name="Cálculo III - Teoría",
                course_id="calc3",
                type=BlockType.CLASS,
                day_of_week=[DayOfWeek.MONDAY, DayOfWeek.WEDNESDAY],
                start_time=time(8, 0),
                end_time=time(10, 0),
                from_date=date(2026, 3, 15),
                to_date=date(2026, 7, 15),
            )
        ],
        events=[
            Event(
                id="parcial_calc3",
                name="Parcial Cálculo III",
                type=EventType.EXAM,
                course_id="calc3",
                date=date(2026, 5, 12),
                start_time=time(10, 0),
                duration_minutes=120,
            ),
            Event(
                id="entrega_bd_1",
                name="Entrega Proyecto BD",
                type=EventType.ENTREGABLE,
                course_id="bd",
                date=date(2026, 4, 5),
            ),
        ],
    )
    save_calendar(cal, cal_file, backup=False)
    return cal_file


@pytest.mark.asyncio
async def test_tool_calendar_read_range(test_calendar_file: Path):
    """calendar_read_range retorna los eventos y clases correspondientes a ese rango."""
    # Rango de la semana del 2026-05-11 al 2026-05-17 (incluye Lunes 11 [clase], Martes 12 [parcial], Miércoles 13 [clase])
    res = await execute_calendar_tool(
        "calendar_read_range",
        {"start_date": "2026-05-11", "end_date": "2026-05-17"},
        calendar_path=test_calendar_file,
    )

    assert "Parcial Cálculo III" in res
    assert "Cálculo III - Teoría" in res
    assert "2026-05-12" in res
    assert "[exam]" in res


@pytest.mark.asyncio
async def test_tool_calendar_read_range_with_type_filter(test_calendar_file: Path):
    """calendar_read_range con filtro 'exam' solo retorna exámenes."""
    res = await execute_calendar_tool(
        "calendar_read_range",
        {"start_date": "2026-05-11", "end_date": "2026-05-17", "types": ["exam"]},
        calendar_path=test_calendar_file,
    )

    assert "Parcial Cálculo III" in res
    assert "Cálculo III - Teoría" not in res


@pytest.mark.asyncio
async def test_tool_calendar_free_slots(test_calendar_file: Path):
    """calendar_free_slots calcula los bloques disponibles entre 6:00 y 23:00."""
    # Lunes 2026-04-06 tiene clase de 8:00 a 10:00
    res = await execute_calendar_tool(
        "calendar_free_slots",
        {"start_date": "2026-04-06", "end_date": "2026-04-06", "min_block_minutes": 60},
        calendar_path=test_calendar_file,
    )

    assert "06:00-08:00" in res
    assert "10:00-23:00" in res


@pytest.mark.asyncio
async def test_tool_calendar_add_event_success(test_calendar_file: Path, monkeypatch):
    """calendar_add_event añade evento, crea backup y devuelve el event_id."""
    backup_dir = test_calendar_file.parent / "backups"
    monkeypatch.setattr(settings, "calendar_backup_dir", str(backup_dir))

    res = await execute_calendar_tool(
        "calendar_add_event",
        {
            "name": "Práctica Calificada 2",
            "date": "2026-04-20",
            "type": "practica",
            "course_id": "calc3",
            "start_time": "08:00",
            "duration_minutes": 90,
            "notes": "Traer calculadora",
        },
        calendar_path=test_calendar_file,
    )

    assert "añadido exitosamente" in res.lower()
    assert "practica_calificada_2_20260420" in res

    # Verificar que el YAML fue efectivamente actualizado en disco
    updated_cal = load_calendar(test_calendar_file)
    added = [e for e in updated_cal.events if "practica_calificada_2" in e.id]
    assert len(added) == 1
    assert added[0].duration_minutes == 90
    assert added[0].notes == "Traer calculadora"

    # Verificar creación de backup
    backups = list(backup_dir.glob("calendar_*.yaml"))
    assert len(backups) == 1


@pytest.mark.asyncio
async def test_tool_calendar_add_event_invalid_course(test_calendar_file: Path):
    """calendar_add_event con course_id inexistente retorna error comprensible."""
    res = await execute_calendar_tool(
        "calendar_add_event",
        {
            "name": "Examen Inexistente",
            "date": "2026-04-20",
            "type": "exam",
            "course_id": "fisica100",
        },
        calendar_path=test_calendar_file,
    )

    assert "Error:" in res
    assert "fisica100" in res
    assert "calc3" in res


@pytest.mark.asyncio
async def test_tool_calendar_list_courses(test_calendar_file: Path):
    """calendar_list_courses lista la información completa de cursos."""
    res = await execute_calendar_tool(
        "calendar_list_courses",
        {},
        calendar_path=test_calendar_file,
    )

    assert "Cálculo III" in res
    assert "Base de Datos" in res
    assert "Dificultad: 4/5" in res
    assert "Series" in res


@pytest.mark.asyncio
async def test_tool_calendar_delete_event(test_calendar_file: Path):
    """calendar_delete_event elimina el evento existente y rechaza IDs inexistentes."""
    # Eliminar existente
    res = await execute_calendar_tool(
        "calendar_delete_event",
        {"event_id": "parcial_calc3"},
        calendar_path=test_calendar_file,
    )
    assert "eliminado exitosamente" in res.lower()

    updated = load_calendar(test_calendar_file)
    assert not any(e.id == "parcial_calc3" for e in updated.events)

    # Intentar eliminar id inexistente
    res_err = await execute_calendar_tool(
        "calendar_delete_event",
        {"event_id": "id_inexistente"},
        calendar_path=test_calendar_file,
    )
    assert "Error:" in res_err
    assert "id_inexistente" in res_err


@pytest.mark.asyncio
async def test_dispatch_via_app_tools(test_calendar_file: Path, monkeypatch):
    """Verifica que execute_tool en app.tools despacha hacia calendar tools."""
    monkeypatch.setattr(settings, "calendar_path", str(test_calendar_file))

    res = await execute_tool("calendar_list_courses", {})
    assert "Cursos del semestre" in res
    assert "calc3" in res


def test_tools_list_contains_calendar_tools():
    """TOOLS de app.tools debe contener las herramientas de archivo y de calendario."""
    tool_names = {t["function"]["name"] for t in TOOLS}
    expected = {
        "read_file",
        "list_directory",
        "calendar_read_range",
        "calendar_free_slots",
        "calendar_add_event",
        "calendar_list_courses",
        "calendar_delete_event",
    }
    assert expected.issubset(tool_names)
