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
    """calendar_read_range retorna los eventos y clases en formato semi-compacto con encabezados en negrita y duraciones."""
    # Rango de la semana del 2026-05-11 al 2026-05-17 (incluye Lunes 11 [clase], Martes 12 [parcial], Miércoles 13 [clase])
    res = await execute_calendar_tool(
        "calendar_read_range",
        {"start_date": "2026-05-11", "end_date": "2026-05-17"},
        calendar_path=test_calendar_file,
    )

    assert "**Lunes 11 de mayo**" in res
    assert "- 08:00-10:00 Cálculo III - Teoría (Cálculo III)" in res
    assert "**Martes 12 de mayo**" in res
    assert "- 10:00-12:00 Parcial Cálculo III (Cálculo III)" in res
    assert "**Miércoles 13 de mayo**" in res
    assert "- 08:00-10:00 Cálculo III - Teoría (Cálculo III)" in res
    assert "**Jueves 14 de mayo**" in res
    assert "- (sin eventos programados)" in res


@pytest.mark.asyncio
async def test_tool_calendar_read_range_semicompact_format_5_days(test_calendar_file: Path):
    """Rango de 5 días genera formato semi-compacto con encabezados de días y duración explícita."""
    res = await execute_calendar_tool(
        "calendar_read_range",
        {"start_date": "2026-05-11", "end_date": "2026-05-15"},
        calendar_path=test_calendar_file,
    )
    assert "**Lunes 11 de mayo**" in res
    assert "- 08:00-10:00 Cálculo III - Teoría (Cálculo III)" in res
    assert "**Martes 12 de mayo**" in res
    assert "- 10:00-12:00 Parcial Cálculo III (Cálculo III)" in res
    assert "**Miércoles 13 de mayo**" in res
    assert "**Jueves 14 de mayo**\n- (sin eventos programados)" in res
    assert "**Viernes 15 de mayo**\n- (sin eventos programados)" in res


@pytest.mark.asyncio
async def test_tool_calendar_read_range_scope_limit_30_days(test_calendar_file: Path):
    """Rango mayor a calendar_read_max_days (7 días) se acota a los primeros 7 días con mensaje explicativo."""
    res = await execute_calendar_tool(
        "calendar_read_range",
        {"start_date": "2026-05-01", "end_date": "2026-05-30"},
        calendar_path=test_calendar_file,
    )
    assert "[Rango completo: 30 días. Mostrando primeros 7. Pedir rangos más específicos para ver más.]" in res
    assert "**Viernes 1 de mayo**" in res
    assert "**Jueves 7 de mayo**" in res
    assert "**Viernes 8 de mayo**" not in res


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
    assert "**Martes 12 de mayo**" in res
    assert "**Lunes 11 de mayo**\n- (sin eventos programados)" in res


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


@pytest.mark.asyncio
async def test_tool_calendar_next_deadline_exam(test_calendar_file: Path):
    """calendar_next_deadline encuentra el próximo examen desde la fecha base."""
    res = await execute_calendar_tool(
        "calendar_next_deadline",
        {"types": ["exam"], "from_date": "2026-04-01"},
        calendar_path=test_calendar_file,
    )
    assert "Próximo: Parcial Cálculo III" in res
    assert "2026-05-12 10:00" in res
    assert "calc3" in res
    assert "faltan 41 días" in res


@pytest.mark.asyncio
async def test_tool_calendar_next_deadline_entregable(test_calendar_file: Path):
    """calendar_next_deadline con types=['entregable'] filtra correctamente."""
    res = await execute_calendar_tool(
        "calendar_next_deadline",
        {"types": ["entregable"], "from_date": "2026-04-01"},
        calendar_path=test_calendar_file,
    )
    assert "Próximo: Entrega Proyecto BD" in res
    assert "2026-04-05" in res
    assert "bd" in res
    assert "faltan 4 días" in res


@pytest.mark.asyncio
async def test_tool_calendar_next_deadline_empty_types(test_calendar_file: Path):
    """calendar_next_deadline sin types o con types=[] busca entre todos y devuelve el más cercano."""
    res = await execute_calendar_tool(
        "calendar_next_deadline",
        {"types": [], "from_date": "2026-04-01"},
        calendar_path=test_calendar_file,
    )
    assert "Próximo: Entrega Proyecto BD" in res

    res_no_types = await execute_calendar_tool(
        "calendar_next_deadline",
        {"from_date": "2026-04-01"},
        calendar_path=test_calendar_file,
    )
    assert "Próximo: Entrega Proyecto BD" in res_no_types


@pytest.mark.asyncio
async def test_tool_calendar_next_deadline_no_future_events(test_calendar_file: Path):
    """calendar_next_deadline cuando no hay eventos futuros devuelve mensaje informativo."""
    res = await execute_calendar_tool(
        "calendar_next_deadline",
        {"types": ["exam"], "from_date": "2026-06-01"},
        calendar_path=test_calendar_file,
    )
    assert "No hay eventos próximos" in res


def test_tools_list_contains_calendar_tools():
    """TOOLS de app.tools debe contener las herramientas de archivo y de calendario."""
    tool_names = {t["function"]["name"] for t in TOOLS}
    expected = {
        "read_file",
        "list_directory",
        "calendar_read_range",
        "calendar_free_slots",
        "calendar_next_deadline",
        "calendar_add_event",
        "calendar_list_courses",
        "calendar_delete_event",
    }
    assert expected.issubset(tool_names)


@pytest.mark.asyncio
async def test_tool_calendar_read_range_structured(test_calendar_file: Path):
    """calendar_read_range con return_structured=True devuelve un dict tipo calendar_card con días y eventos."""
    res = await execute_calendar_tool(
        "calendar_read_range",
        {"start_date": "2026-05-11", "end_date": "2026-05-17"},
        calendar_path=test_calendar_file,
        return_structured=True,
    )
    assert isinstance(res, dict)
    assert res["type"] == "calendar_card"
    assert res["view_type"] == "week"
    assert "Semana del 11 al 17 de mayo" in res["title"]
    assert len(res["days"]) == 7

    # Lunes 11 de mayo tiene clase de Cálculo III
    monday_data = res["days"][0]
    assert monday_data["date"] == "2026-05-11"
    assert "Lunes 11 de mayo" in monday_data["day_name"]
    assert len(monday_data["events"]) >= 1
    ev0 = monday_data["events"][0]
    assert ev0["start"] == "08:00"
    assert ev0["end"] == "10:00"
    assert ev0["name"] == "Cálculo III - Teoría"
    assert ev0["type"] == "class"
    assert ev0["course"] == "Cálculo III"


@pytest.mark.asyncio
async def test_tool_calendar_next_deadline_structured(test_calendar_file: Path):
    """calendar_next_deadline con return_structured=True devuelve el evento evaluativo estructurado."""
    res = await execute_calendar_tool(
        "calendar_next_deadline",
        {"types": ["exam"], "from_date": "2026-04-01"},
        calendar_path=test_calendar_file,
        return_structured=True,
    )
    assert isinstance(res, dict)
    assert res["type"] == "calendar_card"
    assert res["view_type"] == "deadline"
    assert res["deadline"] is not None
    assert res["deadline"]["name"] == "Parcial Cálculo III"
    assert res["deadline"]["date"] == "2026-05-12"
    assert res["deadline"]["type"] == "exam"
    assert res["deadline"]["course"] == "Cálculo III"
    assert "días" in res["deadline"]["days_left_text"]
    assert len(res["days"]) == 1


@pytest.mark.asyncio
async def test_tool_calendar_free_slots_structured(test_calendar_file: Path):
    """calendar_free_slots con return_structured=True devuelve los bloques libres estructurados."""
    res = await execute_calendar_tool(
        "calendar_free_slots",
        {"start_date": "2026-05-11", "end_date": "2026-05-11", "min_block_minutes": 60},
        calendar_path=test_calendar_file,
        return_structured=True,
    )
    assert isinstance(res, dict)
    assert res["type"] == "calendar_card"
    assert res["view_type"] == "free_slots"
    assert len(res["slots"]) > 0
    assert "duration_minutes" in res["slots"][0]
    assert len(res["days"]) == 1
