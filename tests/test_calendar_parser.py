"""
tests/test_calendar_parser.py
Pruebas unitarias para el parser, validador y persistencia del calendario local (Bloque A).
"""
from __future__ import annotations

import os
from datetime import date, time
from pathlib import Path
from unittest.mock import patch
import pytest

from app.calendar import (
    BlockType,
    Calendar,
    CalendarValidationError,
    Course,
    DayOfWeek,
    Event,
    EventType,
    Metadata,
    RecurringBlock,
    load_calendar,
    save_calendar,
    validate_calendar,
)


@pytest.fixture
def sample_valid_calendar() -> Calendar:
    return Calendar(
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
                topics=["Series", "Integrales múltiples"],
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
            ),
            RecurringBlock(
                name="Almuerzo",
                type=BlockType.MEAL,
                day_of_week=[DayOfWeek.MONDAY, DayOfWeek.TUESDAY],
                start_time=time(13, 0),
                end_time=time(14, 0),
                from_date=date(2026, 3, 15),
                to_date=date(2026, 7, 15),
            ),
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
                weight=0.3,
                topics_covered=["Series"],
                notes="Capítulos 1-3",
            ),
            Event(
                id="entrega_bd_1",
                name="Entrega BD",
                type=EventType.ENTREGABLE,
                course_id="bd",
                date=date(2026, 4, 5),
                estimated_hours=10,
            ),
        ],
    )


def test_load_example_calendar():
    """Verifica que el archivo calendar.yaml.example es válido y parsea correctamente."""
    example_path = Path("data/calendar.yaml.example")
    assert example_path.exists(), "El archivo calendar.yaml.example debe existir"

    cal = load_calendar(example_path)
    assert cal.metadata.semester == "2026-1"
    assert len(cal.courses) == 2
    assert cal.courses[0].id == "calc3"
    assert len(cal.recurring_blocks) == 3
    assert len(cal.events) == 2
    assert cal.events[0].id == "parcial_calc3"


def test_load_invalid_course_id_in_recurring_block(tmp_path: Path):
    """Carga de YAML con course_id inexistente en recurring_blocks debe lanzar CalendarValidationError."""
    yaml_content = """
metadata:
  student_name: "Test"
  semester: "2026-1"
  semester_start: 2026-03-15
  semester_end: 2026-07-20
  exam_weeks: {}
courses:
  - id: "calc3"
    name: "Cálculo III"
    difficulty: 3
    sessions_per_week: 2
recurring_blocks:
  - name: "Clase Fantasma"
    course_id: "fisica1"
    type: "class"
    day_of_week: ["monday"]
    start_time: "08:00"
    end_time: "10:00"
    from_date: 2026-03-15
    to_date: 2026-07-15
events: []
"""
    file_path = tmp_path / "calendar.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(CalendarValidationError) as exc_info:
        load_calendar(file_path)
    assert "fisica1" in str(exc_info.value)


def test_load_invalid_course_id_in_event(tmp_path: Path):
    """Carga de YAML con course_id inexistente en events debe lanzar CalendarValidationError."""
    yaml_content = """
metadata:
  student_name: "Test"
  semester: "2026-1"
  semester_start: 2026-03-15
  semester_end: 2026-07-20
  exam_weeks: {}
courses: []
recurring_blocks: []
events:
  - id: "exam_1"
    name: "Examen Inexistente"
    type: "exam"
    course_id: "quimica"
    date: 2026-05-10
"""
    file_path = tmp_path / "calendar.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(CalendarValidationError) as exc_info:
        load_calendar(file_path)
    assert "quimica" in str(exc_info.value)


def test_load_incoherent_dates(tmp_path: Path):
    """Fechas incoherentes (from_date >= to_date o start_time >= end_time) deben lanzar error."""
    yaml_content = """
metadata:
  student_name: "Test"
  semester: "2026-1"
  semester_start: 2026-07-20
  semester_end: 2026-03-15
  exam_weeks: {}
courses: []
recurring_blocks: []
events: []
"""
    file_path = tmp_path / "calendar.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(CalendarValidationError) as exc_info:
        load_calendar(file_path)
    assert "semester_start" in str(exc_info.value)


def test_load_incoherent_times_in_block(tmp_path: Path):
    """Bloque con end_time <= start_time debe lanzar CalendarValidationError."""
    yaml_content = """
metadata:
  student_name: "Test"
  semester: "2026-1"
  semester_start: 2026-03-15
  semester_end: 2026-07-20
  exam_weeks: {}
courses: []
recurring_blocks:
  - name: "Horario invertido"
    type: "study"
    day_of_week: ["monday"]
    start_time: "14:00"
    end_time: "12:00"
    from_date: 2026-03-15
    to_date: 2026-07-15
events: []
"""
    file_path = tmp_path / "calendar.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(CalendarValidationError) as exc_info:
        load_calendar(file_path)
    assert "end_time" in str(exc_info.value)


def test_load_duplicate_event_ids(tmp_path: Path):
    """Eventos con IDs duplicados deben provocar error de validación."""
    yaml_content = """
metadata:
  student_name: "Test"
  semester: "2026-1"
  semester_start: 2026-03-15
  semester_end: 2026-07-20
  exam_weeks: {}
courses: []
recurring_blocks: []
events:
  - id: "dup_id"
    name: "Parcial A"
    type: "exam"
    date: 2026-05-10
  - id: "dup_id"
    name: "Parcial B"
    type: "exam"
    date: 2026-05-12
"""
    file_path = tmp_path / "calendar.yaml"
    file_path.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(CalendarValidationError) as exc_info:
        load_calendar(file_path)
    assert "duplicado" in str(exc_info.value).lower()


def test_save_and_load_roundtrip(tmp_path: Path, sample_valid_calendar: Calendar):
    """Verifica que guardar y volver a cargar produce un modelo idéntico."""
    cal_file = tmp_path / "saved_calendar.yaml"
    save_calendar(sample_valid_calendar, cal_file, backup=False)

    loaded = load_calendar(cal_file)
    assert loaded == sample_valid_calendar


def test_save_with_backup_and_rotation(tmp_path: Path, sample_valid_calendar: Calendar):
    """Verifica la generación de backups y su rotación respetando el límite máximo N."""
    cal_file = tmp_path / "calendar.yaml"
    backup_dir = tmp_path / "backups"

    # Primer guardado (sin backup previo porque el archivo aún no existía)
    save_calendar(sample_valid_calendar, cal_file, backup=True, backup_dir=backup_dir, backup_count=3)
    assert cal_file.exists()
    assert len(list(backup_dir.glob("calendar_*.yaml"))) == 0

    # Realizamos 5 modificaciones guardadas
    for i in range(5):
        sample_valid_calendar.metadata.student_name = f"Estudiante {i}"
        save_calendar(
            sample_valid_calendar,
            cal_file,
            backup=True,
            backup_dir=backup_dir,
            backup_count=3,
        )

    # Debe haber rotado y conservado como máximo 3 backups
    backups = list(backup_dir.glob("calendar_*.yaml"))
    assert len(backups) == 3


def test_atomic_write_does_not_corrupt_on_failure(tmp_path: Path, sample_valid_calendar: Calendar):
    """Verifica que si ocurre un error durante el volcado, el archivo original no se corrompe ni se borra."""
    cal_file = tmp_path / "calendar.yaml"
    save_calendar(sample_valid_calendar, cal_file, backup=False)
    original_content = cal_file.read_text(encoding="utf-8")

    # Simulamos fallo durante os.replace
    with patch("os.replace", side_effect=OSError("Simulated disk error")):
        with pytest.raises(CalendarValidationError):
            save_calendar(sample_valid_calendar, cal_file, backup=False)

    # El archivo original debe permanecer intacto
    assert cal_file.read_text(encoding="utf-8") == original_content
    # El archivo temporal no debe haber quedado abandonado
    assert not (tmp_path / "calendar.yaml.tmp").exists()
