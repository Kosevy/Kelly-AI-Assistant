"""
tests/test_calendar_ergonomics.py — Pruebas para ergonomía y diagnóstico del parser de calendario.

Verifica:
1. Normalización de aliases declarados para BlockType y EventType (case-insensitive, whitespace).
2. Mensaje descriptivo ante tipos desconocidos (valores permitidos + sugerencias de aliases).
3. Errores contextuales con ubicación precisa (section, index, name, field).
4. Detección de claves duplicadas en YAML (DuplicateKeyLoader).
5. Campos extra ignorados silenciosamente en modelo pero con logging INFO.
6. Validación al arrancar y estructura de salud para /health (sin detener el servidor).
"""
from __future__ import annotations

import logging
from datetime import date, time
from pathlib import Path
import pytest
from pydantic import ValidationError

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
    ValidationIssue,
    check_calendar_health,
    load_calendar,
    validate_calendar,
)
from app.config import settings
from app.logging_config import logger


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def capture_logs():
    """Captura los logs emitidos por loguru en memoria durante la prueba."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{level} | {message}")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


@pytest.fixture
def minimal_calendar_dict() -> dict:
    """Diccionario mínimo válido para serializar como YAML en los tests."""
    return {
        "metadata": {
            "student_name": "Test Student",
            "semester": "2026-1",
            "semester_start": "2026-03-15",
            "semester_end": "2026-07-20",
            "exam_weeks": {"parciales": 8},
        },
        "courses": [
            {
                "id": "calc3",
                "name": "Cálculo III",
                "difficulty": 4,
                "sessions_per_week": 2,
                "topics": ["Integrales"],
            }
        ],
        "recurring_blocks": [
            {
                "name": "Clase Cálculo",
                "course_id": "calc3",
                "type": "class",
                "day_of_week": ["monday", "wednesday"],
                "start_time": "08:00",
                "end_time": "10:00",
                "from_date": "2026-03-15",
                "to_date": "2026-07-20",
            }
        ],
        "events": [
            {
                "id": "ev_parcial",
                "name": "Examen Parcial",
                "type": "exam",
                "course_id": "calc3",
                "date": "2026-05-15",
                "start_time": "08:00",
                "duration_minutes": 120,
            }
        ],
    }


# ─── 1. Aliases Normalization ─────────────────────────────────────────────────

def test_aliases_normalization():
    """Verifica que todos los aliases declarados en config.py se resuelvan a sus enums canónicos."""
    # Aliases de bloques recurrentes
    block_alias_cases = [
        ("commute", BlockType.TRANSPORT),
        ("COMMUTE", BlockType.TRANSPORT),
        ("  commute  ", BlockType.TRANSPORT),
        ("transportation", BlockType.TRANSPORT),
        ("Transportation", BlockType.TRANSPORT),
        ("personal", BlockType.MEAL),
        ("PERSONAL", BlockType.MEAL),
    ]
    for raw_val, expected_enum in block_alias_cases:
        block = RecurringBlock(
            name="Test Block",
            type=raw_val,
            day_of_week=[DayOfWeek.MONDAY],
            start_time=time(8, 0),
            end_time=time(10, 0),
            from_date=date(2026, 3, 15),
            to_date=date(2026, 7, 20),
        )
        assert block.type == expected_enum, f"Fallo al normalizar {raw_val} -> {expected_enum}"

    # Aliases de eventos
    event_alias_cases = [
        ("assignment", EventType.ENTREGABLE),
        ("Assignment", EventType.ENTREGABLE),
        ("  task  ", EventType.ENTREGABLE),
        ("deliverable", EventType.ENTREGABLE),
        ("project", EventType.ENTREGABLE),
        ("homework", EventType.ENTREGABLE),
    ]
    for raw_val, expected_enum in event_alias_cases:
        ev = Event(
            id="test_ev",
            name="Test Event",
            type=raw_val,
            date=date(2026, 5, 10),
        )
        assert ev.type == expected_enum, f"Fallo al normalizar {raw_val} -> {expected_enum}"

    # Valores canónicos directos siguen funcionando
    assert RecurringBlock(
        name="Clase",
        type=BlockType.CLASS,
        day_of_week=[DayOfWeek.MONDAY],
        start_time=time(8, 0),
        end_time=time(10, 0),
        from_date=date(2026, 3, 15),
        to_date=date(2026, 7, 20),
    ).type == BlockType.CLASS

    assert Event(
        id="ex1",
        name="Examen",
        type="exam",
        date=date(2026, 5, 10),
    ).type == EventType.EXAM


def test_aliases_unknown_value():
    """Un valor no reconocido lanza error descriptivo con valores válidos y aliases sugeridos."""
    # En RecurringBlock
    with pytest.raises(ValidationError) as exc_info:
        RecurringBlock(
            name="Bloque Inválido",
            type="tipo_inexistente",
            day_of_week=[DayOfWeek.MONDAY],
            start_time=time(8, 0),
            end_time=time(10, 0),
            from_date=date(2026, 3, 15),
            to_date=date(2026, 7, 20),
        )
    err_str = str(exc_info.value)
    assert "type='tipo_inexistente' no válido para bloques recurrentes" in err_str
    assert "Valores permitidos: class, transport, meal, exercise, study, other" in err_str
    assert "commute→transport" in err_str

    # En Event
    with pytest.raises(ValidationError) as exc_info_ev:
        Event(
            id="ev_inv",
            name="Evento Inválido",
            type="tipo_inexistente",
            date=date(2026, 5, 10),
        )
    err_ev_str = str(exc_info_ev.value)
    assert "type='tipo_inexistente' no válido para eventos" in err_ev_str
    assert "Valores permitidos: exam, entregable, presentacion, practica, personal" in err_ev_str
    assert "assignment→entregable" in err_ev_str


# ─── 2. Contextual Errors & Location ──────────────────────────────────────────

def test_contextual_error_includes_location(tmp_path: Path):
    """Verifica que los errores de validación incluyan sección, índice, nombre y campo."""
    # 1. Test unitario de ValidationIssue
    issue = ValidationIssue(
        section="recurring_blocks",
        index=3,
        name="Transporte ida",
        field="course_id",
        message="course_id 'desconocido' no existe en 'courses'.",
    )
    assert issue.location() == 'recurring_blocks[3] "Transporte ida"'
    assert str(issue) == 'recurring_blocks[3] "Transporte ida": course_id: course_id \'desconocido\' no existe en \'courses\'.'

    # 2. Test semántico a través de validate_calendar
    cal = Calendar(
        metadata=Metadata(
            student_name="Test",
            semester="2026-1",
            semester_start=date(2026, 3, 15),
            semester_end=date(2026, 7, 20),
        ),
        courses=[Course(id="calc3", name="Cálculo III", difficulty=4, sessions_per_week=2)],
        recurring_blocks=[
            RecurringBlock(
                name="Transporte",
                course_id="inexistente",
                type=BlockType.TRANSPORT,
                day_of_week=[DayOfWeek.MONDAY],
                start_time=time(7, 0),
                end_time=time(8, 0),
                from_date=date(2026, 3, 15),
                to_date=date(2026, 7, 20),
            )
        ],
        events=[
            Event(
                id="ev_1",
                name="Entrega 1",
                type=EventType.ENTREGABLE,
                course_id="inexistente_ev",
                date=date(2026, 4, 1),
            )
        ],
    )
    issues = validate_calendar(cal)
    assert len(issues) == 2

    # Primer issue: bloque recurrente
    assert issues[0].section == "recurring_blocks"
    assert issues[0].index == 0
    assert issues[0].name == "Transporte"
    assert issues[0].field == "course_id"
    assert "inexistente" in issues[0].message

    # Segundo issue: evento
    assert issues[1].section == "events"
    assert issues[1].index == 0
    assert issues[1].name == "Entrega 1"
    assert issues[1].field == "course_id"
    assert "inexistente_ev" in issues[1].message

    # 3. Test de carga vía load_calendar con CalendarValidationError
    yaml_content = """
metadata:
  student_name: "Test"
  semester: "2026-1"
  semester_start: 2026-03-15
  semester_end: 2026-07-20
courses:
  - id: "calc3"
    name: "Cálculo III"
    difficulty: 4
    sessions_per_week: 2
recurring_blocks:
  - name: "Transporte"
    course_id: "inexistente"
    type: "transport"
    day_of_week: ["monday"]
    start_time: "07:00"
    end_time: "08:00"
    from_date: 2026-03-15
    to_date: 2026-07-20
"""
    cal_file = tmp_path / "calendar_invalid_course.yaml"
    cal_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(CalendarValidationError) as exc_info:
        load_calendar(cal_file)

    err = exc_info.value
    assert len(err.issues) == 1
    assert err.issues[0].name == "Transporte"
    err_list = err.to_list()
    assert len(err_list) == 1
    assert 'recurring_blocks[0] "Transporte"' in err_list[0]


# ─── 3. Duplicate YAML Keys Detection ─────────────────────────────────────────

def test_duplicate_yaml_keys_detected(tmp_path: Path):
    """YAML con claves repetidas en el mismo mapeo falla con mensaje explícito y línea."""
    duplicate_events_yaml = """
metadata:
  student_name: "Test"
  semester: "2026-1"
  semester_start: 2026-03-15
  semester_end: 2026-07-20
courses:
  - id: "calc3"
    name: "Cálculo III"
    difficulty: 4
    sessions_per_week: 2
events:
  - id: "ev1"
    name: "Evento 1"
    type: "exam"
    date: 2026-05-15
events:
  - id: "ev2"
    name: "Evento 2"
    type: "entregable"
    date: 2026-05-20
"""
    cal_file = tmp_path / "calendar_duplicate_keys.yaml"
    cal_file.write_text(duplicate_events_yaml, encoding="utf-8")

    with pytest.raises(CalendarValidationError) as exc_info:
        load_calendar(cal_file)

    msg = str(exc_info.value)
    assert "clave duplicada encontrada: 'events'" in msg
    assert "Fusiona ambos bloques en uno solo." in msg


# ─── 4. Extra Fields Ignored With Warning ─────────────────────────────────────

def test_extra_fields_ignored_with_warning(tmp_path: Path, capture_logs: list[str]):
    """Campos desconocidos son ignorados silenciosamente por el modelo pero loguean INFO."""
    yaml_with_extras = """
metadata:
  student_name: "Test"
  semester: "2026-1"
  semester_start: 2026-03-15
  semester_end: 2026-07-20
  extra_meta_field: "ignored"
courses:
  - id: "calc3"
    name: "Cálculo III"
    difficulty: 4
    sessions_per_week: 2
    credits: 4
recurring_blocks:
  - name: "Clase Calc"
    course_id: "calc3"
    type: "class"
    day_of_week: ["monday"]
    start_time: "08:00"
    end_time: "10:00"
    from_date: 2026-03-15
    to_date: 2026-07-20
    classroom: "B-201"
events:
  - id: "ev1"
    name: "Tarea 1"
    type: "entregable"
    date: 2026-05-10
    week: 6
unknown_root_section:
  foo: "bar"
"""
    cal_file = tmp_path / "calendar_extras.yaml"
    cal_file.write_text(yaml_with_extras, encoding="utf-8")

    cal = load_calendar(cal_file)

    # 1. La carga tuvo éxito a pesar de los campos extra (extra="ignore")
    assert cal.metadata.student_name == "Test"
    assert len(cal.courses) == 1
    assert len(cal.recurring_blocks) == 1
    assert len(cal.events) == 1

    # 2. Los warnings INFO fueron registrados para los campos desconocidos
    log_text = "".join(capture_logs)
    assert "[calendar] Campo 'unknown_root_section' en la raíz" in log_text
    assert "[calendar] Campo 'extra_meta_field' en metadata" in log_text
    assert "[calendar] Campo 'credits' en courses[0] \"Cálculo III\"" in log_text
    assert "[calendar] Campo 'classroom' en recurring_blocks[0] \"Clase Calc\"" in log_text
    assert "[calendar] Campo 'week' en events[0] \"Tarea 1\"" in log_text


# ─── 5. Startup Validation & Health Check ──────────────────────────────────────

def test_startup_validates_calendar_healthy(tmp_path: Path):
    """check_calendar_health con YAML válido retorna status='ok' y lista vacía de errores."""
    valid_yaml = """
metadata:
  student_name: "Kelly Test"
  semester: "2026-1"
  semester_start: 2026-03-15
  semester_end: 2026-07-20
courses:
  - id: "calc3"
    name: "Cálculo III"
    difficulty: 4
    sessions_per_week: 2
recurring_blocks: []
events: []
"""
    cal_file = tmp_path / "valid_cal.yaml"
    cal_file.write_text(valid_yaml, encoding="utf-8")

    health = check_calendar_health(cal_file)
    assert health["status"] == "ok"
    assert health["errors"] == []
    assert health["path"] == str(cal_file)


def test_startup_validates_calendar_error(tmp_path: Path, capture_logs: list[str]):
    """check_calendar_health con YAML inválido retorna status='error' y no lanza excepción."""
    invalid_yaml = """
metadata:
  student_name: "Kelly Test"
  semester: "2026-1"
  semester_start: 2026-07-20
  semester_end: 2026-03-15
courses: []
recurring_blocks: []
events: []
"""
    cal_file = tmp_path / "invalid_cal.yaml"
    cal_file.write_text(invalid_yaml, encoding="utf-8")

    health = check_calendar_health(cal_file)

    assert health["status"] == "error"
    assert len(health["errors"]) >= 1
    assert health["path"] == str(cal_file)
    assert "semester_start" in health["errors"][0]
    # Se logueó como ERROR pero no rompió la ejecución
    log_text = "".join(capture_logs)
    assert "[calendar] ERROR" in log_text


def test_startup_validates_calendar_missing_file(tmp_path: Path):
    """check_calendar_health con archivo inexistente reporta error limpiamente."""
    nonexistent = tmp_path / "nonexistent.yaml"
    health = check_calendar_health(nonexistent)
    assert health["status"] == "error"
    assert len(health["errors"]) == 1
    assert "no existe" in health["errors"][0]


def test_health_endpoint_contract():
    """Verifica que el contrato del endpoint /health degrade cuando el calendario tiene error."""
    # Simula la lógica del endpoint /health de server.py
    def mock_health_response(calendar_health_state: dict) -> tuple[int, dict]:
        components = {
            "stt": {"status": "ok"},
            "llm": {"status": "ok"},
            "tts": {"status": "ok"},
            "calendar": dict(calendar_health_state),
        }
        all_ok = True
        if calendar_health_state.get("status") == "error":
            all_ok = False
        return (200 if all_ok else 503, {"status": "ok" if all_ok else "degraded", "components": components})

    # Caso 1: Calendario saludable
    code, resp = mock_health_response({"status": "ok", "errors": [], "path": "data/calendar.yaml"})
    assert code == 200
    assert resp["status"] == "ok"
    assert resp["components"]["calendar"]["status"] == "ok"

    # Caso 2: Calendario con error -> servicio degradado (503)
    code, resp = mock_health_response({
        "status": "error",
        "errors": ['recurring_blocks[0] "Transporte": course_id no existe'],
        "path": "data/calendar.yaml",
    })
    assert code == 503
    assert resp["status"] == "degraded"
    assert resp["components"]["calendar"]["status"] == "error"
    assert len(resp["components"]["calendar"]["errors"]) == 1
