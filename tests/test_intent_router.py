"""
tests/test_intent_router.py
Pruebas unitarias exhaustivas para app.calendar.intent_router (Bloque A).
"""
from __future__ import annotations

from datetime import datetime
import pytest

from app.calendar.intent_router import detect_calendar_intent


@pytest.fixture
def fixed_thursday() -> datetime:
    """Jueves 15 de octubre de 2026 a las 14:30."""
    return datetime(2026, 10, 15, 14, 30)


class TestReadWeekIntent:
    def test_esta_semana_resolves_monday_to_sunday(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿Qué tengo esta semana?", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "read_week"
        assert res["tool_name"] == "calendar_read_range"
        assert res["params"]["start_date"] == "2026-10-12"  # Lunes
        assert res["params"]["end_date"] == "2026-10-18"    # Domingo

    def test_proxima_semana_resolves_next_monday_to_sunday(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿qué tengo la próxima semana?", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "read_week"
        assert res["tool_name"] == "calendar_read_range"
        assert res["params"]["start_date"] == "2026-10-19"  # Siguiente Lunes
        assert res["params"]["end_date"] == "2026-10-25"    # Siguiente Domingo

    def test_siguiente_semana_alias(self, fixed_thursday: datetime):
        res = detect_calendar_intent("horario de la siguiente semana", now=fixed_thursday)
        assert res is not None
        assert res["params"]["start_date"] == "2026-10-19"
        assert res["params"]["end_date"] == "2026-10-25"

    def test_agenda_de_la_semana_defaults_to_this_week(self, fixed_thursday: datetime):
        res = detect_calendar_intent("agenda de la semana", now=fixed_thursday)
        assert res is not None
        assert res["params"]["start_date"] == "2026-10-12"
        assert res["params"]["end_date"] == "2026-10-18"


class TestDayResolutionIntent:
    def test_hoy_resolves_current_day(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿Qué tengo hoy?", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "read_range"
        assert res["params"]["start_date"] == "2026-10-15"
        assert res["params"]["end_date"] == "2026-10-15"

    def test_manana_resolves_tomorrow(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿Qué hay mañana?", now=fixed_thursday)
        assert res is not None
        assert res["params"]["start_date"] == "2026-10-16"
        assert res["params"]["end_date"] == "2026-10-16"

    def test_el_lunes_resolves_next_monday(self, fixed_thursday: datetime):
        # Desde jueves 15, el próximo lunes es 19
        res = detect_calendar_intent("¿Qué tengo el lunes?", now=fixed_thursday)
        assert res is not None
        assert res["params"]["start_date"] == "2026-10-19"
        assert res["params"]["end_date"] == "2026-10-19"

    def test_el_viernes_resolves_tomorrow_friday(self, fixed_thursday: datetime):
        # Desde jueves 15, este viernes es 16
        res = detect_calendar_intent("horario del viernes", now=fixed_thursday)
        assert res is not None
        assert res["params"]["start_date"] == "2026-10-16"
        assert res["params"]["end_date"] == "2026-10-16"


class TestNextDeadlineIntent:
    def test_proximo_examen_maps_exam_type(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿Cuál es mi próximo examen?", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "next_deadline"
        assert res["tool_name"] == "calendar_next_deadline"
        assert res["params"] == {"types": ["exam"]}

    def test_que_entrega_viene_maps_entregable_type(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿Qué entrega viene?", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "next_deadline"
        assert res["params"] == {"types": ["entregable"]}

    def test_proxima_practica_maps_practica_and_exam_types(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿Cuándo es mi próxima práctica calificada?", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "next_deadline"
        assert res["params"] == {"types": ["practica", "exam"]}

    def test_proxima_pc_alias(self, fixed_thursday: datetime):
        res = detect_calendar_intent("próxima pc", now=fixed_thursday)
        assert res is not None
        assert res["params"] == {"types": ["practica", "exam"]}

    def test_generic_next_deadline_has_no_type_filter(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿Cuál es mi próximo deadline?", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "next_deadline"
        assert res["params"] == {}


class TestFreeSlotsIntent:
    def test_tiempo_para_estudiar(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿Tengo tiempo para estudiar hoy?", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "free_slots"
        assert res["tool_name"] == "calendar_free_slots"
        assert res["params"]["start_date"] == "2026-10-15"
        assert res["params"]["end_date"] == "2026-10-15"

    def test_huecos_libres_esta_semana(self, fixed_thursday: datetime):
        res = detect_calendar_intent("huecos libres esta semana", now=fixed_thursday)
        assert res is not None
        assert res["params"]["start_date"] == "2026-10-12"
        assert res["params"]["end_date"] == "2026-10-18"


class TestRobustnessAndCaseSensitivity:
    def test_uppercase_and_no_question_marks(self, fixed_thursday: datetime):
        res = detect_calendar_intent("CUAL ES MI PROXIMO EXAMEN", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "next_deadline"
        assert res["params"] == {"types": ["exam"]}

    def test_tengo_tiempo_manana(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿tengo tiempo mañana?", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "free_slots"
        assert res["params"]["start_date"] == "2026-10-16"
        assert res["params"]["end_date"] == "2026-10-16"

    def test_mi_horario_general(self, fixed_thursday: datetime):
        res = detect_calendar_intent("mi horario", now=fixed_thursday)
        assert res is not None
        assert res["intent"] == "read_week"
        assert res["params"]["start_date"] == "2026-10-12"
        assert res["params"]["end_date"] == "2026-10-18"


class TestReasoningAndFallbackQueries:
    def test_resume_mi_semana_returns_none(self, fixed_thursday: datetime):
        # Pregunta de razonamiento/resumen: DEBE ir al LLM
        res = detect_calendar_intent("Resume mi semana y dime qué priorizar", now=fixed_thursday)
        assert res is None

    def test_como_estudiar_returns_none(self, fixed_thursday: datetime):
        res = detect_calendar_intent("¿Cómo me preparo para el examen de FAS?", now=fixed_thursday)
        assert res is None

    def test_conversational_queries_return_none(self, fixed_thursday: datetime):
        assert detect_calendar_intent("Hola Kelly, ¿cómo estás?", now=fixed_thursday) is None
        assert detect_calendar_intent("Léeme el archivo README.md", now=fixed_thursday) is None
        assert detect_calendar_intent("Explícame qué es una arquitectura orientada a servicios", now=fixed_thursday) is None
        assert detect_calendar_intent("", now=fixed_thursday) is None
