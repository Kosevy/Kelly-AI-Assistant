"""
app/calendar/models.py — Modelos Pydantic para el sistema de calendario local de Kelly.

Rationale:
- Define la fuente de verdad tipada para calendarios académicos y personales.
- Aplica validaciones de integridad referencial (course_id) y coherencia temporal
  (fechas, horarios y unicidad de identificadores) en tiempo de carga/creación.
"""
from __future__ import annotations

from datetime import date, time
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


class BlockType(str, Enum):
    """Tipos de bloques recurrentes en el horario."""
    CLASS = "class"
    TRANSPORT = "transport"
    MEAL = "meal"
    EXERCISE = "exercise"
    STUDY = "study"
    OTHER = "other"


class EventType(str, Enum):
    """Tipos de eventos puntuales o hitos del calendario."""
    EXAM = "exam"
    ENTREGABLE = "entregable"
    PRESENTACION = "presentacion"
    PRACTICA = "practica"
    PERSONAL = "personal"


class DayOfWeek(str, Enum):
    """Días de la semana para recurrencias."""
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class Metadata(BaseModel):
    """Metadatos globales del semestre académico."""
    student_name: str
    semester: str
    semester_start: date
    semester_end: date
    exam_weeks: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_dates(self) -> Metadata:
        if self.semester_start >= self.semester_end:
            raise ValueError(
                f"semester_start ({self.semester_start}) debe ser anterior a semester_end ({self.semester_end})"
            )
        return self


class Course(BaseModel):
    """Representación de un curso matriculado en el semestre."""
    id: str
    name: str
    difficulty: int = Field(ge=1, le=5)
    sessions_per_week: int = Field(ge=0, le=7)
    topics: list[str] = Field(default_factory=list)


class RecurringBlock(BaseModel):
    """Bloque con recurrencia semanal fija (clases, transporte, comidas)."""
    name: str
    course_id: str | None = None
    type: BlockType
    day_of_week: list[DayOfWeek] = Field(min_length=1)
    start_time: time
    end_time: time
    from_date: date
    to_date: date

    @model_validator(mode="after")
    def _validate_block_coherence(self) -> RecurringBlock:
        if self.from_date >= self.to_date:
            raise ValueError(
                f"Bloque '{self.name}': from_date ({self.from_date}) debe ser estrictamente anterior a to_date ({self.to_date})"
            )
        if self.end_time <= self.start_time:
            raise ValueError(
                f"Bloque '{self.name}': end_time ({self.end_time}) debe ser estrictamente posterior a start_time ({self.start_time})"
            )
        return self


class Event(BaseModel):
    """Evento puntual, hito académico o personal con fecha específica."""
    id: str
    name: str
    type: EventType
    course_id: str | None = None
    date: date
    start_time: time | None = None
    duration_minutes: int | None = None
    weight: float | None = None
    topics_covered: list[str] = Field(default_factory=list)
    estimated_hours: int | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_event_coherence(self) -> Event:
        if self.duration_minutes is not None and self.duration_minutes <= 0:
            raise ValueError(
                f"Evento '{self.id}': duration_minutes ({self.duration_minutes}) debe ser mayor a 0"
            )
        if self.weight is not None and not (0.0 <= self.weight <= 1.0):
            raise ValueError(
                f"Evento '{self.id}': weight ({self.weight}) debe estar comprendido entre 0.0 y 1.0"
            )
        return self


class Calendar(BaseModel):
    """Estructura raíz del calendario local (fuente de verdad YAML)."""
    metadata: Metadata
    courses: list[Course] = Field(default_factory=list)
    recurring_blocks: list[RecurringBlock] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_calendar_integrity(self) -> Calendar:
        course_ids = {c.id for c in self.courses}

        # Validar pertenencia de course_id en bloques recurrentes
        for block in self.recurring_blocks:
            if block.course_id and block.course_id not in course_ids:
                raise ValueError(
                    f"Bloque '{block.name}' referencia course_id inexistente: '{block.course_id}'"
                )

        # Validar pertenencia de course_id en eventos
        for ev in self.events:
            if ev.course_id and ev.course_id not in course_ids:
                raise ValueError(
                    f"Evento '{ev.id}' referencia course_id inexistente: '{ev.course_id}'"
                )

        # Validar identificadores únicos en eventos
        seen_event_ids: set[str] = set()
        for ev in self.events:
            if ev.id in seen_event_ids:
                raise ValueError(f"ID de evento duplicado en el calendario: '{ev.id}'")
            seen_event_ids.add(ev.id)

        return self
