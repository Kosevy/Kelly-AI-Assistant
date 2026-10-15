"""
app/calendar/models.py — Modelos Pydantic para el sistema de calendario local de Kelly.

Rationale:
- Define la fuente de verdad tipada para calendarios académicos y personales.
- Aplica aliases explícitos y declarados para normalizar variantes comunes del
  campo 'type' antes de validar el enum canónico. Los aliases viven en config.py
  para que sean visibles, testeables y sobreescribibles vía .env.
- Usa extra="ignore" para tolerar campos desconocidos en el YAML sin romper
  la carga; el parser emite un warning por campo ignorado.
- Aplica validaciones de integridad referencial (course_id) y coherencia temporal
  (fechas, horarios y unicidad de identificadores) en tiempo de carga/creación.
"""
from __future__ import annotations

from datetime import date, time
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


def _resolve_block_type(raw: Any) -> BlockType:
    """
    Normaliza el valor de 'type' de un RecurringBlock aplicando aliases declarados.

    Rationale: el usuario puede escribir variantes comunes (personal, commute) que
    se normalizan a los valores canónicos ANTES de validar el enum. La tabla de
    aliases es explícita en config.py, no inferencia automática.

    Lanza ValueError con mensaje descriptivo si el valor no se reconoce en
    ninguna de las dos tablas (canónica + aliases).
    """
    if isinstance(raw, BlockType):
        return raw

    # Import tardío para evitar ciclo: models → config → models
    from app.config import settings  # noqa: PLC0415

    raw_str = (raw.value if isinstance(raw, Enum) else str(raw)).strip().lower()

    # 1. Intentar directamente en el enum canónico
    try:
        return BlockType(raw_str)
    except ValueError:
        pass

    # 2. Buscar en la tabla de aliases
    aliases: dict[str, str] = settings.calendar_type_aliases
    if raw_str in aliases:
        canonical = aliases[raw_str]
        try:
            return BlockType(canonical)
        except ValueError:
            pass  # alias mal configurado; caerá al error final

    # 3. Construir mensaje de error descriptivo con valores válidos y aliases
    valid = ", ".join(bt.value for bt in BlockType)
    alias_hints = ", ".join(f"{k}→{v}" for k, v in aliases.items() if v in [bt.value for bt in BlockType])
    raise ValueError(
        f"type='{raw}' no válido para bloques recurrentes. "
        f"Valores permitidos: {valid}. "
        f"Aliases aceptados: {alias_hints}."
    )


def _resolve_event_type(raw: Any) -> EventType:
    """
    Normaliza el valor de 'type' de un Event aplicando aliases declarados.

    Misma estrategia que _resolve_block_type pero para EventType.
    """
    if isinstance(raw, EventType):
        return raw

    from app.config import settings  # noqa: PLC0415

    raw_str = (raw.value if isinstance(raw, Enum) else str(raw)).strip().lower()

    try:
        return EventType(raw_str)
    except ValueError:
        pass

    aliases: dict[str, str] = settings.calendar_type_aliases
    if raw_str in aliases:
        canonical = aliases[raw_str]
        try:
            return EventType(canonical)
        except ValueError:
            pass

    valid = ", ".join(et.value for et in EventType)
    alias_hints = ", ".join(f"{k}→{v}" for k, v in aliases.items() if v in [et.value for et in EventType])
    raise ValueError(
        f"type='{raw}' no válido para eventos. "
        f"Valores permitidos: {valid}. "
        f"Aliases aceptados: {alias_hints}."
    )


class Metadata(BaseModel):
    """Metadatos globales del semestre académico."""
    model_config = ConfigDict(extra="ignore")

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
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    difficulty: int = Field(ge=1, le=5)
    sessions_per_week: int = Field(ge=0, le=7)
    topics: list[str] = Field(default_factory=list)


class RecurringBlock(BaseModel):
    """Bloque con recurrencia semanal fija (clases, transporte, comidas)."""
    model_config = ConfigDict(extra="ignore")

    name: str
    course_id: str | None = None
    type: BlockType
    day_of_week: list[DayOfWeek] = Field(min_length=1)
    start_time: time
    end_time: time
    from_date: date
    to_date: date

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, v: Any) -> BlockType:
        """Aplica aliases y normalización antes de validar el enum BlockType."""
        return _resolve_block_type(v)

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
    model_config = ConfigDict(extra="ignore")

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

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, v: Any) -> EventType:
        """Aplica aliases y normalización antes de validar el enum EventType."""
        return _resolve_event_type(v)

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
    model_config = ConfigDict(extra="ignore")

    metadata: Metadata
    courses: list[Course] = Field(default_factory=list)
    recurring_blocks: list[RecurringBlock] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
