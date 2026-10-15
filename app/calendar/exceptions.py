"""
app/calendar/exceptions.py
Excepciones del sistema de calendario local.

Rationale:
- ValidationIssue estructura el contexto de cada error (sección, índice, nombre,
  campo afectado y mensaje), permitiendo que los errores sean accionables tanto
  en logs como en la respuesta de Kelly al usuario.
- CalendarValidationError acumula una lista de ValidationIssue y genera un
  formato humano numerado con ubicación exacta (e.g. recurring_blocks[3] "Transporte").
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """
    Descripción estructurada de un error de validación del calendario.

    Attributes:
        section:  Sección del YAML afectada (e.g. 'recurring_blocks', 'events', 'metadata').
        index:    Índice 0-based del ítem dentro de la sección (None para errores globales).
        name:     Nombre descriptivo del ítem afectado (e.g. nombre del bloque o evento).
        field:    Campo específico que falló (e.g. 'type', 'course_id'). Vacío si es global.
        message:  Descripción del error con el valor inválido y la corrección esperada.
    """
    section: str
    message: str
    index: int | None = None
    name: str = ""
    field: str = ""

    def location(self) -> str:
        """Retorna la cadena de ubicación legible (e.g. 'recurring_blocks[3] "Transporte"')."""
        parts = [self.section]
        if self.index is not None:
            parts[0] = f"{self.section}[{self.index}]"
        if self.name:
            parts.append(f'"{self.name}"')
        return " ".join(parts)

    def __str__(self) -> str:
        loc = self.location()
        field_prefix = f"{self.field}: " if self.field else ""
        return f"{loc}: {field_prefix}{self.message}"


class CalendarError(Exception):
    """Excepción base para errores del subsistema de calendario."""
    pass


class CalendarValidationError(CalendarError, ValueError):
    """
    Excepción lanzada cuando el archivo o la estructura del calendario
    no cumple las reglas de validación de negocio o esquema.

    Puede construirse con una lista de ValidationIssue (errores tipados con ubicación)
    o con strings planos para compatibilidad con el código existente.
    """

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
        issues: list[ValidationIssue] | None = None,
    ) -> None:
        self.message = message

        # Normalización: si recibe issues estructurados, los convierte a strings
        # y los almacena para acceso tipado.
        self.issues: list[ValidationIssue] = issues or []

        if issues:
            self.errors: list[str] = [str(iss) for iss in issues]
        elif errors:
            self.errors = list(errors)
        else:
            self.errors = [message]

        full_msg = self._format_full()
        super().__init__(full_msg)

    def _format_full(self) -> str:
        """Genera el mensaje completo numerado con ubicación para cada error."""
        if not self.errors or (len(self.errors) == 1 and self.errors[0] == self.message):
            return self.message
        numbered = "\n".join(f"  {i + 1}. {e}" for i, e in enumerate(self.errors))
        return f"{self.message}:\n{numbered}"

    def __str__(self) -> str:
        return self._format_full()

    def to_list(self) -> list[str]:
        """Retorna los errores como lista de strings (para serialización JSON)."""
        return list(self.errors)
