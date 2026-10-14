"""
app/calendar/exceptions.py
Excepciones del sistema de calendario local.
"""
from __future__ import annotations


class CalendarError(Exception):
    """Excepción base para errores del subsistema de calendario."""
    pass


class CalendarValidationError(CalendarError, ValueError):
    """
    Excepción lanzada cuando el archivo o la estructura del calendario
    no cumple las reglas de validación de negocio o esquema.
    """
    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.message = message
        self.errors = errors if errors is not None else [message]
        if errors and (len(errors) > 1 or errors[0] != message):
            full_msg = f"{message}: {'; '.join(errors)}"
        else:
            full_msg = message
        super().__init__(full_msg)

    def __str__(self) -> str:
        if self.errors and (len(self.errors) > 1 or self.errors[0] != self.message):
            return f"{self.message}: {'; '.join(self.errors)}"
        return self.message
