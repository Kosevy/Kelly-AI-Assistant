"""
app/calendar/intent_router.py — Clasificador e interprete determinista de intenciones de calendario.

Rationale:
- Qwen2.5-7B (Q4_K_M) alucina horarios, transpone días y omite eventos al transcribir listas largas.
  Para consultas directas de calendario ("¿qué tengo la próxima semana?", "¿cuál es mi próximo examen?"),
  este router intercepta la solicitud antes del LLM, resolviendo fechas y tipos de forma determinista.
- Funciones puras: no depende de I/O externo. Recibe `now: datetime | None` para total testabilidad.
- Consultas complejas o de razonamiento ("Resume mi semana y dime qué priorizar") retornan `None`,
  permitiendo que el LLM las atienda normalmente con Function Calling híbrido.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any
from unidecode import unidecode

from app.config import settings

DAYS_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
DAYS_ES_DISPLAY = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
DAYS_MAP: dict[str, int] = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2,
    "miércoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5,
    "sábado": 5,
    "domingo": 6,
}

REASONING_KEYWORDS = [
    "resume", "resumen", "resumir", "priorizar", "prioriza", "consejo",
    "recomienda", "recomendacion", "recomendación", "como estudiar", "cómo estudiar",
    "preparar", "preparo", "explica", "explicame", "explícame", "opinas",
]


def _normalize_text(text: str) -> str:
    """Limpia puntuación y normaliza a minúsculas sin acentos para matching robusto."""
    cleaned = re.sub(r"[¿\?¡\!,\.;:\(\)\[\]\"']", " ", text)
    cleaned = " ".join(cleaned.lower().split())
    return unidecode(cleaned)


def _resolve_relative_dates(text_norm: str, now: datetime) -> tuple[date, date] | None:
    """
    Resuelve rangos temporales en español a fechas concretas [start_date, end_date].
    """
    today = now.date()

    # 1. "proxima semana" / "siguiente semana" (debe evaluarse ANTES de "esta semana")
    if "proxima semana" in text_norm or "siguiente semana" in text_norm:
        start_of_week = today - timedelta(days=now.weekday())
        next_monday = start_of_week + timedelta(days=7)
        next_sunday = next_monday + timedelta(days=6)
        return next_monday, next_sunday

    # 2. "esta semana" / "semana actual" / "la semana"
    if "esta semana" in text_norm or "semana actual" in text_norm:
        start_of_week = today - timedelta(days=now.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        return start_of_week, end_of_week

    # 3. "este mes" / "mes actual"
    if "este mes" in text_norm or "mes actual" in text_norm:
        start_of_month = today.replace(day=1)
        if now.month == 12:
            start_of_next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            start_of_next_month = today.replace(month=today.month + 1, day=1)
        end_of_month = start_of_next_month - timedelta(days=1)
        return start_of_month, end_of_month

    # 4. "hoy"
    if re.search(r"\bhoy\b", text_norm):
        return today, today

    # 5. "manana" / "pasado manana"
    if "pasado manana" in text_norm:
        target = today + timedelta(days=2)
        return target, target
    if re.search(r"\bmanana\b", text_norm):
        target = today + timedelta(days=1)
        return target, target

    # 6. "ayer"
    if re.search(r"\bayer\b", text_norm):
        target = today - timedelta(days=1)
        return target, target

    # 7. Día específico: "el lunes", "el martes", "lunes que viene", "este viernes", etc.
    for day_name, weekday_idx in DAYS_MAP.items():
        pattern = rf"\b(?:el|este|proximo|próximo)?\s*{day_name}\b"
        if re.search(pattern, text_norm):
            # Calcular próxima ocurrencia (o hoy si coincide el día y no dice próximo)
            offset = (weekday_idx - now.weekday()) % 7
            if offset == 0 and ("proximo" in text_norm or "que viene" in text_norm):
                offset = 7
            target = today + timedelta(days=offset)
            return target, target

    return None


def detect_calendar_intent(user_text: str, now: datetime | None = None) -> dict[str, Any] | None:
    """
    Analiza el texto del usuario y devuelve la intención de calendario detectada.
    Si no es una consulta directa o requiere razonamiento libre, retorna None (va al LLM).

    Retorna un diccionario con:
      - intent: "read_week" | "read_range" | "next_deadline" | "free_slots"
      - params: dict de argumentos para ejecutar el tool
      - tool_name: nombre del tool subyacente correspondiente
    """
    if not user_text or not user_text.strip():
        return None

    now = now or datetime.now()
    text_norm = _normalize_text(user_text)

    # ── Guard: Si el usuario pide análisis, resumen o consejo, dejar que el LLM razone ──
    for keyword in REASONING_KEYWORDS:
        if re.search(rf"\b{keyword}\b", text_norm):
            return None

    # Obtener patrones configurables
    patterns = settings.calendar_intent_patterns

    # ── 1. Detección de Next Deadline ───────────────────────────────────────
    # "¿Cuál es mi próximo examen?", "¿Qué entrega viene?", "¿Próxima PC?"
    deadline_patterns = patterns.get("next_deadline", [])
    is_deadline = any(p in text_norm for p in deadline_patterns)

    # También detectar disparadores directos de próximo/a + tipo
    if not is_deadline:
        if re.search(r"\b(?:proximo|proxima|siguiente)\s+(?:examen|parcial|final|entrega|tarea|proyecto|practica|pc|evaluacion|deadline)\b", text_norm):
            is_deadline = True
        elif re.search(r"\bque\s+(?:examen|entrega|tarea|practica|evaluacion)\s+viene\b", text_norm):
            is_deadline = True

    if is_deadline:
        # Determinar filtro de types según la palabra clave
        types: list[str] | None = None
        if any(w in text_norm for w in ["examen", "parcial", "final", "eb1", "eb2"]):
            types = ["exam"]
        elif any(w in text_norm for w in ["practica", "pc"]):
            types = ["practica", "exam"]
        elif any(w in text_norm for w in ["entrega", "entregable", "tarea", "proyecto", "tf", "tb", "avance"]):
            types = ["entregable"]

        params: dict[str, Any] = {}
        if types is not None:
            params["types"] = types

        return {
            "intent": "next_deadline",
            "tool_name": "calendar_next_deadline",
            "params": params,
        }

    # ── 2. Detección de Free Slots ──────────────────────────────────────────
    # "¿Tengo tiempo hoy?", "Huecos libres", "¿Cuándo tengo tiempo para estudiar?"
    free_patterns = patterns.get("free_slots", [])
    is_free = any(p in text_norm for p in free_patterns)

    if not is_free:
        if re.search(r"\b(?:tengo|hay)\s+(?:tiempo|espacio|hueco|ratos?)\s+libre\b", text_norm):
            is_free = True
        elif re.search(r"\btiempo\s+para\s+estudiar\b", text_norm):
            is_free = True

    if is_free:
        date_range = _resolve_relative_dates(text_norm, now)
        if date_range:
            start_d, end_d = date_range
        else:
            # Default: consultar disponibilidad para hoy
            start_d = now.date()
            end_d = start_d

        return {
            "intent": "free_slots",
            "tool_name": "calendar_free_slots",
            "params": {
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "min_block_minutes": 30,
            },
        }

    # ── 3. Detección de Read Range / Week ────────────────────────────────────
    # "¿Qué tengo esta semana?", "¿Qué tengo la próxima semana?", "¿Qué tengo el lunes?", "Horario de hoy"
    read_patterns = patterns.get("read_week", [])
    is_read = any(p in text_norm for p in read_patterns)

    if not is_read:
        if re.search(r"\b(?:que\s+tengo|mi\s+agenda|mi\s+horario|ver\s+horario|ver\s+agenda|horario|agenda)\b", text_norm):
            is_read = True
        elif re.search(r"\bque\s+hay\s+(?:hoy|manana|esta\s+semana|la\s+proxima\s+semana|el\s+[a-z]+)\b", text_norm):
            is_read = True

    if is_read:
        date_range = _resolve_relative_dates(text_norm, now)
        if date_range:
            start_d, end_d = date_range
        else:
            # Si preguntó "¿qué tengo?" o "mi horario" sin especificar día/semana, default a esta semana
            start_of_week = now.date() - timedelta(days=now.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            start_d, end_d = start_of_week, end_of_week

        # Determinar si es vista de semana completa o rango corto
        days_diff = (end_d - start_d).days
        intent_type = "read_week" if days_diff >= 5 else "read_range"

        return {
            "intent": intent_type,
            "tool_name": "calendar_read_range",
            "params": {
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
            },
        }

    return None
