"""
app/system_prompt.py
Pure functions to build the dynamic system prompt with pre-computed datetime context and interaction mode.

Rationale:
  - Qwen2.5-7B has limited temporal reasoning capabilities and struggles to reliably compute
    date offsets and calendar week boundaries. Pre-computing resolved dates (esta semana,
    próxima semana, este mes, días individuales) directly in the system prompt removes
    arithmetic burden from the LLM and prevents temporal hallucinations.
  - Windows OS locale often causes strftime("%A") / strftime("%B") to render English names.
    Using explicit Spanish arrays guarantees consistent language regardless of system locale.
  - Kelly operates in two interaction modes: voice and text.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
]


def build_datetime_context(now: datetime | None = None) -> str:
    """
    Construye el bloque de contexto temporal pre-computado.

    Rationale:
      Qwen2.5-7B no calcula rangos de fechas de forma confiable.
      Pre-computar los rangos elimina la aritmética del LLM y reduce
      alucinaciones en consultas temporales.
      Usa nombres en español explícitos para no depender del locale de Windows.
    """
    now = now or datetime.now()
    today = now.date()

    weekday_name = DAYS_ES[now.weekday()]
    month_name = MONTHS_ES[now.month - 1]
    time_str = now.strftime("%H:%M")

    # Días relativos inmediatos
    tomorrow = today + timedelta(days=1)
    tomorrow_name = DAYS_ES[tomorrow.weekday()]

    yesterday = today - timedelta(days=1)
    yesterday_name = DAYS_ES[yesterday.weekday()]

    # Esta semana: lunes a domingo de la semana actual
    start_of_week = today - timedelta(days=now.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    # Próxima semana: lunes a domingo de la siguiente semana
    next_monday = start_of_week + timedelta(days=7)
    next_sunday = next_monday + timedelta(days=6)

    # Este mes: primer día al último día del mes actual
    start_of_month = today.replace(day=1)
    if now.month == 12:
        start_of_next_month = today.replace(year=today.year + 1, month=1, day=1)
    else:
        start_of_next_month = today.replace(month=today.month + 1, day=1)
    end_of_month = start_of_next_month - timedelta(days=1)

    # Días de la semana específicos (próxima ocurrencia sin más contexto)
    day_resolutions = []
    for idx, day_name in enumerate(DAYS_ES):
        offset = (idx - now.weekday()) % 7 or 7
        target_date = today + timedelta(days=offset)
        day_resolutions.append(f'- "el {day_name}" = {target_date.isoformat()}')
    resolved_days_block = "\n".join(day_resolutions)

    return f"""Contexto temporal:
- Fecha actual: {weekday_name}, {today.day} de {month_name} de {today.year}
- Hora actual: {time_str}
- Hoy: {today.isoformat()} ({weekday_name})
- Mañana: {tomorrow.isoformat()} ({tomorrow_name})
- Ayer: {yesterday.isoformat()} ({yesterday_name})

Rangos resueltos:
- "esta semana" = {start_of_week.isoformat()} (lunes) a {end_of_week.isoformat()} (domingo)
- "próxima semana" = {next_monday.isoformat()} (lunes) a {next_sunday.isoformat()} (domingo)
- "este mes" = {start_of_month.isoformat()} a {end_of_month.isoformat()}
{resolved_days_block}"""


def format_input_mode_instruction(input_mode: str) -> str:
    """
    Return a clear format instruction based on interaction mode.

    Pure function: deterministic mapping from input mode string to prompt directive.
    """
    mode = (input_mode or "voice").lower().strip()
    if mode == "voice":
        return (
            "Voz (el usuario te habla por voz y tu respuesta será sintetizada por TTS; "
            "sé concisa, natural y evita markdown complejo o bloques largos de código)."
        )
    return (
        "Texto (el usuario te escribe por teclado; responde con formato completo de Markdown, "
        "tablas, listas o bloques de código según corresponda)."
    )


def build_system_prompt(
    template: str,
    current_datetime: str | None = None,
    input_mode: str = "voice",
    now: datetime | None = None,
) -> str:
    """
    Build dynamic system prompt by injecting datetime context and interaction input_mode.

    If current_datetime is not provided, it is dynamically computed using build_datetime_context(now).
    Replaces {current_datetime} or {datetime_context} with the resolved block.
    Injects or appends the input mode instruction.
    """
    if current_datetime is None:
        current_datetime = build_datetime_context(now)

    mode_text = format_input_mode_instruction(input_mode)
    prompt = template

    if "{datetime_context}" in prompt:
        prompt = prompt.replace("{datetime_context}", current_datetime)
    if "{current_datetime}" in prompt:
        prompt = prompt.replace("{current_datetime}", current_datetime)

    if "{input_mode}" in prompt:
        prompt = prompt.replace("{input_mode}", mode_text)
    else:
        prompt = f"{prompt.rstrip()}\n\nModo de interacción actual: {mode_text}\n"

    return prompt
