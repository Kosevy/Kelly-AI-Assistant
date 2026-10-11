"""
app/system_prompt.py
Pure functions to build the dynamic system prompt with datetime and interaction mode.

Rationale:
  Kelly operates in two interaction modes: voice and text.
  When spoken to, responses must be concise and phonetically natural for TTS synthesis.
  When written to, responses should leverage full Markdown, code blocks, lists, and tables.
  Extracting prompt construction into a pure function enables isolated unit testing
  without requiring a running server or LLM inference.
"""
from __future__ import annotations


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
    current_datetime: str,
    input_mode: str = "voice",
) -> str:
    """
    Build dynamic system prompt by injecting datetime and interaction input_mode.

    If {current_datetime} is present in the template, it is replaced.
    If {input_mode} is present in the template, it is replaced with the mode instruction.
    Otherwise, the active mode is appended as an explicit directive so the model
    adapts its output format accordingly.
    """
    mode_text = format_input_mode_instruction(input_mode)
    prompt = template.replace("{current_datetime}", current_datetime)

    if "{input_mode}" in prompt:
        prompt = prompt.replace("{input_mode}", mode_text)
    else:
        prompt = f"{prompt.rstrip()}\n\nModo de interacción actual: {mode_text}\n"

    return prompt
