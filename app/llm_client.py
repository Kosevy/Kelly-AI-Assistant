"""
app/llm_client.py — Cliente para /v1/chat/completions con streaming SSE y Function Calling nativo.

Rationale:
- Migra del endpoint legado raw /completion al estándar /v1/chat/completions de OpenAI,
  permitiendo el uso nativo de tools y esquemas de funciones JSON en Qwen2.5.
- Implementa el bucle de ejecución de herramientas de forma asíncrona (máximo 3 iteraciones),
  resolviendo dependencias antes de generar la respuesta final.
- Preserva la emisión de tokens en tiempo real (streaming SSE) para mantener la baja latencia
  (TTFT) en la interfaz web y la alimentación ordenada del hilo de síntesis de voz XTTS-v2.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator

import httpx

from app.config import settings
from app.logging_config import logger
from app.tools import TOOLS, execute_tool


def format_chat_messages(
    user_text: str,
    history: list[dict[str, str]],
    system_prompt: str,
    attachments_text: str = "",
) -> list[dict[str, Any]]:
    """
    Construye la lista de mensajes estructurados (messages) para /v1/chat/completions.

    Concatena los archivos adjuntos procesados al final del mensaje del usuario.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    final_user_content = user_text
    if attachments_text:
        final_user_content = f"{user_text}\n\n{attachments_text}"

    messages.append({"role": "user", "content": final_user_content})
    return messages


async def stream_chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = TOOLS,
    server_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    max_tool_iterations: int = 3,
    tool_timeout_s: float = 10.0,
    cancel_event: asyncio.Event | None = None,
    metrics: dict[str, Any] | None = None,
    on_tool_execute: Any | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generador asíncrono que transmite tokens generados por el LLM mediante SSE.

    Si el LLM solicita tool_calls, las ejecuta de forma transparente en un bucle
    (hasta max_tool_iterations) antes de transmitir la respuesta final al usuario.
    Registra el TTFT (Time-To-First-Token) en el diccionario metrics.
    """
    url = server_url or settings.llm_chat_url
    # Rationale: cuando el LLM tiene tools disponibles, usamos una temperature más baja
    # (settings.tool_calling_temperature) para minimizar alucinaciones en la selección
    # y argumentos de tools. Si no hay tools o se pasa explícitamente, usamos la general.
    temp = temperature if temperature is not None else (
        settings.tool_calling_temperature if tools else settings.llm_temperature
    )
    max_tok = max_tokens if max_tokens is not None else settings.llm_max_tokens
    p_val = top_p if top_p is not None else settings.llm_top_p
    rep_pen = repeat_penalty if repeat_penalty is not None else settings.llm_repeat_penalty

    working_messages = list(messages)
    iteration = 0
    t_start = time.perf_counter()

    while iteration < max_tool_iterations:
        iteration += 1
        if cancel_event and cancel_event.is_set():
            break

        payload: dict[str, Any] = {
            "messages": working_messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "top_p": p_val,
            "repeat_penalty": rep_pen,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        tool_calls_accum: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        assistant_content = ""

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    err_msg = err_bytes.decode(errors="replace")
                    logger.error(f"[llm_client] Error HTTP {response.status_code}: {err_msg}")
                    raise RuntimeError(f"Error de llama-server (HTTP {response.status_code}): {err_msg}")

                async for line in response.aiter_lines():
                    if cancel_event and cancel_event.is_set():
                        break

                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    delta = choice.get("delta", {})

                    # Acumulación de fragmentos de llamadas a herramientas (SSE tool_calls)
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_accum:
                                tool_calls_accum[idx] = {
                                    "id": tc.get("id", ""),
                                    "type": tc.get("type", "function"),
                                    "function": {"name": "", "arguments": ""},
                                }
                            if "id" in tc and tc["id"]:
                                tool_calls_accum[idx]["id"] = tc["id"]
                            if "type" in tc and tc["type"]:
                                tool_calls_accum[idx]["type"] = tc["type"]
                            fn = tc.get("function", {})
                            if "name" in fn and fn["name"]:
                                tool_calls_accum[idx]["function"]["name"] += fn["name"]
                            if "arguments" in fn and fn["arguments"]:
                                tool_calls_accum[idx]["function"]["arguments"] += fn["arguments"]

                    # Transmisión inmediata de tokens de respuesta al cliente
                    if "content" in delta and delta["content"]:
                        content_piece = delta["content"]
                        assistant_content += content_piece
                        if metrics is not None and metrics.get("ttft_ms") is None:
                            metrics["ttft_ms"] = (time.perf_counter() - t_start) * 1000.0
                        yield content_piece

        # Si el LLM solicitó herramientas, ejecutarlas y reintentar
        if tool_calls_accum and finish_reason == "tool_calls":
            tool_calls_list = list(tool_calls_accum.values())
            logger.info(f"[llm_client] Iteration {iteration}: LLM solicitó {len(tool_calls_list)} herramientas.")

            working_messages.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": tool_calls_list,
            })

            # Repartir el presupuesto disponible para evitar sobrepasar el context window
            per_tool_max_tokens = min(
                settings.max_file_tokens,
                max(1000, 5000 // max(1, len(tool_calls_list)))
            )
            executed_cache: dict[str, str] = {}

            mode = settings.tool_result_injection_mode
            logger.debug(f"[llm_client] Injecting tool result via mode={mode}")

            for tc in tool_calls_list:
                fn_name = tc.get("function", {}).get("name", "")
                raw_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {"path": raw_args}

                cache_key = f"{fn_name}:{raw_args}"
                if cache_key in executed_cache:
                    tool_result = executed_cache[cache_key]
                else:
                    try:
                        tool_result = await asyncio.wait_for(
                            execute_tool(fn_name, args, max_tokens=per_tool_max_tokens),
                            timeout=tool_timeout_s,
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"[llm_client] Timeout de {tool_timeout_s}s ejecutando '{fn_name}'")
                        tool_result = f"Error: Timeout de {tool_timeout_s}s al ejecutar '{fn_name}'."
                    except Exception as exc:
                        logger.error(f"[llm_client] Error ejecutando '{fn_name}': {exc}")
                        tool_result = f"Error ejecutando '{fn_name}': {exc}"
                    executed_cache[cache_key] = tool_result

                logger.debug(f"[tool_result] {fn_name}({args}) → {tool_result[:500]}...")

                if on_tool_execute is not None:
                    try:
                        hook_res = on_tool_execute(fn_name, args, tool_result)
                        if asyncio.iscoroutine(hook_res):
                            await hook_res
                    except Exception as exc:
                        logger.warning(f"[llm_client] on_tool_execute callback failed: {exc}")

                if mode == "user_message":
                    # 1. Mensaje de assistant vacío o con contexto mínimo (mantiene estructura de turnos)
                    working_messages.append({
                        "role": "assistant",
                        "content": "",
                    })
                    # 2. Mensaje de user con el resultado inyectado
                    if fn_name.startswith("calendar_"):
                        directive = (
                            "[RESULTADO DEL SISTEMA — usa esto solo para responder o resumir]\n"
                            f"El tool `{fn_name}` devolvió:\n\n"
                            f"{tool_result}\n\n"
                            "[FIN DEL RESULTADO]\n\n"
                            "NOTA: El calendario ya fue mostrado al usuario en pantalla como una tarjeta visual interactiva. "
                            "Tú NO necesitas transcribir listas completas de horarios ni eventos. "
                            "Usa los datos exclusivamente para tu resumen conversacional o para responder a la pregunta del usuario."
                        )
                    else:
                        directive = (
                            "[RESULTADO DEL SISTEMA — usa esto literalmente]\n"
                            f"El tool `{fn_name}` devolvió:\n\n"
                            f"{tool_result}\n\n"
                            "[FIN DEL RESULTADO]\n\n"
                            "Basándote EXCLUSIVAMENTE en el resultado anterior, responde la pregunta del usuario. "
                            "NO inventes datos, NO resumas, NO generalices. Si el resultado está vacío, dilo."
                        )
                    working_messages.append({
                        "role": "user",
                        "content": directive,
                    })
                else:
                    # Modo 'tool_role': estándar OpenAI
                    content = tool_result
                    if fn_name.startswith("calendar_"):
                        content = f"{tool_result}\n\n[El calendario ya fue mostrado al usuario en pantalla. Usa los datos solo para tu resumen conversacional.]"
                    working_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": content,
                    })

            if mode == "user_message":
                logger.debug(f"[llm_client] Message injected: {working_messages[-1]['content'][:300]}...")

            # Continuar siguiente iteración con la información de las herramientas inyectada
            continue

        else:
            # Respuesta final terminada (finish_reason == 'stop' o no hubo tool calls)
            break
