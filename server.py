"""
server.py — Core entry point for the Kelly Voice Assistant.

Initialization sequence (order-critical):
1. PyTorch compatibility monkey-patches (BEFORE importing TTS/torchaudio).
2. Configuration and Loguru logging.
3. FastAPI application + STT/TTS model loaders.
4. WebSocket and HTTP route handlers.

WebSocket Protocol (Client -> Server):
  - Binary: Raw audio recording (webm/opus). Triggers STT -> LLM -> TTS pipeline.
  - JSON {"type": "audio_with_history", "history": [...]}:
    Precedes the binary audio chunk to pass client-managed conversational history.
  - JSON {"type": "cancel"}: Immediate barge-in cancellation of the active pipeline.

WebSocket Protocol (Server -> Client):
  - Binary: Synthesized WAV PCM_16 chunks at 24 kHz.
  - JSON {"type": "transcript", "text": "..."}: Text recognized by STT.
  - JSON {"type": "assistant_text", "text": "..."}: Complete LLM response text.
  - JSON {"type": "no_speech"}: VAD filter dropped silent audio.
  - JSON {"type": "error", "message": "..."}: Pipeline error notification.
"""
from __future__ import annotations

# ─── 1. Compatibilidad PyTorch ANTES de cualquier import de torch/TTS ────────
import os
import sys
import ctypes

os.environ["COQUI_TOS_AGREED"] = "1"

import torch
import torchaudio

from patches.torch_compat import apply_all as _apply_patches
_apply_patches()  # noqa: E402  — debe correr antes de TTS

# ─── 2. Configuración y Logging ───────────────────────────────────────────────
from app.config import settings
from app.logging_config import logger, setup_logging

setup_logging(log_level=settings.log_level, logs_dir=settings.logs_dir)

# ─── 3. CUDA DLL path (Windows) ───────────────────────────────────────────────
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(_torch_lib):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ["PATH"]
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetEnvironmentVariableW("PATH", os.environ["PATH"])
    logger.debug(f"[startup] CUDA DLL path añadido: {_torch_lib}")

# ─── 4. Imports principales ───────────────────────────────────────────────────
import asyncio
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from faster_whisper import WhisperModel
from TTS.api import TTS
import pysbd

from app.text_preprocess import normalize_for_tts

# ─── 5. Inicialización de modelos ─────────────────────────────────────────────
logger.info("[startup] Cargando modelo STT (faster-whisper)...")
stt_model = WhisperModel(
    settings.stt_model,
    device=settings.stt_device,
    compute_type=settings.stt_compute_type,
)
logger.info(
    f"[startup] STT listo: {settings.stt_model} | "
    f"device={settings.stt_device} | compute={settings.stt_compute_type}"
)

logger.info("[startup] Cargando modelo TTS (XTTS-v2)...")
_tts_device = settings.tts_device if torch.cuda.is_available() else "cpu"
if settings.tts_device == "cuda" and not torch.cuda.is_available():
    logger.warning(
        "[startup] CUDA no disponible (torch.cuda.is_available() = False). "
        "TTS usará CPU — latencia TTS será ~8-10s por oración. "
        "Verifica que torch está instalado con soporte CUDA (+cu126)."
    )

tts_model = TTS(settings.tts_model)
try:
    tts_model = tts_model.to(_tts_device)
    logger.info(f"[startup] TTS listo: XTTS-v2 | device={_tts_device}")
except Exception as _cuda_err:
    logger.error(
        f"[startup] FALLO al mover TTS a '{_tts_device}': {_cuda_err}\n"
        "  Cayendo a CPU. La latencia TTS será alta (~8-10s por oración)."
    )
    _tts_device = "cpu"
    tts_model = tts_model.to("cpu")
    logger.warning("[startup] TTS cargado en CPU como fallback.")

# ─── 6. Caché de latentes de voz ─────────────────────────────────────────────
_voice_sample = str(settings.get_voice_sample_path())
if not os.path.exists(_voice_sample):
    logger.error(
        f"[startup] No se encontró el archivo de muestra de voz en '{_voice_sample}'. "
        "Por favor proporciona una muestra de voz WAV de 6-10s para la clonación zero-shot (ej. voice_sample.wav). "
        "Consulta el README.md para más instrucciones."
    )
    sys.exit(1)

logger.info(f"[startup] Cargando perfil de voz: {_voice_sample}")

gpt_cond_latent, speaker_embedding = (
    tts_model.synthesizer.tts_model.get_conditioning_latents(
        audio_path=[_voice_sample]
    )
)
logger.info("[startup] Perfil de voz en caché — listo.")

# ─── 7. System prompt ─────────────────────────────────────────────────────────
_prompt_path = settings.get_system_prompt_path()
if _prompt_path.exists():
    _system_prompt_template = _prompt_path.read_text(encoding="utf-8")
    logger.info(f"[startup] System prompt cargado desde: {_prompt_path}")
else:
    logger.warning(
        f"[startup] No se encontró {_prompt_path}. Usando prompt de fallback."
    )
    _system_prompt_template = (
        "Eres Kelly, una asistente personal concisa. "
        "Responde siempre en español neutro en 1-2 oraciones. "
        "No uses markdown ni asteriscos. "
        "Contexto temporal: {current_datetime}."
    )

# ─── 8. pysbd Segmenter ───────────────────────────────────────────────────────
_segmenter = pysbd.Segmenter(language="es", clean=False)

# ─── 9. FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="Kelly Voice Assistant", version="0.3.0")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_prompt_with_history(
    user_text: str,
    history: list[dict[str, str]],
) -> str:
    """
    Construye el prompt ChatML para Qwen2.5 con historial de conversación.

    Formato:
        <|im_start|>system\\n{system}\\n<|im_end|>
        <|im_start|>user\\n{hist_user_1}<|im_end|>
        <|im_start|>assistant\\n{hist_asst_1}<|im_end|>
        ...
        <|im_start|>user\\n{user_text}<|im_end|>
        <|im_start|>assistant\\n

    Si el historial supera history_max_tokens_estimate, descarta los
    turnos más antiguos (en pares user/assistant) hasta caber.
    """
    fecha_hora = datetime.now().strftime("%A, %d de %B de %Y. La hora es %I:%M %p")
    system_prompt = _system_prompt_template.replace("{current_datetime}", fecha_hora)

    # Truncar historial si supera el límite de tokens estimado
    # Estimación burda pero rápida: len(texto) / 4
    max_tok = settings.history_max_tokens_estimate
    working_history = list(history)  # copia local

    # Truncar exceso de turnos (máx N turnos = 2N mensajes)
    max_msgs = settings.history_max_turns * 2
    if len(working_history) > max_msgs:
        working_history = working_history[-max_msgs:]

    # Truncar por tokens estimados
    while working_history:
        estimated_tokens = len(system_prompt) // 4 + len(user_text) // 4
        for msg in working_history:
            estimated_tokens += len(msg.get("content", "")) // 4
        if estimated_tokens <= max_tok:
            break
        # Descarta el par más antiguo (user + assistant = 2 mensajes)
        working_history = working_history[2:]

    if len(working_history) < len(history):
        logger.debug(
            f"[prompt] Historial truncado: {len(history)} → {len(working_history)} mensajes "
            f"(límite ~{max_tok} tokens estimados)"
        )

    # Construir ChatML
    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
    for msg in working_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")

    prompt = "".join(parts)
    estimated_tokens = len(prompt) // 4
    logger.debug(
        f"[prompt] {len(working_history)} msgs historial | "
        f"~{estimated_tokens} tokens estimados | {len(prompt)} chars"
    )
    return prompt


def _synthesize_sentence(text: str) -> bytes:
    """
    Sintetiza una oración con XTTS-v2 usando latentes en caché.
    Devuelve bytes WAV PCM_16 a 24000 Hz.
    Síncrono — llamar con asyncio.to_thread().
    """
    normalized = normalize_for_tts(text)
    if not normalized.strip():
        return b""

    out = tts_model.synthesizer.tts_model.inference(
        normalized,
        settings.stt_language,
        gpt_cond_latent,
        speaker_embedding,
    )
    wav = out["wav"]
    byte_io = io.BytesIO()
    sf.write(byte_io, np.array(wav), settings.tts_sample_rate, format="WAV", subtype="PCM_16")
    return byte_io.getvalue()


def _transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio con faster-whisper + VAD.
    Síncrono — llamar con asyncio.to_thread().
    """
    audio_stream = io.BytesIO(audio_bytes)
    vad_params = {"min_silence_duration_ms": settings.stt_vad_min_silence_ms}

    segments, info = stt_model.transcribe(
        audio_stream,
        language=settings.stt_language,
        vad_filter=settings.stt_vad_filter,
        vad_parameters=vad_params if settings.stt_vad_filter else None,
    )
    return "".join(seg.text for seg in segments).strip()


def _maybe_save_debug_audio(audio_bytes: bytes) -> None:
    """
    Si debug_save_audio está activo, guarda el audio en debug/stt_{ts}.wav.
    Rota automáticamente manteniendo solo los últimos debug_audio_max_files.
    Síncrono — llamar con asyncio.to_thread().
    """
    if not settings.debug_save_audio:
        return

    debug_dir = Path(settings.debug_audio_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = debug_dir / f"stt_{ts}.raw"
    out_path.write_bytes(audio_bytes)
    logger.debug(f"[debug] Audio guardado: {out_path} ({len(audio_bytes)} bytes)")

    # Rotación: eliminar archivos más antiguos si hay más del límite
    existing = sorted(debug_dir.glob("stt_*.raw"), key=lambda p: p.stat().st_mtime)
    while len(existing) > settings.debug_audio_max_files:
        oldest = existing.pop(0)
        oldest.unlink(missing_ok=True)
        logger.debug(f"[debug] Audio rotado (borrado): {oldest.name}")


# ─── Pipeline principal (async con cancelación) ───────────────────────────────

class _PipelineContext:
    """Contexto por sesión WebSocket. Permite cancelar el pipeline activo."""

    def __init__(self) -> None:
        self.cancel_event = asyncio.Event()
        self.active_task: asyncio.Task | None = None
        # Buffer para historial pendiente: el cliente puede enviarlo antes del audio
        self.pending_history: list[dict[str, str]] = []

    def request_cancel(self) -> None:
        self.cancel_event.set()
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()

    def reset(self) -> None:
        self.cancel_event.clear()


async def _run_pipeline(
    websocket: WebSocket,
    audio_bytes: bytes,
    history: list[dict[str, str]],
    ctx: _PipelineContext,
) -> None:
    """
    Pipeline completo: STT → LLM (con historial) → TTS por oración → WebSocket.
    Incluye instrumentación de tiempos, cancelación y debug de audio.
    """
    t_start = time.perf_counter()

    # ── Debug: guardar audio crudo ────────────────────────────────────────────
    if settings.debug_save_audio:
        await asyncio.to_thread(_maybe_save_debug_audio, audio_bytes)

    # ── STT ──────────────────────────────────────────────────────────────────
    logger.debug(f"[pipeline] STT iniciado ({len(audio_bytes)} bytes)")
    user_text = await asyncio.to_thread(_transcribe_audio, audio_bytes)
    t_stt_done = time.perf_counter()
    stt_latency_ms = (t_stt_done - t_start) * 1000

    if not user_text:
        logger.info(f"[pipeline] Audio descartado por VAD (stt={stt_latency_ms:.0f}ms)")
        await websocket.send_json({"type": "no_speech"})
        return

    logger.info(f"[pipeline] STT: {stt_latency_ms:.0f}ms → \"{user_text}\"")

    # Enviar transcript al cliente para que lo añada al historial de UI
    await websocket.send_json({"type": "transcript", "text": user_text})

    if ctx.cancel_event.is_set():
        logger.debug("[pipeline] Cancelado tras STT")
        return

    # ── LLM streaming con historial ───────────────────────────────────────────
    formatted_prompt = _build_prompt_with_history(user_text, history)
    payload = {
        "prompt": formatted_prompt,
        "temperature": settings.llm_temperature,
        "n_predict": settings.llm_max_tokens,
        "top_p": settings.llm_top_p,
        "repeat_penalty": settings.llm_repeat_penalty,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "stream": True,
    }

    t_llm_start = time.perf_counter()
    t_first_token: float | None = None
    buffer = ""
    sentence_count = 0
    tts_first_sent = False
    t_first_audio: float | None = None
    full_response = ""  # Acumula la respuesta completa para el historial

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", settings.llm_server_url, json=payload) as response:
                async for line in response.aiter_lines():
                    if ctx.cancel_event.is_set():
                        logger.info("[pipeline] Cancelado durante LLM streaming")
                        return

                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("content", "")
                    if not token:
                        continue

                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                        llm_ttft_ms = (t_first_token - t_llm_start) * 1000
                        logger.debug(f"[pipeline] LLM TTFT: {llm_ttft_ms:.0f}ms")

                    buffer += token
                    full_response += token

                    # ── Segmentación con pysbd + buffer mínimo ────────────
                    sentences = _segmenter.segment(buffer)

                    if len(sentences) > 1:
                        complete = sentences[:-1]
                        remainder = sentences[-1]

                        for sent in complete:
                            sent = sent.strip()
                            if len(sent) < settings.tts_min_chunk_chars:
                                remainder = sent + " " + remainder
                                continue

                            if ctx.cancel_event.is_set():
                                return

                            await _send_tts_chunk(
                                websocket, sent, ctx,
                                sentence_count, tts_first_sent, t_start
                            )
                            sentence_count += 1
                            if not tts_first_sent:
                                tts_first_sent = True
                                t_first_audio = time.perf_counter()

                        buffer = remainder

        # ── Remanente final ───────────────────────────────────────────────
        if buffer.strip() and len(buffer.strip()) >= settings.tts_min_final_chars:
            if not ctx.cancel_event.is_set():
                await _send_tts_chunk(
                    websocket, buffer.strip(), ctx,
                    sentence_count, tts_first_sent, t_start
                )
                sentence_count += 1
                if not tts_first_sent and t_first_audio is None:
                    t_first_audio = time.perf_counter()
        elif buffer.strip():
            logger.debug(
                f"[pipeline] Remanente descartado ({len(buffer.strip())} chars): "
                f"\"{buffer.strip()}\""
            )

    except httpx.RequestError as e:
        logger.error(f"[pipeline] Error de conexión con LLM: {e}")
        await websocket.send_json({"type": "error", "message": "Error de conexión con el modelo."})
        return
    except asyncio.CancelledError:
        logger.info("[pipeline] Task cancelada externamente")
        return

    # ── Enviar respuesta completa al cliente para el historial ────────────────
    if full_response.strip() and not ctx.cancel_event.is_set():
        await websocket.send_json({
            "type": "assistant_text",
            "text": full_response.strip(),
        })

    # ── Métricas finales ──────────────────────────────────────────────────────
    t_end = time.perf_counter()
    t_first_token = t_first_token or t_llm_start
    metrics = {
        "stt_latency_ms": round(stt_latency_ms, 1),
        "llm_ttft_ms": round((t_first_token - t_llm_start) * 1000, 1),
        "llm_total_ms": round((t_end - t_llm_start) * 1000, 1),
        "tts_first_sentence_ms": round((t_first_audio - t_start) * 1000, 1) if t_first_audio else None,
        "e2e_ms": round((t_first_audio - t_start) * 1000, 1) if t_first_audio else None,
        "sentence_count": sentence_count,
        "history_turns": len(history) // 2,
    }
    logger.info(f"pipeline_metrics {json.dumps(metrics)}")


async def _send_tts_chunk(
    websocket: WebSocket,
    text: str,
    ctx: _PipelineContext,
    sentence_idx: int,
    tts_first_sent: bool,
    t_pipeline_start: float,
) -> None:
    """Sintetiza una oración y la envía por WebSocket. Mide latencia TTS."""
    t_tts_start = time.perf_counter()
    logger.debug(f"[tts] Oración #{sentence_idx}: \"{text}\"")

    wav_bytes = await asyncio.to_thread(_synthesize_sentence, text)

    t_tts_done = time.perf_counter()
    tts_ms = (t_tts_done - t_tts_start) * 1000
    logger.info(f"[tts] #{sentence_idx} | {tts_ms:.0f}ms | {len(wav_bytes)} bytes")

    if ctx.cancel_event.is_set():
        return

    if wav_bytes:
        await websocket.send_bytes(wav_bytes)


# ─── HTTP Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def get_index():
    return FileResponse("index.html")


@app.get("/health")
async def health_check():
    """
    Verifica el estado de todos los componentes.
    200 OK si todo funciona, 503 si alguno falla.
    """
    components: dict[str, Any] = {}
    all_ok = True

    try:
        components["stt"] = {
            "status": "ok",
            "model": settings.stt_model,
            "device": settings.stt_device,
        }
    except Exception as e:
        components["stt"] = {"status": "error", "detail": str(e)}
        all_ok = False

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            llm_health_url = settings.llm_server_url.replace("/completion", "/health")
            resp = await client.get(llm_health_url)
            if resp.status_code == 200:
                components["llm"] = {"status": "ok", "url": settings.llm_server_url}
            else:
                components["llm"] = {"status": "degraded", "http_status": resp.status_code}
                all_ok = False
    except httpx.RequestError as e:
        components["llm"] = {"status": "error", "detail": str(e)}
        all_ok = False

    try:
        tts_loaded = tts_model is not None and gpt_cond_latent is not None
        components["tts"] = {
            "status": "ok" if tts_loaded else "error",
            "device": _tts_device,
            "voice_cached": tts_loaded,
        }
        if not tts_loaded:
            all_ok = False
    except Exception as e:
        components["tts"] = {"status": "error", "detail": str(e)}
        all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(
        content={"status": "ok" if all_ok else "degraded", "components": components},
        status_code=status_code,
    )


# ─── WebSocket Handler ────────────────────────────────────────────────────────

@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    ctx = _PipelineContext()
    logger.info(f"[ws] Cliente conectado: {websocket.client}")

    try:
        while True:
            # ── Recibir dato — C1: capturar RuntimeError además de WebSocketDisconnect
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                logger.info(f"[ws] Cliente desconectado limpiamente: {websocket.client}")
                break
            except RuntimeError as e:
                # Pasa cuando el cliente cierra y se intenta receive() de nuevo
                if "disconnect" in str(e).lower():
                    logger.info(f"[ws] Desconexión detectada (RuntimeError): {websocket.client}")
                else:
                    logger.warning(f"[ws] RuntimeError inesperado: {e}")
                break

            # ── Mensaje binario: audio crudo (protocolo original sin historial)
            if "bytes" in data and data["bytes"]:
                history = ctx.pending_history
                ctx.pending_history = []  # consumir historial pendiente

                ctx.request_cancel()
                await asyncio.sleep(0)
                ctx.reset()

                audio_bytes = data["bytes"]
                ctx.active_task = asyncio.create_task(
                    _run_pipeline(websocket, audio_bytes, history, ctx)
                )

            # ── Mensaje de texto: control JSON
            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    logger.warning(f"[ws] JSON no válido: {data['text'][:100]}")
                    continue

                msg_type = msg.get("type")

                if msg_type == "cancel":
                    # Barge-in: cancelar pipeline activo
                    logger.info("[ws] Barge-in: cancelando pipeline activo")
                    ctx.request_cancel()

                elif msg_type == "audio_with_history":
                    # El cliente envía historial ANTES del audio binario
                    # Almacenar hasta recibir el siguiente mensaje binario
                    history_raw = msg.get("history", [])
                    # Validar estructura mínima
                    ctx.pending_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in history_raw
                        if isinstance(m, dict)
                        and m.get("role") in ("user", "assistant")
                        and isinstance(m.get("content"), str)
                    ]
                    logger.debug(
                        f"[ws] Historial recibido: {len(ctx.pending_history)} mensajes"
                    )

                else:
                    logger.debug(f"[ws] Mensaje de control ignorado: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"[ws] WebSocketDisconnect (outer): {websocket.client}")
    except Exception as e:
        # Cualquier otro error: loguear con nivel warning, no error
        if "disconnect" in str(e).lower():
            logger.info(f"[ws] Desconexión inesperada: {websocket.client}")
        else:
            logger.error(f"[ws] Error inesperado: {e}", exc_info=True)
    finally:
        ctx.request_cancel()
        logger.info(f"[ws] Sesión cerrada: {websocket.client}")