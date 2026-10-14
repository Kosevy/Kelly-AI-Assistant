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
  - JSON {"type": "audio_format", "sample_rate": 24000, "channels": 1, "dtype": "int16"}
    Sent once on connect. Informs frontend of raw PCM stream format.
  - JSON {"type": "audio_chunk_start", "sentence_id": N}: Begin of sentence N audio stream.
  - Binary: Raw PCM int16 samples (no WAV header) for sentence N.
  - JSON {"type": "audio_chunk_end", "sentence_id": N, "chunk_count": M}: End of sentence N.
  - JSON {"type": "transcript", "text": "..."}: Text recognized by STT.
  - JSON {"type": "assistant_text", "text": "..."}: Complete LLM response text.
  - JSON {"type": "no_speech"}: VAD filter dropped silent audio.
  - JSON {"type": "error", "message": "..."}: Pipeline error notification.

TTS Architecture:
  - LLM token streaming and TTS synthesis run concurrently.
  - LLM enqueues sentences into an asyncio.Queue; a single _tts_worker() coroutine
    drains it strictly in FIFO order, guaranteeing ordered audio output.
  - Sentences > TTS_MAX_CHARS chars are split before synthesis to avoid XTTS truncation.
"""
from __future__ import annotations

# ─── 1. PyTorch patches BEFORE any TTS/torchaudio import ─────────────────────
import os
import sys
import ctypes

os.environ["COQUI_TOS_AGREED"] = "1"

import torch
import torchaudio

from patches.torch_compat import apply_all as _apply_patches
_apply_patches()  # noqa: E402

# ─── 2. Configuration & Logging ───────────────────────────────────────────────
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
    logger.debug(f"[startup] CUDA DLL path added: {_torch_lib}")

# ─── 4. Main imports ──────────────────────────────────────────────────────────
import asyncio
import io
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from faster_whisper import WhisperModel
from TTS.api import TTS
import pysbd

from app.text_preprocess import normalize_for_tts
from app.stt_prompt import build_initial_prompt
from app.system_prompt import build_system_prompt
from app.protocol import parse_history_payload, parse_text_message_payload, parse_attachment_ids_payload
from app.tools import TOOLS
from app.calendar import init_calendar_if_missing
from app.llm_client import stream_chat_with_tools, format_chat_messages
from app.file_extractors import extract_by_extension
from app.file_chunker import truncate_to_token_budget, estimate_tokens

# ─── 5. Model initialization ──────────────────────────────────────────────────
logger.info("[startup] Loading STT model (faster-whisper)...")
stt_model = WhisperModel(
    settings.stt_model,
    device=settings.stt_device,
    compute_type=settings.stt_compute_type,
)
logger.info(
    f"[startup] STT ready: {settings.stt_model} | "
    f"device={settings.stt_device} | compute={settings.stt_compute_type}"
)

logger.info("[startup] Loading TTS model (XTTS-v2)...")
_tts_device = settings.tts_device if torch.cuda.is_available() else "cpu"
if settings.tts_device == "cuda" and not torch.cuda.is_available():
    logger.warning(
        "[startup] CUDA unavailable. TTS will run on CPU (~8-10s latency per sentence). "
        "Verify torch is installed with CUDA support (+cu126)."
    )

tts_model = TTS(settings.tts_model)
try:
    tts_model = tts_model.to(_tts_device)
    logger.info(f"[startup] TTS ready: XTTS-v2 | device={_tts_device}")
except Exception as _cuda_err:
    logger.error(
        f"[startup] FAILED to move TTS to '{_tts_device}': {_cuda_err}\n"
        "  Falling back to CPU. TTS latency will be high (~8-10s per sentence)."
    )
    _tts_device = "cpu"
    tts_model = tts_model.to("cpu")
    logger.warning("[startup] TTS loaded on CPU as fallback.")

# ─── 6. Voice conditioning latent cache ───────────────────────────────────────
_voice_sample = str(settings.get_voice_sample_path())
if not os.path.exists(_voice_sample):
    logger.error(
        f"[startup] Voice sample not found at '{_voice_sample}'. "
        "Provide a 6-10s WAV file for zero-shot voice cloning. "
        "See README.md for instructions."
    )
    sys.exit(1)

logger.info(f"[startup] Loading voice profile: {_voice_sample}")
gpt_cond_latent, speaker_embedding = (
    tts_model.synthesizer.tts_model.get_conditioning_latents(
        audio_path=[_voice_sample]
    )
)
logger.info("[startup] Voice profile cached — ready.")

# ─── 7. System prompt ─────────────────────────────────────────────────────────
_prompt_path = settings.get_system_prompt_path()
if _prompt_path.exists():
    _system_prompt_template = _prompt_path.read_text(encoding="utf-8")
    logger.info(f"[startup] System prompt loaded from: {_prompt_path}")
else:
    logger.warning(f"[startup] {_prompt_path} not found. Using fallback prompt.")
    _system_prompt_template = (
        "Eres Kelly, una asistente personal concisa. "
        "Responde siempre en español neutro en 1-2 oraciones. "
        "No uses markdown ni asteriscos. "
        "Contexto temporal: {current_datetime}."
    )

# ─── 8. pysbd Sentence Segmenter ──────────────────────────────────────────────
_segmenter = pysbd.Segmenter(language="es", clean=False)

# ─── 9. Log streaming mode at startup ─────────────────────────────────────────
logger.info(
    f"[startup] TTS mode: inference_stream | "
    f"stream_chunk_size={settings.tts_stream_chunk_size} | "
    f"timeout={settings.tts_stream_timeout_s}s"
)

# ─── 10. FastAPI app with startup lifespan ────────────────────────────────────

from contextlib import asynccontextmanager

# In-memory storage for uploaded file attachments: {attachment_id: metadata_and_text}
UPLOADED_ATTACHMENTS: dict[str, dict[str, Any]] = {}


async def _cleanup_uploads_loop() -> None:
    """
    Tarea periódica en segundo plano que elimina archivos subidos y entradas en memoria
    que hayan superado settings.upload_ttl_minutes.
    """
    while True:
        try:
            await asyncio.sleep(600)  # Cada 10 minutos
            now = time.time()
            ttl_seconds = settings.upload_ttl_minutes * 60
            expired_ids = [
                att_id
                for att_id, meta in UPLOADED_ATTACHMENTS.items()
                if (now - meta.get("created_at", 0)) > ttl_seconds
            ]
            for att_id in expired_ids:
                meta = UPLOADED_ATTACHMENTS.pop(att_id, None)
                if meta and "path" in meta:
                    try:
                        p = Path(meta["path"])
                        if p.exists():
                            p.unlink()
                            logger.info(f"[upload_cleanup] Archivo expirado eliminado: {p.name}")
                    except Exception as err:
                        logger.warning(f"[upload_cleanup] Error borrando {meta.get('path')}: {err}")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"[upload_cleanup] Error en ciclo de limpieza: {exc}")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    FastAPI lifespan handler: initializes calendar from template if missing,
    runs _warmup_llm(), and launches attachment TTL cleanup.
    """
    init_calendar_if_missing()
    await _warmup_llm()
    cleanup_task = asyncio.create_task(_cleanup_uploads_loop(), name="upload_cleanup")
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


async def _warmup_llm() -> None:
    """
    Send a trivial prompt to llama-server to force KV-cache warm-up.
    Non-blocking: logs warning on failure, never raises.
    """
    logger.info("[startup] LLM warmup — triggering KV cache load via chat completions...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                settings.llm_chat_url,
                json={
                    "messages": [{"role": "user", "content": "hola"}],
                    "max_tokens": 3,
                    "temperature": 0.0,
                    "stream": False,
                },
            )
            resp.raise_for_status()
        logger.info("[startup] LLM warmup complete — model loaded in KV cache")
    except Exception as exc:
        logger.warning(
            f"[startup] LLM warmup failed (non-fatal — first query may be slow): {exc}"
        )


app = FastAPI(title="Kelly Voice Assistant", version="0.5.0", lifespan=_lifespan)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _prepare_history(
    history: list[dict[str, str]],
    system_prompt: str,
    user_text: str,
) -> list[dict[str, str]]:
    """
    Truncates history by turn count (max history_max_turns) and by estimated
    token count (max history_max_tokens_estimate, using len/4 heuristic).
    Drops oldest user+assistant pairs first.
    """
    max_tok = settings.history_max_tokens_estimate
    working_history = list(history)

    max_msgs = settings.history_max_turns * 2
    if len(working_history) > max_msgs:
        working_history = working_history[-max_msgs:]

    while working_history:
        estimated_tokens = len(system_prompt) // 4 + len(user_text) // 4
        for msg in working_history:
            estimated_tokens += len(msg.get("content", "")) // 4
        if estimated_tokens <= max_tok:
            break
        working_history = working_history[2:]

    if len(working_history) < len(history):
        logger.debug(
            f"[prompt] History truncated: {len(history)} -> {len(working_history)} messages "
            f"(limit ~{max_tok} estimated tokens)"
        )
    return working_history


def _build_attachments_text(attachment_ids: list[str]) -> str:
    """
    Construye el bloque de texto con los adjuntos para ser inyectado en el prompt.
    """
    if not attachment_ids:
        return ""

    blocks: list[str] = [
        "[Archivos adjuntos provistos por el usuario (su contenido ya está extraído abajo, responde directamente basándote en este texto sin llamar a read_file):]"
    ]
    for att_id in attachment_ids:
        entry = UPLOADED_ATTACHMENTS.get(att_id)
        if not entry:
            logger.warning(f"[attachments] Attachment ID no encontrado: {att_id}")
            continue

        filename = entry.get("filename", "archivo")
        truncated = entry.get("truncated", False)
        status_tag = " (truncado por límite de tokens)" if truncated else ""
        content = entry.get("text", "")

        blocks.append(
            f"--- Adjunto: {filename}{status_tag} ---\n"
            f"{content}\n"
            f"--- Fin adjunto ---"
        )

    if len(blocks) == 1:
        return ""
    return "\n\n".join(blocks)



def _build_prompt_with_history(
    user_text: str,
    history: list[dict[str, str]],
    input_mode: str = "voice",
) -> str:
    """
    Build ChatML prompt for Qwen2.5 with conversational history (legacy fallback).
    """
    fecha_hora = datetime.now().strftime("%A, %d de %B de %Y. La hora es %I:%M %p")
    system_prompt = build_system_prompt(
        _system_prompt_template,
        current_datetime=fecha_hora,
        input_mode=input_mode,
    )
    working_history = _prepare_history(history, system_prompt, user_text)

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
        f"[prompt] {len(working_history)} history msgs | "
        f"~{estimated_tokens} estimated tokens | {len(prompt)} chars"
    )
    return prompt



def _iter_inference_stream(text: str):
    """
    Synchronous generator wrapper around XTTS inference_stream.
    Yields numpy float32 arrays (audio chunks).
    Must be called from a worker thread via asyncio.to_thread.
    """
    normalized = normalize_for_tts(text)
    if not normalized.strip():
        return

    chunks = tts_model.synthesizer.tts_model.inference_stream(
        normalized,
        settings.stt_language,
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=settings.tts_stream_chunk_size,
        enable_text_splitting=False,
    )
    for wav_chunk in chunks:
        # wav_chunk is a 1D torch.Tensor of float32 samples
        yield wav_chunk.cpu().numpy()


def _synthesize_sentence_sync(text: str) -> bytes:
    """
    Synchronous fallback: full inference() returning complete WAV bytes.
    Used when inference_stream fails for graceful degradation.
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


def _transcribe_audio(
    audio_bytes: bytes,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Transcribe audio bytes with faster-whisper.

    Accepts conversation history to build a dynamic initial_prompt.
    All decoding parameters are pulled from settings (externalizable via .env).

    Why initial_prompt from history:
      Whisper's decoder biases toward vocabulary/topics in initial_prompt.
      Using the user's own recent words (not Kelly's responses — those are TTS,
      not audio the model has seen) anchors the decoder to the current topic
      without any hardcoding. Effect is strongest on the first 1-2 words of each
      utterance, which is exactly where the systematic errors were observed.

    Synchronous — call via asyncio.to_thread().
    """
    audio_stream = io.BytesIO(audio_bytes)
    vad_params = {"min_silence_duration_ms": settings.stt_vad_min_silence_ms}

    # Build dynamic initial_prompt from recent user history
    initial_prompt = build_initial_prompt(
        history or [],
        max_chars=settings.stt_initial_prompt_max_chars,
    )
    if initial_prompt:
        logger.debug(
            f"[stt] initial_prompt ({len(initial_prompt)} chars): "
            f"\"{initial_prompt[:100]}{'...' if len(initial_prompt) > 100 else ''}\""
        )
    else:
        logger.debug("[stt] initial_prompt: empty (no history yet)")

    segments, _info = stt_model.transcribe(
        audio_stream,
        language=settings.stt_language,
        initial_prompt=initial_prompt if initial_prompt else None,
        beam_size=settings.stt_beam_size,
        temperature=settings.stt_temperature_fallback,
        no_speech_threshold=settings.stt_no_speech_threshold,
        condition_on_previous_text=settings.stt_condition_on_previous_text,
        vad_filter=settings.stt_vad_filter,
        vad_parameters=vad_params if settings.stt_vad_filter else None,
    )
    return "".join(seg.text for seg in segments).strip()


def _maybe_save_debug_audio(audio_bytes: bytes) -> None:
    """
    Saves raw STT audio for debugging if debug_save_audio is enabled.
    Rotates files to keep at most debug_audio_max_files.
    Synchronous — call via asyncio.to_thread().
    """
    if not settings.debug_save_audio:
        return

    debug_dir = Path(settings.debug_audio_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = debug_dir / f"stt_{ts}.raw"
    out_path.write_bytes(audio_bytes)
    logger.debug(f"[debug] Audio saved: {out_path} ({len(audio_bytes)} bytes)")

    existing = sorted(debug_dir.glob("stt_*.raw"), key=lambda p: p.stat().st_mtime)
    while len(existing) > settings.debug_audio_max_files:
        oldest = existing.pop(0)
        oldest.unlink(missing_ok=True)
        logger.debug(f"[debug] Rotated (deleted): {oldest.name}")


def _float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio samples to raw int16 PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


# Maximum chars per sentence accepted by XTTS for language 'es' is 239.
# We split at 200 to leave headroom for normalization expansion.
_TTS_MAX_CHARS = 200
_TTS_MIN_SPLIT_CHARS = 30  # sub-sentence must be at least this long


def _split_long_sentence(text: str, max_chars: int = _TTS_MAX_CHARS) -> list[str]:
    """
    Split a sentence that exceeds max_chars into shorter sub-sentences.

    Splitting priority:
      1. Comma / semicolon / colon boundaries (keeping punctuation with left part).
      2. Conjunctions: ' y ', ' pero ', ' o ', ' sin embargo ', ' aunque '.
      3. Word-boundary split at midpoint (last resort).

    Each part must be at least _TTS_MIN_SPLIT_CHARS chars. Parts shorter
    than that are merged with the next one.

    Returns a list of strings; always at least one element.
    """
    if len(text) <= max_chars:
        return [text]

    import re
    # Try splitting on punctuation (, ; :) — keep delimiter with left part
    parts = re.split(r'(?<=[,;:])\s+', text)
    if len(parts) > 1:
        # Merge short leading fragments forward
        merged: list[str] = []
        acc = ""
        for p in parts:
            candidate = (acc + " " + p).strip() if acc else p
            if len(candidate) <= max_chars or not acc:
                acc = candidate
            else:
                if len(acc) >= _TTS_MIN_SPLIT_CHARS:
                    merged.append(acc)
                    acc = p
                else:
                    acc = candidate  # still too short, keep accumulating
        if acc:
            merged.append(acc)
        if all(len(m) <= max_chars for m in merged) and len(merged) > 1:
            logger.debug(
                f"[tts] Splitting long sentence ({len(text)} chars) into "
                f"{len(merged)} parts (punctuation split)"
            )
            return merged

    # Try splitting on conjunctions
    conjunction_pattern = r'(?i)\s+(y|pero|o|ni|sin embargo|aunque|porque|entonces)\s+'
    conj_parts = re.split(conjunction_pattern, text)
    # re.split with groups returns alternating content/delimiter
    # Rebuild keeping conjunctions with the right part
    if len(conj_parts) > 1:
        rebuilt: list[str] = []
        i = 0
        while i < len(conj_parts):
            piece = conj_parts[i]
            if i + 1 < len(conj_parts):
                # next element is the conjunction
                conj = conj_parts[i + 1]
                if i + 2 < len(conj_parts):
                    next_piece = conj_parts[i + 2]
                    rebuilt.append(piece.strip())
                    rebuilt.append((conj + " " + next_piece).strip())
                    i += 3
                    continue
            rebuilt.append(piece.strip())
            i += 1

        rebuilt = [r for r in rebuilt if r]
        # Merge parts shorter than min
        merged2: list[str] = []
        acc2 = ""
        for r in rebuilt:
            candidate2 = (acc2 + " " + r).strip() if acc2 else r
            if len(candidate2) <= max_chars:
                acc2 = candidate2
            else:
                if acc2 and len(acc2) >= _TTS_MIN_SPLIT_CHARS:
                    merged2.append(acc2)
                    acc2 = r
                else:
                    acc2 = candidate2
        if acc2:
            merged2.append(acc2)
        if all(len(m) <= max_chars for m in merged2) and len(merged2) > 1:
            logger.debug(
                f"[tts] Splitting long sentence ({len(text)} chars) into "
                f"{len(merged2)} parts (conjunction split)"
            )
            return merged2

    # Last resort: split at word boundary near midpoint
    words = text.split()
    mid = len(words) // 2
    left = " ".join(words[:mid])
    right = " ".join(words[mid:])
    logger.debug(
        f"[tts] Splitting long sentence ({len(text)} chars) into 2 parts (midpoint split)"
    )
    return [p for p in [left, right] if p]


# ─── Pipeline context ─────────────────────────────────────────────────────────

class _PipelineContext:
    """Per-WebSocket session state. Supports pipeline cancellation."""

    def __init__(self) -> None:
        self.cancel_event = asyncio.Event()
        self.active_task: asyncio.Task | None = None
        self.pending_history: list[dict[str, str]] = []
        self.pending_voice_enabled: bool = True
        self.pending_attachment_ids: list[str] = []

    def request_cancel(self) -> None:
        self.cancel_event.set()
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()

    def reset(self) -> None:
        self.cancel_event.clear()


# ─── TTS streaming coroutine ──────────────────────────────────────────────────

async def _stream_tts_sentence(
    websocket: WebSocket,
    sentence_id: int,
    text: str,
    cancel_event: asyncio.Event,
    tts_metrics: dict,
) -> None:
    """
    Stream a single sentence through XTTS inference_stream.

    Sends audio_chunk_start, then raw PCM int16 chunks, then audio_chunk_end.
    Degrades to synchronous inference() if inference_stream fails.
    Checks cancel_event between each chunk for fast barge-in response.
    """
    if cancel_event.is_set():
        return

    t_tts_start = time.perf_counter()
    logger.debug(f"[tts] Sentence #{sentence_id}: \"{text[:80]}{'...' if len(text)>80 else ''}\"")

    await websocket.send_json({"type": "audio_chunk_start", "sentence_id": sentence_id})

    chunk_count = 0
    t_first_chunk: float | None = None
    stream_failed = False

    try:
        loop = asyncio.get_event_loop()
        # maxsize=4: back-pressures the producer if consumer (WS) is slow
        chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        def _producer() -> None:
            """Thread: push numpy arrays (or Exception sentinel) into async queue."""
            try:
                for chunk in _iter_inference_stream(text):
                    if cancel_event.is_set():
                        break
                    fut = asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(chunk), loop
                    )
                    try:
                        fut.result(timeout=settings.tts_stream_timeout_s)
                    except Exception:
                        break  # queue.put timed out or cancelled — stop producing
            except Exception as exc:
                # Send exception as item so consumer can detect failure
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put(exc), loop
                ).result(timeout=5.0)
            finally:
                # Always send sentinel; ignore any error putting it
                try:
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(None), loop
                    ).result(timeout=5.0)
                except Exception:
                    pass

        producer_future = loop.run_in_executor(None, _producer)

        while True:
            if cancel_event.is_set():
                logger.info(f"[tts] #{sentence_id} cancelled mid-stream")
                break

            try:
                item = await asyncio.wait_for(
                    chunk_queue.get(),
                    timeout=settings.tts_stream_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error(f"[tts] #{sentence_id} timed out waiting for chunk")
                stream_failed = True
                break

            if item is None:          # sentinel — generator done
                break
            if isinstance(item, Exception):
                logger.error(f"[tts] inference_stream error on #{sentence_id}: {item}")
                stream_failed = True
                break

            # Happy path: numpy float32 chunk
            if t_first_chunk is None:
                t_first_chunk = time.perf_counter()
                first_ms = (t_first_chunk - t_tts_start) * 1000
                logger.debug(f"[tts] #{sentence_id} first chunk: {first_ms:.0f}ms")
                if tts_metrics.get("tts_first_chunk_ms") is None:
                    tts_metrics["tts_first_chunk_ms"] = round(first_ms, 1)

            if not cancel_event.is_set():
                await websocket.send_bytes(_float32_to_pcm16(item))
            chunk_count += 1

        # Wait for the producer thread to finish cleanly
        try:
            await asyncio.wait_for(asyncio.shield(producer_future), timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    except asyncio.CancelledError:
        logger.info(f"[tts] #{sentence_id} CancelledError — cleaning up")
        # Give producer thread a moment to notice cancel_event and exit
        try:
            await asyncio.wait_for(asyncio.shield(producer_future), timeout=2.0)
        except Exception:
            pass
        raise
    except Exception as exc:
        logger.error(f"[tts] Unexpected error on #{sentence_id}: {exc}", exc_info=True)
        stream_failed = True

    # Graceful degradation: synchronous fallback
    if stream_failed and not cancel_event.is_set():
        logger.warning(f"[tts] #{sentence_id} — falling back to synchronous inference()")
        try:
            wav_bytes = await asyncio.to_thread(_synthesize_sentence_sync, text)
            if wav_bytes and not cancel_event.is_set():
                await websocket.send_bytes(wav_bytes)
                chunk_count = 1
        except Exception as exc2:
            logger.error(f"[tts] Synchronous fallback also failed: {exc2}")

    t_tts_end = time.perf_counter()
    tts_total_ms = (t_tts_end - t_tts_start) * 1000
    tts_metrics["tts_total_streaming_ms"] = (
        tts_metrics.get("tts_total_streaming_ms", 0.0) + tts_total_ms
    )

    logger.info(
        f"[tts] #{sentence_id} done | {chunk_count} chunks | "
        f"total={tts_total_ms:.0f}ms | "
        f"first_chunk={round((t_first_chunk - t_tts_start)*1000):.0f}ms"
        if t_first_chunk else
        f"[tts] #{sentence_id} done | {chunk_count} chunks | total={tts_total_ms:.0f}ms"
    )

    if not cancel_event.is_set():
        await websocket.send_json({
            "type": "audio_chunk_end",
            "sentence_id": sentence_id,
            "chunk_count": chunk_count,
        })


# ─── TTS worker: serial queue consumer ───────────────────────────────────────

async def _tts_worker(
    websocket: WebSocket,
    tts_queue: asyncio.Queue,
    cancel_event: asyncio.Event,
    tts_metrics: dict,
) -> None:
    """
    Single coroutine that drains tts_queue in FIFO order.

    The LLM streaming loop enqueues (sentence_id, text) tuples.
    A sentinel value of None signals the worker to stop.
    This guarantees audio is sent in strict sentence order regardless
    of how long each inference_stream call takes.
    """
    while True:
        try:
            item = await asyncio.wait_for(
                tts_queue.get(),
                timeout=settings.tts_stream_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("[tts_worker] Timed out waiting for next sentence — exiting")
            break
        except asyncio.CancelledError:
            logger.info("[tts_worker] CancelledError — exiting cleanly")
            return

        if item is None:  # Sentinel: LLM is done, no more sentences
            tts_queue.task_done()
            break

        sentence_id, text = item

        if cancel_event.is_set():
            tts_queue.task_done()
            continue  # drain remaining items without processing

        try:
            await _stream_tts_sentence(
                websocket, sentence_id, text, cancel_event, tts_metrics
            )
        except asyncio.CancelledError:
            tts_queue.task_done()
            raise
        except Exception as exc:
            logger.error(f"[tts_worker] Unhandled error on #{sentence_id}: {exc}")
        finally:
            tts_queue.task_done()

    # Drain any remaining items (in case worker exits early due to timeout/cancel)
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
            tts_queue.task_done()
        except asyncio.QueueEmpty:
            break


# ─── Main pipeline ────────────────────────────────────────────────────────────

# ─── LLM and TTS execution core ───────────────────────────────────────────────

async def _run_llm_and_tts(
    websocket: WebSocket,
    user_text: str,
    history: list[dict[str, str]],
    ctx: _PipelineContext,
    input_mode: str = "voice",
    voice_enabled: bool = True,
    t_start: float | None = None,
    stt_latency_ms: float = 0.0,
    attachments_text: str = "",
) -> None:
    """
    Core pipeline: LLM streaming + optional TTS synthesis -> WebSocket.

    Supports:
      - input_mode="voice" or "text": passes mode to dynamic prompt builder.
      - attachments_text: appended to user message for file context injection.
      - Function Calling: transparent tool calling loop via stream_chat_with_tools.
      - voice_enabled=True: spawns _tts_worker to stream synthesized audio chunks concurrently.
      - voice_enabled=False: bypasses TTS completely, saving GPU/CPU compute and latency.
      - llm_chunk streaming: sends {"type": "llm_chunk", "text": token} for live UI updates.
      - assistant_text: sends complete finalized response text at stream completion.
    """
    if t_start is None:
        t_start = time.perf_counter()

    fecha_hora = datetime.now().strftime("%A, %d de %B de %Y. La hora es %I:%M %p")
    system_prompt = build_system_prompt(
        _system_prompt_template,
        current_datetime=fecha_hora,
        input_mode=input_mode,
    )
    working_history = _prepare_history(history, system_prompt, user_text)
    messages = format_chat_messages(
        user_text=user_text,
        history=working_history,
        system_prompt=system_prompt,
        attachments_text=attachments_text,
    )

    t_llm_start = time.perf_counter()
    buffer = ""
    sentence_id = 0
    t_first_tts_enqueue: float | None = None
    full_response = ""

    llm_metrics: dict[str, Any] = {"ttft_ms": None}
    tts_metrics: dict[str, Any] = {
        "tts_first_chunk_ms": None,
        "tts_total_streaming_ms": 0.0,
    }

    tts_queue: asyncio.Queue | None = None
    worker_task: asyncio.Task | None = None

    if voice_enabled:
        tts_queue = asyncio.Queue(maxsize=8)
        worker_task = asyncio.create_task(
            _tts_worker(websocket, tts_queue, ctx.cancel_event, tts_metrics),
            name="tts_worker",
        )

    def _enqueue_sentence(text: str) -> None:
        """Enqueue a sentence (or its split parts) for ordered TTS synthesis."""
        nonlocal sentence_id, t_first_tts_enqueue
        if not voice_enabled or tts_queue is None:
            return
        parts = _split_long_sentence(text, max_chars=_TTS_MAX_CHARS)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            tts_queue.put_nowait((sentence_id, part))
            if t_first_tts_enqueue is None:
                t_first_tts_enqueue = time.perf_counter()
            sentence_id += 1

    try:
        async for token in stream_chat_with_tools(
            messages=messages,
            tools=TOOLS,
            server_url=settings.llm_chat_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            top_p=settings.llm_top_p,
            repeat_penalty=settings.llm_repeat_penalty,
            cancel_event=ctx.cancel_event,
            metrics=llm_metrics,
        ):
            if ctx.cancel_event.is_set():
                logger.info(f"[pipeline] Cancelled during LLM streaming (mode={input_mode})")
                break

            buffer += token
            full_response += token

            # Send token chunk to client for real-time text streaming
            if not ctx.cancel_event.is_set():
                await websocket.send_json({"type": "llm_chunk", "text": token})

            if voice_enabled:
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
                            break
                        _enqueue_sentence(sent)

                    buffer = remainder

        # ── Final remainder for TTS ───────────────────────────────────────────
        if voice_enabled and buffer.strip() and len(buffer.strip()) >= settings.tts_min_final_chars:
            if not ctx.cancel_event.is_set():
                _enqueue_sentence(buffer.strip())
        elif voice_enabled and buffer.strip():
            logger.debug(
                f"[pipeline] Remainder discarded ({len(buffer.strip())} chars): "
                f"\"{buffer.strip()}\""
            )

    except (httpx.RequestError, RuntimeError) as e:
        logger.error(f"[pipeline] LLM connection/execution error: {e}")
        await websocket.send_json({"type": "error", "message": "LLM connection error."})
        if voice_enabled and tts_queue and worker_task:
            await tts_queue.put(None)
            await asyncio.gather(worker_task, return_exceptions=True)
        return
    except asyncio.CancelledError:
        logger.info("[pipeline] Pipeline task CancelledError — cleaning up TTS worker")
        if voice_enabled and tts_queue and worker_task:
            while not tts_queue.empty():
                try:
                    tts_queue.get_nowait()
                    tts_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
        raise

    # ── Signal worker there are no more sentences ──────────────────────────────
    if voice_enabled and tts_queue and worker_task:
        if ctx.cancel_event.is_set():
            while not tts_queue.empty():
                try:
                    tts_queue.get_nowait()
                    tts_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
        else:
            await tts_queue.put(None)
            try:
                await asyncio.wait_for(worker_task, timeout=120.0)
            except asyncio.TimeoutError:
                logger.error("[pipeline] TTS worker timed out after 120s — cancelling")
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
            except asyncio.CancelledError:
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)

    # ── Send full response to client for history ───────────────────────────────
    if full_response.strip() and not ctx.cancel_event.is_set():
        await websocket.send_json({
            "type": "assistant_text",
            "text": full_response.strip(),
        })

    # ── Final metrics ──────────────────────────────────────────────────────────
    t_end = time.perf_counter()
    ttft_ms = llm_metrics.get("ttft_ms")
    if ttft_ms is None:
        ttft_ms = (t_end - t_llm_start) * 1000.0

    metrics: dict[str, Any] = {
        "input_mode": input_mode,
        "voice_enabled": voice_enabled,
        "stt_latency_ms": round(stt_latency_ms, 1),
        "llm_ttft_ms": round(ttft_ms, 1),
        "llm_total_ms": round((t_end - t_llm_start) * 1000, 1),
        "tts_first_chunk_ms": tts_metrics.get("tts_first_chunk_ms"),
        "tts_total_streaming_ms": round(tts_metrics.get("tts_total_streaming_ms", 0.0), 1),
        "sentence_count": sentence_id,
        "history_turns": len(history) // 2,
    }
    if tts_metrics.get("tts_first_chunk_ms") is not None and t_first_tts_enqueue is not None:
        metrics["e2e_first_audio_ms"] = round(
            stt_latency_ms
            + (t_first_tts_enqueue - (t_start + stt_latency_ms / 1000.0)) * 1000
            + tts_metrics["tts_first_chunk_ms"],
            1,
        )
    logger.info(f"pipeline_metrics {json.dumps(metrics)}")


# ─── Pipelines: Audio & Text ─────────────────────────────────────────────────

async def _run_pipeline(
    websocket: WebSocket,
    audio_bytes: bytes,
    history: list[dict[str, str]],
    ctx: _PipelineContext,
    voice_enabled: bool = True,
    attachment_ids: list[str] | None = None,
) -> None:
    """
    Voice pipeline: STT -> LLM streaming -> (optional TTS) -> WebSocket.
    Maintains full backwards compatibility with existing audio callers and tests.
    """
    t_start = time.perf_counter()

    # ── Debug audio save ───────────────────────────────────────────────────────
    if settings.debug_save_audio:
        await asyncio.to_thread(_maybe_save_debug_audio, audio_bytes)

    # ── STT ───────────────────────────────────────────────────────────────────
    logger.debug(f"[pipeline] STT started ({len(audio_bytes)} bytes)")
    user_text = await asyncio.to_thread(_transcribe_audio, audio_bytes, history)
    t_stt_done = time.perf_counter()
    stt_latency_ms = (t_stt_done - t_start) * 1000

    if not user_text:
        logger.info(f"[pipeline] VAD: audio discarded (stt={stt_latency_ms:.0f}ms)")
        await websocket.send_json({"type": "no_speech"})
        return

    logger.info(f"[pipeline] STT: {stt_latency_ms:.0f}ms -> \"{user_text}\"")
    await websocket.send_json({"type": "transcript", "text": user_text})

    if ctx.cancel_event.is_set():
        logger.debug("[pipeline] Cancelled after STT")
        return

    attachments_text = _build_attachments_text(attachment_ids or [])
    await _run_llm_and_tts(
        websocket=websocket,
        user_text=user_text,
        history=history,
        ctx=ctx,
        input_mode="voice",
        voice_enabled=voice_enabled,
        t_start=t_start,
        stt_latency_ms=stt_latency_ms,
        attachments_text=attachments_text,
    )


async def _run_text_pipeline(
    websocket: WebSocket,
    user_text: str,
    history: list[dict[str, str]],
    ctx: _PipelineContext,
    voice_enabled: bool = False,
    attachment_ids: list[str] | None = None,
) -> None:
    """
    Text pipeline: LLM streaming (input_mode="text") -> (optional TTS) -> WebSocket.
    Bypasses STT completely, providing near-instantaneous TTFT and full Markdown responses.
    """
    t_start = time.perf_counter()
    if ctx.cancel_event.is_set():
        logger.debug("[pipeline] Cancelled before text pipeline execution")
        return

    attachments_text = _build_attachments_text(attachment_ids or [])
    await _run_llm_and_tts(
        websocket=websocket,
        user_text=user_text,
        history=history,
        ctx=ctx,
        input_mode="text",
        voice_enabled=voice_enabled,
        t_start=t_start,
        stt_latency_ms=0.0,
        attachments_text=attachments_text,
    )


# ─── HTTP Endpoints ───────────────────────────────────────────────────────────

from fastapi.staticfiles import StaticFiles

# Serve /static/* for PWA assets (icons, service worker, etc.)
# Mounted before route handlers so FastAPI resolves them first.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def get_index():
    return FileResponse("index.html")


@app.get("/manifest.json", include_in_schema=False)
async def get_manifest():
    """
    Serve the PWA web app manifest.

    Dedicated endpoint (not StaticFiles) to ensure Content-Type is
    application/manifest+json, which Chrome requires for installability.
    """
    return FileResponse(
        "static/manifest.json",
        media_type="application/manifest+json",
    )



@app.get("/health")
async def health_check():
    """
    Returns status of all components: STT, LLM, TTS.
    200 OK if all healthy, 503 if any component is degraded.
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
            llm_health_url = settings.llm_chat_url.replace("/v1/chat/completions", "/health")
            resp = await client.get(llm_health_url)
            if resp.status_code == 200:
                components["llm"] = {"status": "ok", "url": settings.llm_chat_url}
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
            "stream_chunk_size": settings.tts_stream_chunk_size,
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


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Recibe un archivo vía multipart. Lo guarda temporalmente en upload_dir.
    Retorna metadata + preview del texto extraído.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido.")

    contents = await file.read()
    size_bytes = len(contents)
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    if size_bytes > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo excede el tamaño máximo permitido de {settings.max_upload_size_mb} MB.",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    attachment_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix
    saved_filename = f"{attachment_id}{suffix}"
    saved_path = upload_dir / saved_filename

    try:
        saved_path.write_bytes(contents)
    except Exception as exc:
        logger.error(f"[upload] Error escribiendo archivo en disco: {exc}")
        raise HTTPException(status_code=500, detail="Error interno al almacenar el archivo.")

    try:
        raw_text = extract_by_extension(saved_path)
    except ValueError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"[upload] Error extrayendo texto de {file.filename}: {exc}")
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error extrayendo texto del archivo: {exc}")

    truncated_text, was_truncated = truncate_to_token_budget(raw_text, settings.max_file_tokens)
    estimated_toks = estimate_tokens(truncated_text)

    preview_text = truncated_text[:200].strip()
    if len(truncated_text) > 200:
        preview_text += "..."

    entry = {
        "attachment_id": attachment_id,
        "filename": file.filename,
        "path": str(saved_path),
        "size_bytes": size_bytes,
        "preview": preview_text,
        "truncated": was_truncated,
        "estimated_tokens": estimated_toks,
        "text": truncated_text,
        "created_at": time.time(),
    }
    UPLOADED_ATTACHMENTS[attachment_id] = entry

    logger.info(
        f"[upload] Archivo cargado: {file.filename} (id={attachment_id}, {size_bytes} bytes, "
        f"~{estimated_toks} tokens, truncado={was_truncated})"
    )

    return {
        "attachment_id": attachment_id,
        "filename": file.filename,
        "size_bytes": size_bytes,
        "preview": preview_text,
        "truncated": was_truncated,
        "estimated_tokens": estimated_toks,
    }


# ─── WebSocket Handler ────────────────────────────────────────────────────────

@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    ctx = _PipelineContext()
    logger.info(f"[ws] Client connected: {websocket.client}")

    # Announce PCM streaming format to client (sent once on connect)
    await websocket.send_json({
        "type": "audio_format",
        "sample_rate": settings.tts_sample_rate,
        "channels": 1,
        "dtype": "int16",
    })

    try:
        while True:
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                logger.info(f"[ws] Client disconnected cleanly: {websocket.client}")
                break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.info(f"[ws] Disconnect detected (RuntimeError): {websocket.client}")
                else:
                    logger.warning(f"[ws] Unexpected RuntimeError: {e}")
                break

            # ── Binary message: raw audio (WebM/Opus)
            if "bytes" in data and data["bytes"]:
                history = ctx.pending_history
                ctx.pending_history = []
                voice_enabled = ctx.pending_voice_enabled
                attachment_ids = ctx.pending_attachment_ids
                ctx.pending_attachment_ids = []

                ctx.request_cancel()
                await asyncio.sleep(0)
                ctx.reset()

                audio_bytes = data["bytes"]
                ctx.active_task = asyncio.create_task(
                    _run_pipeline(
                        websocket=websocket,
                        audio_bytes=audio_bytes,
                        history=history,
                        ctx=ctx,
                        voice_enabled=voice_enabled,
                        attachment_ids=attachment_ids,
                    )
                )

            # ── Text message: control JSON
            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    logger.warning(f"[ws] Invalid JSON: {data['text'][:100]}")
                    continue

                msg_type = msg.get("type")

                if msg_type == "cancel":
                    logger.info("[ws] Barge-in: cancelling active pipeline")
                    ctx.request_cancel()

                elif msg_type == "audio_with_history":
                    history_raw = msg.get("history", [])
                    ctx.pending_history = parse_history_payload(history_raw)
                    ctx.pending_voice_enabled = bool(msg.get("voice_enabled", True))
                    ctx.pending_attachment_ids = parse_attachment_ids_payload(msg.get("attachment_ids", []))
                    logger.debug(
                        f"[ws] History received: {len(ctx.pending_history)} messages | "
                        f"voice_enabled={ctx.pending_voice_enabled} | "
                        f"attachment_ids={len(ctx.pending_attachment_ids)}"
                    )

                elif msg_type == "text_message":
                    try:
                        clean_text, history, voice_enabled, attachment_ids = parse_text_message_payload(msg)
                    except ValueError as err:
                        logger.warning(f"[ws] Invalid text_message: {err}")
                        continue

                    ctx.request_cancel()
                    await asyncio.sleep(0)
                    ctx.reset()

                    logger.info(
                        f"[ws] text_message received: \"{clean_text[:60]}{'...' if len(clean_text) > 60 else ''}\" "
                        f"| history={len(history)} | voice_enabled={voice_enabled} | attachments={len(attachment_ids)}"
                    )

                    ctx.active_task = asyncio.create_task(
                        _run_text_pipeline(
                            websocket=websocket,
                            user_text=clean_text,
                            history=history,
                            ctx=ctx,
                            voice_enabled=voice_enabled,
                            attachment_ids=attachment_ids,
                        )
                    )

                else:
                    logger.debug(f"[ws] Ignored control message: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"[ws] WebSocketDisconnect (outer): {websocket.client}")
    except Exception as e:
        if "disconnect" in str(e).lower():
            logger.info(f"[ws] Unexpected disconnect: {websocket.client}")
        else:
            logger.error(f"[ws] Unexpected error: {e}", exc_info=True)
    finally:
        ctx.request_cancel()
        logger.info(f"[ws] Session closed: {websocket.client}")