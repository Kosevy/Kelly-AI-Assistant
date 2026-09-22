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

# ─── 10. FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(title="Kelly Voice Assistant", version="0.4.0")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_prompt_with_history(
    user_text: str,
    history: list[dict[str, str]],
) -> str:
    """
    Build ChatML prompt for Qwen2.5 with conversational history.

    Truncates history by turn count (max history_max_turns) and by estimated
    token count (max history_max_tokens_estimate, using len/4 heuristic).
    Drops oldest user+assistant pairs first.
    """
    fecha_hora = datetime.now().strftime("%A, %d de %B de %Y. La hora es %I:%M %p")
    system_prompt = _system_prompt_template.replace("{current_datetime}", fecha_hora)

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


def _transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio with faster-whisper + VAD.
    Synchronous — call via asyncio.to_thread().
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


# ─── Pipeline context ─────────────────────────────────────────────────────────

class _PipelineContext:
    """Per-WebSocket session state. Supports pipeline cancellation."""

    def __init__(self) -> None:
        self.cancel_event = asyncio.Event()
        self.active_task: asyncio.Task | None = None
        self.pending_history: list[dict[str, str]] = []

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
    ctx: _PipelineContext,
    tts_metrics: dict,
) -> None:
    """
    Stream a single sentence through XTTS inference_stream.

    Protocol:
      1. Send JSON {"type": "audio_chunk_start", "sentence_id": N}
      2. For each PCM chunk: send raw int16 bytes
      3. Send JSON {"type": "audio_chunk_end", "sentence_id": N, "chunk_count": M}

    Degrades to synchronous inference() if inference_stream raises.
    Updates tts_metrics dict in-place for caller to collect timing.
    """
    if ctx.cancel_event.is_set():
        return

    t_tts_start = time.perf_counter()
    logger.debug(f"[tts] Sentence #{sentence_id}: \"{text[:80]}{'...' if len(text)>80 else ''}\"")

    await websocket.send_json({"type": "audio_chunk_start", "sentence_id": sentence_id})

    chunk_count = 0
    t_first_chunk: float | None = None
    stream_failed = False

    try:
        # Run synchronous generator in a thread pool.
        # We use a queue to bridge the thread-based generator and async WS sends.
        loop = asyncio.get_event_loop()
        chunk_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=4)

        def _producer():
            """Runs in thread: pushes numpy chunks into the asyncio queue."""
            try:
                for chunk in _iter_inference_stream(text):
                    # put_nowait would block if queue is full, use put via run_coroutine
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(chunk), loop
                    ).result(timeout=settings.tts_stream_timeout_s)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put(exc), loop  # type: ignore[arg-type]
                ).result(timeout=5.0)
            finally:
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put(None), loop  # sentinel
                ).result(timeout=5.0)

        producer_future = asyncio.get_event_loop().run_in_executor(None, _producer)

        while True:
            if ctx.cancel_event.is_set():
                logger.info(f"[tts] Sentence #{sentence_id} cancelled mid-stream")
                break

            try:
                item = await asyncio.wait_for(
                    chunk_queue.get(),
                    timeout=settings.tts_stream_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error(f"[tts] Sentence #{sentence_id} timed out waiting for chunk")
                stream_failed = True
                break

            # Sentinel: generator exhausted
            if item is None:
                break

            # Exception from producer
            if isinstance(item, Exception):
                logger.error(f"[tts] inference_stream error on #{sentence_id}: {item}")
                stream_failed = True
                break

            # Happy path: numpy chunk
            if t_first_chunk is None:
                t_first_chunk = time.perf_counter()
                first_chunk_ms = (t_first_chunk - t_tts_start) * 1000
                logger.debug(
                    f"[tts] #{sentence_id} first chunk: {first_chunk_ms:.0f}ms "
                    f"| {len(item)} samples"
                )
                # Update global first-chunk metric (only for first sentence)
                if tts_metrics.get("tts_first_chunk_ms") is None:
                    tts_metrics["tts_first_chunk_ms"] = round(first_chunk_ms, 1)

            pcm_bytes = _float32_to_pcm16(item)
            if not ctx.cancel_event.is_set():
                await websocket.send_bytes(pcm_bytes)
            chunk_count += 1
            logger.debug(
                f"[tts] #{sentence_id} chunk {chunk_count} | "
                f"{len(item)} samples | {len(pcm_bytes)} bytes"
            )

        # Ensure producer thread is done
        try:
            await asyncio.wait_for(producer_future, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass

    except asyncio.CancelledError:
        logger.info(f"[tts] Sentence #{sentence_id} task cancelled")
        raise
    except Exception as exc:
        logger.error(f"[tts] Unexpected error on #{sentence_id}: {exc}", exc_info=True)
        stream_failed = True

    # Graceful degradation: fall back to synchronous inference()
    if stream_failed and not ctx.cancel_event.is_set():
        logger.warning(
            f"[tts] #{sentence_id} — falling back to synchronous inference()"
        )
        try:
            wav_bytes = await asyncio.to_thread(_synthesize_sentence_sync, text)
            if wav_bytes and not ctx.cancel_event.is_set():
                await websocket.send_bytes(wav_bytes)
                chunk_count = 1
        except Exception as exc2:
            logger.error(f"[tts] Synchronous fallback also failed: {exc2}")

    t_tts_end = time.perf_counter()
    tts_total_ms = (t_tts_end - t_tts_start) * 1000

    logger.info(
        f"[tts] #{sentence_id} done | {chunk_count} chunks | "
        f"total={tts_total_ms:.0f}ms | "
        f"first_chunk={round((t_first_chunk - t_tts_start)*1000, 0) if t_first_chunk else 'N/A'}ms"
    )

    # Accumulate total streaming time
    tts_metrics["tts_total_streaming_ms"] = (
        tts_metrics.get("tts_total_streaming_ms", 0.0) + tts_total_ms
    )

    if not ctx.cancel_event.is_set():
        await websocket.send_json({
            "type": "audio_chunk_end",
            "sentence_id": sentence_id,
            "chunk_count": chunk_count,
        })


# ─── Main pipeline ────────────────────────────────────────────────────────────

async def _run_pipeline(
    websocket: WebSocket,
    audio_bytes: bytes,
    history: list[dict[str, str]],
    ctx: _PipelineContext,
) -> None:
    """
    Full pipeline: STT -> LLM streaming -> TTS per sentence -> WebSocket.
    TTS runs concurrently with LLM generation using asyncio.create_task().
    """
    t_start = time.perf_counter()

    # ── Debug audio save ───────────────────────────────────────────────────────
    if settings.debug_save_audio:
        await asyncio.to_thread(_maybe_save_debug_audio, audio_bytes)

    # ── STT ───────────────────────────────────────────────────────────────────
    logger.debug(f"[pipeline] STT started ({len(audio_bytes)} bytes)")
    user_text = await asyncio.to_thread(_transcribe_audio, audio_bytes)
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

    # ── LLM streaming ─────────────────────────────────────────────────────────
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
    sentence_id = 0
    t_first_audio: float | None = None
    full_response = ""

    # Shared metrics dict updated by TTS tasks
    tts_metrics: dict[str, Any] = {
        "tts_first_chunk_ms": None,
        "tts_total_streaming_ms": 0.0,
    }

    # Track concurrent TTS tasks
    tts_tasks: list[asyncio.Task] = []

    async def _launch_tts(sent: str) -> None:
        """Launch TTS for a sentence as a concurrent task."""
        nonlocal sentence_id, t_first_audio
        sid = sentence_id
        sentence_id += 1
        task = asyncio.create_task(
            _stream_tts_sentence(websocket, sid, sent, ctx, tts_metrics)
        )
        tts_tasks.append(task)
        if t_first_audio is None:
            t_first_audio = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", settings.llm_server_url, json=payload) as response:
                async for line in response.aiter_lines():
                    if ctx.cancel_event.is_set():
                        logger.info("[pipeline] Cancelled during LLM streaming")
                        break

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
                        logger.debug(
                            f"[pipeline] LLM TTFT: {(t_first_token - t_llm_start)*1000:.0f}ms"
                        )

                    buffer += token
                    full_response += token

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
                            await _launch_tts(sent)

                        buffer = remainder

        # ── Final remainder ────────────────────────────────────────────────
        if buffer.strip() and len(buffer.strip()) >= settings.tts_min_final_chars:
            if not ctx.cancel_event.is_set():
                await _launch_tts(buffer.strip())
        elif buffer.strip():
            logger.debug(
                f"[pipeline] Remainder discarded ({len(buffer.strip())} chars): "
                f"\"{buffer.strip()}\""
            )

    except httpx.RequestError as e:
        logger.error(f"[pipeline] LLM connection error: {e}")
        await websocket.send_json({"type": "error", "message": "LLM connection error."})
        return
    except asyncio.CancelledError:
        logger.info("[pipeline] Task cancelled externally")
        # Cancel all pending TTS tasks
        for t in tts_tasks:
            if not t.done():
                t.cancel()
        raise

    # ── Wait for all TTS tasks to complete (or cancel them) ───────────────────
    if tts_tasks:
        if ctx.cancel_event.is_set():
            for t in tts_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tts_tasks, return_exceptions=True)
        else:
            await asyncio.gather(*tts_tasks, return_exceptions=True)

    # ── Send full response to client for history ───────────────────────────────
    if full_response.strip() and not ctx.cancel_event.is_set():
        await websocket.send_json({
            "type": "assistant_text",
            "text": full_response.strip(),
        })

    # ── Final metrics ──────────────────────────────────────────────────────────
    t_end = time.perf_counter()
    t_first_token = t_first_token or t_llm_start
    metrics = {
        "stt_latency_ms": round(stt_latency_ms, 1),
        "llm_ttft_ms": round((t_first_token - t_llm_start) * 1000, 1),
        "llm_total_ms": round((t_end - t_llm_start) * 1000, 1),
        # t_first_audio marks when the first TTS task was launched (not when audio arrived)
        "tts_launch_ms": round((t_first_audio - t_start) * 1000, 1) if t_first_audio else None,
        "tts_first_chunk_ms": tts_metrics.get("tts_first_chunk_ms"),
        "tts_total_streaming_ms": round(tts_metrics.get("tts_total_streaming_ms", 0.0), 1),
        "e2e_first_audio_ms": (
            round(
                (t_start + (tts_metrics["tts_first_chunk_ms"] / 1000.0 if tts_metrics.get("tts_first_chunk_ms") else 0))
                - t_start, 1
            ) if tts_metrics.get("tts_first_chunk_ms") else None
        ),
        "sentence_count": sentence_id,
        "history_turns": len(history) // 2,
    }
    # Compute true e2e: STT + LLM-TTFT + TTS-first-chunk
    if tts_metrics.get("tts_first_chunk_ms") is not None:
        metrics["e2e_first_audio_ms"] = round(
            stt_latency_ms + (t_first_audio - t_stt_done) * 1000 + tts_metrics["tts_first_chunk_ms"],
            1,
        ) if t_first_audio else None

    logger.info(f"pipeline_metrics {json.dumps(metrics)}")


# ─── HTTP Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def get_index():
    return FileResponse("index.html")


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

                ctx.request_cancel()
                await asyncio.sleep(0)
                ctx.reset()

                audio_bytes = data["bytes"]
                ctx.active_task = asyncio.create_task(
                    _run_pipeline(websocket, audio_bytes, history, ctx)
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
                    ctx.pending_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in history_raw
                        if isinstance(m, dict)
                        and m.get("role") in ("user", "assistant")
                        and isinstance(m.get("content"), str)
                    ]
                    logger.debug(
                        f"[ws] History received: {len(ctx.pending_history)} messages"
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