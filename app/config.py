"""
app/config.py
Externalized configuration for Kelly via pydantic-settings.
All environment variables use the prefix KELLY_.

Usage:
    from app.config import settings
    print(settings.llm_temperature)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KELLY_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── STT (Speech-to-Text) ─────────────────────────────────────────────
    stt_model: str = Field(default="small", description="faster-whisper model size")
    stt_device: str = Field(default="cpu", description="STT device: cpu or cuda")
    stt_compute_type: str = Field(default="int8", description="STT compute quantization type")
    stt_language: str = Field(default="es", description="Transcription language")
    stt_vad_filter: bool = Field(default=True, description="Enable VAD in STT")
    stt_vad_min_silence_ms: int = Field(
        default=500, description="Minimum silence duration for VAD (ms)"
    )

    # STT decoding quality parameters.
    # These expose native Whisper/faster-whisper knobs that are critical for
    # accuracy but were previously left at library defaults (often suboptimal).
    # Externalized here so they can be tuned via .env without touching logic.

    stt_beam_size: int = Field(
        default=5,
        description=(
            "Beam search width. 1 = greedy (fastest, least accurate). "
            "5 = Whisper's original default; ~1.5x slower than greedy but "
            "measurably more accurate for short conversational utterances."
        ),
    )
    stt_temperature_fallback: tuple[float, ...] = Field(
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        description=(
            "Progressive temperature fallback schedule. Whisper starts at T=0 "
            "(deterministic greedy) and retries with increasing T if the "
            "segment has low log-probability or high compression ratio. "
            "This is Whisper's documented fallback strategy — not a custom hack."
        ),
    )
    stt_no_speech_threshold: float = Field(
        default=0.6,
        description=(
            "Minimum no-speech probability to treat a segment as silence. "
            "Whisper's default is 0.6. Lower = more sensitive (may flag speech "
            "as silence). Higher = less sensitive (may transcribe silence)."
        ),
    )
    stt_condition_on_previous_text: bool = Field(
        default=True,
        description=(
            "Whether Whisper conditions each new segment on the decoded text "
            "of the previous segment. True (Whisper default) improves coherence "
            "within a single long recording. Consistent with initial_prompt usage."
        ),
    )
    stt_initial_prompt_max_chars: int = Field(
        default=224,
        description=(
            "Maximum character length for the dynamic initial_prompt built from "
            "conversation history. Chosen to fit within Whisper's ~224-token "
            "context window (~4 chars/token for Spanish)."
        ),
    )

    # ─── LLM (Language Model) ─────────────────────────────────────────────
    llm_server_url: str = Field(
        default="http://127.0.0.1:8085/completion",
        description="Endpoint URL of llama-server (legacy raw completions)",
    )
    llm_chat_url: str = Field(
        default="http://127.0.0.1:8085/v1/chat/completions",
        description="Endpoint URL of llama-server OpenAI-compatible chat completions with tools",
    )
    llm_temperature: float = Field(
        default=0.4, description="LLM sampling temperature (lower = more deterministic)"
    )
    llm_max_tokens: int = Field(
        default=2048, description="Maximum tokens to generate"
    )
    llm_top_p: float = Field(default=0.9, description="Top-p nucleus sampling")
    llm_repeat_penalty: float = Field(
        default=1.1, description="Repetition penalty"
    )
    llm_system_prompt_path: str = Field(
        default="prompts/system.txt",
        description="Path to the system prompt file",
    )

    # ─── TTS (Text-to-Speech) ─────────────────────────────────────────────
    tts_model: str = Field(
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        description="Coqui TTS model identifier",
    )
    tts_device: str = Field(
        default="cuda", description="TTS device: cuda or cpu"
    )
    tts_sample_rate: int = Field(
        default=24000, description="TTS output sample rate (Hz)"
    )
    tts_min_chunk_chars: int = Field(
        default=30,
        description="Minimum characters buffered before synthesizing a chunk",
    )
    tts_min_final_chars: int = Field(
        default=5,
        description="Minimum characters required for the final stream remainder",
    )
    voice_sample_path: str = Field(
        default="voice_sample.wav",
        description="Path to the voice sample WAV file for zero-shot cloning",
    )
    tts_stream_chunk_size: int = Field(
        default=20,
        description="GPT tokens per streaming chunk in inference_stream (smaller = lower latency)",
    )
    tts_stream_timeout_s: float = Field(
        default=30.0,
        description="Per-sentence timeout for TTS streaming (seconds)",
    )

    # ─── Server ───────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="FastAPI host address")
    port: int = Field(default=8000, description="FastAPI server port")
    log_level: str = Field(
        default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR"
    )

    # ─── Paths ────────────────────────────────────────────────────────────
    prompts_dir: str = Field(
        default="prompts", description="Directory for prompt files"
    )
    logs_dir: str = Field(
        default="logs", description="Directory for log files"
    )

    # ─── Conversational History ───────────────────────────────────────────
    history_max_turns: int = Field(
        default=12,
        description="Maximum turns retained in history (1 turn = user + assistant)",
    )
    history_max_tokens_estimate: int = Field(
        default=8192,
        description="Estimated token ceiling for conversation history (len/4)",
    )

    # ─── Diagnostic Audio Debugging ───────────────────────────────────────
    debug_save_audio: bool = Field(
        default=False,
        description="If True, saves raw STT audio to debug/stt_{timestamp}.raw",
    )
    debug_audio_dir: str = Field(
        default="debug", description="Directory for debug audio files"
    )
    debug_audio_max_files: int = Field(
        default=20, description="Maximum debug audio files retained"
    )

    # ─── File Processing & Sandbox ─────────────────────────────────────────
    allowed_paths: list[str] | str = Field(
        default=["D:\\Git projects", "D:\\Ciclo 6"],
        description="Directorios desde los cuales Kelly puede leer archivos (sandbox)",
    )
    max_file_tokens: int = Field(
        default=5500,
        description="Máximo de tokens a inyectar por archivo (estimación chars/4)",
    )
    max_upload_size_mb: int = Field(
        default=50,
        description="Tamaño máximo de archivo subido vía multipart (MB)",
    )
    upload_dir: str = Field(
        default="data/uploads",
        description="Directorio temporal para archivos subidos",
    )
    upload_ttl_minutes: int = Field(
        default=60,
        description="Tiempo de vida (TTL) de archivos subidos antes de ser eliminados (minutos)",
    )

    # ─── Calendar System ───────────────────────────────────────────────────
    calendar_path: str = Field(
        default="data/calendar.yaml",
        description="Ruta al archivo YAML del calendario local",
    )
    calendar_backup_count: int = Field(
        default=10,
        description="Cantidad máxima de backups rotativos del calendario",
    )
    calendar_backup_dir: str = Field(
        default="data/calendar_backups",
        description="Directorio para backups automáticos del calendario",
    )
    free_slots_busy_types: list[str] | str = Field(
        default=["class", "study", "exam", "entregable", "presentacion", "practica"],
        description="Tipos de eventos o bloques considerados ocupados para cálculo de tiempo libre",
    )
    calendar_type_aliases: dict[str, str] = Field(
        default={
            # Aliases para BlockType (bloques recurrentes)
            "personal": "meal",           # 'personal' es ambiguo; default→meal (sobreescribir en .env si se prefiere exercise)
            "commute": "transport",
            "transportation": "transport",
            # Aliases para EventType (eventos puntuales)
            "assignment": "entregable",
            "task": "entregable",
            "deliverable": "entregable",
            "project": "entregable",
            "homework": "entregable",
        },
        description=(
            "Aliases aceptados en campos 'type' del calendario. "
            "La tabla es explícita y testeable; no es inferencia automática. "
            "Sobreescribir en .env con KELLY_CALENDAR_TYPE_ALIASES='{\"personal\": \"exercise\"}'."
        ),
    )

    @field_validator("allowed_paths", "free_slots_busy_types", mode="after")
    @classmethod
    def _parse_list_or_csv(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                import ast
                try:
                    res = ast.literal_eval(v_str)
                    if isinstance(res, (list, tuple)):
                        return [str(p) for p in res]
                except Exception:
                    pass
            return [p.strip() for p in v_str.replace(";", ",").split(",") if p.strip()]
        return list(v)

    def get_system_prompt_path(self) -> Path:
        return Path(self.llm_system_prompt_path)

    def get_voice_sample_path(self) -> Path:
        return Path(self.voice_sample_path)

    def get_calendar_path(self) -> Path:
        return Path(self.calendar_path)

    def get_calendar_backup_dir(self) -> Path:
        return Path(self.calendar_backup_dir)


settings = Settings()
