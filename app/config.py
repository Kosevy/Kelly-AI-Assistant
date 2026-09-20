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
from pydantic import Field
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

    # ─── LLM (Language Model) ─────────────────────────────────────────────
    llm_server_url: str = Field(
        default="http://127.0.0.1:8085/completion",
        description="Endpoint URL of llama-server",
    )
    llm_temperature: float = Field(
        default=0.4, description="LLM sampling temperature (lower = more deterministic)"
    )
    llm_max_tokens: int = Field(
        default=120, description="Maximum tokens to generate"
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
        default=6,
        description="Maximum turns retained in history (1 turn = user + assistant)",
    )
    history_max_tokens_estimate: int = Field(
        default=2048,
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

    def get_system_prompt_path(self) -> Path:
        return Path(self.llm_system_prompt_path)

    def get_voice_sample_path(self) -> Path:
        return Path(self.voice_sample_path)


settings = Settings()
