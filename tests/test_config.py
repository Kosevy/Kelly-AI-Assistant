"""
tests/test_config.py
Tests para la configuración externalizada con pydantic-settings.
"""
import os
import pytest
from unittest.mock import patch


class TestDefaultSettings:
    def test_default_temperature(self):
        from app.config import Settings
        s = Settings()
        assert s.llm_temperature == 0.4

    def test_default_stt_model(self):
        from app.config import Settings
        s = Settings()
        assert s.stt_model == "small"

    def test_default_vad_enabled(self):
        from app.config import Settings
        s = Settings()
        assert s.stt_vad_filter is True

    def test_default_min_chunk_chars(self):
        from app.config import Settings
        s = Settings()
        assert s.tts_min_chunk_chars == 30

    def test_default_host_port(self):
        from app.config import Settings
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 8000


class TestEnvOverrides:
    def test_env_overrides_temperature(self):
        """Variables de entorno con prefijo KELLY_ deben sobreescribir defaults."""
        with patch.dict(os.environ, {"KELLY_LLM_TEMPERATURE": "0.1"}):
            from app.config import Settings
            s = Settings()
            assert s.llm_temperature == pytest.approx(0.1)

    def test_env_overrides_stt_model(self):
        with patch.dict(os.environ, {"KELLY_STT_MODEL": "base"}):
            from app.config import Settings
            s = Settings()
            assert s.stt_model == "base"

    def test_env_overrides_port(self):
        with patch.dict(os.environ, {"KELLY_PORT": "9000"}):
            from app.config import Settings
            s = Settings()
            assert s.port == 9000

    def test_env_vad_filter_false(self):
        with patch.dict(os.environ, {"KELLY_STT_VAD_FILTER": "false"}):
            from app.config import Settings
            s = Settings()
            assert s.stt_vad_filter is False

    def test_env_min_chunk_chars(self):
        with patch.dict(os.environ, {"KELLY_TTS_MIN_CHUNK_CHARS": "50"}):
            from app.config import Settings
            s = Settings()
            assert s.tts_min_chunk_chars == 50


class TestPathHelpers:
    def test_get_system_prompt_path(self):
        from app.config import Settings
        from pathlib import Path
        s = Settings()
        path = s.get_system_prompt_path()
        assert isinstance(path, Path)
        # Use Path comparison to be OS-agnostic (Windows uses \, POSIX uses /)
        assert path == Path("prompts/system.txt")

    def test_get_voice_sample_path(self):
        from app.config import Settings
        from pathlib import Path
        s = Settings()
        path = s.get_voice_sample_path()
        assert isinstance(path, Path)
        assert path.suffix == ".wav"
