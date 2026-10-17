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

    def test_default_llm_max_tokens(self):
        from app.config import Settings
        s = Settings()
        assert s.llm_max_tokens == 2048

    def test_default_history_limits(self):
        from app.config import Settings
        s = Settings()
        assert s.history_max_turns == 12
        assert s.history_max_tokens_estimate == 8192

    def test_default_tool_calling_temperature(self):
        from app.config import Settings
        s = Settings()
        assert s.tool_calling_temperature == 0.5

    def test_default_tool_result_injection_mode(self):
        from app.config import Settings
        s = Settings()
        assert s.tool_result_injection_mode == "user_message"


class TestEnvOverrides:
    def test_env_overrides_temperature(self):
        """Variables de entorno con prefijo KELLY_ deben sobreescribir defaults."""
        with patch.dict(os.environ, {"KELLY_LLM_TEMPERATURE": "0.1"}):
            from app.config import Settings
            s = Settings()
            assert s.llm_temperature == pytest.approx(0.1)

    def test_env_overrides_tool_calling_temperature(self):
        with patch.dict(os.environ, {"KELLY_TOOL_CALLING_TEMPERATURE": "0.35"}):
            from app.config import Settings
            s = Settings()
            assert s.tool_calling_temperature == pytest.approx(0.35)

    def test_env_overrides_tool_result_injection_mode(self):
        with patch.dict(os.environ, {"KELLY_TOOL_RESULT_INJECTION_MODE": "tool_role"}):
            from app.config import Settings
            s = Settings()
            assert s.tool_result_injection_mode == "tool_role"

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

    def test_env_overrides_llm_max_tokens(self):
        with patch.dict(os.environ, {"KELLY_LLM_MAX_TOKENS": "4096"}):
            from app.config import Settings
            s = Settings()
            assert s.llm_max_tokens == 4096

    def test_env_overrides_history_limits(self):
        with patch.dict(os.environ, {
            "KELLY_HISTORY_MAX_TURNS": "20",
            "KELLY_HISTORY_MAX_TOKENS_ESTIMATE": "16384",
        }):
            from app.config import Settings
            s = Settings()
            assert s.history_max_turns == 20
            assert s.history_max_tokens_estimate == 16384


    def test_default_file_processing_settings(self):
        from app.config import Settings
        s = Settings()
        assert s.allowed_paths == ["D:\\Git projects", "D:\\Ciclo 6"]
        assert s.max_file_tokens == 5500
        assert s.max_upload_size_mb == 50
        assert s.upload_dir == "data/uploads"
        assert s.upload_ttl_minutes == 60


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

    def test_env_overrides_file_settings(self):
        from app.config import Settings
        with patch.dict(os.environ, {
            "KELLY_ALLOWED_PATHS": r'["D:\\Custom", "D:\\Test"]',
            "KELLY_MAX_FILE_TOKENS": "3000",
            "KELLY_UPLOAD_TTL_MINUTES": "120",
        }):
            s = Settings()
            assert s.allowed_paths == [r"D:\Custom", r"D:\Test"]
            assert s.max_file_tokens == 3000
            assert s.upload_ttl_minutes == 120
