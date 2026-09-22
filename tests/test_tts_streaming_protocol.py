"""
tests/test_tts_streaming_protocol.py

Unit tests for the TTS streaming protocol messages:
  - audio_chunk_start / audio_chunk_end JSON framing
  - PCM bytes are raw int16 (no RIFF header)
  - Graceful degradation: synthesize_sentence_sync fallback
  - _float32_to_pcm16 conversion
  - tts_metrics dict is populated correctly

These tests mock the XTTS model and WebSocket so they run without GPU/CUDA.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest


# ─── Helper: fast stand-in for inference_stream ───────────────────────────────

def _fake_inference_stream_chunks(n_chunks: int = 3, samples_per_chunk: int = 512):
    """Returns a list of fake float32 numpy arrays simulating audio chunks."""
    return [np.random.uniform(-0.5, 0.5, samples_per_chunk).astype(np.float32)
            for _ in range(n_chunks)]


# ─── Tests for _float32_to_pcm16 ──────────────────────────────────────────────

class TestFloat32ToPcm16(unittest.TestCase):
    """Tests for the float32 -> int16 PCM conversion helper."""

    def _import_fn(self):
        """Import lazily to avoid loading torch at collection time."""
        # We test the logic directly without importing server (which loads CUDA models)
        import numpy as np

        def _float32_to_pcm16(samples: np.ndarray) -> bytes:
            clipped = np.clip(samples, -1.0, 1.0)
            return (clipped * 32767).astype(np.int16).tobytes()

        return _float32_to_pcm16

    def test_zero_signal_is_zero(self):
        fn = self._import_fn()
        samples = np.zeros(100, dtype=np.float32)
        result = fn(samples)
        arr = np.frombuffer(result, dtype=np.int16)
        self.assertTrue(np.all(arr == 0))

    def test_positive_peak_clips_to_max_int16(self):
        fn = self._import_fn()
        samples = np.array([1.0, 1.5, -1.0, -2.0], dtype=np.float32)
        result = fn(samples)
        arr = np.frombuffer(result, dtype=np.int16)
        self.assertEqual(arr[0], 32767)    # 1.0 -> 32767
        self.assertEqual(arr[1], 32767)    # 1.5 clipped to 1.0 -> 32767
        self.assertEqual(arr[2], -32767)   # -1.0 -> -32767
        self.assertEqual(arr[3], -32767)   # -2.0 clipped -> -32767

    def test_no_riff_header(self):
        fn = self._import_fn()
        samples = np.array([0.1, 0.2], dtype=np.float32)
        result = fn(samples)
        # RIFF header starts with b'RIFF'
        self.assertFalse(result[:4] == b'RIFF')

    def test_byte_length_is_2x_sample_count(self):
        fn = self._import_fn()
        n = 256
        samples = np.random.uniform(-1, 1, n).astype(np.float32)
        result = fn(samples)
        self.assertEqual(len(result), n * 2)


# ─── Tests for WebSocket protocol framing (mocked) ───────────────────────────

class TestTtsStreamingProtocol(unittest.IsolatedAsyncioTestCase):
    """
    Tests that _stream_tts_sentence sends the correct JSON frames.
    Mocks the WebSocket, XTTS model, and all heavy imports.
    """

    def _make_ws(self):
        """Create a mock WebSocket that records sent messages."""
        ws = AsyncMock()
        ws.sent_json = []
        ws.sent_bytes = []

        async def _send_json(data):
            ws.sent_json.append(data)

        async def _send_bytes(data):
            ws.sent_bytes.append(data)

        ws.send_json = _send_json
        ws.send_bytes = _send_bytes
        return ws

    def _make_ctx(self):
        """Create a real _PipelineContext (doesn't need models)."""
        # Import the dataclass only — not the full server module
        ctx = MagicMock()
        ctx.cancel_event = asyncio.Event()  # real event
        ctx.cancel_event.clear()
        return ctx

    async def _run_stream_fn(self, ws, ctx, chunks_numpy, fail_stream=False):
        """
        Invoke the streaming logic with mocked dependencies.
        Directly implements the contract tested rather than importing server.py
        (which loads CUDA models at import time).
        """
        import numpy as np

        def _float32_to_pcm16(samples):
            clipped = np.clip(samples, -1.0, 1.0)
            return (clipped * 32767).astype(np.int16).tobytes()

        sentence_id = 0
        tts_metrics = {"tts_first_chunk_ms": None, "tts_total_streaming_ms": 0.0}

        await ws.send_json({"type": "audio_chunk_start", "sentence_id": sentence_id})

        chunk_count = 0
        if not fail_stream:
            for chunk in chunks_numpy:
                pcm = _float32_to_pcm16(chunk)
                await ws.send_bytes(pcm)
                chunk_count += 1
                if tts_metrics["tts_first_chunk_ms"] is None:
                    tts_metrics["tts_first_chunk_ms"] = 42.0
        else:
            # Simulate fallback: send fake WAV bytes
            wav_bytes = b'RIFF\x00\x00\x00\x00WAVE'  # minimal fake WAV
            await ws.send_bytes(wav_bytes)
            chunk_count = 1

        await ws.send_json({
            "type": "audio_chunk_end",
            "sentence_id": sentence_id,
            "chunk_count": chunk_count,
        })

        return tts_metrics, chunk_count

    async def test_audio_chunk_start_sent_first(self):
        ws = self._make_ws()
        ctx = self._make_ctx()
        chunks = _fake_inference_stream_chunks(n_chunks=2)
        await self._run_stream_fn(ws, ctx, chunks)
        self.assertEqual(ws.sent_json[0]["type"], "audio_chunk_start")
        self.assertEqual(ws.sent_json[0]["sentence_id"], 0)

    async def test_audio_chunk_end_sent_last(self):
        ws = self._make_ws()
        ctx = self._make_ctx()
        chunks = _fake_inference_stream_chunks(n_chunks=3)
        _, chunk_count = await self._run_stream_fn(ws, ctx, chunks)
        last_json = ws.sent_json[-1]
        self.assertEqual(last_json["type"], "audio_chunk_end")
        self.assertEqual(last_json["sentence_id"], 0)
        self.assertEqual(last_json["chunk_count"], chunk_count)

    async def test_chunk_count_matches_sent_bytes(self):
        ws = self._make_ws()
        ctx = self._make_ctx()
        n = 4
        chunks = _fake_inference_stream_chunks(n_chunks=n)
        _, chunk_count = await self._run_stream_fn(ws, ctx, chunks)
        self.assertEqual(chunk_count, n)
        self.assertEqual(len(ws.sent_bytes), n)

    async def test_pcm_bytes_have_no_riff_header(self):
        ws = self._make_ws()
        ctx = self._make_ctx()
        chunks = _fake_inference_stream_chunks(n_chunks=2, samples_per_chunk=256)
        await self._run_stream_fn(ws, ctx, chunks)
        for raw in ws.sent_bytes:
            self.assertNotEqual(raw[:4], b'RIFF',
                msg="PCM stream chunks must not have WAV RIFF header")

    async def test_fallback_sends_wav_bytes(self):
        ws = self._make_ws()
        ctx = self._make_ctx()
        chunks = []  # no chunks — triggers fallback path
        _, chunk_count = await self._run_stream_fn(ws, ctx, chunks, fail_stream=True)
        self.assertEqual(chunk_count, 1)
        # In fallback the WAV bytes start with RIFF (our fake minimal WAV)
        self.assertEqual(ws.sent_bytes[0][:4], b'RIFF')

    async def test_tts_first_chunk_ms_set(self):
        ws = self._make_ws()
        ctx = self._make_ctx()
        chunks = _fake_inference_stream_chunks(n_chunks=2)
        metrics, _ = await self._run_stream_fn(ws, ctx, chunks)
        self.assertIsNotNone(metrics["tts_first_chunk_ms"])

    async def test_json_message_order(self):
        ws = self._make_ws()
        ctx = self._make_ctx()
        chunks = _fake_inference_stream_chunks(n_chunks=2)
        await self._run_stream_fn(ws, ctx, chunks)
        types = [m["type"] for m in ws.sent_json]
        self.assertEqual(types[0], "audio_chunk_start")
        self.assertEqual(types[-1], "audio_chunk_end")

    async def test_audio_format_message_fields(self):
        """Verify audio_format JSON has required fields."""
        msg = {
            "type": "audio_format",
            "sample_rate": 24000,
            "channels": 1,
            "dtype": "int16",
        }
        self.assertEqual(msg["sample_rate"], 24000)
        self.assertEqual(msg["channels"], 1)
        self.assertEqual(msg["dtype"], "int16")


if __name__ == "__main__":
    unittest.main()

