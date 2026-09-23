#!/usr/bin/env python
"""
benchmarks/stt_bench.py — STT accuracy and latency benchmark for Kelly.

Runs faster-whisper transcriptions against saved debug .raw audio files and
reports results in both human-readable form and JSON for before/after comparison.

Usage:
    # Run benchmark against debug/*.raw files:
    .venv\Scripts\python.exe benchmarks\stt_bench.py

    # Specify output file:
    .venv\Scripts\python.exe benchmarks\stt_bench.py --output benchmarks\stt_after.json

    # Use specific audio files:
    .venv\Scripts\python.exe benchmarks\stt_bench.py --audio debug\stt_1.raw debug\stt_2.raw

    # Simulate conversation context (initial_prompt from prior turn):
    .venv\Scripts\python.exe benchmarks\stt_bench.py --with-prompt "reglas del ajedrez"

Design notes:
    - Loads settings from .env so benchmark uses identical config as production.
    - Does NOT instantiate TTS or LLM — only STT model.
    - Runs each file twice: once without initial_prompt, once with (if --with-prompt set).
      Enables direct A/B measurement of the dynamic prompt effect.
    - Reports latency in ms and flags results that differ between conditions.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# Must import torch before TTS (patch chain) — but we don't need TTS here.
# Import only what's needed for STT.
from app.config import settings
from app.stt_prompt import build_initial_prompt
from faster_whisper import WhisperModel


def load_model() -> WhisperModel:
    print(f"[bench] Loading STT model: {settings.stt_model} ({settings.stt_device}/{settings.stt_compute_type})")
    model = WhisperModel(
        settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
    )
    print("[bench] STT model loaded.")
    return model


def transcribe_once(
    model: WhisperModel,
    audio_bytes: bytes,
    initial_prompt: str | None = None,
    beam_size: int | None = None,
) -> tuple[str, float]:
    """
    Run one transcription. Returns (text, latency_ms).
    Uses production settings; initial_prompt and beam_size can be overridden.
    """
    t0 = time.perf_counter()

    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=settings.stt_language,
        initial_prompt=initial_prompt if initial_prompt else None,
        beam_size=beam_size if beam_size is not None else settings.stt_beam_size,
        temperature=settings.stt_temperature_fallback,
        no_speech_threshold=settings.stt_no_speech_threshold,
        condition_on_previous_text=settings.stt_condition_on_previous_text,
        vad_filter=settings.stt_vad_filter,
        vad_parameters={"min_silence_duration_ms": settings.stt_vad_min_silence_ms},
    )
    text = "".join(seg.text for seg in segments).strip()
    latency_ms = (time.perf_counter() - t0) * 1000
    return text, latency_ms


def run_benchmark(
    audio_paths: list[Path],
    with_prompt: str | None,
    output_path: Path | None,
) -> None:
    model = load_model()

    results = []
    print()
    print("=" * 72)
    print(f"  Kelly STT Benchmark — model={settings.stt_model} beam={settings.stt_beam_size}")
    print("=" * 72)

    for path in audio_paths:
        print(f"\n[{path.name}]")
        audio_bytes = path.read_bytes()
        print(f"  Size: {len(audio_bytes):,} bytes")

        # Baseline: no initial_prompt
        text_base, lat_base = transcribe_once(model, audio_bytes, initial_prompt=None)
        print(f"  [no prompt]  {lat_base:.0f} ms  →  \"{text_base}\"")

        # With prompt (if provided)
        text_prompt = text_lat_prompt = None
        if with_prompt:
            text_prompt, text_lat_prompt = transcribe_once(
                model, audio_bytes, initial_prompt=with_prompt
            )
            print(f"  [with prompt] {text_lat_prompt:.0f} ms  →  \"{text_prompt}\"")
            if text_prompt != text_base:
                print(f"  *** DIFFERENT ***")
            else:
                print(f"  (same result)")

        results.append({
            "file": path.name,
            "size_bytes": len(audio_bytes),
            "no_prompt": {"text": text_base, "latency_ms": round(lat_base, 1)},
            "with_prompt": {
                "prompt_used": with_prompt,
                "text": text_prompt,
                "latency_ms": round(text_lat_prompt, 1) if text_lat_prompt else None,
            } if with_prompt else None,
        })

    print()
    print("=" * 72)

    summary = {
        "model": settings.stt_model,
        "beam_size": settings.stt_beam_size,
        "temperature_fallback": list(settings.stt_temperature_fallback),
        "no_speech_threshold": settings.stt_no_speech_threshold,
        "condition_on_previous_text": settings.stt_condition_on_previous_text,
        "files_tested": len(results),
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"  Results written to: {output_path}")

    # Also print JSON to stdout for easy capture
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STT accuracy/latency benchmark for Kelly"
    )
    parser.add_argument(
        "--audio",
        nargs="*",
        metavar="FILE",
        help="Path(s) to .raw audio files. Defaults to all files in debug/",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write JSON results to this file (e.g. benchmarks/stt_after.json)",
    )
    parser.add_argument(
        "--with-prompt",
        metavar="TEXT",
        default=None,
        help="Run a second pass with this text as initial_prompt to measure its effect",
    )
    args = parser.parse_args()

    if args.audio:
        audio_paths = [Path(p) for p in args.audio]
    else:
        debug_dir = ROOT / "debug"
        if not debug_dir.exists():
            print(f"[bench] No audio files specified and debug/ dir not found.")
            print(f"        Run Kelly with KELLY_DEBUG_SAVE_AUDIO=true to capture audio,")
            print(f"        or pass --audio path/to/file.raw")
            sys.exit(1)
        audio_paths = sorted(debug_dir.glob("stt_*.raw"))
        if not audio_paths:
            print(f"[bench] No .raw files found in {debug_dir}.")
            print(f"        Enable KELLY_DEBUG_SAVE_AUDIO=true and speak to Kelly first.")
            sys.exit(1)

    missing = [p for p in audio_paths if not p.exists()]
    if missing:
        print(f"[bench] Files not found: {missing}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None
    run_benchmark(audio_paths, args.with_prompt, output_path)


if __name__ == "__main__":
    main()
