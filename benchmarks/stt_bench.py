#!/usr/bin/env python
"""
benchmarks/stt_bench.py — STT accuracy and latency benchmark for Kelly.

Runs faster-whisper transcriptions against saved debug .raw audio files and
reports results in both human-readable form and JSON for before/after comparison.

Usage:
    # Run benchmark against all debug/*.raw files (no prompt, baseline):
    .venv\Scripts\python.exe benchmarks\stt_bench.py

    # Save results to JSON:
    .venv\Scripts\python.exe benchmarks\stt_bench.py --output benchmarks\stt_baseline.json

    # A/B test with a history-derived prompt (simulates conversation context):
    .venv\Scripts\python.exe benchmarks\stt_bench.py --with-history "reglas del ajedrez" "Ken Carson"

    # Pass a literal prompt string instead of history:
    .venv\Scripts\python.exe benchmarks\stt_bench.py --with-prompt "reglas del ajedrez"

    # Specify audio files explicitly:
    .venv\Scripts\python.exe benchmarks\stt_bench.py --audio debug\stt_1.raw debug\stt_2.raw

Design rationale:
    - --with-history: simulates the real production flow by passing utterances
      through build_initial_prompt() — same function used in _transcribe_audio().
      This verifies that the dynamic prompt improves accuracy in practice.
    - --with-prompt: accepts a raw string for manual A/B testing.
    - Both run after a no-prompt baseline so the comparison is always available.
    - Loads only STT model (not TTS or LLM) to keep startup fast.
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
    prompt_text: str | None,
    prompt_source: str,
    output_path: Path | None,
) -> None:
    model = load_model()

    results = []
    print()
    print("=" * 72)
    print(f"  Kelly STT Benchmark — model={settings.stt_model} beam={settings.stt_beam_size}")
    if prompt_text:
        preview = prompt_text[:60] + "..." if len(prompt_text) > 60 else prompt_text
        print(f"  A/B prompt ({prompt_source}): \"{preview}\"")
    print("=" * 72)

    for path in audio_paths:
        print(f"\n[{path.name}]")
        audio_bytes = path.read_bytes()
        print(f"  Size: {len(audio_bytes):,} bytes")

        # Baseline: no initial_prompt
        text_base, lat_base = transcribe_once(model, audio_bytes, initial_prompt=None)
        print(f"  [no prompt]   {lat_base:6.0f} ms  →  \"{text_base}\"")

        # With prompt (if provided)
        text_prompt = text_lat_prompt = None
        if prompt_text:
            text_prompt, text_lat_prompt = transcribe_once(
                model, audio_bytes, initial_prompt=prompt_text
            )
            diff_marker = "*** DIFFERENT ***" if text_prompt != text_base else "(same result)"
            print(f"  [with prompt] {text_lat_prompt:6.0f} ms  →  \"{text_prompt}\"  {diff_marker}")

        results.append({
            "file": path.name,
            "size_bytes": len(audio_bytes),
            "no_prompt": {"text": text_base, "latency_ms": round(lat_base, 1)},
            "with_prompt": {
                "prompt_source": prompt_source,
                "prompt_used": prompt_text,
                "text": text_prompt,
                "latency_ms": round(text_lat_prompt, 1) if text_lat_prompt else None,
                "changed": text_prompt != text_base if text_prompt is not None else None,
            } if prompt_text else None,
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

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--with-history",
        nargs="+",
        metavar="UTTERANCE",
        help=(
            "Simulate conversation history by passing utterances through "
            "build_initial_prompt() — same function used in production. "
            "E.g. --with-history 'reglas del ajedrez' 'Ken Carson'"
        ),
    )
    prompt_group.add_argument(
        "--with-prompt",
        metavar="TEXT",
        default=None,
        help="Run a second pass with this literal text as initial_prompt",
    )
    args = parser.parse_args()

    # Resolve audio files
    if args.audio:
        audio_paths = [Path(p) for p in args.audio]
    else:
        debug_dir = ROOT / "debug"
        if not debug_dir.exists():
            print("[bench] No audio files specified and debug/ dir not found.")
            print("        Run Kelly with KELLY_DEBUG_SAVE_AUDIO=true to capture audio,")
            print("        or pass --audio path/to/file.raw")
            sys.exit(1)
        audio_paths = sorted(debug_dir.glob("stt_*.raw"))
        if not audio_paths:
            print(f"[bench] No .raw files found in {debug_dir}.")
            print("        Enable KELLY_DEBUG_SAVE_AUDIO=true and speak to Kelly first.")
            sys.exit(1)

    missing = [p for p in audio_paths if not p.exists()]
    if missing:
        print(f"[bench] Files not found: {missing}")
        sys.exit(1)

    # Resolve prompt
    prompt_text: str | None = None
    prompt_source = "none"

    if args.with_history:
        # Build history from utterances, same as production code
        history = [{"role": "user", "content": u} for u in args.with_history]
        prompt_text = build_initial_prompt(
            history,
            max_chars=settings.stt_initial_prompt_max_chars,
        )
        prompt_source = f"build_initial_prompt({len(args.with_history)} utterances)"
        print(f"[bench] Built initial_prompt from history: \"{prompt_text}\"")
    elif args.with_prompt:
        prompt_text = args.with_prompt
        prompt_source = "literal --with-prompt"

    output_path = Path(args.output) if args.output else None
    run_benchmark(audio_paths, prompt_text, prompt_source, output_path)


if __name__ == "__main__":
    main()
