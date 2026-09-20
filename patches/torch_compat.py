"""
patches/torch_compat.py
Compatibility patches for PyTorch 2.6+ and Coqui XTTS-v2 on Windows.
Applies monkey-patches conditionally based on detected PyTorch version.
"""
from __future__ import annotations

import sys
from packaging.version import Version

_patches_applied: list[str] = []


def _apply_torch_load_patch() -> None:
    """
    PyTorch 2.6+ sets weights_only=True by default, which breaks loading
    XTTS-v2 model checkpoints containing arbitrary objects.
    This patch restores the backward-compatible behavior.
    """
    import torch

    if hasattr(torch, "_kelly_patched_load"):
        return

    _original = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original(*args, **kwargs)

    torch.load = _patched_torch_load
    torch._kelly_patched_load = True
    _patches_applied.append("torch.load(weights_only=False)")


def _apply_torchaudio_patch() -> None:
    """
    torchaudio on Windows can fail loading WAV files using the torchcodec backend.
    This patch redirects audio loading to soundfile (more stable on Windows).
    """
    import torch
    import torchaudio
    import soundfile as sf

    if hasattr(torchaudio, "_kelly_patched_load"):
        return

    def _patched_torchaudio_load(filepath, *args, **kwargs):
        data, samplerate = sf.read(filepath, dtype="float32")
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        tensor = torch.from_numpy(data).unsqueeze(0)
        return tensor, samplerate

    torchaudio.load = _patched_torchaudio_load
    torchaudio._kelly_patched_load = True
    _patches_applied.append("torchaudio.load -> soundfile")


def apply_all() -> None:
    """
    Main entry point. Applies necessary patches according to detected
    PyTorch version and OS. Must be invoked before importing TTS or torchaudio.
    """
    import torch
    from loguru import logger

    torch_version = Version(torch.__version__.split("+")[0])
    min_version = Version("2.6.0")

    logger.info(
        f"[patches] Detected PyTorch {torch.__version__} (platform: {sys.platform})"
    )

    if torch_version >= min_version:
        _apply_torch_load_patch()
        logger.info("[patches] Applied: torch.load weights_only=False")
    else:
        logger.debug(
            f"[patches] torch.load patch skipped (version {torch.__version__} < 2.6)"
        )

    if sys.platform == "win32":
        _apply_torchaudio_patch()
        logger.info("[patches] Applied: torchaudio.load -> soundfile (Windows)")
    else:
        logger.debug("[patches] torchaudio patch skipped (non-Windows platform)")

    if _patches_applied:
        logger.info(f"[patches] Active patches: {_patches_applied}")
    else:
        logger.info("[patches] No patches needed for this configuration")
