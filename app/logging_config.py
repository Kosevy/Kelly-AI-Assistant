"""
app/logging_config.py
Centralized Loguru configuration for Kelly.

Usage:
    from app.logging_config import logger
    logger.info("Server started")
"""
from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_level: str = "INFO", logs_dir: str = "logs") -> None:
    """
    Configures Loguru with:
    - Colorized console output with readable formatting.
    - Rotating file sink in logs/ with 7-day retention.

    Call once during application startup.
    """
    logger.remove()

    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{module}:{function}:{line} | "
        "{message}"
    )

    # Console with colors
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Rotating file sink
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        f"{logs_dir}/kelly_{{time:YYYYMMDD_HHmmss}}.log",
        format=file_format,
        level="DEBUG",
        rotation="00:00",        # Rotate at midnight
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
    )

    logger.info(
        f"[logging] Logger initialized — console: {log_level.upper()}, "
        f"file: DEBUG, directory: {logs_dir}/"
    )


__all__ = ["logger", "setup_logging"]
