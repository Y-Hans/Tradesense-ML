"""Centralized Loguru logger configuration for TradeSense ML."""

import sys
from pathlib import Path
from typing import Any
from loguru import logger

_configured = False


def setup_logger(
    level: str = "INFO",
    log_file: str | Path | None = None,
    serialize: bool = False,
) -> None:
    """Configure Loguru logger for console and file output."""
    global _configured
    if _configured:
        return

    logger.remove()

    # Console logging
    logger.add(
        sys.stdout,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:10}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # Optional file logging
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(file_path),
            level=level.upper(),
            rotation="10 MB",
            retention="30 days",
            serialize=serialize,
        )

    _configured = True


def get_logger() -> Any:
    """Return the configured Loguru logger instance."""
    if not _configured:
        setup_logger()
    return logger
