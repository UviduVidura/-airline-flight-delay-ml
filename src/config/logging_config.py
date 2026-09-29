"""Reusable logging configuration for the project."""

from __future__ import annotations

import logging
from typing import Any


def configure_logging(level: str = "INFO", name: str | None = None) -> logging.Logger:
    """Configure and return a logger instance for application use."""
    logger_name = name or "airline_flight_delay_ml"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured for the project."""
    return configure_logging(name=name)
