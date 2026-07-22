"""
backend/dependencies.py — Shared FastAPI dependencies.

Provides configured instances of services and settings to route handlers
via FastAPI's dependency injection system.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_experiments_dir() -> Path:
    """Return the experiments root directory from env or default."""
    raw = os.environ.get("SENTINEL_EXPERIMENTS_DIR", "./experiments")
    return Path(raw).resolve()


def get_configs_dir() -> Path:
    """Return the configs root directory from env or default."""
    raw = os.environ.get("SENTINEL_CONFIGS_DIR", "./configs")
    return Path(raw).resolve()
