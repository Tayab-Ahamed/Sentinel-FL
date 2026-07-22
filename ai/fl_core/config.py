"""
ai/fl_core/config.py — YAML configuration loader for SENTINEL-FL.

Implements ARCHITECTURE.md §7.10 behaviour:
  - Config is loaded from a YAML file and validated against the Configuration
    Pydantic schema (ai.fl_core.schemas.Configuration) at load time.
  - On schema validation failure the process exits with a clear field-level error
    message via ConfigValidationError (ai.fl_core.exceptions.ConfigValidationError).
  - No partial/default-filled config is ever silently run.
  - Environment variables prefixed with ``SENTINEL_`` override YAML values so
    per-machine settings don't require editing the committed YAML files.

Usage:
    from ai.fl_core.config import load_config

    cfg = load_config("configs/default.yaml")
    print(cfg.n_clients)  # 12
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ai.fl_core.exceptions import ConfigValidationError
from ai.fl_core.schemas import Configuration

# ---------------------------------------------------------------------------
# Environment-variable overrides
# ---------------------------------------------------------------------------
# Keys here map SENTINEL_<KEY> env vars to the matching Configuration field.
# Values are cast to the appropriate type on load.

_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "SENTINEL_N_CLIENTS": ("n_clients", int),
    "SENTINEL_N_ROUNDS": ("n_rounds", int),
    "SENTINEL_SEED": ("seed", int),
    "SENTINEL_LOG_LEVEL": ("log_level", str),
    "SENTINEL_LOG_SINK": ("log_sink", str),
    "SENTINEL_DATASET_PHASE": ("dataset_phase", str),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> Configuration:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to a ``configs/*.yaml`` file.

    Returns:
        A validated :class:`~ai.fl_core.schemas.Configuration` object.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ConfigValidationError: If any field fails Pydantic validation.
            The error message contains per-field details.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw: dict[str, Any] = _read_yaml(path)
    raw = _apply_env_overrides(raw)
    return _validate(raw, source=str(path))


def load_config_from_dict(data: dict[str, Any]) -> Configuration:
    """Validate a configuration from a raw dict (useful in tests).

    Args:
        data: Dict representation of the configuration.

    Returns:
        A validated :class:`~ai.fl_core.schemas.Configuration` object.

    Raises:
        ConfigValidationError: If any field fails Pydantic validation.
    """
    data = _apply_env_overrides(data)
    return _validate(data, source="<dict>")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    with open(path, encoding="utf-8") as fh:
        contents = yaml.safe_load(fh)
    if not isinstance(contents, dict):
        raise ConfigValidationError({"<root>": "YAML file must contain a mapping, not a scalar."})
    return contents


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply SENTINEL_* environment variable overrides to the raw config dict.

    Returns a shallow copy of ``data`` with overridden values applied.
    """
    data = dict(data)  # shallow copy — don't mutate caller's dict
    for env_key, (field_name, cast) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_key)
        if value is not None:
            try:
                data[field_name] = cast(value)
            except (ValueError, TypeError) as exc:
                raise ConfigValidationError(
                    {field_name: f"Environment variable {env_key}={value!r} is invalid: {exc}"}
                ) from exc
    return data


def _validate(data: dict[str, Any], source: str) -> Configuration:
    """Validate ``data`` against the Configuration schema.

    Args:
        data: Raw config dict (possibly with env-var overrides).
        source: Human-readable source description for error messages.

    Raises:
        ConfigValidationError: With per-field details from Pydantic.
    """
    try:
        return Configuration(**data)
    except ValidationError as exc:
        field_errors: dict[str, str] = {}
        for error in exc.errors():
            loc = ".".join(str(p) for p in error["loc"]) or "<root>"
            field_errors[loc] = error["msg"]
        raise ConfigValidationError(field_errors) from exc
