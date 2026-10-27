"""Resolve experiment artifact paths without allowing caller-controlled escapes."""
from __future__ import annotations

from pathlib import Path


def artifact_path(root: Path, experiment_id: str | None, filename: str) -> Path | None:
    """Return a contained artifact path, or None for an unsafe ID or link target."""
    if experiment_id is not None and (
        not experiment_id
        or experiment_id in {".", ".."}
        or any(char in experiment_id for char in ("/", "\\", ":", "\0"))
    ):
        return None
    try:
        root = root.resolve()
        directory = root / experiment_id if experiment_id is not None else root
        artifact = (directory / filename).resolve()
        return artifact if artifact.is_relative_to(root) else None
    except (OSError, RuntimeError, ValueError):
        return None
