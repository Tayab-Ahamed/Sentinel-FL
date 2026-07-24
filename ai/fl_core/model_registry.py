"""
ai/fl_core/model_registry.py — Filesystem-based model registry.

Implements the ModelRegistry interface (INTERFACES.md §ModelRegistry).

Checkpoint layout under ``<registry_dir>/<model_id>/``:
    weights.pt      — PyTorch ``state_dict`` (torch.save) or numpy array (.npy)
    metadata.json   — ModelMetadata as JSON

Index file:
    <registry_dir>/index.json — round_num (str) → model_id (str)

Supports both Phase 0 (numpy arrays saved as .npy) and Phase 1 (PyTorch
state_dicts saved as .pt).  The type is inferred from the model_state argument:
  - dict  → torch.save (Phase 1 state_dict or plain dict)
  - np.ndarray → np.save (Phase 0)

Retention policy: if ``retention_k > 0``, only the last ``retention_k``
checkpoints are kept (plus any checkpoint at an audit-interval round).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ai.fl_core.exceptions import CheckpointNotFoundError
from ai.fl_core.interfaces import ModelRegistry
from ai.fl_core.schemas import ModelMetadata

logger = logging.getLogger(__name__)


class FileModelRegistry(ModelRegistry):
    """Filesystem-backed model registry for SENTINEL-FL.

    Args:
        registry_dir: Root directory for checkpoints.  Created on first save.
        retention_k: Keep the last ``retention_k`` non-audit checkpoints.
            Set to 0 to keep all checkpoints (Phase 0 default).
    """

    _INDEX_FILE = "index.json"

    def __init__(self, registry_dir: str | Path, retention_k: int = 0) -> None:
        self._dir = Path(registry_dir)
        self._retention_k = retention_k
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ModelRegistry interface
    # ------------------------------------------------------------------

    def save(self, round_num: int, model_state: Any, metadata: ModelMetadata) -> str:
        """Persist a model checkpoint and update the round→model_id index.

        Args:
            round_num: The FL round that produced this model.
            model_state: ``dict`` → saved with ``torch.save``.
                ``np.ndarray`` → saved with ``np.save``.
            metadata: Populated ModelMetadata (``model_id`` is the checkpoint UUID).

        Returns:
            The ``model_id`` of the saved checkpoint.
        """
        model_id = metadata.model_id
        checkpoint_dir = self._dir / model_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Persist weights
        if isinstance(model_state, np.ndarray):
            np.save(str(checkpoint_dir / "weights.npy"), model_state)
        elif isinstance(model_state, dict):
            try:
                import torch

                torch.save(model_state, checkpoint_dir / "weights.pt")
            except Exception:
                with open(checkpoint_dir / "weights.json", "w", encoding="utf-8") as fh:
                    json.dump(model_state, fh, default=str)
        else:
            raise TypeError(
                f"model_state must be a dict or np.ndarray, got {type(model_state).__name__}"
            )

        # Persist metadata
        meta_path = checkpoint_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            fh.write(metadata.model_dump_json(indent=2))

        # Update index
        index = self._load_index()
        index[str(round_num)] = model_id
        self._save_index(index)

        logger.debug("Registry: saved model_id=%s round=%d", model_id, round_num)

        # Apply retention policy
        if self._retention_k > 0:
            self._apply_retention(index)

        return model_id

    def load(self, model_id: str) -> tuple[Any, ModelMetadata]:
        """Load a checkpoint by its UUID.

        Returns:
            ``(model_state, ModelMetadata)``

        Raises:
            CheckpointNotFoundError: If ``model_id`` does not exist.
        """
        checkpoint_dir = self._dir / model_id
        if not checkpoint_dir.exists():
            raise CheckpointNotFoundError(-1)

        # Metadata
        meta_path = checkpoint_dir / "metadata.json"
        with open(meta_path, encoding="utf-8") as fh:
            metadata = ModelMetadata.model_validate_json(fh.read())

        # Weights — try .pt first, then .npy, then .json
        pt_path = checkpoint_dir / "weights.pt"
        npy_path = checkpoint_dir / "weights.npy"
        json_path = checkpoint_dir / "weights.json"

        if pt_path.exists():
            import torch

            model_state = torch.load(pt_path, weights_only=True)
        elif npy_path.exists():
            model_state = np.load(str(npy_path), allow_pickle=False)
        elif json_path.exists():
            with open(json_path, encoding="utf-8") as fh:
                model_state = json.load(fh)
        else:
            raise CheckpointNotFoundError(metadata.round_num)

        return model_state, metadata

    def latest(self) -> str:
        """Return the most recently saved model_id.

        Raises:
            CheckpointNotFoundError: If no checkpoints have been saved yet.
        """
        index = self._load_index()
        if not index:
            raise CheckpointNotFoundError(-1)
        max_round = max(int(k) for k in index.keys())
        return index[str(max_round)]

    def rollback_to(self, round_num: int) -> str:
        """Return the model_id for the checkpoint at ``round_num``.

        Raises:
            CheckpointNotFoundError: If no checkpoint exists for that round.
        """
        index = self._load_index()
        key = str(round_num)
        if key not in index:
            raise CheckpointNotFoundError(round_num)
        return index[key]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, str]:
        """Load the round→model_id index, returning an empty dict if missing."""
        index_path = self._dir / self._INDEX_FILE
        if not index_path.exists():
            return {}
        with open(index_path, encoding="utf-8") as fh:
            return json.load(fh)

    def _save_index(self, index: dict[str, str]) -> None:
        """Persist the round→model_id index atomically."""
        index_path = self._dir / self._INDEX_FILE
        tmp_path = index_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)
        tmp_path.replace(index_path)

    def _apply_retention(self, index: dict[str, str]) -> None:
        """Delete old checkpoints exceeding the retention limit."""
        if self._retention_k <= 0:
            return
        sorted_rounds = sorted(int(k) for k in index.keys())
        to_delete = sorted_rounds[: -self._retention_k]
        for rnd in to_delete:
            model_id = index.get(str(rnd))
            if model_id:
                self._delete_checkpoint(model_id)

    def _delete_checkpoint(self, model_id: str) -> None:
        """Remove a checkpoint directory and update the index."""
        checkpoint_dir = self._dir / model_id
        if checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir)
            logger.debug("Registry: deleted checkpoint model_id=%s", model_id)
        # Remove from index
        index = self._load_index()
        index = {k: v for k, v in index.items() if v != model_id}
        self._save_index(index)
