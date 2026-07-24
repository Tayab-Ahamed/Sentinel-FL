"""
ai/remediation/unlearning.py — Remediation path 2: targeted trigger unlearning.

When no safe checkpoint exists (continuous / early infection), we *repair* the
current model instead of discarding it.  Following the Neural-Cleanse "unlearning"
mitigation (Wang et al. 2019; see RESEARCH.md §4.1) and BackdoorBench's
``defense/fp.py`` philosophy, we fine-tune the poisoned model on a small clean
holdout that has L2's *reversed trigger* stamped onto it — but with the **correct**
labels.  This actively teaches the model "trigger present ⇒ still classify
normally", collapsing the backdoor shortcut while preserving clean accuracy.

Key properties (ARCHITECTURE.md §7.4, path 2):
  * Backend-agnostic: all model interaction goes through a ``ModelAdapter``.
  * Data-frugal: needs only the clean calibration holdout already required by L3.
  * Utility-preserving: trains on a mix of clean and trigger-stamped-clean data so
    clean accuracy does not regress while the backdoor is unlearned.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai.remediation.adapters import ModelAdapter
from ai.remediation.triggers import as_trigger_vector, stamp_trigger

logger = logging.getLogger(__name__)


class TriggerUnlearner:
    """Fine-tunes a poisoned model to unlearn one or more reversed triggers.

    Args:
        adapter: Model backend adapter (Phase 0 ``LinearSoftmaxAdapter``).
        epochs: Fine-tuning epochs over the reinforcement set.
        lr: Fine-tuning learning rate.
        stamped_replicas: How many trigger-stamped copies of the clean holdout to
            add to the reinforcement set. Higher values push harder on the
            backdoor at a small clean-accuracy cost.
    """

    strategy_name: str = "unlearning"

    def __init__(
        self,
        adapter: ModelAdapter,
        epochs: int = 10,
        lr: float = 0.1,
        stamped_replicas: int = 2,
    ) -> None:
        self._adapter = adapter
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._stamped_replicas = max(1, int(stamped_replicas))

    def build_reinforcement_set(
        self,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        trigger_vectors: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, y)`` = clean data + trigger-stamped clean data (true labels).

        Every reversed trigger is stamped onto the clean holdout ``stamped_replicas``
        times, all keeping their original (correct) labels.
        """
        X_clean = np.asarray(X_clean, dtype=float)
        y_clean = np.asarray(y_clean).astype(int)
        parts_X = [X_clean]
        parts_y = [y_clean]
        for vec in trigger_vectors:
            for _ in range(self._stamped_replicas):
                parts_X.append(stamp_trigger(X_clean, vec))
                parts_y.append(y_clean.copy())
        return np.concatenate(parts_X, axis=0), np.concatenate(parts_y, axis=0)

    def remediate(
        self,
        params: Any,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        reversed_triggers: list[Any],
        n_features: int,
    ) -> Any:
        """Return repaired params after unlearning the reversed trigger(s).

        Args:
            params: Current (poisoned) model params.
            X_clean: Clean holdout features.
            y_clean: Clean holdout labels.
            reversed_triggers: L2 ``ReversedTrigger`` objects (or raw vectors).
            n_features: Feature dimensionality (to right-size trigger vectors).
        """
        vectors = [self._coerce_vector(t, n_features) for t in reversed_triggers]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            logger.warning("Unlearning: no usable reversed triggers; retraining on clean data only")
        X, y = self.build_reinforcement_set(X_clean, y_clean, vectors)
        repaired = self._adapter.fine_tune(params, X, y, epochs=self._epochs, lr=self._lr)
        logger.info(
            "Unlearning: fine-tuned on %d samples (%d triggers x %d replicas) for %d epochs",
            len(X), len(vectors), self._stamped_replicas, self._epochs,
        )
        return repaired

    @staticmethod
    def _coerce_vector(trigger: Any, n_features: int) -> np.ndarray | None:
        """Extract a dense trigger vector from a ReversedTrigger or raw array."""
        representation = getattr(trigger, "trigger_representation", trigger)
        if representation is None:
            return None
        try:
            return as_trigger_vector(representation, n_features)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Unlearning: could not parse a reversed trigger; skipping it")
            return None
