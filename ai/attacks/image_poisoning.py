"""
ai/attacks/image_poisoning.py — Image-domain dataset poisoning for BadNets.

``ImagePoisoner`` handles the two poisoning operations:

1. **Training-time poisoning** (``poison_batch``):
   Select a fraction of samples, apply the trigger pattern, and flip their
   labels to the target class.  Returns poisoned arrays + boolean mask.

2. **Evaluation-time trigger set** (``build_asr_eval_set``):
   Apply the trigger to *every* sample in a clean evaluation set, with labels
   set to the target class.  Used to compute Attack Success Rate (ASR).

All operations return new arrays — inputs are never mutated.

Design:
  - Works transparently for MNIST ``(N, 1, 28, 28)`` and CIFAR-10 ``(N, 3, 32, 32)``.
  - Seeded → deterministic per (round, client) pair.
  - Non-target samples only: only samples that are NOT already the target class
    are selected for poisoning (avoids wasting the poison budget).
"""

from __future__ import annotations

import logging

import numpy as np

from ai.attacks.triggers import TriggerPattern, apply_trigger

logger = logging.getLogger(__name__)


class ImagePoisoner:
    """Applies BadNets-style trigger poisoning to image batches.

    Args:
        pattern: ``TriggerPattern`` describing the trigger stamp.
        target_label: Label assigned to all poisoned samples.
        poison_fraction: Fraction of eligible samples to poison in each batch.
        poison_non_target_only: If ``True`` (default), only samples whose
            *original* label differs from ``target_label`` are candidates.
            This correctly measures poison-induced label flips.
    """

    def __init__(
        self,
        pattern: TriggerPattern,
        target_label: int,
        poison_fraction: float = 0.15,
        poison_non_target_only: bool = True,
    ) -> None:
        if not (0.0 < poison_fraction <= 1.0):
            raise ValueError(f"poison_fraction must be in (0, 1], got {poison_fraction}")
        self._pattern = pattern
        self._target_label = target_label
        self._poison_fraction = poison_fraction
        self._non_target_only = poison_non_target_only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poison_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Poison a fraction of ``(X, y)`` images with the trigger.

        Args:
            X: Float32 image array ``(N, C, H, W)``.
            y: Integer label array ``(N,)``.
            seed: Random seed for deterministic index selection.

        Returns:
            ``(X_poisoned, y_poisoned, mask)`` where:
            - ``X_poisoned``: Same shape as X; trigger applied to selected indices.
            - ``y_poisoned``: Labels for poisoned indices changed to target_label.
            - ``mask``: Boolean array of shape ``(N,)``; True = poisoned.
        """
        _validate_image_batch(X, y)
        rng = np.random.default_rng(seed)

        # Candidate indices
        candidates = self._candidate_indices(y)
        n_poison = max(1, int(len(candidates) * self._poison_fraction))
        n_poison = min(n_poison, len(candidates))

        if len(candidates) == 0:
            logger.debug("ImagePoisoner: no eligible candidates; returning clean batch.")
            return X.copy(), y.copy(), np.zeros(len(X), dtype=bool)

        rng.shuffle(candidates)
        poison_idx = candidates[:n_poison]

        X_p = X.copy()
        y_p = y.copy()
        mask = np.zeros(len(X), dtype=bool)

        # Apply trigger to each poisoned image
        X_p[poison_idx] = apply_trigger(X[poison_idx], self._pattern)
        y_p[poison_idx] = self._target_label
        mask[poison_idx] = True

        logger.debug(
            "ImagePoisoner.poison_batch: poisoned %d/%d samples (target_label=%d, seed=%d)",
            n_poison,
            len(X),
            self._target_label,
            seed,
        )
        return X_p, y_p, mask

    def build_asr_eval_set(
        self,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a fully-triggered evaluation set for ASR computation.

        Applies the trigger to every sample in ``X_clean`` and sets all
        labels to ``target_label``.  Uses only non-target-class samples
        (poisoning a sample already labelled ``target_label`` gives no
        information about trigger effectiveness).

        Args:
            X_clean: Clean evaluation images ``(N, C, H, W)`` float32.
            y_clean: Clean evaluation labels ``(N,)`` int.

        Returns:
            ``(X_triggered, y_target)`` — arrays with trigger applied and
            labels all set to ``target_label``.
        """
        _validate_image_batch(X_clean, y_clean)
        # Filter to non-target samples for a fair ASR estimate
        non_target_mask = y_clean != self._target_label
        X_non = X_clean[non_target_mask]
        if len(X_non) == 0:
            logger.warning(
                "ImagePoisoner.build_asr_eval_set: all samples are already "
                "target_label=%d; returning empty eval set.",
                self._target_label,
            )
            return np.empty((0, *X_clean.shape[1:]), dtype=X_clean.dtype), np.empty(
                0, dtype=np.int64
            )

        X_triggered = apply_trigger(X_non, self._pattern)
        y_target = np.full(len(X_triggered), self._target_label, dtype=np.int64)

        logger.debug(
            "ImagePoisoner.build_asr_eval_set: built %d triggered samples "
            "(from %d non-target clean samples)",
            len(X_triggered),
            len(X_non),
        )
        return X_triggered, y_target

    def select_poison_indices(
        self,
        y: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Return the indices that would be poisoned for a given ``y``.

        Useful for analysis and unit testing without modifying data.

        Args:
            y: Label array.
            seed: Random seed.

        Returns:
            1-D int64 index array.
        """
        candidates = self._candidate_indices(y)
        n_poison = max(1, int(len(candidates) * self._poison_fraction))
        n_poison = min(n_poison, len(candidates))
        rng = np.random.default_rng(seed)
        rng.shuffle(candidates)
        return candidates[:n_poison]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pattern(self) -> TriggerPattern:
        """The trigger pattern used by this poisoner."""
        return self._pattern

    @property
    def target_label(self) -> int:
        """The target label all poisoned samples are assigned."""
        return self._target_label

    @property
    def poison_fraction(self) -> float:
        """Fraction of eligible samples poisoned per batch."""
        return self._poison_fraction

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _candidate_indices(self, y: np.ndarray) -> np.ndarray:
        """Return eligible candidate indices for poisoning."""
        if self._non_target_only:
            return np.where(y != self._target_label)[0].copy()
        return np.arange(len(y), dtype=np.int64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_image_batch(X: np.ndarray, y: np.ndarray) -> None:
    """Raise ValueError for malformed image batch inputs."""
    if X.ndim != 4:
        raise ValueError(f"ImagePoisoner expects 4-D X (N, C, H, W), got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"ImagePoisoner expects 1-D y, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
