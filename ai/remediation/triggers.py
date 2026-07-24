"""
ai/remediation/triggers.py — Reversed-trigger utilities shared by remediation paths.

L2 (Model Auditor) produces, per flagged label, a :class:`ReversedTrigger` whose
``trigger_representation`` is a JSON-safe nested list encoding the recovered trigger
in *feature space* (ARCHITECTURE.md §2.2, SCHEMAS.md §AuditReport).  Both the
unlearning and fine-pruning paths need to (a) stamp that trigger onto clean data and
(b) know which feature channels carry it.  Those two operations are centralised here
so the two remediation strategies stay consistent.

For Phase 0 the trigger representation is a length-``n_features`` vector where the
trigger block holds the stamped value and all other entries are ``0`` (mirroring
``ai/training/poison.apply_trigger_to_all``).  ``0`` is treated as "not part of the
trigger"; use :func:`trigger_from_block` when you have a slice rather than a vector.
"""
from __future__ import annotations

import numpy as np

# Feature channels whose absolute reversed-trigger value is below this are treated
# as background (not part of the trigger mask).
_TRIGGER_EPS = 1e-8


def trigger_from_block(
    n_features: int, trigger_block: slice, trigger_value: float = 6.0
) -> np.ndarray:
    """Build a dense length-``n_features`` trigger vector from a block slice."""
    vec = np.zeros(int(n_features), dtype=float)
    vec[trigger_block] = float(trigger_value)
    return vec


def as_trigger_vector(trigger_representation: object, n_features: int) -> np.ndarray:
    """Coerce a JSON-safe ``trigger_representation`` into a 1-D float vector.

    Accepts nested lists / arrays; flattens and right-sizes to ``n_features`` by
    truncating or zero-padding so a slightly mis-sized reversed trigger never
    crashes remediation.
    """
    arr = np.asarray(trigger_representation, dtype=float).ravel()
    if arr.size == n_features:
        return arr
    fixed = np.zeros(int(n_features), dtype=float)
    k = min(arr.size, n_features)
    fixed[:k] = arr[:k]
    return fixed


def trigger_mask(trigger_vector: np.ndarray, eps: float = _TRIGGER_EPS) -> np.ndarray:
    """Return a boolean mask of the feature channels carrying the trigger."""
    return np.abs(np.asarray(trigger_vector, dtype=float)) > eps


def stamp_trigger(X: np.ndarray, trigger_vector: np.ndarray) -> np.ndarray:
    """Stamp ``trigger_vector`` onto every row of ``X`` (returns a copy).

    Only the masked (non-zero) channels are overwritten, mirroring the additive
    BadNets patch used by ``ai/training/poison``.
    """
    X_stamped = np.asarray(X, dtype=float).copy()
    mask = trigger_mask(trigger_vector)
    if X_stamped.ndim != 2:
        raise ValueError(f"stamp_trigger expects a 2-D feature matrix, got shape {X_stamped.shape}")
    X_stamped[:, mask] = np.asarray(trigger_vector, dtype=float)[mask]
    return X_stamped
