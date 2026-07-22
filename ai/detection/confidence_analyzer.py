"""
ai/detection/confidence_analyzer.py — L3: Prediction confidence analysis utilities.

Pure, stateless functions used by InferenceMonitor and RuntimeSentinelStrategy
to extract confidence signals from model prediction distributions.

Functions:
    softmax_confidence(logits)            → float   (max softmax probability)
    top2_margin(probs)                    → float   (top-1 minus top-2 prob)
    entropy_from_probs(probs)             → float   (Shannon entropy, bits)
    confidence_anomaly_score(c, ref_mean, ref_std) → float [0, 1]
    batch_confidence_stats(probs_batch)   → dict
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Core confidence extractors
# ---------------------------------------------------------------------------


def softmax_confidence(logits: np.ndarray) -> float:
    """Return the maximum softmax probability from raw logits.

    Numerically stable via the log-sum-exp trick.

    Args:
        logits: 1-D array of raw model outputs (before softmax).

    Returns:
        Max softmax probability in [0, 1].
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    return float(np.max(probs))


def top2_margin(probs: np.ndarray) -> float:
    """Return the margin between the top-1 and top-2 predicted probabilities.

    A high margin indicates high confidence in the top prediction.
    A low margin (close to 0) indicates near-tie uncertainty.

    Args:
        probs: 1-D probability vector.  Need not sum to exactly 1.

    Returns:
        top1_prob - top2_prob in [0, 1].  Returns 1.0 for single-class models.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    if len(p) < 2:
        return 1.0
    sorted_p = np.sort(p)[::-1]
    return float(sorted_p[0] - sorted_p[1])


def entropy_from_probs(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector (bits).

    Numerically stable via clipping.  Identical semantics to the Phase-0
    ``entropy()`` function in ``runtime_sentinel.py`` but accepts both
    softmax outputs and raw logits (if they happen to sum to ≈1).

    Args:
        probs: 1-D probability vector; need not sum to exactly 1.

    Returns:
        H(probs) in bits (base-2 logarithm).
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def confidence_anomaly_score(
    confidence: float,
    reference_mean: float,
    reference_std: float,
    clip_z: float = 4.0,
) -> float:
    """Map a confidence value to a [0, 1] anomaly score via sigmoid-squashed z-score.

    A confidence much lower than the reference mean produces a high anomaly score.
    Works for both softmax confidence and top2_margin signals.

    Args:
        confidence: The confidence value for the current input.
        reference_mean: Expected (clean) mean confidence.
        reference_std: Expected (clean) standard deviation.  If 0, returns 0.
        clip_z: Z-score clipping bound before sigmoid.

    Returns:
        Anomaly score in [0, 1].  Higher = more anomalous (lower confidence
        than expected).
    """
    if reference_std < 1e-9:
        return 0.0
    z = (reference_mean - confidence) / reference_std  # positive when confidence is low
    z_clipped = float(np.clip(z, -clip_z, clip_z))
    return float(1.0 / (1.0 + math.exp(-z_clipped)))


# ---------------------------------------------------------------------------
# Batch statistics helper
# ---------------------------------------------------------------------------


def batch_confidence_stats(probs_batch: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a batch of prediction distributions.

    Args:
        probs_batch: 2-D array of shape ``(n_samples, n_classes)``.

    Returns:
        Dict with keys: ``mean_confidence``, ``std_confidence``,
        ``mean_entropy``, ``mean_top2_margin``, ``min_confidence``,
        ``max_confidence``, ``n_samples``.
    """
    probs_batch = np.asarray(probs_batch, dtype=np.float64)
    if probs_batch.ndim == 1:
        probs_batch = probs_batch[None, :]
    n = len(probs_batch)

    confidences = np.max(probs_batch, axis=1)
    entropies = np.array([entropy_from_probs(probs_batch[i]) for i in range(n)])
    margins = np.array([top2_margin(probs_batch[i]) for i in range(n)])

    return {
        "n_samples": n,
        "mean_confidence": round(float(np.mean(confidences)), 6),
        "std_confidence": round(float(np.std(confidences)), 6),
        "min_confidence": round(float(np.min(confidences)), 6),
        "max_confidence": round(float(np.max(confidences)), 6),
        "mean_entropy": round(float(np.mean(entropies)), 6),
        "mean_top2_margin": round(float(np.mean(margins)), 6),
    }
