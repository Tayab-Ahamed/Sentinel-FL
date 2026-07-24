"""
ai/detection/norm_calculator.py — Update-norm computation for L1 Update Guard.

Provides vectorised norm computation over a batch of per-client delta vectors,
followed by statistical outlier detection.  All functions are pure (no side
effects) and operate on NumPy arrays so they are trivially testable.

Norm-based anomaly detection rationale:
  BadNets and gradient-inversion attacks commonly produce updates whose L2
  norm is significantly larger than honest clients' updates (the attacker
  amplifies their delta to overcome honest averaging).  Flagging clients
  beyond ``threshold_z`` standard deviations of the round's norm distribution
  is a lightweight first filter applied before the costlier cosine-similarity
  clustering step.

  Reference: Blanchard et al. "Machine Learning with Adversaries: Byzantine
  Tolerant Gradient Descent" (NeurIPS 2017) — norm bounding as pre-filter.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-client norm computation
# ---------------------------------------------------------------------------


def compute_l2_norms(deltas: list[np.ndarray]) -> np.ndarray:
    """Compute the L2 (Euclidean) norm of each delta vector.

    Args:
        deltas: Per-client flat delta arrays (all must have the same length).

    Returns:
        1-D float32 array of shape ``(n_clients,)``.
    """
    return np.array([float(np.linalg.norm(d)) for d in deltas], dtype=np.float32)


def compute_l1_norms(deltas: list[np.ndarray]) -> np.ndarray:
    """Compute the L1 norm (sum of absolute values) of each delta vector.

    Args:
        deltas: Per-client flat delta arrays.

    Returns:
        1-D float32 array of shape ``(n_clients,)``.
    """
    return np.array([float(np.sum(np.abs(d))) for d in deltas], dtype=np.float32)


def compute_linf_norms(deltas: list[np.ndarray]) -> np.ndarray:
    """Compute the L∞ norm (max absolute value) of each delta vector.

    Args:
        deltas: Per-client flat delta arrays.

    Returns:
        1-D float32 array of shape ``(n_clients,)``.
    """
    return np.array(
        [float(np.max(np.abs(d))) if len(d) > 0 else 0.0 for d in deltas],
        dtype=np.float32,
    )


def compute_norms(
    deltas: list[np.ndarray],
    norm_type: str = "l2",
) -> np.ndarray:
    """Dispatch to the correct norm function by name.

    Args:
        deltas: Per-client flat delta arrays.
        norm_type: ``"l2"``, ``"l1"``, or ``"linf"``.

    Returns:
        1-D float32 norm array.

    Raises:
        ValueError: For unknown norm types.
    """
    if norm_type == "l2":
        return compute_l2_norms(deltas)
    if norm_type == "l1":
        return compute_l1_norms(deltas)
    if norm_type == "linf":
        return compute_linf_norms(deltas)
    raise ValueError(f"Unknown norm_type: {norm_type!r}. Use 'l2', 'l1', or 'linf'.")


# ---------------------------------------------------------------------------
# Statistical outlier detection
# ---------------------------------------------------------------------------


def compute_norm_zscores(norms: np.ndarray) -> np.ndarray:
    """Compute z-scores for a vector of client norms.

    z_i = (norm_i − mean) / std

    If all norms are identical (std = 0), returns a zero array.

    Args:
        norms: 1-D float32 array of per-client norms.

    Returns:
        1-D float32 z-score array of the same shape.
    """
    if len(norms) == 0:
        return np.array([], dtype=np.float32)
    mean = float(np.mean(norms))
    std = float(np.std(norms))
    if std < 1e-12:
        logger.debug("compute_norm_zscores: std ≈ 0; all clients have identical norms.")
        return np.zeros_like(norms, dtype=np.float32)
    return ((norms - mean) / std).astype(np.float32)


def compute_norm_mad_scores(norms: np.ndarray) -> np.ndarray:
    """Compute Median Absolute Deviation (MAD) scores for client norms.

    mad_score_i = |norm_i − median| / (MAD + ε)

    More robust than z-scores when n_clients is small or norms are
    heavy-tailed.

    Args:
        norms: 1-D float32 array of per-client norms.

    Returns:
        1-D float32 MAD-score array.
    """
    if len(norms) == 0:
        return np.array([], dtype=np.float32)
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    if mad < 1e-12:
        logger.debug("compute_norm_mad_scores: MAD ≈ 0; returning absolute deviations.")
        return np.abs(norms - median).astype(np.float32)
    return (np.abs(norms - median) / mad).astype(np.float32)


def flag_norm_outliers(
    norms: np.ndarray,
    threshold_z: float = 3.0,
    method: str = "zscore",
) -> np.ndarray:
    """Return a boolean mask of clients flagged as norm outliers.

    Args:
        norms: 1-D float32 per-client norm array.
        threshold_z: Score threshold.  Clients with score > threshold_z are flagged.
        method: ``"zscore"`` or ``"mad"``.

    Returns:
        Boolean array of shape ``(n_clients,)``; True = flagged.
    """
    if method == "zscore":
        scores = compute_norm_zscores(norms)
    elif method == "mad":
        scores = compute_norm_mad_scores(norms)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'zscore' or 'mad'.")

    flagged = np.abs(scores) > threshold_z
    n_flagged = int(flagged.sum())
    if n_flagged > 0:
        logger.info(
            "NormCalculator: flagged %d/%d clients as norm outliers (method=%s, threshold=%.1f)",
            n_flagged,
            len(norms),
            method,
            threshold_z,
        )
    return flagged


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------


def norm_summary(
    client_ids: list[str],
    norms: np.ndarray,
    zscores: np.ndarray,
    flagged: np.ndarray,
) -> list[dict]:
    """Build a JSON-serialisable per-client norm summary for logging.

    Args:
        client_ids: Client ID strings.
        norms: Per-client norms.
        zscores: Per-client z-scores (or MAD scores).
        flagged: Per-client outlier boolean mask.

    Returns:
        List of dicts with keys ``client_id``, ``norm``, ``zscore``, ``flagged``.
    """
    return [
        {
            "client_id": cid,
            "norm": round(float(n), 6),
            "zscore": round(float(z), 4),
            "flagged": bool(f),
        }
        for cid, n, z, f in zip(client_ids, norms, zscores, flagged)
    ]
