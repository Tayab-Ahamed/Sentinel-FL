"""
ai/detection/anomaly_detector.py — Statistical anomaly scoring for L1 Update Guard.

``UpdateAnomalyDetector`` scores each client's parameter delta against a
reference distribution fitted on honest (clean) rounds, producing a scalar
anomaly score ∈ [0, 1] per client.

Two scoring methods are supported, selectable via ``anomaly_method``:

  ``"zscore"``
      Primary signal: |z-score of L2 norm| normalized to [0, 1] via a
      sigmoid-like squashing function.  Fast, no calibration data needed
      (uses within-round statistics).  Degrades with very small n_clients.

  ``"mad"``
      Uses Median Absolute Deviation instead of standard deviation.
      More robust when n_clients ≤ 8 or norms are heavy-tailed.

Both methods return scores in [0, 1]:
  - 0.0 → perfectly normal (update indistinguishable from peers)
  - 1.0 → extreme outlier

The ``fit()`` method accumulates reference norms from earlier rounds so the
detector's baseline improves over time.  Before any ``fit()`` call (i.e. in
round 1), within-round statistics are used directly.

Design: no external dependencies beyond NumPy.
"""

from __future__ import annotations

import logging

import numpy as np

from ai.detection.norm_calculator import (
    compute_l2_norms,
    compute_norm_mad_scores,
    compute_norm_zscores,
    flag_norm_outliers,
)

logger = logging.getLogger(__name__)

_EPSILON = 1e-12


class UpdateAnomalyDetector:
    """Statistical anomaly detector for FL client update vectors.

    Args:
        anomaly_method: Scoring method — ``"zscore"`` or ``"mad"``.
        norm_type: Which norm to use for update magnitude — ``"l2"``.
            (Currently only L2 is used; kept for future extension.)
        threshold_z: Score threshold above which a client is flagged.
            Applied to the raw z-score / MAD score, not to [0,1]-squashed values.
    """

    def __init__(
        self,
        anomaly_method: str = "zscore",
        norm_type: str = "l2",
        threshold_z: float = 3.0,
    ) -> None:
        if anomaly_method not in ("zscore", "mad"):
            raise ValueError(f"anomaly_method must be 'zscore' or 'mad', got {anomaly_method!r}")
        self._method = anomaly_method
        self._norm_type = norm_type
        self._threshold_z = threshold_z
        # Reference norms accumulated via fit()
        self._reference_norms: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, reference_deltas: list[np.ndarray]) -> None:
        """Accumulate baseline norms from a set of reference (clean) updates.

        Can be called repeatedly; new norms are appended to the reference pool.

        Args:
            reference_deltas: Per-client delta vectors from a clean round.
        """
        if not reference_deltas:
            return
        norms = compute_l2_norms(reference_deltas)
        self._reference_norms.extend(norms.tolist())
        logger.debug(
            "UpdateAnomalyDetector.fit: added %d norms (pool size=%d)",
            len(norms),
            len(self._reference_norms),
        )

    def score_all(self, deltas: list[np.ndarray]) -> np.ndarray:
        """Compute anomaly scores for all clients in the current round.

        Scores are in [0, 1]:  0 = normal, 1 = extreme outlier.

        Args:
            deltas: Per-client delta vectors for the current round.

        Returns:
            1-D float32 array of shape ``(n_clients,)``.
        """
        if not deltas:
            return np.array([], dtype=np.float32)

        norms = compute_l2_norms(deltas)

        # Use reference pool if available; otherwise use within-round stats.
        if len(self._reference_norms) >= 3:
            ref = np.array(self._reference_norms, dtype=np.float32)
            # Compute stats on the reference distribution
            ref_mean = float(np.mean(ref))
            ref_std = float(np.std(ref))
            ref_median = float(np.median(ref))
            ref_mad = float(np.median(np.abs(ref - ref_median)))
            if self._method == "zscore":
                raw_scores = np.abs(norms - ref_mean) / max(ref_std, _EPSILON)
            else:
                raw_scores = np.abs(norms - ref_median) / max(ref_mad, _EPSILON)
        else:
            # No reference data yet — use within-round statistics
            if self._method == "zscore":
                raw_scores = np.abs(compute_norm_zscores(norms))
            else:
                raw_scores = compute_norm_mad_scores(norms)

        # Squash to [0, 1] via sigmoid: s = 1 - exp(-raw/threshold)
        scores = (1.0 - np.exp(-raw_scores / max(self._threshold_z, _EPSILON))).astype(np.float32)
        return np.clip(scores, 0.0, 1.0)

    def score(self, delta: np.ndarray) -> float:
        """Compute anomaly score for a single client update.

        Args:
            delta: Single flat delta vector.

        Returns:
            Anomaly score in [0, 1].
        """
        return float(self.score_all([delta])[0])

    def flag(
        self,
        deltas: list[np.ndarray],
        threshold_z: float | None = None,
    ) -> np.ndarray:
        """Return boolean mask of anomalous clients.

        Uses the raw norm outlier flagging (not the squashed scores) for
        precision — the threshold_z applies directly to z-scores / MAD scores.

        Args:
            deltas: Per-client delta vectors.
            threshold_z: Override threshold (defaults to ``self._threshold_z``).

        Returns:
            Boolean array of shape ``(n_clients,)``; True = anomalous.
        """
        if not deltas:
            return np.array([], dtype=bool)
        norms = compute_l2_norms(deltas)
        thr = threshold_z if threshold_z is not None else self._threshold_z
        return flag_norm_outliers(norms, threshold_z=thr, method=self._method)

    def reset_reference(self) -> None:
        """Clear the accumulated reference norm pool."""
        self._reference_norms.clear()
        logger.debug("UpdateAnomalyDetector: reference pool cleared.")

    @property
    def method(self) -> str:
        """The anomaly scoring method in use."""
        return self._method

    @property
    def threshold_z(self) -> float:
        """The flagging threshold."""
        return self._threshold_z

    @property
    def reference_pool_size(self) -> int:
        """Number of reference norms accumulated via ``fit()``."""
        return len(self._reference_norms)
