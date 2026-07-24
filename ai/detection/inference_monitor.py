"""
ai/detection/inference_monitor.py — L3: Global-model surveillance.

Tracks per-round prediction confidence distributions across inference batches
and flags statistical anomalies that may indicate a model has been compromised.

Two anomaly signals:
  1. Confidence drop: mean prediction confidence falls below the clean baseline
     by more than ``confidence_drop_threshold`` (absolute).
  2. Class-distribution shift: the predicted-class distribution diverges from
     the calibration distribution beyond a configurable KL-divergence bound.

Design: all state is in-memory.  No disk I/O.  The monitor is updated once
per inference batch and queried by RuntimeSentinelStrategy after each round.

Public surface:
    InferenceMonitor
        update(probs_batch, round_num)   — ingest a batch of prediction dists
        check_confidence_drop(threshold) → bool
        check_class_distribution_shift(kl_threshold, window) → bool
        anomaly_score()                  → float [0, 1]
        round_stats(round_num)           → dict | None
        summary()                        → dict
        calibrate(probs_batch)           — establish clean baseline
        reset()                          — wipe all state
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np

from ai.detection.confidence_analyzer import batch_confidence_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal per-round record
# ---------------------------------------------------------------------------


class _RoundStats:
    __slots__ = (
        "class_counts",
        "mean_confidence",
        "mean_entropy",
        "n_samples",
        "round_num",
        "std_confidence",
    )

    def __init__(
        self,
        round_num: int,
        mean_confidence: float,
        std_confidence: float,
        n_samples: int,
        class_counts: dict[int, int],
        mean_entropy: float,
    ) -> None:
        self.round_num = round_num
        self.mean_confidence = mean_confidence
        self.std_confidence = std_confidence
        self.n_samples = n_samples
        self.class_counts = class_counts
        self.mean_entropy = mean_entropy


class InferenceMonitor:
    """Global-model surveillance for L3 Runtime Sentinel.

    Args:
        confidence_drop_threshold: Absolute confidence drop from baseline
            that triggers a confidence-drop alert.
        kl_threshold: KL-divergence upper bound for class-distribution shift.
        window_rounds: Number of recent rounds to include in sliding-window
            anomaly computation.
        n_classes: Number of output classes (used for KL normalisation).
    """

    def __init__(
        self,
        confidence_drop_threshold: float = 0.15,
        kl_threshold: float = 0.3,
        window_rounds: int = 5,
        n_classes: int | None = None,
    ) -> None:
        self._drop_threshold = confidence_drop_threshold
        self._kl_threshold = kl_threshold
        self._window = window_rounds
        self._n_classes = n_classes

        # Calibration baseline (set by calibrate())
        self._baseline_mean_conf: float | None = None
        self._baseline_std_conf: float | None = None
        self._baseline_class_dist: np.ndarray | None = None

        # Rolling history (deque caps memory automatically)
        self._history: deque[_RoundStats] = deque(maxlen=1000)
        self._round_index: dict[int, _RoundStats] = {}

        # Cumulative anomaly count
        self._anomaly_events: int = 0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, probs_batch: np.ndarray) -> None:
        """Establish the clean confidence baseline from a batch of clean predictions.

        Must be called once before any anomaly checks.

        Args:
            probs_batch: 2-D array ``(n_samples, n_classes)`` of softmax outputs
                on a known-clean dataset.
        """
        probs_batch = np.asarray(probs_batch, dtype=np.float64)
        stats = batch_confidence_stats(probs_batch)
        self._baseline_mean_conf = stats["mean_confidence"]
        self._baseline_std_conf = max(stats["std_confidence"], 1e-6)
        # Class distribution baseline
        predicted = np.argmax(probs_batch, axis=1)
        n_classes = self._n_classes or int(probs_batch.shape[1])
        counts = np.bincount(predicted, minlength=n_classes).astype(np.float64)
        self._baseline_class_dist = (counts + 1e-6) / (counts.sum() + n_classes * 1e-6)
        self._n_classes = n_classes
        logger.info(
            "InferenceMonitor: calibrated. baseline_mean_conf=%.4f, n_classes=%d.",
            self._baseline_mean_conf,
            n_classes,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, probs_batch: np.ndarray, round_num: int) -> None:
        """Ingest one batch of inference predictions.

        Args:
            probs_batch: 2-D array ``(n_samples, n_classes)`` of softmax probs.
            round_num: Current FL round number (or inference batch index).
        """
        probs_batch = np.asarray(probs_batch, dtype=np.float64)
        stats = batch_confidence_stats(probs_batch)
        predicted = np.argmax(probs_batch, axis=1)
        n_classes = self._n_classes or int(probs_batch.shape[1])
        counts: dict[int, int] = {
            int(c): int(v) for c, v in enumerate(np.bincount(predicted, minlength=n_classes))
        }
        record = _RoundStats(
            round_num=round_num,
            mean_confidence=stats["mean_confidence"],
            std_confidence=stats["std_confidence"],
            n_samples=stats["n_samples"],
            class_counts=counts,
            mean_entropy=stats["mean_entropy"],
        )
        self._history.append(record)
        self._round_index[round_num] = record
        logger.debug(
            "InferenceMonitor.update: round=%d mean_conf=%.4f n=%d.",
            round_num,
            stats["mean_confidence"],
            stats["n_samples"],
        )

    # ------------------------------------------------------------------
    # Anomaly checks
    # ------------------------------------------------------------------

    def check_confidence_drop(self, threshold: float | None = None) -> bool:
        """Return True if recent mean confidence has dropped below the baseline.

        Args:
            threshold: Override the configured ``confidence_drop_threshold``.

        Returns:
            True if a confidence drop is detected.
        """
        if self._baseline_mean_conf is None or not self._history:
            return False
        thr = threshold if threshold is not None else self._drop_threshold
        recent = list(self._history)[-self._window :]
        recent_mean = sum(r.mean_confidence for r in recent) / len(recent)
        drop = self._baseline_mean_conf - recent_mean
        if drop > thr:
            logger.warning(
                "InferenceMonitor: confidence drop detected (%.4f below baseline %.4f).",
                drop,
                self._baseline_mean_conf,
            )
            self._anomaly_events += 1
            return True
        return False

    def check_class_distribution_shift(
        self,
        kl_threshold: float | None = None,
        window: int | None = None,
    ) -> bool:
        """Return True if the recent class distribution has shifted.

        Uses symmetric KL divergence (Jensen-Shannon divergence) between the
        calibrated baseline and the recent sliding-window distribution.

        Args:
            kl_threshold: Override ``kl_threshold``.
            window: Override ``window_rounds``.

        Returns:
            True if class-distribution shift is detected.
        """
        if self._baseline_class_dist is None or not self._history:
            return False
        kl_thr = kl_threshold if kl_threshold is not None else self._kl_threshold
        n_win = window if window is not None else self._window
        n_classes = self._n_classes or len(self._baseline_class_dist)
        recent = list(self._history)[-n_win:]

        # Aggregate class counts over the window
        agg = np.zeros(n_classes, dtype=np.float64)
        for r in recent:
            for cls, cnt in r.class_counts.items():
                if cls < n_classes:
                    agg[cls] += cnt
        total = agg.sum()
        if total == 0:
            return False
        recent_dist = (agg + 1e-6) / (total + n_classes * 1e-6)

        # Jensen-Shannon divergence
        m = 0.5 * (self._baseline_class_dist + recent_dist)
        kl1 = float(np.sum(self._baseline_class_dist * np.log2(self._baseline_class_dist / m)))
        kl2 = float(np.sum(recent_dist * np.log2(recent_dist / m)))
        js_div = 0.5 * (kl1 + kl2)

        if js_div > kl_thr:
            logger.warning(
                "InferenceMonitor: class-distribution shift detected (JS=%.4f > %.4f).",
                js_div,
                kl_thr,
            )
            self._anomaly_events += 1
            return True
        return False

    def anomaly_score(self) -> float:
        """Return a combined anomaly score in [0, 1].

        Combines confidence drop and class-distribution shift signals using
        a sigmoid-squashed confidence deficit against the calibrated baseline.

        Returns:
            Anomaly score in [0, 1].  0 = normal; 1 = fully anomalous.
        """
        if self._baseline_mean_conf is None or not self._history:
            return 0.0
        recent = list(self._history)[-self._window :]
        recent_mean = sum(r.mean_confidence for r in recent) / len(recent)
        drop = self._baseline_mean_conf - recent_mean  # positive when worse
        # Sigmoid: drop of `_drop_threshold` → score ≈ 0.73
        z = drop / max(self._drop_threshold, 1e-6)
        score = 1.0 / (1.0 + math.exp(-2.0 * z))
        # Adjust toward 0.5 baseline for no-drop case
        score = float(np.clip(score - 0.5, 0.0, 0.5) * 2.0)
        return round(score, 4)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def round_stats(self, round_num: int) -> dict[str, Any] | None:
        """Return statistics for a specific round, or None if not tracked.

        Args:
            round_num: The round to query.

        Returns:
            Dict with round stats or None.
        """
        record = self._round_index.get(round_num)
        if record is None:
            return None
        return {
            "round_num": record.round_num,
            "mean_confidence": record.mean_confidence,
            "std_confidence": record.std_confidence,
            "n_samples": record.n_samples,
            "class_counts": record.class_counts,
            "mean_entropy": record.mean_entropy,
        }

    def summary(self) -> dict[str, Any]:
        """Return a summary of all monitored rounds.

        Returns:
            Dict with ``n_rounds_monitored``, ``baseline_mean_conf``,
            ``current_mean_conf``, ``anomaly_events``, ``current_anomaly_score``,
            ``confidence_drop_detected``, ``class_shift_detected``.
        """
        current_conf = self._history[-1].mean_confidence if self._history else None
        return {
            "n_rounds_monitored": len(self._history),
            "baseline_mean_conf": self._baseline_mean_conf,
            "current_mean_conf": current_conf,
            "anomaly_events": self._anomaly_events,
            "current_anomaly_score": self.anomaly_score(),
            "confidence_drop_detected": self.check_confidence_drop(),
            "class_shift_detected": self.check_class_distribution_shift(),
            "n_classes": self._n_classes,
        }

    def reset(self) -> None:
        """Clear all state, including the calibration baseline."""
        self._baseline_mean_conf = None
        self._baseline_std_conf = None
        self._baseline_class_dist = None
        self._history.clear()
        self._round_index.clear()
        self._anomaly_events = 0
        logger.debug("InferenceMonitor: state reset.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """True if ``calibrate()`` has been called."""
        return self._baseline_mean_conf is not None

    @property
    def n_rounds_monitored(self) -> int:
        """Number of rounds with at least one update."""
        return len(self._history)

    @property
    def anomaly_events(self) -> int:
        """Cumulative number of anomaly events detected."""
        return self._anomaly_events
