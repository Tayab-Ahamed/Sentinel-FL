"""
ai/detection/fusion_classifier.py — L3: Signal fusion classifier.

Implements the logistic fusion of Signal 1 (STRIP entropy) and Signal 2
(activation consistency) described in ARCHITECTURE.md §2.3:

    "a lightweight logistic classifier over both signals, trained on the
    L2 audit's confirmed clean/flagged history — meaning the runtime detector
    improves over the course of the challenge as L2 produces more labeled
    examples."

When fewer than ``min_labels`` L2-confirmed examples are available, the
classifier is not yet calibrated and falls back to a configurable weighted
average (defaulting to Signal 1 only, since Signal 2 requires a PyTorch CNN).

Public surface:
    FusionClassifier
        fit(s1_scores, s2_scores, labels)  — train on L2 audit history
        predict(s1, s2)                    → float fused score in [0, 1]
        predict_batch(s1_arr, s2_arr)      → np.ndarray of fused scores
        is_calibrated                      → bool
        reset()                            — clear trained state
        save_state()                       → dict (JSON-serialisable)
        load_state(state)                  — restore from dict
        n_training_examples                → int
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FusionClassifier:
    """Lightweight logistic fusion over Signal 1 (entropy) + Signal 2 (activation).

    Args:
        signal1_weight: Weight for Signal 1 in the weighted-average fallback.
            Signal 2 gets ``1 - signal1_weight``.  Effective only when not
            calibrated.
        min_labels: Minimum number of L2-confirmed labels before logistic
            training is attempted.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        signal1_weight: float = 1.0,
        min_labels: int = 10,
        random_state: int = 42,
    ) -> None:
        if not 0.0 <= signal1_weight <= 1.0:
            raise ValueError(f"signal1_weight must be in [0, 1], got {signal1_weight}")
        self._w1 = float(signal1_weight)
        self._w2 = 1.0 - self._w1
        self._min_labels = min_labels
        self._rng = random_state
        self._is_calibrated: bool = False
        self._clf: Any | None = None  # sklearn LogisticRegression once trained
        self._n_examples: int = 0
        # History for potential re-training
        self._s1_history: list[float] = []
        self._s2_history: list[float] = []
        self._label_history: list[int] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        s1_scores: list[float] | np.ndarray,
        s2_scores: list[float] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> None:
        """Train the logistic classifier on L2-confirmed examples.

        Args:
            s1_scores: Signal 1 (entropy) scores in [0, ∞).  Lower = more
                suspicious for STRIP-style detectors.
            s2_scores: Signal 2 (activation consistency) scores in [0, 1].
                Higher = more suspicious.
            labels: Binary labels: 1 = trojaned/flagged, 0 = clean.

        Note:
            Appends to internal history so calling ``fit()`` multiple times
            (once per L2 audit) is idempotent and cumulative.
        """
        s1 = np.asarray(s1_scores, dtype=np.float64).ravel()
        s2 = np.asarray(s2_scores, dtype=np.float64).ravel()
        lbl = np.asarray(labels, dtype=np.int32).ravel()

        if not (len(s1) == len(s2) == len(lbl)):
            raise ValueError(
                f"s1_scores ({len(s1)}), s2_scores ({len(s2)}), labels ({len(lbl)}) "
                "must have the same length."
            )

        self._s1_history.extend(s1.tolist())
        self._s2_history.extend(s2.tolist())
        self._label_history.extend(lbl.tolist())
        self._n_examples = len(self._label_history)

        if self._n_examples < self._min_labels:
            logger.debug(
                "FusionClassifier: %d examples available, need %d to calibrate.",
                self._n_examples, self._min_labels,
            )
            return

        if len(set(self._label_history)) < 2:
            logger.debug(
                "FusionClassifier: only one class in labels — cannot calibrate logistic."
            )
            return

        self._train_logistic()

    def _train_logistic(self) -> None:
        """Fit the internal LogisticRegression on accumulated history."""
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            logger.warning("FusionClassifier: sklearn not available; staying in fallback mode.")
            return

        X = np.column_stack([self._s1_history, self._s2_history])
        y = np.array(self._label_history, dtype=np.int32)
        clf = LogisticRegression(
            max_iter=500,
            random_state=self._rng,
            class_weight="balanced",
        )
        clf.fit(X, y)
        self._clf = clf
        self._is_calibrated = True
        logger.info(
            "FusionClassifier: trained on %d examples. is_calibrated=True.",
            len(y),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, s1: float, s2: float = 0.0) -> float:
        """Return a fused anomaly score in [0, 1].

        Args:
            s1: Signal 1 (entropy) score.  For STRIP, lower = more suspicious.
               The classifier internally normalises; pass the raw entropy value.
            s2: Signal 2 (activation consistency) score in [0, 1].
               Defaults to 0.0 when Signal 2 is unavailable.

        Returns:
            Fused score in [0, 1].  Higher = more suspicious.
        """
        if self._is_calibrated and self._clf is not None:
            X = np.array([[s1, s2]])
            proba = self._clf.predict_proba(X)
            # Class ordering: [0=clean, 1=flagged]
            class_order = list(self._clf.classes_)
            if 1 in class_order:
                flagged_idx = class_order.index(1)
                return float(proba[0, flagged_idx])
            return float(proba[0, 1])
        # Fallback weighted average: invert s1 (entropy → suspicion)
        # Normalise s1 using a typical max-entropy bound of log2(n_classes)
        # Default: treat s1 as already a [0, 1] suspicion score
        s1_norm = float(np.clip(1.0 - s1, 0.0, 1.0))  # low entropy = high suspicion
        s2_norm = float(np.clip(s2, 0.0, 1.0))
        return float(self._w1 * s1_norm + self._w2 * s2_norm)

    def predict_batch(
        self,
        s1_scores: list[float] | np.ndarray,
        s2_scores: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Score a batch of inputs.

        Args:
            s1_scores: Array of Signal 1 scores.
            s2_scores: Array of Signal 2 scores.  If None, all zeros.

        Returns:
            1-D float32 array of fused scores in [0, 1].
        """
        s1 = np.asarray(s1_scores, dtype=np.float64).ravel()
        s2_arr = (
            np.zeros_like(s1)
            if s2_scores is None
            else np.asarray(s2_scores, dtype=np.float64).ravel()
        )
        return np.array(
            [self.predict(float(s1[i]), float(s2_arr[i])) for i in range(len(s1))],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all training history and return to fallback mode."""
        self._clf = None
        self._is_calibrated = False
        self._n_examples = 0
        self._s1_history.clear()
        self._s2_history.clear()
        self._label_history.clear()
        logger.debug("FusionClassifier: state reset.")

    def save_state(self) -> dict[str, Any]:
        """Return a JSON-serialisable state snapshot.

        Returns:
            Dict with ``is_calibrated``, ``n_examples``, ``s1_history``,
            ``s2_history``, ``label_history``, and ``signal1_weight``.
        """
        return {
            "is_calibrated": self._is_calibrated,
            "n_examples": self._n_examples,
            "signal1_weight": self._w1,
            "min_labels": self._min_labels,
            "s1_history": self._s1_history.copy(),
            "s2_history": self._s2_history.copy(),
            "label_history": self._label_history.copy(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore from a ``save_state()`` snapshot.

        Args:
            state: Dict previously returned by ``save_state()``.
        """
        self._w1 = float(state.get("signal1_weight", self._w1))
        self._w2 = 1.0 - self._w1
        self._min_labels = int(state.get("min_labels", self._min_labels))
        self._s1_history = list(state.get("s1_history", []))
        self._s2_history = list(state.get("s2_history", []))
        self._label_history = list(state.get("label_history", []))
        self._n_examples = len(self._label_history)
        self._is_calibrated = False
        self._clf = None
        if self._n_examples >= self._min_labels and len(set(self._label_history)) >= 2:
            self._train_logistic()
        logger.debug("FusionClassifier: state loaded (%d examples).", self._n_examples)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """True if the logistic classifier has been trained."""
        return self._is_calibrated

    @property
    def n_training_examples(self) -> int:
        """Total number of L2-confirmed examples seen so far."""
        return self._n_examples
