"""
runtime_sentinel.py — L3: Runtime Sentinel (STRIP-style entropy detector).

Own implementation of the STRIP algorithm (Gao et al., ACSAC 2019, see RESEARCH.md
§1.1) for feature-vector data: perturb an incoming input by linearly blending it with
N randomly drawn clean held-out samples, run the blended inputs through the model, and
measure the Shannon entropy of the resulting predicted-class distribution. A trojaned
input's predictions collapse to the attacker's target class regardless of the blend
(low entropy); a clean input's predictions vary with the blend (high entropy).

This module implements Signal 1 only (entropy). Signal 2 (activation consistency) is
not implemented in this NumPy proof-of-concept because the linear-softmax model used
here has no hidden layer to probe; it is left as a TODO for the PyTorch/CNN production
path (ai/detection/activation_consistency.py) where a penultimate layer exists.

The pure functions (``entropy``, ``strip_score``, ``calibrate_boundary``, ``detect``)
are the Phase 0 NumPy proof-of-concept and are preserved exactly.  The
``StripEntropyDetector`` and ``RuntimeSentinelStrategy`` classes wrap them in the
Detector / DefenseStrategy interface contracts (INTERFACES.md) so the pipeline can
compose them by configuration rather than by calling functions directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ai.fl_core.exceptions import InsufficientCalibrationDataError
from ai.fl_core.interfaces import DefenseStrategy, Detector
from ai.fl_core.schemas import DetectionResult, TrustLedgerEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 0 pure functions (PRESERVED UNCHANGED from original implementation)
# ---------------------------------------------------------------------------


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector (bits).

    Args:
        probs: Probability vector or 1-D array; need not sum to 1 exactly.

    Returns:
        H(probs) in bits (base-2 logarithm).
    """
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def strip_score(
    model: Any,
    x: np.ndarray,
    clean_pool: np.ndarray,
    n_perturb: int = 50,
    rng: np.random.Generator | None = None,
) -> float:
    """Return the normalised entropy (STRIP's H, Eq. 4 in the paper) for one input x.

    Lower H => more likely trojaned.

    Args:
        model: Any object with a ``predict_proba(X)`` method.
        x: Single input feature vector, shape ``(n_features,)``.
        clean_pool: Pool of clean reference samples, shape ``(n, n_features)``.
        n_perturb: Number of blended perturbations to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        Mean Shannon entropy across all blended predictions.
    """
    rng = rng or np.random.default_rng()
    idx = rng.choice(len(clean_pool), size=n_perturb, replace=True)
    blended = 0.5 * x[None, :] + 0.5 * clean_pool[idx]
    probs = model.predict_proba(blended)
    ent_sum = sum(entropy(probs[i]) for i in range(n_perturb))
    return ent_sum / n_perturb


def calibrate_boundary(
    model: Any,
    clean_calib_pool: np.ndarray,
    clean_holdout: np.ndarray,
    target_frr: float = 0.01,
    n_perturb: int = 50,
) -> float:
    """Estimate a detection boundary from benign inputs only (no trojaned samples needed).

    Matches the paper's percentile-based calibration procedure: the boundary is
    the ``target_frr`` quantile of entropy scores computed on clean inputs.

    Args:
        model: Model with ``predict_proba(X)`` method.
        clean_calib_pool: Clean pool used for blending.
        clean_holdout: Clean inputs to score for calibration.
        target_frr: Desired false-rejection rate (e.g. 0.02 = 2%).
        n_perturb: Number of perturbations per input.

    Returns:
        Entropy boundary (float); inputs below this are flagged as trojaned.
    """
    rng = np.random.default_rng(1)
    scores = np.array(
        [
            strip_score(model, clean_holdout[i], clean_calib_pool, n_perturb, rng)
            for i in range(len(clean_holdout))
        ]
    )
    boundary = float(np.quantile(scores, target_frr))
    return boundary


def detect(
    model: Any,
    X: np.ndarray,
    clean_pool: np.ndarray,
    boundary: float,
    n_perturb: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a batch of inputs and return a boolean flag array.

    Args:
        model: Model with ``predict_proba(X)`` method.
        X: Batch of inputs to score, shape ``(n, n_features)``.
        clean_pool: Clean pool for blending.
        boundary: Calibrated entropy boundary.
        n_perturb: Perturbations per input.

    Returns:
        ``(flagged, scores)`` where ``flagged`` is a boolean array
        (True = trojaned) and ``scores`` contains the per-input entropy values.
    """
    rng = np.random.default_rng(2)
    scores = np.array([strip_score(model, X[i], clean_pool, n_perturb, rng) for i in range(len(X))])
    return scores <= boundary, scores


# ---------------------------------------------------------------------------
# Calibration state dataclass
# ---------------------------------------------------------------------------


@dataclass
class StripCalibrationState:
    """Opaque, serialisable calibration state for StripEntropyDetector."""

    boundary: float
    clean_pool: np.ndarray = field(repr=False)
    n_perturb: int = 50
    target_frr: float = 0.02


# ---------------------------------------------------------------------------
# Detector implementation
# ---------------------------------------------------------------------------


class StripEntropyDetector(Detector):
    """L3 Detector — STRIP-style entropy signal (Signal 1).

    Wraps the pure ``strip_score`` / ``calibrate_boundary`` / ``detect``
    functions in the Detector interface so the pipeline composes detectors
    by configuration.

    Args:
        n_perturb: Number of blended perturbations per input.
        target_frr: False-rejection rate for boundary calibration.
        min_calibration_samples: Minimum clean samples required to calibrate.
    """

    name: str = "strip_entropy"
    layer: str = "L3"

    def __init__(
        self,
        n_perturb: int = 50,
        target_frr: float = 0.02,
        min_calibration_samples: int = 30,
    ) -> None:
        self._n_perturb = n_perturb
        self._target_frr = target_frr
        self._min_calibration_samples = min_calibration_samples

    def calibrate(self, clean_reference_data: tuple[Any, np.ndarray]) -> StripCalibrationState:
        """Calibrate the entropy boundary from clean data.

        Args:
            clean_reference_data: Tuple of ``(model, clean_holdout_X)``
                where ``model`` has a ``predict_proba`` method and
                ``clean_holdout_X`` is a 2-D NumPy array.

        Returns:
            StripCalibrationState with the calibrated boundary.

        Raises:
            InsufficientCalibrationDataError: If fewer than
                ``min_calibration_samples`` clean samples are provided.
        """
        model, clean_X = clean_reference_data
        n = len(clean_X)
        if n < self._min_calibration_samples:
            raise InsufficientCalibrationDataError(
                n_provided=n,
                n_required=self._min_calibration_samples,
                detector_name=self.name,
            )
        calib_pool = clean_X[: max(1, n // 2)]
        holdout = clean_X[max(1, n // 2) :]
        boundary = calibrate_boundary(model, calib_pool, holdout, self._target_frr, self._n_perturb)
        return StripCalibrationState(
            boundary=boundary,
            clean_pool=calib_pool,
            n_perturb=self._n_perturb,
            target_frr=self._target_frr,
        )

    def score(
        self,
        input_or_model: tuple[Any, np.ndarray],
        calibration_state: StripCalibrationState,
    ) -> DetectionResult:
        """Score a single input against the calibrated boundary.

        Args:
            input_or_model: Tuple of ``(model, x)`` where ``x`` is a 1-D
                feature vector for one input.
            calibration_state: Previously returned by ``calibrate()``.

        Returns:
            DetectionResult with entropy score and flagged status.
        """
        model, x = input_or_model
        score_val = strip_score(model, x, calibration_state.clean_pool, calibration_state.n_perturb)
        flagged = score_val <= calibration_state.boundary
        return DetectionResult(
            detector_name=self.name,
            layer="L3",
            subject_id=str(id(x)),  # per-input ID; callers should override with a real ID
            score=score_val,
            flagged=flagged,
            boundary=calibration_state.boundary,
            round_num=None,
            explanation=self.explain(
                DetectionResult(
                    detector_name=self.name,
                    layer="L3",
                    subject_id=str(id(x)),
                    score=score_val,
                    flagged=flagged,
                    boundary=calibration_state.boundary,
                    round_num=None,
                )
            ),
        )

    def explain(self, detection_result: DetectionResult) -> str:
        """Return a human-readable explanation for the Trust Ledger."""
        sid = detection_result.subject_id
        score = detection_result.score
        boundary = detection_result.boundary
        flagged = detection_result.flagged
        return (
            f"[strip_entropy] Input '{sid}': mean entropy = {score:.4f} "
            f"({'FLAGGED' if flagged else 'clean'}, boundary = {boundary:.4f}). "
            "Low entropy indicates predictions collapse to one class across perturbations "
            "— consistent with a backdoor trigger."
        )


# ---------------------------------------------------------------------------
# DefenseStrategy implementation
# ---------------------------------------------------------------------------


class RuntimeSentinelStrategy(DefenseStrategy):
    """L3 DefenseStrategy — per-inference STRIP entropy detection.

    Orchestrates one or more L3 Detectors, fuses their scores via
    ``FusionClassifier``, generates ``SentinelAlert`` objects via
    ``AlertManager``, and writes ``TrustLedgerEntry`` records to L4.

    Implements ARCHITECTURE.md §2.3's full L3 pipeline:
      1. Score the input through all active Detectors (Signal 1 + Signal 2).
      2. Fuse scores using a lightweight logistic classifier.
      3. If fused_score > alert_threshold → generate and record an alert.
      4. Write a TrustLedgerEntry for L4.
      5. Emit a structured L3 log event.
      6. Return all ledger entries to the caller.

    Args:
        detectors: List of ``(detector, calibration_state)`` pairs.
        alert_manager: AlertManager for severity classification and history.
        fusion_classifier: FusionClassifier for signal fusion.
        ledger: Optional FileTrustLedger; if provided, entries are written
            directly instead of just returned.
        sentinel_logger: Optional structured Logger for L3 events.
        alert_threshold: Fused score above which an alert is generated.
        latency_budget_ms: Warning threshold for inference latency.
    """

    layer_id: str = "L3"

    def __init__(
        self,
        detectors: list[tuple[Detector, Any]] | None = None,
        alert_manager: Any | None = None,
        fusion_classifier: Any | None = None,
        ledger: Any | None = None,
        sentinel_logger: Any | None = None,
        alert_threshold: float = 0.5,
        latency_budget_ms: float = 50.0,
    ) -> None:
        from ai.detection.alert_manager import AlertManager
        from ai.detection.fusion_classifier import FusionClassifier

        self._detectors: list[tuple[Detector, Any]] = detectors or []
        self._alert_manager: Any = alert_manager or AlertManager(sentinel_logger=sentinel_logger)
        self._fusion: Any = fusion_classifier or FusionClassifier()
        self._ledger = ledger
        self._sentinel_logger = sentinel_logger
        self._alert_threshold = alert_threshold
        self._latency_budget_s = latency_budget_ms / 1000.0

        # Per-detector latency tracking
        self._detector_latencies: dict[str, list[float]] = {}
        self._total_inferences: int = 0
        self._flagged_inferences: int = 0

    # ------------------------------------------------------------------
    # Primary pipeline
    # ------------------------------------------------------------------

    def process(self, context: Any) -> list[TrustLedgerEntry]:
        """Score a single inference context through all active L3 detectors.

        Args:
            context: An ``InferenceContext`` (or any object with ``input_id``,
                ``input_data``, ``predicted_confidence``, ``round_num``).

        Returns:
            A list of ``TrustLedgerEntry`` objects to write to L4.
                Empty if no detectors are active or no flags raised.
        """
        import time

        import numpy as np

        if not self._detectors:
            logger.debug("RuntimeSentinelStrategy.process(): no detectors registered.")
            return []

        self._total_inferences += 1
        t_start = time.perf_counter()

        # Extract input array from context
        input_data = getattr(context, "input_data", None)
        if input_data is None:
            logger.warning("RuntimeSentinelStrategy.process(): context has no input_data.")
            return []
        x = np.asarray(input_data, dtype=np.float32)

        # Reconstruct (model, x) pair — model must be injected separately or
        # carried in context.model if the caller sets it.
        model = getattr(context, "model", None)
        if model is None:
            logger.debug("RuntimeSentinelStrategy.process(): no model in context — skipping score.")
            return []

        # -----------------------------------------------------------
        # Step 1: Score through each detector
        # -----------------------------------------------------------
        detection_results: list[DetectionResult] = []
        s1_score: float | None = None
        s2_score: float = 0.0

        for detector, cal_state in self._detectors:
            d_start = time.perf_counter()
            try:
                result = detector.score((model, x), cal_state)
                # Override subject_id with the real input_id from context
                result = result.model_copy(
                    update={"subject_id": getattr(context, "input_id", result.subject_id)}
                )
                detection_results.append(result)
                # Map detector results to signals
                if detector.name == "strip_entropy":
                    s1_score = result.score
                elif detector.name == "activation_consistency":
                    s2_score = float(result.score) if result.score is not None else 0.0
            except Exception as exc:
                logger.warning(
                    "RuntimeSentinelStrategy: detector '%s' failed: %s",
                    getattr(detector, "name", str(detector)),
                    exc,
                )
            finally:
                d_elapsed = time.perf_counter() - d_start
                name = getattr(detector, "name", "unknown")
                self._detector_latencies.setdefault(name, []).append(d_elapsed)

        # -----------------------------------------------------------
        # Step 2: Fuse signals
        # -----------------------------------------------------------
        if s1_score is None:
            # No entropy score available — use max raw score from any detector
            s1_score = max((dr.score for dr in detection_results), default=0.0)
        fused_score = float(self._fusion.predict(s1_score, s2_score))

        # -----------------------------------------------------------
        # Step 3: Alert generation
        # -----------------------------------------------------------
        entries: list[TrustLedgerEntry] = []
        if fused_score >= self._alert_threshold or any(dr.flagged for dr in detection_results):
            self._flagged_inferences += 1
            alert = self._alert_manager.create_alert(
                detection_results=detection_results,
                fused_score=fused_score,
                context=context,
            )
            self._alert_manager.record_alert(alert)

            # -----------------------------------------------------------
            # Step 4: Write Trust Ledger entries
            # -----------------------------------------------------------
            ledger_entry = self._alert_manager.to_ledger_entry(alert)
            entries.append(ledger_entry)
            if self._ledger is not None:
                try:
                    self._ledger.add_entry(ledger_entry)
                except Exception as exc:
                    logger.warning("RuntimeSentinelStrategy: ledger write failed: %s", exc)

        # -----------------------------------------------------------
        # Step 5: Structured logging
        # -----------------------------------------------------------
        t_elapsed = time.perf_counter() - t_start
        if t_elapsed > self._latency_budget_s:
            logger.warning(
                "RuntimeSentinelStrategy: latency %.1fms exceeds budget %.1fms for input '%s'.",
                t_elapsed * 1000,
                self._latency_budget_s * 1000,
                getattr(context, "input_id", "?"),
            )
        self._emit_inference_event(context, fused_score, bool(entries))

        return entries

    # ------------------------------------------------------------------
    # Registration and calibration helpers
    # ------------------------------------------------------------------

    def add_detector(self, detector: Detector, calibration_state: Any) -> None:
        """Register a new detector with its calibration state.

        Args:
            detector: An object implementing the Detector interface.
            calibration_state: The result of ``detector.calibrate()``.
        """
        self._detectors.append((detector, calibration_state))
        logger.info(
            "RuntimeSentinelStrategy: registered detector '%s'.",
            getattr(detector, "name", str(detector)),
        )

    def calibrate_all(self, model: Any, clean_X: Any) -> None:
        """Calibrate all registered detectors using clean data.

        Replaces the existing calibration states in-place.  Detectors that
        raise ``UnsupportedModelError`` or ``InsufficientCalibrationDataError``
        are removed from the active list and logged as warnings.

        Args:
            model: Model with a ``predict_proba(X)`` method.
            clean_X: Clean reference data (2-D numpy array).
        """
        from ai.fl_core.exceptions import InsufficientCalibrationDataError, UnsupportedModelError

        calibrated: list[tuple[Detector, Any]] = []
        for detector, _ in self._detectors:
            try:
                cal_state = detector.calibrate((model, clean_X))
                calibrated.append((detector, cal_state))
                logger.info(
                    "RuntimeSentinelStrategy: calibrated detector '%s'.",
                    getattr(detector, "name", str(detector)),
                )
            except (UnsupportedModelError, InsufficientCalibrationDataError) as exc:
                logger.warning(
                    "RuntimeSentinelStrategy: detector '%s' excluded (calibration failed): %s",
                    getattr(detector, "name", str(detector)),
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "RuntimeSentinelStrategy: calibrate_all: unexpected error for '%s': %s",
                    getattr(detector, "name", str(detector)),
                    exc,
                )
        self._detectors = calibrated

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        """Return per-detector latency stats and overall detection metrics.

        Returns:
            Dict with ``total_inferences``, ``flagged_inferences``,
            ``flag_rate``, ``active_detectors``, ``per_detector_latency_ms``.
        """
        latency_ms: dict[str, dict[str, float]] = {}
        for name, times in self._detector_latencies.items():
            if times:
                arr = [t * 1000 for t in times]
                latency_ms[name] = {
                    "mean_ms": round(sum(arr) / len(arr), 3),
                    "max_ms": round(max(arr), 3),
                    "n_calls": len(arr),
                }
        return {
            "total_inferences": self._total_inferences,
            "flagged_inferences": self._flagged_inferences,
            "flag_rate": round(self._flagged_inferences / self._total_inferences, 4)
            if self._total_inferences > 0
            else 0.0,
            "active_detectors": [getattr(d, "name", str(d)) for d, _ in self._detectors],
            "per_detector_latency_ms": latency_ms,
            "alert_manager_stats": self._alert_manager.stats(),
            "fusion_is_calibrated": self._fusion.is_calibrated,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit_inference_event(self, context: Any, fused_score: float, flagged: bool) -> None:
        if self._sentinel_logger is None:
            return
        try:
            self._sentinel_logger.log(
                "L3",
                "inference_scored",
                {
                    "input_id": getattr(context, "input_id", "unknown"),
                    "round_num": getattr(context, "round_num", None),
                    "fused_score": fused_score,
                    "flagged": flagged,
                    "total_inferences": self._total_inferences,
                },
            )
        except Exception as exc:
            logger.debug("RuntimeSentinelStrategy: event log failed: %s", exc)
