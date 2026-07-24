"""
ai/detection/alert_manager.py — L3: Alert generation and lifecycle management.

Converts detector verdicts + fused scores into structured ``SentinelAlert``
objects, assigns severity, deduplicates, and manages the alert history.
Provides ``to_ledger_entry()`` to write flags into L4 Trust Ledger.

Severity mapping (configurable via constructor):
    fused_score < low_medium_boundary            → "low"
    low_medium_boundary ≤ score < med_high_bound → "medium"
    score ≥ med_high_boundary                    → "high"

Public surface:
    AlertManager
        create_alert(detection_results, fused_score, context) → SentinelAlert
        record_alert(alert)
        get_active_alerts(min_severity)           → list[SentinelAlert]
        get_alert_history(limit)                  → list[SentinelAlert]
        alert_rate(window_size)                   → float
        clear_old_alerts(max_age_rounds, current_round)
        to_ledger_entry(alert)                    → TrustLedgerEntry
        stats()                                   → dict
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ai.fl_core.schemas import DetectionResult, SentinelAlert, TrustLedgerEntry

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


class AlertManager:
    """Alert lifecycle manager for the L3 Runtime Sentinel.

    Args:
        low_medium_boundary: Fused score below which severity is ``"low"``.
        med_high_boundary: Fused score at or above which severity is ``"high"``.
        max_history: Maximum alerts to keep in memory.
        sentinel_logger: Optional structured logger (emits L3 events).
    """

    def __init__(
        self,
        low_medium_boundary: float = 0.4,
        med_high_boundary: float = 0.7,
        max_history: int = 1000,
        sentinel_logger: Any | None = None,
    ) -> None:
        self._low_med = low_medium_boundary
        self._med_high = med_high_boundary
        self._history: deque[SentinelAlert] = deque(maxlen=max_history)
        self._logger = sentinel_logger
        self._total_created: int = 0
        self._severity_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}

    # ------------------------------------------------------------------
    # Alert creation
    # ------------------------------------------------------------------

    def create_alert(
        self,
        detection_results: list[DetectionResult],
        fused_score: float,
        context: Any,  # InferenceContext
        round_num: int | None = None,
    ) -> SentinelAlert:
        """Build a SentinelAlert from a set of detector verdicts.

        Args:
            detection_results: List of DetectionResult objects from active L3 detectors.
            fused_score: The fused anomaly score in [0, 1].
            context: InferenceContext (used to extract input_id and confidence).
            round_num: Override round number (extracted from context if None).

        Returns:
            A populated SentinelAlert.
        """
        severity = self._classify_severity(fused_score)
        flagged = any(dr.flagged for dr in detection_results)
        input_id = getattr(context, "input_id", "unknown")
        confidence = getattr(context, "predicted_confidence", None)
        rnd = round_num if round_num is not None else getattr(context, "round_num", None)

        # Aggregate explanation from all detectors
        explanations = [dr.explanation for dr in detection_results if dr.explanation]
        combined = (
            " | ".join(explanations)
            if explanations
            else (f"[sentinel] Input '{input_id}': fused_score={fused_score:.4f} ({severity})")
        )

        alert = SentinelAlert(
            input_id=input_id,
            round_num=rnd,
            detector_verdicts=detection_results,
            fused_score=round(float(fused_score), 4),
            flagged=flagged,
            alert_severity=severity,
            confidence_at_flag=confidence,
            explanation=combined,
        )
        self._total_created += 1
        return alert

    def _classify_severity(self, fused_score: float) -> str:
        """Map a fused score to a severity string."""
        if fused_score >= self._med_high:
            return "high"
        if fused_score >= self._low_med:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Recording and retrieval
    # ------------------------------------------------------------------

    def record_alert(self, alert: SentinelAlert) -> None:
        """Persist an alert in the in-memory history and emit a log event.

        Args:
            alert: Alert to record.
        """
        self._history.append(alert)
        self._severity_counts[alert.alert_severity] = (
            self._severity_counts.get(alert.alert_severity, 0) + 1
        )
        logger.info(
            "AlertManager: [%s] Input '%s' flagged (fused=%.4f round=%s).",
            alert.alert_severity.upper(),
            alert.input_id,
            alert.fused_score,
            alert.round_num,
        )
        self._emit_l3_event(alert)

    def get_active_alerts(self, min_severity: str = "low") -> list[SentinelAlert]:
        """Return all recorded alerts at or above the given severity.

        Args:
            min_severity: ``"low"``, ``"medium"``, or ``"high"``.

        Returns:
            Filtered list of SentinelAlert objects, most recent first.
        """
        min_rank = _SEVERITY_ORDER.get(min_severity, 0)
        return [
            a
            for a in reversed(self._history)
            if _SEVERITY_ORDER.get(a.alert_severity, 0) >= min_rank
        ]

    def get_alert_history(self, limit: int | None = None) -> list[SentinelAlert]:
        """Return alert history, most recent first.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of SentinelAlert objects.
        """
        result = list(reversed(self._history))
        if limit is not None:
            result = result[:limit]
        return result

    # ------------------------------------------------------------------
    # Rate and cleanup
    # ------------------------------------------------------------------

    def alert_rate(self, window_size: int = 100) -> float:
        """Fraction of the most recent ``window_size`` alerts that are flagged.

        Args:
            window_size: Number of recent alerts to consider.

        Returns:
            Alert rate in [0, 1].
        """
        recent = list(self._history)[-window_size:]
        if not recent:
            return 0.0
        return sum(1 for a in recent if a.flagged) / len(recent)

    def clear_old_alerts(self, max_age_rounds: int, current_round: int) -> int:
        """Remove alerts older than ``max_age_rounds`` from history.

        Args:
            max_age_rounds: Alerts with round_num < current_round - max_age_rounds
                are removed.
            current_round: The current FL round number.

        Returns:
            Number of alerts removed.
        """
        cutoff = current_round - max_age_rounds
        before = len(self._history)
        filtered = [a for a in self._history if a.round_num is None or a.round_num >= cutoff]
        self._history = deque(filtered, maxlen=self._history.maxlen)
        removed = before - len(self._history)
        if removed:
            logger.debug("AlertManager: removed %d old alerts (cutoff round %d).", removed, cutoff)
        return removed

    # ------------------------------------------------------------------
    # Trust Ledger integration
    # ------------------------------------------------------------------

    def to_ledger_entry(self, alert: SentinelAlert) -> TrustLedgerEntry:
        """Convert a SentinelAlert into a TrustLedgerEntry for L4.

        Args:
            alert: The alert to convert.

        Returns:
            TrustLedgerEntry with subject_type="input".
        """
        return TrustLedgerEntry(
            layer_id="L3",
            subject_type="input",
            subject_id=alert.input_id,
            round_num=alert.round_num,
            score=alert.fused_score,
            reason=alert.explanation,
            evidence={
                "alert_id": alert.alert_id,
                "alert_severity": alert.alert_severity,
                "confidence_at_flag": alert.confidence_at_flag,
                "detector_count": len(alert.detector_verdicts),
                "flagged_by": [dr.detector_name for dr in alert.detector_verdicts if dr.flagged],
            },
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for this AlertManager.

        Returns:
            Dict with ``total_created``, ``total_recorded``, ``severity_counts``,
            ``alert_rate_last_100``.
        """
        return {
            "total_created": self._total_created,
            "total_recorded": len(self._history),
            "severity_counts": dict(self._severity_counts),
            "alert_rate_last_100": round(self.alert_rate(100), 4),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit_l3_event(self, alert: SentinelAlert) -> None:
        if self._logger is None:
            return
        try:
            self._logger.log(
                "L3",
                "input_flagged",
                {
                    "alert_id": alert.alert_id,
                    "input_id": alert.input_id,
                    "round_num": alert.round_num,
                    "fused_score": alert.fused_score,
                    "severity": alert.alert_severity,
                    "explanation": alert.explanation[:200],
                },
            )
        except Exception as exc:
            logger.debug("AlertManager: L3 log failed (non-fatal): %s", exc)
