"""
ai/detection/model_auditor.py — L2: Model Auditor.

Implements the Detector and DefenseStrategy interfaces for periodic, offline
Neural-Cleanse-style trigger reverse-engineering with a within-cohort relative
outlier test (ARCHITECTURE.md §2.2).

Key design decisions (see RESEARCH.md §1.2 and ARCHITECTURE.md §2.2):
  - Replaces global MAD outlier detection with a within-cohort relative test
    (compare each label to a dynamically re-sampled reference subset across rounds),
    which keeps working when many labels are simultaneously infected — Neural
    Cleanse's documented failure mode.
  - Cross-checks L3 runtime detections: if L3 has been flagging inputs near a
    particular predicted class, L2 prioritises auditing that label first.
  - Uses PyTorch autograd (Adam) for trigger optimisation, matching Neural
    Cleanse's own optimizer choice (TECH_STACK.md).

Status: Skeleton — full implementation in Milestone 7 (Phase 1).
"""

from __future__ import annotations

import logging
from typing import Any

from ai.fl_core.interfaces import DefenseStrategy, Detector
from ai.fl_core.schemas import DetectionResult, TrustLedgerEntry

logger = logging.getLogger(__name__)


class ModelAuditorDetector(Detector):
    """L2 Detector — Neural-Cleanse-style per-label trigger reverse engineering.

    Args:
        n_classes: Number of output classes to audit.
        audit_lr: Adam learning rate for trigger optimisation.
        audit_steps: Number of optimisation steps per label.
        outlier_threshold: Anomaly-index threshold for within-cohort relative test.
    """

    name: str = "neural_cleanse_audit"
    layer: str = "L2"

    def __init__(
        self,
        n_classes: int = 4,
        audit_lr: float = 0.01,
        audit_steps: int = 100,
        outlier_threshold: float = 2.0,
    ) -> None:
        self._n_classes = n_classes
        self._audit_lr = audit_lr
        self._audit_steps = audit_steps
        self._outlier_threshold = outlier_threshold

    # ------------------------------------------------------------------
    # Detector interface
    # ------------------------------------------------------------------

    def calibrate(self, clean_reference_data: Any) -> Any:
        """Build the within-cohort reference distribution from clean data.

        Status: Not yet implemented — Milestone 7.
        """
        raise NotImplementedError(
            "ModelAuditorDetector.calibrate() will be implemented in Milestone 7."
        )

    def score(self, input_or_model: Any, calibration_state: Any) -> DetectionResult:
        """Reverse-engineer a trigger for one label and return its anomaly score.

        Status: Not yet implemented — Milestone 7.
        """
        raise NotImplementedError(
            "ModelAuditorDetector.score() will be implemented in Milestone 7."
        )

    def explain(self, detection_result: DetectionResult) -> str:
        """Return a human-readable explanation of the L2 audit result."""
        label = detection_result.subject_id
        score = detection_result.score
        flagged = detection_result.flagged
        return (
            f"[neural_cleanse_audit] Label '{label}': "
            f"reversed-trigger L1 norm = {score:.4f} "
            f"({'FLAGGED' if flagged else 'clean'}, "
            f"within-cohort anomaly boundary = {detection_result.boundary:.4f}). "
            "A small L1 norm indicates a minimal trigger was found — consistent with backdoor."
        )


class ModelAuditorStrategy(DefenseStrategy):
    """L2 DefenseStrategy — orchestrates periodic offline model audits.

    Runs every ``audit_interval_rounds`` rounds, prioritises labels flagged
    by L3, and writes AuditReports + TrustLedgerEntries to L4.

    Args:
        detector: Configured ModelAuditorDetector instance.
        audit_interval_rounds: How often (in rounds) to run the audit.
        l3_flagged_labels_fn: Optional callable returning the set of label IDs
            that L3 has flagged recently (for priority ordering).
    """

    layer_id: str = "L2"

    def __init__(
        self,
        detector: ModelAuditorDetector,
        audit_interval_rounds: int = 5,
        l3_flagged_labels_fn: Any = None,
    ) -> None:
        self._detector = detector
        self._interval = audit_interval_rounds
        self._l3_flagged_labels_fn = l3_flagged_labels_fn

    def process(self, context: Any) -> list[TrustLedgerEntry]:
        """Run the L2 audit if this round is an audit round.

        Status: Not yet implemented — Milestone 7.
        """
        raise NotImplementedError(
            "ModelAuditorStrategy.process() will be implemented in Milestone 7."
        )
