"""
ai/detection/activation_consistency.py — L3 Signal 2: Activation Consistency Detector.

Implements the Detector interface (INTERFACES.md §Detector) for the second,
adaptive-attack-resistant signal described in ARCHITECTURE.md §2.3:

    Track whether the *penultimate-layer activation pattern* of perturbed inputs
    stays consistent the way a trojaned input's does — this remains informative
    even under STRIP's documented entropy-manipulation adaptive attack, because it
    looks at internal representations, not just output entropy.

Signal 2 requires a PyTorch model with a named penultimate layer.  On the
NumPy Phase 0 proof-of-concept (LinearSoftmaxModel) there is no hidden layer to
probe, so this detector raises UnsupportedModelError at registration time,
exactly as specified in INTERFACES.md §Detector.

Status: Skeleton — full implementation in Milestone 6 (Phase 1, PyTorch CNN).
"""

from __future__ import annotations

import logging
from typing import Any

from ai.fl_core.exceptions import UnsupportedModelError
from ai.fl_core.interfaces import Detector
from ai.fl_core.schemas import DetectionResult

logger = logging.getLogger(__name__)


class ActivationConsistencyDetector(Detector):
    """L3 Signal 2 — penultimate-layer activation-consistency detector.

    Args:
        n_perturb: Number of blended perturbations per input (same as STRIP).
        penultimate_layer_name: Name of the PyTorch layer to hook for activations.
    """

    name: str = "activation_consistency"
    layer: str = "L3"

    def __init__(
        self,
        n_perturb: int = 50,
        penultimate_layer_name: str = "penultimate",
    ) -> None:
        self._n_perturb = n_perturb
        self._layer_name = penultimate_layer_name

    # ------------------------------------------------------------------
    # Detector interface
    # ------------------------------------------------------------------

    def calibrate(self, clean_reference_data: Any) -> Any:
        """Estimate the activation-variance boundary from clean data.

        Raises:
            UnsupportedModelError: Always on Phase 0 (no PyTorch model available).

        Status: Not yet implemented — Milestone 6.
        """
        raise UnsupportedModelError(
            self.name,
            "Activation consistency requires a PyTorch CNN with a penultimate layer. "
            "Not available in Phase 0 (NumPy linear model). "
            "Will be enabled in Milestone 6 once the PyTorch training path is active.",
        )

    def score(self, input_or_model: Any, calibration_state: Any) -> DetectionResult:
        """Score one input using penultimate-layer activation variance.

        Raises:
            UnsupportedModelError: Always on Phase 0.

        Status: Not yet implemented — Milestone 6.
        """
        raise UnsupportedModelError(
            self.name,
            "score() requires a PyTorch CNN. Not available in Phase 0.",
        )

    def explain(self, detection_result: DetectionResult) -> str:
        """Return a human-readable explanation of the detection.

        Status: Not yet implemented — Milestone 6.
        """
        score = detection_result.score
        flagged = detection_result.flagged
        sid = detection_result.subject_id
        return (
            f"[activation_consistency] Input '{sid}': "
            f"activation variance = {score:.4f} "
            f"({'FLAGGED' if flagged else 'clean'}, boundary = {detection_result.boundary:.4f}). "
            "High consistency across perturbations indicates a potential trojaned input."
        )
