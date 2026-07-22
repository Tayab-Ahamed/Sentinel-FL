"""
ai/explainability/attack_explainer.py — Attack characterisation from detection evidence.

Maps attack configuration + multi-layer detection evidence into human-readable
``AttackExplanation`` objects.  Answers questions a judge or operator would ask:
  - What type of attack was this?
  - Which label was targeted?
  - What does the trigger look like?
  - What fraction of data was poisoned?
  - Which round did it start?
  - Which clients are suspected?
  - How confident is the detection?

Public surface:
    AttackExplainer
        explain_backdoor(attack_config, detection_results, ledger_entries)
            → AttackExplanation
        trigger_description(trigger_type, trigger_value, target_label)
            → str
        poison_ratio_analysis(n_poisoned, n_total)
            → dict
"""

from __future__ import annotations

import logging
from typing import Any

from ai.fl_core.schemas import AttackExplanation, ChartArtifact

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trigger catalogue
# ---------------------------------------------------------------------------

_TRIGGER_TEMPLATES: dict[str, str] = {
    "pixel_block": (
        "A fixed-value pixel block (value={value}) is injected into a contiguous "
        "spatial region of the input, replacing the original pixels. "
        "Target label: {target}."
    ),
    "blended": (
        "A blended pattern (blend_ratio={value}) is superimposed onto the input "
        "additively.  The trigger is invisible at low ratios. "
        "Target label: {target}."
    ),
    "sinusoidal": (
        "A sinusoidal frequency-domain pattern is added to the input. "
        "Value (amplitude)={value}. Target label: {target}."
    ),
    "unknown": (
        "Trigger type unknown.  Detection evidence suggests a backdoor directed "
        "at label {target}."
    ),
}


# ---------------------------------------------------------------------------
# AttackExplainer
# ---------------------------------------------------------------------------


class AttackExplainer:
    """Produce AttackExplanation objects from attack config and detection evidence.

    Args:
        chart_generator: Optional ChartGenerator.
        high_confidence_threshold: Detection confidence above which the
            attack explanation is reported as high-confidence.
    """

    def __init__(
        self,
        chart_generator: Any | None = None,
        high_confidence_threshold: float = 0.7,
    ) -> None:
        self._charts = chart_generator
        self._high_conf = high_confidence_threshold

    # ------------------------------------------------------------------
    # Explain backdoor
    # ------------------------------------------------------------------

    def explain_backdoor(
        self,
        attack_config: dict[str, Any] | None = None,
        detection_results: list[Any] | None = None,
        ledger_entries: list[Any] | None = None,
    ) -> AttackExplanation:
        """Build an AttackExplanation from attack config + detection evidence.

        Args:
            attack_config: Dict from ``AttackReport`` or ``configs/attack.yaml``
                with keys ``attack_type``, ``target_label``, ``trigger_type``,
                ``trigger_value``, ``poison_fraction``.
            detection_results: List of ``DetectionResult`` from any layer.
            ledger_entries: List of ``TrustLedgerEntry`` flagged entries.

        Returns:
            AttackExplanation.
        """
        cfg = attack_config or {}
        results = detection_results or []
        entries = ledger_entries or []

        attack_type = str(cfg.get("attack_type", "unknown"))
        target_label = cfg.get("target_label")
        trigger_type = str(cfg.get("trigger_type", "unknown"))
        trigger_value = cfg.get("trigger_value")
        poison_fraction = cfg.get("poison_fraction")
        infection_round = cfg.get("infection_round") or cfg.get("start_round")

        # Aggregate detection confidence
        conf = self._aggregate_confidence(results, entries)

        # Suspected clients from ledger
        suspected = list({
            e.get("subject_id") if isinstance(e, dict) else getattr(e, "subject_id", "")
            for e in entries
            if (e.get("subject_type") if isinstance(e, dict) else
                getattr(e, "subject_type", "")) == "client"
        })

        # Trigger description
        trigger_desc = self.trigger_description(trigger_type, trigger_value, target_label)

        # Evidence summary
        evidence_summary = self._build_evidence_summary(
            attack_type, target_label, conf, len(results), len(entries), suspected
        )

        charts: list[ChartArtifact] = []

        return AttackExplanation(
            attack_type=attack_type,
            target_label=int(target_label) if target_label is not None else None,
            trigger_description=trigger_desc,
            poison_fraction=float(poison_fraction) if poison_fraction is not None else None,
            estimated_infection_round=(
                int(infection_round) if infection_round is not None else None
            ),
            suspected_clients=suspected,
            detection_confidence=round(float(conf), 4),
            evidence_summary=evidence_summary,
            chart_artifacts=charts,
        )

    # ------------------------------------------------------------------
    # Trigger description
    # ------------------------------------------------------------------

    @staticmethod
    def trigger_description(
        trigger_type: str,
        trigger_value: Any | None = None,
        target_label: int | None = None,
    ) -> str:
        """Generate a human-readable trigger description.

        Args:
            trigger_type: E.g. ``'pixel_block'``, ``'blended'``, ``'sinusoidal'``.
            trigger_value: The trigger's numeric value or ratio.
            target_label: The attack target class.

        Returns:
            Formatted description string.
        """
        template = _TRIGGER_TEMPLATES.get(trigger_type, _TRIGGER_TEMPLATES["unknown"])
        val_str = f"{trigger_value:.4f}" if isinstance(trigger_value, float) else str(trigger_value)
        return template.format(value=val_str, target=target_label)

    # ------------------------------------------------------------------
    # Poison ratio analysis
    # ------------------------------------------------------------------

    @staticmethod
    def poison_ratio_analysis(n_poisoned: int, n_total: int) -> dict[str, Any]:
        """Analyse the poison fraction and assess attack severity.

        Args:
            n_poisoned: Number of poisoned training samples.
            n_total: Total training samples.

        Returns:
            Dict with ``poison_fraction``, ``severity``, ``clean_samples``,
            ``risk_assessment``.
        """
        if n_total <= 0:
            return {
                "poison_fraction": 0.0,
                "severity": "unknown",
                "clean_samples": 0,
                "n_poisoned": 0,
                "n_total": n_total,
                "risk_assessment": "Cannot assess: n_total is 0.",
            }
        fraction = n_poisoned / n_total
        clean = n_total - n_poisoned

        if fraction < 0.05:
            severity = "low"
            risk = (
                f"Only {fraction:.1%} of data is poisoned — below the typical "
                "5% threshold for reliable backdoor learning.  Attack may be ineffective."
            )
        elif fraction < 0.20:
            severity = "medium"
            risk = (
                f"{fraction:.1%} of data poisoned — within the range where most "
                "backdoor attacks achieve reliable ASR with sufficient triggers."
            )
        else:
            severity = "high"
            risk = (
                f"{fraction:.1%} of data poisoned — extremely high ratio.  "
                "Likely detectable by L1 norm outlier analysis."
            )

        return {
            "poison_fraction": round(fraction, 4),
            "n_poisoned": n_poisoned,
            "clean_samples": clean,
            "n_total": n_total,
            "severity": severity,
            "risk_assessment": risk,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_confidence(
        self, results: list[Any], entries: list[Any]
    ) -> float:
        """Compute aggregated detection confidence across all results."""
        scores = []
        for r in results:
            s = r.get("score") if isinstance(r, dict) else getattr(r, "score", None)
            if s is not None:
                scores.append(float(s))
        for e in entries:
            s = e.get("score") if isinstance(e, dict) else getattr(e, "score", None)
            if s is not None:
                scores.append(float(s))
        if not scores:
            return 0.0
        return float(min(1.0, sum(scores) / len(scores)))

    def _build_evidence_summary(
        self,
        attack_type: str,
        target_label: Any,
        confidence: float,
        n_results: int,
        n_entries: int,
        suspected: list[str],
    ) -> str:
        conf_label = "HIGH" if confidence >= self._high_conf else "MODERATE" if confidence >= 0.4 else "LOW"
        parts = [
            f"Attack type: {attack_type}.",
            f"Target label: {target_label}.",
            f"Detection confidence: {confidence:.4f} ({conf_label}).",
            f"Evidence: {n_results} detector result(s), {n_entries} ledger flag(s).",
        ]
        if suspected:
            parts.append(
                f"Suspected clients: {', '.join(suspected[:5])}"
                + (" ..." if len(suspected) > 5 else ".")
            )
        return " ".join(parts)
