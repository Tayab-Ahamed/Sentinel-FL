"""
ai/explainability/detection_explainer.py — Structured explanations for L1/L2/L3 flags.

Converts ``TrustLedgerEntry`` objects (and their ``evidence`` dicts) into
``DetectionExplanation`` objects — the uniform drilldown type backing
``Visualizer.explainability_drilldown()`` (INTERFACES.md).

Integration points:
    L1 (gradient / cosine): parse UpdateGuard evidence → feature importance
    L2 (model audit):       parse audit evidence → reversed trigger description
    L3 (STRIP / SHAP):      parse sentinel alert evidence → SHAP explanation

Public surface:
    DetectionExplainer
        explain_ledger_entry(entry, shap_explainer, model, clean_X)
            → DetectionExplanation
        explain_l1_flag(client_id, round_num, norm, cosine_sim,
                        cluster, delta, feature_names)
            → DetectionExplanation
        explain_l2_flag(entry_id, label, audit_evidence)
            → DetectionExplanation
        explain_l3_flag(entry_id, alert, shap_explanation)
            → DetectionExplanation
"""

from __future__ import annotations

import logging
from typing import Any

from ai.fl_core.schemas import (
    ChartArtifact,
    DetectionExplanation,
    FeatureImportanceResult,
    SHAPExplanation,
    TrustLedgerEntry,
)

logger = logging.getLogger(__name__)


class DetectionExplainer:
    """Produce DetectionExplanation objects from detection evidence.

    Args:
        chart_generator: Optional ChartGenerator instance.  If provided,
            charts are generated and embedded in the explanation.
        top_k_features: Number of top features to include in structured evidence.
    """

    def __init__(
        self,
        chart_generator: Any | None = None,
        top_k_features: int = 10,
    ) -> None:
        self._charts = chart_generator
        self._top_k = top_k_features

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def explain_ledger_entry(
        self,
        entry: TrustLedgerEntry,
        shap_explainer: Any | None = None,
        model: Any | None = None,
        clean_X: Any | None = None,
    ) -> DetectionExplanation:
        """Build a DetectionExplanation from any TrustLedgerEntry.

        Dispatches to the appropriate layer-specific method based on
        ``entry.layer_id``.

        Args:
            entry: The ledger entry to explain.
            shap_explainer: Optional SHAPExplainer for L3 inputs.
            model: Optional model for SHAP scoring.
            clean_X: Optional clean reference data for SHAP.

        Returns:
            DetectionExplanation.
        """
        layer = entry.layer_id
        evidence = entry.evidence or {}

        if layer == "L1":
            return self.explain_l1_flag(
                client_id=entry.subject_id,
                round_num=entry.round_num,
                norm=evidence.get("l2_norm"),
                cosine_sim=evidence.get("max_cosine_sim"),
                cluster=evidence.get("cluster_id"),
                delta=evidence.get("delta"),
                feature_names=evidence.get("feature_names"),
                entry_id=entry.entry_id,
                reason_string=entry.reason or "",
            )
        elif layer == "L2":
            return self.explain_l2_flag(
                entry_id=entry.entry_id,
                label=evidence.get("audited_label"),
                audit_evidence=evidence,
                reason_string=entry.reason or "",
                round_num=entry.round_num,
                subject_id=entry.subject_id,
            )
        elif layer == "L3":
            return self.explain_l3_flag(
                entry_id=entry.entry_id,
                alert_evidence=evidence,
                reason_string=entry.reason or "",
                round_num=entry.round_num,
                subject_id=entry.subject_id,
            )
        else:
            return DetectionExplanation(
                entry_id=entry.entry_id,
                layer_id=layer,
                subject_id=entry.subject_id,
                subject_type=entry.subject_type,
                round_num=entry.round_num,
                reason_string=entry.reason or "",
                structured_evidence=evidence,
            )

    # ------------------------------------------------------------------
    # Layer-specific builders
    # ------------------------------------------------------------------

    def explain_l1_flag(
        self,
        client_id: str,
        round_num: int | None = None,
        norm: float | None = None,
        cosine_sim: float | None = None,
        cluster: Any | None = None,
        delta: list[float] | None = None,
        feature_names: list[str] | None = None,
        entry_id: str = "",
        reason_string: str = "",
    ) -> DetectionExplanation:
        """Explain an L1 (Update Guard) flag event.

        Args:
            client_id: The flagged client.
            round_num: FL round number.
            norm: L2-norm of the update delta.
            cosine_sim: Maximum cosine similarity to another client.
            cluster: Cluster ID if collusion was detected.
            delta: Raw update delta for gradient importance.
            feature_names: Feature name list.
            entry_id: TrustLedgerEntry.entry_id.
            reason_string: Verbatim from UpdateGuard.

        Returns:
            DetectionExplanation for L1.
        """
        structured: dict[str, Any] = {}
        fi: FeatureImportanceResult | None = None
        charts: list[ChartArtifact] = []

        if norm is not None:
            structured["l2_norm"] = round(float(norm), 4)
        if cosine_sim is not None:
            structured["max_cosine_similarity"] = round(float(cosine_sim), 4)
        if cluster is not None:
            structured["cluster_id"] = cluster

        if delta is not None:
            from ai.explainability.feature_importance import gradient_feature_importance

            ctx = f"L1 round {round_num} client {client_id}"
            fi = gradient_feature_importance(
                delta=delta,
                feature_names=feature_names,
                context=ctx,
            )
            structured["top_gradient_features"] = fi.ranked_features[: self._top_k]

            if self._charts is not None:
                try:
                    chart = self._charts.feature_importance_chart(
                        fi, title=f"Gradient Importance — {client_id} r{round_num}"
                    )
                    charts.append(chart)
                except Exception as exc:
                    logger.debug("L1 chart generation failed: %s", exc)

        narrative = reason_string or (
            f"Client '{client_id}' flagged at round {round_num}. "
            + (f"L2-norm={norm:.4f}. " if norm else "")
            + (f"Max cosine-sim={cosine_sim:.4f}. " if cosine_sim else "")
            + (f"Cluster={cluster}. " if cluster is not None else "")
        )

        return DetectionExplanation(
            entry_id=entry_id or f"L1-{client_id}-r{round_num}",
            layer_id="L1",
            subject_id=client_id,
            subject_type="client",
            round_num=round_num,
            reason_string=narrative,
            structured_evidence=structured,
            feature_importance=fi,
            chart_artifacts=charts,
        )

    def explain_l2_flag(
        self,
        entry_id: str,
        label: int | None = None,
        audit_evidence: dict[str, Any] | None = None,
        reason_string: str = "",
        round_num: int | None = None,
        subject_id: str = "",
    ) -> DetectionExplanation:
        """Explain an L2 (Model Auditor) flag event.

        Args:
            entry_id: TrustLedgerEntry.entry_id.
            label: The audited class label.
            audit_evidence: Evidence dict from the ledger entry.
            reason_string: Verbatim from ModelAuditorDetector.
            round_num: FL round.
            subject_id: Subject identifier.

        Returns:
            DetectionExplanation for L2.
        """
        evidence = dict(audit_evidence or {})
        charts: list[ChartArtifact] = []

        narrative = reason_string or (
            f"Model auditor flagged label {label} at round {round_num}. "
            + ("Trigger reversal detected. " if evidence.get("trigger_reversed") else "")
            + (
                f"Confidence={evidence.get('confidence', ''):.4f}. "
                if isinstance(evidence.get("confidence"), float)
                else ""
            )
        )

        return DetectionExplanation(
            entry_id=entry_id,
            layer_id="L2",
            subject_id=subject_id or str(label),
            subject_type="label",
            round_num=round_num,
            reason_string=narrative,
            structured_evidence=evidence,
            chart_artifacts=charts,
        )

    def explain_l3_flag(
        self,
        entry_id: str,
        alert_evidence: dict[str, Any] | None = None,
        reason_string: str = "",
        round_num: int | None = None,
        subject_id: str = "",
        shap_explanation: SHAPExplanation | None = None,
    ) -> DetectionExplanation:
        """Explain an L3 (Runtime Sentinel) flag event.

        Args:
            entry_id: TrustLedgerEntry.entry_id.
            alert_evidence: Evidence dict from the SentinelAlert.
            reason_string: Verbatim from StripEntropyDetector.explain().
            round_num: FL round.
            subject_id: Input identifier.
            shap_explanation: Pre-computed SHAP explanation (optional).

        Returns:
            DetectionExplanation for L3.
        """
        evidence = dict(alert_evidence or {})
        charts: list[ChartArtifact] = []
        fused_score = evidence.get("fused_score") or evidence.get("score")

        narrative = reason_string or (
            f"Runtime Sentinel flagged input '{subject_id}' at round {round_num}. "
            + (f"Fused anomaly score={float(fused_score):.4f}. " if fused_score else "")
            + (f"Severity={evidence.get('alert_severity', 'unknown')}. ")
        )

        if shap_explanation is not None and self._charts is not None:
            try:
                chart = self._charts.shap_bar_chart(
                    shap_explanation,
                    title=f"SHAP Attribution — {subject_id} r{round_num}",
                )
                charts.append(chart)
            except Exception as exc:
                logger.debug("L3 SHAP chart generation failed: %s", exc)

        return DetectionExplanation(
            entry_id=entry_id,
            layer_id="L3",
            subject_id=subject_id,
            subject_type="input",
            round_num=round_num,
            reason_string=narrative,
            structured_evidence=evidence,
            shap_explanation=shap_explanation,
            chart_artifacts=charts,
        )
