"""
ai/explainability/trust_explainer.py — Client reputation and Trust Ledger explanations.

Reads the ``FileTrustLedger`` to produce ``TrustExplanation`` objects:
  - explain_trust_score: current score + top contributing flags
  - explain_reputation_trajectory: per-round score history
  - rank_clients_by_suspicion: ordered list of all client explanations

Public surface:
    TrustExplainer
        explain_trust_score(client_id, ledger, suspicious_threshold)
            → TrustExplanation
        explain_reputation_trajectory(client_id, ledger, n_rounds)
            → TrustExplanation
        rank_clients_by_suspicion(ledger, top_k, suspicious_threshold)
            → list[TrustExplanation]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ai.fl_core.schemas import ChartArtifact, TrustExplanation

logger = logging.getLogger(__name__)


def _flag_count_narrative(n_flags: int, client_id: str) -> str:
    if n_flags == 0:
        return f"Client '{client_id}' has no recorded flags. Trust score is high."
    noun = "flag" if n_flags == 1 else "flags"
    return (
        f"Client '{client_id}' has accumulated {n_flags} {noun} across detection layers."
    )


class TrustExplainer:
    """Produce TrustExplanation objects from a FileTrustLedger.

    Args:
        suspicious_threshold: Score below which a client is considered suspicious.
        chart_generator: Optional ChartGenerator for trajectory charts.
        top_contributing_k: Number of top ledger entries to include.
    """

    def __init__(
        self,
        suspicious_threshold: float = 0.5,
        chart_generator: Any | None = None,
        top_contributing_k: int = 5,
    ) -> None:
        self._threshold = suspicious_threshold
        self._charts = chart_generator
        self._top_k = top_contributing_k

    # ------------------------------------------------------------------
    # Core explain methods
    # ------------------------------------------------------------------

    def explain_trust_score(
        self,
        client_id: str,
        ledger: Any,  # FileTrustLedger
        suspicious_threshold: float | None = None,
    ) -> TrustExplanation:
        """Explain the current trust score of a client.

        Args:
            client_id: The client to explain.
            ledger: FileTrustLedger instance.
            suspicious_threshold: Override the constructor threshold.

        Returns:
            TrustExplanation with narrative, layer breakdown, and top flags.
        """
        threshold = suspicious_threshold if suspicious_threshold is not None else self._threshold

        # Fetch all entries for this client
        history = self._get_client_history(client_id, ledger)
        current_score = self._current_score(client_id, ledger)
        is_suspicious = current_score < threshold

        layer_breakdown = self._layer_breakdown(history)
        top_entries = self._top_entries(history)

        total_flags = len(history)
        narrative = self._build_narrative(
            client_id, current_score, is_suspicious, total_flags, layer_breakdown
        )

        charts: list[ChartArtifact] = []

        return TrustExplanation(
            client_id=client_id,
            current_score=round(current_score, 4),
            is_suspicious=is_suspicious,
            narrative=narrative,
            score_trajectory=[],  # lightweight version — use trajectory method for full
            top_contributing_entries=top_entries,
            layer_breakdown=layer_breakdown,
            chart_artifacts=charts,
        )

    def explain_reputation_trajectory(
        self,
        client_id: str,
        ledger: Any,
        n_rounds: int | None = None,
    ) -> TrustExplanation:
        """Explain score evolution over rounds.

        Args:
            client_id: Client to trace.
            ledger: FileTrustLedger.
            n_rounds: If set, only include the last n_rounds.

        Returns:
            TrustExplanation with per-round score_trajectory.
        """
        base = self.explain_trust_score(client_id, ledger)
        history = self._get_client_history(client_id, ledger)

        # Build per-round aggregation
        round_map: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"n_flags": 0, "layers": [], "scores": []}
        )
        for entry in history:
            rnd = entry.get("round_num") or 0
            round_map[rnd]["n_flags"] += 1
            layer = entry.get("layer_id", "?")
            if layer not in round_map[rnd]["layers"]:
                round_map[rnd]["layers"].append(layer)
            if "score" in entry:
                round_map[rnd]["scores"].append(entry["score"])

        trajectory = sorted(
            [
                {
                    "round_num": rnd,
                    "n_flags": data["n_flags"],
                    "layers": sorted(data["layers"]),
                    "mean_score": round(
                        sum(data["scores"]) / len(data["scores"]), 4
                    ) if data["scores"] else None,
                }
                for rnd, data in round_map.items()
            ],
            key=lambda d: d["round_num"],
        )
        if n_rounds is not None:
            trajectory = trajectory[-n_rounds:]

        charts: list[ChartArtifact] = []
        if self._charts is not None and trajectory:
            try:
                chart = self._charts.trust_trajectory_chart(
                    TrustExplanation(
                        client_id=client_id,
                        current_score=base.current_score,
                        is_suspicious=base.is_suspicious,
                        narrative=base.narrative,
                        score_trajectory=trajectory,
                        layer_breakdown=base.layer_breakdown,
                    )
                )
                charts.append(chart)
            except Exception as exc:
                logger.debug("TrustExplainer: trajectory chart failed: %s", exc)

        return TrustExplanation(
            client_id=client_id,
            current_score=base.current_score,
            is_suspicious=base.is_suspicious,
            narrative=base.narrative,
            score_trajectory=trajectory,
            top_contributing_entries=base.top_contributing_entries,
            layer_breakdown=base.layer_breakdown,
            chart_artifacts=charts,
        )

    def rank_clients_by_suspicion(
        self,
        ledger: Any,
        top_k: int | None = None,
        suspicious_threshold: float | None = None,
    ) -> list[TrustExplanation]:
        """Return explanations for all known clients, ordered by suspicion.

        Args:
            ledger: FileTrustLedger.
            top_k: Return only the top-k most suspicious clients.
            suspicious_threshold: Override threshold.

        Returns:
            List of TrustExplanation objects, most suspicious first.
        """
        threshold = suspicious_threshold if suspicious_threshold is not None else self._threshold
        client_ids = self._all_client_ids(ledger)

        explanations = []
        for cid in client_ids:
            exp = self.explain_trust_score(cid, ledger, suspicious_threshold=threshold)
            explanations.append(exp)

        # Sort: suspicious first, then by score ascending (lowest score = most suspicious)
        explanations.sort(key=lambda e: (not e.is_suspicious, e.current_score))

        if top_k is not None:
            explanations = explanations[:top_k]
        return explanations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client_history(self, client_id: str, ledger: Any) -> list[dict[str, Any]]:
        """Fetch all ledger entries for a client as plain dicts."""
        try:
            entries = ledger.get_client_history(client_id)
            if entries and hasattr(entries[0], "model_dump"):
                return [e.model_dump() for e in entries]
            return list(entries)
        except Exception as exc:
            logger.debug("TrustExplainer._get_client_history: %s", exc)
            return []

    def _current_score(self, client_id: str, ledger: Any) -> float:
        """Best-effort extraction of current trust score from ledger."""
        try:
            # Try TrustScoreManager-style reputation
            rep = ledger.get_reputation(client_id)
            if isinstance(rep, (int, float)):
                return float(rep)
            if hasattr(rep, "score"):
                return float(rep.score)
        except Exception:
            pass
        # Fallback: derive from flag count
        history = self._get_client_history(client_id, ledger)
        if not history:
            return 1.0
        n_flags = len(history)
        return max(0.0, 1.0 - n_flags * 0.15)

    def _layer_breakdown(self, history: list[dict[str, Any]]) -> dict[str, int]:
        breakdown: dict[str, int] = defaultdict(int)
        for entry in history:
            layer = entry.get("layer_id", "?")
            breakdown[layer] += 1
        return dict(breakdown)

    def _top_entries(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return top-k entries by score (highest = most concerning)."""
        scored = [e for e in history if e.get("score") is not None]
        sorted_entries = sorted(scored, key=lambda e: e.get("score", 0), reverse=True)
        result = []
        for e in sorted_entries[: self._top_k]:
            result.append({
                "entry_id": e.get("entry_id", ""),
                "layer_id": e.get("layer_id", ""),
                "round_num": e.get("round_num"),
                "score": e.get("score"),
                "reason": (e.get("reason") or "")[:120],
            })
        return result

    def _all_client_ids(self, ledger: Any) -> list[str]:
        """Extract all unique client IDs from the ledger."""
        try:
            snapshot = ledger.export_snapshot()
            if isinstance(snapshot, list):
                return list({
                    e.get("subject_id", "") or (e.subject_id if hasattr(e, "subject_id") else "")
                    for e in snapshot
                    if (e.get("subject_type") if isinstance(e, dict) else
                        getattr(e, "subject_type", "")) == "client"
                })
            # Alternate: try get_all_clients() method
            return list(ledger.get_all_clients())
        except Exception as exc:
            logger.debug("TrustExplainer._all_client_ids: %s", exc)
            return []

    def _build_narrative(
        self,
        client_id: str,
        score: float,
        is_suspicious: bool,
        total_flags: int,
        layer_breakdown: dict[str, int],
    ) -> str:
        status = "SUSPICIOUS" if is_suspicious else "TRUSTED"
        parts = [
            f"Client '{client_id}' [{status}] — trust score: {score:.4f}.",
            _flag_count_narrative(total_flags, client_id),
        ]
        if layer_breakdown:
            layers_str = ", ".join(
                f"{layer}: {count}" for layer, count in sorted(layer_breakdown.items())
            )
            parts.append(f"Flag breakdown by layer: {layers_str}.")
        return " ".join(parts)
