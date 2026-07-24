"""
ai/detection/reputation_engine.py — L4: Reputation Engine.

Analytics layer above ``FileTrustLedger`` that produces structured reports,
visualization data, and cross-layer metrics for the dashboard (API.md §4–7)
and evaluation pipeline.

Separation of concerns:
  - ``FileTrustLedger``  → I/O, persistence, in-memory index, score cache
  - ``ReputationEngine`` → analytics, derived metrics, multi-client reports

All public methods return JSON-serialisable dicts so they can be served
directly by the FastAPI backend (Milestone 7+) without further transformation.

Design: stateless analytics — the engine holds a reference to the ledger and
recomputes metrics on demand.  Expensive computations (e.g. ``cross_layer_correlation``)
are cached with a round-stamp so repeated calls in the same round are free.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from ai.detection.trust_ledger import FileTrustLedger
from ai.fl_core.schemas import TrustLedgerQuery

logger = logging.getLogger(__name__)


class ReputationEngine:
    """Analytics engine for the L4 Trust Ledger.

    Args:
        ledger: The ``FileTrustLedger`` instance to analyse.
        suspicious_threshold: Score threshold for "suspicious" classification.
            Falls back to ``ledger.suspicious_threshold`` if None.
    """

    def __init__(
        self,
        ledger: FileTrustLedger,
        suspicious_threshold: float | None = None,
    ) -> None:
        self._ledger = ledger
        self._threshold = (
            suspicious_threshold
            if suspicious_threshold is not None
            else ledger.suspicious_threshold
        )
        # Simple round-keyed computation cache
        self._cache: dict[str, Any] = {}
        self._cache_round: int = -1

    # ------------------------------------------------------------------
    # Per-client report
    # ------------------------------------------------------------------

    def client_reputation_report(
        self,
        client_id: str,
        n_rounds: int | None = None,
    ) -> dict[str, Any]:
        """Full reputation dossier for a single client.

        Args:
            client_id: Client identifier.
            n_rounds: If set, only include entries from the last n_rounds.

        Returns:
            Dict with: ``client_id``, ``current_score``, ``is_suspicious``,
            ``total_flag_count``, ``layer_breakdown`` (flags per layer),
            ``round_history`` (per-round score and entry list),
            ``most_recent_reason``.
        """
        ts = self._ledger.get_score(client_id)
        current_score = ts.score if ts else 0.0
        entries = self._ledger.get_client_history(client_id, max_rounds=n_rounds)

        # Layer breakdown: count flags per layer
        layer_breakdown: dict[str, int] = defaultdict(int)
        for e in entries:
            layer_breakdown[e.layer_id] += 1

        # Round history: group entries by round_num
        round_groups: dict[int, list[dict]] = defaultdict(list)
        for e in entries:
            rnd = e.round_num if e.round_num is not None else -1
            round_groups[rnd].append(
                {
                    "entry_id": e.entry_id,
                    "layer_id": e.layer_id,
                    "score": e.score,
                    "reason": e.reason,
                }
            )

        round_history = [
            {
                "round_num": rnd,
                "n_flags": len(evts),
                "max_score": max(ev["score"] for ev in evts),
                "events": evts,
            }
            for rnd, evts in sorted(round_groups.items())
        ]

        most_recent_reason = entries[0].reason if entries else ""

        return {
            "client_id": client_id,
            "current_score": round(current_score, 4),
            "is_suspicious": current_score >= self._threshold,
            "total_flag_count": len(entries),
            "layer_breakdown": dict(layer_breakdown),
            "round_history": round_history,
            "most_recent_reason": most_recent_reason,
            "contributing_event_count": len(ts.contributing_events) if ts else 0,
        }

    # ------------------------------------------------------------------
    # Heatmap (multi-client, multi-round)
    # ------------------------------------------------------------------

    def heatmap_data(
        self,
        round_range: tuple[int | None, int | None] = (None, None),
    ) -> dict[str, Any]:
        """Client × round trust-score matrix for the dashboard heatmap.

        Delegates to ``FileTrustLedger.reputation_heatmap()`` and enriches
        with summary statistics per client and per round.

        Args:
            round_range: ``(round_min, round_max)`` inclusive.

        Returns:
            Full heatmap dict with client_ids, rounds, matrix, current_scores,
            plus ``per_client_summary`` and ``per_round_summary``.
        """
        base = self._ledger.reputation_heatmap(round_range)

        client_ids = base["client_ids"]
        rounds = base["rounds"]
        matrix = base["matrix"]

        # Per-client summary: max score, n_flagged_rounds
        per_client: list[dict] = []
        for i, cid in enumerate(client_ids):
            row = [v for v in matrix[i] if v is not None]
            per_client.append(
                {
                    "client_id": cid,
                    "max_score": round(max(row), 4) if row else 0.0,
                    "mean_score": round(sum(row) / len(row), 4) if row else 0.0,
                    "n_flagged_rounds": sum(
                        1 for v in row if v is not None and v >= self._threshold
                    ),
                    "current_score": base["current_scores"].get(cid, 0.0),
                }
            )

        # Per-round summary: mean + max across all clients
        per_round: list[dict] = []
        for j, rnd in enumerate(rounds):
            col = [matrix[i][j] for i in range(len(client_ids)) if matrix[i][j] is not None]
            per_round.append(
                {
                    "round_num": rnd,
                    "mean_score": round(sum(col) / len(col), 4) if col else 0.0,
                    "max_score": round(max(col), 4) if col else 0.0,
                    "n_flagged_clients": sum(1 for v in col if v >= self._threshold),
                }
            )

        return {
            **base,
            "per_client_summary": per_client,
            "per_round_summary": per_round,
        }

    # ------------------------------------------------------------------
    # Flag rate by layer
    # ------------------------------------------------------------------

    def flag_rate_by_layer(self) -> dict[str, float]:
        """Fraction of all ledger entries produced by each layer.

        Returns:
            Dict ``{layer_id: fraction}`` e.g. ``{"L1": 0.72, "L2": 0.28}``.
        """
        all_entries = self._ledger.load_entries()
        if not all_entries:
            return {}
        counts: dict[str, int] = defaultdict(int)
        for e in all_entries:
            counts[e.layer_id] += 1
        total = len(all_entries)
        return {layer: round(count / total, 4) for layer, count in sorted(counts.items())}

    # ------------------------------------------------------------------
    # Score distribution
    # ------------------------------------------------------------------

    def score_distribution(
        self,
        round_num: int | None = None,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """Histogram of current trust scores across all clients.

        Args:
            round_num: If set, only considers entries up to this round.
            n_bins: Number of histogram bins (evenly spaced in [0, 1]).

        Returns:
            Dict with ``bins``, ``counts``, ``mean``, ``std``, ``n_suspicious``.
        """
        if round_num is not None:
            # Rebuild scores up to round_num from ledger
            q = TrustLedgerQuery(round_max=round_num)
            entries = self._ledger.query(q)
            score_map: dict[str, float] = {}
            for e in entries:
                score_map[e.subject_id] = min(1.0, score_map.get(e.subject_id, 0.0) + e.score * 0.5)
            scores = list(score_map.values())
        else:
            scores = [ts.score for ts in self._ledger.get_all_scores()]

        if not scores:
            return {"bins": [], "counts": [], "mean": 0.0, "std": 0.0, "n_suspicious": 0}

        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        counts = [0] * n_bins
        for s in scores:
            idx = min(int(s * n_bins), n_bins - 1)
            counts[idx] += 1

        mean_s = sum(scores) / len(scores)
        variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
        std_s = math.sqrt(variance)

        return {
            "bins": [round(b, 2) for b in bin_edges[:-1]],
            "counts": counts,
            "mean": round(mean_s, 4),
            "std": round(std_s, 4),
            "n_suspicious": sum(1 for s in scores if s >= self._threshold),
            "total_clients": len(scores),
        }

    # ------------------------------------------------------------------
    # Suspicious timeline
    # ------------------------------------------------------------------

    def suspicious_timeline(
        self,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Per-round count of clients above the suspicious threshold.

        Args:
            threshold: Override suspicious threshold.

        Returns:
            List of ``{round_num, n_suspicious, suspicious_client_ids}``
            dicts sorted by round_num ascending.
        """
        thr = threshold if threshold is not None else self._threshold
        all_entries = self._ledger.load_entries()

        # Group by round
        round_entries: dict[int, list] = defaultdict(list)
        for e in all_entries:
            if e.round_num is not None:
                round_entries[e.round_num].append(e)

        timeline: list[dict] = []
        running_scores: dict[str, float] = {}
        for rnd in sorted(round_entries.keys()):
            for e in round_entries[rnd]:
                running_scores[e.subject_id] = min(
                    1.0,
                    running_scores.get(e.subject_id, 0.0) + e.score * 0.5,
                )
            suspicious = [cid for cid, s in running_scores.items() if s >= thr]
            timeline.append(
                {
                    "round_num": rnd,
                    "n_suspicious": len(suspicious),
                    "suspicious_client_ids": sorted(suspicious),
                }
            )
        return timeline

    # ------------------------------------------------------------------
    # Cross-layer correlation
    # ------------------------------------------------------------------

    def cross_layer_correlation(self) -> dict[str, Any]:
        """Compute how often each pair of layers flag the same client.

        Useful for the ablation study — a high L1–L2 overlap means the
        two layers agree; a low overlap means they provide independent signal.

        Returns:
            Dict with ``agreement_matrix`` (layer × layer fraction overlap)
            and ``layer_ids`` list.
        """
        # Build per-layer client sets
        layer_clients: dict[str, set[str]] = defaultdict(set)
        for e in self._ledger.load_entries():
            layer_clients[e.layer_id].add(e.subject_id)

        layer_ids = sorted(layer_clients.keys())
        n = len(layer_ids)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

        for i, li in enumerate(layer_ids):
            for j, lj in enumerate(layer_ids):
                set_i = layer_clients[li]
                set_j = layer_clients[lj]
                union = set_i | set_j
                if not union:
                    matrix[i][j] = 1.0
                else:
                    # Jaccard similarity
                    matrix[i][j] = round(len(set_i & set_j) / len(union), 4)

        return {
            "layer_ids": layer_ids,
            "agreement_matrix": matrix,
        }

    # ------------------------------------------------------------------
    # Experiment-level metrics
    # ------------------------------------------------------------------

    def compute_all_metrics(self, experiment_id: str) -> dict[str, Any]:
        """Compute the full Trust Ledger metric set for a completed experiment.

        Args:
            experiment_id: Identifier string (used as a label in output only).

        Returns:
            Dict with all Trust Ledger metrics.
        """
        all_scores = self._ledger.get_all_scores()
        n = len(all_scores)
        stats = self._ledger.get_stats()

        return {
            "experiment_id": experiment_id,
            "total_ledger_entries": stats["total_entries"],
            "n_tracked_clients": n,
            "n_suspicious_clients": stats["suspicious_count"],
            "suspicious_fraction": round(stats["suspicious_count"] / n, 4) if n else 0.0,
            "mean_trust_score": stats["mean_score"],
            "top_suspicious": [
                {"client_id": ts.subject_id, "score": round(ts.score, 4)}
                for ts in self._ledger.top_k_suspicious(5)
            ],
            "flag_rate_by_layer": self.flag_rate_by_layer(),
            "score_distribution": self.score_distribution(),
        }
