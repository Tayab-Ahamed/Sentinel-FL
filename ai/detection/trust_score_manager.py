"""
ai/detection/trust_score_manager.py — Per-client trust score tracking for L1.

``TrustScoreManager`` maintains an in-memory trust score for every client
observed during a federated learning experiment.  It is the bridge between
the anomaly signals produced by ``UpdateAnomalyDetector`` / collusion
clustering and the durable ``FileTrustLedger``.

Trust score semantics (SCHEMAS.md §TrustScore):
  - Score ∈ [0, 1].  0 = fully trusted, 1 = fully flagged.
  - Per-round update: ``score = min(1.0, score * (1-decay) + anomaly_score * weight)``
  - Exponential decay ensures that a previously suspicious client can recover
    if their subsequent updates are benign.

Client ranking:
  - ``rank_clients()`` returns client IDs sorted ascending by trust score
    (most trusted = lowest score = first).
  - ``get_suspicious_clients(threshold)`` returns all clients with score ≥ threshold.

Ledger integration:
  - If a ``FileTrustLedger`` is provided, every score update creates a
    ``TrustLedgerEntry`` and calls ``ledger.add_entry()``.
  - The manager itself never reads from the ledger.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai.fl_core.schemas import TrustLedgerEntry, TrustScore

logger = logging.getLogger(__name__)


class TrustScoreManager:
    """Manages per-client trust scores with exponential decay and ledger integration.

    Args:
        decay_rate: Per-round exponential decay applied to all scores.
            0.0 = no decay; 0.1 = 10% reduction per round.
        weight: Coefficient for anomaly score contribution:
            ``score += anomaly_score * weight``.
        ledger: Optional ``FileTrustLedger``; if provided, every update is
            written as a ``TrustLedgerEntry``.
    """

    def __init__(
        self,
        decay_rate: float = 0.1,
        weight: float = 0.5,
        ledger: Any | None = None,  # FileTrustLedger (avoid circular import)
    ) -> None:
        if not (0.0 <= decay_rate <= 1.0):
            raise ValueError(f"decay_rate must be in [0, 1], got {decay_rate}")
        if not (0.0 < weight <= 1.0):
            raise ValueError(f"weight must be in (0, 1], got {weight}")
        self._decay = decay_rate
        self._weight = weight
        self._ledger = ledger
        # client_id → TrustScore
        self._scores: dict[str, TrustScore] = {}

    # ------------------------------------------------------------------
    # Core update methods
    # ------------------------------------------------------------------

    def update(
        self,
        client_id: str,
        anomaly_score: float,
        round_num: int,
        reason: str = "",
        evidence: dict | None = None,
    ) -> TrustScore:
        """Update the trust score for a single client.

        Args:
            client_id: Client identifier string.
            anomaly_score: Anomaly signal in [0, 1] from the detector.
            round_num: Current FL round.
            reason: Human-readable explanation for the ledger entry.
            evidence: Raw evidence dict (norm, z-score, cluster info).

        Returns:
            Updated ``TrustScore`` for ``client_id``.
        """
        existing = self._scores.get(client_id)
        prev_score = existing.score if existing else 0.0
        prev_events = existing.contributing_events if existing else []

        # Update formula
        new_score = float(
            min(1.0, prev_score * (1.0 - self._decay) + anomaly_score * self._weight)
        )

        trust_score = TrustScore(
            subject_type="client",
            subject_id=client_id,
            score=new_score,
            last_updated_round=round_num,
            contributing_events=prev_events,  # appended after ledger write
        )
        self._scores[client_id] = trust_score

        logger.debug(
            "TrustScoreManager: client=%s round=%d anomaly=%.4f "
            "prev=%.4f → new=%.4f",
            client_id, round_num, anomaly_score, prev_score, new_score,
        )

        # Persist to ledger if attached
        if self._ledger is not None and anomaly_score > 0.0:
            entry = TrustLedgerEntry(
                layer_id="L1",
                subject_type="client",
                subject_id=client_id,
                round_num=round_num,
                score=float(anomaly_score),
                reason=reason or f"Anomaly score={anomaly_score:.4f}",
                evidence=evidence or {},
            )
            try:
                self._ledger.add_entry(entry)
                trust_score.contributing_events = [*prev_events, entry.entry_id][-50:]
                self._scores[client_id] = trust_score
            except Exception as exc:
                logger.warning("TrustScoreManager: ledger write failed: %s", exc)

        return trust_score

    def update_batch(
        self,
        client_ids: list[str],
        anomaly_scores: np.ndarray | list[float],
        round_num: int,
        reasons: list[str] | None = None,
        evidences: list[dict] | None = None,
    ) -> list[TrustScore]:
        """Update trust scores for all clients in a round.

        Args:
            client_ids: Client ID strings (same order as ``anomaly_scores``).
            anomaly_scores: Per-client anomaly scores in [0, 1].
            round_num: Current FL round.
            reasons: Optional per-client reason strings.
            evidences: Optional per-client evidence dicts.

        Returns:
            List of updated ``TrustScore`` objects, one per client.
        """
        scores = list(anomaly_scores)
        reasons = reasons or [""] * len(client_ids)
        evidences = evidences or [{}] * len(client_ids)

        results = []
        for cid, ascore, reason, ev in zip(client_ids, scores, reasons, evidences):
            ts = self.update(
                client_id=cid,
                anomaly_score=float(ascore),
                round_num=round_num,
                reason=reason,
                evidence=ev,
            )
            results.append(ts)
        return results

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def decay_all(self, round_num: int) -> None:
        """Apply exponential decay to all client trust scores.

        Called once per FL round after all per-client updates for the round.

        Args:
            round_num: The round to stamp on decayed scores.
        """
        for _cid, ts in self._scores.items():
            ts.score = max(0.0, ts.score * (1.0 - self._decay))
            ts.last_updated_round = round_num
        logger.debug(
            "TrustScoreManager.decay_all: decayed %d client scores (rate=%.2f)",
            len(self._scores), self._decay,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_score(self, client_id: str) -> TrustScore | None:
        """Return the current TrustScore for a client, or None if unknown."""
        return self._scores.get(client_id)

    def get_all_scores(self) -> list[TrustScore]:
        """Return all current TrustScore objects."""
        return list(self._scores.values())

    def rank_clients(
        self, client_ids: list[str] | None = None
    ) -> list[str]:
        """Return client IDs sorted ascending by trust score (most trusted first).

        Args:
            client_ids: Optional subset of client IDs to rank.  If None,
                ranks all known clients.

        Returns:
            List of client IDs, most trusted (lowest score) first.
        """
        ids = client_ids if client_ids is not None else list(self._scores.keys())
        return sorted(ids, key=lambda cid: self._scores.get(cid, _zero_score(cid)).score)

    def get_suspicious_clients(self, threshold: float = 0.5) -> list[str]:
        """Return IDs of clients whose trust score ≥ threshold.

        Args:
            threshold: Score threshold in [0, 1].

        Returns:
            List of client IDs with elevated trust scores (suspicious).
        """
        return [
            cid for cid, ts in self._scores.items()
            if ts.score >= threshold
        ]

    def score_vector(self, client_ids: list[str]) -> np.ndarray:
        """Return trust scores as a float32 array in the same order as ``client_ids``.

        Args:
            client_ids: Ordered list of client IDs.

        Returns:
            Float32 array of shape ``(n,)``.
        """
        return np.array(
            [self._scores.get(cid, _zero_score(cid)).score for cid in client_ids],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable ``{client_id: score}`` mapping."""
        return {cid: round(ts.score, 6) for cid, ts in self._scores.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_score(client_id: str) -> TrustScore:
    """Return a zero TrustScore for an unknown client."""
    return TrustScore(
        subject_type="client",
        subject_id=client_id,
        score=0.0,
        last_updated_round=0,
    )
