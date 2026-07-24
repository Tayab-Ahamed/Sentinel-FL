"""
update_guard.py — L1: Update Guard.

Wraps a base robust-aggregation rule (Multi-Krum, in fl_engine.multi_krum) with a
second, complementary signal: pairwise cosine-similarity clustering over each client's
*residual* update (their update minus the current robust aggregate). This targets a
specific gap identified in RESEARCH.md — a trigger fragmented across several colluding
clients, each individually mild enough to survive Krum's per-client outlier filter, but
collectively pointing the same unusual direction.

Relationship to prior art (see RESEARCH.md §4): FoolsGold (Fung et al., RAID 2020,
implemented in FedML/python/fedml/core/security/defense/foolsgold_defense.py) already
uses pairwise cosine similarity of client updates to down-weight sybil-like colluders.
This module differs in three ways: (1) it clusters *residuals after robust aggregation*
rather than raw last-layer gradients, so it is a second, independent layer behind
Multi-Krum rather than a replacement for it; (2) it feeds a bounded [0,1] collusion
score per client into the shared Trust Ledger (L4) instead of directly reweighting the
aggregation, keeping the two defense signals auditable and separable for the ablation
study in IMPLEMENTATION_PLAN.md; (3) it flags *clusters* (>=2 correlated clients) rather
than scoring clients independently, which is what a fragmented-trigger attack requires.

The pure functions (``cosine_sim_matrix``, ``detect_collusion_clusters``) are the
Phase 0 NumPy proof-of-concept and are PRESERVED UNCHANGED.  ``CollusionGuardStrategy``
wraps them in the DefenseStrategy interface contract (INTERFACES.md).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai.fl_core.interfaces import DefenseStrategy
from ai.fl_core.schemas import TrustLedgerEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 0 pure functions (PRESERVED UNCHANGED from original implementation)
# ---------------------------------------------------------------------------


def cosine_sim_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Compute the pairwise cosine-similarity matrix for a list of vectors.

    Args:
        vectors: List of 1-D NumPy arrays of equal length.

    Returns:
        Square matrix of shape ``(n, n)`` with values in ``[-1, 1]``.
    """
    stacked = np.stack(vectors, axis=0)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normed = stacked / norms
    return normed @ normed.T


def detect_collusion_clusters(
    client_updates: list[np.ndarray],
    aggregate: np.ndarray,
    sim_threshold: float = 0.85,
    min_cluster_size: int = 2,
) -> dict[str, Any]:
    """Detect colluding client clusters from residual update directions.

    residual_i = update_i - aggregate

    Clients whose residuals are highly cosine-similar to each other (>= sim_threshold)
    form a candidate collusion cluster. A cluster of size >= min_cluster_size is flagged.
    Isolated high-residual clients are NOT flagged here — that is Multi-Krum's job;
    this layer is specifically for *correlated groups*.

    Args:
        client_updates: List of flat parameter delta arrays, one per client.
        aggregate: The current robust aggregate delta (e.g. Multi-Krum output).
        sim_threshold: Cosine-similarity threshold for cluster membership.
        min_cluster_size: Minimum clients in a cluster to flag it.

    Returns:
        Dict with keys:
            - ``"collusion_score"`` (list[float]): per-client score in [0, 1]
            - ``"flagged_clusters"`` (list[list[int]]): indices of flagged clusters
            - ``"similarity_matrix"`` (list[list[float]]): full pairwise sim matrix
    """
    residuals = [u - aggregate for u in client_updates]
    sims = cosine_sim_matrix(residuals)
    n = len(client_updates)

    adj = sims >= sim_threshold
    np.fill_diagonal(adj, False)

    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            comp.append(node)
            neighbors = np.where(adj[node])[0]
            stack.extend(neighbors.tolist())
        if len(comp) >= min_cluster_size:
            clusters.append(sorted(comp))

    collusion_score = np.zeros(n)
    for cluster in clusters:
        # score = mean pairwise similarity within the cluster, assigned to each member
        idx = np.array(cluster)
        sub = sims[np.ix_(idx, idx)]
        mean_sim = (sub.sum() - len(idx)) / max(1, (len(idx) * (len(idx) - 1)))
        for i in cluster:
            collusion_score[i] = max(collusion_score[i], mean_sim)

    return {
        "collusion_score": collusion_score.tolist(),
        "flagged_clusters": clusters,
        "similarity_matrix": sims.tolist(),
    }


# ---------------------------------------------------------------------------
# DefenseStrategy implementation
# ---------------------------------------------------------------------------


class CollusionGuardStrategy(DefenseStrategy):
    """L1 DefenseStrategy — Multi-Krum + residual collusion clustering.

    Wraps ``detect_collusion_clusters()`` in the DefenseStrategy interface so
    the L1 layer can be composed by configuration and its output written to the
    Trust Ledger (L4) in a structured, auditable way.

    Args:
        sim_threshold: Cosine-similarity threshold for cluster membership.
        min_cluster_size: Minimum cluster size to flag.
    """

    layer_id: str = "L1"

    def __init__(
        self,
        sim_threshold: float = 0.85,
        min_cluster_size: int = 2,
    ) -> None:
        self._sim_threshold = sim_threshold
        self._min_cluster_size = min_cluster_size

    def process(self, context: Any) -> list[TrustLedgerEntry]:
        """Run collusion detection on client updates from the round context.

        Args:
            context: RoundContext carrying:
                - ``client_updates`` (list[np.ndarray]): raw delta vectors
                - ``aggregate`` (np.ndarray): current robust aggregate delta
                - ``client_ids`` (list[str]): corresponding client IDs
                - ``round_num`` (int): current FL round number

        Returns:
            List of TrustLedgerEntry objects for each flagged cluster.
        """
        client_updates: list[np.ndarray] = context.client_updates
        aggregate: np.ndarray = context.aggregate
        client_ids: list[str] = context.client_ids
        round_num: int = context.round_num

        result = detect_collusion_clusters(
            client_updates=client_updates,
            aggregate=aggregate,
            sim_threshold=self._sim_threshold,
            min_cluster_size=self._min_cluster_size,
        )

        entries: list[TrustLedgerEntry] = []
        for cluster_indices in result["flagged_clusters"]:
            cluster_ids = [client_ids[i] for i in cluster_indices]
            scores = [result["collusion_score"][i] for i in cluster_indices]
            mean_score = float(np.mean(scores))
            sim_matrix_slice = [
                [result["similarity_matrix"][i][j] for j in cluster_indices]
                for i in cluster_indices
            ]
            reason = self._explain_cluster(cluster_ids, mean_score, sim_matrix_slice)

            for cid, score in zip(cluster_ids, scores):
                entry = TrustLedgerEntry(
                    layer_id="L1",
                    subject_type="client",
                    subject_id=cid,
                    round_num=round_num,
                    score=float(score),
                    reason=reason,
                    evidence={
                        "cluster_ids": cluster_ids,
                        "sim_matrix_slice": sim_matrix_slice,
                        "sim_threshold": self._sim_threshold,
                    },
                )
                entries.append(entry)
                logger.info(
                    "L1 cluster flag: client=%s round=%d cluster=%s score=%.3f",
                    cid,
                    round_num,
                    cluster_ids,
                    score,
                )

        return entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _explain_cluster(
        self,
        cluster_ids: list[str],
        mean_score: float,
        sim_matrix: list[list[float]],
    ) -> str:
        """Generate the human-readable reason string for the Trust Ledger."""
        ids_str = ", ".join(cluster_ids)
        # Format the max off-diagonal similarity for the explanation
        max_sim = (
            max(
                sim_matrix[i][j]
                for i in range(len(sim_matrix))
                for j in range(len(sim_matrix[i]))
                if i != j
            )
            if len(sim_matrix) > 1
            else 0.0
        )
        return (
            f"Clients [{ids_str}] form a residual-update cluster "
            f"(max pairwise cosine similarity = {max_sim:.2f}, "
            f"mean collusion score = {mean_score:.4f}). "
            "Their update directions are unusually similar after Multi-Krum aggregation, "
            "consistent with a fragmented trigger split across colluding clients."
        )


# ---------------------------------------------------------------------------
# Milestone 5: UpdateGuardResult + UpdateGuard orchestrator
# ---------------------------------------------------------------------------


from dataclasses import dataclass


@dataclass
class UpdateGuardResult:
    """Structured result produced by ``UpdateGuard.process_round()``.

    Contains all per-round L1 analysis outputs:
    norms, similarity matrix, collusion clusters, anomaly scores, and trust
    scores.  Serialisable via ``to_dict()``.
    """

    round_num: int
    client_ids: list[str]
    l2_norms: list[float]
    norm_zscores: list[float]
    norm_flagged: list[bool]
    similarity_matrix: list[list[float]]
    flagged_clusters: list[list[int]]
    collusion_scores: list[float]
    anomaly_scores: list[float]
    trust_scores: list[float]
    ranked_clients: list[str]
    excluded_clients: list[str]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all fields."""
        import dataclasses

        return dataclasses.asdict(self)

    def summary(self) -> str:
        """One-line summary for log output."""
        n_flagged = sum(self.norm_flagged)
        n_clusters = len(self.flagged_clusters)
        n_excluded = len(self.excluded_clients)
        return (
            f"Round {self.round_num:3d} | "
            f"norm_outliers={n_flagged} | "
            f"collusion_clusters={n_clusters} | "
            f"excluded={n_excluded} | "
            f"max_trust={max(self.trust_scores, default=0.0):.3f}"
        )


class UpdateGuard:
    """L1 Update Guard — full per-round anomaly pipeline.

    Composes gradient extraction, norm calculation, cosine-similarity
    collusion detection, anomaly scoring, and trust score management into a
    single ``process_round()`` call.

    The existing Phase-0 ``CollusionGuardStrategy`` is kept unchanged and
    still usable directly.  ``UpdateGuard`` is the Milestone 5 production
    orchestrator.

    Args:
        sim_threshold: Cosine-similarity threshold for collusion clustering.
        min_cluster_size: Minimum cluster size to flag.
        anomaly_method: ``"zscore"`` or ``"mad"``.
        norm_outlier_threshold_z: Z-score threshold for norm outlier flagging.
        norm_type: Norm used for magnitude computation (``"l2"``/``"l1"``/``"linf"``).
        exclude_flagged_clients: If True, norm outliers + colluders are removed
            from the returned ``excluded_clients`` list for use by the strategy.
        trust_score_decay: Exponential decay rate per round.
        trust_score_weight: Coefficient for anomaly score → trust score.
        log_similarity_matrix: If True, log the full similarity matrix.
        sentinel_logger: Optional StructuredLogger instance for JSON-lines output.
        ledger: Optional FileTrustLedger for persistent trust score storage.
    """

    def __init__(
        self,
        sim_threshold: float = 0.85,
        min_cluster_size: int = 2,
        anomaly_method: str = "zscore",
        norm_outlier_threshold_z: float = 3.0,
        norm_type: str = "l2",
        exclude_flagged_clients: bool = False,
        trust_score_decay: float = 0.1,
        trust_score_weight: float = 0.5,
        log_similarity_matrix: bool = False,
        sentinel_logger: Any | None = None,
        ledger: Any | None = None,
    ) -> None:
        from ai.detection.anomaly_detector import UpdateAnomalyDetector
        from ai.detection.trust_score_manager import TrustScoreManager

        self._sim_threshold = sim_threshold
        self._min_cluster_size = min_cluster_size
        self._norm_type = norm_type
        self._norm_threshold_z = norm_outlier_threshold_z
        self._exclude_flagged = exclude_flagged_clients
        self._log_sim_matrix = log_similarity_matrix
        self._sentinel_logger = sentinel_logger
        self._ledger = ledger

        self._anomaly_detector = UpdateAnomalyDetector(
            anomaly_method=anomaly_method,
            norm_type=norm_type,
            threshold_z=norm_outlier_threshold_z,
        )
        self._trust_manager = TrustScoreManager(
            decay_rate=trust_score_decay,
            weight=trust_score_weight,
            ledger=ledger,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        sentinel_logger: Any | None = None,
        ledger: Any | None = None,
    ) -> UpdateGuard:
        """Build an UpdateGuard from a ``Configuration`` object.

        Reads from ``config.update_guard`` sub-config (if present) with
        fallback to root-level collusion fields.

        Args:
            config: Configuration object (or duck-typed equivalent).
            sentinel_logger: Optional StructuredLogger.
            ledger: Optional FileTrustLedger.

        Returns:
            Configured ``UpdateGuard`` instance.
        """
        ug = getattr(config, "update_guard", None)

        def _get(attr: str, default: Any) -> Any:
            if ug is not None and hasattr(ug, attr):
                return getattr(ug, attr)
            # Fall back to root-level config fields
            return getattr(config, attr, default)

        return cls(
            sim_threshold=_get("sim_threshold", getattr(config, "collusion_sim_threshold", 0.85)),
            min_cluster_size=_get(
                "min_cluster_size", getattr(config, "collusion_min_cluster_size", 2)
            ),
            anomaly_method=_get("anomaly_method", "zscore"),
            norm_outlier_threshold_z=_get("norm_outlier_threshold_z", 3.0),
            norm_type=_get("norm_type", "l2"),
            exclude_flagged_clients=_get("exclude_flagged_clients", False),
            trust_score_decay=_get("trust_score_decay", 0.1),
            trust_score_weight=_get("trust_score_weight", 0.5),
            log_similarity_matrix=_get("log_similarity_matrix", False),
            sentinel_logger=sentinel_logger,
            ledger=ledger,
        )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def process_round(
        self,
        round_num: int,
        client_ids: list[str],
        deltas: list[np.ndarray],
    ) -> UpdateGuardResult:
        """Run the full L1 Update Guard pipeline for one FL round.

        Steps:
          1. Compute L2 norms + z-scores; flag norm outliers.
          2. Compute cosine-similarity matrix on deltas.
          3. Detect collusion clusters from residual directions.
          4. Score each update with the anomaly detector.
          5. Update per-client trust scores.
          6. Rank clients by trust score.
          7. Log everything to StructuredLogger.
          8. Write TrustLedgerEntry for flagged clients.

        Args:
            round_num: Current FL round number.
            client_ids: Client ID strings (same order as ``deltas``).
            deltas: Per-client flat parameter delta vectors.

        Returns:
            ``UpdateGuardResult`` with all computed signals.
        """
        from ai.detection.norm_calculator import (
            compute_norm_zscores,
            compute_norms,
            flag_norm_outliers,
        )

        n = len(client_ids)
        logger.info("UpdateGuard.process_round: round=%d n_clients=%d", round_num, n)

        if n == 0:
            return self._empty_result(round_num)

        # ── 1. Norms ────────────────────────────────────────────────────
        norms = compute_norms(deltas, self._norm_type)
        zscores = compute_norm_zscores(norms)
        norm_flagged = flag_norm_outliers(
            norms,
            threshold_z=self._norm_threshold_z,
            method=self._anomaly_detector.method,
        )
        flagged_by_norm = {client_ids[i] for i, f in enumerate(norm_flagged) if f}

        # ── 2+3. Cosine similarity + collusion clustering ────────────────
        # Use mean-of-all as aggregate proxy (no Multi-Krum here;
        # CollusionGuardStrategy handles Krum integration separately).
        delta_mean = np.mean(np.stack(deltas), axis=0) if deltas else np.zeros(1)
        collusion_result = detect_collusion_clusters(
            deltas,
            aggregate=delta_mean,
            sim_threshold=self._sim_threshold,
            min_cluster_size=self._min_cluster_size,
        )
        flagged_clusters: list[list[int]] = collusion_result["flagged_clusters"]
        collusion_scores: list[float] = collusion_result["collusion_score"]
        sim_matrix: list[list[float]] = collusion_result["similarity_matrix"]

        flagged_by_collusion = {client_ids[i] for cluster in flagged_clusters for i in cluster}

        # ── 4. Anomaly scores ────────────────────────────────────────────
        anomaly_scores_arr = self._anomaly_detector.score_all(deltas)
        anomaly_scores: list[float] = anomaly_scores_arr.tolist()

        # ── 5. Trust scores ──────────────────────────────────────────────
        reasons = self._build_reasons(client_ids, norm_flagged, flagged_clusters, anomaly_scores)
        evidences = [
            {
                "norm": round(float(norms[i]), 6),
                "zscore": round(float(zscores[i]), 4),
                "collusion_score": round(float(collusion_scores[i]), 4),
                "anomaly_score": round(float(anomaly_scores[i]), 4),
            }
            for i in range(n)
        ]
        trust_score_objs = self._trust_manager.update_batch(
            client_ids=client_ids,
            anomaly_scores=anomaly_scores_arr,
            round_num=round_num,
            reasons=reasons,
            evidences=evidences,
        )
        trust_scores = [ts.score for ts in trust_score_objs]

        # ── 6. Client ranking ────────────────────────────────────────────
        ranked_clients = self._trust_manager.rank_clients(client_ids)

        # ── 7. Excluded clients ──────────────────────────────────────────
        excluded: list[str] = []
        if self._exclude_flagged:
            excluded = sorted(flagged_by_norm | flagged_by_collusion)

        # ── 8. Logging ───────────────────────────────────────────────────
        self._log_round(
            round_num=round_num,
            client_ids=client_ids,
            norms=norms,
            zscores=zscores,
            norm_flagged=norm_flagged,
            flagged_clusters=flagged_clusters,
            collusion_scores=collusion_scores,
            anomaly_scores=anomaly_scores,
            trust_scores=trust_scores,
            ranked_clients=ranked_clients,
            excluded=excluded,
            sim_matrix=sim_matrix if self._log_sim_matrix else None,
        )

        result = UpdateGuardResult(
            round_num=round_num,
            client_ids=client_ids,
            l2_norms=[round(float(n), 6) for n in norms],
            norm_zscores=[round(float(z), 4) for z in zscores],
            norm_flagged=norm_flagged.tolist(),
            similarity_matrix=sim_matrix,
            flagged_clusters=flagged_clusters,
            collusion_scores=[round(float(s), 4) for s in collusion_scores],
            anomaly_scores=[round(float(s), 4) for s in anomaly_scores],
            trust_scores=[round(float(s), 4) for s in trust_scores],
            ranked_clients=ranked_clients,
            excluded_clients=excluded,
        )
        logger.info("UpdateGuard: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def trust_manager(self) -> Any:
        """The underlying TrustScoreManager."""
        return self._trust_manager

    @property
    def anomaly_detector(self) -> Any:
        """The underlying UpdateAnomalyDetector."""
        return self._anomaly_detector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_reasons(
        self,
        client_ids: list[str],
        norm_flagged: np.ndarray,
        flagged_clusters: list[list[int]],
        anomaly_scores: list[float],
    ) -> list[str]:
        """Build per-client human-readable reason strings."""
        cluster_map: dict[int, list[int]] = {}
        for cluster in flagged_clusters:
            for idx in cluster:
                cluster_map[idx] = cluster

        reasons = []
        for i, _cid in enumerate(client_ids):
            parts = []
            if norm_flagged[i]:
                parts.append("norm outlier")
            if i in cluster_map:
                peer_ids = [client_ids[j] for j in cluster_map[i] if j != i]
                parts.append(f"colluding with [{', '.join(peer_ids)}]")
            if not parts:
                parts.append(f"anomaly_score={anomaly_scores[i]:.4f}")
            reasons.append("; ".join(parts))
        return reasons

    def _log_round(
        self,
        round_num: int,
        client_ids: list[str],
        norms: np.ndarray,
        zscores: np.ndarray,
        norm_flagged: np.ndarray,
        flagged_clusters: list[list[int]],
        collusion_scores: list[float],
        anomaly_scores: list[float],
        trust_scores: list[float],
        ranked_clients: list[str],
        excluded: list[str],
        sim_matrix: list[list[float]] | None,
    ) -> None:
        """Emit structured log event for this round's Update Guard results."""
        if self._sentinel_logger is None:
            return
        payload: dict[str, Any] = {
            "round": round_num,
            "n_clients": len(client_ids),
            "norm_outliers": [cid for cid, f in zip(client_ids, norm_flagged) if f],
            "flagged_clusters": [[client_ids[i] for i in cluster] for cluster in flagged_clusters],
            "excluded_clients": excluded,
            "trust_scores": {cid: round(float(s), 4) for cid, s in zip(client_ids, trust_scores)},
            "ranked_clients": ranked_clients,
            "norms": {cid: round(float(n), 6) for cid, n in zip(client_ids, norms)},
        }
        if sim_matrix is not None:
            payload["similarity_matrix"] = sim_matrix
        try:
            self._sentinel_logger.log("L1", "update_guard_round", payload)
        except Exception as exc:
            logger.warning("UpdateGuard: log write failed: %s", exc)

    def _empty_result(self, round_num: int) -> UpdateGuardResult:
        """Return an empty result for rounds with no clients."""
        return UpdateGuardResult(
            round_num=round_num,
            client_ids=[],
            l2_norms=[],
            norm_zscores=[],
            norm_flagged=[],
            similarity_matrix=[],
            flagged_clusters=[],
            collusion_scores=[],
            anomaly_scores=[],
            trust_scores=[],
            ranked_clients=[],
            excluded_clients=[],
        )
