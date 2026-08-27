"""
tests/test_update_guard.py — Unit tests for L1 Update Guard (TESTING.md §2).

Test requirements from TESTING.md:
  - detect_collusion_clusters(): a synthetic case with a known 3-client tight
    cluster must be recovered exactly.
  - A case with no collusion must return zero clusters (no false positives on
    independent random residuals).
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.detection.update_guard import (
    CollusionGuardStrategy,
    cosine_sim_matrix,
    detect_collusion_clusters,
)


class TestCosineSimMatrix:
    def test_self_similarity_is_one(self):
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_sim_matrix([v, v])
        assert abs(sim[0, 1] - 1.0) < 1e-6

    def test_orthogonal_vectors_have_zero_similarity(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        sim = cosine_sim_matrix([v1, v2])
        assert abs(sim[0, 1]) < 1e-6

    def test_opposite_vectors_have_neg_one_similarity(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        sim = cosine_sim_matrix([v1, v2])
        assert abs(sim[0, 1] + 1.0) < 1e-6

    def test_zero_vector_handled_without_nan(self):
        v0 = np.zeros(5)
        v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        sim = cosine_sim_matrix([v0, v1])
        assert not np.any(np.isnan(sim))

    def test_symmetric(self):
        rng = np.random.default_rng(0)
        vecs = [rng.normal(size=10) for _ in range(5)]
        sim = cosine_sim_matrix(vecs)
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)


class TestDetectCollusionClusters:
    """TESTING.md requirement: known tight cluster recovered exactly; no FP on random."""

    def _make_colluding_setup(self) -> tuple[list[np.ndarray], np.ndarray]:
        """Create 5 clients where 0,1,2 collude and 3,4 are honest."""
        rng = np.random.default_rng(42)
        n_features = 20

        # Honest clients — independent random residuals
        honest = [rng.normal(0, 1, n_features) for _ in range(2)]

        # Colluding clients — share the same direction with small noise
        shared_direction = rng.normal(0, 1, n_features)
        shared_direction /= np.linalg.norm(shared_direction)
        colluders = [shared_direction * 5.0 + rng.normal(0, 0.05, n_features) for _ in range(3)]

        updates = honest + colluders  # indices [0,1] honest, [2,3,4] colluding
        # Build an aggregate that is close to the honest mean
        aggregate = np.mean(honest, axis=0)
        return updates, aggregate

    def test_known_cluster_recovered_exactly(self):
        updates, aggregate = self._make_colluding_setup()
        result = detect_collusion_clusters(
            updates, aggregate, sim_threshold=0.90, min_cluster_size=2
        )
        clusters = result["flagged_clusters"]
        # The 3 colluding clients (indices 2,3,4) must appear in exactly one cluster
        assert len(clusters) == 1, f"Expected 1 cluster, got {clusters}"
        assert set(clusters[0]) == {2, 3, 4}, f"Expected {{2,3,4}}, got {clusters[0]}"

    def test_no_false_positives_on_independent_residuals(self):
        rng = np.random.default_rng(99)
        n_features = 30
        # 8 independent random clients — no collusion
        updates = [rng.normal(0, 1, n_features) for _ in range(8)]
        aggregate = np.mean(updates, axis=0)
        result = detect_collusion_clusters(
            updates, aggregate, sim_threshold=0.85, min_cluster_size=2
        )
        assert (
            result["flagged_clusters"] == []
        ), f"Unexpected clusters: {result['flagged_clusters']}"

    def test_collusion_score_is_nonzero_for_flagged_clients(self):
        updates, aggregate = self._make_colluding_setup()
        result = detect_collusion_clusters(updates, aggregate)
        scores = result["collusion_score"]
        # Indices 0,1 (honest) should have score 0.0
        assert scores[0] == pytest.approx(0.0)
        assert scores[1] == pytest.approx(0.0)
        # Indices 2,3,4 (colluding) should have positive score
        assert scores[2] > 0
        assert scores[3] > 0
        assert scores[4] > 0

    def test_similarity_matrix_is_square(self):
        rng = np.random.default_rng(7)
        updates = [rng.normal(size=10) for _ in range(4)]
        aggregate = rng.normal(size=10)
        result = detect_collusion_clusters(updates, aggregate)
        sim = np.array(result["similarity_matrix"])
        assert sim.shape == (4, 4)

    def test_min_cluster_size_respected(self):
        """With min_cluster_size=4, a 3-client cluster should not be flagged."""
        updates, aggregate = self._make_colluding_setup()
        result = detect_collusion_clusters(
            updates, aggregate, sim_threshold=0.90, min_cluster_size=4
        )
        # 3 colluders < min_cluster_size=4 → should not be flagged
        assert result["flagged_clusters"] == []


class TestCollusionGuardStrategy:
    """Smoke test: strategy produces TrustLedgerEntry objects for flagged clusters."""

    def test_process_returns_trust_ledger_entries(self, toy_client_updates):
        from types import SimpleNamespace

        updates, aggregate = toy_client_updates
        client_ids = [f"client_{i:02d}" for i in range(len(updates))]

        ctx = SimpleNamespace(
            client_updates=updates,
            aggregate=aggregate,
            client_ids=client_ids,
            round_num=1,
        )
        strategy = CollusionGuardStrategy(sim_threshold=0.90, min_cluster_size=2)
        entries = strategy.process(ctx)

        # We know clients 2 and 3 in toy_client_updates are colluding
        flagged_ids = {e.subject_id for e in entries}
        assert "client_02" in flagged_ids or "client_03" in flagged_ids

        for entry in entries:
            assert entry.layer_id == "L1"
            assert entry.subject_type == "client"
            assert 0.0 <= entry.score <= 1.0
            assert entry.reason  # non-empty explanation
