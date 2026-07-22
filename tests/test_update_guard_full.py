"""
tests/test_update_guard_full.py — Integration tests for UpdateGuard orchestrator.

Tests the full process_round() pipeline end-to-end:
  - norms computed and logged
  - collusion clusters detected
  - anomaly scores produced
  - trust scores updated
  - clients ranked
  - excluded clients populated when flag is set
  - UpdateGuardResult structure correct
  - from_config factory
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ai.detection.update_guard import UpdateGuard

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_deltas(n=6, dim=20, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim).astype(np.float32) for _ in range(n)]


def _colluding_setup(n_clients=6, n_colluders=3, dim=20, seed=0):
    """6 clients: first n_colluders share direction, rest random."""
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(dim)
    shared /= np.linalg.norm(shared)
    deltas = []
    for i in range(n_clients):
        if i < n_colluders:
            deltas.append((shared * 5.0 + rng.standard_normal(dim) * 0.05).astype(np.float32))
        else:
            deltas.append(rng.standard_normal(dim).astype(np.float32))
    return deltas


def _guard(**kwargs):
    defaults = dict(sim_threshold=0.90, min_cluster_size=2)
    defaults.update(kwargs)
    return UpdateGuard(**defaults)


def _cids(n):
    return [f"client_{i:02d}" for i in range(n)]


# ---------------------------------------------------------------------------
# UpdateGuardResult structure
# ---------------------------------------------------------------------------


class TestUpdateGuardResultStructure:
    def test_to_dict_is_json_serialisable(self):
        import json
        guard = _guard()
        deltas = _random_deltas(n=4)
        result = guard.process_round(0, _cids(4), deltas)
        d = result.to_dict()
        json.dumps(d)  # must not raise

    def test_summary_is_string(self):
        guard = _guard()
        result = guard.process_round(0, _cids(4), _random_deltas(n=4))
        assert isinstance(result.summary(), str)

    def test_all_list_fields_have_correct_length(self):
        n = 5
        guard = _guard()
        result = guard.process_round(1, _cids(n), _random_deltas(n=n))
        assert len(result.l2_norms) == n
        assert len(result.norm_zscores) == n
        assert len(result.norm_flagged) == n
        assert len(result.collusion_scores) == n
        assert len(result.anomaly_scores) == n
        assert len(result.trust_scores) == n
        assert len(result.ranked_clients) == n

    def test_round_num_matches(self):
        guard = _guard()
        result = guard.process_round(7, _cids(3), _random_deltas(n=3))
        assert result.round_num == 7

    def test_client_ids_preserved(self):
        cids = ["alice", "bob", "carol"]
        guard = _guard()
        result = guard.process_round(0, cids, _random_deltas(n=3))
        assert result.client_ids == cids


# ---------------------------------------------------------------------------
# Norm computation
# ---------------------------------------------------------------------------


class TestNorms:
    def test_l2_norms_are_nonneg(self):
        guard = _guard()
        result = guard.process_round(0, _cids(5), _random_deltas(n=5))
        assert all(n >= 0.0 for n in result.l2_norms)

    def test_norm_flagged_is_list_of_bool(self):
        guard = _guard()
        result = guard.process_round(0, _cids(5), _random_deltas(n=5))
        assert all(isinstance(f, bool) for f in result.norm_flagged)

    def test_zero_delta_has_zero_norm(self):
        guard = _guard()
        deltas = [np.zeros(10, dtype=np.float32)] * 4
        result = guard.process_round(0, _cids(4), deltas)
        assert all(n == pytest.approx(0.0) for n in result.l2_norms)


# ---------------------------------------------------------------------------
# Collusion detection
# ---------------------------------------------------------------------------


class TestCollusionDetection:
    def test_detects_colluding_cluster(self):
        deltas = _colluding_setup(n_clients=6, n_colluders=3)
        guard = _guard(sim_threshold=0.90, min_cluster_size=2)
        result = guard.process_round(0, _cids(6), deltas)
        assert len(result.flagged_clusters) >= 1

    def test_colluders_have_higher_collusion_scores(self):
        deltas = _colluding_setup(n_clients=6, n_colluders=3)
        guard = _guard(sim_threshold=0.90, min_cluster_size=2)
        result = guard.process_round(0, _cids(6), deltas)
        colluder_scores = result.collusion_scores[:3]
        honest_scores = result.collusion_scores[3:]
        assert max(colluder_scores) > max(honest_scores)

    def test_no_clusters_with_independent_updates(self):
        rng = np.random.default_rng(42)
        # Purely random deltas should not form clusters
        deltas = [rng.standard_normal(50).astype(np.float32) for _ in range(8)]
        guard = _guard(sim_threshold=0.90, min_cluster_size=2)
        result = guard.process_round(0, _cids(8), deltas)
        assert result.flagged_clusters == []

    def test_similarity_matrix_is_n_by_n(self):
        n = 5
        guard = _guard()
        result = guard.process_round(0, _cids(n), _random_deltas(n=n))
        assert len(result.similarity_matrix) == n
        assert all(len(row) == n for row in result.similarity_matrix)


# ---------------------------------------------------------------------------
# Anomaly scores
# ---------------------------------------------------------------------------


class TestAnomalyScores:
    def test_scores_in_range(self):
        guard = _guard()
        result = guard.process_round(0, _cids(6), _random_deltas(n=6))
        assert all(0.0 <= s <= 1.0 for s in result.anomaly_scores)

    def test_outlier_has_higher_score(self):
        rng = np.random.default_rng(0)
        base = [rng.standard_normal(20).astype(np.float32) for _ in range(6)]
        outlier = (rng.standard_normal(20) * 50).astype(np.float32)
        deltas = base + [outlier]
        guard = _guard()
        result = guard.process_round(0, _cids(7), deltas)
        assert result.anomaly_scores[-1] > result.anomaly_scores[0]


# ---------------------------------------------------------------------------
# Trust scores
# ---------------------------------------------------------------------------


class TestTrustScores:
    def test_trust_scores_in_range(self):
        guard = _guard()
        result = guard.process_round(0, _cids(5), _random_deltas(n=5))
        assert all(0.0 <= s <= 1.0 for s in result.trust_scores)

    def test_trust_accumulates_over_rounds(self):
        deltas = _colluding_setup(n_clients=5, n_colluders=2)
        guard = _guard(sim_threshold=0.90)
        r1 = guard.process_round(0, _cids(5), deltas)
        r2 = guard.process_round(1, _cids(5), deltas)
        # Colluders' trust should grow over rounds
        assert max(r2.trust_scores[:2]) >= max(r1.trust_scores[:2])


# ---------------------------------------------------------------------------
# Client ranking
# ---------------------------------------------------------------------------


class TestRanking:
    def test_ranked_clients_contains_all(self):
        n = 5
        guard = _guard()
        result = guard.process_round(0, _cids(n), _random_deltas(n=n))
        assert set(result.ranked_clients) == set(_cids(n))

    def test_most_trusted_is_first(self):
        """Client with lowest trust score should appear first."""
        guard = _guard(exclude_flagged_clients=False, trust_score_weight=1.0,
                       trust_score_decay=0.0)
        # Drive one client's score high
        cids = _cids(4)
        deltas = _random_deltas(n=4)
        # Artificially inflate client_00's anomaly by giving it a huge norm
        deltas[0] = np.ones(20, dtype=np.float32) * 1000.0
        result = guard.process_round(0, cids, deltas)
        # client_00 should be last (most suspicious)
        assert result.ranked_clients[-1] == "client_00"


# ---------------------------------------------------------------------------
# Excluded clients
# ---------------------------------------------------------------------------


class TestExcludedClients:
    def test_excluded_empty_when_flag_off(self):
        guard = _guard(exclude_flagged_clients=False)
        deltas = _colluding_setup(n_clients=6, n_colluders=3)
        result = guard.process_round(0, _cids(6), deltas)
        assert result.excluded_clients == []

    def test_excluded_populated_when_flag_on(self):
        # Force a norm outlier by scaling one delta massively
        rng = np.random.default_rng(0)
        base = [rng.standard_normal(20).astype(np.float32) for _ in range(5)]
        base.append(np.ones(20, dtype=np.float32) * 1000.0)
        guard = _guard(
            exclude_flagged_clients=True,
            norm_outlier_threshold_z=1.0,  # low threshold to catch the outlier
        )
        result = guard.process_round(0, _cids(6), base)
        assert len(result.excluded_clients) > 0

    def test_excluded_are_subset_of_client_ids(self):
        rng = np.random.default_rng(0)
        base = [rng.standard_normal(20).astype(np.float32) for _ in range(6)]
        guard = _guard(exclude_flagged_clients=True)
        result = guard.process_round(0, _cids(6), base)
        assert all(cid in _cids(6) for cid in result.excluded_clients)


# ---------------------------------------------------------------------------
# Empty round (no clients)
# ---------------------------------------------------------------------------


class TestEmptyRound:
    def test_empty_result_returned(self):
        guard = _guard()
        result = guard.process_round(5, [], [])
        assert result.round_num == 5
        assert result.client_ids == []
        assert result.l2_norms == []
        assert result.ranked_clients == []


# ---------------------------------------------------------------------------
# Structured logging integration
# ---------------------------------------------------------------------------


class TestLogging:
    def test_logger_called_if_provided(self):
        mock_logger = MagicMock()
        guard = _guard(sentinel_logger=mock_logger)
        guard.process_round(0, _cids(4), _random_deltas(n=4))
        mock_logger.log.assert_called()

    def test_logger_not_called_if_none(self):
        guard = _guard(sentinel_logger=None)
        # Should run without error — no logger attached
        result = guard.process_round(0, _cids(4), _random_deltas(n=4))
        assert result is not None


# ---------------------------------------------------------------------------
# from_config factory
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_basic(self):
        cfg = SimpleNamespace(
            collusion_sim_threshold=0.85,
            collusion_min_cluster_size=2,
            update_guard=None,
        )
        guard = UpdateGuard.from_config(cfg)
        assert isinstance(guard, UpdateGuard)

    def test_from_config_reads_update_guard_subconfig(self):
        ug_cfg = SimpleNamespace(
            sim_threshold=0.92,
            min_cluster_size=3,
            anomaly_method="mad",
            norm_outlier_threshold_z=2.5,
            norm_type="l2",
            exclude_flagged_clients=True,
            trust_score_decay=0.05,
            trust_score_weight=0.3,
            log_similarity_matrix=False,
        )
        cfg = SimpleNamespace(update_guard=ug_cfg)
        guard = UpdateGuard.from_config(cfg)
        assert guard._sim_threshold == pytest.approx(0.92)
        assert guard._min_cluster_size == 3
        assert guard._exclude_flagged is True

    def test_from_config_with_ledger(self, tmp_path):
        from ai.detection.trust_ledger import FileTrustLedger
        ledger = FileTrustLedger(tmp_path / "ledger.jsonl")
        cfg = SimpleNamespace(update_guard=None)
        guard = UpdateGuard.from_config(cfg, ledger=ledger)
        assert guard._ledger is ledger


# ---------------------------------------------------------------------------
# Trust accessor properties
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_trust_manager_accessible(self):
        guard = _guard()
        from ai.detection.trust_score_manager import TrustScoreManager
        assert isinstance(guard.trust_manager, TrustScoreManager)

    def test_anomaly_detector_accessible(self):
        guard = _guard()
        from ai.detection.anomaly_detector import UpdateAnomalyDetector
        assert isinstance(guard.anomaly_detector, UpdateAnomalyDetector)
