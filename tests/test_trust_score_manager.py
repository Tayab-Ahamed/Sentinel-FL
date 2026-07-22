"""
tests/test_trust_score_manager.py — Unit tests for TrustScoreManager.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.detection.trust_score_manager import TrustScoreManager, _zero_score
from ai.fl_core.schemas import TrustScore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mgr(decay=0.1, weight=0.5, ledger=None):
    return TrustScoreManager(decay_rate=decay, weight=weight, ledger=ledger)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError, match="decay_rate"):
            TrustScoreManager(decay_rate=1.5)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="weight"):
            TrustScoreManager(weight=0.0)

    def test_zero_decay_accepted(self):
        m = TrustScoreManager(decay_rate=0.0)
        assert m._decay == 0.0


# ---------------------------------------------------------------------------
# update (single client)
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_returns_trust_score(self):
        m = _mgr()
        ts = m.update("c0", anomaly_score=0.4, round_num=1)
        assert isinstance(ts, TrustScore)

    def test_score_is_in_range(self):
        m = _mgr()
        ts = m.update("c0", 0.8, round_num=1)
        assert 0.0 <= ts.score <= 1.0

    def test_zero_anomaly_keeps_score_at_zero_first_round(self):
        m = _mgr()
        ts = m.update("c0", 0.0, round_num=1)
        assert ts.score == pytest.approx(0.0)

    def test_score_increases_with_anomaly(self):
        m = _mgr(decay=0.0, weight=0.5)
        ts1 = m.update("c0", 0.5, round_num=1)
        ts2 = m.update("c0", 0.5, round_num=2)
        assert ts2.score > ts1.score

    def test_score_capped_at_one(self):
        m = _mgr(decay=0.0, weight=1.0)
        for _ in range(10):
            ts = m.update("c0", 1.0, round_num=1)
        assert ts.score <= 1.0

    def test_subject_id_set(self):
        m = _mgr()
        ts = m.update("alice", 0.3, round_num=3)
        assert ts.subject_id == "alice"

    def test_last_updated_round_set(self):
        m = _mgr()
        ts = m.update("c0", 0.2, round_num=7)
        assert ts.last_updated_round == 7

    def test_update_stored_and_retrievable(self):
        m = _mgr()
        m.update("c0", 0.5, round_num=1)
        stored = m.get_score("c0")
        assert stored is not None
        assert 0.0 < stored.score <= 1.0

    def test_unknown_client_returns_none(self):
        m = _mgr()
        assert m.get_score("ghost") is None


# ---------------------------------------------------------------------------
# update_batch
# ---------------------------------------------------------------------------


class TestUpdateBatch:
    def test_returns_one_score_per_client(self):
        m = _mgr()
        cids = ["c0", "c1", "c2"]
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        results = m.update_batch(cids, scores, round_num=1)
        assert len(results) == 3

    def test_all_scores_in_range(self):
        m = _mgr()
        cids = [f"c{i}" for i in range(5)]
        scores = np.random.default_rng(0).uniform(0, 1, 5).astype(np.float32)
        results = m.update_batch(cids, scores, round_num=1)
        for ts in results:
            assert 0.0 <= ts.score <= 1.0

    def test_order_preserved(self):
        m = _mgr()
        cids = ["alpha", "beta", "gamma"]
        results = m.update_batch(cids, [0.1, 0.5, 0.9], round_num=1)
        assert [ts.subject_id for ts in results] == cids


# ---------------------------------------------------------------------------
# decay_all
# ---------------------------------------------------------------------------


class TestDecayAll:
    def test_decay_reduces_scores(self):
        m = _mgr(decay=0.2)
        m.update("c0", 0.8, round_num=1)
        score_before = m.get_score("c0").score
        m.decay_all(round_num=2)
        score_after = m.get_score("c0").score
        assert score_after < score_before

    def test_zero_decay_no_change(self):
        m = _mgr(decay=0.0)
        m.update("c0", 0.5, round_num=1)
        before = m.get_score("c0").score
        m.decay_all(round_num=2)
        after = m.get_score("c0").score
        assert after == pytest.approx(before, rel=1e-5)

    def test_score_never_goes_negative(self):
        m = _mgr(decay=1.0)
        m.update("c0", 0.5, round_num=1)
        m.decay_all(round_num=2)
        assert m.get_score("c0").score >= 0.0


# ---------------------------------------------------------------------------
# rank_clients
# ---------------------------------------------------------------------------


class TestRankClients:
    def test_most_trusted_first(self):
        m = _mgr(decay=0.0, weight=1.0)
        m.update("c_high", 0.9, round_num=1)
        m.update("c_low", 0.1, round_num=1)
        m.update("c_mid", 0.5, round_num=1)
        ranked = m.rank_clients(["c_high", "c_low", "c_mid"])
        assert ranked[0] == "c_low"  # lowest score = most trusted

    def test_subset_only_ranked(self):
        m = _mgr()
        for cid in ["c0", "c1", "c2", "c3"]:
            m.update(cid, 0.3, round_num=1)
        ranked = m.rank_clients(["c0", "c2"])
        assert set(ranked) == {"c0", "c2"}

    def test_unknown_clients_get_zero_score(self):
        m = _mgr()
        # No updates yet, but rank_clients with unknown IDs should still work
        ranked = m.rank_clients(["phantom"])
        assert ranked == ["phantom"]


# ---------------------------------------------------------------------------
# get_suspicious_clients
# ---------------------------------------------------------------------------


class TestGetSuspiciousClients:
    def test_high_score_client_flagged(self):
        m = _mgr(decay=0.0, weight=1.0)
        m.update("c_bad", 0.9, round_num=1)
        m.update("c_good", 0.05, round_num=1)
        sus = m.get_suspicious_clients(threshold=0.4)
        assert "c_bad" in sus
        assert "c_good" not in sus

    def test_all_below_threshold_returns_empty(self):
        m = _mgr()
        for cid in ["c0", "c1"]:
            m.update(cid, 0.0, round_num=1)
        assert m.get_suspicious_clients(threshold=0.5) == []


# ---------------------------------------------------------------------------
# score_vector
# ---------------------------------------------------------------------------


class TestScoreVector:
    def test_shape_matches_client_list(self):
        m = _mgr()
        cids = ["c0", "c1", "c2"]
        m.update_batch(cids, [0.2, 0.4, 0.6], round_num=1)
        vec = m.score_vector(cids)
        assert vec.shape == (3,)

    def test_dtype_float32(self):
        m = _mgr()
        m.update("c0", 0.5, round_num=1)
        assert m.score_vector(["c0"]).dtype == np.float32

    def test_unknown_client_score_is_zero(self):
        m = _mgr()
        vec = m.score_vector(["unknown"])
        assert vec[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_returns_dict(self):
        m = _mgr()
        m.update("c0", 0.5, round_num=1)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "c0" in d

    def test_values_are_floats(self):
        m = _mgr()
        m.update("c0", 0.3, round_num=1)
        d = m.to_dict()
        assert isinstance(d["c0"], float)


# ---------------------------------------------------------------------------
# _zero_score helper
# ---------------------------------------------------------------------------


def test_zero_score_helper():
    ts = _zero_score("x")
    assert ts.score == 0.0
    assert ts.subject_id == "x"
