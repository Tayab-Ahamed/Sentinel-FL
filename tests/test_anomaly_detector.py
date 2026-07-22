"""
tests/test_anomaly_detector.py — Unit tests for UpdateAnomalyDetector.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.detection.anomaly_detector import UpdateAnomalyDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normal_deltas(n=8, dim=20, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim).astype(np.float32) * scale for _ in range(n)]


def _outlier_delta(dim=20, scale=100.0, seed=99):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32) * scale


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="anomaly_method"):
            UpdateAnomalyDetector(anomaly_method="unknown")

    def test_default_method_is_zscore(self):
        d = UpdateAnomalyDetector()
        assert d.method == "zscore"

    def test_mad_method_accepted(self):
        d = UpdateAnomalyDetector(anomaly_method="mad")
        assert d.method == "mad"

    def test_initial_reference_pool_empty(self):
        d = UpdateAnomalyDetector()
        assert d.reference_pool_size == 0


# ---------------------------------------------------------------------------
# score_all — within-round statistics (no fit() called)
# ---------------------------------------------------------------------------


class TestScoreAllWithinRound:
    def test_output_shape(self):
        d = UpdateAnomalyDetector()
        deltas = _normal_deltas(n=6)
        scores = d.score_all(deltas)
        assert scores.shape == (6,)

    def test_scores_in_range(self):
        d = UpdateAnomalyDetector()
        scores = d.score_all(_normal_deltas(n=10))
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_outlier_scores_higher_than_normal(self):
        d = UpdateAnomalyDetector()
        deltas = _normal_deltas(n=7) + [_outlier_delta()]
        scores = d.score_all(deltas)
        # Outlier (last) should score higher than the mean of normal clients
        assert scores[-1] > scores[:-1].mean()

    def test_dtype_float32(self):
        d = UpdateAnomalyDetector()
        scores = d.score_all(_normal_deltas())
        assert scores.dtype == np.float32

    def test_empty_returns_empty(self):
        d = UpdateAnomalyDetector()
        assert len(d.score_all([])) == 0

    def test_mad_method_also_scores_in_range(self):
        d = UpdateAnomalyDetector(anomaly_method="mad")
        scores = d.score_all(_normal_deltas(n=8))
        assert np.all((scores >= 0.0) & (scores <= 1.0))


# ---------------------------------------------------------------------------
# score_all — with reference pool (fit() called)
# ---------------------------------------------------------------------------


class TestScoreAllWithReference:
    def test_scores_still_in_range_with_reference(self):
        d = UpdateAnomalyDetector()
        # Fit on clean data from previous rounds
        for _ in range(3):
            d.fit(_normal_deltas(n=5))
        scores = d.score_all(_normal_deltas(n=6))
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    def test_outlier_scored_high_relative_to_reference(self):
        d = UpdateAnomalyDetector()
        # Fit on low-norm deltas
        d.fit([np.ones(20, dtype=np.float32) * 0.01 for _ in range(10)])
        # Feed one massive outlier — all normal scores will be ~1.0 from
        # sigmoid, outlier at least as high
        deltas = _normal_deltas(n=5, scale=0.01) + [_outlier_delta(scale=50.0)]
        scores = d.score_all(deltas)
        assert scores[-1] >= scores[:-1].mean()

    def test_fit_grows_reference_pool(self):
        d = UpdateAnomalyDetector()
        d.fit(_normal_deltas(n=4))
        assert d.reference_pool_size == 4
        d.fit(_normal_deltas(n=3))
        assert d.reference_pool_size == 7

    def test_reset_clears_pool(self):
        d = UpdateAnomalyDetector()
        d.fit(_normal_deltas(n=5))
        d.reset_reference()
        assert d.reference_pool_size == 0


# ---------------------------------------------------------------------------
# score (single delta)
# ---------------------------------------------------------------------------


class TestScoreSingle:
    def test_returns_float(self):
        d = UpdateAnomalyDetector()
        # Need at least 2 deltas so within-round stats work; call score_all first
        # to warm up detector (score() uses score_all([delta]))
        val = d.score(np.ones(10, dtype=np.float32))
        assert isinstance(val, float)

    def test_in_range(self):
        d = UpdateAnomalyDetector()
        val = d.score(np.ones(10, dtype=np.float32))
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# flag
# ---------------------------------------------------------------------------


class TestFlag:
    def test_returns_bool_array(self):
        d = UpdateAnomalyDetector()
        deltas = _normal_deltas(n=6)
        flagged = d.flag(deltas)
        assert flagged.dtype == bool

    def test_empty_returns_empty(self):
        d = UpdateAnomalyDetector()
        flagged = d.flag([])
        assert len(flagged) == 0

    def test_outlier_flagged_with_low_threshold(self):
        d = UpdateAnomalyDetector(threshold_z=1.0)
        # All identical except outlier → outlier must be flagged
        base = np.ones(20, dtype=np.float32)
        outlier = np.ones(20, dtype=np.float32) * 500.0
        deltas = [base] * 7 + [outlier]
        flagged = d.flag(deltas, threshold_z=1.0)
        assert flagged[-1]

    def test_identical_deltas_not_flagged(self):
        d = UpdateAnomalyDetector(threshold_z=3.0)
        deltas = [np.ones(10, dtype=np.float32)] * 5
        flagged = d.flag(deltas)
        assert not np.any(flagged)

    def test_override_threshold_respected(self):
        d = UpdateAnomalyDetector(threshold_z=100.0)
        deltas = _normal_deltas(n=5) + [_outlier_delta()]
        # Override to very low threshold → should flag the outlier
        flagged = d.flag(deltas, threshold_z=0.5)
        assert np.any(flagged)
