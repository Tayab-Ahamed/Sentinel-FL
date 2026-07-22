"""
tests/test_norm_calculator.py — Unit tests for NormCalculator functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.detection.norm_calculator import (
    compute_l1_norms,
    compute_l2_norms,
    compute_linf_norms,
    compute_norm_mad_scores,
    compute_norm_zscores,
    compute_norms,
    flag_norm_outliers,
    norm_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deltas(n=6, dim=20, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim).astype(np.float32) for _ in range(n)]


def _norms_with_outlier():
    # Outlier at 200x normal magnitude ensures z-score well above 3.0
    norms = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 200.0], dtype=np.float32)
    return norms  # last one is an outlier


# ---------------------------------------------------------------------------
# compute_l2_norms
# ---------------------------------------------------------------------------


class TestComputeL2Norms:
    def test_output_shape(self):
        deltas = _deltas(n=5)
        assert compute_l2_norms(deltas).shape == (5,)

    def test_known_value(self):
        d = [np.array([3.0, 4.0], dtype=np.float32)]
        norms = compute_l2_norms(d)
        assert abs(norms[0] - 5.0) < 1e-5

    def test_zero_delta_has_zero_norm(self):
        d = [np.zeros(10, dtype=np.float32)]
        assert compute_l2_norms(d)[0] == pytest.approx(0.0)

    def test_dtype_float32(self):
        assert compute_l2_norms(_deltas()). dtype == np.float32

    def test_all_norms_nonnegative(self):
        assert np.all(compute_l2_norms(_deltas()) >= 0.0)


# ---------------------------------------------------------------------------
# compute_l1_norms
# ---------------------------------------------------------------------------


class TestComputeL1Norms:
    def test_known_value(self):
        d = [np.array([1.0, -2.0, 3.0], dtype=np.float32)]
        assert compute_l1_norms(d)[0] == pytest.approx(6.0)

    def test_shape(self):
        assert compute_l1_norms(_deltas(n=4)).shape == (4,)


# ---------------------------------------------------------------------------
# compute_linf_norms
# ---------------------------------------------------------------------------


class TestComputeLinfNorms:
    def test_known_value(self):
        d = [np.array([1.0, -5.0, 3.0], dtype=np.float32)]
        assert compute_linf_norms(d)[0] == pytest.approx(5.0)

    def test_empty_delta_returns_zero(self):
        d = [np.array([], dtype=np.float32)]
        assert compute_linf_norms(d)[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_norms dispatcher
# ---------------------------------------------------------------------------


class TestComputeNorms:
    def test_dispatches_l2(self):
        deltas = _deltas(n=3)
        np.testing.assert_allclose(
            compute_norms(deltas, "l2"), compute_l2_norms(deltas), rtol=1e-5
        )

    def test_dispatches_l1(self):
        deltas = _deltas(n=3)
        np.testing.assert_allclose(
            compute_norms(deltas, "l1"), compute_l1_norms(deltas), rtol=1e-5
        )

    def test_dispatches_linf(self):
        deltas = _deltas(n=3)
        np.testing.assert_allclose(
            compute_norms(deltas, "linf"), compute_linf_norms(deltas), rtol=1e-5
        )

    def test_unknown_norm_type_raises(self):
        with pytest.raises(ValueError, match="norm_type"):
            compute_norms(_deltas(), "l5")


# ---------------------------------------------------------------------------
# compute_norm_zscores
# ---------------------------------------------------------------------------


class TestComputeNormZScores:
    def test_output_shape(self):
        norms = np.ones(6, dtype=np.float32)
        assert compute_norm_zscores(norms).shape == (6,)

    def test_all_identical_returns_zero(self):
        norms = np.ones(5, dtype=np.float32) * 2.0
        zs = compute_norm_zscores(norms)
        np.testing.assert_allclose(zs, 0.0, atol=1e-5)

    def test_outlier_has_high_zscore(self):
        norms = _norms_with_outlier()
        zs = compute_norm_zscores(norms)
        assert abs(zs[-1]) > 2.0  # outlier at index 5 should stand out

    def test_mean_is_approximately_zero(self):
        norms = np.random.default_rng(0).standard_normal(20).astype(np.float32) + 5.0
        zs = compute_norm_zscores(norms)
        assert abs(float(np.mean(zs))) < 1e-4

    def test_std_is_approximately_one(self):
        norms = np.random.default_rng(0).standard_normal(20).astype(np.float32) + 5.0
        zs = compute_norm_zscores(norms)
        assert abs(float(np.std(zs)) - 1.0) < 1e-4

    def test_empty_returns_empty(self):
        assert len(compute_norm_zscores(np.array([]))) == 0


# ---------------------------------------------------------------------------
# compute_norm_mad_scores
# ---------------------------------------------------------------------------


class TestComputeNormMADScores:
    def test_output_shape(self):
        norms = np.ones(5, dtype=np.float32)
        assert compute_norm_mad_scores(norms).shape == (5,)

    def test_outlier_has_high_mad(self):
        norms = _norms_with_outlier()
        mad_scores = compute_norm_mad_scores(norms)
        assert mad_scores[-1] >= mad_scores[:-1].max()

    def test_empty_returns_empty(self):
        assert len(compute_norm_mad_scores(np.array([]))) == 0

    def test_nonnegative(self):
        norms = np.abs(np.random.default_rng(0).standard_normal(10).astype(np.float32)) + 1.0
        assert np.all(compute_norm_mad_scores(norms) >= 0.0)


# ---------------------------------------------------------------------------
# flag_norm_outliers
# ---------------------------------------------------------------------------


class TestFlagNormOutliers:
    def test_only_outlier_flagged(self):
        norms = _norms_with_outlier()
        # With n=6, the outlier's z-score is ~2.24 (bounded by sqrt(n-1)≈2.24).
        # Use threshold=2.0 to ensure it's flagged; honest clients are at ≈-0.45.
        flagged = flag_norm_outliers(norms, threshold_z=2.0, method="zscore")
        assert flagged[-1]  # last is the outlier
        assert not any(flagged[:-1])

    def test_mad_method_flags_outlier(self):
        norms = _norms_with_outlier()
        flagged = flag_norm_outliers(norms, threshold_z=5.0, method="mad")
        assert flagged[-1]

    def test_returns_bool_array(self):
        norms = np.ones(5, dtype=np.float32)
        flagged = flag_norm_outliers(norms)
        assert flagged.dtype == bool

    def test_no_outliers_all_false(self):
        norms = np.ones(5, dtype=np.float32)  # all identical → std=0 → no outliers
        flagged = flag_norm_outliers(norms, threshold_z=3.0)
        assert not np.any(flagged)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            flag_norm_outliers(np.ones(5, dtype=np.float32), method="hmm")

    def test_high_threshold_no_flags(self):
        norms = _norms_with_outlier()
        flagged = flag_norm_outliers(norms, threshold_z=100.0)
        assert not np.any(flagged)


# ---------------------------------------------------------------------------
# norm_summary
# ---------------------------------------------------------------------------


class TestNormSummary:
    def test_returns_list_of_dicts(self):
        cids = ["c0", "c1", "c2"]
        norms = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        zs = compute_norm_zscores(norms)
        flagged = np.array([False, False, True])
        summary = norm_summary(cids, norms, zs, flagged)
        assert len(summary) == 3
        assert all("client_id" in d and "norm" in d and "flagged" in d for d in summary)

    def test_flagged_field_matches_mask(self):
        cids = ["c0", "c1"]
        norms = np.array([1.0, 2.0], dtype=np.float32)
        zs = np.zeros(2, dtype=np.float32)
        flagged = np.array([False, True])
        summary = norm_summary(cids, norms, zs, flagged)
        assert summary[0]["flagged"] is False
        assert summary[1]["flagged"] is True
