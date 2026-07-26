"""
tests/test_gradient_extractor.py — Unit tests for GradientExtractor.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.detection.gradient_extractor import (
    GradientExtractor,
    extract_delta,
    flatten_params,
    normalize_all_deltas,
    normalize_delta,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _params(shapes=((10,), (5,)), seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


def _perturbed(params, noise=0.1, seed=1):
    rng = np.random.default_rng(seed)
    return [p + rng.standard_normal(p.shape).astype(np.float32) * noise for p in params]


# ---------------------------------------------------------------------------
# extract_delta
# ---------------------------------------------------------------------------


class TestExtractDelta:
    def test_output_is_flat_1d(self):
        p, q = _params(), _perturbed(_params())
        d = extract_delta(p, q)
        assert d.ndim == 1

    def test_output_size_matches_total_params(self):
        shapes = ((10,), (5, 3), (7,))
        p = _params(shapes)
        q = _perturbed(p)
        d = extract_delta(p, q)
        total = sum(np.prod(s) for s in shapes)
        assert len(d) == total

    def test_zero_delta_for_identical_params(self):
        p = _params()
        d = extract_delta(p, p)
        np.testing.assert_allclose(d, 0.0, atol=1e-6)

    def test_sign_is_correct(self):
        p = [np.zeros(5, dtype=np.float32)]
        q = [np.ones(5, dtype=np.float32)]
        d = extract_delta(p, q)
        np.testing.assert_allclose(d, 1.0)

    def test_dtype_is_float32(self):
        p, q = _params(), _perturbed(_params())
        assert extract_delta(p, q).dtype == np.float32

    def test_length_mismatch_raises(self):
        p = _params(((10,), (5,)))
        q = _params(((10,),))
        with pytest.raises(ValueError, match="length mismatch"):
            extract_delta(p, q)

    def test_original_params_not_mutated(self):
        p = _params()
        p_copy = [x.copy() for x in p]
        q = _perturbed(p)
        extract_delta(p, q)
        for a, b in zip(p, p_copy):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# flatten_params
# ---------------------------------------------------------------------------


class TestFlattenParams:
    def test_output_is_1d(self):
        p = _params(((3, 4), (5,)))
        flat = flatten_params(p)
        assert flat.ndim == 1

    def test_total_length(self):
        p = _params(((3, 4), (5,)))
        assert len(flatten_params(p)) == 3 * 4 + 5

    def test_dtype_float32(self):
        assert flatten_params(_params()).dtype == np.float32


# ---------------------------------------------------------------------------
# normalize_delta
# ---------------------------------------------------------------------------


class TestNormalizeDelta:
    def test_l2_normalized_has_unit_norm(self):
        d = np.array([3.0, 4.0], dtype=np.float32)
        n = normalize_delta(d, "l2")
        assert abs(np.linalg.norm(n) - 1.0) < 1e-5

    def test_l1_normalized_sums_to_one(self):
        d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        n = normalize_delta(d, "l1")
        assert abs(np.sum(np.abs(n)) - 1.0) < 1e-5

    def test_linf_max_abs_is_one(self):
        d = np.array([1.0, -5.0, 3.0], dtype=np.float32)
        n = normalize_delta(d, "linf")
        assert abs(np.max(np.abs(n)) - 1.0) < 1e-5

    def test_zero_vector_returns_zeros(self):
        d = np.zeros(10, dtype=np.float32)
        n = normalize_delta(d, "l2")
        np.testing.assert_array_equal(n, 0.0)

    def test_invalid_norm_type_raises(self):
        with pytest.raises(ValueError, match="norm_type"):
            normalize_delta(np.ones(5, dtype=np.float32), "l5")

    def test_output_dtype_is_float32(self):
        d = np.ones(4, dtype=np.float64)
        assert normalize_delta(d, "l2").dtype == np.float32


# ---------------------------------------------------------------------------
# normalize_all_deltas
# ---------------------------------------------------------------------------


class TestNormalizeAllDeltas:
    def test_returns_same_count(self):
        deltas = [np.ones(5, dtype=np.float32) for _ in range(6)]
        result = normalize_all_deltas(deltas)
        assert len(result) == 6

    def test_each_element_is_normalized(self):
        rng = np.random.default_rng(0)
        deltas = [rng.standard_normal(10).astype(np.float32) for _ in range(4)]
        normed = normalize_all_deltas(deltas, "l2")
        for n in normed:
            assert abs(np.linalg.norm(n) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# GradientExtractor (stateful)
# ---------------------------------------------------------------------------


class TestGradientExtractor:
    def test_total_params(self):
        p = _params(((10,), (5, 3)))
        ge = GradientExtractor(p)
        assert ge.total_params == 25

    def test_extract_round_deltas_count(self):
        """extract_round_deltas returns one delta per result."""
        from unittest.mock import MagicMock

        flwr_common = pytest.importorskip("flwr.common")
        ndarrays_to_parameters = flwr_common.ndarrays_to_parameters

        p = _params()
        ge = GradientExtractor(p)

        def _make_res(noise):
            m = MagicMock()
            q = _perturbed(p, noise=noise)
            m.parameters = ndarrays_to_parameters(q)
            return m

        results = [(MagicMock(), _make_res(0.1)) for _ in range(5)]
        deltas = ge.extract_round_deltas(results)
        assert len(deltas) == 5

    def test_extract_round_deltas_shape(self):
        from unittest.mock import MagicMock

        flwr_common = pytest.importorskip("flwr.common")
        ndarrays_to_parameters = flwr_common.ndarrays_to_parameters

        shapes = ((8,), (4, 3))
        p = _params(shapes)
        ge = GradientExtractor(p)
        q = _perturbed(p)
        m = MagicMock()
        m.parameters = ndarrays_to_parameters(q)
        deltas = ge.extract_round_deltas([(MagicMock(), m)])
        total = sum(np.prod(s) for s in shapes)
        assert deltas[0].shape == (total,)

    def test_update_params_changes_prev(self):
        p = _params()
        ge = GradientExtractor(p)
        q = _perturbed(p)
        ge.update_params(q)
        for a, b in zip(ge.prev_params, q):
            np.testing.assert_array_equal(a, b)

    def test_prev_params_are_copies(self):
        """Modifying original params after init must not affect stored params."""
        p = _params()
        ge = GradientExtractor(p)
        p[0][:] = 999.0
        assert not np.allclose(ge.prev_params[0], 999.0)
