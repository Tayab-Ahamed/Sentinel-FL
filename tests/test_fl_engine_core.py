"""
tests/test_fl_engine_core.py — Tests for ai/fl_core/fl_engine.py.

Covers:
  LinearSoftmaxModel: params, predict_proba, predict
  local_train: shape, determinism, gradient descent
  fedavg: weighted average, equal weights
  multi_krum: basic selection, known outlier exclusion
"""
from __future__ import annotations

import numpy as np

from ai.fl_core.fl_engine import LinearSoftmaxModel, fedavg, local_train, multi_krum

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(n_features=10, n_classes=4) -> LinearSoftmaxModel:
    return LinearSoftmaxModel(n_features, n_classes)


def _random_data(n=200, n_features=10, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, n)
    return X, y


# ---------------------------------------------------------------------------
# LinearSoftmaxModel
# ---------------------------------------------------------------------------


class TestLinearSoftmaxModel:
    def test_params_shape(self):
        m = _model(10, 4)
        p = m.get_params()
        assert p.shape == (10 * 4 + 4,)  # W flat + b

    def test_set_get_roundtrip(self):
        m = _model(8, 3)
        p = m.get_params()
        p2 = p + 1.0
        m.set_params(p2)
        np.testing.assert_allclose(m.get_params(), p2)

    def test_predict_proba_sums_to_one(self):
        m = _model(10, 4)
        X = np.random.default_rng(0).standard_normal((50, 10)).astype(np.float32)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(50), atol=1e-5)

    def test_predict_shape(self):
        m = _model(10, 4)
        X = np.zeros((30, 10), dtype=np.float32)
        preds = m.predict(X)
        assert preds.shape == (30,)
        assert preds.min() >= 0
        assert preds.max() < 4

    def test_predict_selects_argmax(self):
        m = _model(5, 3)
        X, _ = _random_data(100, 5, 3, seed=42)
        proba = m.predict_proba(X)
        preds = m.predict(X)
        np.testing.assert_array_equal(preds, proba.argmax(axis=1))

    def test_deterministic_init(self):
        """Two models with same RNG seed should have identical params."""
        m1 = _model(6, 3)
        m2 = _model(6, 3)
        np.testing.assert_array_equal(m1.get_params(), m2.get_params())

    def test_single_sample_predict(self):
        m = _model(10, 4)
        X = np.zeros((1, 10), dtype=np.float32)
        preds = m.predict(X)
        assert preds.shape == (1,)


# ---------------------------------------------------------------------------
# local_train
# ---------------------------------------------------------------------------


class TestLocalTrain:
    def test_output_shape(self):
        m = _model(10, 4)
        init_p = m.get_params()
        X, y = _random_data(100, 10, 4, 0)
        new_p = local_train(init_p, 10, 4, X, y, epochs=3, lr=0.1)
        assert new_p.shape == init_p.shape

    def test_params_change_after_training(self):
        m = _model(10, 4)
        init_p = m.get_params()
        X, y = _random_data(100, 10, 4, 0)
        new_p = local_train(init_p.copy(), 10, 4, X, y, epochs=5, lr=0.1)
        assert not np.allclose(new_p, init_p), "Params must change after training"

    def test_init_params_not_mutated(self):
        m = _model(10, 4)
        init_p = m.get_params().copy()
        X, y = _random_data(100, 10, 4, 0)
        local_train(m.get_params(), 10, 4, X, y, epochs=5, lr=0.1)
        # original init_p unchanged
        np.testing.assert_array_equal(m.get_params(), init_p)

    def test_loss_decreases_on_separable_data(self):
        """Accuracy on separable data should improve with training."""
        np.random.seed(42)
        n, nf, nc = 300, 5, 2
        X = np.vstack([np.random.randn(n // 2, nf) + 2, np.random.randn(n // 2, nf) - 2]).astype(
            np.float32
        )
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        m = LinearSoftmaxModel(nf, nc)
        p0 = m.get_params()
        m.set_params(p0)
        acc_before = (m.predict(X) == y).mean()
        new_p = local_train(p0, nf, nc, X, y, epochs=20, lr=0.3)
        m.set_params(new_p)
        acc_after = (m.predict(X) == y).mean()
        assert acc_after >= acc_before


# ---------------------------------------------------------------------------
# fedavg
# ---------------------------------------------------------------------------


class TestFedAvg:
    def test_equal_weights_is_mean(self):
        u1 = np.array([1.0, 2.0, 3.0])
        u2 = np.array([3.0, 4.0, 5.0])
        result = fedavg([u1, u2], [1, 1])
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_weighted_average(self):
        u1 = np.array([0.0, 0.0])
        u2 = np.array([1.0, 1.0])
        result = fedavg([u1, u2], [3, 1])  # 75% u1, 25% u2
        np.testing.assert_allclose(result, [0.25, 0.25])

    def test_single_client(self):
        u = np.array([5.0, 6.0, 7.0])
        result = fedavg([u], [100])
        np.testing.assert_allclose(result, u)

    def test_output_shape_matches_inputs(self):
        updates = [np.random.randn(20) for _ in range(5)]
        weights = [10] * 5
        result = fedavg(updates, weights)
        assert result.shape == (20,)

    def test_weights_normalized(self):
        """Result must not depend on absolute weight magnitudes."""
        u1 = np.ones(5)
        u2 = np.zeros(5)
        r1 = fedavg([u1, u2], [1, 1])
        r2 = fedavg([u1, u2], [100, 100])
        np.testing.assert_allclose(r1, r2)


# ---------------------------------------------------------------------------
# multi_krum
# ---------------------------------------------------------------------------


class TestMultiKrum:
    def test_basic_selection(self):
        updates = [np.ones(10) * i for i in range(8)]
        agg, selected = multi_krum(updates, num_malicious_assumed=2, num_to_select=5)
        assert len(selected) == 5
        assert len(set(selected)) == 5  # unique
        assert agg.shape == (10,)

    def test_excludes_outlier(self):
        """An obvious outlier (large norm) should not be selected."""
        rng = np.random.default_rng(0)
        honest = [rng.standard_normal(20).astype(np.float32) for _ in range(8)]
        outlier = np.ones(20, dtype=np.float32) * 1000.0
        updates = honest + [outlier]  # index 8 = outlier
        _agg, selected = multi_krum(updates, num_malicious_assumed=1, num_to_select=8)
        assert 8 not in selected, "Outlier (index 8) should be excluded"

    def test_aggregate_is_mean_of_selected(self):
        updates = [np.ones(5) * i for i in range(6)]
        agg, selected = multi_krum(updates, num_malicious_assumed=1, num_to_select=4)
        expected = np.mean([updates[i] for i in selected], axis=0)
        np.testing.assert_allclose(agg, expected)

    def test_selected_indices_sorted(self):
        updates = [np.random.randn(10) for _ in range(6)]
        _agg, selected = multi_krum(updates, num_malicious_assumed=1, num_to_select=3)
        assert selected == sorted(selected)

    def test_select_all_minus_one(self):
        updates = [np.zeros(5)] * 6
        agg, selected = multi_krum(updates, num_malicious_assumed=1, num_to_select=5)
        assert len(selected) == 5
