"""
tests/test_partitioning.py — Unit tests for IIDPartitioner and DirichletPartitioner.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.training.partitioning import DirichletPartitioner, IIDPartitioner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_dataset() -> tuple[np.ndarray, np.ndarray]:
    """100 samples, 5 classes, evenly distributed."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = np.repeat(np.arange(5), 20).astype(np.int64)
    rng.shuffle(y)
    return X, y


@pytest.fixture
def medium_dataset() -> tuple[np.ndarray, np.ndarray]:
    """10_000 samples, 10 classes (like MNIST subset)."""
    rng = np.random.default_rng(1)
    n = 10_000
    X = rng.standard_normal((n, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, size=n).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# IIDPartitioner tests
# ---------------------------------------------------------------------------


class TestIIDPartitioner:
    def test_returns_correct_n_clients(self, small_dataset):
        X, y = small_dataset
        parts = IIDPartitioner().partition(X, y, 5, seed=42)
        assert len(parts) == 5

    def test_indices_cover_all_samples(self, small_dataset):
        X, y = small_dataset
        parts = IIDPartitioner().partition(X, y, 5, seed=42)
        all_idx = np.concatenate(parts)
        assert set(all_idx.tolist()) == set(range(len(X)))

    def test_no_duplicate_indices(self, small_dataset):
        X, y = small_dataset
        parts = IIDPartitioner().partition(X, y, 5, seed=42)
        all_idx = np.concatenate(parts)
        assert len(all_idx) == len(set(all_idx.tolist()))

    def test_equal_sizes_when_divisible(self):
        """100 samples / 5 clients = 20 each."""
        X = np.zeros((100, 4), dtype=np.float32)
        y = np.zeros(100, dtype=np.int64)
        parts = IIDPartitioner().partition(X, y, 5, seed=0)
        sizes = [len(p) for p in parts]
        assert sizes == [20, 20, 20, 20, 20]

    def test_near_equal_sizes_when_indivisible(self):
        """101 samples / 5 clients → sizes in {20, 21}."""
        X = np.zeros((101, 4), dtype=np.float32)
        y = np.zeros(101, dtype=np.int64)
        parts = IIDPartitioner().partition(X, y, 5, seed=0)
        sizes = [len(p) for p in parts]
        assert max(sizes) - min(sizes) <= 1

    def test_reproducible_with_same_seed(self, small_dataset):
        X, y = small_dataset
        p1 = IIDPartitioner().partition(X, y, 4, seed=7)
        p2 = IIDPartitioner().partition(X, y, 4, seed=7)
        for a, b in zip(p1, p2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_splits(self, small_dataset):
        X, y = small_dataset
        p1 = IIDPartitioner().partition(X, y, 4, seed=1)
        p2 = IIDPartitioner().partition(X, y, 4, seed=2)
        assert not all(
            np.array_equal(a, b) for a, b in zip(p1, p2)
        ), "Different seeds should produce different splits"

    def test_raises_on_too_few_samples(self):
        X = np.zeros((3, 4), dtype=np.float32)
        y = np.zeros(3, dtype=np.int64)
        with pytest.raises(ValueError, match="Cannot partition"):
            IIDPartitioner().partition(X, y, 10, seed=0)

    def test_raises_on_empty_dataset(self):
        X = np.zeros((0, 4), dtype=np.float32)
        y = np.zeros(0, dtype=np.int64)
        with pytest.raises(ValueError):
            IIDPartitioner().partition(X, y, 3, seed=0)

    def test_single_client(self, small_dataset):
        X, y = small_dataset
        parts = IIDPartitioner().partition(X, y, 1, seed=0)
        assert len(parts) == 1
        assert len(parts[0]) == len(X)


# ---------------------------------------------------------------------------
# DirichletPartitioner tests
# ---------------------------------------------------------------------------


class TestDirichletPartitioner:
    def test_returns_correct_n_clients(self, small_dataset):
        X, y = small_dataset
        parts = DirichletPartitioner(alpha=0.5, n_classes=5).partition(X, y, 5, seed=42)
        assert len(parts) == 5

    def test_indices_cover_all_samples(self, small_dataset):
        X, y = small_dataset
        parts = DirichletPartitioner(alpha=0.5, n_classes=5).partition(X, y, 5, seed=42)
        all_idx = np.concatenate(parts)
        assert set(all_idx.tolist()) == set(range(len(X)))

    def test_no_duplicate_indices(self, small_dataset):
        X, y = small_dataset
        parts = DirichletPartitioner(alpha=0.5, n_classes=5).partition(X, y, 5, seed=42)
        all_idx = np.concatenate(parts)
        assert len(all_idx) == len(set(all_idx.tolist()))

    def test_reproducible_with_same_seed(self, small_dataset):
        X, y = small_dataset
        p = DirichletPartitioner(alpha=0.5, n_classes=5)
        p1 = p.partition(X, y, 4, seed=99)
        p2 = p.partition(X, y, 4, seed=99)
        for a, b in zip(p1, p2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_splits(self, small_dataset):
        X, y = small_dataset
        p = DirichletPartitioner(alpha=0.5, n_classes=5)
        p1 = p.partition(X, y, 4, seed=1)
        p2 = p.partition(X, y, 4, seed=2)
        assert not all(np.array_equal(a, b) for a, b in zip(p1, p2))

    def test_low_alpha_produces_skewed_distribution(self, medium_dataset):
        """With alpha=0.01, class distributions should be highly skewed."""
        X, y = medium_dataset
        parts = DirichletPartitioner(alpha=0.01, n_classes=10).partition(X, y, 5, seed=42)
        # Count dominant classes per client
        dominant_counts = []
        for idx in parts:
            labels = y[idx]
            counts = np.bincount(labels, minlength=10)
            dominant_counts.append(counts.max() / len(labels))
        # With extreme alpha, at least one client should have >50% from one class
        assert max(dominant_counts) > 0.5, "Low alpha should produce skewed partitions"

    def test_high_alpha_approaches_iid(self, medium_dataset):
        """With alpha=100, all clients should have roughly uniform class distribution."""
        X, y = medium_dataset
        parts = DirichletPartitioner(alpha=100.0, n_classes=10).partition(X, y, 10, seed=42)
        for idx in parts:
            labels = y[idx]
            if len(labels) < 10:
                continue
            counts = np.bincount(labels, minlength=10)
            fractions = counts / len(labels)
            # Each class should be within ±15% of 10% expected fraction
            assert np.all(np.abs(fractions - 0.1) < 0.15), (
                f"High alpha should approximate IID. Got fractions: {fractions}"
            )

    def test_raises_on_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be"):
            DirichletPartitioner(alpha=0.0)

    def test_minimum_samples_guaranteed(self):
        """No client should be completely empty even with extreme alpha."""
        X = np.zeros((50, 4), dtype=np.float32)
        y = np.array([0] * 48 + [1] * 2, dtype=np.int64)
        parts = DirichletPartitioner(alpha=0.001, n_classes=2).partition(X, y, 5, seed=42)
        for i, p in enumerate(parts):
            assert len(p) >= 1, f"Client {i} got 0 samples"

    def test_auto_detect_n_classes(self, small_dataset):
        """n_classes=None should auto-detect from y.max()+1."""
        X, y = small_dataset
        # 5-class dataset, don't pass n_classes explicitly
        parts = DirichletPartitioner(alpha=0.5).partition(X, y, 4, seed=0)
        assert len(parts) == 4
        all_idx = np.concatenate(parts)
        assert len(all_idx) == len(X)
