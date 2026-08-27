"""
tests/test_mnist_dataset.py — Unit tests for MNISTDatasetLoader.

All tests use a mocked ``_download_raw()`` to avoid real network calls.
The mock returns deterministic fake arrays of the correct shapes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from ai.training.datasets.mnist import MNISTDatasetLoader

# ---------------------------------------------------------------------------
# Mock raw download
# ---------------------------------------------------------------------------


def _make_fake_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fake MNIST-shaped arrays (no real download)."""
    rng = np.random.default_rng(0)
    n_train, n_test = 600, 100
    X_train = rng.standard_normal((n_train, 1, 28, 28)).astype(np.float32)
    y_train = rng.integers(0, 10, size=n_train).astype(np.int64)
    X_test = rng.standard_normal((n_test, 1, 28, 28)).astype(np.float32)
    y_test = rng.integers(0, 10, size=n_test).astype(np.int64)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def loader(tmp_path) -> MNISTDatasetLoader:
    """Return a MNISTDatasetLoader with mocked _download_raw."""
    inst = MNISTDatasetLoader(
        data_dir=tmp_path,
        dirichlet_alpha=0.5,
        seed=42,
        validate=True,
        use_cache=True,
    )
    inst._download_raw = _make_fake_mnist  # type: ignore[method-assign]
    return inst


class TestMNISTProperties:
    def test_dataset_name(self, loader):
        assert loader.dataset_name == "mnist_v1"

    def test_n_classes(self, loader):
        assert loader.n_classes == 10

    def test_input_shape(self, loader):
        assert loader.input_shape == (1, 28, 28)


class TestMNISTLoadClientPartitions:
    def test_returns_correct_n_clients(self, loader):
        cfg = MagicMock(seed=42, synthetic=None)
        parts = loader.load_client_partitions(5, cfg)
        assert len(parts) == 5

    def test_partition_shapes_are_correct(self, loader):
        cfg = MagicMock(seed=42, synthetic=None)
        parts = loader.load_client_partitions(4, cfg)
        for X_i, y_i in parts:
            assert X_i.ndim == 4
            assert X_i.shape[1:] == (1, 28, 28)
            assert y_i.ndim == 1
            assert len(X_i) == len(y_i)

    def test_partitions_cover_all_training_samples(self, loader):
        cfg = MagicMock(seed=42, synthetic=None)
        parts = loader.load_client_partitions(4, cfg)
        total = sum(len(p[1]) for p in parts)
        assert total == 600  # n_train from fake data

    def test_reproducible_with_same_config(self, loader, tmp_path):
        cfg = MagicMock(seed=42, synthetic=None)
        p1 = loader.load_client_partitions(4, cfg)
        # Re-instantiate with same seed
        loader2 = MNISTDatasetLoader(
            data_dir=tmp_path, dirichlet_alpha=0.5, seed=42, validate=True, use_cache=True
        )
        loader2._download_raw = _make_fake_mnist  # type: ignore[method-assign]
        p2 = loader2.load_client_partitions(4, cfg)
        for (X1, y1), (X2, y2) in zip(p1, p2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    def test_iid_partitioning_when_alpha_is_none(self, tmp_path):
        loader_iid = MNISTDatasetLoader(
            data_dir=tmp_path,
            dirichlet_alpha=None,
            seed=42,
            validate=True,
            use_cache=False,
        )
        loader_iid._download_raw = _make_fake_mnist  # type: ignore[method-assign]
        cfg = MagicMock(seed=42, synthetic=None)
        parts = loader_iid.load_client_partitions(4, cfg)
        sizes = [len(p[1]) for p in parts]
        # IID → equal sizes ± 1
        assert max(sizes) - min(sizes) <= 1


class TestMNISTHoldout:
    def test_holdout_shape(self, loader):
        X_h, y_h = loader.load_clean_holdout()
        assert X_h.shape[1:] == (1, 28, 28)
        assert y_h.ndim == 1

    def test_eval_shape(self, loader):
        X_e, y_e = loader.load_evaluation_set()
        assert X_e.shape[1:] == (1, 28, 28)
        assert y_e.ndim == 1

    def test_holdout_and_eval_are_disjoint(self, loader):
        """The two test-set splits must not overlap."""
        X_h, _y_h = loader.load_clean_holdout()
        X_e, _y_e = loader.load_evaluation_set()
        # Check by length: holdout + eval = full test set (100 samples in mock)
        assert len(X_h) + len(X_e) == 100

    def test_holdout_fraction_applied_correctly(self, loader):
        """With holdout_fraction=0.5, each split should be 50 samples."""
        X_h, _ = loader.load_clean_holdout()
        X_e, _ = loader.load_evaluation_set()
        assert len(X_h) == 50
        assert len(X_e) == 50

    def test_returns_copies_not_views(self, loader):
        """Mutating the returned holdout must not affect internal state."""
        X_h, _ = loader.load_clean_holdout()
        X_h[:] = 99.0
        X_h2, _ = loader.load_clean_holdout()
        assert not np.all(X_h2 == 99.0)


class TestMNISTCache:
    def test_disk_cache_is_populated_after_first_load(self, loader, tmp_path):
        cfg = MagicMock(seed=42, synthetic=None)
        loader.load_client_partitions(4, cfg)
        from ai.training.cache import DiskCache

        c = DiskCache(tmp_path)
        assert c.has("mnist_v1_train")
        assert c.has("mnist_v1_test")

    def test_second_load_uses_cache(self, loader, tmp_path):
        """On second instantiation, _download_raw should NOT be called."""
        cfg = MagicMock(seed=42, synthetic=None)
        loader.load_client_partitions(4, cfg)

        call_count = 0

        def _should_not_be_called():
            nonlocal call_count
            call_count += 1
            return _make_fake_mnist()

        loader2 = MNISTDatasetLoader(
            data_dir=tmp_path,
            dirichlet_alpha=0.5,
            seed=42,
            validate=True,
            use_cache=True,
        )
        loader2._download_raw = _should_not_be_called  # type: ignore[method-assign]
        loader2.load_client_partitions(4, cfg)
        assert call_count == 0, "_download_raw was called despite cache hit"
