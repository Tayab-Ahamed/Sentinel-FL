"""
tests/test_cifar10_dataset.py — Unit tests for CIFAR10DatasetLoader.

All tests use a mocked ``_download_raw()`` to avoid real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from ai.training.datasets.cifar10 import CIFAR10_CLASSES, CIFAR10DatasetLoader

# ---------------------------------------------------------------------------
# Mock raw download
# ---------------------------------------------------------------------------


def _make_fake_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fake CIFAR-10-shaped arrays (no real download)."""
    rng = np.random.default_rng(1)
    n_train, n_test = 500, 100
    X_train = rng.standard_normal((n_train, 3, 32, 32)).astype(np.float32)
    y_train = rng.integers(0, 10, size=n_train).astype(np.int64)
    X_test = rng.standard_normal((n_test, 3, 32, 32)).astype(np.float32)
    y_test = rng.integers(0, 10, size=n_test).astype(np.int64)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def loader(tmp_path) -> CIFAR10DatasetLoader:
    inst = CIFAR10DatasetLoader(
        data_dir=tmp_path,
        dirichlet_alpha=0.5,
        seed=42,
        validate=True,
        use_cache=True,
    )
    inst._download_raw = _make_fake_cifar10  # type: ignore[method-assign]
    return inst


class TestCIFAR10Properties:
    def test_dataset_name(self, loader):
        assert loader.dataset_name == "cifar10_v1"

    def test_n_classes(self, loader):
        assert loader.n_classes == 10

    def test_input_shape(self, loader):
        assert loader.input_shape == (3, 32, 32)

    def test_class_name_returns_string(self, loader):
        assert loader.class_name(0) == "airplane"
        assert loader.class_name(9) == "truck"

    def test_class_name_unknown_label(self, loader):
        assert "unknown" in loader.class_name(99).lower()

    def test_class_constants_length(self):
        assert len(CIFAR10_CLASSES) == 10


class TestCIFAR10LoadClientPartitions:
    def test_returns_correct_n_clients(self, loader):
        cfg = MagicMock(seed=42, synthetic=None)
        parts = loader.load_client_partitions(5, cfg)
        assert len(parts) == 5

    def test_partition_shapes_are_correct(self, loader):
        cfg = MagicMock(seed=42, synthetic=None)
        parts = loader.load_client_partitions(4, cfg)
        for X_i, y_i in parts:
            assert X_i.ndim == 4
            assert X_i.shape[1:] == (3, 32, 32)
            assert y_i.ndim == 1
            assert len(X_i) == len(y_i)

    def test_partitions_cover_all_training_samples(self, loader):
        cfg = MagicMock(seed=42, synthetic=None)
        parts = loader.load_client_partitions(4, cfg)
        total = sum(len(p[1]) for p in parts)
        assert total == 500

    def test_reproducible_with_same_seed(self, loader, tmp_path):
        cfg = MagicMock(seed=99, synthetic=None)
        p1 = loader.load_client_partitions(4, cfg)
        loader2 = CIFAR10DatasetLoader(
            data_dir=tmp_path, dirichlet_alpha=0.5, seed=42,
            validate=True, use_cache=True,
        )
        loader2._download_raw = _make_fake_cifar10  # type: ignore[method-assign]
        p2 = loader2.load_client_partitions(4, cfg)
        for (X1, _), (X2, _) in zip(p1, p2):
            np.testing.assert_array_equal(X1, X2)

    def test_iid_partitioning_when_alpha_is_none(self, tmp_path):
        loader_iid = CIFAR10DatasetLoader(
            data_dir=tmp_path, dirichlet_alpha=None, seed=42,
            validate=True, use_cache=False,
        )
        loader_iid._download_raw = _make_fake_cifar10  # type: ignore[method-assign]
        cfg = MagicMock(seed=42, synthetic=None)
        parts = loader_iid.load_client_partitions(5, cfg)
        sizes = [len(p[1]) for p in parts]
        assert max(sizes) - min(sizes) <= 1  # IID → near-equal sizes


class TestCIFAR10Holdout:
    def test_holdout_shape(self, loader):
        X_h, y_h = loader.load_clean_holdout()
        assert X_h.shape[1:] == (3, 32, 32)
        assert y_h.ndim == 1

    def test_eval_shape(self, loader):
        X_e, y_e = loader.load_evaluation_set()
        assert X_e.shape[1:] == (3, 32, 32)
        assert y_e.ndim == 1

    def test_holdout_and_eval_sum_to_test_set(self, loader):
        X_h, _ = loader.load_clean_holdout()
        X_e, _ = loader.load_evaluation_set()
        assert len(X_h) + len(X_e) == 100

    def test_holdout_fraction_50pct(self, loader):
        X_h, _ = loader.load_clean_holdout()
        X_e, _ = loader.load_evaluation_set()
        assert len(X_h) == 50
        assert len(X_e) == 50

    def test_returns_copies(self, loader):
        X_h, _ = loader.load_clean_holdout()
        original_val = X_h[0, 0, 0, 0]
        X_h[0, 0, 0, 0] = 9999.0
        X_h2, _ = loader.load_clean_holdout()
        assert X_h2[0, 0, 0, 0] == pytest.approx(original_val)


class TestCIFAR10Cache:
    def test_disk_cache_populated_after_load(self, loader, tmp_path):
        cfg = MagicMock(seed=42, synthetic=None)
        loader.load_client_partitions(4, cfg)
        from ai.training.cache import DiskCache
        c = DiskCache(tmp_path)
        assert c.has("cifar10_v1_train")
        assert c.has("cifar10_v1_test")

    def test_second_load_uses_cache(self, loader, tmp_path):
        cfg = MagicMock(seed=42, synthetic=None)
        loader.load_client_partitions(4, cfg)

        call_count = 0

        def _noop():
            nonlocal call_count
            call_count += 1
            return _make_fake_cifar10()

        loader2 = CIFAR10DatasetLoader(
            data_dir=tmp_path, dirichlet_alpha=0.5, seed=42,
            validate=True, use_cache=True,
        )
        loader2._download_raw = _noop  # type: ignore[method-assign]
        loader2.load_client_partitions(4, cfg)
        assert call_count == 0
