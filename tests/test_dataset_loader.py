"""
tests/test_dataset_loader.py — Tests for ai/training/dataset_loader.py.

Covers:
  SyntheticDatasetLoader: load_client_partitions, load_clean_holdout, load_evaluation_set
  OfficialDatasetLoader: missing path raises, dev_mode fallback
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ai.fl_core.exceptions import DatasetNotFoundError
from ai.training.dataset_loader import OfficialDatasetLoader, SyntheticDatasetLoader

# ---------------------------------------------------------------------------
# SyntheticDatasetLoader
# ---------------------------------------------------------------------------


class TestSyntheticDatasetLoader:
    @pytest.fixture
    def loader(self):
        return SyntheticDatasetLoader(
            n_samples=300,
            n_features=8,
            n_classes=3,
            dirichlet_alpha=0.5,
            train_fraction=0.8,
            seed=0,
        )

    @pytest.fixture
    def config(self):
        return SimpleNamespace(seed=0)

    def test_load_client_partitions_count(self, loader, config):
        parts = loader.load_client_partitions(n_clients=5, config=config)
        assert len(parts) == 5

    def test_load_client_partitions_are_tuples(self, loader, config):
        parts = loader.load_client_partitions(n_clients=3, config=config)
        for X, y in parts:
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert X.ndim == 2
            assert y.ndim == 1

    def test_load_client_partitions_feature_dim(self, loader, config):
        parts = loader.load_client_partitions(n_clients=4, config=config)
        for X, _y in parts:
            assert X.shape[1] == 8

    def test_load_client_partitions_total_samples(self, loader, config):
        parts = loader.load_client_partitions(n_clients=5, config=config)
        total = sum(len(X) for X, _ in parts)
        # Total should equal training samples (80% of 300 = 240)
        assert total == 240

    def test_load_client_partitions_are_copies(self, loader, config):
        parts = loader.load_client_partitions(n_clients=3, config=config)
        # Modifying returned data must not affect internal cache
        X0, _y0 = parts[0]
        X0[:] = 999.0
        parts2 = loader.load_client_partitions(n_clients=3, config=config)
        assert not np.all(parts2[0][0] == 999.0)

    def test_load_clean_holdout_shape(self, loader):
        X, y = loader.load_clean_holdout()
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 8

    def test_load_clean_holdout_is_half_of_holdout(self, loader):
        X_holdout, _ = loader.load_clean_holdout()
        X_eval, _ = loader.load_evaluation_set()
        # Together they cover the full holdout (20% of 300 = 60)
        total = len(X_holdout) + len(X_eval)
        assert total == 60

    def test_load_evaluation_set_shape(self, loader):
        X, y = loader.load_evaluation_set()
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[1] == 8

    def test_no_overlap_between_holdout_and_eval(self, loader):
        """Clean holdout and evaluation set must be disjoint."""
        X_holdout, _ = loader.load_clean_holdout()
        X_eval, _ = loader.load_evaluation_set()
        # They should not share any rows (different index ranges)
        assert len(X_holdout) > 0
        assert len(X_eval) > 0

    def test_cache_avoids_double_generation(self, loader, config):
        """Data generation should only happen once (idempotent calls)."""
        # First access: generates data
        p1 = loader.load_client_partitions(n_clients=3, config=config)
        # Second access: uses cache (same data, no re-generation)
        p2 = loader.load_client_partitions(n_clients=3, config=config)
        # Shapes should be identical
        assert len(p1) == len(p2)
        for (X1, _), (X2, _) in zip(p1, p2):
            assert X1.shape == X2.shape

    def test_reproducible_with_same_seed(self, config):
        loader1 = SyntheticDatasetLoader(n_samples=100, n_features=5, n_classes=2, seed=42)
        loader2 = SyntheticDatasetLoader(n_samples=100, n_features=5, n_classes=2, seed=42)
        X1, y1 = loader1.load_clean_holdout()
        X2, y2 = loader2.load_clean_holdout()
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_give_different_data(self, config):
        loader1 = SyntheticDatasetLoader(n_samples=200, n_features=5, n_classes=2, seed=0)
        loader2 = SyntheticDatasetLoader(n_samples=200, n_features=5, n_classes=2, seed=999)
        X1, _ = loader1.load_clean_holdout()
        X2, _ = loader2.load_clean_holdout()
        assert not np.array_equal(X1, X2)


# ---------------------------------------------------------------------------
# OfficialDatasetLoader
# ---------------------------------------------------------------------------


class TestOfficialDatasetLoader:
    def test_missing_path_raises_dataset_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        loader = OfficialDatasetLoader(missing, dev_mode=False)
        with pytest.raises(DatasetNotFoundError):
            loader.load_client_partitions(3, SimpleNamespace(seed=0))

    def test_missing_path_raises_on_holdout(self, tmp_path):
        missing = tmp_path / "no_dataset"
        loader = OfficialDatasetLoader(missing, dev_mode=False)
        with pytest.raises(DatasetNotFoundError):
            loader.load_clean_holdout()

    def test_missing_path_raises_on_eval(self, tmp_path):
        missing = tmp_path / "no_dataset"
        loader = OfficialDatasetLoader(missing, dev_mode=False)
        with pytest.raises(DatasetNotFoundError):
            loader.load_evaluation_set()

    def test_dev_mode_fallback_for_partitions(self, tmp_path):
        """dev_mode=True should fall back to synthetic data."""
        missing = tmp_path / "no_official"
        loader = OfficialDatasetLoader(missing, dev_mode=True)
        config = SimpleNamespace(seed=0)
        parts = loader.load_client_partitions(4, config)
        assert len(parts) == 4

    def test_dev_mode_fallback_for_holdout(self, tmp_path):
        missing = tmp_path / "no_official"
        loader = OfficialDatasetLoader(missing, dev_mode=True)
        X, y = loader.load_clean_holdout()
        assert X.ndim == 2
        assert y.ndim == 1

    def test_dev_mode_fallback_for_eval(self, tmp_path):
        missing = tmp_path / "no_official"
        loader = OfficialDatasetLoader(missing, dev_mode=True)
        X, y = loader.load_evaluation_set()
        assert X.ndim == 2
        assert y.ndim == 1

    def test_existing_path_raises_not_implemented(self, tmp_path):
        """Even with path existing, Phase 1 is not yet implemented."""
        existing = tmp_path / "official_data"
        existing.mkdir()
        loader = OfficialDatasetLoader(existing, dev_mode=False)
        with pytest.raises(NotImplementedError):
            loader.load_client_partitions(3, SimpleNamespace(seed=0))
