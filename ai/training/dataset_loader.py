"""
ai/training/dataset_loader.py — DatasetLoader implementations.

Implements the DatasetLoader interface (INTERFACES.md §DatasetLoader) for:
  - Phase 0: Synthetic Gaussian-blob data with Dirichlet non-IID partitioning
    (backed by ai.training.poison.make_dataset + dirichlet_partition).
  - Phase 1: Official GSC26 Challenge 1 dataset (stub — available from 13 July).

Both loaders expose the same interface so ai/fl_core and ai/detection never
special-case which phase is active (ARCHITECTURE.md §7.11).

Failure contract:
  - Phase 1 dataset missing → raise DatasetNotFoundError unless --dev-mode.
  - Never silently fall back to Phase 0 data in production mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ai.fl_core.exceptions import DatasetNotFoundError
from ai.fl_core.interfaces import DatasetLoader
from ai.training.poison import dirichlet_partition, make_dataset

logger = logging.getLogger(__name__)


class SyntheticDatasetLoader(DatasetLoader):
    """Phase 0: Gaussian-blob synthetic data with Dirichlet non-IID partitioning.

    Args:
        n_samples: Total number of synthetic samples.
        n_features: Feature dimensionality.
        n_classes: Number of target classes.
        dirichlet_alpha: Dirichlet concentration parameter (smaller = more skewed).
        train_fraction: Fraction of data used for training vs. held-out.
        seed: Base random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 3000,
        n_features: int = 20,
        n_classes: int = 4,
        dirichlet_alpha: float = 0.5,
        train_fraction: float = 0.85,
        seed: int = 42,
    ) -> None:
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_classes = n_classes
        self._dirichlet_alpha = dirichlet_alpha
        self._train_fraction = train_fraction
        self._seed = seed
        # Cache so we generate the data once
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._X_holdout: np.ndarray | None = None
        self._y_holdout: np.ndarray | None = None

    def _ensure_data(self) -> None:
        """Generate and cache the synthetic dataset on first access."""
        if self._X is not None:
            return
        self._X, self._y = make_dataset(
            self._n_samples, self._n_features, self._n_classes, seed=self._seed
        )
        split = int(self._n_samples * self._train_fraction)
        self._X_train = self._X[:split]
        self._y_train = self._y[:split]
        self._X_holdout = self._X[split:]
        self._y_holdout = self._y[split:]
        logger.info(
            "SyntheticDatasetLoader: generated %d samples (%d train, %d holdout), "
            "n_features=%d, n_classes=%d",
            self._n_samples,
            split,
            self._n_samples - split,
            self._n_features,
            self._n_classes,
        )

    def load_client_partitions(
        self, n_clients: int, config: Any
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Partition training data across n_clients using Dirichlet distribution.

        Args:
            n_clients: Number of simulated FL clients.
            config: Configuration object (used for seed).

        Returns:
            List of ``(X_client, y_client)`` tuples.
        """
        self._ensure_data()
        assert self._X_train is not None and self._y_train is not None
        seed = getattr(config, "seed", self._seed) + 7  # offset to differ from data seed
        indices = dirichlet_partition(
            len(self._X_train),
            n_clients,
            self._y_train,
            self._n_classes,
            alpha=self._dirichlet_alpha,
            seed=seed,
        )
        partitions = [(self._X_train[idx].copy(), self._y_train[idx].copy()) for idx in indices]
        logger.info(
            "SyntheticDatasetLoader: partitioned into %d clients (Dirichlet α=%.2f)",
            n_clients,
            self._dirichlet_alpha,
        )
        return partitions

    def load_clean_holdout(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the server-side clean validation set."""
        self._ensure_data()
        assert self._X_holdout is not None and self._y_holdout is not None
        # Use the first half of holdout as the clean calibration pool.
        mid = len(self._X_holdout) // 2
        return self._X_holdout[:mid].copy(), self._y_holdout[:mid].copy()

    def load_evaluation_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the held-out evaluation set (never seen during training)."""
        self._ensure_data()
        assert self._X_holdout is not None and self._y_holdout is not None
        # Use the second half as the final evaluation set.
        mid = len(self._X_holdout) // 2
        return self._X_holdout[mid:].copy(), self._y_holdout[mid:].copy()


class OfficialDatasetLoader(DatasetLoader):
    """Phase 1: Official GSC26 Challenge 1 dataset loader.

    Reads from the dataset root pointed to by ``dataset_path``.
    Raises ``DatasetNotFoundError`` if the path is missing or malformed,
    unless ``dev_mode=True`` is passed (then falls back to SyntheticDatasetLoader).

    Status: Skeleton — implementation blocked on dataset release (13 July).

    Args:
        dataset_path: Path to the official dataset root directory.
        dev_mode: If True, fall back to synthetic data when the official
            dataset is missing (for local development only).
    """

    def __init__(self, dataset_path: str | Path, dev_mode: bool = False) -> None:
        self._path = Path(dataset_path)
        self._dev_mode = dev_mode
        self._fallback = SyntheticDatasetLoader() if dev_mode else None

    def _check_path(self) -> None:
        """Raise DatasetNotFoundError if the dataset root is missing."""
        if not self._path.exists():
            if self._dev_mode and self._fallback is not None:
                logger.warning(
                    "OfficialDatasetLoader: dataset not found at %s. "
                    "Falling back to synthetic data (dev_mode=True).",
                    self._path,
                )
                return
            raise DatasetNotFoundError(str(self._path), phase="phase1_official")

    def load_client_partitions(self, n_clients: int, config: Any) -> list[tuple[Any, Any]]:
        """Load Phase 1 client partitions.

        Status: Not yet implemented — will be completed once the dataset is released.
        """
        self._check_path()
        if self._dev_mode and self._fallback is not None and not self._path.exists():
            return self._fallback.load_client_partitions(n_clients, config)
        raise NotImplementedError(
            "OfficialDatasetLoader will be implemented in Phase 1 "
            "once the official GSC26 dataset is released (13 July)."
        )

    def load_clean_holdout(self) -> tuple[Any, Any]:
        """Load Phase 1 clean holdout set.

        Status: Not yet implemented.
        """
        self._check_path()
        if self._dev_mode and self._fallback is not None and not self._path.exists():
            return self._fallback.load_clean_holdout()
        raise NotImplementedError(
            "OfficialDatasetLoader.load_clean_holdout() will be implemented in Phase 1."
        )

    def load_evaluation_set(self) -> tuple[Any, Any]:
        """Load Phase 1 evaluation set.

        Status: Not yet implemented.
        """
        self._check_path()
        if self._dev_mode and self._fallback is not None and not self._path.exists():
            return self._fallback.load_evaluation_set()
        raise NotImplementedError(
            "OfficialDatasetLoader.load_evaluation_set() will be implemented in Phase 1."
        )
