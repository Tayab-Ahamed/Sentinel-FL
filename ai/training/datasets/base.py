"""
ai/training/datasets/base.py — BaseDatasetLoader: shared lifecycle for all concrete loaders.

``BaseDatasetLoader`` implements the ``DatasetLoader`` interface and provides
the complete download → cache → validate → partition lifecycle.  Concrete
subclasses (MNIST, CIFAR-10, etc.) only need to implement two methods:

    ``_download_raw()`` → ``(X_train, y_train, X_test, y_test)``
        Downloads (or loads from torchvision/disk) and returns raw numpy arrays.
        Must NOT call the cache — the base class handles caching.

    ``dataset_name`` property → ``str``
        A stable, version-tagged name used as the disk-cache key.

Lifecycle (called on first access via ``_ensure_data()``):
  1. Check ``DiskCache`` for ``<dataset_name>_train`` and ``<dataset_name>_test``.
  2. On cache miss: call ``_download_raw()`` and ``cache.put()``.
  3. Validate arrays via ``DatasetValidator``.
  4. Cache training and test splits in memory.

Partitioning (``load_client_partitions``):
  - Reads ``dirichlet_alpha`` from the config object (or falls back to the
    constructor default).
  - ``None`` alpha → ``IIDPartitioner``; any float → ``DirichletPartitioner``.

Holdout/evaluation split:
  - ``load_clean_holdout()`` → first ``holdout_fraction`` of test set.
  - ``load_evaluation_set()`` → remainder of test set.
  - Both splits are fully disjoint from all client partitions.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ai.fl_core.interfaces import DatasetLoader
from ai.training.cache import DiskCache
from ai.training.partitioning import DirichletPartitioner, IIDPartitioner
from ai.training.validation import DatasetValidator

logger = logging.getLogger(__name__)


class BaseDatasetLoader(DatasetLoader):
    """Abstract base for MNIST, CIFAR-10, and future torchvision loaders.

    Args:
        data_dir: Root directory for downloads and ``.cache/``.
        dirichlet_alpha: Dirichlet concentration for non-IID partitioning.
            ``None`` → IID equal splits.
        holdout_fraction: Fraction of the *test* set used as the clean holdout.
            The rest becomes the evaluation set.
        seed: Default random seed (overridden by ``config.seed`` at runtime).
        validate: If ``True``, run ``DatasetValidator`` after loading.
        use_cache: If ``True``, use ``DiskCache`` to avoid re-downloading.
    """

    def __init__(
        self,
        data_dir: str | Path = "datasets",
        dirichlet_alpha: float | None = 0.5,
        holdout_fraction: float = 0.5,
        seed: int = 42,
        validate: bool = True,
        use_cache: bool = True,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._dirichlet_alpha = dirichlet_alpha
        self._holdout_fraction = holdout_fraction
        self._seed = seed
        self._validate = validate
        self._use_cache = use_cache

        self._cache = DiskCache(data_dir)
        self._validator = DatasetValidator()

        # In-memory cache (populated on first _ensure_data call)
        self._train_X: np.ndarray | None = None
        self._train_y: np.ndarray | None = None
        self._test_X: np.ndarray | None = None
        self._test_y: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Stable, version-tagged name used as disk-cache key.

        Example: ``"mnist_v1"`` or ``"cifar10_v1"``.
        """

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Number of target classes (e.g. 10 for MNIST/CIFAR-10)."""

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """Per-sample input shape without batch dim (e.g. ``(1, 28, 28)``)."""

    @abstractmethod
    def _download_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Download/load the raw dataset and return numpy arrays.

        Returns:
            ``(X_train, y_train, X_test, y_test)`` — all dtype float32/int64.
            Shapes: X ``(N, *input_shape)``, y ``(N,)``.

        Note:
            Do NOT call the cache here.  The base class handles caching.
        """

    # ------------------------------------------------------------------
    # DatasetLoader interface
    # ------------------------------------------------------------------

    def load_client_partitions(
        self, n_clients: int, config: Any
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Partition training data across ``n_clients``.

        Reads ``dirichlet_alpha`` and ``seed`` from ``config`` if present,
        otherwise falls back to constructor defaults.

        Args:
            n_clients: Number of simulated clients.
            config: Configuration object (duck-typed: reads ``.seed`` and
                ``.synthetic.dirichlet_alpha`` if available).

        Returns:
            List of ``(X_i, y_i)`` tuples, one per client.
        """
        self._ensure_data()
        X, y = self._train_X, self._train_y
        assert X is not None and y is not None

        seed = int(getattr(config, "seed", self._seed))
        alpha = self._resolve_alpha(config)

        if alpha is None:
            partitioner = IIDPartitioner()
            label = "IID"
        else:
            partitioner = DirichletPartitioner(alpha=alpha, n_classes=self.n_classes)
            label = f"Dirichlet(alpha={alpha})"

        index_lists = partitioner.partition(X, y, n_clients, seed=seed)
        partitions = [(X[idx].copy(), y[idx].copy()) for idx in index_lists]

        logger.info(
            "%s.load_client_partitions: %d clients, %s, sizes min=%d max=%d",
            self.__class__.__name__,
            n_clients,
            label,
            min(len(idx) for idx in index_lists),
            max(len(idx) for idx in index_lists),
        )
        return partitions

    def load_clean_holdout(self) -> tuple[np.ndarray, np.ndarray]:
        """Return server-side clean validation set (first ``holdout_fraction`` of test).

        Used by L3 STRIP calibration and L2 Neural Cleanse.

        Returns:
            ``(X_holdout, y_holdout)`` — copies, not views.
        """
        self._ensure_data()
        assert self._test_X is not None and self._test_y is not None
        mid = int(len(self._test_X) * self._holdout_fraction)
        return self._test_X[:mid].copy(), self._test_y[:mid].copy()

    def load_evaluation_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Return final evaluation set (last ``1 - holdout_fraction`` of test).

        Never seen during training; used only for final C-Acc / ASR metrics.

        Returns:
            ``(X_eval, y_eval)`` — copies, not views.
        """
        self._ensure_data()
        assert self._test_X is not None and self._test_y is not None
        mid = int(len(self._test_X) * self._holdout_fraction)
        return self._test_X[mid:].copy(), self._test_y[mid:].copy()

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    @property
    def n_train(self) -> int:
        """Number of training samples (0 if not yet loaded)."""
        return len(self._train_X) if self._train_X is not None else 0

    @property
    def n_test(self) -> int:
        """Number of test samples (0 if not yet loaded)."""
        return len(self._test_X) if self._test_X is not None else 0

    def preload(self) -> None:
        """Eagerly download and cache the dataset (optional, for prefetching)."""
        self._ensure_data()

    # ------------------------------------------------------------------
    # Internal lifecycle
    # ------------------------------------------------------------------

    def _ensure_data(self) -> None:
        """Download, cache, and validate the dataset on first access."""
        if self._train_X is not None:
            return  # already in memory

        train_key = f"{self.dataset_name}_train"
        test_key = f"{self.dataset_name}_test"

        # Try disk cache first
        if self._use_cache:
            cached_train = self._cache.get(train_key)
            cached_test = self._cache.get(test_key)
            if cached_train is not None and cached_test is not None:
                self._train_X, self._train_y = cached_train
                self._test_X, self._test_y = cached_test
                logger.info(
                    "%s: loaded from cache (train=%d, test=%d)",
                    self.__class__.__name__,
                    len(self._train_X),
                    len(self._test_X),
                )
                return

        # Download / load from source
        logger.info("%s: downloading dataset from source …", self.__class__.__name__)
        X_train, y_train, X_test, y_test = self._download_raw()

        # Validate
        if self._validate:
            self._run_validation(X_train, y_train, "train")
            self._run_validation(X_test, y_test, "test")

        # Write to disk cache
        if self._use_cache:
            self._cache.put(train_key, X_train, y_train)
            self._cache.put(test_key, X_test, y_test)

        self._train_X, self._train_y = X_train, y_train
        self._test_X, self._test_y = X_test, y_test
        logger.info(
            "%s: loaded (train=%d, test=%d)",
            self.__class__.__name__,
            len(X_train),
            len(X_test),
        )

    def _run_validation(self, X: np.ndarray, y: np.ndarray, split: str) -> None:
        """Validate arrays and log — raises on errors if validation fails."""
        result = self._validator.validate(
            X,
            y,
            expected_shape=self.input_shape,
            n_classes=self.n_classes,
            dataset_name=f"{self.dataset_name}_{split}",
        )
        if not result.is_valid:
            raise ValueError(
                f"{self.__class__.__name__} validation failed for {split} split:\n"
                + "\n".join(result.errors)
            )

    def _resolve_alpha(self, config: Any) -> float | None:
        """Extract dirichlet_alpha from config, falling back to constructor default."""
        # Try config.synthetic.dirichlet_alpha (from YAML)
        synthetic = getattr(config, "synthetic", None)
        if synthetic is not None:
            alpha = getattr(synthetic, "dirichlet_alpha", None)
            if alpha is not None:
                return float(alpha)
        # Fall back to constructor default
        return self._dirichlet_alpha
