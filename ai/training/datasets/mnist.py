"""
ai/training/datasets/mnist.py — MNIST DatasetLoader.

Downloads MNIST via ``torchvision.datasets.MNIST``, normalises to
``mean=0.1307, std=0.3081`` (standard MNIST statistics), and returns
``float32`` arrays of shape ``(N, 1, 28, 28)``.

Cache key: ``"mnist_v1_train"`` / ``"mnist_v1_test"``
  Bump the version suffix whenever normalisation or preprocessing changes.

This module replaces the original ``ai/training/mnist_loader.py``
(which is kept as a backward-compatible shim).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms

from ai.training.datasets.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MNIST normalisation statistics (from the full training set)
# ---------------------------------------------------------------------------
_MNIST_MEAN: tuple[float, ...] = (0.1307,)
_MNIST_STD: tuple[float, ...]  = (0.3081,)

_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
    ]
)


class MNISTDatasetLoader(BaseDatasetLoader):
    """DatasetLoader for MNIST (28×28 greyscale, 10 classes).

    Args:
        data_dir: Root directory for downloads and disk cache.
        dirichlet_alpha: Dirichlet α for non-IID partitioning (``None`` = IID).
        holdout_fraction: Fraction of test set used as clean holdout.
        seed: Default random seed.
        validate: Run ``DatasetValidator`` after loading.
        use_cache: Use disk cache to avoid re-downloading.
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
        super().__init__(
            data_dir=data_dir,
            dirichlet_alpha=dirichlet_alpha,
            holdout_fraction=holdout_fraction,
            seed=seed,
            validate=validate,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    # BaseDatasetLoader abstract properties
    # ------------------------------------------------------------------

    @property
    def dataset_name(self) -> str:
        return "mnist_v1"

    @property
    def n_classes(self) -> int:
        return 10

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 28, 28)

    # ------------------------------------------------------------------
    # Download implementation
    # ------------------------------------------------------------------

    def _download_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Download MNIST via torchvision and convert to float32/int64 arrays.

        Returns:
            ``(X_train, y_train, X_test, y_test)``
            X shape: ``(N, 1, 28, 28)`` float32 in normalised range.
            y shape: ``(N,)`` int64 in ``[0, 9]``.
        """
        logger.info("MNISTDatasetLoader: downloading MNIST to %s …", self._data_dir)
        train_ds = datasets.MNIST(
            str(self._data_dir), train=True, download=True, transform=_TRANSFORM
        )
        test_ds = datasets.MNIST(
            str(self._data_dir), train=False, download=True, transform=_TRANSFORM
        )
        X_train, y_train = _dataset_to_numpy(train_ds)
        X_test, y_test = _dataset_to_numpy(test_ds)
        logger.info(
            "MNISTDatasetLoader: train=%d test=%d",
            len(X_train), len(X_test),
        )
        return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset_to_numpy(
    dataset: datasets.MNIST,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a torchvision Dataset to ``(X_float32, y_int64)`` numpy arrays.

    Loads the entire dataset in a single batch to avoid Python-loop overhead.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0
    )
    X_t, y_t = next(iter(loader))
    return X_t.numpy().astype(np.float32), y_t.numpy().astype(np.int64)
