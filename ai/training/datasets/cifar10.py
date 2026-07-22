"""
ai/training/datasets/cifar10.py — CIFAR-10 DatasetLoader.

Downloads CIFAR-10 via ``torchvision.datasets.CIFAR10``, normalises with
per-channel statistics (standard values from the literature), and returns
``float32`` arrays of shape ``(N, 3, 32, 32)``.

CIFAR-10 statistics (computed on training set):
  mean = [0.4914, 0.4822, 0.4465]  (R, G, B)
  std  = [0.2023, 0.1994, 0.2010]

Cache key: ``"cifar10_v1_train"`` / ``"cifar10_v1_test"``
  Bump the version suffix whenever normalisation or preprocessing changes.

Classes (in label order 0–9):
  airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
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
# CIFAR-10 normalisation statistics
# ---------------------------------------------------------------------------
_CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD: tuple[float, float, float]  = (0.2023, 0.1994, 0.2010)

_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
)

# Human-readable class names for logging and reports
CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


class CIFAR10DatasetLoader(BaseDatasetLoader):
    """DatasetLoader for CIFAR-10 (32×32 RGB, 10 classes).

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
        return "cifar10_v1"

    @property
    def n_classes(self) -> int:
        return 10

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, 32, 32)

    # ------------------------------------------------------------------
    # Download implementation
    # ------------------------------------------------------------------

    def _download_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Download CIFAR-10 via torchvision and convert to float32/int64 arrays.

        Returns:
            ``(X_train, y_train, X_test, y_test)``
            X shape: ``(N, 3, 32, 32)`` float32 in normalised range.
            y shape: ``(N,)`` int64 in ``[0, 9]``.
        """
        logger.info(
            "CIFAR10DatasetLoader: downloading CIFAR-10 to %s …", self._data_dir
        )
        train_ds = datasets.CIFAR10(
            str(self._data_dir), train=True, download=True, transform=_TRANSFORM
        )
        test_ds = datasets.CIFAR10(
            str(self._data_dir), train=False, download=True, transform=_TRANSFORM
        )
        X_train, y_train = _dataset_to_numpy(train_ds)
        X_test, y_test = _dataset_to_numpy(test_ds)
        logger.info(
            "CIFAR10DatasetLoader: train=%d test=%d",
            len(X_train), len(X_test),
        )
        return X_train, y_train, X_test, y_test

    def class_name(self, label: int) -> str:
        """Return the human-readable class name for a label index.

        Args:
            label: Integer class label in ``[0, 9]``.

        Returns:
            Class name string (e.g. ``"airplane"``).
        """
        if 0 <= label < len(CIFAR10_CLASSES):
            return CIFAR10_CLASSES[label]
        return f"unknown_label_{label}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset_to_numpy(
    dataset: datasets.CIFAR10,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a torchvision CIFAR-10 Dataset to ``(X_float32, y_int64)``."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0
    )
    X_t, y_t = next(iter(loader))
    return X_t.numpy().astype(np.float32), y_t.numpy().astype(np.int64)
