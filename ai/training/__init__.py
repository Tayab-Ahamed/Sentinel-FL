"""
ai/training/__init__.py — Training package public re-exports.
"""

from ai.training.cache import DiskCache
from ai.training.dataset_loader import OfficialDatasetLoader, SyntheticDatasetLoader
from ai.training.datasets import (
    BaseDatasetLoader,
    CIFAR10DatasetLoader,
    DatasetRegistry,
    MNISTDatasetLoader,
)
from ai.training.partitioning import DirichletPartitioner, IIDPartitioner
from ai.training.poison import BadNetsAttackSimulator
from ai.training.validation import DatasetValidator, ValidationResult

__all__ = [
    # Loaders
    "BadNetsAttackSimulator",
    "BaseDatasetLoader",
    "CIFAR10DatasetLoader",
    "DatasetRegistry",
    "MNISTDatasetLoader",
    "OfficialDatasetLoader",
    "SyntheticDatasetLoader",
    # Partitioning
    "DirichletPartitioner",
    "IIDPartitioner",
    # Cache
    "DiskCache",
    # Validation
    "DatasetValidator",
    "ValidationResult",
]
