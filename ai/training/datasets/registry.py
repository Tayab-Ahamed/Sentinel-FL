"""
ai/training/datasets/registry.py — DatasetRegistry factory.

Maps dataset names (as used in configs) to concrete ``DatasetLoader``
subclasses.  New datasets self-register by calling
``DatasetRegistry.register()``.

Pre-registered datasets:
  - ``"mnist"``           → ``MNISTDatasetLoader``
  - ``"cifar10"``         → ``CIFAR10DatasetLoader``
  - ``"synthetic"``       → ``SyntheticDatasetLoader`` (Phase 0)
  - ``"phase1_official"`` → ``OfficialDatasetLoader`` (Phase 1 stub)

Usage::

    from ai.training.datasets.registry import DatasetRegistry
    loader = DatasetRegistry.get_loader("mnist", config)
    partitions = loader.load_client_partitions(10, config)

Config keys read by ``get_loader``:
  - ``config.dataset_name``        — dataset to load (``"mnist"``, etc.)
  - ``config.data_dir``            — download / cache root (default: ``"datasets"``)
  - ``config.seed``                — random seed
  - ``config.synthetic.dirichlet_alpha`` — alpha for non-IID
  - ``config.phase1_dataset_path`` — path for OfficialDatasetLoader
"""

from __future__ import annotations

import logging
from typing import Any

from ai.fl_core.interfaces import DatasetLoader

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Singleton factory mapping dataset name → DatasetLoader subclass.

    All built-in loaders are registered at module import time.  Third-party
    or experiment-specific loaders can call ``DatasetRegistry.register()``
    at any time before ``get_loader()`` is called.
    """

    _registry: dict[str, type[DatasetLoader]] = {}

    @classmethod
    def register(cls, name: str, loader_class: type[DatasetLoader]) -> None:
        """Register a loader class under a string name.

        Args:
            name: Short, lowercase name (e.g. ``"mnist"``).  Must be unique.
            loader_class: A concrete subclass of ``DatasetLoader``.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in cls._registry:
            raise ValueError(
                f"DatasetRegistry: '{name}' is already registered "
                f"(class={cls._registry[name].__name__}). "
                "Use DatasetRegistry.unregister() first if replacement is intended."
            )
        cls._registry[name] = loader_class
        logger.debug("DatasetRegistry: registered '%s' → %s", name, loader_class.__name__)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a registration (primarily for tests).

        Args:
            name: Name to remove.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"DatasetRegistry: '{name}' is not registered.")
        del cls._registry[name]

    @classmethod
    def get_loader(
        cls,
        name: str,
        config: Any,
        *,
        data_dir: str | None = None,
    ) -> DatasetLoader:
        """Construct and return a configured DatasetLoader.

        Args:
            name: Registered dataset name (e.g. ``"mnist"``).  If ``None``,
                falls back to ``config.dataset_name``.
            config: Configuration object.  Read for ``seed``, ``data_dir``,
                ``synthetic.dirichlet_alpha``, and ``phase1_dataset_path``.
            data_dir: Override the download/cache directory.  Falls back to
                ``config.data_dir`` then ``"datasets"``.

        Returns:
            A fully configured ``DatasetLoader`` instance.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"DatasetRegistry: unknown dataset '{name}'. "
                f"Available: {available}"
            )

        loader_class = cls._registry[name]
        resolved_dir = data_dir or getattr(config, "data_dir", "datasets")
        seed = int(getattr(config, "seed", 42))

        # Read dirichlet_alpha from config.synthetic (YAML structure)
        synthetic = getattr(config, "synthetic", None)
        alpha: float | None = (
            float(getattr(synthetic, "dirichlet_alpha", 0.5))
            if synthetic is not None
            else 0.5
        )

        # Build kwargs common to BaseDatasetLoader subclasses
        common_kwargs: dict[str, Any] = {
            "data_dir": resolved_dir,
            "seed": seed,
        }

        # ------------------------------------------------------------------
        # Dataset-specific construction
        # ------------------------------------------------------------------
        if name in ("mnist", "cifar10"):
            loader = loader_class(
                **common_kwargs,
                dirichlet_alpha=alpha,
            )

        elif name == "synthetic":
            # SyntheticDatasetLoader has its own signature
            n_samples = int(getattr(synthetic, "n_samples", 3000)) if synthetic else 3000
            n_features = int(getattr(synthetic, "n_features", 20)) if synthetic else 20
            n_classes = int(getattr(synthetic, "n_classes", 4)) if synthetic else 4
            loader = loader_class(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                dirichlet_alpha=alpha,
                seed=seed,
            )

        elif name == "phase1_official":
            dataset_path = getattr(config, "phase1_dataset_path", "datasets/phase1")
            dev_mode = bool(getattr(config, "dev_mode", False))
            loader = loader_class(dataset_path=dataset_path, dev_mode=dev_mode)

        else:
            # Generic fallback — pass common kwargs and hope for the best
            try:
                loader = loader_class(**common_kwargs)
            except TypeError:
                loader = loader_class()

        logger.info(
            "DatasetRegistry.get_loader('%s'): constructed %s",
            name, loader_class.__name__,
        )
        return loader

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return sorted list of all registered dataset names."""
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return ``True`` if ``name`` is registered."""
        return name in cls._registry


# ---------------------------------------------------------------------------
# Register all built-in loaders at import time
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Register MNIST, CIFAR-10, Synthetic, and Phase1 loaders."""
    from ai.training.dataset_loader import OfficialDatasetLoader, SyntheticDatasetLoader
    from ai.training.datasets.cifar10 import CIFAR10DatasetLoader
    from ai.training.datasets.mnist import MNISTDatasetLoader

    DatasetRegistry.register("mnist", MNISTDatasetLoader)
    DatasetRegistry.register("cifar10", CIFAR10DatasetLoader)
    DatasetRegistry.register("synthetic", SyntheticDatasetLoader)
    DatasetRegistry.register("phase1_official", OfficialDatasetLoader)


_register_builtins()
