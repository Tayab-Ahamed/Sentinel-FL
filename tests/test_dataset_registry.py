"""
tests/test_dataset_registry.py — Unit tests for DatasetRegistry.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai.fl_core.interfaces import DatasetLoader
from ai.training.dataset_loader import OfficialDatasetLoader, SyntheticDatasetLoader
from ai.training.datasets.cifar10 import CIFAR10DatasetLoader
from ai.training.datasets.mnist import MNISTDatasetLoader
from ai.training.datasets.registry import DatasetRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(**kwargs) -> MagicMock:
    """Return a mock config with sensible defaults."""
    defaults = {
        "seed": 42,
        "synthetic": None,
        "data_dir": "datasets",
        "phase1_dataset_path": "datasets/phase1",
        "dev_mode": False,
    }
    defaults.update(kwargs)
    return MagicMock(**defaults)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------


class TestBuiltinRegistrations:
    def test_mnist_is_registered(self):
        assert DatasetRegistry.is_registered("mnist")

    def test_cifar10_is_registered(self):
        assert DatasetRegistry.is_registered("cifar10")

    def test_synthetic_is_registered(self):
        assert DatasetRegistry.is_registered("synthetic")

    def test_phase1_official_is_registered(self):
        assert DatasetRegistry.is_registered("phase1_official")

    def test_list_registered_returns_all_builtins(self):
        registered = DatasetRegistry.list_registered()
        assert "mnist" in registered
        assert "cifar10" in registered
        assert "synthetic" in registered
        assert "phase1_official" in registered


# ---------------------------------------------------------------------------
# get_loader
# ---------------------------------------------------------------------------


class TestGetLoader:
    def test_get_mnist_returns_mnist_loader(self, tmp_path):
        cfg = _make_cfg(data_dir=str(tmp_path))
        loader = DatasetRegistry.get_loader("mnist", cfg)
        assert isinstance(loader, MNISTDatasetLoader)

    def test_get_cifar10_returns_cifar10_loader(self, tmp_path):
        cfg = _make_cfg(data_dir=str(tmp_path))
        loader = DatasetRegistry.get_loader("cifar10", cfg)
        assert isinstance(loader, CIFAR10DatasetLoader)

    def test_get_synthetic_returns_synthetic_loader(self):
        cfg = _make_cfg()
        loader = DatasetRegistry.get_loader("synthetic", cfg)
        assert isinstance(loader, SyntheticDatasetLoader)

    def test_get_phase1_returns_official_loader(self):
        cfg = _make_cfg()
        loader = DatasetRegistry.get_loader("phase1_official", cfg)
        assert isinstance(loader, OfficialDatasetLoader)

    def test_get_loader_returns_dataset_loader_interface(self, tmp_path):
        cfg = _make_cfg(data_dir=str(tmp_path))
        loader = DatasetRegistry.get_loader("mnist", cfg)
        assert isinstance(loader, DatasetLoader)

    def test_unknown_name_raises_key_error(self):
        cfg = _make_cfg()
        with pytest.raises(KeyError, match="unknown dataset"):
            DatasetRegistry.get_loader("nonexistent_dataset", cfg)

    def test_data_dir_override(self, tmp_path):
        cfg = _make_cfg(data_dir="wrong_dir")
        loader = DatasetRegistry.get_loader("mnist", cfg, data_dir=str(tmp_path))
        assert str(loader._data_dir) == str(tmp_path)

    def test_seed_propagated_from_config(self, tmp_path):
        cfg = _make_cfg(seed=99, data_dir=str(tmp_path))
        loader = DatasetRegistry.get_loader("mnist", cfg)
        assert loader._seed == 99

    def test_synthetic_reads_config_params(self):
        """SyntheticDatasetLoader should pick up n_samples etc from config."""
        synthetic_cfg = MagicMock(n_samples=500, n_features=10, n_classes=3, dirichlet_alpha=0.3)
        cfg = _make_cfg(synthetic=synthetic_cfg)
        loader = DatasetRegistry.get_loader("synthetic", cfg)
        assert isinstance(loader, SyntheticDatasetLoader)


# ---------------------------------------------------------------------------
# register / unregister
# ---------------------------------------------------------------------------


class TestRegisterUnregister:
    def test_register_custom_loader(self):
        class DummyLoader(DatasetLoader):
            def load_client_partitions(self, n, cfg):
                return []

            def load_clean_holdout(self):
                return (None, None)  # type: ignore[return-value]

            def load_evaluation_set(self):
                return (None, None)  # type: ignore[return-value]

        name = "test_dummy_9999"
        # Ensure clean state
        if DatasetRegistry.is_registered(name):
            DatasetRegistry.unregister(name)

        DatasetRegistry.register(name, DummyLoader)
        assert DatasetRegistry.is_registered(name)

        # Clean up
        DatasetRegistry.unregister(name)
        assert not DatasetRegistry.is_registered(name)

    def test_double_register_raises_value_error(self):
        class AnotherLoader(DatasetLoader):
            def load_client_partitions(self, n, cfg):
                return []

            def load_clean_holdout(self):
                return (None, None)  # type: ignore[return-value]

            def load_evaluation_set(self):
                return (None, None)  # type: ignore[return-value]

        name = "test_double_9999"
        if DatasetRegistry.is_registered(name):
            DatasetRegistry.unregister(name)

        DatasetRegistry.register(name, AnotherLoader)
        with pytest.raises(ValueError, match="already registered"):
            DatasetRegistry.register(name, AnotherLoader)

        DatasetRegistry.unregister(name)

    def test_unregister_nonexistent_raises_key_error(self):
        with pytest.raises(KeyError):
            DatasetRegistry.unregister("this_does_not_exist_xyz")
