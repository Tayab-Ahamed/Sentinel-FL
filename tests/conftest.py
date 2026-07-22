"""
tests/conftest.py — Shared pytest fixtures for SENTINEL-FL test suite.

See TESTING.md for the four test tiers and reproducibility conventions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai.fl_core.config import load_config_from_dict
from ai.fl_core.schemas import Configuration

# ---------------------------------------------------------------------------
# Minimal valid configuration fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config() -> Configuration:
    """Return a minimal valid Configuration object for unit tests."""
    return load_config_from_dict(
        {
            "n_clients": 4,
            "n_rounds": 3,
            "min_clients": 2,
            "aggregator": "multi_krum",
            "krum_f": 1,
            "krum_select": 3,
            "collusion_sim_threshold": 0.85,
            "collusion_min_cluster_size": 2,
            "audit_interval_rounds": 5,
            "detectors": ["strip_entropy"],
            "strip_n_perturb": 10,
            "strip_target_frr": 0.02,
            "dataset_phase": "phase0_synthetic",
            "synthetic": {
                "n_samples": 200,
                "n_features": 10,
                "n_classes": 3,
                "dirichlet_alpha": 0.5,
            },
            "attack": {
                "type": "badnet_colluding",
                "target_class": 0,
                "trigger_block_start": 0,
                "trigger_block_end": 2,
                "trigger_value": 5.0,
                "poison_fraction": 0.2,
                "malicious_client_indices": [0],
            },
            "local_epochs": 2,
            "local_lr": 0.1,
            "seed": 0,
            "log_level": "WARNING",
            "log_sink": "stdout",
        }
    )


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a tiny (200, 10) synthetic dataset for fast unit tests."""
    from ai.training.poison import make_dataset

    return make_dataset(n_samples=200, n_features=10, n_classes=3, seed=0)


@pytest.fixture
def toy_client_updates() -> tuple[list[np.ndarray], np.ndarray]:
    """Return 4 toy client update vectors for aggregation unit tests.

    Client 0 and 1 are honest; clients 2 and 3 are colluding (similar directions).
    """
    rng = np.random.default_rng(42)
    # Honest clients — random independent directions
    u0 = rng.normal(0, 1, 20)
    u1 = rng.normal(0, 1, 20)
    # Colluding clients — same direction + small noise
    base = rng.normal(0, 1, 20)
    base = base / np.linalg.norm(base)
    u2 = base + rng.normal(0, 0.05, 20)
    u3 = base + rng.normal(0, 0.05, 20)
    updates = [u0, u1, u2, u3]
    aggregate = np.mean(updates, axis=0)
    return updates, aggregate


# ---------------------------------------------------------------------------
# Temporary directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory path for test artefacts."""
    return tmp_path
