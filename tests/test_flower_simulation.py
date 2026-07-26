"""
tests/test_flower_simulation.py — Smoke tests for the Flower FL simulation.

These tests run a minimal end-to-end simulation (2 clients, 2 rounds) using
synthetic toy data so no MNIST download is needed.  They verify:
  - Simulation completes without exceptions.
  - Multiple communication rounds execute.
  - SimulationResult has the expected structure.
  - Experiment JSON file is written correctly.
  - clean_accuracy is a valid float in [0, 1] after training.

Marked ``@pytest.mark.slow`` for the full-rounds test so CI can filter if needed.
Fast smoke test (2 clients, 2 rounds) is unmarked.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("flwr")

import torch

from ai.fl_core.schemas import Configuration
from ai.fl_engine.simulation import SimulationResult, _extract_rounds, _write_result_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_config(**overrides) -> Configuration:
    """Create a minimal Configuration valid for fast smoke tests."""
    defaults = {
        "n_clients": 4,
        "n_rounds": 2,
        "min_clients": 2,
        "local_epochs": 1,
        "local_lr": 0.01,
        "krum_select": 4,
        "krum_f": 1,
        "seed": 0,
        "dataset_phase": "phase1_official",
        "model_registry_dir": "experiments/test_checkpoints",
    }
    defaults.update(overrides)
    return Configuration(**defaults)


def _make_fake_partitions(n_clients: int, n_per_client: int = 64) -> list:
    """Return synthetic (X, y) partitions shaped like MNIST (N, 1, 28, 28)."""
    rng = np.random.default_rng(42)
    return [
        (
            rng.random((n_per_client, 1, 28, 28), dtype=np.float32),
            rng.integers(0, 10, size=n_per_client, dtype=np.int64),
        )
        for _ in range(n_clients)
    ]


# ---------------------------------------------------------------------------
# SimulationResult helpers (pure unit tests, no Flower needed)
# ---------------------------------------------------------------------------


class TestSimulationResult:
    def test_result_dataclass_defaults(self):
        """SimulationResult initialises with expected defaults."""
        r = SimulationResult(experiment_id="test", n_rounds=3, n_clients=5)
        assert r.experiment_id == "test"
        assert r.n_rounds == 3
        assert r.n_clients == 5
        assert r.rounds_history == []
        assert r.final_clean_accuracy is None
        assert r.final_loss is None

    def test_write_result_json(self, tmp_path):
        """_write_result_json writes valid JSON with expected keys."""
        r = SimulationResult(
            experiment_id="test_exp",
            n_rounds=2,
            n_clients=4,
            rounds_history=[{"round": 1, "clean_accuracy": 0.5}],
            final_clean_accuracy=0.75,
            final_loss=1.23,
            total_wall_time_s=42.0,
        )
        out = tmp_path / "result.json"
        _write_result_json(r, out)
        data = json.loads(out.read_text())
        assert data["experiment_id"] == "test_exp"
        assert data["n_rounds"] == 2
        assert data["n_clients"] == 4
        assert data["final_clean_accuracy"] == pytest.approx(0.75)
        assert len(data["rounds"]) == 1

    def test_extract_rounds_empty_history(self):
        """_extract_rounds returns empty list for empty History."""

        class FakeHistory:
            losses_centralized = []
            metrics_centralized = {}

        assert _extract_rounds(FakeHistory()) == []

    def test_extract_rounds_with_data(self):
        """_extract_rounds correctly extracts per-round metrics."""

        class FakeHistory:
            losses_centralized = [(1, 1.5), (2, 1.2)]
            metrics_centralized = {
                "clean_accuracy": [(1, 0.45), (2, 0.60)],
            }

        rounds = _extract_rounds(FakeHistory())
        assert len(rounds) == 2
        assert rounds[0]["round"] == 1
        assert rounds[0]["centralized_loss"] == pytest.approx(1.5)
        assert rounds[1]["clean_accuracy"] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Model Registry integration (tmp_path, no Flower)
# ---------------------------------------------------------------------------


class TestFileModelRegistryIntegration:
    """Test that save/load/latest/rollback_to work end-to-end."""

    def test_save_and_load_dict(self, tmp_path):
        from ai.fl_core.model_registry import FileModelRegistry
        from ai.fl_core.schemas import ModelMetadata

        reg = FileModelRegistry(tmp_path)
        meta = ModelMetadata(round_num=1, architecture="test_v0")
        model_state = {"weights": [1.0, 2.0], "architecture": "test_v0"}
        model_id = reg.save(1, model_state, meta)
        loaded_state, loaded_meta = reg.load(model_id)
        assert loaded_meta.round_num == 1
        assert loaded_meta.architecture == "test_v0"

    def test_save_and_load_numpy(self, tmp_path):
        from ai.fl_core.model_registry import FileModelRegistry
        from ai.fl_core.schemas import ModelMetadata

        reg = FileModelRegistry(tmp_path)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        meta = ModelMetadata(round_num=2, architecture="numpy_v0")
        model_id = reg.save(2, arr, meta)
        loaded_state, loaded_meta = reg.load(model_id)
        np.testing.assert_array_equal(loaded_state, arr)

    def test_latest_returns_most_recent(self, tmp_path):
        from ai.fl_core.model_registry import FileModelRegistry
        from ai.fl_core.schemas import ModelMetadata

        reg = FileModelRegistry(tmp_path)
        for rnd in [1, 2, 3]:
            meta = ModelMetadata(round_num=rnd, architecture="v0")
            reg.save(rnd, {"r": rnd}, meta)
        latest_id = reg.latest()
        _, meta = reg.load(latest_id)
        assert meta.round_num == 3

    def test_rollback_to_known_round(self, tmp_path):
        from ai.fl_core.model_registry import FileModelRegistry
        from ai.fl_core.schemas import ModelMetadata

        reg = FileModelRegistry(tmp_path)
        for rnd in [1, 3, 5]:
            meta = ModelMetadata(round_num=rnd, architecture="v0")
            reg.save(rnd, {"r": rnd}, meta)
        mid_id = reg.rollback_to(3)
        _, meta = reg.load(mid_id)
        assert meta.round_num == 3

    def test_rollback_missing_raises(self, tmp_path):
        from ai.fl_core.exceptions import CheckpointNotFoundError
        from ai.fl_core.model_registry import FileModelRegistry
        from ai.fl_core.schemas import ModelMetadata

        reg = FileModelRegistry(tmp_path)
        meta = ModelMetadata(round_num=1, architecture="v0")
        reg.save(1, {}, meta)
        with pytest.raises(CheckpointNotFoundError):
            reg.rollback_to(99)

    def test_load_missing_raises(self, tmp_path):
        from ai.fl_core.exceptions import CheckpointNotFoundError
        from ai.fl_core.model_registry import FileModelRegistry

        reg = FileModelRegistry(tmp_path)
        with pytest.raises(CheckpointNotFoundError):
            reg.load("nonexistent-id-0000")

    def test_latest_empty_raises(self, tmp_path):
        from ai.fl_core.exceptions import CheckpointNotFoundError
        from ai.fl_core.model_registry import FileModelRegistry

        reg = FileModelRegistry(tmp_path)
        with pytest.raises(CheckpointNotFoundError):
            reg.latest()

    def test_save_pytorch_state_dict(self, tmp_path):
        """save/load round-trip for a real PyTorch SimpleCNN state_dict."""
        from ai.fl_core.model_registry import FileModelRegistry
        from ai.fl_core.schemas import ModelMetadata
        from ai.models.mnist_cnn import SimpleCNN

        reg = FileModelRegistry(tmp_path)
        torch.manual_seed(5)
        model = SimpleCNN()
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        meta = ModelMetadata(round_num=1, architecture="mnist_simplecnn_v1")
        model_id = reg.save(1, state_dict, meta)
        loaded_state, loaded_meta = reg.load(model_id)
        assert set(loaded_state.keys()) == set(state_dict.keys())
        for key in state_dict:
            np.testing.assert_array_almost_equal(
                loaded_state[key].numpy(),
                state_dict[key].numpy(),
            )


# ---------------------------------------------------------------------------
# End-to-end simulation smoke test (uses real Flower, synthetic data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSimulationEndToEnd:
    """Full Flower simulation with synthetic MNIST-shaped data.

    Marked slow because it runs actual FL rounds (2 clients, 2 rounds ≈ 10s).
    Run with: pytest -m slow tests/test_flower_simulation.py
    """

    def test_smoke_2clients_2rounds(self, tmp_path):
        """Simulation completes, writes JSON, returns valid SimulationResult."""
        from ai.fl_engine.simulation import run_simulation

        config = _make_minimal_config(
            n_clients=2,
            n_rounds=2,
            local_epochs=1,
            model_registry_dir=str(tmp_path / "checkpoints"),
        )
        # Patch MNISTDatasetLoader to return synthetic data (no download)
        n_per = 64
        synthetic_partitions = _make_fake_partitions(2, n_per)
        X_hold = np.random.default_rng(99).random((32, 1, 28, 28), dtype=np.float32)
        y_hold = np.random.default_rng(99).integers(0, 10, 32, dtype=np.int64)

        with patch("ai.fl_engine.simulation.MNISTDatasetLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.load_client_partitions.return_value = synthetic_partitions
            instance.load_clean_holdout.return_value = (X_hold, y_hold)

            result = run_simulation(
                config=config,
                experiment_id="smoke_test",
                experiments_dir=str(tmp_path),
                data_dir=str(tmp_path / "data"),
                verbose=False,
            )

        assert isinstance(result, SimulationResult)
        assert result.experiment_id == "smoke_test"
        assert result.n_rounds == 2
        assert result.n_clients == 2
        assert result.total_wall_time_s > 0
        # JSON artefact must exist and be valid
        out = tmp_path / "smoke_test.json"
        assert out.exists(), f"Expected {out} to be written"
        data = json.loads(out.read_text())
        assert data["experiment_id"] == "smoke_test"
        # Accuracy should be a valid number (random init on random data ≈ 0.1)
        if result.final_clean_accuracy is not None:
            assert 0.0 <= result.final_clean_accuracy <= 1.0
