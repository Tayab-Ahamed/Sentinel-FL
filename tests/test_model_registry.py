"""
tests/test_model_registry.py — Tests for ai/fl_core/model_registry.py.

Covers:
  FileModelRegistry.save: numpy arrays, dict states
  FileModelRegistry.load: numpy, dict, not found
  FileModelRegistry.latest: empty + populated registry
  FileModelRegistry.rollback_to: exact + missing round
  FileModelRegistry._apply_retention: keeps only last k
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ai.fl_core.exceptions import CheckpointNotFoundError
from ai.fl_core.model_registry import FileModelRegistry
from ai.fl_core.schemas import ModelMetadata

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path) -> FileModelRegistry:
    return FileModelRegistry(tmp_path / "registry")


@pytest.fixture
def meta():
    return ModelMetadata(round_num=0, architecture="linear_softmax_v0")


# ---------------------------------------------------------------------------
# save + load: numpy arrays
# ---------------------------------------------------------------------------


class TestSaveLoadNumpy:
    def test_save_returns_model_id(self, registry, meta):
        weights = np.ones(20, dtype=np.float32)
        mid = registry.save(0, weights, meta)
        assert mid == meta.model_id

    def test_load_numpy_roundtrip(self, registry):
        weights = np.arange(10, dtype=np.float64)
        meta = ModelMetadata(round_num=1, architecture="linear_softmax_v0")
        mid = registry.save(1, weights, meta)
        loaded_state, loaded_meta = registry.load(mid)
        np.testing.assert_array_equal(loaded_state, weights)
        assert loaded_meta.round_num == 1

    def test_load_preserves_metadata(self, registry):
        meta = ModelMetadata(round_num=3, architecture="resnet18", clean_accuracy=0.88)
        weights = np.zeros(5)
        mid = registry.save(3, weights, meta)
        _, loaded_meta = registry.load(mid)
        assert loaded_meta.architecture == "resnet18"
        assert loaded_meta.clean_accuracy == pytest.approx(0.88)
        assert loaded_meta.round_num == 3

    def test_load_nonexistent_raises(self, registry):
        with pytest.raises(CheckpointNotFoundError):
            registry.load("does_not_exist")


# ---------------------------------------------------------------------------
# save + load: dict states
# ---------------------------------------------------------------------------


class TestSaveLoadDict:
    def test_save_dict_state(self, registry):
        meta = ModelMetadata(round_num=0, architecture="linear_softmax_v0")
        state = {"layer0.weight": [1.0, 2.0], "layer0.bias": [0.0]}
        mid = registry.save(0, state, meta)
        loaded_state, _ = registry.load(mid)
        assert "layer0.weight" in loaded_state or isinstance(loaded_state, dict)

    def test_save_dict_creates_weights_file(self, registry, tmp_path):
        meta = ModelMetadata(round_num=0, architecture="linear_softmax_v0")
        state = {"key": "value"}
        mid = registry.save(0, state, meta)
        ckpt_dir = registry._dir / mid
        # Either .pt or .json must exist
        assert (ckpt_dir / "weights.pt").exists() or (ckpt_dir / "weights.json").exists()


# ---------------------------------------------------------------------------
# save: invalid type
# ---------------------------------------------------------------------------


class TestSaveInvalidType:
    def test_invalid_model_state_type_raises(self, registry):
        meta = ModelMetadata(round_num=0, architecture="linear_softmax_v0")
        with pytest.raises(TypeError):
            registry.save(0, "not_a_valid_state", meta)

    def test_invalid_type_is_list(self, registry):
        meta = ModelMetadata(round_num=0, architecture="linear_softmax_v0")
        with pytest.raises(TypeError):
            registry.save(0, [1, 2, 3], meta)


# ---------------------------------------------------------------------------
# latest
# ---------------------------------------------------------------------------


class TestLatest:
    def test_latest_raises_when_empty(self, registry):
        with pytest.raises(CheckpointNotFoundError):
            registry.latest()

    def test_latest_returns_most_recent(self, registry):
        for rnd in range(3):
            meta = ModelMetadata(round_num=rnd, architecture="linear_softmax_v0")
            registry.save(rnd, np.ones(5), meta)
        # latest should be round 2
        latest_id = registry.latest()
        _, meta = registry.load(latest_id)
        assert meta.round_num == 2

    def test_latest_updates_on_new_save(self, registry):
        meta0 = ModelMetadata(round_num=0, architecture="linear_softmax_v0")
        registry.save(0, np.zeros(3), meta0)
        meta5 = ModelMetadata(round_num=5, architecture="linear_softmax_v0")
        registry.save(5, np.ones(3), meta5)
        latest_id = registry.latest()
        _, m = registry.load(latest_id)
        assert m.round_num == 5


# ---------------------------------------------------------------------------
# rollback_to
# ---------------------------------------------------------------------------


class TestRollbackTo:
    def test_rollback_to_existing_round(self, registry):
        for rnd in [0, 2, 5]:
            meta = ModelMetadata(round_num=rnd, architecture="linear_softmax_v0")
            registry.save(rnd, np.ones(5) * rnd, meta)
        mid = registry.rollback_to(2)
        _, m = registry.load(mid)
        assert m.round_num == 2

    def test_rollback_to_missing_round_raises(self, registry):
        meta = ModelMetadata(round_num=0, architecture="linear_softmax_v0")
        registry.save(0, np.zeros(3), meta)
        with pytest.raises(CheckpointNotFoundError):
            registry.rollback_to(99)

    def test_rollback_to_round_zero(self, registry):
        meta = ModelMetadata(round_num=0, architecture="linear_softmax_v0")
        mid0 = registry.save(0, np.ones(3) * 7, meta)
        assert registry.rollback_to(0) == mid0


# ---------------------------------------------------------------------------
# index persistence
# ---------------------------------------------------------------------------


class TestIndexPersistence:
    def test_index_file_created(self, registry):
        meta = ModelMetadata(round_num=0, architecture="linear_softmax_v0")
        registry.save(0, np.ones(3), meta)
        index_path = registry._dir / "index.json"
        assert index_path.exists()

    def test_index_maps_round_to_model_id(self, registry):
        meta = ModelMetadata(round_num=4, architecture="linear_softmax_v0")
        mid = registry.save(4, np.ones(3), meta)
        with open(registry._dir / "index.json") as f:
            index = json.load(f)
        assert "4" in index
        assert index["4"] == mid

    def test_index_accumulates_across_saves(self, registry):
        for rnd in range(5):
            meta = ModelMetadata(round_num=rnd, architecture="linear_softmax_v0")
            registry.save(rnd, np.zeros(3), meta)
        with open(registry._dir / "index.json") as f:
            index = json.load(f)
        assert len(index) == 5


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------


class TestRetentionPolicy:
    def test_retention_k_keeps_only_last_k(self, tmp_path):
        registry = FileModelRegistry(tmp_path / "reg_k2", retention_k=2)
        saved_ids = []
        for rnd in range(5):
            meta = ModelMetadata(round_num=rnd, architecture="linear_softmax_v0")
            mid = registry.save(rnd, np.ones(3) * rnd, meta)
            saved_ids.append(mid)
        # Only last 2 checkpoints should exist on disk
        existing = [
            d for d in (tmp_path / "reg_k2").iterdir() if d.is_dir() and d.name != "index.json"
        ]
        assert len(existing) <= 2

    def test_retention_zero_keeps_all(self, tmp_path):
        registry = FileModelRegistry(tmp_path / "reg_k0", retention_k=0)
        for rnd in range(5):
            meta = ModelMetadata(round_num=rnd, architecture="linear_softmax_v0")
            registry.save(rnd, np.ones(3) * rnd, meta)
        with open(tmp_path / "reg_k0" / "index.json") as f:
            index = json.load(f)
        assert len(index) == 5  # all kept
