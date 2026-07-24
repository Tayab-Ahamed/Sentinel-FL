"""PyTorch/CNN coverage for the L5 remediation backend.

No network or dataset download is needed. The module is skipped cleanly in a Phase-0
install and runs in CI's ``.[dev,phase1]`` environment.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.remediation import TorchFinePruner, TorchModelAdapter, TriggerUnlearner
from ai.remediation.triggers import stamp_trigger


class TinyCNN(nn.Module):
    """Small deterministic classifier used only for adapter tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return F.log_softmax(self.fc(x), dim=1)


def _factory():
    torch.manual_seed(7)
    return TinyCNN()


def _params():
    return [p.detach().numpy().copy() for p in _factory().parameters()]


def _data(n=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.2, size=(n, 1, 4, 4)).astype(np.float32)
    y = (X[:, :, :2, :2].mean(axis=(1, 2, 3)) > 0).astype(np.int64)
    X[y == 1, :, :2, :2] += 1.0
    X[y == 0, :, :2, :2] -= 1.0
    return X, y


@pytest.fixture
def adapter():
    return TorchModelAdapter(
        _factory, device="cpu", batch_size=8, momentum=0.0, architecture="tiny_cnn_test"
    )


class TestTorchModelAdapter:
    def test_clone_is_deep(self, adapter):
        params = _params()
        cloned = adapter.clone(params)
        cloned[0].flat[0] = 99
        assert params[0].flat[0] != 99

    def test_predict_and_proba_shapes(self, adapter):
        X, _ = _data(17)
        pred = adapter.predict(_params(), X)
        proba = adapter.predict_proba(_params(), X)
        assert pred.shape == (17,)
        assert pred.dtype == np.int64
        assert proba.shape == (17, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fine_tune_improves_accuracy(self, adapter):
        X, y = _data(64)
        params = _params()
        before = float((adapter.predict(params, X) == y).mean())
        trained = adapter.fine_tune(params, X, y, epochs=15, lr=0.05)
        after = float((adapter.predict(trained, X) == y).mean())
        assert after >= before
        assert after >= 0.9

    def test_empty_fine_tune_returns_copy(self, adapter):
        params = _params()
        out = adapter.fine_tune(params, np.empty((0, 1, 4, 4)), np.empty(0), 2, 0.1)
        assert all(np.array_equal(a, b) for a, b in zip(params, out, strict=True))
        assert all(a is not b for a, b in zip(params, out, strict=True))

    def test_parameter_count_mismatch_is_clear(self, adapter):
        X, _ = _data(2)
        with pytest.raises(ValueError, match="parameter count mismatch"):
            adapter.predict(_params()[:-1], X)

    def test_nonfinite_loss_guard(self, adapter):
        X, y = _data(8)
        X[0, 0, 0, 0] = np.nan
        with pytest.raises(FloatingPointError, match="non-finite"):
            adapter.fine_tune(_params(), X, y, epochs=1, lr=0.1)


class TestTorchFinePruning:
    def test_prune_dormant_channels_emits_evidence(self, adapter):
        X, _ = _data(16)
        pruned, evidence = adapter.prune_dormant_channels(_params(), X, prune_fraction=0.25)
        assert len(pruned) == len(_params())
        assert evidence["layer"] == "conv"
        assert evidence["channels_total"] == 4
        assert evidence["channels_pruned"] == 1
        idx = evidence["pruned_indices"][0]
        assert np.all(pruned[0][idx] == 0)

    def test_pruner_finetunes_after_channel_surgery(self, adapter):
        X, y = _data(24)
        pruner = TorchFinePruner(adapter, finetune_epochs=1, finetune_lr=0.01, prune_fraction=0.25)
        out = pruner.remediate(_params(), X, y, [], n_features=16)
        assert len(out) == len(_params())
        assert pruner.last_evidence["channels_pruned"] == 1


class TestImageTriggerUnlearning:
    def test_nd_trigger_stamping_preserves_shape(self):
        X, _ = _data(5)
        trigger = np.zeros(16)
        trigger[-4:] = 5
        out = stamp_trigger(X, trigger)
        assert out.shape == X.shape
        assert np.all(out.reshape(5, -1)[:, -4:] == 5)

    def test_unlearning_accepts_image_tensors(self, adapter):
        X, y = _data(16)
        trigger = np.zeros(16)
        trigger[-4:] = 5
        unlearner = TriggerUnlearner(adapter, epochs=1, lr=0.01, stamped_replicas=1)
        out = unlearner.remediate(_params(), X, y, [trigger], n_features=16)
        assert len(out) == len(_params())
