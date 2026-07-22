"""
tests/test_mnist_cnn.py — Unit tests for SimpleCNN and parameter helpers.

Tests follow TESTING.md conventions:
  - Pure NumPy/PyTorch, no network access.
  - Deterministic with manual_seed.
  - Fast: no MNIST download, no FL round overhead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ai.models.mnist_cnn import SimpleCNN, get_model_parameters, set_model_parameters


class TestSimpleCNN:
    """Forward pass shape and output validity tests."""

    def test_output_shape_single_input(self):
        """Single image → output shape (1, 10)."""
        model = SimpleCNN()
        model.eval()
        x = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10), f"Expected (1, 10), got {out.shape}"

    def test_output_shape_batch(self):
        """Batch of 32 images → output shape (32, 10)."""
        model = SimpleCNN()
        model.eval()
        x = torch.zeros(32, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32, 10)

    def test_output_is_log_softmax(self):
        """Output should be log-probabilities: exp sums to ~1, all values ≤ 0."""
        torch.manual_seed(0)
        model = SimpleCNN()
        model.eval()
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        # All log-probs must be ≤ 0
        assert (out <= 0).all(), "log_softmax output must be ≤ 0"
        # exp(out) must sum to ~1 per row
        probs = out.exp()
        row_sums = probs.sum(dim=1)
        np.testing.assert_allclose(
            row_sums.numpy(), np.ones(4), atol=1e-5
        )

    def test_num_classes_override(self):
        """num_classes kwarg changes last layer width."""
        model = SimpleCNN(num_classes=5)
        model.eval()
        x = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_parameter_count_reasonable(self):
        """Model should have a sensible parameter count (not trivial, not huge)."""
        model = SimpleCNN()
        total = sum(p.numel() for p in model.parameters())
        # 2-block CNN: ~421K params; allow generous range
        assert 50_000 < total < 2_000_000, f"Unexpected param count: {total}"

    def test_different_seeds_different_outputs(self):
        """Two different initialisations should produce different outputs."""
        torch.manual_seed(0)
        m1 = SimpleCNN()
        torch.manual_seed(99)
        m2 = SimpleCNN()
        x = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            o1, o2 = m1(x), m2(x)
        assert not torch.allclose(o1, o2)


class TestParameterHelpers:
    """Tests for get_model_parameters / set_model_parameters round-trip."""

    def test_get_returns_list_of_ndarrays(self):
        """get_model_parameters returns a list of numpy arrays."""
        model = SimpleCNN()
        params = get_model_parameters(model)
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_round_trip_preserves_values(self):
        """set_model_parameters(get_model_parameters(m)) is identity."""
        torch.manual_seed(7)
        model = SimpleCNN()
        params_before = get_model_parameters(model)
        set_model_parameters(model, params_before)
        params_after = get_model_parameters(model)
        assert len(params_before) == len(params_after)
        for arr_before, arr_after in zip(params_before, params_after, strict=True):
            np.testing.assert_array_equal(arr_before, arr_after)

    def test_set_updates_model_weights(self):
        """After set_model_parameters with zeros, all parameters are zero."""
        model = SimpleCNN()
        zero_params = [np.zeros_like(p) for p in get_model_parameters(model)]
        set_model_parameters(model, zero_params)
        for p in model.parameters():
            assert (p.data == 0).all()

    def test_forward_after_set_parameters(self):
        """Model produces valid output after set_model_parameters."""
        torch.manual_seed(3)
        model = SimpleCNN()
        params = get_model_parameters(model)
        set_model_parameters(model, params)
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)
        assert torch.isfinite(out).all()

    def test_mismatched_params_raises(self):
        """set_model_parameters raises when parameter count doesn't match."""
        model = SimpleCNN()
        too_few = get_model_parameters(model)[:-1]  # drop last param
        with pytest.raises((ValueError, TypeError)):
            set_model_parameters(model, too_few)
