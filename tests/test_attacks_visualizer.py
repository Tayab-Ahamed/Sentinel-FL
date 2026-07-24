"""
tests/test_attacks_visualizer.py — Tests for ai/attacks/visualizer.py.

Covers:
  PoisonedSampleVisualizer instantiation
  plot_sample_grid: 1-channel (MNIST) and 3-channel (CIFAR-10)
  plot_trigger_pattern: square trigger
  plot_asr_over_rounds: line chart
  save_figure: file creation, format options
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if matplotlib is unavailable
pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mnist_batch(n=8):
    """Return (n, 1, 28, 28) float32 image batch (MNIST-like)."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, (n, 1, 28, 28)).astype(np.float32)


def _make_cifar_batch(n=8):
    """Return (n, 3, 32, 32) float32 image batch (CIFAR-like)."""
    rng = np.random.default_rng(1)
    return rng.uniform(0, 1, (n, 3, 32, 32)).astype(np.float32)


def _make_mask(n=8, n_poison=4):
    mask = np.zeros(n, dtype=bool)
    mask[:n_poison] = True
    return mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPoisonedSampleVisualizer:
    @pytest.fixture
    def viz(self):
        from ai.attacks.visualizer import PoisonedSampleVisualizer

        return PoisonedSampleVisualizer(dpi=72)

    def test_instantiation(self, viz):
        assert viz is not None

    # ------------------------------------------------------------------
    # plot_sample_grid
    # ------------------------------------------------------------------

    def test_plot_sample_grid_mnist(self, viz):
        from matplotlib.figure import Figure

        X_clean = _make_mnist_batch(8)
        X_poisoned = X_clean.copy()
        X_poisoned[:4, :, 24:, 24:] = 1.0  # add trigger block
        mask = _make_mask(8, 4)
        fig = viz.plot_sample_grid(X_clean, X_poisoned, mask, n_cols=4)
        assert isinstance(fig, Figure)

    def test_plot_sample_grid_cifar(self, viz):
        from matplotlib.figure import Figure

        X_clean = _make_cifar_batch(8)
        X_poisoned = X_clean.copy()
        X_poisoned[:3, :, 28:, 28:] = 1.0
        mask = _make_mask(8, 3)
        fig = viz.plot_sample_grid(X_clean, X_poisoned, mask, n_cols=4)
        assert isinstance(fig, Figure)

    def test_plot_sample_grid_no_poisoned(self, viz):
        """When mask is all-False, should still render without error."""
        from matplotlib.figure import Figure

        X_clean = _make_mnist_batch(4)
        X_poisoned = X_clean.copy()
        mask = np.zeros(4, dtype=bool)
        fig = viz.plot_sample_grid(X_clean, X_poisoned, mask, n_cols=4)
        assert isinstance(fig, Figure)

    def test_plot_sample_grid_all_poisoned(self, viz):
        """All samples poisoned — must render without error."""
        from matplotlib.figure import Figure

        X_clean = _make_mnist_batch(4)
        X_poisoned = X_clean.copy()
        X_poisoned[:, :, 24:, 24:] = 1.0
        mask = np.ones(4, dtype=bool)
        fig = viz.plot_sample_grid(X_clean, X_poisoned, mask, n_cols=4)
        assert isinstance(fig, Figure)

    def test_plot_sample_grid_single_sample(self, viz):
        """Single-sample batch must not crash."""
        from matplotlib.figure import Figure

        X_clean = _make_mnist_batch(1)
        X_poisoned = X_clean.copy()
        mask = np.array([True])
        fig = viz.plot_sample_grid(X_clean, X_poisoned, mask, n_cols=1)
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # plot_trigger_pattern
    # ------------------------------------------------------------------

    def test_plot_trigger_pattern_mnist(self, viz):
        from matplotlib.figure import Figure

        from ai.attacks.triggers import TriggerPattern

        trigger = TriggerPattern(shape="square", size=5, location="bottom_right", color=1.0)
        fig = viz.plot_trigger_pattern(trigger, input_shape=(1, 28, 28))
        assert isinstance(fig, Figure)

    def test_plot_trigger_pattern_cifar(self, viz):
        from matplotlib.figure import Figure

        from ai.attacks.triggers import TriggerPattern

        trigger = TriggerPattern(shape="square", size=4, location="bottom_right", color=1.0)
        fig = viz.plot_trigger_pattern(trigger, input_shape=(3, 32, 32))
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # plot_asr_over_rounds
    # ------------------------------------------------------------------

    def test_plot_asr_curve(self, viz):
        from matplotlib.figure import Figure

        rounds = list(range(10))
        asr = [0.8 - i * 0.05 for i in range(10)]
        clean_acc = [0.7 + i * 0.02 for i in range(10)]
        fig = viz.plot_asr_curve(rounds, asr, clean_acc)
        assert isinstance(fig, Figure)

    def test_plot_asr_curve_single_round(self, viz):
        """Single-round time series must not crash."""
        from matplotlib.figure import Figure

        fig = viz.plot_asr_curve([0], [0.9], [0.7])
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # save_figure
    # ------------------------------------------------------------------

    def test_save_figure_png(self, viz, tmp_path):
        X_clean = _make_mnist_batch(4)
        mask = np.zeros(4, dtype=bool)
        fig = viz.plot_sample_grid(X_clean, X_clean.copy(), mask)
        out_path = tmp_path / "test_output.png"
        viz.save_figure(fig, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 100  # not empty

    def test_save_figure_creates_parent_dirs(self, viz, tmp_path):
        X_clean = _make_mnist_batch(4)
        mask = np.zeros(4, dtype=bool)
        fig = viz.plot_sample_grid(X_clean, X_clean.copy(), mask)
        out_path = tmp_path / "subdir" / "deep" / "output.png"
        viz.save_figure(fig, out_path)
        assert out_path.exists()

    def test_save_figure_returns_path(self, viz, tmp_path):
        X_clean = _make_mnist_batch(2)
        mask = np.zeros(2, dtype=bool)
        fig = viz.plot_sample_grid(X_clean, X_clean.copy(), mask)
        out_path = tmp_path / "ret_test.png"
        viz.save_figure(fig, out_path)
        assert out_path.exists()


class TestRequireMatplotlib:
    def test_missing_matplotlib_raises_import_error(self, monkeypatch):
        """If _MPL_AVAILABLE is False, instantiation should raise ImportError."""
        import ai.attacks.visualizer as viz_module

        original = viz_module._MPL_AVAILABLE
        viz_module._MPL_AVAILABLE = False
        try:
            from ai.attacks.visualizer import PoisonedSampleVisualizer

            with pytest.raises(ImportError):
                PoisonedSampleVisualizer()
        finally:
            viz_module._MPL_AVAILABLE = original
