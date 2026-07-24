"""
tests/test_triggers.py — Unit tests for TriggerPattern and apply_trigger.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.attacks.triggers import (
    TriggerFactory,
    TriggerPattern,
    apply_trigger,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mnist_batch(n: int = 8) -> np.ndarray:
    """Return a fake MNIST batch (N, 1, 28, 28) float32."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 1, 28, 28)).astype(np.float32)


def _cifar_batch(n: int = 8) -> np.ndarray:
    """Return a fake CIFAR-10 batch (N, 3, 32, 32) float32."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((n, 3, 32, 32)).astype(np.float32)


# ---------------------------------------------------------------------------
# TriggerPattern construction
# ---------------------------------------------------------------------------


class TestTriggerPatternConstruction:
    def test_default_pattern_is_valid(self):
        p = TriggerPattern()
        assert p.shape == "square"
        assert p.size == 4
        assert p.location == "bottom_right"
        assert p.opacity == 1.0

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="size"):
            TriggerPattern(size=0)

    def test_invalid_opacity_raises(self):
        with pytest.raises(ValueError, match="opacity"):
            TriggerPattern(opacity=1.5)

    def test_factory_make_square(self):
        p = TriggerFactory.make_square(size=3, location="top_left", color=0.5)
        assert p.shape == "square"
        assert p.size == 3

    def test_factory_make_cross(self):
        p = TriggerFactory.make_cross(size=5)
        assert p.shape == "cross"

    def test_factory_make_checkerboard(self):
        p = TriggerFactory.make_checkerboard(size=4)
        assert p.shape == "checkerboard"

    def test_factory_make_random_noise(self):
        p = TriggerFactory.make_random_noise(size=4, seed=7)
        assert p.shape == "random_noise"
        assert p.seed == 7


# ---------------------------------------------------------------------------
# Stamp generation
# ---------------------------------------------------------------------------


class TestStampGeneration:
    @pytest.mark.parametrize("shape", ["square", "cross", "checkerboard", "random_noise"])
    def test_stamp_shape_is_correct(self, shape):
        p = TriggerPattern(shape=shape, size=4)
        stamp = p.stamp(n_channels=1, h_img=28, w_img=28)
        assert stamp.shape == (1, 4, 4)

    def test_stamp_multichannel(self):
        p = TriggerPattern(shape="square", size=4, color=(0.5, 0.6, 0.7))
        stamp = p.stamp(n_channels=3, h_img=32, w_img=32)
        assert stamp.shape == (3, 4, 4)

    def test_square_stamp_is_all_ones_times_color(self):
        p = TriggerPattern(shape="square", size=3, color=0.8)
        stamp = p.stamp(n_channels=1, h_img=28, w_img=28)
        np.testing.assert_allclose(stamp[0], 0.8 * np.ones((3, 3), dtype=np.float32))

    def test_cross_stamp_has_correct_structure(self):
        p = TriggerPattern(shape="cross", size=5)
        stamp = p.stamp(n_channels=1, h_img=28, w_img=28)
        # Centre row and column should be 1.0
        mid = 2
        assert float(stamp[0, mid, :].mean()) == pytest.approx(1.0)
        assert float(stamp[0, :, mid].mean()) == pytest.approx(1.0)

    def test_checkerboard_has_alternating_pattern(self):
        p = TriggerPattern(shape="checkerboard", size=4)
        stamp = p.stamp(n_channels=1, h_img=28, w_img=28)
        m = stamp[0]
        # Each pixel should be 0 or 1
        assert set(m.flatten().tolist()).issubset({0.0, 1.0})
        # Alternating: (0,0) and (0,1) should differ
        assert m[0, 0] != m[0, 1]

    def test_random_noise_is_reproducible(self):
        p1 = TriggerPattern(shape="random_noise", size=4, seed=42)
        p2 = TriggerPattern(shape="random_noise", size=4, seed=42)
        s1 = p1.stamp(1, 28, 28)
        s2 = p2.stamp(1, 28, 28)
        np.testing.assert_array_equal(s1, s2)

    def test_color_tuple_wrong_length_raises(self):
        p = TriggerPattern(shape="square", size=4, color=(0.5, 0.5))  # 2 vs 3 channels
        with pytest.raises(ValueError, match="color tuple length"):
            p.stamp(n_channels=3, h_img=32, w_img=32)


# ---------------------------------------------------------------------------
# Patch slice
# ---------------------------------------------------------------------------


class TestPatchSlice:
    def test_bottom_right_location(self):
        p = TriggerPattern(size=4, location="bottom_right")
        r, c = p.get_patch_slice(28, 28)
        assert r == slice(24, 28)
        assert c == slice(24, 28)

    def test_top_left_location(self):
        p = TriggerPattern(size=4, location="top_left")
        r, c = p.get_patch_slice(28, 28)
        assert r == slice(0, 4)
        assert c == slice(0, 4)

    def test_center_location(self):
        p = TriggerPattern(size=4, location="center")
        r, c = p.get_patch_slice(28, 28)
        assert r == slice(12, 16)
        assert c == slice(12, 16)

    def test_clamps_size_to_image_dims(self):
        p = TriggerPattern(size=100, location="bottom_right")
        r, c = p.get_patch_slice(28, 28)
        # Should clamp to (28, 28) patch
        assert r.stop - r.start == 28
        assert c.stop - c.start == 28


# ---------------------------------------------------------------------------
# apply_trigger — MNIST
# ---------------------------------------------------------------------------


class TestApplyTriggerMNIST:
    def test_returns_copy_not_view(self):
        X = _mnist_batch()
        pattern = TriggerFactory.make_square(size=4)
        X_t = apply_trigger(X, pattern)
        X[0] = 0.0  # mutate original
        assert not np.array_equal(X_t[0], X[0])

    def test_output_shape_unchanged(self):
        X = _mnist_batch()
        X_t = apply_trigger(X, TriggerFactory.make_square())
        assert X_t.shape == X.shape

    def test_trigger_region_is_modified(self):
        X = np.zeros((4, 1, 28, 28), dtype=np.float32)
        pattern = TriggerFactory.make_square(size=4, location="bottom_right", color=1.0)
        X_t = apply_trigger(X, pattern)
        patch = X_t[0, 0, 24:28, 24:28]
        np.testing.assert_allclose(patch, 1.0)

    def test_non_trigger_region_is_unchanged(self):
        X = np.zeros((4, 1, 28, 28), dtype=np.float32)
        pattern = TriggerFactory.make_square(size=4, location="bottom_right", color=1.0)
        X_t = apply_trigger(X, pattern)
        # Top-left region should still be 0
        np.testing.assert_allclose(X_t[0, 0, :20, :20], 0.0)

    def test_opacity_blends_correctly(self):
        X = np.ones((2, 1, 28, 28), dtype=np.float32)  # all 1s
        pattern = TriggerPattern(
            shape="square", size=4, location="bottom_right", color=0.0, opacity=0.5
        )  # black, 50% blend
        X_t = apply_trigger(X, pattern)
        patch = X_t[0, 0, 24:28, 24:28]
        # 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        np.testing.assert_allclose(patch, 0.5, atol=1e-6)

    def test_single_image_3d_input(self):
        img = np.zeros((1, 28, 28), dtype=np.float32)
        pattern = TriggerFactory.make_square(size=4, color=1.0)
        out = apply_trigger(img, pattern)
        assert out.shape == (1, 28, 28)

    def test_wrong_ndim_raises(self):
        bad = np.zeros((28, 28), dtype=np.float32)
        with pytest.raises(ValueError, match="3-D|4-D"):
            apply_trigger(bad, TriggerFactory.make_square())


# ---------------------------------------------------------------------------
# apply_trigger — CIFAR-10 (3-channel)
# ---------------------------------------------------------------------------


class TestApplyTriggerCIFAR:
    def test_output_shape_unchanged(self):
        X = _cifar_batch()
        X_t = apply_trigger(X, TriggerFactory.make_square(size=4))
        assert X_t.shape == X.shape

    def test_trigger_region_modified_all_channels(self):
        X = np.zeros((2, 3, 32, 32), dtype=np.float32)
        pattern = TriggerFactory.make_square(size=4, location="bottom_right", color=(1.0, 0.5, 0.0))
        X_t = apply_trigger(X, pattern)
        # Channel 0 → 1.0, Channel 1 → 0.5, Channel 2 → 0.0
        np.testing.assert_allclose(X_t[0, 0, 28:32, 28:32], 1.0)
        np.testing.assert_allclose(X_t[0, 1, 28:32, 28:32], 0.5)
        np.testing.assert_allclose(X_t[0, 2, 28:32, 28:32], 0.0)

    def test_all_shapes_work(self):
        for shape in ("square", "cross", "checkerboard", "random_noise"):
            X = _cifar_batch()
            pattern = TriggerPattern(shape=shape, size=4)
            X_t = apply_trigger(X, pattern)
            assert X_t.shape == X.shape
