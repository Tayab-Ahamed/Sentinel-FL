"""
tests/test_image_poisoning.py — Unit tests for ImagePoisoner.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.attacks.image_poisoning import ImagePoisoner
from ai.attacks.triggers import TriggerFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch(n: int = 50, shape: tuple = (1, 28, 28)) -> tuple[np.ndarray, np.ndarray]:
    """Fake image batch with balanced 5-class labels."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, *shape)).astype(np.float32)
    y = np.tile(np.arange(5), n // 5 + 1)[:n].astype(np.int64)
    return X, y


def _default_poisoner(target: int = 0, frac: float = 0.3) -> ImagePoisoner:
    pattern = TriggerFactory.make_square(size=4, location="bottom_right", color=1.0)
    return ImagePoisoner(pattern=pattern, target_label=target, poison_fraction=frac)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestImagePoisonerConstruction:
    def test_invalid_fraction_zero(self):
        with pytest.raises(ValueError, match="poison_fraction"):
            ImagePoisoner(
                pattern=TriggerFactory.make_square(),
                target_label=0,
                poison_fraction=0.0,
            )

    def test_invalid_fraction_above_one(self):
        with pytest.raises(ValueError, match="poison_fraction"):
            ImagePoisoner(
                pattern=TriggerFactory.make_square(),
                target_label=0,
                poison_fraction=1.5,
            )

    def test_properties_accessible(self):
        p = TriggerFactory.make_square(size=3)
        poisoner = ImagePoisoner(pattern=p, target_label=2, poison_fraction=0.2)
        assert poisoner.target_label == 2
        assert poisoner.poison_fraction == pytest.approx(0.2)
        assert poisoner.pattern is p


# ---------------------------------------------------------------------------
# poison_batch
# ---------------------------------------------------------------------------


class TestPoisonBatch:
    def test_returns_three_items(self):
        X, y = _batch()
        poisoner = _default_poisoner()
        result = poisoner.poison_batch(X, y, seed=0)
        assert len(result) == 3

    def test_output_shapes_match_input(self):
        X, y = _batch()
        poisoner = _default_poisoner()
        X_p, y_p, mask = poisoner.poison_batch(X, y, seed=0)
        assert X_p.shape == X.shape
        assert y_p.shape == y.shape
        assert mask.shape == (len(X),)

    def test_mask_is_bool(self):
        X, y = _batch()
        X_p, y_p, mask = _default_poisoner().poison_batch(X, y, seed=0)
        assert mask.dtype == bool

    def test_poisoned_labels_set_to_target(self):
        X, y = _batch()
        target = 0
        X_p, y_p, mask = _default_poisoner(target=target).poison_batch(X, y, seed=0)
        assert np.all(y_p[mask] == target)

    def test_honest_labels_unchanged(self):
        X, y = _batch()
        X_p, y_p, mask = _default_poisoner().poison_batch(X, y, seed=0)
        np.testing.assert_array_equal(y_p[~mask], y[~mask])

    def test_poison_fraction_respected(self):
        """Actual poisoned fraction should be ~= configured fraction."""
        X, y = _batch(n=100)
        poisoner = _default_poisoner(frac=0.2)
        _, _, mask = poisoner.poison_batch(X, y, seed=0)
        # Non-target samples: 80 (target=0, so 4/5 * 100 = 80)
        n_non_target = int((y != 0).sum())
        expected = int(n_non_target * 0.2)
        assert int(mask.sum()) == expected

    def test_original_X_not_mutated(self):
        X, y = _batch()
        X_original = X.copy()
        _default_poisoner().poison_batch(X, y, seed=0)
        np.testing.assert_array_equal(X, X_original)

    def test_original_y_not_mutated(self):
        X, y = _batch()
        y_original = y.copy()
        _default_poisoner().poison_batch(X, y, seed=0)
        np.testing.assert_array_equal(y, y_original)

    def test_trigger_applied_to_poisoned_images(self):
        """Poisoned images should differ from clean ones in the trigger region."""
        X = np.zeros((20, 1, 28, 28), dtype=np.float32)
        y = np.array([1] * 20, dtype=np.int64)  # no target (target=0)
        poisoner = _default_poisoner(target=0, frac=1.0)
        X_p, _, mask = poisoner.poison_batch(X, y, seed=0)
        # Trigger patch (bottom-right 4×4) should be 1.0
        poisoned_patches = X_p[mask, 0, 24:28, 24:28]
        np.testing.assert_allclose(poisoned_patches, 1.0)

    def test_reproducible_with_same_seed(self):
        X, y = _batch(n=80)
        poisoner = _default_poisoner()
        _, y1, m1 = poisoner.poison_batch(X, y, seed=99)
        _, y2, m2 = poisoner.poison_batch(X, y, seed=99)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_give_different_masks(self):
        X, y = _batch(n=80)
        poisoner = _default_poisoner()
        _, _, m1 = poisoner.poison_batch(X, y, seed=1)
        _, _, m2 = poisoner.poison_batch(X, y, seed=2)
        assert not np.array_equal(m1, m2)

    def test_4d_input_required(self):
        X_bad = np.zeros((10, 28, 28), dtype=np.float32)  # 3D, missing channel
        y = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="4-D"):
            _default_poisoner().poison_batch(X_bad, y)

    def test_empty_candidates_returns_clean(self):
        """If all samples are already target class, no poisoning happens."""
        X = np.zeros((10, 1, 28, 28), dtype=np.float32)
        y = np.zeros(10, dtype=np.int64)  # all already target_label=0
        poisoner = _default_poisoner(target=0)
        X_p, y_p, mask = poisoner.poison_batch(X, y, seed=0)
        assert not np.any(mask)
        np.testing.assert_array_equal(X_p, X)
        np.testing.assert_array_equal(y_p, y)

    def test_cifar10_shape_works(self):
        X, y = _batch(shape=(3, 32, 32))
        X_p, y_p, mask = _default_poisoner().poison_batch(X, y, seed=0)
        assert X_p.shape == X.shape


# ---------------------------------------------------------------------------
# build_asr_eval_set
# ---------------------------------------------------------------------------


class TestBuildASREvalSet:
    def test_output_shapes_correct(self):
        X, y = _batch(n=50)
        poisoner = _default_poisoner(target=0)
        X_t, y_t = poisoner.build_asr_eval_set(X, y)
        assert X_t.ndim == 4
        assert y_t.ndim == 1
        assert X_t.shape[0] == y_t.shape[0]

    def test_all_triggered_labels_are_target(self):
        X, y = _batch(n=50)
        poisoner = _default_poisoner(target=0)
        _, y_t = poisoner.build_asr_eval_set(X, y)
        assert np.all(y_t == 0)

    def test_only_non_target_samples_included(self):
        X, y = _batch(n=50)  # 5 classes, 10 each; target=0 → 10 excluded
        poisoner = _default_poisoner(target=0)
        X_t, _ = poisoner.build_asr_eval_set(X, y)
        n_non_target = int((y != 0).sum())
        assert len(X_t) == n_non_target

    def test_original_X_not_mutated(self):
        X, y = _batch()
        X_original = X.copy()
        _default_poisoner().build_asr_eval_set(X, y)
        np.testing.assert_array_equal(X, X_original)

    def test_all_target_returns_empty(self):
        X = np.zeros((5, 1, 28, 28), dtype=np.float32)
        y = np.zeros(5, dtype=np.int64)  # all target=0
        poisoner = _default_poisoner(target=0)
        X_t, y_t = poisoner.build_asr_eval_set(X, y)
        assert len(X_t) == 0
        assert len(y_t) == 0


# ---------------------------------------------------------------------------
# select_poison_indices
# ---------------------------------------------------------------------------


class TestSelectPoisonIndices:
    def test_returns_correct_count(self):
        X, y = _batch(n=100)
        poisoner = _default_poisoner(target=0, frac=0.5)
        idx = poisoner.select_poison_indices(y, seed=0)
        # 80 non-target samples * 0.5 = 40
        assert len(idx) == 40

    def test_all_indices_are_non_target(self):
        X, y = _batch(n=50)
        poisoner = _default_poisoner(target=0)
        idx = poisoner.select_poison_indices(y, seed=0)
        assert np.all(y[idx] != 0)

    def test_reproducible(self):
        X, y = _batch(n=50)
        poisoner = _default_poisoner()
        idx1 = poisoner.select_poison_indices(y, seed=5)
        idx2 = poisoner.select_poison_indices(y, seed=5)
        np.testing.assert_array_equal(idx1, idx2)
