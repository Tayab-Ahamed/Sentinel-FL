"""
tests/test_poison.py — Unit tests for ai/training/poison.py (TESTING.md §2).

Test requirements from TESTING.md:
  - dirichlet_partition(): partition sizes sum to n_samples; sizes sum correctly.
  - inject_trigger(): exactly poison_fraction * n rows modified; mask matches.
"""

from __future__ import annotations

import numpy as np

from ai.training.poison import (
    BadNetsAttackSimulator,
    apply_trigger_to_all,
    dirichlet_partition,
    inject_trigger,
    make_dataset,
)

# ---------------------------------------------------------------------------
# make_dataset
# ---------------------------------------------------------------------------


class TestMakeDataset:
    def test_shape(self):
        X, y = make_dataset(100, 10, 4, seed=0)
        assert X.shape == (100, 10)
        assert y.shape == (100,)

    def test_all_classes_present(self):
        _X, y = make_dataset(500, 5, 3, seed=0)
        assert len(np.unique(y)) == 3

    def test_reproducible(self):
        X1, y1 = make_dataset(50, 5, 2, seed=7)
        X2, y2 = make_dataset(50, 5, 2, seed=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# dirichlet_partition
# ---------------------------------------------------------------------------


class TestDirichletPartition:
    """TESTING.md: partition sizes sum to n_samples."""

    def test_partition_sizes_sum_to_n_samples(self):
        n_samples = 300
        n_clients = 6
        _, y = make_dataset(n_samples, 10, 4, seed=0)
        indices = dirichlet_partition(n_samples, n_clients, y, 4, alpha=0.5, seed=0)
        total = sum(len(idx) for idx in indices)
        assert total == n_samples, f"Expected {n_samples}, got {total}"

    def test_returns_n_clients_partitions(self):
        n_clients = 8
        _, y = make_dataset(200, 10, 4, seed=1)
        indices = dirichlet_partition(200, n_clients, y, 4, alpha=0.5, seed=0)
        assert len(indices) == n_clients

    def test_no_overlap_between_partitions(self):
        n_samples = 100
        _, y = make_dataset(n_samples, 5, 3, seed=2)
        indices = dirichlet_partition(n_samples, 4, y, 3, seed=0)
        all_indices = np.concatenate(indices)
        # All indices are unique (no overlap)
        assert len(all_indices) == len(np.unique(all_indices))

    def test_non_iid_skew_with_low_alpha(self):
        """Low alpha → more skewed partitions (some clients have few samples)."""
        _, y = make_dataset(1000, 10, 5, seed=3)
        indices_low = dirichlet_partition(1000, 10, y, 5, alpha=0.01, seed=0)
        sizes_low = [len(idx) for idx in indices_low]
        # With low alpha, size variance should be high
        assert np.std(sizes_low) > 10, "Expected high variance with alpha=0.01"

    def test_reproducible(self):
        _, y = make_dataset(200, 5, 3, seed=0)
        i1 = dirichlet_partition(200, 4, y, 3, seed=42)
        i2 = dirichlet_partition(200, 4, y, 3, seed=42)
        for a, b in zip(i1, i2, strict=False):
            np.testing.assert_array_equal(sorted(a), sorted(b))


# ---------------------------------------------------------------------------
# inject_trigger
# ---------------------------------------------------------------------------


class TestInjectTrigger:
    """TESTING.md: exactly poison_fraction * n rows modified; mask matches."""

    def test_exact_poison_count(self):
        X, y = make_dataset(100, 10, 3, seed=0)
        poison_fraction = 0.20
        _X_p, _y_p, mask = inject_trigger(X, y, target_class=0, trigger_block=slice(0, 3),
                                          trigger_value=6.0, poison_fraction=poison_fraction, seed=0)
        expected = int(100 * poison_fraction)
        assert int(mask.sum()) == expected, f"Expected {expected} poisoned rows, got {mask.sum()}"

    def test_mask_matches_modified_rows(self):
        X, y = make_dataset(50, 10, 2, seed=1)
        trigger_block = slice(1, 4)
        trigger_value = 9.0
        X_p, _y_p, mask = inject_trigger(X, y, target_class=1, trigger_block=trigger_block,
                                          trigger_value=trigger_value, poison_fraction=0.3, seed=2)
        # Every row where mask=True must have trigger_value in the trigger_block
        for i in range(len(X)):
            if mask[i]:
                assert np.all(X_p[i, trigger_block] == trigger_value), (
                    f"Row {i} flagged in mask but trigger not set"
                )
            else:
                # Non-poisoned rows should be unchanged
                np.testing.assert_array_equal(X_p[i], X[i])

    def test_label_flip_for_poisoned_rows(self):
        X, y = make_dataset(100, 8, 4, seed=3)
        target_class = 2
        _X_p, y_p, mask = inject_trigger(X, y, target_class=target_class,
                                          trigger_block=slice(0, 2), poison_fraction=0.25, seed=4)
        # All poisoned rows should have y_p == target_class
        assert np.all(y_p[mask] == target_class)
        # Non-poisoned rows should have original labels
        np.testing.assert_array_equal(y_p[~mask], y[~mask])

    def test_original_arrays_not_modified(self):
        X, y = make_dataset(80, 6, 3, seed=5)
        X_orig = X.copy()
        y_orig = y.copy()
        inject_trigger(X, y, target_class=0, trigger_block=slice(0, 2), poison_fraction=0.2, seed=0)
        np.testing.assert_array_equal(X, X_orig)
        np.testing.assert_array_equal(y, y_orig)


# ---------------------------------------------------------------------------
# apply_trigger_to_all
# ---------------------------------------------------------------------------


class TestApplyTriggerToAll:
    def test_all_rows_have_trigger(self):
        X, _ = make_dataset(50, 10, 2, seed=0)
        trigger_block = slice(0, 3)
        trigger_value = 7.0
        X_t = apply_trigger_to_all(X, trigger_block, trigger_value)
        assert np.all(X_t[:, trigger_block] == trigger_value)

    def test_non_trigger_features_unchanged(self):
        X, _ = make_dataset(30, 10, 2, seed=0)
        X_t = apply_trigger_to_all(X, trigger_block=slice(0, 2), trigger_value=5.0)
        np.testing.assert_array_equal(X_t[:, 2:], X[:, 2:])

    def test_original_not_modified(self):
        X, _ = make_dataset(20, 5, 2, seed=0)
        X_orig = X.copy()
        apply_trigger_to_all(X, slice(0, 2), 5.0)
        np.testing.assert_array_equal(X, X_orig)


# ---------------------------------------------------------------------------
# BadNetsAttackSimulator
# ---------------------------------------------------------------------------


class TestBadNetsAttackSimulator:
    def test_honest_client_data_unchanged(self):
        X, y = make_dataset(100, 10, 3, seed=0)
        sim = BadNetsAttackSimulator(malicious_client_indices=[99])  # client 0 is honest
        X_p, _y_p, mask = sim.poison_client_data(X, y, "client_00", round_num=0, config=object())
        assert not mask.any(), "Honest client should have all-False mask"
        np.testing.assert_array_equal(X_p, X)

    def test_malicious_client_data_poisoned(self):
        X, y = make_dataset(100, 10, 3, seed=0)
        sim = BadNetsAttackSimulator(malicious_client_indices=[2], poison_fraction=0.3)
        _X_p, _y_p, mask = sim.poison_client_data(X, y, "client_02", round_num=5, config=object())
        assert mask.any(), "Malicious client should have some poisoned rows"
        expected = int(100 * 0.3)
        assert int(mask.sum()) == expected

    def test_build_trigger_eval_set_applies_to_all(self):
        X, _ = make_dataset(50, 10, 2, seed=0)
        sim = BadNetsAttackSimulator(trigger_block=slice(0, 3), trigger_value=6.0)
        X_t = sim.build_trigger_eval_set(X)
        assert np.all(X_t[:, 0:3] == 6.0)
        np.testing.assert_array_equal(X_t[:, 3:], X[:, 3:])
