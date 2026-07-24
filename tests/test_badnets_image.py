"""
tests/test_badnets_image.py — Unit tests for BadNetsImageAttack.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from ai.attacks.badnets import BadNetsImageAttack, _parse_client_id
from ai.attacks.triggers import TriggerFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch(n: int = 40, channels: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, channels, 28, 28)).astype(np.float32)
    y = rng.integers(0, 5, size=n).astype(np.int64)
    return X, y


def _default_attacker(
    target: int = 0,
    malicious: list[int] | None = None,
    frac: float = 0.3,
) -> BadNetsImageAttack:
    return BadNetsImageAttack(
        target_label=target,
        poison_fraction=frac,
        malicious_client_indices=malicious if malicious is not None else [2, 5],
        pattern=TriggerFactory.make_square(size=4),
        seed=42,
    )


def _cfg() -> MagicMock:
    return MagicMock(seed=42)


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestBadNetsImageAttackProperties:
    def test_name_attribute(self):
        assert BadNetsImageAttack.name == "badnets_image"

    def test_target_label_property(self):
        a = _default_attacker(target=3)
        assert a.target_label == 3

    def test_malicious_client_indices_property(self):
        a = _default_attacker(malicious=[1, 4, 7])
        assert a.malicious_client_indices == {1, 4, 7}

    def test_trigger_pattern_property(self):
        pattern = TriggerFactory.make_cross(size=5)
        a = BadNetsImageAttack(pattern=pattern)
        assert a.trigger_pattern is pattern


# ---------------------------------------------------------------------------
# poison_client_data — honest clients
# ---------------------------------------------------------------------------


class TestHonestClientUnchanged:
    def test_honest_client_returns_original_X(self):
        X, y = _batch()
        attacker = _default_attacker(malicious=[2, 5])
        X_p, y_p, mask = attacker.poison_client_data(X, y, "client_00", 0, _cfg())
        np.testing.assert_array_equal(X_p, X)

    def test_honest_client_returns_original_y(self):
        X, y = _batch()
        attacker = _default_attacker(malicious=[2, 5])
        X_p, y_p, mask = attacker.poison_client_data(X, y, "client_00", 0, _cfg())
        np.testing.assert_array_equal(y_p, y)

    def test_honest_client_mask_all_false(self):
        X, y = _batch()
        attacker = _default_attacker(malicious=[2, 5])
        _, _, mask = attacker.poison_client_data(X, y, "client_01", 0, _cfg())
        assert not np.any(mask)

    def test_non_numeric_honest_client_id(self):
        X, y = _batch()
        attacker = _default_attacker(malicious=[2, 5])
        _, _, mask = attacker.poison_client_data(X, y, "some_random_name", 0, _cfg())
        assert not np.any(mask)


# ---------------------------------------------------------------------------
# poison_client_data — malicious clients
# ---------------------------------------------------------------------------


class TestMaliciousClientPoisoned:
    def test_malicious_client_produces_nonzero_mask(self):
        X, y = _batch(n=60)
        attacker = _default_attacker(malicious=[2], frac=0.3)
        _, _, mask = attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        assert np.any(mask)

    def test_poisoned_labels_are_target(self):
        X, y = _batch(n=60)
        attacker = _default_attacker(malicious=[2], target=0, frac=0.5)
        _, y_p, mask = attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        assert np.all(y_p[mask] == 0)

    def test_honest_labels_unchanged_for_malicious_client(self):
        X, y = _batch(n=60)
        attacker = _default_attacker(malicious=[2], frac=0.3)
        _, y_p, mask = attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        np.testing.assert_array_equal(y_p[~mask], y[~mask])

    def test_output_shape_unchanged(self):
        X, y = _batch(n=40)
        attacker = _default_attacker(malicious=[2])
        X_p, y_p, mask = attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        assert X_p.shape == X.shape
        assert y_p.shape == y.shape
        assert mask.shape == (len(X),)

    def test_mask_dtype_is_bool(self):
        X, y = _batch()
        attacker = _default_attacker(malicious=[2])
        _, _, mask = attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        assert mask.dtype == bool

    def test_original_X_not_mutated(self):
        X, y = _batch()
        X_orig = X.copy()
        attacker = _default_attacker(malicious=[2])
        attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        np.testing.assert_array_equal(X, X_orig)

    def test_round_num_changes_poisoning(self):
        """Different rounds should produce different poison sets (different seeds)."""
        X, y = _batch(n=60)
        attacker = _default_attacker(malicious=[2], frac=0.3)
        _, _, m0 = attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        _, _, m1 = attacker.poison_client_data(X, y, "client_02", 1, _cfg())
        assert not np.array_equal(m0, m1), "Different rounds must give different seeds"

    def test_same_round_same_client_is_deterministic(self):
        X, y = _batch(n=60)
        attacker = _default_attacker(malicious=[2])
        _, y1, m1 = attacker.poison_client_data(X, y, "client_02", 3, _cfg())
        _, y2, m2 = attacker.poison_client_data(X, y, "client_02", 3, _cfg())
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(y1, y2)

    def test_too_few_samples_skips_poisoning(self):
        X = np.zeros((2, 1, 28, 28), dtype=np.float32)
        y = np.ones(2, dtype=np.int64)
        attacker = _default_attacker(malicious=[2])
        _, _, mask = attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        assert not np.any(mask)

    def test_cifar10_shape_works(self):
        X, y = _batch(n=40, channels=3)
        X = np.zeros((40, 3, 32, 32), dtype=np.float32)
        attacker = _default_attacker(malicious=[2])
        X_p, y_p, mask = attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        assert X_p.shape == X.shape


# ---------------------------------------------------------------------------
# build_trigger_eval_set
# ---------------------------------------------------------------------------


class TestBuildTriggerEvalSet:
    def test_output_shape_unchanged(self):
        X = np.zeros((10, 1, 28, 28), dtype=np.float32)
        attacker = _default_attacker()
        X_t = attacker.build_trigger_eval_set(X)
        assert X_t.shape == X.shape

    def test_trigger_region_modified(self):
        X = np.zeros((5, 1, 28, 28), dtype=np.float32)
        attacker = _default_attacker()
        X_t = attacker.build_trigger_eval_set(X)
        patch = X_t[0, 0, 24:28, 24:28]
        np.testing.assert_allclose(patch, 1.0)

    def test_original_unchanged(self):
        X = np.zeros((5, 1, 28, 28), dtype=np.float32)
        X_orig = X.copy()
        attacker = _default_attacker()
        attacker.build_trigger_eval_set(X)
        np.testing.assert_array_equal(X, X_orig)


# ---------------------------------------------------------------------------
# Round reports
# ---------------------------------------------------------------------------


class TestRoundReports:
    def test_round_report_created_after_poison(self):
        X, y = _batch(n=40)
        attacker = _default_attacker(malicious=[2])
        attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        report = attacker.get_round_report(0)
        assert report is not None
        assert report.round_num == 0

    def test_round_report_has_malicious_client(self):
        X, y = _batch(n=40)
        attacker = _default_attacker(malicious=[2])
        attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        report = attacker.get_round_report(0)
        assert "client_02" in report.malicious_clients

    def test_all_round_reports(self):
        X, y = _batch(n=40)
        attacker = _default_attacker(malicious=[2])
        attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        attacker.poison_client_data(X, y, "client_02", 1, _cfg())
        reports = attacker.all_round_reports()
        assert len(reports) == 2

    def test_attach_eval_result(self):
        X, y = _batch(n=40)
        attacker = _default_attacker(malicious=[2])
        attacker.poison_client_data(X, y, "client_02", 0, _cfg())
        attacker.attach_eval_result(0, asr=0.87, clean_acc=0.93)
        report = attacker.get_round_report(0)
        assert report.asr == pytest.approx(0.87)
        assert report.clean_acc == pytest.approx(0.93)

    def test_get_round_report_nonexistent_returns_none(self):
        attacker = _default_attacker()
        assert attacker.get_round_report(999) is None


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_explicit_malicious_indices(self):
        cfg = MagicMock(
            seed=42,
            n_clients=10,
            attack=MagicMock(
                target_label=3,
                poison_fraction=0.2,
                malicious_client_indices=[1, 4],
                malicious_client_fraction=0.25,
            ),
            trigger=MagicMock(
                shape="cross",
                size=5,
                location="top_left",
                color=0.8,
                opacity=1.0,
                seed=0,
            ),
        )
        attacker = BadNetsImageAttack.from_config(cfg)
        assert attacker.target_label == 3
        assert attacker.malicious_client_indices == {1, 4}

    def test_from_config_uses_fraction_when_list_empty(self):
        cfg = MagicMock(
            seed=42,
            n_clients=10,
            attack=MagicMock(
                target_label=0,
                poison_fraction=0.15,
                malicious_client_indices=[],
                malicious_client_fraction=0.3,
            ),
            trigger=MagicMock(
                shape="square",
                size=4,
                location="bottom_right",
                color=1.0,
                opacity=1.0,
                seed=0,
            ),
        )
        attacker = BadNetsImageAttack.from_config(cfg)
        # 30% of 10 = 3 malicious clients
        assert len(attacker.malicious_client_indices) == 3

    def test_from_config_no_attack_uses_defaults(self):
        cfg = MagicMock(seed=42, attack=None, trigger=None)
        attacker = BadNetsImageAttack.from_config(cfg)
        assert attacker.target_label == 0


# ---------------------------------------------------------------------------
# _parse_client_id helper
# ---------------------------------------------------------------------------


class TestParseClientId:
    def test_standard_format(self):
        assert _parse_client_id("client_02") == 2
        assert _parse_client_id("client_10") == 10

    def test_plain_numeric(self):
        assert _parse_client_id("7") == 7

    def test_fallback_on_unparseable(self):
        result = _parse_client_id("abc_xyz")
        assert isinstance(result, int)
