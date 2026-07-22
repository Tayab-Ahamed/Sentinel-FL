"""
tests/test_asr_evaluator.py — Unit tests for AttackSuccessRateEvaluator.

Uses a tiny stub model so tests run without real training:
  - ``AlwaysTargetModel``: always predicts ``target_label`` → ASR = 1.0
  - ``AlwaysWrongModel``: always predicts ``target_label + 1`` → ASR = 0.0
  - ``PerfectCleanModel``: correct predictions → C-Acc = 1.0
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from ai.attacks.asr_evaluator import AttackSuccessRateEvaluator
from ai.attacks.badnets import BadNetsImageAttack
from ai.attacks.triggers import TriggerFactory

# ---------------------------------------------------------------------------
# Stub models
# ---------------------------------------------------------------------------


class AlwaysTargetModel(nn.Module):
    """Always predicts label ``target``."""

    def __init__(self, target: int = 0, n_classes: int = 5) -> None:
        super().__init__()
        self._target = target
        self._n = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        logits = torch.zeros(n, self._n)
        logits[:, self._target] = 10.0  # argmax → target
        return logits


class AlwaysWrongModel(nn.Module):
    """Always predicts ``(target + 1) % n_classes``."""

    def __init__(self, target: int = 0, n_classes: int = 5) -> None:
        super().__init__()
        self._wrong = (target + 1) % n_classes
        self._n = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        logits = torch.zeros(n, self._n)
        logits[:, self._wrong] = 10.0
        return logits


class PerfectModel(nn.Module):
    """Returns logits that always pick the correct label (from y stored in a buffer)."""

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()
        self._n = n_classes
        self._labels: list[int] = []

    def set_labels(self, y: np.ndarray) -> None:
        self._labels = y.tolist()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        logits = torch.zeros(n, self._n)
        for i in range(n):
            if i < len(self._labels):
                logits[i, self._labels[i]] = 10.0
        return logits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluator() -> AttackSuccessRateEvaluator:
    return AttackSuccessRateEvaluator(device="cpu", batch_size=32)


@pytest.fixture
def attacker() -> BadNetsImageAttack:
    return BadNetsImageAttack(
        target_label=0,
        poison_fraction=0.3,
        malicious_client_indices=[2],
        pattern=TriggerFactory.make_square(size=4),
        seed=42,
    )


def _clean_data(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Balanced 5-class MNIST-shaped eval set."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, 1, 28, 28)).astype(np.float32)
    y = np.tile(np.arange(5), n // 5 + 1)[:n].astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# compute_asr
# ---------------------------------------------------------------------------


class TestComputeASR:
    def test_asr_is_one_when_model_always_predicts_target(self, evaluator):
        X_triggered = np.zeros((20, 1, 28, 28), dtype=np.float32)
        model = AlwaysTargetModel(target=0)
        asr = evaluator.compute_asr(model, X_triggered, target_label=0)
        assert asr == pytest.approx(1.0)

    def test_asr_is_zero_when_model_never_predicts_target(self, evaluator):
        X_triggered = np.zeros((20, 1, 28, 28), dtype=np.float32)
        model = AlwaysWrongModel(target=0)
        asr = evaluator.compute_asr(model, X_triggered, target_label=0)
        assert asr == pytest.approx(0.0)

    def test_asr_in_range(self, evaluator):
        X_triggered = np.zeros((30, 1, 28, 28), dtype=np.float32)
        model = AlwaysTargetModel(target=2)
        asr = evaluator.compute_asr(model, X_triggered, target_label=2)
        assert 0.0 <= asr <= 1.0

    def test_empty_triggered_returns_zero(self, evaluator):
        X_empty = np.zeros((0, 1, 28, 28), dtype=np.float32)
        model = AlwaysTargetModel()
        asr = evaluator.compute_asr(model, X_empty, target_label=0)
        assert asr == pytest.approx(0.0)

    def test_asr_works_with_cifar_shape(self, evaluator):
        X = np.zeros((10, 3, 32, 32), dtype=np.float32)
        model = AlwaysTargetModel(target=1, n_classes=10)
        asr = evaluator.compute_asr(model, X, target_label=1)
        assert asr == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_clean_accuracy
# ---------------------------------------------------------------------------


class TestComputeCleanAccuracy:
    def test_perfect_model_gives_acc_one(self, evaluator):
        X, y = _clean_data(n=30)
        model = PerfectModel(n_classes=5)
        model.set_labels(y)
        acc = evaluator.compute_clean_accuracy(model, X, y)
        assert acc == pytest.approx(1.0)

    def test_always_wrong_model_gives_low_acc(self, evaluator):
        X, y = _clean_data(n=30)
        model = AlwaysWrongModel(target=0, n_classes=5)
        acc = evaluator.compute_clean_accuracy(model, X, y)
        # Only class-1 samples will be correct (wrong predicts 1, some y=1)
        assert 0.0 <= acc <= 1.0

    def test_empty_clean_returns_zero(self, evaluator):
        X_empty = np.zeros((0, 1, 28, 28), dtype=np.float32)
        y_empty = np.zeros(0, dtype=np.int64)
        model = AlwaysTargetModel()
        acc = evaluator.compute_clean_accuracy(model, X_empty, y_empty)
        assert acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_round
# ---------------------------------------------------------------------------


class TestEvaluateRound:
    def test_returns_attack_eval_result(self, evaluator, attacker):
        from ai.attacks.attack_report import AttackEvalResult
        X, y = _clean_data(n=50)
        model = AlwaysTargetModel(target=0, n_classes=5)
        result = evaluator.evaluate_round(
            model=model, attacker=attacker,
            X_clean_eval=X, y_clean_eval=y,
            round_num=0,
        )
        assert isinstance(result, AttackEvalResult)

    def test_result_round_num_matches(self, evaluator, attacker):
        X, y = _clean_data()
        model = AlwaysTargetModel()
        result = evaluator.evaluate_round(model, attacker, X, y, round_num=5)
        assert result.round_num == 5

    def test_result_target_label_matches(self, evaluator, attacker):
        X, y = _clean_data()
        model = AlwaysTargetModel()
        result = evaluator.evaluate_round(model, attacker, X, y, round_num=0)
        assert result.target_label == attacker.target_label

    def test_asr_high_when_model_always_predicts_target(self, evaluator, attacker):
        X, y = _clean_data(n=50)
        model = AlwaysTargetModel(target=0, n_classes=5)
        result = evaluator.evaluate_round(model, attacker, X, y, round_num=0)
        assert result.asr == pytest.approx(1.0)

    def test_clean_acc_returned(self, evaluator, attacker):
        X, y = _clean_data(n=30)
        model = PerfectModel(n_classes=5)
        model.set_labels(y)
        result = evaluator.evaluate_round(model, attacker, X, y, round_num=1)
        assert result.clean_acc == pytest.approx(1.0)

    def test_result_attached_to_round_report(self, evaluator, attacker):
        X, y = _clean_data()
        # First create a round report by poisoning
        X_client, y_client = _clean_data(n=40)
        from unittest.mock import MagicMock
        attacker.poison_client_data(X_client, y_client, "client_02", 0, MagicMock(seed=42))

        model = AlwaysTargetModel()
        evaluator.evaluate_round(model, attacker, X, y, round_num=0)
        report = attacker.get_round_report(0)
        assert report is not None
        assert report.asr is not None

    def test_n_triggered_and_n_clean_are_positive(self, evaluator, attacker):
        X, y = _clean_data(n=50)
        model = AlwaysTargetModel()
        result = evaluator.evaluate_round(model, attacker, X, y, round_num=0)
        assert result.n_triggered > 0
        assert result.n_clean > 0


# ---------------------------------------------------------------------------
# evaluate_all_rounds
# ---------------------------------------------------------------------------


class TestEvaluateAllRounds:
    def test_returns_one_result_per_model(self, evaluator, attacker):
        X, y = _clean_data(n=30)
        models = [(r, AlwaysTargetModel()) for r in range(3)]
        results = evaluator.evaluate_all_rounds(models, attacker, X, y)
        assert len(results) == 3

    def test_round_numbers_match(self, evaluator, attacker):
        X, y = _clean_data(n=30)
        models = [(5, AlwaysTargetModel()), (10, AlwaysTargetModel())]
        results = evaluator.evaluate_all_rounds(models, attacker, X, y)
        assert results[0].round_num == 5
        assert results[1].round_num == 10
