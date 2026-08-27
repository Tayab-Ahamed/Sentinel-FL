"""
tests/test_remediation.py — Tests for the L5 Remediation Engine (ARCHITECTURE.md §7.4).

Dependency-light: uses only NumPy + Pydantic (the Phase 0 stack), so the whole
file runs without PyTorch / scikit-learn.  Covers:

  triggers.py       — vector coercion, masking, stamping
  adapters.py       — LinearSoftmaxAdapter predict / fine_tune / weight surgery
  rollback.py       — target selection + registry integration + failure paths
  unlearning.py     — reinforcement-set construction + ASR reduction
  pruning.py        — channel pruning + backend guard
  remediation_engine— escalation policy, acceptance criteria, ledger audit trail,
                       RemediationFailedError + attached report
  schema            — RemediationReport + Configuration remediation fields
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.fl_core.exceptions import RemediationFailedError
from ai.fl_core.fl_engine import LinearSoftmaxModel, local_train
from ai.fl_core.model_registry import FileModelRegistry
from ai.fl_core.schemas import (
    AuditReport,
    Configuration,
    DetectionResult,
    ModelMetadata,
    RemediationReport,
    ReversedTrigger,
)
from ai.remediation import (
    FinePruner,
    LinearSoftmaxAdapter,
    RemediationEngine,
    RollbackRemediator,
    TriggerUnlearner,
)
from ai.remediation.rollback import RollbackFailed
from ai.remediation.triggers import (
    as_trigger_vector,
    stamp_trigger,
    trigger_from_block,
    trigger_mask,
)

N_FEATURES = 20
N_CLASSES = 4
TARGET = 0
TRIGGER_BLOCK = slice(0, 3)
TRIGGER_VALUE = 6.0


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_data(seed: int = 42, n: int = 800):
    """Gaussian-blob multi-class data (mirrors ai/training/poison.make_dataset)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(N_CLASSES, N_FEATURES))
    y = rng.integers(0, N_CLASSES, size=n)
    X = centers[y] + rng.normal(0, 1.0, size=(n, N_FEATURES))
    return X, y


def _stamp_block(X: np.ndarray) -> np.ndarray:
    X_t = X.copy()
    X_t[:, TRIGGER_BLOCK] = TRIGGER_VALUE
    return X_t


def _train_poisoned_params(seed: int = 0):
    """Train a model whose weights contain a strong trigger->TARGET backdoor."""
    X, y = _make_data(seed=1, n=1200)
    # Heavy poisoning so the backdoor is reliably learned.
    rng = np.random.default_rng(seed)
    n_poison = int(0.4 * len(X))
    idx = rng.choice(len(X), n_poison, replace=False)
    Xp, yp = X.copy(), y.copy()
    Xp[idx, TRIGGER_BLOCK] = TRIGGER_VALUE
    yp[idx] = TARGET
    params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
    params = local_train(params, N_FEATURES, N_CLASSES, Xp, yp, epochs=60, lr=0.3)
    return params


def _reversed_trigger() -> ReversedTrigger:
    vec = trigger_from_block(N_FEATURES, TRIGGER_BLOCK, TRIGGER_VALUE)
    return ReversedTrigger(
        label=TARGET, trigger_representation=vec.tolist(), l1_norm=float(np.abs(vec).sum())
    )


def _audit_report(round_num: int = 10) -> AuditReport:
    det = DetectionResult(
        detector_name="neural_cleanse_audit",
        layer="L2",
        subject_id=str(TARGET),
        score=0.5,
        flagged=True,
        boundary=2.0,
        round_num=round_num,
        explanation="reversed trigger found",
    )
    return AuditReport(
        round_num=round_num,
        per_label_results=[det],
        flagged_labels=[TARGET],
        reversed_triggers=[_reversed_trigger()],
    )


@pytest.fixture
def adapter() -> LinearSoftmaxAdapter:
    return LinearSoftmaxAdapter(N_FEATURES, N_CLASSES)


@pytest.fixture
def clean_holdout():
    return _make_data(seed=7, n=400)


# ---------------------------------------------------------------------------
# triggers.py
# ---------------------------------------------------------------------------


class TestTriggers:
    def test_trigger_from_block_shape_and_values(self):
        vec = trigger_from_block(N_FEATURES, TRIGGER_BLOCK, TRIGGER_VALUE)
        assert vec.shape == (N_FEATURES,)
        assert np.all(vec[TRIGGER_BLOCK] == TRIGGER_VALUE)
        assert np.all(vec[3:] == 0.0)

    def test_as_trigger_vector_from_nested_list(self):
        vec = as_trigger_vector([[1.0, 2.0], [3.0, 4.0]], 4)
        assert vec.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_as_trigger_vector_pads_and_truncates(self):
        assert as_trigger_vector([1, 2], 4).tolist() == [1.0, 2.0, 0.0, 0.0]
        assert as_trigger_vector([1, 2, 3, 4, 5], 3).tolist() == [1.0, 2.0, 3.0]

    def test_trigger_mask(self):
        mask = trigger_mask(np.array([0.0, 6.0, 0.0, 3.0]))
        assert mask.tolist() == [False, True, False, True]

    def test_stamp_trigger_overwrites_masked_channels_only(self):
        X = np.ones((5, N_FEATURES))
        vec = trigger_from_block(N_FEATURES, TRIGGER_BLOCK, TRIGGER_VALUE)
        out = stamp_trigger(X, vec)
        assert np.all(out[:, TRIGGER_BLOCK] == TRIGGER_VALUE)
        assert np.all(out[:, 3:] == 1.0)  # untouched

    def test_stamp_trigger_rejects_1d(self):
        with pytest.raises(ValueError):
            stamp_trigger(np.ones(N_FEATURES), np.ones(N_FEATURES))


# ---------------------------------------------------------------------------
# adapters.py
# ---------------------------------------------------------------------------


class TestLinearSoftmaxAdapter:
    def test_clone_is_deep_copy(self, adapter):
        p = np.ones(5)
        c = adapter.clone(p)
        c[0] = 99
        assert p[0] == 1.0

    def test_predict_shape_and_dtype(self, adapter, clean_holdout):
        X, _y = clean_holdout
        params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
        preds = adapter.predict(params, X)
        assert preds.shape == (len(X),)
        assert preds.dtype == np.int64

    def test_fine_tune_improves_clean_accuracy(self, adapter, clean_holdout):
        X, y = clean_holdout
        params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
        acc_before = float((adapter.predict(params, X) == y).mean())
        trained = adapter.fine_tune(params, X, y, epochs=40, lr=0.3)
        acc_after = float((adapter.predict(trained, X) == y).mean())
        assert acc_after > acc_before

    def test_fine_tune_empty_returns_clone(self, adapter):
        params = np.ones(5)
        out = adapter.fine_tune(params, np.empty((0, N_FEATURES)), np.empty(0), 5, 0.1)
        assert np.allclose(out, params)

    def test_weight_matrix_roundtrip(self, adapter):
        params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
        W = adapter.get_weight_matrix(params)
        assert W.shape == (N_CLASSES, N_FEATURES)
        W[:, 0] = 0.0
        new_params = adapter.set_weight_matrix(params, W)
        assert np.all(adapter.get_weight_matrix(new_params)[:, 0] == 0.0)


class TestSourceOnlyASR:
    """Scientific regression tests for the standard source-only ASR definition."""

    def test_excludes_examples_already_in_target_class(self, adapter):
        engine = RemediationEngine(adapter, strategies=())
        params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
        X = np.zeros((4, N_FEATURES))
        # Zero parameters predict class 0 (TARGET) for every sample. Two samples
        # are already target class and must not inflate the attack denominator.
        y = np.array([TARGET, 1, TARGET, 2])
        assert engine._asr(params, X, TARGET, y) == pytest.approx(1.0)

    def test_all_target_ground_truth_has_no_attack_surface(self, adapter):
        engine = RemediationEngine(adapter, strategies=())
        params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
        X = np.zeros((3, N_FEATURES))
        assert engine._asr(params, X, TARGET, np.full(3, TARGET)) == 0.0

    def test_label_length_mismatch_rejected(self, adapter):
        engine = RemediationEngine(adapter, strategies=())
        params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
        with pytest.raises(ValueError, match="length must match"):
            engine._asr(params, np.zeros((3, N_FEATURES)), TARGET, np.zeros(2))


# ---------------------------------------------------------------------------
# rollback.py
# ---------------------------------------------------------------------------


class TestRollback:
    def _registry_with_rounds(self, tmp_path, rounds):
        reg = FileModelRegistry(tmp_path / "ckpts")
        for r in rounds:
            params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params() + r
            reg.save(r, params, ModelMetadata(round_num=r, architecture="linear_softmax_v0"))
        return reg

    def test_available_rounds(self, tmp_path):
        reg = self._registry_with_rounds(tmp_path, [0, 5, 10])
        rem = RollbackRemediator(reg)
        assert rem.available_rounds() == [0, 5, 10]

    def test_select_target_strictly_before(self, tmp_path):
        reg = self._registry_with_rounds(tmp_path, [0, 5, 10])
        rem = RollbackRemediator(reg)
        assert rem.select_target_round(10) == 5
        assert rem.select_target_round(6) == 5

    def test_select_target_none_returns_earliest(self, tmp_path):
        reg = self._registry_with_rounds(tmp_path, [0, 5, 10])
        assert RollbackRemediator(reg).select_target_round(None) == 0

    def test_no_older_checkpoint_raises(self, tmp_path):
        reg = self._registry_with_rounds(tmp_path, [5, 10])
        with pytest.raises(RollbackFailed):
            RollbackRemediator(reg).select_target_round(5)

    def test_empty_registry_raises(self, tmp_path):
        reg = FileModelRegistry(tmp_path / "empty")
        with pytest.raises(RollbackFailed):
            RollbackRemediator(reg).select_target_round(10)

    def test_remediate_returns_state_and_id(self, tmp_path):
        reg = self._registry_with_rounds(tmp_path, [0, 5, 10])
        state, target_round, model_id = RollbackRemediator(reg).remediate(10)
        assert target_round == 5
        assert isinstance(model_id, str)
        assert state.shape == (N_CLASSES * N_FEATURES + N_CLASSES,)


# ---------------------------------------------------------------------------
# unlearning.py
# ---------------------------------------------------------------------------


class TestUnlearning:
    def test_reinforcement_set_size(self, adapter, clean_holdout):
        X, y = clean_holdout
        unl = TriggerUnlearner(adapter, stamped_replicas=2)
        vec = trigger_from_block(N_FEATURES, TRIGGER_BLOCK, TRIGGER_VALUE)
        Xr, yr = unl.build_reinforcement_set(X, y, [vec])
        # clean + 2 stamped replicas = 3x
        assert len(Xr) == 3 * len(X)
        assert len(yr) == 3 * len(y)

    def test_unlearning_reduces_asr(self, adapter, clean_holdout):
        X, y = clean_holdout
        params = _train_poisoned_params()
        X_trig = _stamp_block(X)
        asr_before = float((adapter.predict(params, X_trig) == TARGET).mean())
        unl = TriggerUnlearner(adapter, epochs=30, lr=0.2, stamped_replicas=3)
        repaired = unl.remediate(params, X, y, [_reversed_trigger()], N_FEATURES)
        asr_after = float((adapter.predict(repaired, X_trig) == TARGET).mean())
        assert asr_before > 0.5  # sanity: backdoor was learned
        assert asr_after < asr_before

    def test_unlearning_without_triggers_runs(self, adapter, clean_holdout):
        X, y = clean_holdout
        params = _train_poisoned_params()
        repaired = TriggerUnlearner(adapter, epochs=5).remediate(params, X, y, [], N_FEATURES)
        assert repaired.shape == params.shape


# ---------------------------------------------------------------------------
# pruning.py
# ---------------------------------------------------------------------------


class TestPruning:
    def test_pruning_zeros_trigger_channels_then_finetunes(self, adapter, clean_holdout):
        X, y = clean_holdout
        params = _train_poisoned_params()
        pruner = FinePruner(adapter, finetune_epochs=0)  # no recovery -> weights stay zeroed
        repaired = pruner.remediate(params, X, y, [_reversed_trigger()], N_FEATURES)
        W = adapter.get_weight_matrix(repaired)
        assert np.allclose(W[:, TRIGGER_BLOCK], 0.0)

    def test_pruning_reduces_asr(self, adapter, clean_holdout):
        X, y = clean_holdout
        params = _train_poisoned_params()
        X_trig = _stamp_block(X)
        asr_before = float((adapter.predict(params, X_trig) == TARGET).mean())
        repaired = FinePruner(adapter, finetune_epochs=10, finetune_lr=0.2).remediate(
            params, X, y, [_reversed_trigger()], N_FEATURES
        )
        asr_after = float((adapter.predict(repaired, X_trig) == TARGET).mean())
        assert asr_after < asr_before

    def test_pruning_respects_max_fraction(self, adapter, clean_holdout):
        X, y = clean_holdout
        params = _train_poisoned_params()
        # A diffuse "trigger" over all features, capped at 10%.
        full = ReversedTrigger(
            label=TARGET,
            trigger_representation=np.ones(N_FEATURES).tolist(),
            l1_norm=float(N_FEATURES),
        )
        pruner = FinePruner(adapter, finetune_epochs=0, max_prune_fraction=0.1)
        repaired = pruner.remediate(params, X, y, [full], N_FEATURES)
        W = adapter.get_weight_matrix(repaired)
        n_zeroed = int(np.all(W == 0.0, axis=0).sum())
        assert n_zeroed <= int(0.1 * N_FEATURES) + 1

    def test_pruning_rejects_non_linear_adapter(self, clean_holdout):
        class Dummy:
            architecture = "cnn"

        with pytest.raises(TypeError):
            FinePruner(Dummy()).remediate(  # type: ignore[arg-type]
                np.ones(5), *clean_holdout, [_reversed_trigger()], N_FEATURES
            )


# ---------------------------------------------------------------------------
# remediation_engine.py
# ---------------------------------------------------------------------------


class _RecordingLedger:
    def __init__(self):
        self.entries = []

    def add_entry(self, entry):
        self.entries.append(entry)


class TestRemediationEngine:
    def _registry_with_clean_checkpoint(self, tmp_path, clean_params):
        reg = FileModelRegistry(tmp_path / "ckpts")
        reg.save(0, clean_params, ModelMetadata(round_num=0, architecture="linear_softmax_v0"))
        reg.save(5, clean_params, ModelMetadata(round_num=5, architecture="linear_softmax_v0"))
        return reg

    def test_rollback_path_wins_first(self, adapter, clean_holdout, tmp_path):
        X, y = clean_holdout
        clean_params = local_train(
            LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params(),
            N_FEATURES,
            N_CLASSES,
            X,
            y,
            epochs=40,
            lr=0.3,
        )
        reg = self._registry_with_clean_checkpoint(tmp_path, clean_params)
        poisoned = _train_poisoned_params()
        ledger = _RecordingLedger()
        engine = RemediationEngine(adapter, registry=reg, ledger=ledger, asr_threshold=0.3)
        _remediated, report = engine.remediate(
            poisoned,
            _audit_report(round_num=10),
            X,
            y,
            _stamp_block(X),
            TARGET,
            suspected_infection_round=6,
        )
        assert report.success is True
        assert report.strategy_succeeded == "rollback"
        assert report.asr_after <= 0.3
        assert report.rolled_back_model_id is not None
        assert len(ledger.entries) == 1

    def test_unlearning_path_when_no_registry(self, adapter, clean_holdout):
        X, y = clean_holdout
        poisoned = _train_poisoned_params()
        engine = RemediationEngine(
            adapter,
            registry=None,
            asr_threshold=0.3,
            strategies=("rollback", "unlearning", "pruning"),
            unlearning_epochs=40,
            unlearning_lr=0.25,
        )
        _remediated, report = engine.remediate(
            poisoned,
            _audit_report(),
            X,
            y,
            _stamp_block(X),
            TARGET,
        )
        assert report.success is True
        # rollback is skipped (no registry) -> unlearning or pruning succeeds
        assert report.strategy_succeeded in {"unlearning", "pruning"}
        assert report.asr_after < report.asr_before

    def test_failure_raises_with_report(self, adapter, clean_holdout):
        X, y = clean_holdout
        poisoned = _train_poisoned_params()
        # Impossible threshold + only rollback (no registry) => guaranteed failure.
        engine = RemediationEngine(
            adapter,
            registry=None,
            asr_threshold=0.0,
            strategies=("rollback",),
        )
        with pytest.raises(RemediationFailedError) as excinfo:
            engine.remediate(poisoned, _audit_report(), X, y, _stamp_block(X), TARGET)
        assert hasattr(excinfo.value, "report")
        assert excinfo.value.report.manual_review_required is True
        assert excinfo.value.report.success is False

    def test_failure_no_raise_returns_report(self, adapter, clean_holdout):
        X, y = clean_holdout
        poisoned = _train_poisoned_params()
        engine = RemediationEngine(
            adapter,
            registry=None,
            asr_threshold=0.0,
            strategies=("rollback",),
        )
        params, report = engine.remediate(
            poisoned,
            _audit_report(),
            X,
            y,
            _stamp_block(X),
            TARGET,
            raise_on_failure=False,
        )
        assert report.manual_review_required is True
        assert params.shape == poisoned.shape

    def test_from_config(self, adapter):
        cfg = Configuration()
        engine = RemediationEngine.from_config(adapter, cfg)
        assert engine._asr_threshold == cfg.remediation_asr_threshold
        assert engine._strategies == tuple(cfg.remediation_strategies)

    def test_ledger_never_gates_control_flow(self, adapter, clean_holdout):
        class BrokenLedger:
            def add_entry(self, entry):
                raise OSError("disk full")

        X, y = clean_holdout
        poisoned = _train_poisoned_params()
        engine = RemediationEngine(
            adapter,
            ledger=BrokenLedger(),
            asr_threshold=0.3,
            strategies=("unlearning",),
            unlearning_epochs=40,
            unlearning_lr=0.25,
        )
        _, report = engine.remediate(poisoned, _audit_report(), X, y, _stamp_block(X), TARGET)
        assert report.success is True  # broken ledger did not break remediation


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


class TestRemediationSchema:
    def test_report_roundtrip(self):
        r = RemediationReport(
            round_num=10,
            asr_before=0.9,
            asr_after=0.05,
            clean_accuracy_before=0.8,
            clean_accuracy_after=0.79,
            asr_threshold=0.2,
            success=True,
            manual_review_required=False,
        )
        assert RemediationReport.model_validate_json(r.model_dump_json()).success is True

    def test_report_rejects_out_of_range_asr(self):
        with pytest.raises(Exception):
            RemediationReport(
                round_num=1,
                asr_before=1.5,
                asr_after=0.0,
                clean_accuracy_before=0.5,
                clean_accuracy_after=0.5,
                asr_threshold=0.2,
                success=False,
                manual_review_required=True,
            )

    def test_configuration_remediation_defaults(self):
        cfg = Configuration()
        assert cfg.remediation_enabled is True
        assert cfg.remediation_strategies == ["rollback", "unlearning", "pruning"]
        assert 0.0 <= cfg.remediation_asr_threshold <= 1.0
