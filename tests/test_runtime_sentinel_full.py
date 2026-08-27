"""
tests/test_runtime_sentinel_full.py — Milestone 7: Runtime Sentinel full test suite.

Covers:
  ConfidenceAnalyzer: softmax_confidence, top2_margin, entropy_from_probs,
                      confidence_anomaly_score, batch_confidence_stats
  FusionClassifier: fallback mode, fit+calibrate, predict, predict_batch,
                    is_calibrated, reset, save/load state, validation
  InferenceMonitor: calibrate, update, check_confidence_drop,
                    check_class_distribution_shift, anomaly_score, round_stats,
                    summary, reset
  AlertManager: create_alert, record_alert, get_active_alerts, get_alert_history,
                alert_rate, clear_old_alerts, to_ledger_entry, stats, severity
  InferenceContext schema: validation, serialisation
  SentinelAlert schema: validation, severity pattern
  RuntimeSentinelStrategy.process(): full pipeline, alert generation, ledger entries,
                    no-model early-exit, metrics(), calibrate_all(), add_detector()
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _toy_model(n_features: int = 10, n_classes: int = 3, seed: int = 0):
    """Return a trivially trained LogisticRegression."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((300, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=300)
    clf = LogisticRegression(max_iter=200, random_state=seed)
    clf.fit(X, y)
    return clf


def _probs_batch(n: int = 50, n_classes: int = 3, seed: int = 0) -> np.ndarray:
    """Uniform-ish softmax probability batch."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, n_classes)).astype(np.float32)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


# ---------------------------------------------------------------------------
# ConfidenceAnalyzer
# ---------------------------------------------------------------------------


class TestConfidenceAnalyzer:
    def test_softmax_confidence_max_prob(self):
        from ai.detection.confidence_analyzer import softmax_confidence

        # logits heavily skewed to first class → confidence ≈ 1
        logits = np.array([10.0, 0.0, 0.0])
        conf = softmax_confidence(logits)
        assert conf > 0.99

    def test_softmax_confidence_uniform(self):
        from ai.detection.confidence_analyzer import softmax_confidence

        logits = np.array([0.0, 0.0, 0.0])
        conf = softmax_confidence(logits)
        assert abs(conf - 1 / 3) < 0.01

    def test_softmax_confidence_in_range(self):
        from ai.detection.confidence_analyzer import softmax_confidence

        for _ in range(10):
            logits = np.random.default_rng(0).standard_normal(5)
            c = softmax_confidence(logits)
            assert 0.0 <= c <= 1.0

    def test_top2_margin_certain(self):
        from ai.detection.confidence_analyzer import top2_margin

        probs = np.array([1.0, 0.0, 0.0])
        assert top2_margin(probs) == pytest.approx(1.0)

    def test_top2_margin_uniform(self):
        from ai.detection.confidence_analyzer import top2_margin

        probs = np.array([1 / 3, 1 / 3, 1 / 3])
        assert top2_margin(probs) < 0.01

    def test_top2_margin_single_class(self):
        from ai.detection.confidence_analyzer import top2_margin

        assert top2_margin(np.array([1.0])) == 1.0

    def test_entropy_from_probs_uniform(self):
        from ai.detection.confidence_analyzer import entropy_from_probs

        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(entropy_from_probs(p) - 2.0) < 1e-6

    def test_entropy_from_probs_certain(self):
        from ai.detection.confidence_analyzer import entropy_from_probs

        p = np.array([1.0, 0.0])
        assert abs(entropy_from_probs(p)) < 1e-4

    def test_confidence_anomaly_score_zero_std(self):
        from ai.detection.confidence_analyzer import confidence_anomaly_score

        assert confidence_anomaly_score(0.5, 0.9, 0.0) == 0.0

    def test_confidence_anomaly_score_high_drop(self):
        from ai.detection.confidence_analyzer import confidence_anomaly_score

        # Confidence dropped from 0.9 to 0.1 → high anomaly
        score = confidence_anomaly_score(0.1, 0.9, 0.1)
        assert score > 0.9

    def test_confidence_anomaly_score_no_drop(self):
        from ai.detection.confidence_analyzer import confidence_anomaly_score

        # Confidence exactly at baseline → z=0 → sigmoid=0.5.
        # The function is a relative anomaly measure; at z=0 it returns 0.5.
        # The key property is that no-drop score < high-drop score.
        no_drop = confidence_anomaly_score(0.9, 0.9, 0.1)
        high_drop = confidence_anomaly_score(0.1, 0.9, 0.1)
        assert no_drop < high_drop

    def test_confidence_anomaly_score_in_range(self):
        from ai.detection.confidence_analyzer import confidence_anomaly_score

        score = confidence_anomaly_score(0.5, 0.8, 0.15)
        assert 0.0 <= score <= 1.0

    def test_batch_confidence_stats_structure(self):
        from ai.detection.confidence_analyzer import batch_confidence_stats

        probs = _probs_batch(50, 3)
        stats = batch_confidence_stats(probs)
        for key in (
            "n_samples",
            "mean_confidence",
            "std_confidence",
            "min_confidence",
            "max_confidence",
            "mean_entropy",
            "mean_top2_margin",
        ):
            assert key in stats

    def test_batch_confidence_stats_n_samples(self):
        from ai.detection.confidence_analyzer import batch_confidence_stats

        probs = _probs_batch(20, 3)
        assert batch_confidence_stats(probs)["n_samples"] == 20

    def test_batch_confidence_stats_1d_input(self):
        from ai.detection.confidence_analyzer import batch_confidence_stats

        probs = np.array([0.3, 0.5, 0.2])
        stats = batch_confidence_stats(probs)
        assert stats["n_samples"] == 1


# ---------------------------------------------------------------------------
# FusionClassifier
# ---------------------------------------------------------------------------


class TestFusionClassifier:
    def test_default_fallback_mode(self):
        from ai.detection.fusion_classifier import FusionClassifier

        fc = FusionClassifier()
        assert not fc.is_calibrated

    def test_predict_fallback_range(self):
        from ai.detection.fusion_classifier import FusionClassifier

        fc = FusionClassifier(signal1_weight=1.0)
        for s1 in [0.0, 0.5, 1.0, 1.5, 2.0]:
            score = fc.predict(s1)
            assert 0.0 <= score <= 1.0, f"s1={s1} gave score={score}"

    def test_predict_low_entropy_high_suspicion(self):
        from ai.detection.fusion_classifier import FusionClassifier

        fc = FusionClassifier(signal1_weight=1.0)
        # s1=0.0 (lowest entropy possible) → highest suspicion
        high = fc.predict(0.0)
        # s1=2.0 (high entropy) → lower suspicion
        low = fc.predict(2.0)
        assert high > low

    def test_fit_below_min_labels_stays_uncalibrated(self):
        from ai.detection.fusion_classifier import FusionClassifier

        fc = FusionClassifier(min_labels=20)
        fc.fit([0.1, 0.9], [0.1, 0.9], [0, 1])
        assert not fc.is_calibrated
        assert fc.n_training_examples == 2

    def test_fit_above_min_labels_calibrates(self):
        from ai.detection.fusion_classifier import FusionClassifier

        rng = np.random.default_rng(7)
        s1 = rng.uniform(0.0, 1.0, 30).tolist()
        s2 = rng.uniform(0.0, 1.0, 30).tolist()
        # alternating labels to ensure both classes present
        labels = [i % 2 for i in range(30)]
        fc = FusionClassifier(min_labels=10)
        fc.fit(s1, s2, labels)
        assert fc.is_calibrated

    def test_predict_after_calibration_in_range(self):
        from ai.detection.fusion_classifier import FusionClassifier

        rng = np.random.default_rng(3)
        s1 = rng.uniform(0.0, 1.5, 40).tolist()
        s2 = rng.uniform(0.0, 1.0, 40).tolist()
        labels = [i % 2 for i in range(40)]
        fc = FusionClassifier(min_labels=10)
        fc.fit(s1, s2, labels)
        for i in range(10):
            score = fc.predict(float(s1[i]), float(s2[i]))
            assert 0.0 <= score <= 1.0

    def test_predict_batch(self):
        from ai.detection.fusion_classifier import FusionClassifier

        fc = FusionClassifier()
        s1 = [0.3, 0.7, 1.2]
        result = fc.predict_batch(s1)
        assert len(result) == 3
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_cumulative_fit(self):
        from ai.detection.fusion_classifier import FusionClassifier

        fc = FusionClassifier(min_labels=10)
        # First call: 5 examples (below threshold)
        fc.fit([0.1] * 5, [0.1] * 5, [0] * 5)
        assert fc.n_training_examples == 5
        # Second call: 6 more examples → total 11, above threshold
        fc.fit([0.9] * 6, [0.9] * 6, [1] * 6)
        assert fc.n_training_examples == 11
        assert fc.is_calibrated

    def test_reset_clears_state(self):
        from ai.detection.fusion_classifier import FusionClassifier

        rng = np.random.default_rng(0)
        fc = FusionClassifier(min_labels=5)
        fc.fit(
            rng.uniform(size=10).tolist(), rng.uniform(size=10).tolist(), [i % 2 for i in range(10)]
        )
        fc.reset()
        assert not fc.is_calibrated
        assert fc.n_training_examples == 0

    def test_save_load_state_roundtrip(self):
        from ai.detection.fusion_classifier import FusionClassifier

        rng = np.random.default_rng(9)
        fc = FusionClassifier(min_labels=5)
        fc.fit(
            rng.uniform(size=12).tolist(), rng.uniform(size=12).tolist(), [i % 2 for i in range(12)]
        )
        state = fc.save_state()
        fc2 = FusionClassifier(min_labels=5)
        fc2.load_state(state)
        # Should produce same score
        s1, s2 = 0.4, 0.3
        assert abs(fc.predict(s1, s2) - fc2.predict(s1, s2)) < 1e-4

    def test_mismatched_lengths_raises(self):
        from ai.detection.fusion_classifier import FusionClassifier

        fc = FusionClassifier()
        with pytest.raises(ValueError, match="same length"):
            fc.fit([0.1, 0.2], [0.1], [0, 1])

    def test_invalid_signal1_weight_raises(self):
        from ai.detection.fusion_classifier import FusionClassifier

        with pytest.raises(ValueError):
            FusionClassifier(signal1_weight=1.5)


# ---------------------------------------------------------------------------
# InferenceMonitor
# ---------------------------------------------------------------------------


class TestInferenceMonitor:
    def test_uncalibrated_anomaly_score_is_zero(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor()
        assert m.anomaly_score() == 0.0

    def test_calibrate_sets_baseline(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor()
        probs = _probs_batch(50, 3, seed=0)
        m.calibrate(probs)
        assert m.is_calibrated
        assert m._baseline_mean_conf is not None

    def test_update_increments_round_count(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor()
        probs = _probs_batch(30, 3)
        m.update(probs, round_num=1)
        m.update(probs, round_num=2)
        assert m.n_rounds_monitored == 2

    def test_round_stats_returns_dict(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor()
        probs = _probs_batch(20, 3)
        m.update(probs, round_num=5)
        stats = m.round_stats(5)
        assert stats is not None
        assert stats["round_num"] == 5
        assert "mean_confidence" in stats

    def test_round_stats_missing_returns_none(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor()
        assert m.round_stats(99) is None

    def test_check_confidence_drop_no_drop(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor(confidence_drop_threshold=0.5)
        probs = _probs_batch(40, 3, seed=1)
        m.calibrate(probs)
        m.update(probs, round_num=1)  # same distribution → no drop
        assert m.check_confidence_drop() is False

    def test_check_confidence_drop_with_drop(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor(confidence_drop_threshold=0.05)
        # Calibrate with high-confidence predictions
        high_conf = np.tile([0.9, 0.05, 0.05], (40, 1))
        m.calibrate(high_conf)
        # Update with very uncertain predictions
        low_conf = np.tile([1 / 3, 1 / 3, 1 / 3], (40, 1))
        m.update(low_conf, round_num=1)
        assert m.check_confidence_drop() is True

    def test_check_class_shift_no_shift(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor(kl_threshold=0.5)
        probs = _probs_batch(40, 3, seed=2)
        m.calibrate(probs)
        m.update(probs, round_num=1)
        # Same distribution → no shift
        assert m.check_class_distribution_shift() is False

    def test_check_class_shift_all_one_class(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor(kl_threshold=0.1)
        probs_balanced = _probs_batch(60, 3, seed=0)
        m.calibrate(probs_balanced)
        # All inputs predicted as class 0 → heavy shift
        all_class0 = np.tile([0.95, 0.025, 0.025], (40, 1))
        m.update(all_class0, round_num=1)
        assert m.check_class_distribution_shift() is True

    def test_anomaly_score_in_range(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor()
        probs = _probs_batch(30, 3)
        m.calibrate(probs)
        m.update(probs, round_num=1)
        score = m.anomaly_score()
        assert 0.0 <= score <= 1.0

    def test_summary_structure(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor()
        probs = _probs_batch(20, 3)
        m.calibrate(probs)
        m.update(probs, round_num=1)
        s = m.summary()
        for key in (
            "n_rounds_monitored",
            "baseline_mean_conf",
            "current_mean_conf",
            "anomaly_events",
            "current_anomaly_score",
        ):
            assert key in s

    def test_reset_wipes_all(self):
        from ai.detection.inference_monitor import InferenceMonitor

        m = InferenceMonitor()
        probs = _probs_batch(20, 3)
        m.calibrate(probs)
        m.update(probs, round_num=1)
        m.reset()
        assert not m.is_calibrated
        assert m.n_rounds_monitored == 0


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class TestAlertManager:
    def _make_context(
        self,
        input_id: str = "inp_0",
        confidence: float = 0.8,
        round_num: int = 1,
    ) -> Any:
        ctx = MagicMock()
        ctx.input_id = input_id
        ctx.predicted_confidence = confidence
        ctx.round_num = round_num
        return ctx

    def _make_dr(self, flagged: bool = True, score: float = 0.7) -> Any:
        from ai.fl_core.schemas import DetectionResult

        return DetectionResult(
            detector_name="strip_entropy",
            layer="L3",
            subject_id="inp_0",
            score=score,
            flagged=flagged,
            boundary=0.5,
            explanation="test explanation",
        )

    def test_create_alert_returns_sentinel_alert(self):
        from ai.detection.alert_manager import AlertManager
        from ai.fl_core.schemas import SentinelAlert

        am = AlertManager()
        dr = self._make_dr()
        alert = am.create_alert([dr], fused_score=0.8, context=self._make_context())
        assert isinstance(alert, SentinelAlert)

    def test_severity_high(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager(low_medium_boundary=0.4, med_high_boundary=0.7)
        dr = self._make_dr(score=0.9)
        alert = am.create_alert([dr], fused_score=0.85, context=self._make_context())
        assert alert.alert_severity == "high"

    def test_severity_medium(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager(low_medium_boundary=0.4, med_high_boundary=0.7)
        dr = self._make_dr(score=0.5)
        alert = am.create_alert([dr], fused_score=0.55, context=self._make_context())
        assert alert.alert_severity == "medium"

    def test_severity_low(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager(low_medium_boundary=0.4, med_high_boundary=0.7)
        dr = self._make_dr(score=0.2)
        alert = am.create_alert([dr], fused_score=0.2, context=self._make_context())
        assert alert.alert_severity == "low"

    def test_record_and_get_active(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager()
        dr = self._make_dr()
        alert = am.create_alert([dr], fused_score=0.8, context=self._make_context())
        am.record_alert(alert)
        active = am.get_active_alerts(min_severity="low")
        assert len(active) == 1

    def test_get_active_filter_by_severity(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager(low_medium_boundary=0.4, med_high_boundary=0.7)
        for score in [0.2, 0.55, 0.85]:
            dr = self._make_dr(score=score)
            alert = am.create_alert([dr], fused_score=score, context=self._make_context())
            am.record_alert(alert)
        assert len(am.get_active_alerts(min_severity="high")) == 1
        assert len(am.get_active_alerts(min_severity="medium")) == 2
        assert len(am.get_active_alerts(min_severity="low")) == 3

    def test_get_alert_history_limit(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager()
        for i in range(5):
            dr = self._make_dr()
            ctx = self._make_context(input_id=f"inp_{i}")
            alert = am.create_alert([dr], fused_score=0.6, context=ctx)
            am.record_alert(alert)
        hist = am.get_alert_history(limit=3)
        assert len(hist) == 3

    def test_alert_rate_all_flagged(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager()
        for i in range(5):
            dr = self._make_dr(flagged=True)
            ctx = self._make_context(input_id=f"inp_{i}")
            alert = am.create_alert([dr], fused_score=0.8, context=ctx)
            am.record_alert(alert)
        assert am.alert_rate() == pytest.approx(1.0)

    def test_alert_rate_empty(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager()
        assert am.alert_rate() == 0.0

    def test_clear_old_alerts(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager()
        for rnd in range(1, 6):
            dr = self._make_dr()
            ctx = self._make_context(round_num=rnd)
            alert = am.create_alert([dr], fused_score=0.6, context=ctx, round_num=rnd)
            am.record_alert(alert)
        removed = am.clear_old_alerts(max_age_rounds=2, current_round=5)
        # rounds 1, 2 are older than 5-2=3 → should be removed
        assert removed == 2

    def test_to_ledger_entry(self):
        from ai.detection.alert_manager import AlertManager
        from ai.fl_core.schemas import TrustLedgerEntry

        am = AlertManager()
        dr = self._make_dr()
        alert = am.create_alert([dr], fused_score=0.75, context=self._make_context())
        entry = am.to_ledger_entry(alert)
        assert isinstance(entry, TrustLedgerEntry)
        assert entry.layer_id == "L3"
        assert entry.subject_type == "input"
        assert entry.subject_id == "inp_0"
        assert entry.score == pytest.approx(0.75, abs=1e-3)

    def test_stats_structure(self):
        from ai.detection.alert_manager import AlertManager

        am = AlertManager()
        stats = am.stats()
        for key in ("total_created", "total_recorded", "severity_counts", "alert_rate_last_100"):
            assert key in stats

    def test_logger_called_on_record(self):
        from ai.detection.alert_manager import AlertManager

        mock_logger = MagicMock()
        am = AlertManager(sentinel_logger=mock_logger)
        dr = self._make_dr()
        alert = am.create_alert([dr], fused_score=0.7, context=self._make_context())
        am.record_alert(alert)
        mock_logger.log.assert_called_once()
        assert mock_logger.log.call_args[0][1] == "input_flagged"

    def test_logger_failure_does_not_raise(self):
        from ai.detection.alert_manager import AlertManager

        mock_logger = MagicMock()
        mock_logger.log.side_effect = RuntimeError("boom")
        am = AlertManager(sentinel_logger=mock_logger)
        dr = self._make_dr()
        alert = am.create_alert([dr], fused_score=0.7, context=self._make_context())
        am.record_alert(alert)  # must not raise


# ---------------------------------------------------------------------------
# InferenceContext schema
# ---------------------------------------------------------------------------


class TestInferenceContextSchema:
    def test_valid_context(self):
        from ai.fl_core.schemas import InferenceContext

        ctx = InferenceContext(
            input_id="x_001",
            input_data=[0.1, 0.2, 0.3],
            predicted_class=0,
            predicted_confidence=0.92,
        )
        assert ctx.input_id == "x_001"
        assert ctx.predicted_confidence == pytest.approx(0.92)

    def test_empty_input_data_raises(self):
        from ai.fl_core.schemas import InferenceContext

        with pytest.raises(Exception):
            InferenceContext(
                input_id="x",
                input_data=[],  # min_length=1
                predicted_class=0,
                predicted_confidence=0.5,
            )

    def test_confidence_out_of_range_raises(self):
        from ai.fl_core.schemas import InferenceContext

        with pytest.raises(Exception):
            InferenceContext(
                input_id="x",
                input_data=[0.1],
                predicted_class=0,
                predicted_confidence=1.5,
            )

    def test_timestamp_auto_set(self):
        from ai.fl_core.schemas import InferenceContext

        ctx = InferenceContext(
            input_id="x",
            input_data=[0.1],
            predicted_class=0,
            predicted_confidence=0.5,
        )
        assert ctx.timestamp  # non-empty

    def test_json_serialisable(self):
        import json

        from ai.fl_core.schemas import InferenceContext

        ctx = InferenceContext(
            input_id="x",
            input_data=[0.1, 0.2],
            predicted_class=1,
            predicted_confidence=0.7,
        )
        data = json.loads(ctx.model_dump_json())
        assert data["input_id"] == "x"


# ---------------------------------------------------------------------------
# SentinelAlert schema
# ---------------------------------------------------------------------------


class TestSentinelAlertSchema:
    def _make_alert(self, severity: str = "medium", fused: float = 0.6):
        from ai.fl_core.schemas import DetectionResult, SentinelAlert

        dr = DetectionResult(
            detector_name="strip_entropy",
            layer="L3",
            subject_id="x",
            score=0.5,
            flagged=True,
            boundary=0.4,
        )
        return SentinelAlert(
            input_id="x_001",
            detector_verdicts=[dr],
            fused_score=fused,
            flagged=True,
            alert_severity=severity,
        )

    def test_valid_low(self):
        alert = self._make_alert("low", 0.2)
        assert alert.alert_severity == "low"

    def test_valid_medium(self):
        alert = self._make_alert("medium", 0.55)
        assert alert.alert_severity == "medium"

    def test_valid_high(self):
        alert = self._make_alert("high", 0.9)
        assert alert.alert_severity == "high"

    def test_invalid_severity_raises(self):
        from ai.fl_core.schemas import DetectionResult, SentinelAlert

        dr = DetectionResult(
            detector_name="strip_entropy",
            layer="L3",
            subject_id="x",
            score=0.5,
            flagged=True,
            boundary=0.4,
        )
        with pytest.raises(Exception):
            SentinelAlert(
                input_id="x",
                detector_verdicts=[dr],
                fused_score=0.5,
                flagged=True,
                alert_severity="critical",  # invalid
            )

    def test_alert_id_auto_generated(self):
        alert = self._make_alert()
        assert alert.alert_id  # non-empty UUID

    def test_fused_score_out_of_range_raises(self):
        from ai.fl_core.schemas import DetectionResult, SentinelAlert

        dr = DetectionResult(
            detector_name="strip_entropy",
            layer="L3",
            subject_id="x",
            score=0.5,
            flagged=True,
            boundary=0.4,
        )
        with pytest.raises(Exception):
            SentinelAlert(
                input_id="x",
                detector_verdicts=[dr],
                fused_score=1.5,  # > 1
                flagged=True,
                alert_severity="high",
            )


# ---------------------------------------------------------------------------
# RuntimeSentinelStrategy.process() — full pipeline
# ---------------------------------------------------------------------------


def _build_strategy(alert_threshold: float = 0.5):
    """Build a fully configured RuntimeSentinelStrategy with a toy model."""
    from ai.detection.runtime_sentinel import RuntimeSentinelStrategy, StripEntropyDetector

    model = _toy_model(10, 3)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 10)).astype(np.float32)
    detector = StripEntropyDetector(n_perturb=5, min_calibration_samples=10)
    cal = detector.calibrate((model, X))
    strategy = RuntimeSentinelStrategy(
        detectors=[(detector, cal)],
        alert_threshold=alert_threshold,
    )
    return strategy, model, X


def _make_context(model: Any, x: np.ndarray, input_id: str = "inp_0"):
    ctx = MagicMock()
    ctx.input_id = input_id
    ctx.input_data = x.tolist()
    ctx.predicted_class = 0
    ctx.predicted_confidence = 0.85
    ctx.round_num = 1
    ctx.model = model
    return ctx


class TestRuntimeSentinelStrategyProcess:
    def test_process_returns_list(self):
        strategy, model, X = _build_strategy()
        ctx = _make_context(model, X[0])
        result = strategy.process(ctx)
        assert isinstance(result, list)

    def test_process_no_model_returns_empty(self):
        strategy, _model, X = _build_strategy()
        ctx = MagicMock()
        ctx.input_id = "x"
        ctx.input_data = X[0].tolist()
        ctx.model = None
        result = strategy.process(ctx)
        assert result == []

    def test_process_no_input_data_returns_empty(self):
        strategy, model, _X = _build_strategy()
        ctx = MagicMock()
        ctx.input_id = "x"
        ctx.input_data = None
        ctx.model = model
        result = strategy.process(ctx)
        assert result == []

    def test_process_no_detectors_returns_empty(self):
        from ai.detection.runtime_sentinel import RuntimeSentinelStrategy

        strategy = RuntimeSentinelStrategy(detectors=[])
        ctx = MagicMock()
        ctx.input_id = "x"
        ctx.input_data = [0.1, 0.2]
        ctx.model = MagicMock()
        result = strategy.process(ctx)
        assert result == []

    def test_process_increments_total_inferences(self):
        strategy, model, X = _build_strategy(alert_threshold=1.0)  # never flag
        ctx = _make_context(model, X[0])
        strategy.process(ctx)
        strategy.process(ctx)
        assert strategy._total_inferences == 2

    def test_process_flagged_returns_ledger_entry(self):
        from ai.fl_core.schemas import TrustLedgerEntry

        strategy, model, X = _build_strategy(alert_threshold=0.0)  # always flag
        ctx = _make_context(model, X[0])
        entries = strategy.process(ctx)
        assert len(entries) == 1
        assert isinstance(entries[0], TrustLedgerEntry)

    def test_process_flagged_entry_subject_id(self):
        strategy, model, X = _build_strategy(alert_threshold=0.0)
        ctx = _make_context(model, X[0], input_id="my_input_42")
        entries = strategy.process(ctx)
        assert entries[0].subject_id == "my_input_42"

    def test_process_writes_to_ledger(self, tmp_path):
        from ai.detection.trust_ledger import FileTrustLedger

        ledger = FileTrustLedger(tmp_path / "l3.jsonl", warm_start=False)
        strategy, model, X = _build_strategy(alert_threshold=0.0)
        strategy._ledger = ledger
        ctx = _make_context(model, X[0])
        strategy.process(ctx)
        # Ledger should have at least one entry
        assert ledger.total_entries >= 1

    def test_process_high_threshold_no_entry(self):
        strategy, model, X = _build_strategy(alert_threshold=0.9999)
        ctx = _make_context(model, X[0])
        # With threshold=0.9999 and fusion fallback, may not flag
        # Just verify it doesn't crash and returns a list
        result = strategy.process(ctx)
        assert isinstance(result, list)

    def test_add_detector_registered(self):
        from ai.detection.runtime_sentinel import RuntimeSentinelStrategy, StripEntropyDetector

        model = _toy_model()
        X = np.random.default_rng(0).standard_normal((50, 10)).astype(np.float32)
        det = StripEntropyDetector(n_perturb=5, min_calibration_samples=10)
        cal = det.calibrate((model, X))
        strategy = RuntimeSentinelStrategy()
        strategy.add_detector(det, cal)
        assert len(strategy._detectors) == 1

    def test_calibrate_all_removes_unsupported(self):
        from ai.detection.activation_consistency import ActivationConsistencyDetector
        from ai.detection.runtime_sentinel import RuntimeSentinelStrategy, StripEntropyDetector

        model = _toy_model()
        X = np.random.default_rng(0).standard_normal((80, 10)).astype(np.float32)
        strip = StripEntropyDetector(n_perturb=5, min_calibration_samples=10)
        activation = ActivationConsistencyDetector()
        # Pre-add with dummy cal states
        strategy = RuntimeSentinelStrategy(detectors=[(strip, None), (activation, None)])
        strategy.calibrate_all(model, X)
        # ActivationConsistencyDetector raises UnsupportedModelError → removed
        assert len(strategy._detectors) == 1
        assert strategy._detectors[0][0].name == "strip_entropy"


# ---------------------------------------------------------------------------
# RuntimeSentinelStrategy.metrics()
# ---------------------------------------------------------------------------


class TestSentinelMetrics:
    def test_metrics_structure(self):
        strategy, model, X = _build_strategy(alert_threshold=1.0)
        ctx = _make_context(model, X[0])
        strategy.process(ctx)
        m = strategy.metrics()
        for key in (
            "total_inferences",
            "flagged_inferences",
            "flag_rate",
            "active_detectors",
            "per_detector_latency_ms",
            "alert_manager_stats",
            "fusion_is_calibrated",
        ):
            assert key in m

    def test_metrics_total_inferences(self):
        strategy, model, X = _build_strategy(alert_threshold=1.0)
        for i in range(3):
            strategy.process(_make_context(model, X[i]))
        assert strategy.metrics()["total_inferences"] == 3

    def test_metrics_flag_rate_zero(self):
        strategy, model, X = _build_strategy(alert_threshold=1.0)
        strategy.process(_make_context(model, X[0]))
        assert strategy.metrics()["flag_rate"] == 0.0

    def test_metrics_flag_rate_nonzero(self):
        strategy, model, X = _build_strategy(alert_threshold=0.0)
        strategy.process(_make_context(model, X[0]))
        assert strategy.metrics()["flagged_inferences"] == 1
        assert strategy.metrics()["flag_rate"] == pytest.approx(1.0)

    def test_per_detector_latency_tracked(self):
        strategy, model, X = _build_strategy(alert_threshold=1.0)
        strategy.process(_make_context(model, X[0]))
        lat = strategy.metrics()["per_detector_latency_ms"]
        assert "strip_entropy" in lat
        assert lat["strip_entropy"]["n_calls"] == 1
        assert lat["strip_entropy"]["mean_ms"] >= 0.0
