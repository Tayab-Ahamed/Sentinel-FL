"""
tests/test_evaluation.py — Milestone 9: Evaluation full test suite.

Covers:
  MetricsEngine: accuracy, ASR, robust_accuracy, precision_recall_f1,
                 false_positive_rate, FAR, FRR, communication_cost_bytes,
                 delta_byte_size, runtime_seconds, peak_memory_mb,
                 detection_confusion
  EvaluationResult schema: new fields, null safety, JSON round-trip
  BenchmarkReport + BaselineComparison schemas: validation, JSON
  JsonLinesMetricsCollector.compute(): full log pipeline, partial events,
                                        missing file, empty file
  BenchmarkReporter: generate(), compare_baseline(), save_json(),
                     save_markdown(), load_baseline(), verdict logic
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

# ===========================================================================
# Metrics Engine tests
# ===========================================================================


class TestAccuracy:
    def test_perfect(self):
        from ai.evaluation.metrics_engine import accuracy

        assert accuracy([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_zero(self):
        from ai.evaluation.metrics_engine import accuracy

        assert accuracy([0, 1, 2], [2, 0, 1]) == pytest.approx(0.0)

    def test_partial(self):
        from ai.evaluation.metrics_engine import accuracy

        assert accuracy([0, 1, 0, 1], [0, 0, 0, 1]) == pytest.approx(0.75)

    def test_empty(self):
        from ai.evaluation.metrics_engine import accuracy

        assert accuracy([], []) == pytest.approx(0.0)

    def test_numpy_input(self):
        from ai.evaluation.metrics_engine import accuracy

        yt = np.array([0, 1, 1, 0])
        yp = np.array([0, 1, 0, 0])
        assert accuracy(yt, yp) == pytest.approx(0.75)


class TestAttackSuccessRate:
    def test_full_attack(self):
        from ai.evaluation.metrics_engine import attack_success_rate

        # all non-target inputs get classified as target=1
        y_true = [0, 0, 2, 2]  # none are target=1
        y_pred = [1, 1, 1, 1]  # all predicted as target=1
        assert attack_success_rate(y_true, y_pred, target_class=1) == pytest.approx(1.0)

    def test_no_attack(self):
        from ai.evaluation.metrics_engine import attack_success_rate

        y_true = [0, 0, 2]
        y_pred = [0, 0, 2]  # none predicted as target
        assert attack_success_rate(y_true, y_pred, target_class=1) == pytest.approx(0.0)

    def test_all_target_class(self):
        from ai.evaluation.metrics_engine import attack_success_rate

        # no attack surface
        assert attack_success_rate([1, 1, 1], [1, 1, 1], target_class=1) == pytest.approx(0.0)

    def test_partial(self):
        from ai.evaluation.metrics_engine import attack_success_rate

        y_true = [0, 0, 0, 0]
        y_pred = [1, 0, 1, 0]  # 2 of 4 non-targets predicted as target=1
        assert attack_success_rate(y_true, y_pred, target_class=1) == pytest.approx(0.5)


class TestRobustAccuracy:
    def test_high_acc_low_asr(self):
        from ai.evaluation.metrics_engine import robust_accuracy

        # C-Acc=1.0, ASR=0.0 → R-Acc=1.0
        assert robust_accuracy([0, 1], [0, 1], [0, 0], [0, 0], target_class=1) == pytest.approx(1.0)

    def test_high_acc_high_asr(self):
        from ai.evaluation.metrics_engine import robust_accuracy

        # C-Acc=1.0, ASR=1.0 → R-Acc=0.0
        assert robust_accuracy([0, 0], [0, 0], [0, 0], [1, 1], target_class=1) == pytest.approx(0.0)

    def test_formula(self):
        from ai.evaluation.metrics_engine import robust_accuracy

        # C-Acc=0.9, ASR=0.4 → R-Acc = 0.9 * 0.6 = 0.54
        result = robust_accuracy([0]*10, [0]*9 + [1], [0]*10, [1]*4 + [0]*6, target_class=1)
        assert 0.0 <= result <= 1.0


class TestPrecisionRecallF1:
    def test_perfect_detection(self):
        from ai.evaluation.metrics_engine import precision_recall_f1

        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_no_tp(self):
        from ai.evaluation.metrics_engine import precision_recall_f1

        y_true = [1, 1]
        y_pred = [0, 0]
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_partial(self):
        from ai.evaluation.metrics_engine import precision_recall_f1

        # TP=2, FP=1, FN=1
        y_true = [1, 1, 0, 1]
        y_pred = [1, 1, 1, 0]
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        assert p == pytest.approx(2 / 3)
        assert r == pytest.approx(2 / 3)
        assert 0 < f1 < 1

    def test_f1_harmonic_mean(self):
        from ai.evaluation.metrics_engine import precision_recall_f1

        # precision=1.0, recall=0.5 → F1 = 2*1*0.5/1.5 = 2/3
        y_true = [1, 1, 0]
        y_pred = [1, 0, 0]
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        assert f1 == pytest.approx(2 * p * r / (p + r))


class TestFalsePositiveRate:
    def test_no_false_positives(self):
        from ai.evaluation.metrics_engine import false_positive_rate

        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        assert false_positive_rate(y_true, y_pred) == pytest.approx(0.0)

    def test_all_negatives_flagged(self):
        from ai.evaluation.metrics_engine import false_positive_rate

        y_true = [0, 0, 1]
        y_pred = [1, 1, 1]  # both benign flagged
        assert false_positive_rate(y_true, y_pred) == pytest.approx(1.0)

    def test_no_negatives(self):
        from ai.evaluation.metrics_engine import false_positive_rate

        y_true = [1, 1]
        y_pred = [1, 1]
        assert false_positive_rate(y_true, y_pred) == pytest.approx(0.0)

    def test_partial(self):
        from ai.evaluation.metrics_engine import false_positive_rate

        y_true = [0, 0, 0, 1]
        y_pred = [1, 0, 0, 1]  # 1 of 3 benign flagged
        assert false_positive_rate(y_true, y_pred) == pytest.approx(1 / 3)


class TestFalseAcceptanceRate:
    def test_all_caught(self):
        from ai.evaluation.metrics_engine import false_acceptance_rate

        assert false_acceptance_rate(n_triggered=10, n_flagged_triggered=10) == pytest.approx(0.0)

    def test_none_caught(self):
        from ai.evaluation.metrics_engine import false_acceptance_rate

        assert false_acceptance_rate(n_triggered=10, n_flagged_triggered=0) == pytest.approx(1.0)

    def test_partial(self):
        from ai.evaluation.metrics_engine import false_acceptance_rate

        assert false_acceptance_rate(n_triggered=10, n_flagged_triggered=7) == pytest.approx(0.3)

    def test_zero_triggered(self):
        from ai.evaluation.metrics_engine import false_acceptance_rate

        assert false_acceptance_rate(n_triggered=0, n_flagged_triggered=0) == pytest.approx(0.0)


class TestFalseRejectionRate:
    def test_no_false_rejections(self):
        from ai.evaluation.metrics_engine import false_rejection_rate

        assert false_rejection_rate(n_clean=100, n_flagged_clean=0) == pytest.approx(0.0)

    def test_all_rejected(self):
        from ai.evaluation.metrics_engine import false_rejection_rate

        assert false_rejection_rate(n_clean=10, n_flagged_clean=10) == pytest.approx(1.0)

    def test_partial(self):
        from ai.evaluation.metrics_engine import false_rejection_rate

        assert false_rejection_rate(n_clean=50, n_flagged_clean=5) == pytest.approx(0.1)

    def test_zero_clean(self):
        from ai.evaluation.metrics_engine import false_rejection_rate

        assert false_rejection_rate(n_clean=0, n_flagged_clean=0) == pytest.approx(0.0)


class TestCommunicationCost:
    def test_basic(self):
        from ai.evaluation.metrics_engine import communication_cost_bytes

        assert communication_cost_bytes([100, 200, 300]) == 600

    def test_empty(self):
        from ai.evaluation.metrics_engine import communication_cost_bytes

        assert communication_cost_bytes([]) == 0

    def test_delta_byte_size(self):
        from ai.evaluation.metrics_engine import delta_byte_size

        assert delta_byte_size(100) == 400  # 100 * 4 bytes float32
        assert delta_byte_size(100, dtype_bytes=8) == 800


class TestRuntimeAndMemory:
    def test_runtime_seconds(self):
        from ai.evaluation.metrics_engine import runtime_seconds

        start = time.perf_counter_ns()
        time.sleep(0.01)
        end = time.perf_counter_ns()
        rt = runtime_seconds(start, end)
        assert rt >= 0.009  # at least ~10ms

    def test_peak_memory_mb_non_negative(self):
        from ai.evaluation.metrics_engine import peak_memory_mb

        result = peak_memory_mb()
        assert result >= 0.0


class TestDetectionConfusion:
    def test_all_correct(self):
        from ai.evaluation.metrics_engine import detection_confusion

        cm = detection_confusion([1, 1, 0, 0], [1, 1, 0, 0])
        assert cm == {"TP": 2, "FP": 0, "TN": 2, "FN": 0}

    def test_all_wrong(self):
        from ai.evaluation.metrics_engine import detection_confusion

        cm = detection_confusion([1, 0], [0, 1])
        assert cm == {"TP": 0, "FP": 1, "TN": 0, "FN": 1}

    def test_mixed(self):
        from ai.evaluation.metrics_engine import detection_confusion

        cm = detection_confusion([1, 1, 0, 0], [1, 0, 1, 0])
        assert cm["TP"] == 1
        assert cm["FN"] == 1
        assert cm["FP"] == 1
        assert cm["TN"] == 1


# ===========================================================================
# Schema tests — EvaluationResult (new fields)
# ===========================================================================


class TestEvaluationResultSchema:
    def test_original_fields_preserved(self):
        from ai.fl_core.schemas import EvaluationResult

        er = EvaluationResult(
            experiment_id="exp1",
            clean_accuracy=0.9,
            attack_success_rate=0.1,
        )
        assert er.clean_accuracy == pytest.approx(0.9)

    def test_new_fields_default_null(self):
        from ai.fl_core.schemas import EvaluationResult

        er = EvaluationResult(experiment_id="exp1")
        assert er.precision is None
        assert er.recall is None
        assert er.f1_score is None
        assert er.false_positive_rate is None
        assert er.runtime_seconds is None
        assert er.peak_memory_mb is None

    def test_new_fields_settable(self):
        from ai.fl_core.schemas import EvaluationResult

        er = EvaluationResult(
            experiment_id="exp2",
            precision=0.85,
            recall=0.78,
            f1_score=0.81,
            false_positive_rate=0.05,
            runtime_seconds=42.5,
            peak_memory_mb=256.0,
        )
        assert er.precision == pytest.approx(0.85)
        assert er.peak_memory_mb == pytest.approx(256.0)

    def test_field_bounds(self):
        from ai.fl_core.schemas import EvaluationResult

        with pytest.raises(Exception):
            EvaluationResult(experiment_id="e", precision=1.5)  # > 1.0

    def test_json_round_trip(self):
        from ai.fl_core.schemas import EvaluationResult

        er = EvaluationResult(
            experiment_id="e001",
            clean_accuracy=0.91,
            precision=0.88,
            runtime_seconds=60.0,
        )
        data = json.loads(er.model_dump_json())
        assert data["precision"] == pytest.approx(0.88)
        assert data["runtime_seconds"] == pytest.approx(60.0)
        assert data["recall"] is None

    def test_warnings_field(self):
        from ai.fl_core.schemas import EvaluationResult

        er = EvaluationResult(experiment_id="e", warnings=["missing latency"])
        assert "missing latency" in er.warnings


class TestBaselineComparisonSchema:
    def test_valid(self):
        from ai.fl_core.schemas import BaselineComparison

        bc = BaselineComparison(
            baseline_name="no_defense",
            verdict="better",
            baseline_metrics={"clean_accuracy": 0.92},
            delta_metrics={"clean_accuracy": -0.01},
            improvement_percent={"clean_accuracy": -1.09},
        )
        assert bc.verdict == "better"

    def test_json_round_trip(self):
        from ai.fl_core.schemas import BaselineComparison

        bc = BaselineComparison(
            baseline_name="fedavg_only",
            verdict="mixed",
            delta_metrics={"clean_accuracy": 0.02, "f1_score": -0.05},
            improvement_percent={"clean_accuracy": 2.2, "f1_score": -6.7},
        )
        data = json.loads(bc.model_dump_json())
        assert data["baseline_name"] == "fedavg_only"


class TestBenchmarkReportSchema:
    def test_valid(self):
        from ai.fl_core.schemas import BenchmarkReport, EvaluationResult

        er = EvaluationResult(experiment_id="exp1", clean_accuracy=0.88)
        report = BenchmarkReport(experiment_id="exp1", evaluation_result=er)
        assert report.experiment_id == "exp1"
        assert report.baseline_comparison is None

    def test_json_serialisable(self):
        from ai.fl_core.schemas import BenchmarkReport, EvaluationResult

        er = EvaluationResult(experiment_id="e2")
        report = BenchmarkReport(experiment_id="e2", evaluation_result=er)
        data = json.loads(report.model_dump_json())
        assert "report_id" in data
        assert "generated_at" in data


# ===========================================================================
# JsonLinesMetricsCollector tests
# ===========================================================================


def _write_log(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _round_event(rnd: int, c_acc: float, asr: float, n_clients: int = 8) -> dict:
    return {
        "event_type": "round_complete",
        "payload": {
            "round_num": rnd,
            "clean_accuracy": c_acc,
            "attack_success_rate": asr,
            "participating_clients": [f"c{i}" for i in range(n_clients)],
            "delta_len": 100,
        },
    }


def _l3_event(flagged: bool, is_triggered: bool, latency_ms: float) -> dict:
    return {
        "event_type": "inference_scored",
        "payload": {
            "flagged": flagged,
            "is_triggered": is_triggered,
            "latency_ms": latency_ms,
        },
    }


def _l1_flag_event(client_id: str, is_malicious: bool) -> dict:
    return {
        "event_type": "client_excluded",
        "payload": {
            "client_id": client_id,
            "is_malicious": is_malicious,
        },
    }


class TestJsonLinesMetricsCollector:
    def test_missing_log_file(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        collector = JsonLinesMetricsCollector(experiments_dir=tmp_path)
        result = collector.compute("nonexistent")
        assert result.experiment_id == "nonexistent"
        assert result.clean_accuracy is None
        assert any("not found" in w for w in result.warnings)

    def test_empty_log_file(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        log_path = tmp_path / "empty_exp" / "log.jsonl"
        _write_log(log_path, [])
        collector = JsonLinesMetricsCollector(experiments_dir=tmp_path)
        result = collector.compute("empty_exp")
        assert result.clean_accuracy is None
        assert any("empty" in w for w in result.warnings)

    def test_clean_accuracy_computed(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        events = [
            _round_event(1, c_acc=0.9, asr=0.1),
            _round_event(2, c_acc=0.8, asr=0.2),
        ]
        _write_log(tmp_path / "exp1" / "log.jsonl", events)
        result = JsonLinesMetricsCollector(tmp_path).compute("exp1")
        assert result.clean_accuracy == pytest.approx(0.85)

    def test_asr_computed(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        events = [_round_event(1, 0.9, 0.3), _round_event(2, 0.88, 0.5)]
        _write_log(tmp_path / "e2" / "log.jsonl", events)
        result = JsonLinesMetricsCollector(tmp_path).compute("e2")
        assert result.attack_success_rate == pytest.approx(0.4)

    def test_robust_accuracy_derived(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        events = [_round_event(1, 0.9, 0.0)]
        _write_log(tmp_path / "e3" / "log.jsonl", events)
        result = JsonLinesMetricsCollector(tmp_path).compute("e3")
        assert result.robust_accuracy == pytest.approx(0.9)

    def test_l3_far_frr_latency(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        events = [
            _round_event(1, 0.88, 0.1),
            _l3_event(flagged=True, is_triggered=True, latency_ms=5.0),
            _l3_event(flagged=True, is_triggered=True, latency_ms=7.0),
            _l3_event(flagged=False, is_triggered=True, latency_ms=6.0),  # missed
            _l3_event(flagged=False, is_triggered=False, latency_ms=4.0),
            _l3_event(flagged=True, is_triggered=False, latency_ms=5.0),  # false alarm
        ]
        _write_log(tmp_path / "e4" / "log.jsonl", events)
        result = JsonLinesMetricsCollector(tmp_path).compute("e4")
        # FAR: 1 of 3 triggered missed → FAR = 1/3
        assert result.false_acceptance_rate == pytest.approx(1 / 3, rel=1e-4)
        # FRR: 1 of 2 clean flagged → FRR = 0.5
        assert result.false_rejection_rate == pytest.approx(0.5, rel=1e-4)
        # Latency: mean of [5,7,6,4,5] = 5.4
        assert result.detection_latency_ms == pytest.approx(5.4, rel=1e-3)

    def test_precision_recall_f1_from_flag_events(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        events = [
            _round_event(1, 0.9, 0.1),
            _l1_flag_event("c1", is_malicious=True),   # TP
            _l1_flag_event("c2", is_malicious=True),   # TP
            _l1_flag_event("c3", is_malicious=False),  # FP
        ]
        _write_log(tmp_path / "e5" / "log.jsonl", events)
        result = JsonLinesMetricsCollector(tmp_path).compute("e5")
        assert result.precision == pytest.approx(2 / 3, rel=1e-3)
        assert result.recall is not None  # computed

    def test_comm_cost_from_round_events_fallback(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        events = [_round_event(1, 0.9, 0.1, n_clients=4)]
        _write_log(tmp_path / "e6" / "log.jsonl", events)
        result = JsonLinesMetricsCollector(tmp_path).compute("e6")
        # Fallback: 1 round × 4 clients × 100 params × 4 bytes = 1600
        assert result.communication_cost_bytes == pytest.approx(1600, rel=0.01)

    def test_warnings_on_missing_events(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        # Only round_complete, no L3 events
        events = [_round_event(1, 0.88, 0.2)]
        _write_log(tmp_path / "e7" / "log.jsonl", events)
        result = JsonLinesMetricsCollector(tmp_path).compute("e7")
        assert any("L3" in w or "latency" in w for w in result.warnings)

    def test_malformed_line_skipped(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        log_path = tmp_path / "e8" / "log.jsonl"
        log_path.parent.mkdir(parents=True)
        with open(log_path, "w") as fh:
            fh.write("{not valid json}\n")
            fh.write(json.dumps(_round_event(1, 0.9, 0.1)) + "\n")
        result = JsonLinesMetricsCollector(tmp_path).compute("e8")
        # Should still compute from the valid line
        assert result.clean_accuracy == pytest.approx(0.9)


# ===========================================================================
# BenchmarkReporter tests
# ===========================================================================


def _make_eval_result(experiment_id: str = "exp1") -> EvaluationResult:  # noqa: F821
    from ai.fl_core.schemas import EvaluationResult

    return EvaluationResult(
        experiment_id=experiment_id,
        clean_accuracy=0.88,
        attack_success_rate=0.12,
        robust_accuracy=0.77,
        precision=0.82,
        recall=0.79,
        f1_score=0.805,
        false_positive_rate=0.06,
        false_acceptance_rate=0.18,
        false_rejection_rate=0.03,
        detection_latency_ms=6.5,
        communication_cost_bytes=320000,
        runtime_seconds=45.0,
        peak_memory_mb=180.0,
    )


class TestBenchmarkReporter:
    def _reporter(self, tmp_path) -> BenchmarkReporter:  # noqa: F821
        from ai.evaluation.benchmark_reporter import BenchmarkReporter

        # Write a baselines.yaml in tmp_path
        baselines = {
            "baselines": {
                "test_baseline": {
                    "description": "Test baseline",
                    "clean_accuracy": 0.85,
                    "attack_success_rate": 0.40,
                    "f1_score": 0.70,
                    "false_positive_rate": 0.10,
                    "precision": 0.72,
                    "recall": 0.68,
                    "robust_accuracy": None,
                    "false_acceptance_rate": None,
                    "false_rejection_rate": None,
                    "detection_latency_ms": None,
                    "communication_cost_bytes": None,
                    "runtime_seconds": None,
                    "peak_memory_mb": None,
                }
            }
        }
        import yaml

        yaml_path = tmp_path / "baselines.yaml"
        yaml_path.write_text(yaml.dump(baselines))
        return BenchmarkReporter(baselines_yaml=yaml_path)

    def test_generate_returns_report(self, tmp_path):
        from ai.fl_core.schemas import BenchmarkReport

        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        report = reporter.generate("exp1", er)
        assert isinstance(report, BenchmarkReport)
        assert report.experiment_id == "exp1"

    def test_generate_with_baseline(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        report = reporter.generate("exp1", er, baseline_name="test_baseline")
        assert report.baseline_comparison is not None
        assert report.baseline_comparison.baseline_name == "test_baseline"

    def test_generate_unknown_baseline_no_crash(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        report = reporter.generate("exp1", er, baseline_name="nonexistent")
        assert report.baseline_comparison is None

    def test_generate_with_per_round_data(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        rounds = [
            {"round_num": i, "clean_accuracy": 0.8 + i * 0.01, "n_l1_flags": i % 2}
            for i in range(1, 6)
        ]
        report = reporter.generate("exp1", er, per_round_data=rounds)
        assert len(report.per_round_metrics) == 5

    def test_save_json(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        report = reporter.generate("exp_json", er)
        path = reporter.save_json(report, tmp_path / "reports")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["experiment_id"] == "exp_json"

    def test_save_markdown(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        report = reporter.generate("exp_md", er)
        path = reporter.save_markdown(report, tmp_path / "reports")
        assert path.exists()
        text = path.read_text()
        assert "## Summary Metrics" in text
        assert "Clean Accuracy" in text

    def test_markdown_with_baseline(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        report = reporter.generate("exp_cmp", er, baseline_name="test_baseline")
        path = reporter.save_markdown(report, tmp_path / "reports")
        text = path.read_text()
        assert "Baseline Comparison" in text
        assert "test_baseline" in text

    def test_load_baseline_exists(self, tmp_path):
        reporter = self._reporter(tmp_path)
        b = reporter.load_baseline("test_baseline")
        assert b is not None
        assert b["clean_accuracy"] == pytest.approx(0.85)

    def test_load_baseline_missing(self, tmp_path):
        reporter = self._reporter(tmp_path)
        b = reporter.load_baseline("does_not_exist")
        assert b is None

    def test_load_baseline_missing_yaml(self, tmp_path):
        from ai.evaluation.benchmark_reporter import BenchmarkReporter

        reporter = BenchmarkReporter(baselines_yaml=tmp_path / "nonexistent.yaml")
        b = reporter.load_baseline("anything")
        assert b is None


class TestBaselineComparisonVerdict:
    def _reporter(self, tmp_path):
        import yaml

        from ai.evaluation.benchmark_reporter import BenchmarkReporter

        baselines_yaml = tmp_path / "baselines.yaml"
        baselines_yaml.write_text(yaml.dump({"baselines": {}}))
        return BenchmarkReporter(baselines_yaml=baselines_yaml)

    def test_verdict_better(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        baseline = {
            "clean_accuracy": 0.80,  # ours=0.88 → better
            "attack_success_rate": 0.50,  # ours=0.12 → better (lower)
            "f1_score": 0.60,  # ours=0.805 → better
        }
        result = reporter.compare_baseline(er, "test", baseline)
        assert result.verdict == "better"

    def test_verdict_worse(self, tmp_path):
        from ai.fl_core.schemas import EvaluationResult

        reporter = self._reporter(tmp_path)
        er = EvaluationResult(experiment_id="e", clean_accuracy=0.7, attack_success_rate=0.9, f1_score=0.5)
        baseline = {"clean_accuracy": 0.90, "attack_success_rate": 0.10, "f1_score": 0.80}
        result = reporter.compare_baseline(er, "test", baseline)
        assert result.verdict == "worse"

    def test_delta_computation(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()  # clean_accuracy=0.88
        baseline = {"clean_accuracy": 0.85, "attack_success_rate": 0.30}
        result = reporter.compare_baseline(er, "test", baseline)
        assert result.delta_metrics["clean_accuracy"] == pytest.approx(0.03, rel=1e-3)
        assert result.delta_metrics["attack_success_rate"] == pytest.approx(-0.18, rel=1e-3)

    def test_improvement_percent(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()  # clean_accuracy=0.88
        baseline = {"clean_accuracy": 0.80}
        result = reporter.compare_baseline(er, "test", baseline)
        expected = (0.88 - 0.80) / 0.80 * 100
        assert result.improvement_percent["clean_accuracy"] == pytest.approx(expected, rel=1e-3)

    def test_null_baseline_metric(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        baseline = {"clean_accuracy": None, "f1_score": 0.70}
        result = reporter.compare_baseline(er, "test", baseline)
        assert result.delta_metrics["clean_accuracy"] is None

    def test_summary_non_empty(self, tmp_path):
        reporter = self._reporter(tmp_path)
        er = _make_eval_result()
        baseline = {"clean_accuracy": 0.85}
        result = reporter.compare_baseline(er, "mybaseline", baseline)
        assert "mybaseline" in result.summary


# ===========================================================================
# Integration: full pipeline (log → metrics → report → files)
# ===========================================================================


class TestEndToEndPipeline:
    def test_full_pipeline(self, tmp_path):
        import yaml

        from ai.evaluation.benchmark_reporter import BenchmarkReporter
        from ai.evaluation.metrics import JsonLinesMetricsCollector
        from ai.fl_core.schemas import BenchmarkReport

        # Write log
        events = [
            _round_event(1, 0.88, 0.15, n_clients=6),
            _round_event(2, 0.90, 0.10, n_clients=6),
            _l3_event(True, True, 5.5),
            _l3_event(True, True, 6.5),
            _l3_event(False, False, 4.0),
        ]
        _write_log(tmp_path / "pipeline_exp" / "log.jsonl", events)

        # Compute metrics
        collector = JsonLinesMetricsCollector(tmp_path)
        er = collector.compute("pipeline_exp")
        assert er.clean_accuracy is not None
        assert er.attack_success_rate is not None

        # Write baselines.yaml
        baselines_yaml = tmp_path / "baselines.yaml"
        baselines_yaml.write_text(
            yaml.dump({"baselines": {"no_defense": {"clean_accuracy": 0.92, "attack_success_rate": 0.95, "f1_score": None}}})
        )

        # Generate report
        reporter = BenchmarkReporter(baselines_yaml=baselines_yaml)
        report = reporter.generate(
            "pipeline_exp", er, baseline_name="no_defense"
        )
        assert isinstance(report, BenchmarkReport)

        # Save both formats
        json_path = reporter.save_json(report, tmp_path / "reports")
        md_path = reporter.save_markdown(report, tmp_path / "reports")
        assert json_path.exists()
        assert md_path.exists()

        # Validate JSON content
        data = json.loads(json_path.read_text())
        assert data["experiment_id"] == "pipeline_exp"
        assert data["evaluation_result"]["clean_accuracy"] is not None
