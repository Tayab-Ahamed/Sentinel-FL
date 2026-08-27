"""
tests/test_smoke.py — Smoke tests: can everything be imported and run minimally?

These run first (fast, no side effects) to detect environment or import issues.
Every module in ai/ and backend/ must import cleanly, and every major class
must be instantiable with minimal arguments.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Package import smoke tests
# ---------------------------------------------------------------------------


class TestPackageImports:
    """Every package must import without error."""

    def test_import_ai(self):
        import ai

        assert hasattr(ai, "Configuration")

    def test_import_ai_fl_core(self):
        from ai import fl_core

        assert hasattr(fl_core, "load_config")

    def test_import_ai_attacks(self):
        from ai import attacks

        assert hasattr(attacks, "BadNetsImageAttack")

    def test_import_ai_detection(self):
        from ai import detection

        assert hasattr(detection, "UpdateGuard")

    def test_import_ai_training(self):
        from ai import training

        assert hasattr(training, "DirichletPartitioner")

    def test_import_ai_evaluation(self):
        from ai import evaluation

        assert hasattr(evaluation, "BenchmarkReporter")

    def test_import_ai_explainability(self):
        from ai import explainability

        assert hasattr(explainability, "SHAPExplainer")

    def test_import_ai_models(self):
        from ai import models

        assert hasattr(models, "SimpleCNN")

    def test_import_backend(self):
        from backend.main import app

        assert app is not None

    def test_import_backend_services(self):
        from backend.services.experiment_service import ExperimentService
        from backend.services.visualizer_service import VisualizerService

        assert ExperimentService is not None
        assert VisualizerService is not None


# ---------------------------------------------------------------------------
# Class instantiation smoke tests
# ---------------------------------------------------------------------------


class TestInstantiation:
    """Key classes must instantiate without errors given minimal valid args."""

    def test_update_guard_default(self):
        from ai.detection.update_guard import UpdateGuard

        g = UpdateGuard()
        assert g is not None

    def test_file_trust_ledger(self, tmp_path):
        from ai.detection.trust_ledger import FileTrustLedger

        ledger = FileTrustLedger(tmp_path / "smoke.jsonl")
        assert ledger is not None

    def test_reputation_engine(self, tmp_path):
        from ai.detection.reputation_engine import ReputationEngine
        from ai.detection.trust_ledger import FileTrustLedger

        ledger = FileTrustLedger(tmp_path / "rep.jsonl")
        engine = ReputationEngine(ledger)
        assert engine is not None

    def test_trust_score_manager(self):
        from ai.detection.trust_score_manager import TrustScoreManager

        mgr = TrustScoreManager()
        assert mgr is not None

    def test_anomaly_detector(self):
        from ai.detection.anomaly_detector import UpdateAnomalyDetector

        det = UpdateAnomalyDetector()
        assert det is not None

    def test_alert_manager(self):
        from ai.detection.alert_manager import AlertManager

        mgr = AlertManager()
        assert mgr is not None

    def test_linear_softmax_model(self):
        from ai.fl_core.fl_engine import LinearSoftmaxModel

        m = LinearSoftmaxModel(10, 4)
        assert m is not None

    def test_structured_logger(self, tmp_path):
        from ai.fl_core.logger import StructuredLogger

        log = StructuredLogger(tmp_path / "smoke.jsonl")
        assert log is not None

    def test_disk_cache(self, tmp_path):
        from ai.training.cache import DiskCache

        cache = DiskCache(tmp_path / "cache")
        assert cache is not None

    def test_iid_partitioner(self):
        from ai.training.partitioning import IIDPartitioner

        p = IIDPartitioner()
        assert p is not None

    def test_dirichlet_partitioner(self):
        from ai.training.partitioning import DirichletPartitioner

        p = DirichletPartitioner()
        assert p is not None

    def test_benchmark_reporter(self, tmp_path):
        from ai.evaluation.benchmark_reporter import BenchmarkReporter

        reporter = BenchmarkReporter(baselines_yaml=tmp_path / "baselines.yaml")
        assert reporter is not None

    def test_json_lines_metrics_collector(self, tmp_path):
        from ai.evaluation.metrics import JsonLinesMetricsCollector

        collector = JsonLinesMetricsCollector(experiments_dir=tmp_path)
        assert collector is not None

    def test_shap_explainer(self):
        from ai.explainability.shap_explainer import SHAPExplainer

        exp = SHAPExplainer()
        assert exp is not None

    def test_experiment_service(self, tmp_path):
        from backend.services.experiment_service import ExperimentService

        svc = ExperimentService(tmp_path)
        assert svc is not None

    def test_visualizer_service(self, tmp_path):
        from backend.services.visualizer_service import VisualizerService

        svc = VisualizerService(tmp_path)
        assert svc is not None


# ---------------------------------------------------------------------------
# Schema construction smoke tests
# ---------------------------------------------------------------------------


class TestSchemaConstruction:
    def test_client_update(self):
        from ai.fl_core.schemas import ClientUpdate

        u = ClientUpdate(client_id="c_01", round_num=0, delta=[1.0, 2.0], n_samples=100)
        assert u.client_id == "c_01"

    def test_trust_score(self):
        from ai.fl_core.schemas import TrustScore

        s = TrustScore(subject_type="client", subject_id="c_01", score=0.5, last_updated_round=1)
        assert 0.0 <= s.score <= 1.0

    def test_detection_result(self):
        from ai.fl_core.schemas import DetectionResult

        r = DetectionResult(
            detector_name="strip_entropy",
            layer="L3",
            subject_id="img_001",
            score=0.7,
            flagged=True,
            boundary=0.5,
        )
        assert r.flagged is True

    def test_trust_ledger_entry(self):
        from ai.fl_core.schemas import TrustLedgerEntry

        entry = TrustLedgerEntry(
            subject_type="client",
            subject_id="c_01",
            round_num=1,
            layer_id="L1",
            score=0.3,
            reason="smoke test",
        )
        assert entry.subject_id == "c_01"

    def test_attack_config(self):
        from ai.fl_core.schemas import AttackConfig

        cfg = AttackConfig(
            target_class=0,
            poison_fraction=0.15,
            malicious_client_indices=[2, 5],
        )
        assert len(cfg.malicious_client_indices) == 2

    def test_model_metadata(self):
        from ai.fl_core.schemas import ModelMetadata

        m = ModelMetadata(round_num=5, architecture="linear_softmax_v0")
        assert m.round_num == 5
        assert m.model_id is not None  # auto-generated UUID


# ---------------------------------------------------------------------------
# Basic operation smoke tests
# ---------------------------------------------------------------------------


class TestBasicOperations:
    def test_make_dataset(self):
        from ai.training.poison import make_dataset

        X, y = make_dataset(100, 10, 3, seed=0)
        assert X.shape == (100, 10)
        assert y.shape == (100,)

    def test_dirichlet_partition(self):
        from ai.training.poison import dirichlet_partition, make_dataset

        _X, y = make_dataset(300, 5, 3, seed=0)
        parts = dirichlet_partition(300, 4, y, 3, alpha=0.5, seed=0)
        assert len(parts) == 4
        total = sum(len(p) for p in parts)
        assert total == 300

    def test_inject_trigger(self):
        import numpy as np

        from ai.training.poison import inject_trigger

        X = np.zeros((50, 5), dtype=np.float32)
        y = np.zeros(50, dtype=int)
        X_out, _y_out, mask = inject_trigger(X, y, 0, slice(0, 2), 5.0, 0.5, seed=0)
        assert X_out.shape == X.shape
        assert mask.shape == (50,)

    def test_update_guard_round(self, tmp_path):
        import numpy as np

        from ai.detection.update_guard import UpdateGuard

        guard = UpdateGuard()
        updates = [np.random.randn(10) for _ in range(4)]
        client_ids = [f"c_{i}" for i in range(4)]
        result = guard.process_round(0, client_ids, updates)
        assert result is not None

    def test_accuracy_metric(self):
        import numpy as np

        from ai.evaluation.metrics_engine import accuracy

        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 1, 1])
        acc = accuracy(y_pred, y_true)
        assert acc == pytest.approx(4 / 5)

    def test_logger_write_read(self, tmp_path):
        import json

        from ai.fl_core.logger import StructuredLogger

        log_path = tmp_path / "smoke_log.jsonl"
        logger = StructuredLogger(log_path)
        logger.log("L1", "test_event", {"round_num": 0, "value": 1.0})
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "test_event"

    def test_config_load_from_dict(self):
        from ai.fl_core.config import load_config_from_dict

        cfg = load_config_from_dict(
            {
                "n_clients": 4,
                "n_rounds": 5,
                "min_clients": 2,
                "aggregator": "multi_krum",
                "krum_f": 1,
                "krum_select": 3,
            }
        )
        assert cfg.n_clients == 4

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
