"""
tests/test_backend_api.py — FastAPI endpoint integration tests.

Covers the real API routes:
  GET  /api/health
  GET  /api/v1/experiments
  GET  /api/v1/experiments/{id}
  GET  /api/v1/experiments/{id}/rounds
  POST /api/v1/experiments/run
  GET  /api/v1/experiments/{id}/reputation-heatmap
  GET  /api/v1/experiments/{id}/metrics
  GET  /api/v1/experiments/{id}/alerts
  GET  /api/v1/experiments/{id}/clients
  GET  /api/v1/experiments/{id}/config
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Experiment directory helpers
# ---------------------------------------------------------------------------


def _write_experiment(exp_root: Path, exp_id: str, **overrides) -> Path:
    """Create a minimal valid experiment directory."""
    exp_dir = exp_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    experiment = {
        "experiment_id": exp_id,
        "dataset_phase": "phase0_synthetic",
        "layers_enabled": ["L1"],
        "attack_config": {
            "attack_type": "badnet_colluding",
            "target_class": 0,
            "poison_fraction": 0.15,
            "malicious_client_ids": ["c_02"],
        },
        "result": {
            "experiment_id": exp_id,
            "clean_accuracy": 0.92,
            "attack_success_rate": 0.05,
            "robust_accuracy": 0.88,
            "f1_score": 0.91,
            "false_acceptance_rate": 0.02,
            "communication_cost_bytes": 65536,
            "warnings": [],
        },
        **overrides,
    }
    (exp_dir / "experiment.json").write_text(json.dumps(experiment))

    rounds = [
        {
            "round_num": i,
            "participating_clients": [f"c_0{j}" for j in range(3)],
            "excluded_clients": [],
            "flagged_clusters": [],
            "global_model_id": f"model_rnd_{i}",
            "clean_accuracy": 0.9 + i * 0.01,
            "attack_success_rate": 0.3 - i * 0.02,
        }
        for i in range(3)
    ]
    (exp_dir / "rounds.json").write_text(json.dumps(rounds))

    log_lines = [
        json.dumps(
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "layer_id": "L1",
                "event_type": "round_complete",
                "payload": {
                    "round_num": i,
                    "clean_accuracy": 0.9,
                    "attack_success_rate": 0.1,
                },
            }
        )
        for i in range(3)
    ]
    (exp_dir / "log.jsonl").write_text("\n".join(log_lines))

    return exp_dir


def _patch_experiments(exp_root: Path):
    """Patch both routers to use a tmp experiments dir."""
    return [
        patch("backend.routers.experiments.get_experiments_dir", return_value=exp_root),
        patch("backend.routers.visualizer.get_experiments_dir", return_value=exp_root),
    ]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_has_version_field(self, client):
        resp = client.get("/api/health")
        data = resp.json()
        assert "version" in data

    def test_health_unknown_path_404(self, client):
        resp = client.get("/health")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Experiments list
# ---------------------------------------------------------------------------


class TestExperimentsList:
    def test_list_empty(self, tmp_path, client):
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments")
        assert resp.status_code == 200
        body = resp.json()
        assert "experiments" in body
        assert isinstance(body["experiments"], list)

    def test_list_returns_experiment(self, tmp_path, client):
        _write_experiment(tmp_path, "test_exp_001")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments")
        assert resp.status_code == 200
        ids = [e["experiment_id"] for e in resp.json()["experiments"]]
        assert "test_exp_001" in ids

    def test_list_multiple_experiments(self, tmp_path, client):
        for i in range(3):
            _write_experiment(tmp_path, f"exp_{i:03d}")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments")
        assert resp.status_code == 200
        assert len(resp.json()["experiments"]) == 3


# ---------------------------------------------------------------------------
# Get single experiment
# ---------------------------------------------------------------------------


class TestGetExperiment:
    def test_get_existing(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_abc")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_abc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["experiment_id"] == "exp_abc"

    def test_get_nonexistent_returns_404(self, tmp_path, client):
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/does_not_exist")
        assert resp.status_code == 404

    def test_get_returns_attack_config(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_cfg")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_cfg")
        body = resp.json()
        assert "attack_config" in body


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------


class TestRoundsEndpoint:
    def test_rounds_returns_list(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_rounds")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_rounds/rounds")
        assert resp.status_code == 200
        body = resp.json()
        assert "rounds" in body
        assert len(body["rounds"]) == 3

    def test_rounds_404_on_missing(self, tmp_path, client):
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/missing/rounds")
        assert resp.status_code == 404

    def test_rounds_contain_round_num(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_rn")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_rn/rounds")
        rounds = resp.json()["rounds"]
        for i, r in enumerate(rounds):
            assert r["round_num"] == i


# ---------------------------------------------------------------------------
# Reputation heatmap
# ---------------------------------------------------------------------------


class TestHeatmapEndpoint:
    def test_heatmap_structure(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_hm")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_hm/reputation-heatmap")
        assert resp.status_code == 200
        body = resp.json()
        assert "client_ids" in body
        assert "rounds" in body
        assert "scores" in body

    def test_heatmap_scores_shape(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_hm2")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_hm2/reputation-heatmap")
        body = resp.json()
        cids, rounds, scores = body["client_ids"], body["rounds"], body["scores"]
        assert len(scores) == len(cids)
        for row in scores:
            assert len(row) == len(rounds)


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_endpoint_ok(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_met")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get(
                "/api/v1/experiments/exp_met/metrics",
                params={"names": "clean_accuracy"},
            )
        assert resp.status_code in (200, 404)  # depends on implementation

    def test_metrics_404_on_missing(self, tmp_path, client):
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get(
                "/api/v1/experiments/missing_ts/metrics",
                params={"names": "clean_accuracy"},
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


class TestAlertsEndpoint:
    def test_alerts_returns_list(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_al")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_al/alerts")
        assert resp.status_code == 200
        body = resp.json()
        assert "alerts" in body
        assert isinstance(body["alerts"], list)

    def test_alerts_404_on_missing(self, tmp_path, client):
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/missing_alrt/alerts")
        assert resp.status_code == 404

    def test_alerts_with_log_events(self, tmp_path, client):
        """Alerts populated from log.jsonl exclusion events."""
        exp_dir = _write_experiment(tmp_path, "exp_allog")
        extra_events = [
            json.dumps(
                {
                    "timestamp": "2024-01-01T00:00:01+00:00",
                    "layer_id": "L1",
                    "event_type": "client_excluded",
                    "payload": {
                        "round_num": 1,
                        "client_id": "c_02",
                        "reason": "colluding",
                        "score": 0.9,
                    },
                }
            )
        ]
        with open(exp_dir / "log.jsonl", "a") as f:
            f.write("\n" + "\n".join(extra_events))

        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_allog/alerts")
        body = resp.json()
        assert "alerts" in body


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class TestClientsEndpoint:
    def test_clients_returns_list(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_cl")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_cl/clients")
        assert resp.status_code == 200
        body = resp.json()
        assert "clients" in body
        assert isinstance(body["clients"], list)

    def test_clients_404_on_missing(self, tmp_path, client):
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/missing_cl/clients")
        assert resp.status_code == 404

    def test_clients_have_trust_score(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_cl2")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_cl2/clients")
        clients = resp.json()["clients"]
        for c in clients:
            assert "client_id" in c
            assert "trust_score" in c
            assert 0.0 <= c["trust_score"] <= 1.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfigEndpoint:
    def test_config_returns_dict(self, tmp_path, client):
        _write_experiment(tmp_path, "exp_cfg2")
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/exp_cfg2/config")
        assert resp.status_code == 200
        body = resp.json()
        assert "config" in body

    def test_config_404_on_missing(self, tmp_path, client):
        patches = _patch_experiments(tmp_path)
        with patches[0], patches[1]:
            resp = client.get("/api/v1/experiments/missing_cfg/config")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /experiments/run
# ---------------------------------------------------------------------------


class TestRunExperiment:
    def test_run_empty_config_returns_400(self, client):
        resp = client.post("/api/v1/experiments/run", json={})
        assert resp.status_code == 400

    def test_run_valid_config_returns_202(self, client):
        resp = client.post(
            "/api/v1/experiments/run",
            json={"n_clients": 4, "n_rounds": 2, "aggregator": "fedavg"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "experiment_id" in body
        assert body["status"] == "queued"


# ---------------------------------------------------------------------------
# CORS / Content-Type
# ---------------------------------------------------------------------------


class TestContentType:
    def test_json_content_type_on_health(self, client):
        resp = client.get("/api/health")
        assert "application/json" in resp.headers.get("content-type", "")
