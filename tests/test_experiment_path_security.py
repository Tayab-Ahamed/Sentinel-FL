"""Experiment APIs must not read artifacts selected outside their configured root."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from backend.services.experiment_service import ExperimentService
from backend.services.remediation_service import RemediationService


def write_artifacts(directory: Path, experiment_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment.json").write_text(
        json.dumps({"experiment_id": experiment_id, "layers_enabled": ["L1"]}),
        encoding="utf-8",
    )
    (directory / "rounds.json").write_text(
        json.dumps([{"round_num": 7, "experiment_id": experiment_id}]),
        encoding="utf-8",
    )
    (directory / "remediation.json").write_text(
        json.dumps({"experiment_id": experiment_id, "manual_review_required": True}),
        encoding="utf-8",
    )


@pytest.fixture
def artifact_tree(tmp_path):
    root = tmp_path / "experiments"
    write_artifacts(root, "ROOT_DOCUMENT")
    write_artifacts(tmp_path, "PARENT_DOCUMENT")
    write_artifacts(tmp_path / "outside", "OUTSIDE_DOCUMENT")
    write_artifacts(root / "nested" / "child", "NESTED_DOCUMENT")
    write_artifacts(root / "exp_valid-1.v2", "exp_valid-1.v2")
    return root


@pytest.mark.parametrize("escape", ["../outside", r"..\outside", ".", "..", "", "nested/child", r"nested\child", "absolute"])
@pytest.mark.parametrize(
    ("method", "missing"), [("get_experiment", None), ("experiment_exists", False), ("get_rounds", [])]
)
def test_unsafe_ids_cannot_read_real_artifacts(artifact_tree, escape, method, missing):
    experiment_id = (
        str((artifact_tree.parent / "outside").resolve()) if escape == "absolute" else escape
    )
    service = ExperimentService(artifact_tree)

    assert getattr(service, method)(experiment_id) == missing


@pytest.mark.parametrize("escape", ["../outside", r"..\outside", ".", "..", "", "nested/child", r"nested\child", "absolute"])
def test_unsafe_remediation_ids_cannot_read_real_reports(artifact_tree, escape):
    experiment_id = (
        str((artifact_tree.parent / "outside").resolve()) if escape == "absolute" else escape
    )
    service = RemediationService(artifact_tree)

    assert service.get_report(experiment_id) is None


def test_valid_experiment_and_rounds_still_load(artifact_tree):
    service = ExperimentService(artifact_tree)

    assert service.get_experiment("exp_valid-1.v2")["experiment_id"] == "exp_valid-1.v2"
    assert service.experiment_exists("exp_valid-1.v2")
    assert service.get_rounds("exp_valid-1.v2") == [
        {"round_num": 7, "experiment_id": "exp_valid-1.v2"}
    ]
    assert [entry["experiment_id"] for entry in service.list_experiments()] == ["exp_valid-1.v2"]


def test_demo_adapter_still_loads(artifact_tree):
    (artifact_tree / "demo_results.json").write_text(
        json.dumps({"multikrum+guard": {"clean_accuracy": 0.92, "attack_success_rate": 0.04}}),
        encoding="utf-8",
    )
    service = ExperimentService(artifact_tree)

    assert service.experiment_exists("demo")
    assert service.get_experiment("demo")["result"]["clean_accuracy"] == 0.92


def test_linked_experiment_cannot_escape_root(artifact_tree):
    outside = artifact_tree.parent / "outside"
    link = artifact_tree / "linked"
    if os.name == "nt":
        # Directory junctions do not require Windows' optional symlink privilege.
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(outside, target_is_directory=True)
    service = ExperimentService(artifact_tree)

    assert service.get_experiment("linked") is None
    assert not service.experiment_exists("linked")
    assert service.get_rounds("linked") == []
    assert "OUTSIDE_DOCUMENT" not in {
        item["experiment_id"] for item in service.list_experiments()
    }
    remediation = RemediationService(artifact_tree)
    assert remediation.get_report("linked") is None
    assert "OUTSIDE_DOCUMENT" not in {
        item["experiment_id"] for item in remediation.list_manual_review()
    }


def test_encoded_backslash_traversal_is_not_exposed_by_api(artifact_tree, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app

    monkeypatch.setattr(
        "backend.routers.experiments.get_experiments_dir", lambda: artifact_tree
    )
    monkeypatch.setattr(
        "backend.routers.visualizer.get_experiments_dir", lambda: artifact_tree
    )
    monkeypatch.setattr(
        "backend.routers.remediation.get_experiments_dir", lambda: artifact_tree
    )
    with TestClient(app) as client:
        detail = client.get("/api/v1/experiments/..%5Coutside")
        rounds = client.get("/api/v1/experiments/..%5Coutside/rounds")
        metrics = client.get("/api/v1/experiments/..%5Coutside/metrics?names=clean_accuracy")
        remediation = client.get("/api/v1/experiments/..%5Coutside/remediation")

    assert detail.status_code == 404
    assert rounds.status_code == 404
    assert metrics.status_code == 404
    assert remediation.status_code == 404
