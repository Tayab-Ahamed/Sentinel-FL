"""
backend/routers/visualizer.py — Visualizer endpoints.

Implements API.md §4–7 plus M10 dashboard endpoints:
  GET /experiments/{id}/reputation-heatmap
  GET /experiments/{id}/metrics?names=...
  GET /experiments/{id}/audits/{round_num}
  GET /trust-ledger/{entry_id}
  GET /experiments/{id}/alerts         (M10)
  GET /experiments/{id}/clients        (M10)
  GET /experiments/{id}/config         (M10)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.dependencies import get_experiments_dir
from backend.services.experiment_service import ExperimentService
from backend.services.visualizer_service import VisualizerService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["visualizer"])

KNOWN_METRICS = {
    "clean_accuracy",
    "attack_success_rate",
    "robust_accuracy",
    "false_acceptance_rate",
    "false_rejection_rate",
    "detection_latency_ms",
    "communication_cost_bytes",
    # M9 additions
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "runtime_seconds",
    "peak_memory_mb",
}


# ---------------------------------------------------------------------------
# GET /experiments/{id}/reputation-heatmap  (API.md §4)
# ---------------------------------------------------------------------------


@router.get("/experiments/{experiment_id}/reputation-heatmap")
async def reputation_heatmap(experiment_id: str) -> dict[str, Any]:
    """Return client × round trust-score matrix.

    Requires L1 to be enabled for this experiment.
    """
    exp_service = ExperimentService(get_experiments_dir())
    experiment = exp_service.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "experiment_not_found", "experiment_id": experiment_id},
        )
    if "L1" not in experiment.get("layers_enabled", []):
        raise HTTPException(
            status_code=400,
            detail={"error": "layer_not_enabled", "layer": "L1"},
        )

    viz = VisualizerService(get_experiments_dir())
    return viz.reputation_heatmap(experiment_id)


# ---------------------------------------------------------------------------
# GET /experiments/{id}/metrics  (API.md §5)
# ---------------------------------------------------------------------------


@router.get("/experiments/{experiment_id}/metrics")
async def metric_timeseries(
    experiment_id: str,
    names: str = Query(..., description="Comma-separated metric names"),
) -> dict[str, Any]:
    """Return time-series data for the requested metrics."""
    exp_service = ExperimentService(get_experiments_dir())
    if not exp_service.experiment_exists(experiment_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "experiment_not_found", "experiment_id": experiment_id},
        )

    requested = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in requested if n not in KNOWN_METRICS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail={"error": "unknown_metric", "name": unknown[0]},
        )

    viz = VisualizerService(get_experiments_dir())
    return viz.metric_timeseries(experiment_id, requested)


# ---------------------------------------------------------------------------
# GET /experiments/{id}/audits/{round_num}  (API.md §6)
# ---------------------------------------------------------------------------


@router.get("/experiments/{experiment_id}/audits/{round_num}")
async def audit_report(experiment_id: str, round_num: int) -> dict[str, Any]:
    """Return the L2 audit report for a given audited round."""
    exp_service = ExperimentService(get_experiments_dir())
    if not exp_service.experiment_exists(experiment_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "experiment_not_found", "experiment_id": experiment_id},
        )

    viz = VisualizerService(get_experiments_dir())
    report = viz.audit_report(experiment_id, round_num)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "audit_not_found", "round_num": round_num},
        )
    return report


# ---------------------------------------------------------------------------
# GET /trust-ledger/{entry_id}  (API.md §7)
# ---------------------------------------------------------------------------


@router.get("/trust-ledger/{entry_id}")
async def trust_ledger_entry(entry_id: str) -> dict[str, Any]:
    """Return the human-readable reason and raw evidence for a specific flag."""
    viz = VisualizerService(get_experiments_dir())
    entry = viz.explainability_drilldown(entry_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "entry_not_found"},
        )
    return entry


# ---------------------------------------------------------------------------
# GET /experiments/{id}/alerts  (M10)
# ---------------------------------------------------------------------------


@router.get("/experiments/{experiment_id}/alerts")
async def experiment_alerts(
    experiment_id: str,
    limit: int = 200,
) -> dict[str, Any]:
    """Return detection alert events for the dashboard alerts page.

    Parses log.jsonl for: client_excluded, cluster_flagged, input_flagged,
    inference_scored (flagged=True), audit_flagged.
    """
    exp_service = ExperimentService(get_experiments_dir())
    if not exp_service.experiment_exists(experiment_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "experiment_not_found", "experiment_id": experiment_id},
        )

    viz = VisualizerService(get_experiments_dir())
    alerts = viz.alerts(experiment_id, limit=limit)
    return {"alerts": alerts, "count": len(alerts)}


# ---------------------------------------------------------------------------
# GET /experiments/{id}/clients  (M10)
# ---------------------------------------------------------------------------


@router.get("/experiments/{experiment_id}/clients")
async def experiment_clients(experiment_id: str) -> dict[str, Any]:
    """Return client list with trust scores, flag counts, and status."""
    exp_service = ExperimentService(get_experiments_dir())
    if not exp_service.experiment_exists(experiment_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "experiment_not_found", "experiment_id": experiment_id},
        )

    viz = VisualizerService(get_experiments_dir())
    clients = viz.clients(experiment_id)
    return {"clients": clients, "count": len(clients)}


# ---------------------------------------------------------------------------
# GET /experiments/{id}/config  (M10)
# ---------------------------------------------------------------------------


@router.get("/experiments/{experiment_id}/config")
async def experiment_config(experiment_id: str) -> dict[str, Any]:
    """Return the configuration used for this experiment."""
    exp_service = ExperimentService(get_experiments_dir())
    if not exp_service.experiment_exists(experiment_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "experiment_not_found", "experiment_id": experiment_id},
        )

    viz = VisualizerService(get_experiments_dir())
    config = viz.experiment_config(experiment_id)
    return {"config": config or {}}
