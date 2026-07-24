"""
backend/routers/experiments.py — Experiment endpoints.

Implements API.md §1–3 and §8:
  GET  /experiments                — list all experiments
  GET  /experiments/{id}           — full experiment detail
  GET  /experiments/{id}/rounds    — per-round timeline data
  POST /experiments/run            — trigger a run (local dev only)

All GET endpoints read from the experiments/ directory (pre-computed artefacts).
POST /run is single-run-at-a-time; concurrency is out of scope (API.md §8).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.dependencies import get_experiments_dir
from backend.services.experiment_service import ExperimentService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["experiments"])

# Singleton flag — only one run allowed at a time (API.md §8)
_run_in_progress: bool = False


# ---------------------------------------------------------------------------
# GET /experiments
# ---------------------------------------------------------------------------


@router.get("/experiments")
async def list_experiments(
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List all recorded experiments.

    Returns an empty list if none exist.
    """
    service = ExperimentService(get_experiments_dir())
    experiments = service.list_experiments(limit=limit, offset=offset)
    return {"experiments": experiments}


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}
# ---------------------------------------------------------------------------


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Return full detail for one experiment.

    Raises 404 if the experiment does not exist.
    """
    service = ExperimentService(get_experiments_dir())
    experiment = service.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "experiment_not_found", "experiment_id": experiment_id},
        )
    return experiment


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/rounds
# ---------------------------------------------------------------------------


@router.get("/experiments/{experiment_id}/rounds")
async def get_rounds(experiment_id: str) -> dict[str, Any]:
    """Return per-round timeline data for the reputation heatmap and metric charts."""
    service = ExperimentService(get_experiments_dir())
    if not service.experiment_exists(experiment_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "experiment_not_found", "experiment_id": experiment_id},
        )
    rounds = service.get_rounds(experiment_id)
    return {"rounds": rounds}


# ---------------------------------------------------------------------------
# POST /experiments/run  (local dev only)
# ---------------------------------------------------------------------------


@router.post("/experiments/run", status_code=202)
async def run_experiment(
    config: dict[str, Any], background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """Trigger a new experiment run with the provided configuration.

    Single-run-at-a-time (API.md §8 note on concurrency).
    Returns 409 if a run is already in progress.
    """
    global _run_in_progress

    if _run_in_progress:
        raise HTTPException(
            status_code=409,
            detail={"error": "experiment_already_running"},
        )

    # Basic config validation — full Pydantic validation is done inside the script
    if not config:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_config", "details": ["Request body must not be empty."]},
        )

    import uuid

    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
    logger.info("Queueing experiment run: %s", experiment_id)

    background_tasks.add_task(_run_experiment_bg, experiment_id, config)
    return {"experiment_id": experiment_id, "status": "queued"}


async def _run_experiment_bg(experiment_id: str, config: dict[str, Any]) -> None:
    """Background task: invoke scripts/run_demo.py with the given config."""
    global _run_in_progress
    _run_in_progress = True
    try:
        logger.info("Starting background experiment run: %s", experiment_id)
        # TODO(Milestone 8): wire to the full config-driven run script
        logger.warning("Background run not yet implemented (Milestone 8): %s", experiment_id)
    finally:
        _run_in_progress = False
        logger.info("Background experiment run complete: %s", experiment_id)
