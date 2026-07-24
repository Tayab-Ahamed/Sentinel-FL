"""
backend/routers/remediation.py — L5 Remediation endpoints.

  GET /experiments/{id}/remediation        — remediation report for an experiment
  GET /remediation/manual-review           — all reports needing manual review

Read-only artefact serving, consistent with the experiments/visualizer routers.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.dependencies import get_experiments_dir
from backend.services.remediation_service import RemediationService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["remediation"])


@router.get("/experiments/{experiment_id}/remediation")
async def get_remediation(experiment_id: str) -> dict[str, Any]:
    """Return the remediation report for one experiment.

    Raises 404 if no remediation artefact exists for the experiment.
    """
    service = RemediationService(get_experiments_dir())
    report = service.get_report(experiment_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "remediation_not_found", "experiment_id": experiment_id},
        )
    return {"experiment_id": experiment_id, "remediation": report}


@router.get("/remediation/manual-review")
async def list_manual_review() -> dict[str, Any]:
    """Return every remediation report flagged ``manual_review_required``."""
    service = RemediationService(get_experiments_dir())
    pending = service.list_manual_review()
    return {"count": len(pending), "reports": pending}
