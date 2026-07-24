"""
backend/services/remediation_service.py — Serves L5 remediation artefacts.

Remediation runs write a ``RemediationReport`` (ai/fl_core/schemas.py) to
``experiments/<experiment_id>/remediation.json`` (or the top-level
``experiments/remediation_results.json`` produced by
``scripts/run_remediation_demo.py``).  This service exposes those reports to the
dashboard's remediation panel and manual-review queue.

Read-only and defensive: a missing or malformed artefact yields ``None`` / ``[]``
rather than raising, matching ``ExperimentService`` conventions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RemediationService:
    """Reads remediation reports from the experiments/ directory.

    Args:
        experiments_dir: Root directory where experiment artefacts live.
    """

    _DEMO_FILE = "remediation_results.json"

    def __init__(self, experiments_dir: Path) -> None:
        self._dir = experiments_dir

    def get_report(self, experiment_id: str) -> dict[str, Any] | None:
        """Return the remediation report for one experiment, or None."""
        if experiment_id == "demo":
            return self._load_json(self._dir / self._DEMO_FILE)
        return self._load_json(self._dir / experiment_id / "remediation.json")

    def list_manual_review(self) -> list[dict[str, Any]]:
        """Return every remediation report currently flagged for manual review.

        Scans the demo artefact plus each ``experiments/<id>/remediation.json``.
        """
        pending: list[dict[str, Any]] = []
        for report in self._iter_reports():
            if report.get("manual_review_required"):
                pending.append(report)
        return pending

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_reports(self):
        demo = self._load_json(self._dir / self._DEMO_FILE)
        if demo:
            yield from self._as_reports(demo)
        if not self._dir.exists():
            return
        for subdir in sorted(self._dir.iterdir()):
            if not subdir.is_dir():
                continue
            data = self._load_json(subdir / "remediation.json")
            if data:
                yield from self._as_reports(data)

    @staticmethod
    def _as_reports(data: Any):
        """Normalise a loaded artefact into an iterable of report dicts."""
        if isinstance(data, list):
            yield from (d for d in data if isinstance(d, dict))
        elif isinstance(data, dict):
            # run_remediation_demo.py writes {"reports": [...], ...}
            if isinstance(data.get("reports"), list):
                yield from (d for d in data["reports"] if isinstance(d, dict))
            else:
                yield data

    @staticmethod
    def _load_json(path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load remediation artefact %s: %s", path, exc)
            return None
