"""
backend/services/experiment_service.py — Reads experiment artefacts from disk.

Experiments are stored as JSON files under experiments/<experiment_id>/:
  - experiment.json   — Experiment schema object
  - rounds.json       — list of TrainingRound objects
  - log.jsonl         — structured JSON-lines log

For Phase 0, the existing experiments/demo_results.json is also served
by mapping its content to the Experiment schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentService:
    """Reads experiment artefacts from the experiments/ directory.

    Args:
        experiments_dir: Root directory where experiment subdirectories live.
    """

    _DEMO_EXPERIMENT_ID = "demo"
    _DEMO_RESULTS_FILE = "demo_results.json"

    def __init__(self, experiments_dir: Path) -> None:
        self._dir = experiments_dir

    def list_experiments(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return all experiments, including the Phase 0 demo results."""
        experiments = []

        # Serve the Phase 0 demo results as a named experiment
        demo = self._load_demo_results()
        if demo:
            experiments.append(demo)

        # Scan for experiment subdirectories
        for subdir in sorted(self._dir.iterdir()):
            if not subdir.is_dir():
                continue
            exp_file = subdir / "experiment.json"
            if exp_file.exists():
                try:
                    with open(exp_file, encoding="utf-8") as fh:
                        experiments.append(json.load(fh))
                except Exception as exc:
                    logger.warning("Failed to load experiment from %s: %s", exp_file, exc)

        return experiments[offset : offset + limit]

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Return one experiment by ID, or None if not found."""
        if experiment_id == self._DEMO_EXPERIMENT_ID:
            return self._load_demo_results()

        exp_file = self._dir / experiment_id / "experiment.json"
        if not exp_file.exists():
            return None
        try:
            with open(exp_file, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load experiment %s: %s", experiment_id, exc)
            return None

    def experiment_exists(self, experiment_id: str) -> bool:
        """Return True if the experiment is known."""
        if experiment_id == self._DEMO_EXPERIMENT_ID:
            return (self._dir / self._DEMO_RESULTS_FILE).exists()
        return (self._dir / experiment_id / "experiment.json").exists()

    def get_rounds(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return per-round data for the given experiment."""
        rounds_file = self._dir / experiment_id / "rounds.json"
        if not rounds_file.exists():
            return []
        try:
            with open(rounds_file, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load rounds for %s: %s", experiment_id, exc)
            return []

    # ------------------------------------------------------------------
    # Phase 0 demo results adapter
    # ------------------------------------------------------------------

    def _load_demo_results(self) -> dict[str, Any] | None:
        """Load the Phase 0 demo_results.json and map to Experiment schema."""
        demo_file = self._dir / self._DEMO_RESULTS_FILE
        if not demo_file.exists():
            return None
        try:
            with open(demo_file, encoding="utf-8") as fh:
                raw = json.load(fh)
            # Build a minimal Experiment-compatible dict from the demo results
            multikrum_guard = raw.get("multikrum+guard", {})
            return {
                "experiment_id": self._DEMO_EXPERIMENT_ID,
                "config_ref": "configs/default.yaml",
                "dataset_phase": "phase0_synthetic",
                "layers_enabled": ["L1", "L3"],
                "attack_config": {
                    "attack_id": "demo_attack",
                    "attack_type": "badnet_colluding",
                    "malicious_client_ids": ["client_02", "client_05", "client_09"],
                    "target_class": 0,
                    "poison_fraction": 0.15,
                    "rounds_active": list(range(20)),
                },
                "result": {
                    "experiment_id": self._DEMO_EXPERIMENT_ID,
                    "clean_accuracy": multikrum_guard.get("clean_accuracy"),
                    "attack_success_rate": multikrum_guard.get("attack_success_rate"),
                    "robust_accuracy": None,
                    "false_acceptance_rate": multikrum_guard.get("strip_frr_on_clean"),
                    "false_rejection_rate": None,
                    "detection_latency_ms": None,
                    "communication_cost_bytes": None,
                    "warnings": [],
                },
                "seeds": {"global": 42},
                "_raw": raw,  # expose raw multi-strategy results for the dashboard
            }
        except Exception as exc:
            logger.warning("Failed to load demo results: %s", exc)
            return None
