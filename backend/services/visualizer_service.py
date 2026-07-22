"""
backend/services/visualizer_service.py — Builds visualisation JSON for the dashboard.

Implements the data transformations backing Visualizer endpoints:
  - reputation_heatmap: client × round trust-score matrix
  - metric_timeseries: per-round metric values
  - audit_report: L2 reversed triggers + anomaly scores
  - explainability_drilldown: Trust Ledger entry detail
  - alerts: alert events from log.jsonl
  - clients: client list with trust scores and flag counts
  - experiment_config: parsed configuration for the config viewer

All methods return None on not-found (routers convert to 404).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric field → friendly display name
# ---------------------------------------------------------------------------
_METRIC_DISPLAY = {
    "clean_accuracy": "Clean Accuracy",
    "attack_success_rate": "Attack Success Rate",
    "robust_accuracy": "Robust Accuracy",
    "false_acceptance_rate": "False Acceptance Rate",
    "false_rejection_rate": "False Rejection Rate",
    "detection_latency_ms": "Detection Latency (ms)",
    "communication_cost_bytes": "Communication Cost (bytes)",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score",
    "false_positive_rate": "False Positive Rate",
    "runtime_seconds": "Runtime (s)",
    "peak_memory_mb": "Peak Memory (MB)",
}


class VisualizerService:
    """Builds JSON payloads for the React dashboard.

    Args:
        experiments_dir: Root experiments directory.
    """

    def __init__(self, experiments_dir: Path) -> None:
        self._dir = experiments_dir

    # ------------------------------------------------------------------
    # Reputation heatmap  (API.md §4)
    # ------------------------------------------------------------------

    def reputation_heatmap(self, experiment_id: str) -> dict[str, Any]:
        """Return client × round trust-score matrix.

        Reads trust_ledger.jsonl (if present) and builds the matrix.
        Falls back to an empty matrix with a note if the file is missing.
        """
        ledger_file = self._dir / experiment_id / "trust_ledger.jsonl"
        if not ledger_file.exists():
            # Try to synthesise from rounds data
            return self._heatmap_from_rounds(experiment_id)

        # Parse ledger entries — build {client_id: {round_num: score}}
        client_rounds: dict[str, dict[int, float]] = {}
        try:
            with open(ledger_file, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("subject_type") != "client":
                        continue
                    cid = entry.get("subject_id", "unknown")
                    rnd = entry.get("round_num", 0) or 0
                    score = float(entry.get("score", 0.0))
                    if cid not in client_rounds:
                        client_rounds[cid] = {}
                    # Worst score wins if multiple entries per round
                    client_rounds[cid][rnd] = max(
                        client_rounds[cid].get(rnd, 0.0), score
                    )
        except Exception as exc:
            logger.warning("Failed to read trust_ledger for %s: %s", experiment_id, exc)
            return {"client_ids": [], "rounds": [], "scores": []}

        if not client_rounds:
            return {"client_ids": [], "rounds": [], "scores": []}

        all_rounds = sorted({r for cr in client_rounds.values() for r in cr})
        client_ids = sorted(client_rounds.keys())
        scores = [
            [client_rounds[cid].get(r, 0.0) for r in all_rounds]
            for cid in client_ids
        ]
        return {"client_ids": client_ids, "rounds": all_rounds, "scores": scores}

    # ------------------------------------------------------------------
    # Metric timeseries  (API.md §5)
    # ------------------------------------------------------------------

    def metric_timeseries(
        self, experiment_id: str, metric_names: list[str]
    ) -> dict[str, Any]:
        """Return per-round time-series data for the requested metrics."""
        rounds = self._load_rounds(experiment_id)
        series: dict[str, list[dict[str, Any]]] = {name: [] for name in metric_names}

        for rnd in rounds:
            round_num = rnd.get("round_num", rnd.get("round", 0))
            for name in metric_names:
                value = rnd.get(name)
                if value is not None:
                    series[name].append({
                        "metric_name": name,
                        "round_num": round_num,
                        "value": float(value),
                    })

        # Also try to read from log.jsonl round_complete events
        if all(len(v) == 0 for v in series.values()):
            log_path = self._dir / experiment_id / "log.jsonl"
            series = self._timeseries_from_log(log_path, metric_names)

        return {
            "series": series,
            "display_names": {
                name: _METRIC_DISPLAY.get(name, name) for name in metric_names
            },
        }

    # ------------------------------------------------------------------
    # Audit report  (API.md §6)
    # ------------------------------------------------------------------

    def audit_report(self, experiment_id: str, round_num: int) -> dict[str, Any] | None:
        """Return the L2 audit report for a given round."""
        audit_file = self._dir / experiment_id / "audits" / f"round_{round_num}.json"
        if not audit_file.exists():
            return None
        try:
            with open(audit_file, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning(
                "Failed to load audit report %s round %d: %s", experiment_id, round_num, exc
            )
            return None

    # ------------------------------------------------------------------
    # Explainability drilldown  (API.md §7)
    # ------------------------------------------------------------------

    def explainability_drilldown(self, entry_id: str) -> dict[str, Any] | None:
        """Return the Trust Ledger entry detail for one flag."""
        for ledger_file in self._dir.rglob("trust_ledger.jsonl"):
            try:
                with open(ledger_file, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if entry.get("entry_id") == entry_id:
                            return entry
            except Exception as exc:
                logger.warning("Error reading ledger file %s: %s", ledger_file, exc)
        return None

    # ------------------------------------------------------------------
    # Alerts  (new — M10)
    # ------------------------------------------------------------------

    def alerts(self, experiment_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Return alert events from log.jsonl for the dashboard alerts page.

        Parses log.jsonl for events of types: client_excluded, cluster_flagged,
        input_flagged, inference_scored (flagged=True), and audit_flagged.

        Returns a list of alert dicts with: id, round_num, layer_id, severity,
        event_type, subject_id, message, timestamp.
        """
        log_path = self._dir / experiment_id / "log.jsonl"
        alerts: list[dict[str, Any]] = []

        _alert_event_types = {
            "client_excluded",
            "cluster_flagged",
            "input_flagged",
            "audit_flagged",
        }

        if not log_path.exists():
            return []

        try:
            with open(log_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("event_type", "")
                    payload = event.get("payload", {})

                    # inference_scored: only include if flagged
                    if event_type == "inference_scored":
                        if not payload.get("flagged", False):
                            continue
                        event_type = "input_flagged"

                    if event_type not in _alert_event_types:
                        continue

                    layer = event.get("layer_id", payload.get("layer_id", "L1"))
                    severity = self._alert_severity(event_type, payload)
                    subject = (
                        payload.get("client_id")
                        or payload.get("input_id")
                        or payload.get("label")
                        or "—"
                    )
                    message = self._alert_message(event_type, payload, layer)

                    alerts.append({
                        "id": f"{experiment_id}_{len(alerts)}",
                        "round_num": event.get("round_num") or payload.get("round_num"),
                        "layer_id": layer,
                        "severity": severity,
                        "event_type": event_type,
                        "subject_id": str(subject),
                        "message": message,
                        "timestamp": event.get("timestamp", ""),
                    })

        except Exception as exc:
            logger.warning("Failed to read alerts for %s: %s", experiment_id, exc)

        return alerts[-limit:]

    # ------------------------------------------------------------------
    # Clients  (new — M10)
    # ------------------------------------------------------------------

    def clients(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return client list with trust scores and flag counts.

        Reads trust_ledger.jsonl (if present) to compute per-client stats.
        Falls back to reading from rounds data.
        """
        # Build client stats from ledger
        client_stats: dict[str, dict[str, Any]] = {}

        ledger_file = self._dir / experiment_id / "trust_ledger.jsonl"
        if ledger_file.exists():
            try:
                with open(ledger_file, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if entry.get("subject_type") != "client":
                            continue
                        cid = entry.get("subject_id", "unknown")
                        if cid not in client_stats:
                            client_stats[cid] = {
                                "client_id": cid,
                                "trust_score": 0.0,
                                "flag_count": 0,
                                "layers_flagged": set(),
                                "last_round": 0,
                                "status": "active",
                            }
                        s = client_stats[cid]
                        s["flag_count"] += 1
                        s["trust_score"] = max(s["trust_score"], float(entry.get("score", 0.0)))
                        s["layers_flagged"].add(entry.get("layer_id", "?"))
                        s["last_round"] = max(s["last_round"], entry.get("round_num") or 0)
            except Exception as exc:
                logger.warning("Failed to read ledger for clients: %s", exc)

        # Convert sets to lists for JSON serialisation
        result = []
        for stats in client_stats.values():
            stats["layers_flagged"] = sorted(stats["layers_flagged"])
            stats["is_suspicious"] = stats["trust_score"] >= 0.5
            result.append(stats)

        # If no ledger data, try to infer from rounds
        if not result:
            result = self._clients_from_rounds(experiment_id)

        return sorted(result, key=lambda c: c["trust_score"], reverse=True)

    # ------------------------------------------------------------------
    # Experiment config  (new — M10)
    # ------------------------------------------------------------------

    def experiment_config(self, experiment_id: str) -> dict[str, Any] | None:
        """Return the configuration used for this experiment."""
        exp_file = self._dir / experiment_id / "experiment.json"
        if exp_file.exists():
            try:
                with open(exp_file, encoding="utf-8") as fh:
                    exp = json.load(fh)
                return exp.get("config", exp.get("attack_config", {}))
            except Exception as exc:
                logger.warning("Failed to load config for %s: %s", experiment_id, exc)

        # Demo experiment: load from default.yaml if available
        config_file = self._dir.parent / "configs" / "default.yaml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file, encoding="utf-8") as fh:
                    return yaml.safe_load(fh)
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_rounds(self, experiment_id: str) -> list[dict[str, Any]]:
        rounds_file = self._dir / experiment_id / "rounds.json"
        if not rounds_file.exists():
            return []
        try:
            with open(rounds_file, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load rounds for %s: %s", experiment_id, exc)
            return []

    def _timeseries_from_log(
        self, log_path: Path, metric_names: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        series: dict[str, list[dict[str, Any]]] = {name: [] for name in metric_names}
        if not log_path.exists():
            return series
        try:
            with open(log_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event_type") != "round_complete":
                        continue
                    payload = event.get("payload", {})
                    round_num = event.get("round_num") or payload.get("round_num", 0)
                    for name in metric_names:
                        value = payload.get(name)
                        if value is not None:
                            series[name].append({
                                "metric_name": name,
                                "round_num": round_num,
                                "value": float(value),
                            })
        except Exception as exc:
            logger.warning("Failed to read log for timeseries: %s", exc)
        return series

    def _heatmap_from_rounds(self, experiment_id: str) -> dict[str, Any]:
        """Synthesise a minimal heatmap from rounds data when no ledger exists."""
        rounds = self._load_rounds(experiment_id)
        if not rounds:
            return {"client_ids": [], "rounds": [], "scores": [], "_note": "No ledger data."}

        all_clients: set[str] = set()
        for r in rounds:
            all_clients.update(r.get("participating_clients", []))
            all_clients.update(r.get("excluded_clients", []))

        client_ids = sorted(all_clients)
        round_nums = [r.get("round_num", i) for i, r in enumerate(rounds)]
        scores = []
        for cid in client_ids:
            row = []
            for r in rounds:
                excluded = r.get("excluded_clients", [])
                row.append(0.8 if cid in excluded else 0.1)
            scores.append(row)
        return {"client_ids": client_ids, "rounds": round_nums, "scores": scores}

    def _clients_from_rounds(self, experiment_id: str) -> list[dict[str, Any]]:
        """Infer client list from rounds data."""
        rounds = self._load_rounds(experiment_id)
        client_flags: dict[str, int] = {}
        for r in rounds:
            for cid in r.get("excluded_clients", []):
                client_flags[cid] = client_flags.get(cid, 0) + 1
            for cid in r.get("participating_clients", []):
                if cid not in client_flags:
                    client_flags[cid] = 0
        total_rounds = max(len(rounds), 1)
        return [
            {
                "client_id": cid,
                "trust_score": min(1.0, flags / total_rounds),
                "flag_count": flags,
                "layers_flagged": ["L1"] if flags > 0 else [],
                "last_round": total_rounds,
                "status": "suspicious" if flags > 2 else "active",
                "is_suspicious": flags > 2,
            }
            for cid, flags in sorted(client_flags.items(), key=lambda x: -x[1])
        ]

    @staticmethod
    def _alert_severity(event_type: str, payload: dict[str, Any]) -> str:
        if event_type == "cluster_flagged":
            return "high"
        if event_type == "client_excluded":
            return "high" if payload.get("is_malicious") else "medium"
        if event_type == "audit_flagged":
            return "high"
        return "medium"

    @staticmethod
    def _alert_message(event_type: str, payload: dict[str, Any], layer: str) -> str:
        if event_type == "client_excluded":
            cid = payload.get("client_id", "unknown")
            return f"[{layer}] Client {cid} excluded — {payload.get('reason', 'anomalous update')}"
        if event_type == "cluster_flagged":
            clients = payload.get("client_ids", [])
            return f"[{layer}] Collusion cluster detected: {', '.join(str(c) for c in clients[:4])}"
        if event_type == "input_flagged":
            return f"[{layer}] Backdoored input detected (confidence={payload.get('confidence', '?')})"
        if event_type == "audit_flagged":
            label = payload.get("label", "?")
            return f"[{layer}] L2 audit flagged label {label}"
        return f"[{layer}] {event_type}"
