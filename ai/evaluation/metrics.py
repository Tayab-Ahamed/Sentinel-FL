"""
ai/evaluation/metrics.py — MetricsCollector implementation (Milestone 9).

Reads structured JSON-lines logs produced by StructuredLogger and computes
the full metric set (SCHEMAS.md §EvaluationResult, BENCHMARK.md §4):

  Classification metrics (from ``round_complete`` events):
    - Clean Accuracy (C-Acc)
    - Attack Success Rate (ASR)
    - Robust Accuracy (R-Acc)

  Detection metrics (from ``client_excluded``, ``cluster_flagged``,
                     ``input_flagged`` events):
    - Precision, Recall, F1
    - False Positive Rate (FPR)
    - False Acceptance Rate / False Rejection Rate (L3)

  Operational metrics:
    - Detection latency (ms/input from ``inference_scored`` events)
    - Communication cost (bytes from ``communication`` events)

Missing log events → EvaluationResult fields are null with a warnings list
(ARCHITECTURE.md §7.9 — preserved from skeleton).

Event types consumed from log.jsonl:
  ``round_complete``     payload: {clean_accuracy, attack_success_rate,
                                   participating_clients, excluded_clients}
  ``client_excluded``    payload: {client_id, layer_id, is_malicious (opt)}
  ``cluster_flagged``    payload: {client_ids, is_malicious (opt)}
  ``input_flagged``      payload: {input_id, flagged, is_triggered (opt),
                                   latency_ms}
  ``inference_scored``   payload: {latency_ms, flagged, is_triggered (opt)}
  ``communication``      payload: {delta_len, dtype_bytes (opt)}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ai.evaluation.metrics_engine import (
    communication_cost_bytes,
    delta_byte_size,
    false_acceptance_rate,
    false_positive_rate,
    false_rejection_rate,
    precision_recall_f1,
)
from ai.fl_core.interfaces import MetricsCollector
from ai.fl_core.schemas import EvaluationResult

logger = logging.getLogger(__name__)


class JsonLinesMetricsCollector(MetricsCollector):
    """MetricsCollector that reads JSON-lines experiment logs.

    Args:
        experiments_dir: Root directory where experiment logs are stored.
            Each experiment has a subdirectory named by ``experiment_id``
            containing a ``log.jsonl`` file.
    """

    def __init__(self, experiments_dir: str | Path = "experiments") -> None:
        self._dir = Path(experiments_dir)

    def compute(self, experiment_id: str) -> EvaluationResult:
        """Compute all metrics for a completed experiment.

        Reads ``experiments/<experiment_id>/log.jsonl``, extracts metric
        events by ``event_type``, and returns a fully populated
        EvaluationResult.

        Missing events → null fields with warnings (ARCHITECTURE.md §7.9).

        Args:
            experiment_id: Identifier of the experiment to evaluate.

        Returns:
            EvaluationResult with all computable metrics filled in.
        """
        log_path = self._dir / experiment_id / "log.jsonl"
        warnings: list[str] = []

        if not log_path.exists():
            logger.warning(
                "No log file found for experiment '%s' at %s", experiment_id, log_path
            )
            warnings.append(f"Log file not found: {log_path}")
            return EvaluationResult(experiment_id=experiment_id, warnings=warnings)

        events = self._load_events(log_path)
        if not events:
            warnings.append("Log file is empty or contains no parseable events.")
            return EvaluationResult(experiment_id=experiment_id, warnings=warnings)

        logger.info(
            "JsonLinesMetricsCollector: loaded %d events for experiment '%s'.",
            len(events),
            experiment_id,
        )

        # ----------------------------------------------------------------
        # Partition events by type
        # ----------------------------------------------------------------
        round_events = self._filter(events, "round_complete")
        l1_flag_events = self._filter(events, "client_excluded") + self._filter(
            events, "cluster_flagged"
        )
        l3_events = self._filter(events, "input_flagged") + self._filter(
            events, "inference_scored"
        )
        comm_events = self._filter(events, "communication")

        # ----------------------------------------------------------------
        # Clean accuracy (mean over rounds)
        # ----------------------------------------------------------------
        clean_accuracy: float | None = None
        c_acc_values = [
            float(e["payload"]["clean_accuracy"])
            for e in round_events
            if "clean_accuracy" in e.get("payload", {})
        ]
        if c_acc_values:
            clean_accuracy = round(sum(c_acc_values) / len(c_acc_values), 6)
        else:
            warnings.append("No 'clean_accuracy' in round_complete events.")

        # ----------------------------------------------------------------
        # Attack Success Rate (mean over rounds)
        # ----------------------------------------------------------------
        asr: float | None = None
        asr_values = [
            float(e["payload"]["attack_success_rate"])
            for e in round_events
            if "attack_success_rate" in e.get("payload", {})
        ]
        if asr_values:
            asr = round(sum(asr_values) / len(asr_values), 6)
        else:
            warnings.append("No 'attack_success_rate' in round_complete events.")

        # ----------------------------------------------------------------
        # Robust accuracy (C-Acc × (1 − ASR))
        # ----------------------------------------------------------------
        r_acc: float | None = None
        if clean_accuracy is not None and asr is not None:
            r_acc = round(clean_accuracy * (1.0 - asr), 6)

        # ----------------------------------------------------------------
        # Precision / Recall / F1 / FPR  (from flag events with ground truth)
        # ----------------------------------------------------------------
        prec: float | None = None
        rec: float | None = None
        f1: float | None = None
        fpr: float | None = None

        gt_flags, pred_flags = self._extract_flag_labels(l1_flag_events)
        if gt_flags and len(gt_flags) >= 2:
            prec, rec, f1 = precision_recall_f1(gt_flags, pred_flags, pos_label=1)
            fpr = false_positive_rate(gt_flags, pred_flags, neg_label=0)
        else:
            warnings.append(
                "Insufficient ground-truth flag data for precision/recall/F1/FPR "
                "(need 'is_malicious' field in client_excluded/cluster_flagged events)."
            )

        # ----------------------------------------------------------------
        # L3 FAR / FRR / latency
        # ----------------------------------------------------------------
        far: float | None = None
        frr: float | None = None
        latency: float | None = None

        n_triggered, n_flagged_triggered, n_clean, n_flagged_clean, latencies = (
            self._extract_l3_stats(l3_events)
        )
        if n_triggered > 0 or n_clean > 0:
            if n_triggered > 0:
                far = round(false_acceptance_rate(n_triggered, n_flagged_triggered), 6)
            else:
                warnings.append("No triggered L3 events found — FAR not computed.")
            if n_clean > 0:
                frr = round(false_rejection_rate(n_clean, n_flagged_clean), 6)
            else:
                warnings.append("No clean L3 events found — FRR not computed.")
        else:
            warnings.append("No L3 inference events found — FAR/FRR not computed.")

        if latencies:
            latency = round(sum(latencies) / len(latencies), 4)
        else:
            warnings.append("No latency data in L3 events.")

        # ----------------------------------------------------------------
        # Communication cost
        # ----------------------------------------------------------------
        comm_cost: int | None = None
        byte_sizes = [
            delta_byte_size(
                int(e["payload"].get("delta_len", 0)),
                int(e["payload"].get("dtype_bytes", 4)),
            )
            for e in comm_events
            if e.get("payload", {}).get("delta_len")
        ]
        if byte_sizes:
            comm_cost = communication_cost_bytes(byte_sizes)
        else:
            # Fallback: estimate from excluded-client count and round events
            n_rounds = len(round_events)
            n_clients_mean = self._mean_participating(round_events)
            delta_len_mean = self._mean_delta_len(round_events)
            if n_rounds > 0 and n_clients_mean > 0 and delta_len_mean > 0:
                comm_cost = int(n_rounds * n_clients_mean * delta_byte_size(int(delta_len_mean)))
                warnings.append(
                    "Communication cost estimated from round participation "
                    "(no 'communication' events found)."
                )
            else:
                warnings.append("Communication cost could not be computed.")

        return EvaluationResult(
            experiment_id=experiment_id,
            clean_accuracy=clean_accuracy,
            attack_success_rate=asr,
            robust_accuracy=r_acc,
            false_acceptance_rate=far,
            false_rejection_rate=frr,
            detection_latency_ms=latency,
            communication_cost_bytes=comm_cost,
            precision=round(prec, 6) if prec is not None else None,
            recall=round(rec, 6) if rec is not None else None,
            f1_score=round(f1, 6) if f1 is not None else None,
            false_positive_rate=round(fpr, 6) if fpr is not None else None,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_events(self, log_path: Path) -> list[dict[str, Any]]:
        """Parse a JSON-lines log file into a list of event dicts."""
        events: list[dict[str, Any]] = []
        with open(log_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed log line %d: %s", lineno, exc)
        return events

    def _filter(
        self, events: list[dict[str, Any]], event_type: str
    ) -> list[dict[str, Any]]:
        """Return events matching a given event_type."""
        return [e for e in events if e.get("event_type") == event_type]

    def _extract_flag_labels(
        self, flag_events: list[dict[str, Any]]
    ) -> tuple[list[int], list[int]]:
        """Extract (y_true, y_pred) binary labels from L1 flag events.

        Only events that include ``is_malicious`` in their payload are used.
        ``y_pred = 1`` always (the event means a flag was raised).
        ``y_true = 1`` if ``is_malicious=True``, else ``0``.
        """
        y_true: list[int] = []
        y_pred: list[int] = []
        for event in flag_events:
            payload = event.get("payload", {})
            if "is_malicious" in payload:
                y_true.append(1 if payload["is_malicious"] else 0)
                y_pred.append(1)
        return y_true, y_pred

    def _extract_l3_stats(
        self, l3_events: list[dict[str, Any]]
    ) -> tuple[int, int, int, int, list[float]]:
        """Extract L3 statistics from inference events.

        Returns:
            (n_triggered, n_flagged_triggered, n_clean, n_flagged_clean, latencies)
        """
        n_triggered = 0
        n_flagged_triggered = 0
        n_clean = 0
        n_flagged_clean = 0
        latencies: list[float] = []

        for event in l3_events:
            payload = event.get("payload", {})
            flagged = bool(payload.get("flagged", False))
            is_triggered = payload.get("is_triggered")
            lat = payload.get("latency_ms")

            if lat is not None:
                latencies.append(float(lat))

            if is_triggered is True:
                n_triggered += 1
                if flagged:
                    n_flagged_triggered += 1
            elif is_triggered is False:
                n_clean += 1
                if flagged:
                    n_flagged_clean += 1

        return n_triggered, n_flagged_triggered, n_clean, n_flagged_clean, latencies

    def _mean_participating(self, round_events: list[dict[str, Any]]) -> float:
        """Mean number of participating clients across rounds."""
        counts = [
            len(e.get("payload", {}).get("participating_clients", []))
            for e in round_events
            if "participating_clients" in e.get("payload", {})
        ]
        return float(sum(counts) / len(counts)) if counts else 0.0

    def _mean_delta_len(self, round_events: list[dict[str, Any]]) -> float:
        """Mean delta length from round events (if present)."""
        lengths = [
            float(e.get("payload", {}).get("delta_len", 0))
            for e in round_events
            if e.get("payload", {}).get("delta_len")
        ]
        return float(sum(lengths) / len(lengths)) if lengths else 0.0
