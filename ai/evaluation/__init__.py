"""
ai/evaluation/__init__.py — Evaluation package public re-exports.
"""

from ai.evaluation.benchmark_reporter import BenchmarkReporter
from ai.evaluation.metrics import JsonLinesMetricsCollector
from ai.evaluation.metrics_engine import (
    accuracy,
    attack_success_rate,
    communication_cost_bytes,
    delta_byte_size,
    detection_confusion,
    false_acceptance_rate,
    false_positive_rate,
    false_rejection_rate,
    peak_memory_mb,
    precision_recall_f1,
    robust_accuracy,
    runtime_seconds,
)

__all__ = [
    # Collector / reporter
    "JsonLinesMetricsCollector",
    "BenchmarkReporter",
    # Pure metric functions
    "accuracy",
    "attack_success_rate",
    "robust_accuracy",
    "precision_recall_f1",
    "false_positive_rate",
    "false_acceptance_rate",
    "false_rejection_rate",
    "communication_cost_bytes",
    "delta_byte_size",
    "runtime_seconds",
    "peak_memory_mb",
    "detection_confusion",
]
