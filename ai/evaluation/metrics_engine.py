"""
ai/evaluation/metrics_engine.py — Pure-function metric computation engine.

No I/O, no side-effects.  All functions receive arrays or scalars and return
numbers.  Consumed by JsonLinesMetricsCollector and BenchmarkReporter.

Functions:
    accuracy(y_true, y_pred) → float
    attack_success_rate(y_true, y_pred, target_class) → float
    robust_accuracy(y_true_clean, y_pred_clean,
                    y_true_triggered, y_pred_triggered) → float
    precision_recall_f1(y_true, y_pred, pos_label) → tuple[float, float, float]
    false_positive_rate(y_true, y_pred, neg_label) → float
    false_acceptance_rate(n_triggered, n_flagged_triggered) → float
    false_rejection_rate(n_clean, n_flagged_clean) → float
    communication_cost_bytes(deltas_sizes) → int
    runtime_seconds(start_ns, end_ns) → float
    peak_memory_mb() → float
    detection_confusion(y_true_flags, y_pred_flags)
        → dict[str, int]          # TP, FP, TN, FN
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification accuracy
# ---------------------------------------------------------------------------


def accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """Fraction of predictions that match ground truth.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        Accuracy in [0, 1].  Returns 0.0 on empty input.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp))


# ---------------------------------------------------------------------------
# Attack Success Rate
# ---------------------------------------------------------------------------


def attack_success_rate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    target_class: int,
) -> float:
    """Fraction of triggered inputs successfully mis-classified to target_class.

    Only inputs that are genuinely NOT target_class in the clean setting
    (i.e. the attack actually needs to change the prediction) are considered.
    If all inputs belong to target_class there is no attack surface → returns 0.

    Args:
        y_true: Ground-truth class labels (clean, before trigger injection).
        y_pred: Predicted labels on TRIGGERED inputs.
        target_class: The backdoor's target class.

    Returns:
        ASR in [0, 1].
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    non_target = yt != target_class
    denom = int(non_target.sum())
    if denom == 0:
        return 0.0
    return float(np.mean(yp[non_target] == target_class))


# ---------------------------------------------------------------------------
# Robust accuracy
# ---------------------------------------------------------------------------


def robust_accuracy(
    y_true_clean: Sequence[int],
    y_pred_clean: Sequence[int],
    y_true_triggered: Sequence[int],
    y_pred_triggered: Sequence[int],
    target_class: int,
) -> float:
    """Joint accuracy on clean inputs AND resistance to the backdoor.

    Defined as: C-Acc * (1 − ASR).  Both factors must be high for a strong
    defence — a model that rejects all inputs has C-Acc=0; a model that
    always predicts the target has ASR=1.

    Args:
        y_true_clean: Clean-set ground-truth labels.
        y_pred_clean: Predictions on clean inputs.
        y_true_triggered: Triggered-set ground-truth (same as clean).
        y_pred_triggered: Predictions on triggered inputs.
        target_class: Backdoor target class.

    Returns:
        Robust accuracy in [0, 1].
    """
    c_acc = accuracy(y_true_clean, y_pred_clean)
    asr = attack_success_rate(y_true_triggered, y_pred_triggered, target_class)
    return float(c_acc * (1.0 - asr))


# ---------------------------------------------------------------------------
# Detection precision / recall / F1
# ---------------------------------------------------------------------------


def precision_recall_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    pos_label: int = 1,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 for binary detection.

    Args:
        y_true: Ground-truth binary labels (1=malicious, 0=benign).
        y_pred: Predicted binary labels.
        pos_label: Which value is the positive (malicious) class.

    Returns:
        ``(precision, recall, f1)`` each in [0, 1].
        Returns ``(0, 0, 0)`` if there are no true positives in either set.
    """
    yt = (np.asarray(y_true) == pos_label).astype(int)
    yp = (np.asarray(y_pred) == pos_label).astype(int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


# ---------------------------------------------------------------------------
# False Positive Rate (FPR)
# ---------------------------------------------------------------------------


def false_positive_rate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    neg_label: int = 0,
) -> float:
    """FP / (FP + TN) — fraction of benign inputs/clients incorrectly flagged.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        neg_label: Which value is the negative (benign) class.

    Returns:
        FPR in [0, 1].  Returns 0.0 when there are no negatives.
    """
    yt = (np.asarray(y_true) == neg_label).astype(int)
    yp = (np.asarray(y_pred) == neg_label).astype(int)

    # Negate: treat positives as the "flagged" class
    fp = int(np.sum((yt == 1) & (yp == 0)))  # benign but predicted malicious
    tn = int(np.sum((yt == 1) & (yp == 1)))  # benign and predicted benign

    if (fp + tn) == 0:
        return 0.0
    return float(fp / (fp + tn))


# ---------------------------------------------------------------------------
# False Acceptance Rate / False Rejection Rate  (L3 — STRIP-specific)
# ---------------------------------------------------------------------------


def false_acceptance_rate(
    n_triggered: int,
    n_flagged_triggered: int,
) -> float:
    """FAR = trojaned inputs NOT flagged / total trojaned inputs.

    A lower FAR means the detector catches more backdoored inputs.

    Args:
        n_triggered: Total number of triggered (poisoned) test inputs.
        n_flagged_triggered: How many of them were flagged by L3.

    Returns:
        FAR in [0, 1].  Returns 0.0 when n_triggered == 0.
    """
    if n_triggered <= 0:
        return 0.0
    not_flagged = max(0, n_triggered - n_flagged_triggered)
    return float(not_flagged / n_triggered)


def false_rejection_rate(
    n_clean: int,
    n_flagged_clean: int,
) -> float:
    """FRR = clean inputs incorrectly flagged / total clean inputs.

    A lower FRR means fewer false alarms on normal traffic.

    Args:
        n_clean: Total number of clean (benign) test inputs.
        n_flagged_clean: How many of them were flagged by L3.

    Returns:
        FRR in [0, 1].  Returns 0.0 when n_clean == 0.
    """
    if n_clean <= 0:
        return 0.0
    return float(min(n_flagged_clean, n_clean) / n_clean)


# ---------------------------------------------------------------------------
# Communication cost
# ---------------------------------------------------------------------------


def communication_cost_bytes(delta_sizes: Sequence[int]) -> int:
    """Total bytes transferred across all client updates.

    Args:
        delta_sizes: Byte-length of each ``ClientUpdate.delta`` serialised as
            float32.  For a delta of length N, byte size = N * 4.

    Returns:
        Total bytes as an integer.
    """
    return int(sum(delta_sizes))


def delta_byte_size(delta_len: int, dtype_bytes: int = 4) -> int:
    """Byte size of one flattened delta of ``delta_len`` parameters.

    Args:
        delta_len: Number of float parameters in the delta.
        dtype_bytes: Bytes per parameter (4 = float32, 8 = float64).

    Returns:
        Byte size.
    """
    return delta_len * dtype_bytes


# ---------------------------------------------------------------------------
# Runtime / memory
# ---------------------------------------------------------------------------


def runtime_seconds(start_ns: int, end_ns: int) -> float:
    """Convert nanosecond timestamps to elapsed seconds.

    Args:
        start_ns: ``time.perf_counter_ns()`` at start.
        end_ns: ``time.perf_counter_ns()`` at end.

    Returns:
        Elapsed time in seconds, rounded to 6 decimal places.
    """
    return round((end_ns - start_ns) / 1e9, 6)


def peak_memory_mb() -> float:
    """Return current process RSS memory usage in MB.

    Uses ``psutil`` if available; falls back to 0.0 with a warning.

    Returns:
        RSS in megabytes, or 0.0 if psutil is not installed.
    """
    try:
        import os

        import psutil

        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        return round(rss / (1024 * 1024), 2)
    except ImportError:
        logger.debug("peak_memory_mb: psutil not installed — returning 0.0.")
        return 0.0
    except Exception as exc:
        logger.debug("peak_memory_mb: %s — returning 0.0.", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Detection confusion matrix
# ---------------------------------------------------------------------------


def detection_confusion(
    y_true_flags: Sequence[int],
    y_pred_flags: Sequence[int],
) -> dict[str, int]:
    """Compute TP/FP/TN/FN counts for binary detection labels.

    Args:
        y_true_flags: Ground-truth binary flags (1=malicious, 0=benign).
        y_pred_flags: Predicted binary flags.

    Returns:
        Dict with keys ``'TP'``, ``'FP'``, ``'TN'``, ``'FN'``.
    """
    yt = np.asarray(y_true_flags)
    yp = np.asarray(y_pred_flags)
    return {
        "TP": int(np.sum((yt == 1) & (yp == 1))),
        "FP": int(np.sum((yt == 0) & (yp == 1))),
        "TN": int(np.sum((yt == 0) & (yp == 0))),
        "FN": int(np.sum((yt == 1) & (yp == 0))),
    }
