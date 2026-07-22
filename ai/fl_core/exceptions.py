"""
ai/fl_core/exceptions.py — Custom exceptions for SENTINEL-FL.

Every exception is raised at a specific, documented boundary so callers can
react explicitly rather than catching a generic RuntimeError.  See INTERFACES.md
for the contracts that define when each exception is raised.
"""

from __future__ import annotations


class SentinelError(Exception):
    """Base class for all SENTINEL-FL exceptions."""


# ---------------------------------------------------------------------------
# Aggregation / FL-round exceptions
# ---------------------------------------------------------------------------


class InsufficientClientsError(SentinelError):
    """Raised by an Aggregator when fewer than ``min_clients`` updates are received.

    The round is aborted; the caller must retry with the same global model.
    Never silently degrade to plain averaging.  See INTERFACES.md#Aggregator.

    Args:
        received: Number of client updates actually received.
        required: Minimum number required to proceed.
    """

    def __init__(self, received: int, required: int) -> None:
        self.received = received
        self.required = required
        super().__init__(
            f"Insufficient clients for aggregation: received {received}, "
            f"required at least {required}."
        )


# ---------------------------------------------------------------------------
# Detector / calibration exceptions
# ---------------------------------------------------------------------------


class InsufficientCalibrationDataError(SentinelError):
    """Raised by a Detector when the clean reference set is too small to calibrate.

    A detector that cannot calibrate is excluded from that round's ensemble and
    logged; it never blocks the round.  See INTERFACES.md#Detector.

    Args:
        n_provided: Number of clean samples provided.
        n_required: Minimum required by the detector's calibration procedure.
        detector_name: Human-readable detector identifier (e.g. ``"strip_entropy"``).
    """

    def __init__(self, n_provided: int, n_required: int, detector_name: str = "") -> None:
        self.n_provided = n_provided
        self.n_required = n_required
        self.detector_name = detector_name
        prefix = f"[{detector_name}] " if detector_name else ""
        super().__init__(
            f"{prefix}Calibration data too small: "
            f"got {n_provided} samples, need at least {n_required}."
        )


class UnsupportedModelError(SentinelError):
    """Raised when a Detector is registered against an incompatible model architecture.

    Must be raised at registration time, not at score time.  See INTERFACES.md#Detector.

    Args:
        detector_name: Identifier of the detector that cannot support this model.
        reason: Short explanation of the incompatibility.
    """

    def __init__(self, detector_name: str, reason: str = "") -> None:
        self.detector_name = detector_name
        self.reason = reason
        msg = f"Detector '{detector_name}' does not support this model architecture."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Model Registry exceptions
# ---------------------------------------------------------------------------


class CheckpointNotFoundError(SentinelError):
    """Raised by ModelRegistry.rollback_to() when the requested round has no checkpoint.

    The caller must fall back to the nearest earlier checkpoint explicitly;
    never silently use a different round.  See INTERFACES.md#ModelRegistry.

    Args:
        round_num: The round number that was requested.
    """

    def __init__(self, round_num: int) -> None:
        self.round_num = round_num
        super().__init__(
            f"No model checkpoint found for round {round_num}. "
            "Fall back to the nearest earlier checkpoint explicitly."
        )


# ---------------------------------------------------------------------------
# Configuration exceptions
# ---------------------------------------------------------------------------


class ConfigValidationError(SentinelError):
    """Raised when a loaded YAML config fails Pydantic schema validation.

    The process exits with a clear field-level error message; no partial or
    default-filled config is ever silently run.  See ARCHITECTURE.md §7.10.

    Args:
        field_errors: Mapping of field path → error message.
    """

    def __init__(self, field_errors: dict[str, str]) -> None:
        self.field_errors = field_errors
        lines = "\n".join(f"  {field}: {msg}" for field, msg in field_errors.items())
        super().__init__(f"Configuration validation failed:\n{lines}")


# ---------------------------------------------------------------------------
# Dataset exceptions
# ---------------------------------------------------------------------------


class DatasetNotFoundError(SentinelError):
    """Raised when the Phase 1 official dataset is missing or malformed.

    Falls back to Phase 0 synthetic loader ONLY in explicit ``--dev-mode``;
    otherwise hard-fails.  See ARCHITECTURE.md §7.11 and DATASETS.md.

    Args:
        path: The path that was searched.
        phase: Dataset phase string, e.g. ``"phase1_official"``.
    """

    def __init__(self, path: str, phase: str = "") -> None:
        self.path = path
        self.phase = phase
        prefix = f"[{phase}] " if phase else ""
        super().__init__(
            f"{prefix}Dataset not found or malformed at: {path}. "
            "Pass --dev-mode to fall back to Phase 0 synthetic data."
        )


# ---------------------------------------------------------------------------
# Remediation exceptions
# ---------------------------------------------------------------------------


class RemediationFailedError(SentinelError):
    """Raised when both rollback and unlearning mitigation paths fail.

    Triggers a ``manual_review_required`` flag on the dashboard.
    See ARCHITECTURE.md §7.4.

    Args:
        reason: Description of what was attempted and why it failed.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            f"Remediation failed (both rollback and unlearning exhausted): {reason}. "
            "Manual review required."
        )
