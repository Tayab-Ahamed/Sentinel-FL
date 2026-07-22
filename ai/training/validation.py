"""
ai/training/validation.py — Dataset validation utilities.

``DatasetValidator`` runs structural checks on ``(X, y)`` numpy arrays
before they enter the training pipeline.  Validation is non-destructive:
it returns a ``ValidationResult`` dataclass rather than raising, so callers
can decide whether to abort or emit a warning.

Checks performed:
  - Shape consistency between X and y (same leading dimension).
  - Expected input shape (ignoring batch dim).
  - Label range: all y values in ``[0, n_classes)``.
  - At least one sample per class (warns if fewer than ``min_per_class``).
  - No NaN or Inf values in X.
  - Dtype checks (X should be float32, y should be integer).

Usage::

    validator = DatasetValidator()
    result = validator.validate(X, y, expected_shape=(1, 28, 28), n_classes=10)
    if not result.is_valid:
        raise ValueError(result.errors)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a dataset validation pass."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"ValidationResult(is_valid={self.is_valid})"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


class DatasetValidator:
    """Validates ``(X, y)`` numpy pairs before training.

    Args:
        min_per_class: Warn (not error) when any class has fewer than this
            many samples.  Useful to catch extreme Dirichlet partitions.
    """

    def __init__(self, min_per_class: int = 1) -> None:
        self._min_per_class = min_per_class

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expected_shape: tuple[int, ...] | None = None,
        n_classes: int | None = None,
        dataset_name: str = "dataset",
    ) -> ValidationResult:
        """Run all checks on ``(X, y)`` and return a ``ValidationResult``.

        Args:
            X: Feature array; first dim is batch.
            y: Integer label array; must be 1-D.
            expected_shape: Expected per-sample shape (e.g. ``(1, 28, 28)``).
                Checked against ``X.shape[1:]``.  Skip check if ``None``.
            n_classes: Expected number of classes. Labels must be in
                ``[0, n_classes)``.  Skip check if ``None``.
            dataset_name: Used in error/warning messages.

        Returns:
            ``ValidationResult`` with ``is_valid=True`` if no errors.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── 1. Basic type checks ───────────────────────────────────────
        if not isinstance(X, np.ndarray):
            errors.append(f"[{dataset_name}] X must be a numpy ndarray, got {type(X).__name__}")
        if not isinstance(y, np.ndarray):
            errors.append(f"[{dataset_name}] y must be a numpy ndarray, got {type(y).__name__}")
        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # ── 2. Non-empty ──────────────────────────────────────────────
        if len(X) == 0:
            errors.append(f"[{dataset_name}] X is empty (0 samples).")
        if len(y) == 0:
            errors.append(f"[{dataset_name}] y is empty (0 samples).")
        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # ── 3. Shape consistency ──────────────────────────────────────
        if X.shape[0] != y.shape[0]:
            errors.append(
                f"[{dataset_name}] X and y have different leading dimensions: "
                f"X={X.shape[0]}, y={y.shape[0]}"
            )

        if y.ndim != 1:
            errors.append(
                f"[{dataset_name}] y must be 1-D, got shape {y.shape}"
            )

        # ── 4. Expected per-sample shape ──────────────────────────────
        if expected_shape is not None:
            actual = tuple(X.shape[1:])
            if actual != expected_shape:
                errors.append(
                    f"[{dataset_name}] Expected per-sample shape {expected_shape}, "
                    f"got {actual}"
                )

        # ── 5. NaN / Inf in X ─────────────────────────────────────────
        if np.issubdtype(X.dtype, np.floating):
            n_nan = int(np.isnan(X).sum())
            n_inf = int(np.isinf(X).sum())
            if n_nan > 0:
                errors.append(
                    f"[{dataset_name}] X contains {n_nan} NaN value(s)."
                )
            if n_inf > 0:
                errors.append(
                    f"[{dataset_name}] X contains {n_inf} Inf value(s)."
                )

        # ── 6. Label range ────────────────────────────────────────────
        if n_classes is not None and y.ndim == 1:
            if not np.issubdtype(y.dtype, np.integer):
                warnings.append(
                    f"[{dataset_name}] y dtype is {y.dtype}, expected integer."
                )
            y_min, y_max = int(y.min()), int(y.max())
            if y_min < 0:
                errors.append(
                    f"[{dataset_name}] Labels must be >= 0, found min={y_min}."
                )
            if y_max >= n_classes:
                errors.append(
                    f"[{dataset_name}] Labels must be < {n_classes}, found max={y_max}."
                )

            # ── 7. Per-class sample count ─────────────────────────────
            for cls in range(n_classes):
                count = int((y == cls).sum())
                if count < self._min_per_class:
                    warnings.append(
                        f"[{dataset_name}] Class {cls} has only {count} sample(s) "
                        f"(min_per_class={self._min_per_class})."
                    )

        # ── 8. Dtype advisory ─────────────────────────────────────────
        if X.dtype != np.float32:
            warnings.append(
                f"[{dataset_name}] X dtype is {X.dtype}; float32 recommended for "
                "PyTorch compatibility."
            )

        is_valid = len(errors) == 0
        result = ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
        if not is_valid:
            logger.error("Dataset validation failed for %s:\n%s", dataset_name, result)
        elif warnings:
            logger.warning("Dataset validation warnings for %s:\n%s", dataset_name, result)
        else:
            logger.debug(
                "Dataset validation passed for %s (n=%d)", dataset_name, len(X)
            )
        return result

    def validate_partition(
        self,
        partitions: list[tuple[np.ndarray, np.ndarray]],
        n_classes: int | None = None,
        expected_shape: tuple[int, ...] | None = None,
    ) -> ValidationResult:
        """Validate a list of ``(X, y)`` client partitions.

        Args:
            partitions: List of ``(X_i, y_i)`` tuples.
            n_classes: Expected number of classes.
            expected_shape: Expected per-sample shape.

        Returns:
            Merged ``ValidationResult`` across all partitions.
        """
        all_errors: list[str] = []
        all_warnings: list[str] = []

        if not partitions:
            return ValidationResult(
                is_valid=False,
                errors=["Partition list is empty."],
            )

        for i, (X_i, y_i) in enumerate(partitions):
            r = self.validate(
                X_i, y_i,
                expected_shape=expected_shape,
                n_classes=n_classes,
                dataset_name=f"client_{i:02d}",
            )
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
        )
