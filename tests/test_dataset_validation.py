"""
tests/test_dataset_validation.py — Unit tests for DatasetValidator.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.training.validation import DatasetValidator, ValidationResult


@pytest.fixture
def validator() -> DatasetValidator:
    return DatasetValidator(min_per_class=1)


@pytest.fixture
def clean_mnist_batch() -> tuple[np.ndarray, np.ndarray]:
    """Minimal clean MNIST-like batch: 50 samples, shape (1, 28, 28)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 1, 28, 28)).astype(np.float32)
    y = np.tile(np.arange(10), 5).astype(np.int64)  # 5 samples per class
    return X, y


class TestValidationResult:
    def test_valid_result_is_truthy(self):
        r = ValidationResult(is_valid=True)
        assert r.is_valid

    def test_str_includes_error_messages(self):
        r = ValidationResult(is_valid=False, errors=["X is empty"])
        assert "ERROR: X is empty" in str(r)

    def test_str_includes_warnings(self):
        r = ValidationResult(is_valid=True, warnings=["dtype is float64"])
        assert "WARN:" in str(r)


class TestDatasetValidator:
    def test_clean_batch_passes(self, validator, clean_mnist_batch):
        X, y = clean_mnist_batch
        result = validator.validate(
            X, y, expected_shape=(1, 28, 28), n_classes=10
        )
        assert result.is_valid
        assert result.errors == []

    def test_nan_in_X_is_error(self, validator, clean_mnist_batch):
        X, y = clean_mnist_batch
        X[0, 0, 0, 0] = np.nan
        result = validator.validate(X, y)
        assert not result.is_valid
        assert any("NaN" in e for e in result.errors)

    def test_inf_in_X_is_error(self, validator, clean_mnist_batch):
        X, y = clean_mnist_batch
        X[1, 0, 5, 5] = np.inf
        result = validator.validate(X, y)
        assert not result.is_valid
        assert any("Inf" in e for e in result.errors)

    def test_shape_mismatch_is_error(self, validator, clean_mnist_batch):
        X, y = clean_mnist_batch
        result = validator.validate(
            X, y, expected_shape=(3, 32, 32)  # wrong shape for MNIST
        )
        assert not result.is_valid
        assert any("shape" in e.lower() for e in result.errors)

    def test_correct_shape_passes(self, validator, clean_mnist_batch):
        X, y = clean_mnist_batch
        result = validator.validate(X, y, expected_shape=(1, 28, 28))
        assert result.is_valid

    def test_label_above_range_is_error(self, validator, clean_mnist_batch):
        X, y = clean_mnist_batch
        y[0] = 10  # out of range for 10-class problem
        result = validator.validate(X, y, n_classes=10)
        assert not result.is_valid
        assert any("10" in e for e in result.errors)

    def test_negative_label_is_error(self, validator, clean_mnist_batch):
        X, y = clean_mnist_batch
        y[0] = -1
        result = validator.validate(X, y, n_classes=10)
        assert not result.is_valid
        assert any("0" in e for e in result.errors)

    def test_mismatched_n_samples_is_error(self, validator):
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(9, dtype=np.int64)  # one fewer than X
        result = validator.validate(X, y)
        assert not result.is_valid
        assert any("leading" in e.lower() or "dimension" in e.lower() for e in result.errors)

    def test_empty_X_is_error(self, validator):
        X = np.zeros((0, 4), dtype=np.float32)
        y = np.zeros(0, dtype=np.int64)
        result = validator.validate(X, y)
        assert not result.is_valid

    def test_float64_X_gives_warning_not_error(self, validator, clean_mnist_batch):
        X, y = clean_mnist_batch
        X = X.astype(np.float64)
        result = validator.validate(X, y)
        assert result.is_valid  # float64 is a warning, not an error
        assert any("float32" in w for w in result.warnings)

    def test_2d_y_is_error(self, validator, clean_mnist_batch):
        X, y = clean_mnist_batch
        y_2d = y.reshape(5, 10)
        result = validator.validate(X, y_2d)
        assert not result.is_valid

    def test_missing_class_gives_warning(self):
        """If a class has 0 samples, warn (not error)."""
        v = DatasetValidator(min_per_class=1)
        X = np.zeros((10, 4), dtype=np.float32)
        # Only classes 0..8, missing class 9
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0], dtype=np.int64)
        result = v.validate(X, y, n_classes=10)
        assert result.is_valid  # warnings don't fail validation
        assert any("9" in w for w in result.warnings)

    def test_non_array_inputs(self, validator):
        result = validator.validate([1, 2], np.array([0]))
        assert not result.is_valid
        assert any("ndarray" in e for e in result.errors)


class TestValidatePartition:
    def test_valid_partitions_pass(self, validator):
        partitions = [
            (np.ones((20, 1, 28, 28), dtype=np.float32), np.zeros(20, dtype=np.int64)),
            (np.ones((15, 1, 28, 28), dtype=np.float32), np.zeros(15, dtype=np.int64)),
        ]
        result = validator.validate_partition(partitions, n_classes=10)
        assert result.is_valid

    def test_empty_partition_list_fails(self, validator):
        result = validator.validate_partition([])
        assert not result.is_valid

    def test_bad_client_errors_aggregate(self, validator):
        partitions = [
            (np.ones((10, 1, 28, 28), dtype=np.float32), np.zeros(10, dtype=np.int64)),
            (np.full((10, 1, 28, 28), np.nan, dtype=np.float32), np.zeros(10, dtype=np.int64)),
        ]
        result = validator.validate_partition(partitions)
        assert not result.is_valid
        # Should mention client_01 (the bad one)
        assert any("client_01" in e for e in result.errors)
