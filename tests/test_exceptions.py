"""
tests/test_exceptions.py — Unit tests for custom exception hierarchy.
"""

from __future__ import annotations

import pytest

from ai.fl_core.exceptions import (
    CheckpointNotFoundError,
    ConfigValidationError,
    DatasetNotFoundError,
    InsufficientCalibrationDataError,
    InsufficientClientsError,
    RemediationFailedError,
    SentinelError,
    UnsupportedModelError,
)


class TestExceptionHierarchy:
    def test_all_subclass_sentinel_error(self):
        exceptions = [
            InsufficientClientsError(1, 3),
            InsufficientCalibrationDataError(5, 30, "strip_entropy"),
            UnsupportedModelError("activation_consistency", "no penultimate layer"),
            CheckpointNotFoundError(7),
            ConfigValidationError({"field": "error msg"}),
            DatasetNotFoundError("/path/to/data", "phase1_official"),
            RemediationFailedError("rollback and unlearning both failed"),
        ]
        for exc in exceptions:
            assert isinstance(exc, SentinelError), f"{type(exc).__name__} not a SentinelError"

    def test_insufficient_clients_message(self):
        exc = InsufficientClientsError(received=2, required=5)
        assert "2" in str(exc)
        assert "5" in str(exc)
        assert exc.received == 2
        assert exc.required == 5

    def test_insufficient_calibration_message(self):
        exc = InsufficientCalibrationDataError(10, 30, "strip_entropy")
        assert "strip_entropy" in str(exc)
        assert exc.n_provided == 10
        assert exc.n_required == 30

    def test_unsupported_model_message(self):
        exc = UnsupportedModelError("activation_consistency", "no hidden layer")
        assert "activation_consistency" in str(exc)
        assert "no hidden layer" in str(exc)

    def test_checkpoint_not_found_message(self):
        exc = CheckpointNotFoundError(42)
        assert "42" in str(exc)
        assert exc.round_num == 42

    def test_config_validation_error_message(self):
        exc = ConfigValidationError({"n_clients": "must be > 0", "krum_f": "invalid"})
        assert "n_clients" in str(exc)
        assert "krum_f" in str(exc)
        assert "n_clients" in exc.field_errors

    def test_dataset_not_found_message(self):
        exc = DatasetNotFoundError("/no/such/path", "phase1_official")
        assert "phase1_official" in str(exc)
        assert "--dev-mode" in str(exc)

    def test_remediation_failed_message(self):
        exc = RemediationFailedError("both paths exhausted")
        assert "Manual review" in str(exc)


class TestExceptionRaising:
    def test_can_catch_as_sentinel_error(self):
        with pytest.raises(SentinelError):
            raise InsufficientClientsError(0, 3)

    def test_can_catch_as_specific_type(self):
        with pytest.raises(InsufficientClientsError):
            raise InsufficientClientsError(0, 3)
