"""
tests/test_schemas.py — Unit tests for Pydantic schema models (TESTING.md §2).

Tests that every schema serialises/deserialises correctly and that field
validation rules match SCHEMAS.md.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai.fl_core.schemas import (
    ClientUpdate,
    Configuration,
    DetectionResult,
    EvaluationResult,
    ModelMetadata,
    TrainingRound,
    TrustLedgerEntry,
    TrustScore,
)

# ---------------------------------------------------------------------------
# ClientUpdate
# ---------------------------------------------------------------------------


class TestClientUpdate:
    def test_valid_construction(self):
        cu = ClientUpdate(
            client_id="client_07",
            round_num=12,
            delta=[0.001, -0.002, 0.003],
            n_samples=480,
        )
        assert cu.client_id == "client_07"
        assert cu.round_num == 12
        assert cu.n_samples == 480
        assert cu.signature is None

    def test_json_roundtrip(self):
        cu = ClientUpdate(client_id="c1", round_num=0, delta=[1.0], n_samples=10)
        raw = cu.model_dump_json()
        cu2 = ClientUpdate.model_validate_json(raw)
        assert cu2.client_id == cu.client_id
        assert cu2.delta == cu.delta

    def test_negative_round_num_invalid(self):
        with pytest.raises(ValidationError):
            ClientUpdate(client_id="c1", round_num=-1, delta=[0.0], n_samples=10)

    def test_zero_n_samples_invalid(self):
        with pytest.raises(ValidationError):
            ClientUpdate(client_id="c1", round_num=0, delta=[0.0], n_samples=0)


# ---------------------------------------------------------------------------
# ModelMetadata
# ---------------------------------------------------------------------------


class TestModelMetadata:
    def test_valid_construction(self):
        mm = ModelMetadata(round_num=5, architecture="linear_softmax_v0")
        assert mm.round_num == 5
        assert mm.model_id  # auto-generated UUID

    def test_json_roundtrip(self):
        mm = ModelMetadata(round_num=1, architecture="test_arch")
        mm2 = ModelMetadata.model_validate(json.loads(mm.model_dump_json()))
        assert mm2.model_id == mm.model_id


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


class TestDetectionResult:
    def test_valid_l3_result(self):
        dr = DetectionResult(
            detector_name="strip_entropy",
            layer="L3",
            subject_id="input_001",
            score=0.45,
            flagged=True,
            boundary=0.54,
        )
        assert dr.flagged is True
        assert dr.round_num is None  # L3 has no round_num

    def test_invalid_layer(self):
        with pytest.raises(ValidationError):
            DetectionResult(
                detector_name="x",
                layer="L1",  # invalid — only L2/L3 allowed
                subject_id="x",
                score=0.0,
                flagged=False,
                boundary=0.0,
            )

    def test_json_roundtrip(self):
        dr = DetectionResult(
            detector_name="strip_entropy",
            layer="L3",
            subject_id="x",
            score=0.7,
            flagged=False,
            boundary=0.5,
        )
        dr2 = DetectionResult.model_validate(json.loads(dr.model_dump_json()))
        assert dr2.score == dr.score


# ---------------------------------------------------------------------------
# TrustScore
# ---------------------------------------------------------------------------


class TestTrustScore:
    def test_score_bounds(self):
        ts = TrustScore(
            subject_type="client",
            subject_id="client_01",
            score=0.5,
            last_updated_round=3,
        )
        assert 0.0 <= ts.score <= 1.0

    def test_score_above_1_invalid(self):
        with pytest.raises(ValidationError):
            TrustScore(
                subject_type="client",
                subject_id="c",
                score=1.5,  # > 1.0, invalid
                last_updated_round=0,
            )


# ---------------------------------------------------------------------------
# TrainingRound
# ---------------------------------------------------------------------------


class TestTrainingRound:
    def test_valid_round(self):
        tr = TrainingRound(
            round_num=5,
            participating_clients=["c0", "c1", "c2"],
            global_model_id="model-uuid-xyz",
        )
        assert tr.excluded_clients == []
        assert tr.flagged_clusters == []

    def test_json_roundtrip(self):
        tr = TrainingRound(
            round_num=0,
            participating_clients=["c0"],
            excluded_clients=["c1"],
            global_model_id="m1",
        )
        tr2 = TrainingRound.model_validate(json.loads(tr.model_dump_json()))
        assert tr2.excluded_clients == ["c1"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_defaults(self):
        cfg = Configuration()
        assert cfg.n_clients == 12
        assert cfg.n_rounds == 20
        assert cfg.aggregator == "multi_krum"

    def test_krum_select_exceeds_n_clients_invalid(self):
        with pytest.raises(ValidationError):
            Configuration(n_clients=5, krum_select=10)

    def test_min_clients_exceeds_n_clients_invalid(self):
        with pytest.raises(ValidationError):
            Configuration(n_clients=4, min_clients=10)

    def test_valid_override(self):
        cfg = Configuration(n_clients=20, n_rounds=5, krum_select=15)
        assert cfg.n_clients == 20
        assert cfg.n_rounds == 5

    def test_json_roundtrip(self):
        cfg = Configuration()
        cfg2 = Configuration.model_validate(json.loads(cfg.model_dump_json()))
        assert cfg2.n_clients == cfg.n_clients


# ---------------------------------------------------------------------------
# TrustLedgerEntry
# ---------------------------------------------------------------------------


class TestTrustLedgerEntry:
    def test_valid_entry(self):
        entry = TrustLedgerEntry(
            layer_id="L1",
            subject_type="client",
            subject_id="client_02",
            round_num=5,
            score=0.8,
            reason="Test reason",
        )
        assert entry.entry_id  # auto-generated

    def test_score_must_be_0_to_1(self):
        with pytest.raises(ValidationError):
            TrustLedgerEntry(
                layer_id="L1",
                subject_type="client",
                subject_id="c",
                score=1.5,  # invalid
                reason="x",
            )

    def test_json_roundtrip(self):
        entry = TrustLedgerEntry(
            layer_id="L3",
            subject_type="input",
            subject_id="inp_001",
            score=0.3,
            reason="Low entropy",
        )
        entry2 = TrustLedgerEntry.model_validate(json.loads(entry.model_dump_json()))
        assert entry2.reason == entry.reason


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_all_null_fields_allowed(self):
        """Missing metrics should be null with a warnings list."""
        er = EvaluationResult(experiment_id="exp_001")
        assert er.clean_accuracy is None
        assert er.attack_success_rate is None
        assert er.warnings == []

    def test_valid_result(self):
        er = EvaluationResult(
            experiment_id="exp_001",
            clean_accuracy=0.95,
            attack_success_rate=0.02,
            robust_accuracy=0.88,
        )
        assert er.clean_accuracy == 0.95
