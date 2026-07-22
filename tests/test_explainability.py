"""
tests/test_explainability.py — Milestone 8: Explainability full test suite.

Covers:
  Schemas: SHAPExplanation, FeatureImportanceResult, DetectionExplanation,
           TrustExplanation, AttackExplanation, ChartArtifact
  SHAPExplainer: fit, explain_input (permutation fallback), explain_batch,
                 top_features, is_fitted, unfitted error
  Feature importance: permutation_importance, coefficient_importance,
                      gradient_feature_importance
  DetectionExplainer: explain_l1_flag, explain_l2_flag, explain_l3_flag,
                      explain_ledger_entry dispatch
  TrustExplainer: explain_trust_score, explain_reputation_trajectory,
                  rank_clients_by_suspicion
  AttackExplainer: explain_backdoor, trigger_description, poison_ratio_analysis
  ChartGenerator: shap_bar_chart, feature_importance_chart,
                  trust_trajectory_chart, reputation_heatmap_chart,
                  alert_timeline_chart, save_all
  Integration: JSON round-trip of all explanation schemas
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _toy_model(n_features: int = 10, n_classes: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((200, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, 200)
    clf = LogisticRegression(max_iter=300, random_state=seed)
    clf.fit(X, y)
    return clf


def _random_X(n: int = 80, n_feat: int = 10, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, n_feat)).astype(np.float32)


def _random_y(n: int = 80, n_classes: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_classes, n)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSHAPExplanationSchema:
    def test_valid(self):
        from ai.fl_core.schemas import SHAPExplanation

        exp = SHAPExplanation(
            input_id="x0",
            predicted_class=1,
            base_value=0.33,
            shap_values=[0.1, -0.2, 0.05],
            feature_names=["a", "b", "c"],
        )
        assert exp.input_id == "x0"
        assert len(exp.shap_values) == 3

    def test_length_mismatch_raises(self):
        from ai.fl_core.schemas import SHAPExplanation

        with pytest.raises(Exception):
            SHAPExplanation(
                input_id="x",
                predicted_class=0,
                base_value=0.0,
                shap_values=[0.1, 0.2],
                feature_names=["a"],  # mismatch
            )

    def test_json_round_trip(self):
        from ai.fl_core.schemas import SHAPExplanation

        exp = SHAPExplanation(
            input_id="x1",
            predicted_class=0,
            base_value=0.5,
            shap_values=[0.1, 0.2],
            feature_names=["f0", "f1"],
        )
        data = json.loads(exp.model_dump_json())
        assert data["input_id"] == "x1"
        assert data["shap_values"] == [0.1, 0.2]


class TestFeatureImportanceResultSchema:
    def test_valid(self):
        from ai.fl_core.schemas import FeatureImportanceResult

        r = FeatureImportanceResult(
            method="permutation",
            feature_names=["f0", "f1"],
            importance_scores=[0.3, 0.1],
        )
        assert r.method == "permutation"

    def test_length_mismatch_raises(self):
        from ai.fl_core.schemas import FeatureImportanceResult

        with pytest.raises(Exception):
            FeatureImportanceResult(
                method="permutation",
                feature_names=["f0"],
                importance_scores=[0.3, 0.1],  # mismatch
            )

    def test_json_round_trip(self):
        from ai.fl_core.schemas import FeatureImportanceResult

        r = FeatureImportanceResult(
            method="coefficient",
            feature_names=["a", "b"],
            importance_scores=[0.5, 0.2],
        )
        data = json.loads(r.model_dump_json())
        assert data["method"] == "coefficient"


class TestDetectionExplanationSchema:
    def test_valid(self):
        from ai.fl_core.schemas import DetectionExplanation

        exp = DetectionExplanation(
            entry_id="e001",
            layer_id="L1",
            subject_id="client_02",
            subject_type="client",
            reason_string="High cosine similarity detected.",
        )
        assert exp.layer_id == "L1"
        assert exp.chart_artifacts == []

    def test_json_serialisable(self):
        from ai.fl_core.schemas import DetectionExplanation

        exp = DetectionExplanation(
            entry_id="e002",
            layer_id="L3",
            subject_id="input_42",
            subject_type="input",
            reason_string="STRIP flagged.",
        )
        data = json.loads(exp.model_dump_json())
        assert data["layer_id"] == "L3"


class TestTrustExplanationSchema:
    def test_valid(self):
        from ai.fl_core.schemas import TrustExplanation

        t = TrustExplanation(
            client_id="c01",
            current_score=0.3,
            is_suspicious=True,
            narrative="Client is suspicious.",
        )
        assert t.is_suspicious is True

    def test_score_out_of_range(self):
        from ai.fl_core.schemas import TrustExplanation

        with pytest.raises(Exception):
            TrustExplanation(
                client_id="c01",
                current_score=1.5,  # > 1.0
                is_suspicious=True,
                narrative="",
            )


class TestAttackExplanationSchema:
    def test_valid(self):
        from ai.fl_core.schemas import AttackExplanation

        a = AttackExplanation(
            attack_type="BadNets",
            target_label=7,
            detection_confidence=0.85,
        )
        assert a.attack_type == "BadNets"

    def test_confidence_out_of_range(self):
        from ai.fl_core.schemas import AttackExplanation

        with pytest.raises(Exception):
            AttackExplanation(attack_type="BadNets", detection_confidence=1.5)


class TestChartArtifactSchema:
    def test_valid(self):
        from ai.fl_core.schemas import ChartArtifact

        art = ChartArtifact(chart_type="shap_bar", png_b64="abc123")
        assert art.chart_type == "shap_bar"
        assert art.width_px == 800

    def test_json_round_trip(self):
        from ai.fl_core.schemas import ChartArtifact

        art = ChartArtifact(chart_type="feature_importance", png_b64="xyz", title="Test")
        data = json.loads(art.model_dump_json())
        assert data["png_b64"] == "xyz"


# ---------------------------------------------------------------------------
# SHAPExplainer tests (permutation fallback — no shap package required)
# ---------------------------------------------------------------------------


class TestSHAPExplainer:
    """Tests use the permutation fallback path so shap package is not required."""

    def _build(self, n_feat: int = 10) -> tuple:
        from ai.explainability.shap_explainer import SHAPExplainer

        model = _toy_model(n_feat)
        X = _random_X(80, n_feat)
        exp = SHAPExplainer(n_background=20, nsamples=20, top_k=5)
        return exp, model, X

    def test_not_fitted_initially(self):
        from ai.explainability.shap_explainer import SHAPExplainer

        assert not SHAPExplainer().is_fitted

    def test_unfitted_raises(self):
        from ai.explainability.shap_explainer import SHAPExplainer

        exp = SHAPExplainer()
        with pytest.raises(RuntimeError, match="fit()"):
            exp.explain_input(np.zeros(5))

    def test_fit_sets_is_fitted(self):
        exp, model, X = self._build()
        # Force permutation path
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X)
        assert exp.is_fitted

    def test_explain_input_returns_schema(self):
        from ai.fl_core.schemas import SHAPExplanation

        exp, model, X = self._build()
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X)
            result = exp.explain_input(X[0], input_id="x0", predicted_class=0)
        assert isinstance(result, SHAPExplanation)

    def test_explain_input_shap_len_matches_features(self):
        exp, model, X = self._build(n_feat=10)
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X)
            result = exp.explain_input(X[0])
        assert len(result.shap_values) == len(result.feature_names)

    def test_explain_input_top_k_populated(self):
        exp, model, X = self._build()
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X)
            result = exp.explain_input(X[0])
        assert len(result.top_k_features) <= 5
        assert result.top_k_features[0]["rank"] == 1

    def test_explain_input_method_permutation(self):
        exp, model, X = self._build()
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X)
            result = exp.explain_input(X[0])
        assert result.method == "permutation"

    def test_explain_batch_length(self):
        exp, model, X = self._build()
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X)
            results = exp.explain_batch(X[:5])
        assert len(results) == 5

    def test_explain_batch_ids(self):
        exp, model, X = self._build()
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X)
            ids = [f"inp_{i}" for i in range(4)]
            results = exp.explain_batch(X[:4], input_ids=ids)
        assert [r.input_id for r in results] == ids

    def test_top_features_list(self):
        exp, model, X = self._build()
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X)
            top = exp.top_features(X[0], k=3)
        assert len(top) == 3
        assert all("name" in t and "shap_value" in t and "rank" in t for t in top)

    def test_feature_names_default(self):
        exp, model, X = self._build(n_feat=10)
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X)
        assert exp.feature_names == [f"f{i}" for i in range(10)]

    def test_feature_names_custom(self):
        exp, model, X = self._build(n_feat=5)
        names = ["alpha", "beta", "gamma", "delta", "epsilon"]
        with patch("ai.explainability.shap_explainer._check_shap", return_value=False):
            exp.fit(model, X, feature_names=names)
        assert exp.feature_names == names


# ---------------------------------------------------------------------------
# Feature importance tests
# ---------------------------------------------------------------------------


class TestPermutationImportance:
    def test_returns_correct_schema(self):
        from ai.explainability.feature_importance import permutation_importance
        from ai.fl_core.schemas import FeatureImportanceResult

        model = _toy_model()
        X = _random_X()
        y = _random_y()
        result = permutation_importance(model, X, y, n_repeats=3)
        assert isinstance(result, FeatureImportanceResult)
        assert result.method == "permutation"

    def test_length_matches_features(self):
        from ai.explainability.feature_importance import permutation_importance

        model = _toy_model(n_features=8)
        X = _random_X(60, 8)
        y = _random_y(60)
        result = permutation_importance(model, X, y, n_repeats=2)
        assert len(result.importance_scores) == 8
        assert len(result.feature_names) == 8

    def test_ranked_features_sorted(self):
        from ai.explainability.feature_importance import permutation_importance

        model = _toy_model()
        X = _random_X()
        y = _random_y()
        result = permutation_importance(model, X, y, n_repeats=3)
        scores = [r["score"] for r in result.ranked_features]
        assert scores == sorted(scores, reverse=True)

    def test_custom_feature_names(self):
        from ai.explainability.feature_importance import permutation_importance

        model = _toy_model(n_features=5)
        X = _random_X(50, 5)
        y = _random_y(50)
        names = ["w0", "w1", "w2", "w3", "w4"]
        result = permutation_importance(model, X, y, n_repeats=2, feature_names=names)
        assert result.feature_names == names

    def test_context_field(self):
        from ai.explainability.feature_importance import permutation_importance

        model = _toy_model()
        X, y = _random_X(), _random_y()
        result = permutation_importance(model, X, y, n_repeats=2, context="test context")
        assert result.context == "test context"


class TestCoefficientImportance:
    def test_returns_schema(self):
        from ai.explainability.feature_importance import coefficient_importance
        from ai.fl_core.schemas import FeatureImportanceResult

        model = _toy_model()
        result = coefficient_importance(model)
        assert isinstance(result, FeatureImportanceResult)
        assert result.method == "coefficient"

    def test_all_scores_nonneg(self):
        from ai.explainability.feature_importance import coefficient_importance

        model = _toy_model()
        result = coefficient_importance(model)
        assert all(s >= 0 for s in result.importance_scores)

    def test_no_coef_raises(self):
        from ai.explainability.feature_importance import coefficient_importance

        with pytest.raises(ValueError, match="coef_"):
            coefficient_importance(MagicMock(spec=[]))  # no coef_

    def test_feature_names_length(self):
        from ai.explainability.feature_importance import coefficient_importance

        n = 10
        model = _toy_model(n_features=n)
        result = coefficient_importance(model)
        assert len(result.importance_scores) == n


class TestGradientFeatureImportance:
    def test_returns_schema(self):
        from ai.explainability.feature_importance import gradient_feature_importance
        from ai.fl_core.schemas import FeatureImportanceResult

        delta = np.random.default_rng(0).standard_normal(10).tolist()
        result = gradient_feature_importance(delta)
        assert isinstance(result, FeatureImportanceResult)
        assert result.method == "gradient"

    def test_scores_are_abs_values(self):
        from ai.explainability.feature_importance import gradient_feature_importance

        delta = [-2.0, 1.0, -0.5]
        result = gradient_feature_importance(delta)
        assert result.importance_scores == pytest.approx([2.0, 1.0, 0.5])

    def test_ranked_descending(self):
        from ai.explainability.feature_importance import gradient_feature_importance

        delta = [0.1, 0.9, 0.5]
        result = gradient_feature_importance(delta)
        scores = [r["score"] for r in result.ranked_features]
        assert scores == sorted(scores, reverse=True)

    def test_block_aggregation(self):
        from ai.explainability.feature_importance import gradient_feature_importance

        # 6 params, 2 features → 3 params per feature
        delta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = gradient_feature_importance(delta, n_params_per_feature=3)
        # feature 0: mean(|[1,2,3]|) = 2.0; feature 1: mean(|[4,5,6]|) = 5.0
        assert len(result.importance_scores) == 2
        assert result.importance_scores[0] == pytest.approx(2.0)
        assert result.importance_scores[1] == pytest.approx(5.0)

    def test_custom_names(self):
        from ai.explainability.feature_importance import gradient_feature_importance

        delta = [0.1, 0.2, 0.3]
        result = gradient_feature_importance(delta, feature_names=["x", "y", "z"])
        assert result.feature_names == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# DetectionExplainer tests
# ---------------------------------------------------------------------------


class TestDetectionExplainer:
    def test_explain_l1_flag_basic(self):
        from ai.explainability.detection_explainer import DetectionExplainer
        from ai.fl_core.schemas import DetectionExplanation

        de = DetectionExplainer()
        result = de.explain_l1_flag(
            client_id="c01",
            round_num=3,
            norm=4.2,
            cosine_sim=0.95,
        )
        assert isinstance(result, DetectionExplanation)
        assert result.layer_id == "L1"
        assert result.subject_id == "c01"
        assert "4.2" in result.reason_string or "L1" in result.reason_string or "c01" in result.reason_string

    def test_explain_l1_with_delta(self):
        from ai.explainability.detection_explainer import DetectionExplainer

        de = DetectionExplainer()
        delta = np.random.default_rng(0).standard_normal(10).tolist()
        result = de.explain_l1_flag(
            client_id="c02",
            round_num=1,
            delta=delta,
            feature_names=[f"f{i}" for i in range(10)],
        )
        assert result.feature_importance is not None
        assert result.feature_importance.method == "gradient"

    def test_explain_l2_flag(self):
        from ai.explainability.detection_explainer import DetectionExplainer
        from ai.fl_core.schemas import DetectionExplanation

        de = DetectionExplainer()
        result = de.explain_l2_flag(
            entry_id="e_l2_001",
            label=7,
            audit_evidence={"confidence": 0.91, "trigger_reversed": True},
            reason_string="[model_auditor] Label 7 flagged.",
        )
        assert isinstance(result, DetectionExplanation)
        assert result.layer_id == "L2"
        assert result.subject_type == "label"

    def test_explain_l3_flag(self):
        from ai.explainability.detection_explainer import DetectionExplainer
        from ai.fl_core.schemas import DetectionExplanation

        de = DetectionExplainer()
        result = de.explain_l3_flag(
            entry_id="e_l3_001",
            alert_evidence={"fused_score": 0.82, "alert_severity": "high"},
            reason_string="[strip_entropy] Input flagged.",
            subject_id="inp_99",
        )
        assert isinstance(result, DetectionExplanation)
        assert result.layer_id == "L3"
        assert result.subject_type == "input"

    def test_explain_l3_with_shap(self):
        from ai.explainability.detection_explainer import DetectionExplainer
        from ai.fl_core.schemas import SHAPExplanation

        de = DetectionExplainer()
        shap_exp = SHAPExplanation(
            input_id="inp_0",
            predicted_class=1,
            base_value=0.33,
            shap_values=[0.1, -0.2],
            feature_names=["f0", "f1"],
        )
        result = de.explain_l3_flag(
            entry_id="e003",
            shap_explanation=shap_exp,
        )
        assert result.shap_explanation is not None
        assert result.shap_explanation.input_id == "inp_0"

    def test_explain_ledger_entry_l1_dispatch(self):
        from ai.explainability.detection_explainer import DetectionExplainer
        from ai.fl_core.schemas import TrustLedgerEntry

        de = DetectionExplainer()
        entry = TrustLedgerEntry(
            layer_id="L1",
            subject_type="client",
            subject_id="c_test",
            round_num=5,
            score=0.7,
            reason="High norm detected.",
            evidence={"l2_norm": 4.5},
        )
        result = de.explain_ledger_entry(entry)
        assert result.layer_id == "L1"
        assert result.subject_id == "c_test"

    def test_explain_ledger_entry_l2_dispatch(self):
        from ai.explainability.detection_explainer import DetectionExplainer
        from ai.fl_core.schemas import TrustLedgerEntry

        de = DetectionExplainer()
        entry = TrustLedgerEntry(
            layer_id="L2",
            subject_type="label",
            subject_id="label_3",
            round_num=2,
            score=0.88,
            reason="Trigger reversed.",
        )
        result = de.explain_ledger_entry(entry)
        assert result.layer_id == "L2"

    def test_explain_ledger_entry_l3_dispatch(self):
        from ai.explainability.detection_explainer import DetectionExplainer
        from ai.fl_core.schemas import TrustLedgerEntry

        de = DetectionExplainer()
        entry = TrustLedgerEntry(
            layer_id="L3",
            subject_type="input",
            subject_id="inp_001",
            score=0.75,
            reason="STRIP entropy flagged.",
            evidence={"fused_score": 0.75, "alert_severity": "high"},
        )
        result = de.explain_ledger_entry(entry)
        assert result.layer_id == "L3"


# ---------------------------------------------------------------------------
# TrustExplainer tests
# ---------------------------------------------------------------------------


def _make_mock_ledger(client_ids: list[str], n_flags_per_client: int = 2) -> MagicMock:
    """Build a mock FileTrustLedger."""
    from ai.fl_core.schemas import TrustLedgerEntry

    ledger = MagicMock()

    def _history(cid):
        return [
            TrustLedgerEntry(
                layer_id=["L1", "L2"][i % 2],
                subject_type="client",
                subject_id=cid,
                round_num=i + 1,
                score=0.6 + i * 0.1,
                reason=f"Flag {i} for {cid}.",
            )
            for i in range(n_flags_per_client)
        ]

    ledger.get_client_history.side_effect = _history
    ledger.get_reputation.side_effect = lambda cid: 0.4 if cid == client_ids[0] else 0.8
    ledger.export_snapshot.return_value = [
        {"subject_id": cid, "subject_type": "client", "layer_id": "L1", "score": 0.5}
        for cid in client_ids
    ]
    return ledger


class TestTrustExplainer:
    def test_explain_trust_score_returns_schema(self):
        from ai.explainability.trust_explainer import TrustExplainer
        from ai.fl_core.schemas import TrustExplanation

        te = TrustExplainer(suspicious_threshold=0.5)
        ledger = _make_mock_ledger(["c01", "c02"])
        result = te.explain_trust_score("c01", ledger)
        assert isinstance(result, TrustExplanation)
        assert result.client_id == "c01"

    def test_suspicious_classification(self):
        from ai.explainability.trust_explainer import TrustExplainer

        te = TrustExplainer(suspicious_threshold=0.5)
        ledger = _make_mock_ledger(["c01"])
        result = te.explain_trust_score("c01", ledger)
        # c01 gets score 0.4 from mock → suspicious
        assert result.is_suspicious is True

    def test_trusted_classification(self):
        from ai.explainability.trust_explainer import TrustExplainer

        te = TrustExplainer(suspicious_threshold=0.5)
        ledger = _make_mock_ledger(["c01", "c02"])
        result = te.explain_trust_score("c02", ledger)
        assert result.is_suspicious is False  # c02 gets score 0.8

    def test_layer_breakdown_populated(self):
        from ai.explainability.trust_explainer import TrustExplainer

        te = TrustExplainer()
        ledger = _make_mock_ledger(["c01"], n_flags_per_client=4)
        result = te.explain_trust_score("c01", ledger)
        # 4 entries alternating L1/L2
        assert "L1" in result.layer_breakdown
        assert "L2" in result.layer_breakdown

    def test_top_contributing_entries_populated(self):
        from ai.explainability.trust_explainer import TrustExplainer

        te = TrustExplainer()
        ledger = _make_mock_ledger(["c01"], n_flags_per_client=3)
        result = te.explain_trust_score("c01", ledger)
        assert len(result.top_contributing_entries) <= 5

    def test_narrative_non_empty(self):
        from ai.explainability.trust_explainer import TrustExplainer

        te = TrustExplainer()
        ledger = _make_mock_ledger(["c01"])
        result = te.explain_trust_score("c01", ledger)
        assert len(result.narrative) > 10

    def test_explain_reputation_trajectory_rounds(self):
        from ai.explainability.trust_explainer import TrustExplainer

        te = TrustExplainer()
        ledger = _make_mock_ledger(["c01"], n_flags_per_client=4)
        result = te.explain_reputation_trajectory("c01", ledger)
        assert isinstance(result.score_trajectory, list)

    def test_rank_clients_returns_list(self):
        from ai.explainability.trust_explainer import TrustExplainer

        te = TrustExplainer(suspicious_threshold=0.5)
        ledger = _make_mock_ledger(["c01", "c02", "c03"])
        ranked = te.rank_clients_by_suspicion(ledger)
        assert len(ranked) == 3

    def test_rank_clients_suspicious_first(self):
        from ai.explainability.trust_explainer import TrustExplainer

        te = TrustExplainer(suspicious_threshold=0.5)
        ledger = _make_mock_ledger(["c01", "c02"])
        ranked = te.rank_clients_by_suspicion(ledger)
        # c01 is suspicious (0.4), c02 is not (0.8)
        assert ranked[0].client_id == "c01"

    def test_rank_clients_top_k(self):
        from ai.explainability.trust_explainer import TrustExplainer

        te = TrustExplainer()
        ledger = _make_mock_ledger(["c01", "c02", "c03", "c04"])
        ranked = te.rank_clients_by_suspicion(ledger, top_k=2)
        assert len(ranked) == 2


# ---------------------------------------------------------------------------
# AttackExplainer tests
# ---------------------------------------------------------------------------


class TestAttackExplainer:
    def test_explain_backdoor_basic(self):
        from ai.explainability.attack_explainer import AttackExplainer
        from ai.fl_core.schemas import AttackExplanation

        ae = AttackExplainer()
        result = ae.explain_backdoor(
            attack_config={
                "attack_type": "BadNets",
                "target_label": 7,
                "trigger_type": "pixel_block",
                "trigger_value": 6.0,
                "poison_fraction": 0.1,
            }
        )
        assert isinstance(result, AttackExplanation)
        assert result.attack_type == "BadNets"
        assert result.target_label == 7

    def test_explain_backdoor_no_config(self):
        from ai.explainability.attack_explainer import AttackExplainer
        from ai.fl_core.schemas import AttackExplanation

        ae = AttackExplainer()
        result = ae.explain_backdoor()
        assert isinstance(result, AttackExplanation)
        assert result.attack_type == "unknown"

    def test_trigger_description_pixel_block(self):
        from ai.explainability.attack_explainer import AttackExplainer

        desc = AttackExplainer.trigger_description("pixel_block", 6.0, 7)
        assert "pixel" in desc.lower()
        assert "7" in desc

    def test_trigger_description_blended(self):
        from ai.explainability.attack_explainer import AttackExplainer

        desc = AttackExplainer.trigger_description("blended", 0.2, 3)
        assert "blend" in desc.lower()
        assert "3" in desc

    def test_trigger_description_unknown(self):
        from ai.explainability.attack_explainer import AttackExplainer

        desc = AttackExplainer.trigger_description("unknown_type", None, 1)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_poison_ratio_low(self):
        from ai.explainability.attack_explainer import AttackExplainer

        r = AttackExplainer.poison_ratio_analysis(20, 1000)
        assert r["severity"] == "low"
        assert r["poison_fraction"] == pytest.approx(0.02)

    def test_poison_ratio_medium(self):
        from ai.explainability.attack_explainer import AttackExplainer

        r = AttackExplainer.poison_ratio_analysis(100, 1000)
        assert r["severity"] == "medium"

    def test_poison_ratio_high(self):
        from ai.explainability.attack_explainer import AttackExplainer

        r = AttackExplainer.poison_ratio_analysis(300, 1000)
        assert r["severity"] == "high"

    def test_poison_ratio_zero_total(self):
        from ai.explainability.attack_explainer import AttackExplainer

        r = AttackExplainer.poison_ratio_analysis(0, 0)
        assert r["severity"] == "unknown"

    def test_evidence_summary_non_empty(self):
        from ai.explainability.attack_explainer import AttackExplainer

        ae = AttackExplainer()
        result = ae.explain_backdoor(
            attack_config={"attack_type": "BadNets", "target_label": 2}
        )
        assert len(result.evidence_summary) > 10

    def test_suspected_clients_from_entries(self):
        from ai.explainability.attack_explainer import AttackExplainer

        ae = AttackExplainer()
        entries = [
            {"subject_id": "c01", "subject_type": "client", "score": 0.8},
            {"subject_id": "c02", "subject_type": "client", "score": 0.7},
        ]
        result = ae.explain_backdoor(ledger_entries=entries)
        assert "c01" in result.suspected_clients or "c02" in result.suspected_clients


# ---------------------------------------------------------------------------
# ChartGenerator tests
# ---------------------------------------------------------------------------


class TestChartGenerator:
    def _make_shap_exp(self, n: int = 5):
        from ai.fl_core.schemas import SHAPExplanation

        return SHAPExplanation(
            input_id="x0",
            predicted_class=0,
            base_value=0.33,
            shap_values=[round(v, 3) for v in np.random.default_rng(0).uniform(-1, 1, n).tolist()],
            feature_names=[f"f{i}" for i in range(n)],
            top_k_features=[
                {"rank": i + 1, "name": f"f{i}", "shap_value": 0.1 * (n - i)}
                for i in range(n)
            ],
        )

    def _make_fi_result(self, n: int = 5):
        from ai.fl_core.schemas import FeatureImportanceResult

        scores = list(reversed([i * 0.1 for i in range(n)]))
        return FeatureImportanceResult(
            method="permutation",
            feature_names=[f"f{i}" for i in range(n)],
            importance_scores=scores,
            ranked_features=[{"rank": i + 1, "name": f"f{i}", "score": scores[i]} for i in range(n)],
        )

    def _make_trust_exp(self):
        from ai.fl_core.schemas import TrustExplanation

        return TrustExplanation(
            client_id="c01",
            current_score=0.4,
            is_suspicious=True,
            narrative="Suspicious client.",
            score_trajectory=[
                {"round_num": i, "n_flags": i % 3, "layers": ["L1"]}
                for i in range(1, 6)
            ],
        )

    def test_shap_bar_chart_returns_artifact(self):
        from ai.explainability.chart_generator import ChartGenerator
        from ai.fl_core.schemas import ChartArtifact

        cg = ChartGenerator()
        result = cg.shap_bar_chart(self._make_shap_exp(), title="Test SHAP")
        assert isinstance(result, ChartArtifact)
        assert result.chart_type == "shap_bar"

    def test_shap_bar_chart_has_png_or_empty(self):
        from ai.explainability.chart_generator import ChartGenerator

        cg = ChartGenerator()
        result = cg.shap_bar_chart(self._make_shap_exp())
        # Either has valid b64 or empty (matplotlib missing)
        assert isinstance(result.png_b64, str)

    def test_feature_importance_chart_returns_artifact(self):
        from ai.explainability.chart_generator import ChartGenerator
        from ai.fl_core.schemas import ChartArtifact

        cg = ChartGenerator()
        result = cg.feature_importance_chart(self._make_fi_result())
        assert isinstance(result, ChartArtifact)
        assert result.chart_type == "feature_importance"

    def test_trust_trajectory_chart_returns_artifact(self):
        from ai.explainability.chart_generator import ChartGenerator
        from ai.fl_core.schemas import ChartArtifact

        cg = ChartGenerator()
        result = cg.trust_trajectory_chart(self._make_trust_exp())
        assert isinstance(result, ChartArtifact)
        assert result.chart_type == "trust_trajectory"

    def test_trust_trajectory_empty_traj(self):
        from ai.explainability.chart_generator import ChartGenerator
        from ai.fl_core.schemas import TrustExplanation

        cg = ChartGenerator()
        exp = TrustExplanation(
            client_id="c00",
            current_score=1.0,
            is_suspicious=False,
            narrative="Clean.",
            score_trajectory=[],
        )
        result = cg.trust_trajectory_chart(exp)
        assert isinstance(result.png_b64, str)

    def test_reputation_heatmap_returns_artifact(self):
        from ai.explainability.chart_generator import ChartGenerator
        from ai.fl_core.schemas import ChartArtifact

        cg = ChartGenerator()
        heatmap_data = {
            "c01": {"L1": 3, "L2": 1},
            "c02": {"L1": 0, "L2": 2},
        }
        result = cg.reputation_heatmap_chart(heatmap_data)
        assert isinstance(result, ChartArtifact)
        assert result.chart_type == "reputation_heatmap"

    def test_alert_timeline_returns_artifact(self):
        from ai.explainability.chart_generator import ChartGenerator
        from ai.fl_core.schemas import ChartArtifact

        cg = ChartGenerator()
        alerts = [
            {"round_num": 1, "alert_severity": "low"},
            {"round_num": 3, "alert_severity": "high"},
            {"round_num": 5, "alert_severity": "medium"},
        ]
        result = cg.alert_timeline_chart(alerts)
        assert isinstance(result, ChartArtifact)
        assert result.chart_type == "alert_timeline"

    def test_save_all_writes_files(self, tmp_path):
        from ai.explainability.chart_generator import ChartGenerator
        from ai.fl_core.schemas import ChartArtifact

        cg = ChartGenerator()
        # Create an artifact with a real tiny PNG (1x1 white pixel)
        tiny_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
            b"\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        import base64
        b64 = base64.b64encode(tiny_png).decode()
        art = ChartArtifact(chart_type="shap_bar", png_b64=b64, title="test")
        paths = cg.save_all([art], output_dir=tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_save_all_skips_empty_b64(self, tmp_path):
        from ai.explainability.chart_generator import ChartGenerator
        from ai.fl_core.schemas import ChartArtifact

        cg = ChartGenerator()
        art = ChartArtifact(chart_type="shap_bar", png_b64="", title="empty")
        paths = cg.save_all([art], output_dir=tmp_path)
        assert len(paths) == 0


# ---------------------------------------------------------------------------
# JSON round-trip integration
# ---------------------------------------------------------------------------


class TestJSONRoundTrip:
    def test_detection_explanation_json(self):
        from ai.fl_core.schemas import DetectionExplanation

        exp = DetectionExplanation(
            entry_id="e001",
            layer_id="L1",
            subject_id="c01",
            subject_type="client",
            reason_string="High norm.",
            structured_evidence={"l2_norm": 4.5, "cosine_sim": 0.92},
        )
        data = json.loads(exp.model_dump_json())
        assert data["entry_id"] == "e001"
        assert data["structured_evidence"]["l2_norm"] == 4.5

    def test_attack_explanation_json(self):
        from ai.fl_core.schemas import AttackExplanation

        a = AttackExplanation(
            attack_type="BadNets",
            target_label=3,
            trigger_description="pixel block at corner",
            poison_fraction=0.1,
            detection_confidence=0.85,
        )
        data = json.loads(a.model_dump_json())
        assert data["attack_type"] == "BadNets"
        assert math.isclose(data["detection_confidence"], 0.85, rel_tol=1e-5)

    def test_trust_explanation_json(self):
        from ai.fl_core.schemas import TrustExplanation

        t = TrustExplanation(
            client_id="c42",
            current_score=0.55,
            is_suspicious=False,
            narrative="Looks clean.",
            layer_breakdown={"L1": 1},
        )
        data = json.loads(t.model_dump_json())
        assert data["client_id"] == "c42"
        assert data["layer_breakdown"] == {"L1": 1}
