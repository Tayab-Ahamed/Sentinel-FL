"""
ai/explainability/__init__.py — Explainability package public re-exports.
"""

from ai.explainability.attack_explainer import AttackExplainer
from ai.explainability.chart_generator import ChartGenerator
from ai.explainability.detection_explainer import DetectionExplainer
from ai.explainability.feature_importance import (
    coefficient_importance,
    gradient_feature_importance,
    permutation_importance,
)
from ai.explainability.shap_explainer import SHAPExplainer
from ai.explainability.trust_explainer import TrustExplainer

__all__ = [
    # Explainer classes
    "SHAPExplainer",
    "DetectionExplainer",
    "TrustExplainer",
    "AttackExplainer",
    "ChartGenerator",
    # Feature importance functions
    "permutation_importance",
    "coefficient_importance",
    "gradient_feature_importance",
]
