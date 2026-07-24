"""
ai/explainability/feature_importance.py — Feature importance utilities.

Three methods:
    permutation_importance  — drop-column shuffle (sklearn-compatible).
    coefficient_importance  — |coef| for linear models.
    gradient_feature_importance — gradient magnitude from ClientUpdate.delta.

All return ``FeatureImportanceResult`` (SCHEMAS.md §FeatureImportanceResult).

Public surface:
    permutation_importance(model, X, y, n_repeats, feature_names, context)
        → FeatureImportanceResult
    coefficient_importance(model, feature_names, context)
        → FeatureImportanceResult
    gradient_feature_importance(delta, feature_names, context)
        → FeatureImportanceResult
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai.fl_core.schemas import FeatureImportanceResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_ranked(feature_names: list[str], scores: list[float]) -> list[dict[str, Any]]:
    """Return features sorted by score descending with rank."""
    pairs = sorted(enumerate(scores), key=lambda iv: iv[1], reverse=True)
    return [
        {"rank": rank + 1, "name": feature_names[i], "score": round(scores[i], 6)}
        for rank, (i, _) in enumerate(pairs)
    ]


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


def permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    feature_names: list[str] | None = None,
    context: str = "",
    random_state: int = 0,
) -> FeatureImportanceResult:
    """Compute permutation feature importance.

    For each feature j, randomly shuffle column j ``n_repeats`` times and
    measure the mean drop in accuracy.  A larger drop = more important feature.

    Args:
        model: Any object with a ``predict(X)`` or ``predict_proba(X)`` method.
        X: 2-D array ``(n_samples, n_features)``.
        y: 1-D ground-truth labels.
        n_repeats: Number of shuffle repeats per feature.
        feature_names: Feature name list.  Defaults to ``['f0', ...]``.
        context: Optional description string.
        random_state: RNG seed.

    Returns:
        FeatureImportanceResult with ``method='permutation'``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_features = X.shape[1]
    names = feature_names or [f"f{i}" for i in range(n_features)]
    rng = np.random.default_rng(random_state)

    # Baseline accuracy
    if hasattr(model, "predict"):
        baseline_acc = float(np.mean(model.predict(X) == y))
    else:
        raise ValueError("model must have a predict() method.")

    importances = np.zeros(n_features)
    for j in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_acc = float(np.mean(model.predict(X_perm) == y))
            drops.append(baseline_acc - perm_acc)
        importances[j] = float(np.mean(drops))

    scores = importances.tolist()
    return FeatureImportanceResult(
        method="permutation",
        feature_names=names,
        importance_scores=scores,
        ranked_features=_build_ranked(names, scores),
        context=context,
    )


# ---------------------------------------------------------------------------
# Coefficient importance
# ---------------------------------------------------------------------------


def coefficient_importance(
    model: Any,
    feature_names: list[str] | None = None,
    context: str = "",
) -> FeatureImportanceResult:
    """Extract feature importance from the absolute value of model coefficients.

    Works for any model with a ``coef_`` attribute (sklearn LogisticRegression,
    LinearSVC, Ridge, etc.).  For multi-class models, takes the mean of
    ``|coef_|`` across classes.

    Args:
        model: Linear model with ``coef_`` attribute.
        feature_names: Feature names.
        context: Optional description.

    Returns:
        FeatureImportanceResult with ``method='coefficient'``.

    Raises:
        ValueError: If model has no ``coef_`` attribute.
    """
    if not hasattr(model, "coef_"):
        raise ValueError(
            f"coefficient_importance: model '{type(model).__name__}' has no coef_ "
            "attribute.  Use permutation_importance for non-linear models."
        )
    coef = np.asarray(model.coef_)
    if coef.ndim == 2:
        # multi-class: mean |coef| across classes
        importance = np.mean(np.abs(coef), axis=0)
    else:
        importance = np.abs(coef)

    n_features = len(importance)
    names = feature_names or [f"f{i}" for i in range(n_features)]
    scores = importance.tolist()
    return FeatureImportanceResult(
        method="coefficient",
        feature_names=names,
        importance_scores=scores,
        ranked_features=_build_ranked(names, scores),
        context=context,
    )


# ---------------------------------------------------------------------------
# Gradient feature importance  (L1 — ClientUpdate.delta)
# ---------------------------------------------------------------------------


def gradient_feature_importance(
    delta: list[float] | np.ndarray,
    feature_names: list[str] | None = None,
    context: str = "",
    aggregate: str = "abs_mean",
    n_params_per_feature: int | None = None,
) -> FeatureImportanceResult:
    """Map a client update gradient to per-feature importance scores.

    For a flat ``ClientUpdate.delta``, the gradient magnitude (abs value)
    approximates how much each weight changed.  If the model has
    ``n_params_per_feature`` parameters per input feature (e.g. one weight
    per feature in a linear model), consecutive blocks are averaged.

    Args:
        delta: Flattened model parameter delta from ``ClientUpdate.delta``.
        feature_names: Feature names.  Defaults to parameter indices.
        context: Optional description (e.g. 'L1 round 4 client_02').
        aggregate: How to aggregate parameter blocks: ``'abs_mean'`` (default)
            or ``'max_abs'``.
        n_params_per_feature: If set, ``delta`` is split into blocks of this
            size, one per feature.  Otherwise each parameter is treated as one
            feature.

    Returns:
        FeatureImportanceResult with ``method='gradient'``.
    """
    arr = np.asarray(delta, dtype=np.float64).ravel()
    if n_params_per_feature and n_params_per_feature > 1:
        n_feat = len(arr) // n_params_per_feature
        remainder = len(arr) % n_params_per_feature
        arr_trimmed = arr[: n_feat * n_params_per_feature]
        blocks = arr_trimmed.reshape(n_feat, n_params_per_feature)
        if aggregate == "max_abs":
            importance = np.max(np.abs(blocks), axis=1)
        else:
            importance = np.mean(np.abs(blocks), axis=1)
        if remainder:
            logger.debug(
                "gradient_feature_importance: %d trailing parameters ignored "
                "(not divisible by n_params_per_feature=%d).",
                remainder,
                n_params_per_feature,
            )
    else:
        importance = np.abs(arr)
        n_feat = len(arr)

    names = feature_names or [f"p{i}" for i in range(n_feat)]
    # Clip names to match importance length
    names = (
        names[:n_feat]
        if len(names) >= n_feat
        else (names + [f"p{i}" for i in range(len(names), n_feat)])
    )
    scores = importance.tolist()
    return FeatureImportanceResult(
        method="gradient",
        feature_names=names,
        importance_scores=scores,
        ranked_features=_build_ranked(names, scores),
        context=context,
    )
