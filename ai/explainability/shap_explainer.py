"""
ai/explainability/shap_explainer.py — SHAP-based feature attribution for SENTINEL-FL.

Uses ``shap.KernelExplainer`` (model-agnostic) so this works with the
sklearn ``LogisticRegression`` in Phase 0 and any ``predict_proba`` callable
in Phase 1.

When the ``shap`` package is not installed, ``SHAPExplainer`` falls back to
``permutation_importance`` automatically — callers get a valid
``SHAPExplanation`` regardless; the ``method`` field records which path was taken.

Public surface:
    SHAPExplainer
        fit(model, background_data, feature_names)  — build explainer
        explain_input(x, input_id, predicted_class) → SHAPExplanation
        explain_batch(X, input_ids, predicted_classes) → list[SHAPExplanation]
        top_features(x, k)                          → list[dict]
        is_fitted                                   → bool
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai.fl_core.schemas import SHAPExplanation

logger = logging.getLogger(__name__)

# Lazily imported so the module loads even without shap installed.
_SHAP_AVAILABLE: bool | None = None


def _check_shap() -> bool:
    global _SHAP_AVAILABLE
    if _SHAP_AVAILABLE is None:
        try:
            import shap  # noqa: F401

            _SHAP_AVAILABLE = True
        except ImportError:
            _SHAP_AVAILABLE = False
            logger.warning(
                "SHAPExplainer: `shap` package not installed — "
                "falling back to permutation importance."
            )
    return _SHAP_AVAILABLE


def _build_top_k(
    feature_names: list[str], shap_values: list[float], k: int
) -> list[dict[str, Any]]:
    """Return the top-k features sorted by |shap_value| descending."""
    pairs = sorted(
        enumerate(shap_values), key=lambda iv: abs(iv[1]), reverse=True
    )
    return [
        {"rank": rank + 1, "name": feature_names[i], "shap_value": round(shap_values[i], 6)}
        for rank, (i, _) in enumerate(pairs[:k])
    ]


class SHAPExplainer:
    """Model-agnostic feature attribution via SHAP KernelExplainer (or fallback).

    Args:
        n_background: Number of background samples for KernelExplainer.
        nsamples: Number of samples per SHAP estimation call.
        top_k: Number of top features to include in each explanation.
        random_state: Seed for background sampling reproducibility.
    """

    def __init__(
        self,
        n_background: int = 50,
        nsamples: int = 100,
        top_k: int = 10,
        random_state: int = 42,
    ) -> None:
        self._n_background = n_background
        self._nsamples = nsamples
        self._top_k = top_k
        self._rng = np.random.default_rng(random_state)
        self._model: Any | None = None
        self._background: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._explainer: Any | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """Build the SHAP explainer from a model and clean background data.

        Args:
            model: Any object with a ``predict_proba(X)`` method.
            background_data: 2-D array ``(n_samples, n_features)`` of clean
                reference data used to marginalise out feature effects.
            feature_names: Optional list of feature names.  Defaults to
                ``['f0', 'f1', ...]``.
        """
        bg = np.asarray(background_data, dtype=np.float64)
        n_bg = min(self._n_background, len(bg))
        idx = self._rng.choice(len(bg), size=n_bg, replace=False)
        self._background = bg[idx]
        self._model = model
        n_features = bg.shape[1]
        self._feature_names = feature_names or [f"f{i}" for i in range(n_features)]

        if _check_shap():
            import shap

            self._explainer = shap.KernelExplainer(
                model.predict_proba, self._background
            )
            logger.info(
                "SHAPExplainer: KernelExplainer built (n_background=%d, n_features=%d).",
                n_bg, n_features,
            )
        else:
            self._explainer = None
            logger.info(
                "SHAPExplainer: using permutation fallback (n_background=%d).", n_bg
            )
        self._is_fitted = True

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_input(
        self,
        x: np.ndarray,
        input_id: str = "input",
        predicted_class: int = 0,
    ) -> SHAPExplanation:
        """Explain a single input vector.

        Args:
            x: 1-D feature vector.
            input_id: Identifier for this input.
            predicted_class: The model's argmax prediction.

        Returns:
            SHAPExplanation with per-feature SHAP values and top-k ranking.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "SHAPExplainer.fit() must be called before explain_input()."
            )
        x_2d = np.asarray(x, dtype=np.float64).reshape(1, -1)
        n_features = x_2d.shape[1]
        n_feat = len(self._feature_names)
        names = (
            self._feature_names[:n_features]
            if n_feat >= n_features
            else self._feature_names + [f"f{i}" for i in range(n_feat, n_features)]
        )

        if self._explainer is not None:
            raw = self._explainer.shap_values(x_2d, nsamples=self._nsamples)
            # KernelExplainer returns list (per class) or 2-D array
            if isinstance(raw, list):
                sv = np.asarray(raw[predicted_class]).ravel().tolist()
                base = float(self._explainer.expected_value[predicted_class])
            else:
                sv = np.asarray(raw).ravel().tolist()
                base = float(self._explainer.expected_value)
            method = "kernel_shap"
        else:
            sv, base = self._permutation_shap(x_2d, predicted_class)
            method = "permutation"

        top_k = _build_top_k(names, sv, self._top_k)
        return SHAPExplanation(
            input_id=input_id,
            predicted_class=predicted_class,
            base_value=round(base, 6),
            shap_values=[round(v, 6) for v in sv],
            feature_names=names,
            top_k_features=top_k,
            method=method,
        )

    def explain_batch(
        self,
        X: np.ndarray,
        input_ids: list[str] | None = None,
        predicted_classes: list[int] | None = None,
    ) -> list[SHAPExplanation]:
        """Explain a batch of inputs.

        Args:
            X: 2-D array ``(n_samples, n_features)``.
            input_ids: Optional list of identifiers.
            predicted_classes: Optional list of argmax predictions.

        Returns:
            List of SHAPExplanation objects, one per row.
        """
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        ids = input_ids or [f"input_{i}" for i in range(n)]
        preds = predicted_classes or [0] * n
        return [self.explain_input(X[i], ids[i], preds[i]) for i in range(n)]

    def top_features(
        self,
        x: np.ndarray,
        k: int | None = None,
        input_id: str = "input",
        predicted_class: int = 0,
    ) -> list[dict[str, Any]]:
        """Return the top-k most influential features for a single input.

        Args:
            x: 1-D feature vector.
            k: Override the default top_k.
            input_id: Identifier.
            predicted_class: Argmax prediction.

        Returns:
            List of dicts with ``rank``, ``name``, ``shap_value``.
        """
        exp = self.explain_input(x, input_id=input_id, predicted_class=predicted_class)
        k = k or self._top_k
        return exp.top_k_features[:k]

    # ------------------------------------------------------------------
    # Permutation fallback
    # ------------------------------------------------------------------

    def _permutation_shap(
        self, x_2d: np.ndarray, predicted_class: int
    ) -> tuple[list[float], float]:
        """Approximate SHAP values via feature permutation on background data."""
        assert self._background is not None and self._model is not None
        n_features = x_2d.shape[1]
        sv = np.zeros(n_features)
        base_preds = self._model.predict_proba(self._background)
        base = float(np.mean(base_preds[:, predicted_class]))

        for j in range(n_features):
            perturbed = self._background.copy()
            perturbed[:, j] = x_2d[0, j]
            pred_with = self._model.predict_proba(perturbed)
            sv[j] = float(np.mean(pred_with[:, predicted_class])) - base

        return sv.tolist(), base

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True if ``fit()`` has been called."""
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        """Feature names used at fit time."""
        return list(self._feature_names)
