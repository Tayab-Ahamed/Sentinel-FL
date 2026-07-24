"""
ai/remediation/adapters.py — Model backend adapters for the Remediation Engine.

The Remediation Engine (``remediation_engine.py``) is deliberately model-agnostic:
it reasons about *parameters*, *predictions*, and *fine-tuning* through the
``ModelAdapter`` protocol below, never about a concrete tensor backend.  This keeps
the escalation policy (rollback → unlearning → fine-pruning) identical across:

  * **Phase 0** — the pure-NumPy ``LinearSoftmaxModel`` reference implementation
    (``ai/fl_core/fl_engine.py``), used for the runnable proof-of-concept and the
    full deterministic test-suite.
  * **Phase 1** — a PyTorch CNN (``ai/models/mnist_cnn.py``); a
    ``TorchModelAdapter`` can be dropped in without touching any remediation logic
    (see ARCHITECTURE.md §7.4 and §7.12).

Design contract (mirrors INTERFACES.md conventions):
  * ``params`` is an opaque, backend-specific model-state object. For Phase 0 it is
    a flat ``np.ndarray`` (``LinearSoftmaxModel.get_params()``); adapters must copy
    rather than mutate the params they are handed.
  * ``predict(params, X) -> np.ndarray`` returns int64 predicted labels.
  * ``fine_tune(params, X, y, epochs, lr) -> params`` returns *new* params.
  * ``clone(params) -> params`` returns a defensive deep copy.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ai.fl_core.fl_engine import LinearSoftmaxModel, local_train


@runtime_checkable
class ModelAdapter(Protocol):
    """Backend-agnostic view of a model used by the Remediation Engine."""

    #: Architecture tag mirrored into ``ModelMetadata.architecture``.
    architecture: str

    def clone(self, params: Any) -> Any:
        """Return a defensive deep copy of ``params``."""
        ...

    def predict(self, params: Any, X: np.ndarray) -> np.ndarray:
        """Return int64 predicted labels for ``X`` under ``params``."""
        ...

    def fine_tune(
        self, params: Any, X: np.ndarray, y: np.ndarray, epochs: int, lr: float
    ) -> Any:
        """Fine-tune ``params`` on ``(X, y)`` and return the updated params."""
        ...


class LinearSoftmaxAdapter:
    """``ModelAdapter`` for the Phase 0 :class:`LinearSoftmaxModel`.

    Wraps the pure-NumPy softmax classifier and its ``local_train`` SGD routine so
    the Remediation Engine can rollback, unlearn, and fine-prune without importing
    any FL-engine internals directly.

    Args:
        n_features: Input feature dimensionality.
        n_classes: Number of output classes.
    """

    architecture: str = "linear_softmax_v0"

    def __init__(self, n_features: int, n_classes: int) -> None:
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)

    # ------------------------------------------------------------------
    # ModelAdapter protocol
    # ------------------------------------------------------------------

    def clone(self, params: Any) -> np.ndarray:
        return np.array(params, dtype=float, copy=True)

    def predict(self, params: Any, X: np.ndarray) -> np.ndarray:
        model = self._materialize(params)
        return model.predict(np.asarray(X, dtype=float)).astype(np.int64)

    def predict_proba(self, params: Any, X: np.ndarray) -> np.ndarray:
        model = self._materialize(params)
        return model.predict_proba(np.asarray(X, dtype=float))

    def fine_tune(
        self, params: Any, X: np.ndarray, y: np.ndarray, epochs: int, lr: float
    ) -> np.ndarray:
        if len(X) == 0:
            return self.clone(params)
        return local_train(
            np.asarray(params, dtype=float),
            self.n_features,
            self.n_classes,
            np.asarray(X, dtype=float),
            np.asarray(y).astype(int),
            epochs=int(epochs),
            lr=float(lr),
        )

    # ------------------------------------------------------------------
    # Linear-model specifics used by fine-pruning
    # ------------------------------------------------------------------

    def get_weight_matrix(self, params: Any) -> np.ndarray:
        """Return the ``(n_classes, n_features)`` weight matrix ``W``."""
        model = self._materialize(params)
        return model.W.copy()

    def set_weight_matrix(self, params: Any, W: np.ndarray) -> np.ndarray:
        """Return new flat params with ``W`` substituted (bias preserved)."""
        model = self._materialize(params)
        model.W = np.asarray(W, dtype=float).reshape(self.n_classes, self.n_features)
        return model.get_params()

    def _materialize(self, params: Any) -> LinearSoftmaxModel:
        model = LinearSoftmaxModel(self.n_features, self.n_classes)
        model.set_params(np.asarray(params, dtype=float))
        return model
