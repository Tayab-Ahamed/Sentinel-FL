"""
ai/remediation/adapters.py — Model backend adapters for the Remediation Engine.

The Remediation Engine (``remediation_engine.py``) is deliberately model-agnostic:
it reasons about *parameters*, *predictions*, and *fine-tuning* through the
``ModelAdapter`` protocol below, never about a concrete tensor backend.  This keeps
the escalation policy (rollback → unlearning → fine-pruning) identical across:

  * **Phase 0** — the pure-NumPy ``LinearSoftmaxModel`` reference implementation
    (``ai/fl_core/fl_engine.py``), used for the runnable proof-of-concept and the
    full deterministic test-suite.
  * **Phase 1** — PyTorch CNNs through ``TorchModelAdapter``.  Torch is imported
    lazily, so Phase-0 installations remain dependency-light and import-safe.

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

    def fine_tune(self, params: Any, X: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> Any:
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


class TorchModelAdapter:
    """Generic :class:`ModelAdapter` for PyTorch classifiers.

    The adapter exchanges parameters as ``list[np.ndarray]`` (Flower's native
    convention), owns all device movement, and never mutates caller-owned arrays.
    PyTorch is imported only when the adapter is instantiated.

    Args:
        model_factory: Zero-argument callable returning a fresh ``nn.Module``.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"`` (default).
        batch_size: Mini-batch size for prediction and fine-tuning.
        optimizer: ``"sgd"`` or ``"adam"``.
        momentum: SGD momentum.
        weight_decay: Optimizer L2 regularisation.
        architecture: Human-readable model architecture tag.
    """

    def __init__(
        self,
        model_factory: Any,
        device: str = "auto",
        batch_size: int = 64,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        architecture: str = "pytorch_cnn",
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "TorchModelAdapter requires the 'phase1' extra: pip install -e '.[phase1]'"
            ) from exc
        if not callable(model_factory):
            raise TypeError("model_factory must be callable")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'")
        self._torch = torch
        self._model_factory = model_factory
        self._device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        self._batch_size = max(1, int(batch_size))
        self._optimizer = optimizer
        self._momentum = float(momentum)
        self._weight_decay = float(weight_decay)
        self.architecture = str(architecture)

    def clone(self, params: Any) -> list[np.ndarray]:
        return [np.array(p, copy=True) for p in params]

    def _materialize(self, params: Any) -> Any:
        model = self._model_factory().to(self._device)
        arrays = list(params)
        model_params = list(model.parameters())
        if len(arrays) != len(model_params):
            raise ValueError(
                f"parameter count mismatch: model expects {len(model_params)}, got {len(arrays)}"
            )
        with self._torch.no_grad():
            for tensor, arr in zip(model_params, arrays, strict=True):
                value = self._torch.as_tensor(arr, dtype=tensor.dtype, device=self._device)
                if tuple(value.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"parameter shape mismatch: expected {tuple(tensor.shape)}, "
                        f"got {tuple(value.shape)}"
                    )
                tensor.copy_(value)
        return model

    @staticmethod
    def _extract(model: Any) -> list[np.ndarray]:
        return [p.detach().cpu().numpy().copy() for p in model.parameters()]

    def _input_tensor(self, X: np.ndarray) -> Any:
        return self._torch.as_tensor(np.asarray(X), dtype=self._torch.float32, device=self._device)

    def predict(self, params: Any, X: np.ndarray) -> np.ndarray:
        model = self._materialize(params)
        model.eval()
        X_t = self._input_tensor(X)
        outputs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(X_t), self._batch_size):
                logits = model(X_t[start : start + self._batch_size])
                outputs.append(logits.argmax(dim=1).cpu().numpy())
        if not outputs:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(outputs).astype(np.int64, copy=False)

    def predict_proba(self, params: Any, X: np.ndarray) -> np.ndarray:
        model = self._materialize(params)
        model.eval()
        X_t = self._input_tensor(X)
        outputs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(X_t), self._batch_size):
                logits = model(X_t[start : start + self._batch_size])
                outputs.append(self._torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, 0), dtype=float)

    def fine_tune(
        self, params: Any, X: np.ndarray, y: np.ndarray, epochs: int, lr: float
    ) -> list[np.ndarray]:
        if len(X) == 0 or int(epochs) <= 0:
            return self.clone(params)
        model = self._materialize(params)
        model.train()
        X_t = self._input_tensor(X)
        y_t = self._torch.as_tensor(np.asarray(y), dtype=self._torch.long, device=self._device)
        if self._optimizer == "adam":
            optimiser = self._torch.optim.Adam(
                model.parameters(), lr=float(lr), weight_decay=self._weight_decay
            )
        else:
            optimiser = self._torch.optim.SGD(
                model.parameters(),
                lr=float(lr),
                momentum=self._momentum,
                weight_decay=self._weight_decay,
            )
        generator = self._torch.Generator(device="cpu").manual_seed(42)
        for _ in range(int(epochs)):
            order = self._torch.randperm(len(X_t), generator=generator)
            for start in range(0, len(order), self._batch_size):
                idx = order[start : start + self._batch_size].to(self._device)
                optimiser.zero_grad(set_to_none=True)
                loss = self._torch.nn.functional.nll_loss(model(X_t[idx]), y_t[idx])
                if not self._torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss during remediation fine-tuning")
                loss.backward()
                self._torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimiser.step()
        return self._extract(model)

    def prune_dormant_channels(
        self,
        params: Any,
        X_clean: np.ndarray,
        prune_fraction: float = 0.1,
        layer_name: str | None = None,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        """Zero channels with the lowest clean activation in the last Conv2d layer.

        Dormant-on-clean channels are the classic fine-pruning candidates: they can
        encode a backdoor while contributing little to normal predictions.  The
        returned evidence is JSON-safe for audit reports.
        """
        model = self._materialize(params)
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, self._torch.nn.Conv2d)]
        if not convs:
            raise TypeError("fine-pruning requires at least one Conv2d layer")
        if layer_name is None:
            chosen_name, layer = convs[-1]
        else:
            matches = [(n, m) for n, m in convs if n == layer_name]
            if not matches:
                raise ValueError(f"Conv2d layer '{layer_name}' not found")
            chosen_name, layer = matches[0]

        activations: list[Any] = []
        handle = layer.register_forward_hook(
            lambda _m, _i, output: activations.append(output.detach().abs().mean(dim=(0, 2, 3)))
        )
        model.eval()
        X_t = self._input_tensor(X_clean)
        with self._torch.inference_mode():
            for start in range(0, len(X_t), self._batch_size):
                model(X_t[start : start + self._batch_size])
        handle.remove()
        if not activations:
            raise ValueError("clean calibration set is empty")
        scores = self._torch.stack(activations).mean(dim=0)
        n_channels = int(scores.numel())
        fraction = min(max(float(prune_fraction), 0.0), 0.5)
        n_prune = max(1, round(n_channels * fraction)) if fraction > 0 else 0
        indices = self._torch.argsort(scores)[:n_prune]
        with self._torch.no_grad():
            layer.weight[indices] = 0
            if layer.bias is not None:
                layer.bias[indices] = 0
        evidence = {
            "layer": chosen_name,
            "channels_total": n_channels,
            "channels_pruned": n_prune,
            "pruned_indices": [int(i) for i in indices.cpu().tolist()],
            "mean_clean_activation": float(scores.mean().cpu()),
        }
        return self._extract(model), evidence
