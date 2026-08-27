"""
fl_engine.py — minimal, dependency-light federated learning engine.

This is a pure NumPy reference implementation used to (a) validate the SENTINEL-FL
algorithm design end-to-end and (b) serve as the runnable proof-of-concept for Phase 0.
The full ML path targets PyTorch + Flower (see ai/models/ and ai/fl_engine/) for deep learning
models and benchmark image datasets; the core algorithms (Multi-Krum selection,
residual-collusion clustering, unlearning) operate across both backends.

Model: multinomial logistic regression (softmax classifier). Kept simple deliberately
so the FL/defense logic is clean, fast, and transparent. Swapping in a CNN under PyTorch
does not change any of the defense-layer architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class LinearSoftmaxModel:
    """W: (n_classes, n_features), b: (n_classes,) — flattened for FL transport."""

    n_features: int
    n_classes: int
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(0)
        self.W = rng.normal(0, 0.01, size=(self.n_classes, self.n_features))
        self.b = np.zeros(self.n_classes)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    def set_params(self, flat: np.ndarray) -> None:
        n_w = self.n_classes * self.n_features
        self.W = flat[:n_w].reshape(self.n_classes, self.n_features).copy()
        self.b = flat[n_w:].copy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W.T + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def local_train(
    model_params: np.ndarray,
    n_features: int,
    n_classes: int,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 5,
    lr: float = 0.1,
) -> np.ndarray:
    """One client's local SGD training. Returns updated flat params (not the delta;
    caller computes delta = new - old, matching standard FL update semantics)."""
    model = LinearSoftmaxModel(n_features, n_classes)
    model.set_params(model_params.copy())
    n = X.shape[0]
    y_onehot = np.eye(n_classes)[y]
    for _ in range(epochs):
        idx = np.random.permutation(n)
        Xs, ys = X[idx], y_onehot[idx]
        probs = softmax(Xs @ model.W.T + model.b)
        grad_logits = (probs - ys) / n  # (n, n_classes)
        grad_W = grad_logits.T @ Xs
        grad_b = grad_logits.sum(axis=0)
        model.W -= lr * grad_W
        model.b -= lr * grad_b
    return model.get_params()


def fedavg(client_params: list[np.ndarray], weights: list[float]) -> np.ndarray:
    weights = np.array(weights) / np.sum(weights)
    stacked = np.stack(client_params, axis=0)
    return (stacked * weights[:, None]).sum(axis=0)


def multi_krum(
    client_updates: list[np.ndarray], num_malicious_assumed: int, num_to_select: int
) -> tuple[np.ndarray, list[int]]:
    """Blanchard et al. 2017 Multi-Krum, applied to raw update vectors (deltas).
    Reference behavior matches Flower's flwr.serverapp.strategy.MultiKrum semantics
    (see RESEARCH.md for why we defer to Flower's implementation in the production
    path rather than re-deriving Krum ourselves there)."""
    n = len(client_updates)
    f = num_malicious_assumed
    stacked = np.stack(client_updates, axis=0)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.sum((stacked[i] - stacked[j]) ** 2)
    scores = []
    n_closest = max(1, n - f - 2)
    for i in range(n):
        d = np.sort(dists[i])[1 : 1 + n_closest]  # exclude self (distance 0)
        scores.append(d.sum())
    order = np.argsort(scores)
    selected = sorted(order[:num_to_select].tolist())
    selected_updates = [client_updates[i] for i in selected]
    aggregate = np.mean(selected_updates, axis=0)
    return aggregate, selected
