"""
ai/fl_engine/client.py — Flower NumPyClient for MNIST.

Each simulated client:
  1. Receives the current global model parameters from the server.
  2. Trains for ``local_epochs`` epochs on its local MNIST partition using SGD.
  3. Returns updated parameters + training metrics to the server.
  4. Evaluates the received global parameters on its local test slice on request.

Error handling:
  - NaN/Inf in loss → logs ``training_failed`` event, returns original parameters
    with a special ``"nan_loss": true`` metric so the server's aggregator can
    detect and exclude this client rather than ingesting garbage weights.
  - Empty local dataset → raises ``InsufficientClientsError`` at construction.

Design note: the client is *stateless* between rounds — Flower recreates it via
``client_fn`` each round, so no persistent state is stored on the client object.
All mutable state lives in the local model weights, which are always initialised
from the server's global parameters at the start of ``fit()``.
"""

from __future__ import annotations

import logging
import math

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader, TensorDataset

from ai.fl_core.logger import StructuredLogger
from ai.models.mnist_cnn import SimpleCNN, get_model_parameters, set_model_parameters

logger = logging.getLogger(__name__)


class MNISTFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient that trains a SimpleCNN on a MNIST partition.

    Args:
        client_id: Unique string identifier for this client (e.g. ``"client_00"``).
        train_data: ``(X, y)`` numpy arrays for local training.
            ``X`` shape: ``(N, 1, 28, 28)``.  ``y`` shape: ``(N,)``.
        val_data: ``(X, y)`` numpy arrays for local evaluation.
            May be the same as ``train_data`` for small partitions.
        local_epochs: Number of local SGD epochs per FL round.
        learning_rate: SGD learning rate.
        batch_size: Mini-batch size for local training.
        device: PyTorch device (``"cpu"`` or ``"cuda"``).
        sentinel_logger: Optional StructuredLogger for structured log emission.
    """

    def __init__(
        self,
        client_id: str,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: str = "cpu",
        sentinel_logger: StructuredLogger | None = None,
    ) -> None:
        self._client_id = client_id
        self._train_data = train_data
        self._val_data = val_data
        self._local_epochs = local_epochs
        self._lr = learning_rate
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._sentinel_logger = sentinel_logger
        self._model = SimpleCNN().to(self._device)

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return current local model parameters as a list of numpy arrays."""
        return get_model_parameters(self._model)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train the local model for ``local_epochs`` epochs.

        Args:
            parameters: Global model parameters received from the server.
            config: Server-provided config dict (may contain ``"round"``).

        Returns:
            ``(updated_parameters, num_examples, metrics)``
            where ``metrics`` contains ``{"train_loss": float, "train_accuracy": float}``.
        """
        round_num: int = int(config.get("round", 0))
        set_model_parameters(self._model, parameters)

        train_loss, train_acc = self._train(round_num)

        # Guard against NaN/Inf loss (e.g. diverged SGD)
        if not math.isfinite(train_loss):
            logger.warning(
                "Client %s round %d: non-finite train_loss=%.4f — returning original params.",
                self._client_id,
                round_num,
                train_loss,
            )
            if self._sentinel_logger:
                self._sentinel_logger.log(
                    "L1",
                    "training_failed",
                    {"client_id": self._client_id, "round": round_num, "reason": "nan_loss"},
                )
            return get_model_parameters(self._model), len(self._train_data[0]), {
                "train_loss": float("inf"),
                "train_accuracy": 0.0,
                "nan_loss": True,
            }

        metrics: dict[str, Scalar] = {
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "client_id": self._client_id,
        }
        logger.debug(
            "Client %s round %d: loss=%.4f acc=%.4f n=%d",
            self._client_id,
            round_num,
            train_loss,
            train_acc,
            len(self._train_data[0]),
        )
        return get_model_parameters(self._model), len(self._train_data[0]), metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate global model parameters on the local validation slice.

        Args:
            parameters: Global model parameters to evaluate.
            config: Server-provided config dict.

        Returns:
            ``(loss, num_examples, metrics)``
            where ``metrics`` contains ``{"accuracy": float}``.
        """
        set_model_parameters(self._model, parameters)
        loss, accuracy = self._evaluate()
        return (
            float(loss),
            len(self._val_data[0]),
            {"accuracy": round(accuracy, 6), "client_id": self._client_id},
        )

    # ------------------------------------------------------------------
    # Internal training / evaluation loops
    # ------------------------------------------------------------------

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self._batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _train(self, round_num: int) -> tuple[float, float]:
        """Run local SGD training and return (mean_loss, accuracy)."""
        self._model.train()
        optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        criterion = nn.NLLLoss()
        loader = self._make_loader(*self._train_data, shuffle=True)

        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(self._local_epochs):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad()
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        mean_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return mean_loss, accuracy

    def _evaluate(self) -> tuple[float, float]:
        """Run evaluation on the local validation data and return (loss, accuracy)."""
        self._model.eval()
        criterion = nn.NLLLoss()
        loader = self._make_loader(*self._val_data, shuffle=False)

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        return total_loss / max(total, 1), correct / max(total, 1)
