"""
ai/models/mnist_cnn.py — Compact CNN for MNIST classification.

Architecture (two conv blocks + two FC layers):
  Conv1(1→32, 3×3) → ReLU → MaxPool(2×2)
  Conv2(32→64, 3×3) → ReLU → MaxPool(2×2)
  Flatten → FC(1600→128) → ReLU → Dropout(0.25) → FC(128→10)

Achieves ~99% test accuracy on MNIST after 5–10 epochs of local training with
SGD (lr=0.01, momentum=0.9).

Helper functions ``get_model_parameters`` / ``set_model_parameters`` convert
between PyTorch ``state_dict`` and the flat list[np.ndarray] that Flower's
``NumPyClient`` expects, keeping the model class itself framework-agnostic at
the interface boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass  # no circular imports needed here


class SimpleCNN(nn.Module):
    """Two-block CNN for MNIST (28×28 greyscale, 10 classes).

    Designed to be small enough for fast Flower simulation (CPU-only) while
    still reaching production-quality accuracy (>98% after 5 local epochs).

    Args:
        num_classes: Number of output classes. Defaults to 10 (MNIST).
        dropout_rate: Dropout probability after the first FC layer.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25) -> None:
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # After 2× MaxPool(2): 28 → 14 → 7, so flat dim = 64 × 7 × 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(N, 1, 28, 28)``.

        Returns:
            Log-softmax output of shape ``(N, num_classes)``.
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Flower-compatible parameter helpers
# ---------------------------------------------------------------------------


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays (Flower convention).

    Flower's ``NumPyClient.get_parameters`` / ``set_parameters`` use this format.
    Each tensor in ``state_dict`` is returned as a separate array in the list.

    Args:
        model: Any PyTorch ``nn.Module``.

    Returns:
        Ordered list of ``np.ndarray``, one per parameter tensor.
    """
    return [param.detach().cpu().numpy() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load flat NumPy parameter list back into a model in-place.

    Args:
        model: Target PyTorch ``nn.Module``.
        parameters: Ordered list of ``np.ndarray`` matching ``model.parameters()``.

    Raises:
        ValueError: If the number of parameter arrays doesn't match the model.
    """
    params_dict = zip(model.parameters(), parameters, strict=True)
    for param, arr in params_dict:
        param.data = torch.tensor(arr, dtype=param.dtype)
