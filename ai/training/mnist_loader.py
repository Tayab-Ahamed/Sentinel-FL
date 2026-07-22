"""
ai/training/mnist_loader.py — Backward-compatible shim.

The full MNIST implementation has moved to
``ai/training/datasets/mnist.py`` (Milestone 3).

This module re-exports ``MNISTDatasetLoader`` from the new location so all
existing callers (``ai/fl_engine/simulation.py``, tests, etc.) continue to
work without modification.

.. deprecated::
    Import directly from ``ai.training.datasets.mnist`` or
    ``ai.training`` in new code.
"""

from __future__ import annotations

# Re-export from the canonical location
from ai.training.datasets.mnist import MNISTDatasetLoader

__all__ = ["MNISTDatasetLoader"]
