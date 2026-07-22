"""
ai/fl_engine/__init__.py — FL engine package re-exports.
"""

from ai.fl_engine.client import MNISTFlowerClient
from ai.fl_engine.simulation import run_simulation
from ai.fl_engine.strategy import SentinelFedAvg

__all__ = [
    "MNISTFlowerClient",
    "SentinelFedAvg",
    "run_simulation",
]
