"""
ai/fl_core — Federated learning core: engine, schemas, interfaces, config, logger.

Public re-exports so callers can do:
    from ai.fl_core import Configuration, StructuredLogger, load_config
"""

from ai.fl_core.config import load_config, load_config_from_dict
from ai.fl_core.exceptions import (
    CheckpointNotFoundError,
    ConfigValidationError,
    DatasetNotFoundError,
    InsufficientCalibrationDataError,
    InsufficientClientsError,
    RemediationFailedError,
    SentinelError,
    UnsupportedModelError,
)
from ai.fl_core.interfaces import (
    Aggregator,
    AttackSimulator,
    DatasetLoader,
    DefenseStrategy,
    Detector,
    Logger,
    MetricsCollector,
    ModelRegistry,
    Visualizer,
)
from ai.fl_core.logger import StructuredLogger, make_logger
from ai.fl_core.schemas import (
    AttackConfig,
    AttackReport,
    AuditReport,
    ClientUpdate,
    Configuration,
    DetectionResult,
    EvaluationResult,
    Experiment,
    LogEntry,
    Metric,
    ModelMetadata,
    ReversedTrigger,
    TrainingRound,
    TrustLedgerEntry,
    TrustScore,
)

__all__ = [
    # Config
    "load_config",
    "load_config_from_dict",
    # Exceptions
    "CheckpointNotFoundError",
    "ConfigValidationError",
    "DatasetNotFoundError",
    "InsufficientCalibrationDataError",
    "InsufficientClientsError",
    "RemediationFailedError",
    "SentinelError",
    "UnsupportedModelError",
    # Interfaces
    "Aggregator",
    "AttackSimulator",
    "DatasetLoader",
    "DefenseStrategy",
    "Detector",
    "Logger",
    "MetricsCollector",
    "ModelRegistry",
    "Visualizer",
    # Logger
    "StructuredLogger",
    "make_logger",
    # Schemas
    "AttackConfig",
    "AttackReport",
    "AuditReport",
    "ClientUpdate",
    "Configuration",
    "DetectionResult",
    "Experiment",
    "EvaluationResult",
    "LogEntry",
    "Metric",
    "ModelMetadata",
    "ReversedTrigger",
    "TrainingRound",
    "TrustLedgerEntry",
    "TrustScore",
]
