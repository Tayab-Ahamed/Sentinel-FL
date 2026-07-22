"""
ai/detection/__init__.py — Detection layer public re-exports.

Imports all detector/strategy classes so callers can do:
    from ai.detection import CollusionGuardStrategy, UpdateGuard, ReputationEngine
"""

from ai.detection.activation_consistency import ActivationConsistencyDetector
from ai.detection.alert_manager import AlertManager
from ai.detection.anomaly_detector import UpdateAnomalyDetector
from ai.detection.confidence_analyzer import (
    batch_confidence_stats,
    confidence_anomaly_score,
    entropy_from_probs,
    softmax_confidence,
    top2_margin,
)
from ai.detection.fusion_classifier import FusionClassifier
from ai.detection.gradient_extractor import GradientExtractor
from ai.detection.inference_monitor import InferenceMonitor
from ai.detection.model_auditor import ModelAuditorDetector, ModelAuditorStrategy
from ai.detection.norm_calculator import (
    compute_l2_norms,
    compute_norm_zscores,
    flag_norm_outliers,
)
from ai.detection.reputation_engine import ReputationEngine
from ai.detection.runtime_sentinel import RuntimeSentinelStrategy, StripEntropyDetector
from ai.detection.trust_ledger import FileTrustLedger
from ai.detection.trust_score_manager import TrustScoreManager
from ai.detection.update_guard import (
    CollusionGuardStrategy,
    UpdateGuard,
    UpdateGuardResult,
    cosine_sim_matrix,
    detect_collusion_clusters,
)

__all__ = [
    # Phase 0 (preserved)
    "CollusionGuardStrategy",
    "cosine_sim_matrix",
    "detect_collusion_clusters",
    # Milestone 5
    "GradientExtractor",
    "UpdateAnomalyDetector",
    "TrustScoreManager",
    "UpdateGuard",
    "UpdateGuardResult",
    # Milestone 6
    "ReputationEngine",
    # Milestone 7
    "AlertManager",
    "FusionClassifier",
    "InferenceMonitor",
    "batch_confidence_stats",
    "confidence_anomaly_score",
    "entropy_from_probs",
    "softmax_confidence",
    "top2_margin",
    # Norm utils
    "compute_l2_norms",
    "compute_norm_zscores",
    "flag_norm_outliers",
    # Other layers (unchanged)
    "ActivationConsistencyDetector",
    "FileTrustLedger",
    "ModelAuditorDetector",
    "ModelAuditorStrategy",
    "RuntimeSentinelStrategy",
    "StripEntropyDetector",
]
