"""
ai/attacks/__init__.py — Attack engine public re-exports.

Image-domain BadNets attack engine for SENTINEL-FL Milestone 4.
"""

from ai.attacks.asr_evaluator import AttackSuccessRateEvaluator
from ai.attacks.attack_report import AttackEvalResult, PoisonRoundReport
from ai.attacks.badnets import BadNetsImageAttack
from ai.attacks.image_poisoning import ImagePoisoner
from ai.attacks.triggers import (
    TriggerFactory,
    TriggerPattern,
    apply_trigger,
)

__all__ = [
    # Core attack
    "BadNetsImageAttack",
    "ImagePoisoner",
    # Trigger
    "TriggerFactory",
    "TriggerPattern",
    "apply_trigger",
    # Evaluation
    "AttackSuccessRateEvaluator",
    # Reports
    "AttackEvalResult",
    "PoisonRoundReport",
]
