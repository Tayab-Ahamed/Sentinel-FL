"""
ai/attacks/attack_report.py — Per-round poison statistics dataclass.

``PoisonRoundReport`` captures the ground-truth state of the attack for one
FL round.  It is produced by ``BadNetsImageAttack`` and stored alongside the
experiment log — never exposed to the defense pipeline during training.

``AttackEvalResult`` is the paired evaluation record: ASR and clean accuracy
measured after a round completes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# PoisonRoundReport
# ---------------------------------------------------------------------------


@dataclass
class PoisonRoundReport:
    """Ground-truth attack state for one FL round.

    Args:
        round_num: FL round index (0-based).
        malicious_clients: Client IDs that were active this round.
        total_poisoned_samples: Aggregate poisoned samples across all malicious clients.
        poison_fraction_target: Configured poison fraction (may differ from actual).
        poison_fraction_actual: Measured fraction (poisoned / total client samples).
        target_label: Label to which poisoned samples are flipped.
        trigger_shape: Human-readable trigger description.
        trigger_location: Where the trigger is placed on the image.
        trigger_size: Trigger patch side length in pixels.
    """

    round_num: int
    malicious_clients: list[str]
    total_poisoned_samples: int
    poison_fraction_target: float
    poison_fraction_actual: float
    target_label: int
    trigger_shape: str
    trigger_location: str
    trigger_size: int
    # Filled after ASR evaluation
    asr: float | None = None
    clean_acc: float | None = None
    # Optional extras
    per_client_poisoned: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """One-line human-readable summary."""
        asr_str = f"{self.asr:.1%}" if self.asr is not None else "n/a"
        acc_str = f"{self.clean_acc:.1%}" if self.clean_acc is not None else "n/a"
        return (
            f"Round {self.round_num:3d} | "
            f"Malicious: {len(self.malicious_clients)} clients | "
            f"Poisoned: {self.total_poisoned_samples} samples "
            f"(frac={self.poison_fraction_actual:.1%}) | "
            f"ASR: {asr_str} | C-Acc: {acc_str}"
        )


# ---------------------------------------------------------------------------
# AttackEvalResult
# ---------------------------------------------------------------------------


@dataclass
class AttackEvalResult:
    """Evaluation metrics for one FL round after attack.

    Produced by ``AttackSuccessRateEvaluator``.

    Args:
        round_num: FL round.
        asr: Attack Success Rate: fraction of triggered inputs predicted as
            ``target_label`` by the current global model.
        clean_acc: Clean accuracy: fraction of clean inputs correctly classified.
        n_triggered: Number of triggered samples used in ASR computation.
        n_clean: Number of clean samples used in accuracy computation.
        target_label: The backdoor target class.
    """

    round_num: int
    asr: float
    clean_acc: float
    n_triggered: int
    n_clean: int
    target_label: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """One-line summary string."""
        return (
            f"Round {self.round_num:3d} | "
            f"ASR: {self.asr:.1%} (n={self.n_triggered}) | "
            f"C-Acc: {self.clean_acc:.1%} (n={self.n_clean})"
        )
