"""
ai/attacks/badnets.py — BadNets image-domain AttackSimulator.

``BadNetsImageAttack`` implements the ``AttackSimulator`` interface
(INTERFACES.md §AttackSimulator) for 2-D image datasets (MNIST, CIFAR-10).

It is a clean re-implementation of the canonical BadNets attack
(Gu et al. 2017, "BadNets: Identifying Vulnerabilities in the Machine
Learning Model Supply Chain") adapted for the federated learning setting.

The existing ``BadNetsAttackSimulator`` in ``ai/training/poison.py`` handles
Phase 0 Gaussian-blob (1-D feature-vector) data and is kept unchanged.
This class targets image data and lives in the dedicated attack engine package.

Attack semantics:
  - A configurable subset of FL clients are designated as *malicious*.
  - Each malicious client stamps a trigger pattern onto a fraction of their
    local training data and flips those samples' labels to the target class.
  - Honest clients return data unchanged.
  - After every round, ``build_trigger_eval_set()`` produces a fully-triggered
    evaluation set for ASR computation.

Seed strategy (deterministic):
  ``seed = base_seed + round_num * 1000 + client_numeric_id``
  This gives different but reproducible poisoning per client per round.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai.attacks.attack_report import PoisonRoundReport
from ai.attacks.image_poisoning import ImagePoisoner
from ai.attacks.triggers import TriggerFactory, TriggerPattern, apply_trigger
from ai.fl_core.interfaces import AttackSimulator

logger = logging.getLogger(__name__)


class BadNetsImageAttack(AttackSimulator):
    """BadNets attack for image-domain federated learning datasets.

    Args:
        target_label: Backdoor target class (all triggered inputs → this label).
        poison_fraction: Fraction of each malicious client's data to poison.
        malicious_client_indices: Set of numeric client IDs that are malicious.
            If empty, no client is poisoned.
        pattern: ``TriggerPattern`` describing the trigger stamp.
        seed: Base random seed for deterministic poisoning.
        poison_non_target_only: If True, only non-target-class samples are
            candidates for poisoning (default: True).
    """

    name: str = "badnets_image"

    def __init__(
        self,
        target_label: int = 0,
        poison_fraction: float = 0.15,
        malicious_client_indices: list[int] | None = None,
        pattern: TriggerPattern | None = None,
        seed: int = 42,
        poison_non_target_only: bool = True,
    ) -> None:
        self._target_label = target_label
        self._poison_fraction = poison_fraction
        self._malicious: set[int] = set(malicious_client_indices or [2, 5])
        self._pattern = pattern or TriggerFactory.make_square(
            size=4, location="bottom_right", color=1.0, opacity=1.0
        )
        self._seed = seed
        self._poisoner = ImagePoisoner(
            pattern=self._pattern,
            target_label=target_label,
            poison_fraction=poison_fraction,
            poison_non_target_only=poison_non_target_only,
        )
        # Round reports accumulated across the experiment
        self._round_reports: list[PoisonRoundReport] = []

    # ------------------------------------------------------------------
    # AttackSimulator interface
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> BadNetsImageAttack:
        """Construct from a ``Configuration`` or duck-typed config object.

        Reads ``config.attack.*`` and ``config.trigger.*`` fields.

        Args:
            config: Configuration object with ``attack`` and ``trigger``
                sub-configs.

        Returns:
            Fully configured ``BadNetsImageAttack``.
        """
        attack_cfg = getattr(config, "attack", None)
        pattern = TriggerFactory.from_config(config)
        seed = int(getattr(config, "seed", 42))

        if attack_cfg is None:
            logger.warning("BadNetsImageAttack.from_config: no attack sub-config; using defaults.")
            return cls(pattern=pattern, seed=seed)

        # Resolve malicious_client_indices: explicit list takes priority;
        # otherwise use malicious_client_fraction of n_clients.
        explicit = getattr(attack_cfg, "malicious_client_indices", None)
        if explicit is not None and len(list(explicit)) > 0:
            mal_indices = list(explicit)
        else:
            n_clients = int(getattr(config, "n_clients", 12))
            frac = float(getattr(attack_cfg, "malicious_client_fraction", 0.25))
            n_mal = max(1, round(n_clients * frac))
            mal_indices = list(range(n_mal))

        return cls(
            target_label=int(getattr(attack_cfg, "target_label", 0)),
            poison_fraction=float(getattr(attack_cfg, "poison_fraction", 0.15)),
            malicious_client_indices=mal_indices,
            pattern=pattern,
            seed=seed,
        )

    def poison_client_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: str,
        round_num: int,
        config: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject BadNets trigger into a malicious client's training data.

        Honest clients receive their data unchanged with an all-False mask.

        Args:
            X: Client image batch ``(N, C, H, W)`` float32.
            y: Client label array ``(N,)`` int64.
            client_id: Client identifier (e.g. ``"client_02"``).
            round_num: Current FL round (used for seed variation).
            config: Active configuration (unused by this attacker).

        Returns:
            ``(X_poisoned, y_poisoned, mask)`` — mask is bool array of shape (N,).
        """
        cid_int = _parse_client_id(client_id)

        if cid_int not in self._malicious:
            # Honest client — return unchanged data
            return X.copy(), y.copy(), np.zeros(len(X), dtype=bool)

        if len(X) <= 2:
            logger.debug(
                "BadNetsImageAttack: client %s has too few samples (%d); skipping.",
                client_id,
                len(X),
            )
            return X.copy(), y.copy(), np.zeros(len(X), dtype=bool)

        # Per-client, per-round deterministic seed
        seed = self._seed + round_num * 1000 + cid_int
        X_p, y_p, mask = self._poisoner.poison_batch(X, y, seed=seed)

        n_poisoned = int(mask.sum())
        logger.info(
            "BadNetsImageAttack: [round %d] client %s → %d/%d poisoned (target_label=%d)",
            round_num,
            client_id,
            n_poisoned,
            len(X),
            self._target_label,
        )

        # Track per-client counts for round report
        self._track_poison(
            round_num=round_num,
            client_id=client_id,
            n_poisoned=n_poisoned,
            n_total=len(X),
        )
        return X_p, y_p, mask

    def build_trigger_eval_set(self, X_clean: np.ndarray) -> np.ndarray:
        """Stamp the trigger onto every clean sample for ASR evaluation.

        Args:
            X_clean: Clean evaluation images ``(N, C, H, W)`` float32.

        Returns:
            ``X_triggered`` — same shape with trigger applied to every sample.
        """
        return apply_trigger(X_clean, self._pattern)

    # ------------------------------------------------------------------
    # Extended API (beyond base AttackSimulator interface)
    # ------------------------------------------------------------------

    def build_asr_eval_set(
        self,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a fully-triggered eval set (non-target samples only).

        Args:
            X_clean: Clean evaluation images.
            y_clean: Clean evaluation labels.

        Returns:
            ``(X_triggered, y_target)`` for ASR computation.
        """
        return self._poisoner.build_asr_eval_set(X_clean, y_clean)

    def get_round_report(self, round_num: int) -> PoisonRoundReport | None:
        """Return the poison report for a specific round, if available."""
        for r in self._round_reports:
            if r.round_num == round_num:
                return r
        return None

    def all_round_reports(self) -> list[PoisonRoundReport]:
        """Return all accumulated round reports (one per round)."""
        return list(self._round_reports)

    def attach_eval_result(self, round_num: int, asr: float, clean_acc: float) -> None:
        """Attach ASR and clean accuracy to an existing round report.

        Args:
            round_num: Round to update.
            asr: Attack Success Rate computed after this round.
            clean_acc: Clean accuracy computed after this round.
        """
        report = self.get_round_report(round_num)
        if report is not None:
            report.asr = asr
            report.clean_acc = clean_acc

    @property
    def target_label(self) -> int:
        """The backdoor target class."""
        return self._target_label

    @property
    def malicious_client_indices(self) -> set[int]:
        """Set of numeric client IDs designated as malicious."""
        return set(self._malicious)

    @property
    def trigger_pattern(self) -> TriggerPattern:
        """The trigger pattern used by this attack."""
        return self._pattern

    # ------------------------------------------------------------------
    # Internal tracking
    # ------------------------------------------------------------------

    def _track_poison(
        self,
        round_num: int,
        client_id: str,
        n_poisoned: int,
        n_total: int,
    ) -> None:
        """Update or create the PoisonRoundReport for this round."""
        report = self.get_round_report(round_num)
        if report is None:
            report = PoisonRoundReport(
                round_num=round_num,
                malicious_clients=[],
                total_poisoned_samples=0,
                poison_fraction_target=self._poison_fraction,
                poison_fraction_actual=0.0,
                target_label=self._target_label,
                trigger_shape=self._pattern.shape,
                trigger_location=self._pattern.location,
                trigger_size=self._pattern.size,
            )
            self._round_reports.append(report)

        if client_id not in report.malicious_clients:
            report.malicious_clients.append(client_id)
        report.total_poisoned_samples += n_poisoned
        report.per_client_poisoned[client_id] = n_poisoned
        # Recompute actual fraction (approximation — average over clients seen so far)
        total_samples = sum(
            n_poisoned + 1  # minimal estimate; real total not stored
            for _ in report.malicious_clients
        )
        report.poison_fraction_actual = report.total_poisoned_samples / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_client_id(client_id: str) -> int:
    """Extract numeric suffix from a client ID string.

    Examples::
        "client_02" → 2
        "client_9"  → 9
        "42"        → 42

    Falls back to ``hash(client_id) % 10_000`` on parse failure.
    """
    try:
        return int(client_id.split("_")[-1])
    except (ValueError, IndexError):
        return hash(client_id) % 10_000
