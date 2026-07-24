"""
ai/remediation/rollback.py — Remediation path 1: checkpoint rollback.

The fastest, safest response to a confirmed backdoor is to restore the last
global model that pre-dates the infection (ARCHITECTURE.md §7.4, path 1).  This
module is a thin, well-tested wrapper over the ``ModelRegistry`` contract
(INTERFACES.md §ModelRegistry) that:

  1. Picks a rollback target round *strictly before* the suspected infection round.
  2. Falls back to the nearest *earlier* checkpoint when the exact round is missing
     — never silently jumping to a later (possibly still-poisoned) checkpoint, per
     the ``CheckpointNotFoundError`` contract.
  3. Returns the restored params + the ``model_id`` used, for the audit trail.

Because an L2 audit runs every ``audit_interval_rounds`` rounds, the infection is
typically detected some rounds after it began; the audit round itself is therefore
*not* a safe rollback target.  Callers pass ``suspected_infection_round`` (usually
the first audit round that flagged a label, minus the audit interval) and this
module selects the newest checkpoint strictly older than it.
"""
from __future__ import annotations

import logging
from typing import Any

from ai.fl_core.exceptions import CheckpointNotFoundError

logger = logging.getLogger(__name__)


class RollbackFailed(Exception):
    """Internal signal that no safe pre-infection checkpoint exists.

    Caught by the Remediation Engine, which then escalates to unlearning /
    fine-pruning before ultimately raising ``RemediationFailedError``.
    """


class RollbackRemediator:
    """Restores a pre-infection checkpoint from a :class:`ModelRegistry`.

    Args:
        registry: Any object implementing the ``ModelRegistry`` interface
            (``load``, ``rollback_to``; optionally ``available_rounds``).
    """

    strategy_name: str = "rollback"

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def available_rounds(self) -> list[int]:
        """Return the sorted list of rounds with a checkpoint in the registry.

        Uses the registry's own index when exposed; otherwise probes via
        ``rollback_to`` is avoided (too expensive) and an empty list is returned.
        """
        # FileModelRegistry keeps a private round->model_id index.
        index = getattr(self._registry, "_load_index", None)
        if callable(index):
            try:
                return sorted(int(k) for k in index().keys())
            except Exception:  # pragma: no cover - defensive
                return []
        explicit = getattr(self._registry, "available_rounds", None)
        if callable(explicit):
            try:
                return sorted(int(r) for r in explicit())
            except Exception:  # pragma: no cover - defensive
                return []
        return []

    def select_target_round(self, suspected_infection_round: int | None) -> int:
        """Return the newest checkpoint round strictly before the infection round.

        Args:
            suspected_infection_round: Earliest round believed to be poisoned.
                ``None`` means "unknown" — fall back to the earliest checkpoint,
                which is the most conservative clean model available.

        Raises:
            RollbackFailed: If no eligible checkpoint exists.
        """
        rounds = self.available_rounds()
        if not rounds:
            raise RollbackFailed("registry exposes no checkpoints to roll back to")

        if suspected_infection_round is None:
            return rounds[0]

        candidates = [r for r in rounds if r < suspected_infection_round]
        if not candidates:
            raise RollbackFailed(
                f"no checkpoint strictly older than suspected infection round "
                f"{suspected_infection_round}; earliest available is {rounds[0]}"
            )
        return max(candidates)

    def remediate(self, suspected_infection_round: int | None) -> tuple[Any, int, str]:
        """Load the best pre-infection checkpoint.

        Returns:
            ``(model_state, target_round, model_id)``.

        Raises:
            RollbackFailed: If no safe checkpoint can be loaded.
        """
        target_round = self.select_target_round(suspected_infection_round)
        try:
            model_id = self._registry.rollback_to(target_round)
            model_state, _meta = self._registry.load(model_id)
        except CheckpointNotFoundError as exc:
            raise RollbackFailed(str(exc)) from exc
        logger.info(
            "Rollback: restored round %d (model_id=%s) as pre-infection model",
            target_round, model_id,
        )
        return model_state, target_round, model_id
