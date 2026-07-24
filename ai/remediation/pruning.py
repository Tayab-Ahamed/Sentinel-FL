"""
ai/remediation/pruning.py — Remediation path 3: fine-pruning.

Last-resort repair when both rollback and unlearning fail to drive the Attack
Success Rate below threshold.  Fine-pruning (Liu et al. 2018, ``BackdoorBench
defense/fp.py``; RESEARCH.md §4.4) removes the parameters that carry the backdoor
and then briefly fine-tunes on clean data to recover any lost utility.

Backend specialisation
----------------------
Classic fine-pruning prunes *dormant-on-clean / active-on-trigger* convolutional
channels.  The Phase 0 reference model is a linear softmax classifier with no
hidden units, so the equivalent backdoor pathway is the set of **input feature
channels that constitute the reversed trigger**: those columns of the weight
matrix ``W`` are what a BadNets patch exploits.  We therefore:

  1. Identify trigger feature channels from L2's reversed trigger.
  2. Zero (prune) those columns of ``W`` — severing the trigger→target pathway.
  3. Fine-tune on the clean holdout to recover clean accuracy.

The implemented ``TorchFinePruner`` ranks hidden channels by clean activation and
prunes the lowest fraction; the orchestration contract is identical.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ai.remediation.adapters import LinearSoftmaxAdapter, TorchModelAdapter
from ai.remediation.triggers import as_trigger_vector, trigger_mask

logger = logging.getLogger(__name__)


class FinePruner:
    """Prunes the trigger-carrying weights of a linear model, then fine-tunes.

    Args:
        adapter: A ``LinearSoftmaxAdapter`` (Phase 0). Other adapters are rejected
            with a clear message so the engine can skip this path gracefully.
        finetune_epochs: Clean-data recovery epochs after pruning.
        finetune_lr: Recovery learning rate.
        max_prune_fraction: Safety cap — never prune more than this fraction of
            input channels, even if the reversed trigger is diffuse.
    """

    strategy_name: str = "pruning"

    def __init__(
        self,
        adapter: LinearSoftmaxAdapter,
        finetune_epochs: int = 5,
        finetune_lr: float = 0.1,
        max_prune_fraction: float = 0.5,
    ) -> None:
        self._adapter = adapter
        self._finetune_epochs = int(finetune_epochs)
        self._finetune_lr = float(finetune_lr)
        self._max_prune_fraction = float(max_prune_fraction)

    def _prune_channels(self, reversed_triggers: list[Any], n_features: int) -> np.ndarray:
        """Union of trigger-feature masks across all reversed triggers."""
        mask = np.zeros(int(n_features), dtype=bool)
        for trig in reversed_triggers:
            representation = getattr(trig, "trigger_representation", trig)
            if representation is None:
                continue
            try:
                vec = as_trigger_vector(representation, n_features)
            except Exception:  # pragma: no cover - defensive
                continue
            mask |= trigger_mask(vec)
        # Enforce the safety cap on how many channels may be pruned.
        max_channels = int(self._max_prune_fraction * n_features)
        if max_channels >= 1 and mask.sum() > max_channels:
            kept = np.where(mask)[0][:max_channels]
            capped = np.zeros_like(mask)
            capped[kept] = True
            mask = capped
        return mask

    def remediate(
        self,
        params: Any,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        reversed_triggers: list[Any],
        n_features: int,
    ) -> Any:
        """Return repaired params after pruning trigger channels + fine-tuning."""
        if not isinstance(self._adapter, LinearSoftmaxAdapter):
            raise TypeError(
                "FinePruner Phase 0 path requires a LinearSoftmaxAdapter; "
                "use TorchFinePruner for CNN models (Phase 1)."
            )
        mask = self._prune_channels(reversed_triggers, n_features)
        W = self._adapter.get_weight_matrix(params)
        n_pruned = int(mask.sum())
        if n_pruned:
            W[:, mask] = 0.0
        pruned_params = self._adapter.set_weight_matrix(params, W)
        repaired = self._adapter.fine_tune(
            pruned_params,
            np.asarray(X_clean, dtype=float),
            np.asarray(y_clean).astype(int),
            epochs=self._finetune_epochs,
            lr=self._finetune_lr,
        )
        logger.info(
            "Fine-pruning: zeroed %d/%d trigger channels then fine-tuned %d epochs",
            n_pruned,
            n_features,
            self._finetune_epochs,
        )
        return repaired


class TorchFinePruner:
    """Activation-aware fine-pruning for convolutional PyTorch models.

    Channels with the lowest mean absolute activation on clean calibration data
    are zeroed in the final convolutional layer, then the model is fine-tuned on
    clean data to recover utility. This follows the defensive principle of Liu et
    al. (2018) while keeping all backend details inside ``TorchModelAdapter``.
    """

    strategy_name: str = "pruning"

    def __init__(
        self,
        adapter: TorchModelAdapter,
        finetune_epochs: int = 5,
        finetune_lr: float = 0.01,
        prune_fraction: float = 0.1,
    ) -> None:
        self._adapter = adapter
        self._finetune_epochs = int(finetune_epochs)
        self._finetune_lr = float(finetune_lr)
        self._prune_fraction = float(prune_fraction)
        self.last_evidence: dict[str, Any] = {}

    def remediate(
        self,
        params: Any,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        reversed_triggers: list[Any],
        n_features: int,
    ) -> Any:
        del reversed_triggers, n_features  # activation ranking is trigger-agnostic
        pruned, evidence = self._adapter.prune_dormant_channels(
            params, X_clean, prune_fraction=self._prune_fraction
        )
        self.last_evidence = evidence
        repaired = self._adapter.fine_tune(
            pruned,
            X_clean,
            y_clean,
            epochs=self._finetune_epochs,
            lr=self._finetune_lr,
        )
        logger.info(
            "Torch fine-pruning: pruned %d/%d channels in %s then fine-tuned %d epochs",
            evidence["channels_pruned"],
            evidence["channels_total"],
            evidence["layer"],
            self._finetune_epochs,
        )
        return repaired
