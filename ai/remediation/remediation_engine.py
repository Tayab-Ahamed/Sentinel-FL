"""
ai/remediation/remediation_engine.py — the Remediation Engine (orchestrator).

This is the component named directly in the challenge title ("...and their
**Remediation**").  Once L2/L3 confirm a backdoor, the engine restores a trusted
model through an escalating policy and produces an auditable
:class:`RemediationReport` (ARCHITECTURE.md §7.4).

Escalation policy (each step is measured against a held-out ASR eval set):

    1. rollback   — restore the last pre-infection checkpoint.
    2. unlearning — fine-tune away L2's reversed trigger on clean data.
    3. pruning    — fine-prune the trigger-carrying weights, then recover.

A step is *accepted* when it drives ASR to/under ``asr_threshold`` **and** keeps
clean accuracy within ``max_clean_accuracy_drop`` of the pre-remediation model.
The first accepted step wins; the model is not degraded further.  If every enabled
step is exhausted without success, the engine raises
:class:`RemediationFailedError` (which flips the dashboard's
``manual_review_required`` flag).  Every attempt — successful or not — is written to
the Trust Ledger (L4) when one is supplied, so the remediation itself is auditable.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ai.fl_core.exceptions import RemediationFailedError
from ai.fl_core.schemas import RemediationReport, TrustLedgerEntry
from ai.remediation.adapters import LinearSoftmaxAdapter, ModelAdapter, TorchModelAdapter
from ai.remediation.pruning import FinePruner, TorchFinePruner
from ai.remediation.rollback import RollbackFailed, RollbackRemediator
from ai.remediation.unlearning import TriggerUnlearner

logger = logging.getLogger(__name__)

_DEFAULT_STRATEGIES = ("rollback", "unlearning", "pruning")


class RemediationEngine:
    """Escalating backdoor remediation orchestrator.

    Args:
        adapter: Model backend adapter (Phase 0 ``LinearSoftmaxAdapter``).
        registry: Optional ``ModelRegistry`` (enables the rollback path).
        ledger: Optional Trust Ledger (L4) for an audit trail. Must expose
            ``add_entry(TrustLedgerEntry)``.
        asr_threshold: Attack-success-rate at/under which remediation succeeds.
        max_clean_accuracy_drop: Max tolerated clean-accuracy regression vs. the
            pre-remediation model for a step to be accepted.
        strategies: Ordered subset of ``{rollback, unlearning, pruning}``.
        unlearning_epochs / unlearning_lr: Passed to :class:`TriggerUnlearner`.
        pruning_finetune_epochs / pruning_finetune_lr: Passed to :class:`FinePruner`.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        registry: Any | None = None,
        ledger: Any | None = None,
        asr_threshold: float = 0.2,
        max_clean_accuracy_drop: float = 0.1,
        strategies: tuple[str, ...] | list[str] = _DEFAULT_STRATEGIES,
        unlearning_epochs: int = 10,
        unlearning_lr: float = 0.1,
        pruning_finetune_epochs: int = 5,
        pruning_finetune_lr: float = 0.1,
    ) -> None:
        self._adapter = adapter
        self._registry = registry
        self._ledger = ledger
        self._asr_threshold = float(asr_threshold)
        self._max_clean_drop = float(max_clean_accuracy_drop)
        self._strategies = tuple(strategies)

        self._rollback = RollbackRemediator(registry) if registry is not None else None
        self._unlearner = TriggerUnlearner(adapter, epochs=unlearning_epochs, lr=unlearning_lr)
        self._pruner = (
            FinePruner(
                adapter,
                finetune_epochs=pruning_finetune_epochs,
                finetune_lr=pruning_finetune_lr,
            )
            if isinstance(adapter, LinearSoftmaxAdapter)
            else TorchFinePruner(
                adapter,
                finetune_epochs=pruning_finetune_epochs,
                finetune_lr=pruning_finetune_lr,
            )
            if isinstance(adapter, TorchModelAdapter)
            else None
        )

    # ------------------------------------------------------------------
    # Config constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        adapter: ModelAdapter,
        config: Any,
        registry: Any | None = None,
        ledger: Any | None = None,
    ) -> RemediationEngine:
        """Build an engine from a :class:`Configuration` object."""
        return cls(
            adapter=adapter,
            registry=registry,
            ledger=ledger,
            asr_threshold=getattr(config, "remediation_asr_threshold", 0.2),
            max_clean_accuracy_drop=getattr(config, "remediation_max_clean_accuracy_drop", 0.1),
            strategies=tuple(getattr(config, "remediation_strategies", _DEFAULT_STRATEGIES)),
            unlearning_epochs=getattr(config, "unlearning_epochs", 10),
            unlearning_lr=getattr(config, "unlearning_lr", 0.1),
            pruning_finetune_epochs=getattr(config, "pruning_finetune_epochs", 5),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _asr(
        self,
        params: Any,
        X_triggered: np.ndarray,
        target_label: int,
        y_triggered: np.ndarray | None = None,
    ) -> float:
        if X_triggered is None or len(X_triggered) == 0:
            return 0.0
        preds = self._adapter.predict(params, X_triggered)
        if y_triggered is None:
            # Backward-compatible fallback for callers that already pre-filtered
            # their triggered evaluation set to non-target source classes.
            return float(np.mean(preds == target_label))
        y_true = np.asarray(y_triggered).astype(int)
        if len(y_true) != len(preds):
            raise ValueError("y_triggered length must match X_triggered")
        source_mask = y_true != target_label
        if not np.any(source_mask):
            return 0.0
        return float(np.mean(preds[source_mask] == target_label))

    def _clean_acc(self, params: Any, X_clean: np.ndarray, y_clean: np.ndarray) -> float:
        if X_clean is None or len(X_clean) == 0:
            return 0.0
        preds = self._adapter.predict(params, X_clean)
        return float(np.mean(preds == np.asarray(y_clean).astype(int)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remediate(
        self,
        params: Any,
        audit_report: Any,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        X_triggered: np.ndarray,
        target_label: int,
        n_features: int | None = None,
        suspected_infection_round: int | None = None,
        raise_on_failure: bool = True,
        y_triggered: np.ndarray | None = None,
    ) -> tuple[Any, RemediationReport]:
        """Run the escalating remediation policy.

        Args:
            params: The current (poisoned) global model params.
            audit_report: L2 ``AuditReport`` (supplies ``reversed_triggers`` and
                ``flagged_labels``). May be ``None`` for rollback-only recovery.
            X_clean, y_clean: Clean holdout for utility checks + repair training.
            X_triggered: Trigger-stamped eval set for ASR measurement.
            target_label: Backdoor target class.
            n_features: Feature dim; inferred from ``X_clean`` when omitted.
            suspected_infection_round: Earliest poisoned round (for rollback).
            raise_on_failure: If True (default, per ARCHITECTURE.md §7.4), raise
                ``RemediationFailedError`` when no step succeeds. The failing
                :class:`RemediationReport` is attached to the exception as
                ``.report``. If False, return ``(best_params, report)``.
            y_triggered: Optional clean ground-truth labels corresponding to
                ``X_triggered``. When supplied, ASR follows the standard source-only
                definition and excludes samples already belonging to ``target_label``.

        Returns:
            ``(remediated_params, RemediationReport)`` on success (or on failure
            when ``raise_on_failure=False``).
        """
        t0 = time.perf_counter()
        if n_features is None:
            n_features = int(np.prod(np.asarray(X_clean).shape[1:]))
        reversed_triggers = list(getattr(audit_report, "reversed_triggers", []) or [])
        round_num = int(getattr(audit_report, "round_num", 0) or 0)

        asr_before = self._asr(params, X_triggered, target_label, y_triggered)
        clean_before = self._clean_acc(params, X_clean, y_clean)
        logger.info(
            "Remediation start: ASR=%.3f C-Acc=%.3f (threshold ASR<=%.3f)",
            asr_before,
            clean_before,
            self._asr_threshold,
        )

        per_strategy: list[dict[str, Any]] = []
        best_params = self._adapter.clone(params)

        for name in self._strategies:
            candidate, note = self._run_step(
                name,
                params=params,
                reversed_triggers=reversed_triggers,
                X_clean=X_clean,
                y_clean=y_clean,
                n_features=n_features,
                suspected_infection_round=suspected_infection_round,
            )
            if candidate is None:
                per_strategy.append({"strategy": name, "status": "skipped", "detail": note})
                logger.info("Remediation step '%s' skipped: %s", name, note)
                continue

            asr_after = self._asr(candidate, X_triggered, target_label, y_triggered)
            clean_after = self._clean_acc(candidate, X_clean, y_clean)
            accepted = (
                asr_after <= self._asr_threshold
                and clean_after >= clean_before - self._max_clean_drop
            )
            per_strategy.append(
                {
                    "strategy": name,
                    "status": "accepted" if accepted else "rejected",
                    "asr_after": round(asr_after, 4),
                    "clean_acc_after": round(clean_after, 4),
                    "detail": note,
                }
            )
            logger.info(
                "Remediation step '%s': ASR %.3f->%.3f C-Acc %.3f->%.3f (%s)",
                name,
                asr_before,
                asr_after,
                clean_before,
                clean_after,
                "ACCEPTED" if accepted else "rejected",
            )

            if accepted:
                report = self._build_report(
                    round_num=round_num,
                    suspected_infection_round=suspected_infection_round,
                    strategies_attempted=[s["strategy"] for s in per_strategy],
                    strategy_succeeded=name,
                    asr_before=asr_before,
                    asr_after=asr_after,
                    clean_before=clean_before,
                    clean_after=clean_after,
                    success=True,
                    manual_review_required=False,
                    reason=f"Backdoor remediated via {name}: {note}",
                    per_strategy=per_strategy,
                    rolled_back_model_id=note if name == "rollback" else None,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )
                self._log_to_ledger(report)
                return candidate, report

        # No strategy succeeded -> manual review.
        reason = (
            f"exhausted strategies {list(self._strategies)}; "
            f"lowest ASR still above threshold {self._asr_threshold}"
        )
        report = self._build_report(
            round_num=round_num,
            suspected_infection_round=suspected_infection_round,
            strategies_attempted=[s["strategy"] for s in per_strategy],
            strategy_succeeded=None,
            asr_before=asr_before,
            asr_after=asr_before,
            clean_before=clean_before,
            clean_after=clean_before,
            success=False,
            manual_review_required=True,
            reason=reason,
            per_strategy=per_strategy,
            rolled_back_model_id=None,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
        self._log_to_ledger(report)
        if raise_on_failure:
            exc = RemediationFailedError(reason)
            exc.report = report  # type: ignore[attr-defined]
            raise exc
        return best_params, report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_step(
        self,
        name: str,
        *,
        params: Any,
        reversed_triggers: list[Any],
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        n_features: int,
        suspected_infection_round: int | None,
    ) -> tuple[Any, str]:
        """Execute one remediation strategy. Returns ``(candidate_params, note)``;
        ``candidate_params`` is ``None`` when the step is unavailable/failed."""
        try:
            if name == "rollback":
                if self._rollback is None:
                    return None, "no model registry configured"
                restored, target_round, model_id = self._rollback.remediate(
                    suspected_infection_round
                )
                return restored, f"restored round {target_round} (model_id={model_id})"

            if name == "unlearning":
                repaired = self._unlearner.remediate(
                    params, X_clean, y_clean, reversed_triggers, n_features
                )
                return repaired, "fine-tuned reversed trigger with correct labels"

            if name == "pruning":
                if self._pruner is None:
                    return None, "fine-pruning unavailable for this model backend"
                repaired = self._pruner.remediate(
                    params, X_clean, y_clean, reversed_triggers, n_features
                )
                return repaired, "pruned trigger channels + fine-tuned"

            return None, f"unknown strategy '{name}'"
        except RollbackFailed as exc:
            return None, f"rollback unavailable: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Remediation step '%s' raised", name)
            return None, f"error: {exc}"

    def _build_report(self, **kwargs: Any) -> RemediationReport:
        elapsed_ms = kwargs.pop("elapsed_ms", None)
        report = RemediationReport(
            round_num=kwargs["round_num"],
            suspected_infection_round=kwargs["suspected_infection_round"],
            strategies_attempted=kwargs["strategies_attempted"],
            strategy_succeeded=kwargs["strategy_succeeded"],
            asr_before=kwargs["asr_before"],
            asr_after=kwargs["asr_after"],
            clean_accuracy_before=kwargs["clean_before"],
            clean_accuracy_after=kwargs["clean_after"],
            asr_threshold=self._asr_threshold,
            success=kwargs["success"],
            manual_review_required=kwargs["manual_review_required"],
            reason=kwargs["reason"],
            per_strategy=kwargs["per_strategy"],
            rolled_back_model_id=kwargs["rolled_back_model_id"],
            elapsed_ms=elapsed_ms,
        )
        return report

    def _log_to_ledger(self, report: RemediationReport) -> None:
        """Write a remediation record to the Trust Ledger (never raises)."""
        if self._ledger is None:
            return
        try:
            entry = TrustLedgerEntry(
                layer_id="L5",
                subject_type="model",
                subject_id=f"global_model_round_{report.round_num}",
                round_num=report.round_num,
                score=1.0 if report.manual_review_required else 0.0,
                reason=report.reason,
                evidence={
                    "strategy_succeeded": report.strategy_succeeded,
                    "asr_before": report.asr_before,
                    "asr_after": report.asr_after,
                    "success": report.success,
                    "manual_review_required": report.manual_review_required,
                },
            )
            self._ledger.add_entry(entry)
        except Exception:  # pragma: no cover - ledger writes never gate control flow
            logger.warning("Remediation: failed to write audit entry to ledger", exc_info=True)
