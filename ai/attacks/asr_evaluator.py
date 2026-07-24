"""
ai/attacks/asr_evaluator.py — Attack Success Rate evaluation.

``AttackSuccessRateEvaluator`` computes two key metrics after each FL round:

  **ASR (Attack Success Rate)**:
    Fraction of triggered (poisoned) inputs that the current global model
    classifies as the target backdoor label.  ASR = 1.0 means every
    triggered input fools the model.

  **Clean Accuracy (C-Acc)**:
    Fraction of clean (unmodified) inputs correctly classified.  Used to
    confirm the backdoor does not degrade overall utility.

The evaluator works with any PyTorch model that accepts ``(N, C, H, W)``
float32 tensors and returns logits of shape ``(N, n_classes)``.

Usage::

    evaluator = AttackSuccessRateEvaluator(device="cpu")
    result = evaluator.evaluate_round(
        model=global_model,
        attacker=badnets_attack,
        X_clean_eval=X_test,
        y_clean_eval=y_test,
        round_num=5,
    )
    print(result.summary())
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn

from ai.attacks.attack_report import AttackEvalResult

logger = logging.getLogger(__name__)

# Default batch size for inference (avoids OOM on large eval sets)
_EVAL_BATCH_SIZE = 256


class AttackSuccessRateEvaluator:
    """Evaluates ASR and clean accuracy for a BadNets attack each FL round.

    Args:
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
        batch_size: Mini-batch size for inference.
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = _EVAL_BATCH_SIZE,
    ) -> None:
        self._device = torch.device(device)
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_asr(
        self,
        model: nn.Module,
        X_triggered: np.ndarray,
        target_label: int,
    ) -> float:
        """Compute Attack Success Rate.

        Args:
            model: Global PyTorch model (set to eval mode internally).
            X_triggered: Triggered evaluation images ``(N, C, H, W)`` float32.
            target_label: The backdoor target label.

        Returns:
            ASR in [0, 1].  Returns 0.0 if ``X_triggered`` is empty.
        """
        if len(X_triggered) == 0:
            logger.warning("compute_asr: X_triggered is empty; returning 0.0")
            return 0.0

        preds = self._predict(model, X_triggered)
        asr = float((preds == target_label).mean())
        logger.debug(
            "compute_asr: ASR=%.4f (target_label=%d, n=%d)",
            asr,
            target_label,
            len(X_triggered),
        )
        return asr

    def compute_clean_accuracy(
        self,
        model: nn.Module,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
    ) -> float:
        """Compute clean accuracy.

        Args:
            model: Global PyTorch model.
            X_clean: Clean evaluation images ``(N, C, H, W)`` float32.
            y_clean: True labels ``(N,)`` int.

        Returns:
            Accuracy in [0, 1].  Returns 0.0 if empty.
        """
        if len(X_clean) == 0:
            logger.warning("compute_clean_accuracy: X_clean is empty; returning 0.0")
            return 0.0

        preds = self._predict(model, X_clean)
        acc = float((preds == y_clean).mean())
        logger.debug("compute_clean_accuracy: C-Acc=%.4f (n=%d)", acc, len(X_clean))
        return acc

    def evaluate_round(
        self,
        model: nn.Module,
        attacker: object,  # BadNetsImageAttack (avoid circular import)
        X_clean_eval: np.ndarray,
        y_clean_eval: np.ndarray,
        round_num: int,
    ) -> AttackEvalResult:
        """Compute ASR + clean accuracy for one FL round.

        Builds the triggered eval set via ``attacker.build_asr_eval_set()``,
        then computes both metrics in a single forward pass each.

        Args:
            model: Current global model.
            attacker: ``BadNetsImageAttack`` (or any object with
                ``build_asr_eval_set(X, y)`` and ``target_label``).
            X_clean_eval: Clean evaluation images.
            y_clean_eval: Clean evaluation labels.
            round_num: Current FL round.

        Returns:
            ``AttackEvalResult`` with ASR, C-Acc, sample counts.
        """
        t0 = time.perf_counter()

        # Build triggered eval set (non-target clean samples + trigger)
        X_triggered, y_target = attacker.build_asr_eval_set(X_clean_eval, y_clean_eval)

        # Compute metrics
        asr = self.compute_asr(model, X_triggered, attacker.target_label)
        clean_acc = self.compute_clean_accuracy(model, X_clean_eval, y_clean_eval)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "AttackSuccessRateEvaluator [round %d]: ASR=%.3f C-Acc=%.3f "
            "(n_triggered=%d n_clean=%d, %.1f ms)",
            round_num,
            asr,
            clean_acc,
            len(X_triggered),
            len(X_clean_eval),
            elapsed_ms,
        )

        result = AttackEvalResult(
            round_num=round_num,
            asr=asr,
            clean_acc=clean_acc,
            n_triggered=len(X_triggered),
            n_clean=len(X_clean_eval),
            target_label=attacker.target_label,
        )

        # Attach metrics back to the attacker's round report
        if hasattr(attacker, "attach_eval_result"):
            attacker.attach_eval_result(round_num, asr, clean_acc)

        return result

    def evaluate_all_rounds(
        self,
        model_history: list[tuple[int, nn.Module]],
        attacker: object,
        X_clean_eval: np.ndarray,
        y_clean_eval: np.ndarray,
    ) -> list[AttackEvalResult]:
        """Evaluate ASR and C-Acc for a list of historical models.

        Args:
            model_history: List of ``(round_num, model)`` pairs.
            attacker: Attack simulator.
            X_clean_eval: Clean evaluation images.
            y_clean_eval: Clean evaluation labels.

        Returns:
            List of ``AttackEvalResult``, one per round.
        """
        results = []
        for round_num, model in model_history:
            result = self.evaluate_round(
                model=model,
                attacker=attacker,
                X_clean_eval=X_clean_eval,
                y_clean_eval=y_clean_eval,
                round_num=round_num,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Run batched inference and return int64 predicted labels.

        Args:
            model: PyTorch model (moved to device).
            X: Float32 image array ``(N, C, H, W)``.

        Returns:
            Predicted class labels ``(N,)`` as int64 numpy array.
        """
        model.eval()
        model.to(self._device)
        all_preds: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(X), self._batch_size):
                batch = torch.from_numpy(X[start : start + self._batch_size]).to(self._device)
                logits = model(batch)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)

        return np.concatenate(all_preds, axis=0).astype(np.int64)
