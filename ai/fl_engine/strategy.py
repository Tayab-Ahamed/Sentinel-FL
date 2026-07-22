"""
ai/fl_engine/strategy.py — SentinelFedAvg: FedAvg with structured logging.

Subclasses Flower's ``FedAvg`` strategy to intercept every round's aggregate
result and emit structured JSON-lines log events (ARCHITECTURE.md §6) that:
  - Record per-round centralized evaluation metrics (clean accuracy, loss).
  - Record per-round federated evaluation metrics (weighted mean of client metrics).
  - Emit a ``round_complete`` event that the dashboard's MetricsCollector reads.

Flower API (1.8+):
  - ``aggregate_fit`` → called after collecting all client ``fit()`` results.
  - ``aggregate_evaluate`` → called after collecting all client ``evaluate()`` results.
  - ``evaluate`` → centralized (server-side) evaluation, called every round.

The strategy does NOT implement defense layers (L1–L3) — that is Milestone 3.
For Milestone 2 this is a clean FedAvg baseline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ai.fl_core.logger import StructuredLogger
from ai.models.mnist_cnn import SimpleCNN, set_model_parameters

logger = logging.getLogger(__name__)

# Flower type alias for the aggregation results list
FitResultsList = list[tuple[ClientProxy, FitRes]]
EvalResultsList = list[tuple[ClientProxy, EvaluateRes]]


def _weighted_average(metrics_list: list[tuple[int, Metrics]]) -> Metrics:
    """Compute sample-weighted mean of scalar metrics across clients.

    Args:
        metrics_list: List of ``(num_examples, metrics_dict)`` tuples.

    Returns:
        Dict of weighted-average metric values.
    """
    if not metrics_list:
        return {}
    total_examples = sum(n for n, _ in metrics_list)
    if total_examples == 0:
        return {}
    keys = [k for k in metrics_list[0][1] if isinstance(metrics_list[0][1][k], (int, float))]
    averaged: dict[str, float] = {}
    for key in keys:
        averaged[key] = sum(
            n * float(m.get(key, 0.0)) for n, m in metrics_list
        ) / total_examples
    return averaged


class SentinelFedAvg(FedAvg):
    """FedAvg strategy extended with SENTINEL-FL structured logging.

    All FL metrics are emitted as ``round_complete`` / ``fit_aggregate`` /
    ``eval_aggregate`` JSON-lines events via ``StructuredLogger``, which the
    backend's ``JsonLinesMetricsCollector`` reads to produce ``EvaluationResult``.

    Args:
        sentinel_logger: StructuredLogger instance.  All events are written here.
        eval_model: A freshly instantiated ``SimpleCNN`` used for centralized
            evaluation.  The server holds one copy of the global model.
        eval_fn_data: ``(X_val, y_val)`` numpy arrays for centralized evaluation.
            If ``None``, centralized evaluation is skipped.
        **kwargs: Passed to ``FedAvg.__init__`` (min_fit_clients, etc.).
    """

    def __init__(
        self,
        sentinel_logger: StructuredLogger,
        eval_model: SimpleCNN,
        eval_fn_data: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> None:
        # Centralized eval is wired via the `evaluate_fn` FedAvg kwarg.
        if eval_fn_data is not None:
            kwargs["evaluate_fn"] = self._make_evaluate_fn(eval_model, eval_fn_data)
        super().__init__(**kwargs)
        self._sentinel_logger = sentinel_logger
        self._eval_model = eval_model

    # ------------------------------------------------------------------
    # FedAvg overrides
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: FitResultsList,
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results and log summary metrics."""
        if failures:
            logger.warning(
                "Round %d: %d fit failures (clients that did not return).",
                server_round,
                len(failures),
            )

        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is not None:
            # Compute weighted-average client training metrics
            client_metrics = [(r.num_examples, r.metrics) for _, r in results]
            avg_metrics = _weighted_average(client_metrics)
            n_total = sum(r.num_examples for _, r in results)

            self._sentinel_logger.set_round(server_round)
            self._sentinel_logger.log(
                "L1",
                "fit_aggregate",
                {
                    "round": server_round,
                    "n_clients": len(results),
                    "n_failures": len(failures),
                    "n_examples": n_total,
                    "avg_train_loss": round(avg_metrics.get("train_loss", float("nan")), 6),
                    "avg_train_accuracy": round(avg_metrics.get("train_accuracy", 0.0), 6),
                },
            )
            logger.info(
                "Round %d fit aggregate: %d clients, %d examples, "
                "avg_loss=%.4f avg_acc=%.4f",
                server_round,
                len(results),
                n_total,
                avg_metrics.get("train_loss", float("nan")),
                avg_metrics.get("train_accuracy", 0.0),
            )

        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: EvalResultsList,
        failures: list[Any],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation results and log summary metrics."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if results:
            client_metrics = [(r.num_examples, r.metrics) for _, r in results]
            avg_metrics = _weighted_average(client_metrics)
            n_total = sum(r.num_examples for _, r in results)

            self._sentinel_logger.log(
                "L1",
                "eval_aggregate",
                {
                    "round": server_round,
                    "n_clients": len(results),
                    "n_examples": n_total,
                    "avg_val_accuracy": round(avg_metrics.get("accuracy", 0.0), 6),
                    "avg_val_loss": round(float(aggregated_loss or 0.0), 6),
                },
            )
            logger.info(
                "Round %d eval aggregate: %d clients, %d examples, "
                "avg_accuracy=%.4f",
                server_round,
                len(results),
                n_total,
                avg_metrics.get("accuracy", 0.0),
            )

        return aggregated_loss, aggregated_metrics

    # ------------------------------------------------------------------
    # Centralized evaluation helper
    # ------------------------------------------------------------------

    def _make_evaluate_fn(
        self,
        model: SimpleCNN,
        data: tuple[np.ndarray, np.ndarray],
    ):
        """Build a Flower-compatible evaluate function for centralized evaluation.

        The returned function is passed to ``FedAvg(evaluate_fn=...)``.
        It evaluates the global model on the server's clean holdout set.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_val, y_val = data
        X_t = torch.tensor(X_val, dtype=torch.float32)
        y_t = torch.tensor(y_val, dtype=torch.long)
        val_dataset = TensorDataset(X_t, y_t)

        def evaluate_fn(
            server_round: int,
            parameters: NDArrays,
            config: dict[str, Scalar],
        ) -> tuple[float, dict[str, Scalar]]:
            """Centralized evaluation on the server's clean holdout."""
            set_model_parameters(model, parameters)
            model.eval()
            criterion = nn.NLLLoss()
            loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    total_loss += loss.item() * len(y_batch)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += len(y_batch)

            loss_val = total_loss / max(total, 1)
            accuracy = correct / max(total, 1)
            self._sentinel_logger.log(
                "L1",
                "centralized_eval",
                {
                    "round": server_round,
                    "clean_accuracy": round(accuracy, 6),
                    "loss": round(loss_val, 6),
                    "n_examples": total,
                },
            )
            logger.info(
                "Round %d centralized eval: clean_accuracy=%.4f loss=%.4f",
                server_round,
                accuracy,
                loss_val,
            )
            return float(loss_val), {"clean_accuracy": float(accuracy)}

        return evaluate_fn


# ---------------------------------------------------------------------------
# Milestone 5: SentinelFedAvgWithGuard
# ---------------------------------------------------------------------------


class SentinelFedAvgWithGuard(SentinelFedAvg):
    """FedAvg strategy with integrated L1 Update Guard.

    Extends ``SentinelFedAvg`` by running ``UpdateGuard.process_round()``
    inside ``aggregate_fit()`` before delegation to FedAvg.  Maintains a
    copy of the previous round's global parameters so it can extract
    per-client delta vectors from each ``FitRes``.

    The existing ``SentinelFedAvg`` is completely unchanged; this class
    adds a single override of ``aggregate_fit()``.

    Args:
        update_guard: Configured ``UpdateGuard`` instance.
        initial_params: Global model parameters at round 0 (before training).
            Used to compute deltas in the first round.
        **kwargs: Passed to ``SentinelFedAvg.__init__``.
    """

    def __init__(
        self,
        update_guard: Any,          # UpdateGuard (avoid circular import)
        initial_params: NDArrays,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._update_guard = update_guard
        self._prev_params: NDArrays = initial_params
        logger.info("SentinelFedAvgWithGuard: Update Guard wired into aggregation.")

    def aggregate_fit(
        self,
        server_round: int,
        results: FitResultsList,
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Any]]:
        """Run Update Guard analysis then delegate to FedAvg aggregation.

        Steps:
          1. Extract per-client delta vectors from FitRes objects.
          2. Run ``UpdateGuard.process_round()`` for anomaly analysis.
          3. If ``exclude_flagged_clients=True``, remove flagged clients.
          4. Delegate to ``SentinelFedAvg.aggregate_fit()`` with (filtered) results.
          5. Update stored global parameters for the next round.

        Args:
            server_round: Current FL round number.
            results: Flower ``(ClientProxy, FitRes)`` pairs.
            failures: Failed client responses.

        Returns:
            Aggregated ``Parameters`` and metrics dict.
        """
        from flwr.common import parameters_to_ndarrays

        from ai.detection.gradient_extractor import extract_all_deltas

        if failures:
            logger.warning(
                "SentinelFedAvgWithGuard: %d fit failures in round %d.",
                len(failures), server_round,
            )

        # ── 1. Extract deltas ────────────────────────────────────────────
        client_ids = [proxy.cid for proxy, _ in results]
        try:
            deltas = extract_all_deltas(self._prev_params, results)
        except Exception as exc:
            logger.warning(
                "SentinelFedAvgWithGuard: delta extraction failed: %s. "
                "Skipping Update Guard for round %d.",
                exc, server_round,
            )
            return super().aggregate_fit(server_round, results, failures)

        # ── 2. Update Guard analysis ─────────────────────────────────────
        try:
            guard_result = self._update_guard.process_round(
                round_num=server_round,
                client_ids=client_ids,
                deltas=deltas,
            )
            logger.info(
                "SentinelFedAvgWithGuard round %d: %s",
                server_round, guard_result.summary(),
            )
        except Exception as exc:
            logger.warning(
                "SentinelFedAvgWithGuard: Update Guard failed: %s. "
                "Proceeding with unfiltered aggregation.",
                exc,
            )
            guard_result = None

        # ── 3. Optionally filter flagged clients ─────────────────────────
        filtered_results = results
        if guard_result is not None and guard_result.excluded_clients:
            excluded_set = set(guard_result.excluded_clients)
            filtered_results = [
                (proxy, fit_res)
                for proxy, fit_res in results
                if proxy.cid not in excluded_set
            ]
            n_excluded = len(results) - len(filtered_results)
            logger.info(
                "SentinelFedAvgWithGuard: excluded %d/%d clients from aggregation "
                "(round %d): %s",
                n_excluded, len(results), server_round, guard_result.excluded_clients,
            )
            if not filtered_results:
                logger.warning(
                    "SentinelFedAvgWithGuard: ALL clients excluded — "
                    "falling back to full result set to avoid empty aggregation."
                )
                filtered_results = results

        # ── 4. Delegate to parent strategy ───────────────────────────────
        new_params, metrics = super().aggregate_fit(
            server_round, filtered_results, failures
        )

        # ── 5. Update stored params for next round ───────────────────────
        if new_params is not None:
            try:
                self._prev_params = parameters_to_ndarrays(new_params)
            except Exception as exc:
                logger.warning(
                    "SentinelFedAvgWithGuard: failed to update prev_params: %s", exc
                )

        return new_params, metrics

