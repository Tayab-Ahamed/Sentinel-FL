"""
ai/fl_engine/simulation.py — Flower-based FL simulation runner.

Implements a complete federated learning simulation loop using Flower's
strategy and client interfaces directly, without requiring Ray.  This
approach is compatible with all Python versions and environments, and is
functionally equivalent to Flower's ``start_simulation`` with a single-process
backend — the same round-by-round logic, just orchestrated in plain Python.

Architecture:
  Each round:
    1. Server sends global parameters to all selected clients.
    2. Each client runs ``fit()`` (local training) and returns updated parameters.
    3. Strategy ``aggregate_fit()`` combines the results (FedAvg).
    4. Selected clients run ``evaluate()`` on the new parameters.
    5. Strategy ``aggregate_evaluate()`` computes mean metrics.
    6. If ``evaluate_fn`` provided, run centralized server-side eval.

Returns a ``SimulationResult`` dataclass with per-round metrics, suitable for
writing to ``experiments/<experiment_id>.json``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from ai.fl_core.config import load_config
from ai.fl_core.logger import make_logger
from ai.fl_core.model_registry import FileModelRegistry
from ai.fl_core.schemas import Configuration, ModelMetadata
from ai.fl_engine.client import MNISTFlowerClient
from ai.fl_engine.strategy import SentinelFedAvg
from ai.models.mnist_cnn import SimpleCNN, get_model_parameters
from ai.training.mnist_loader import MNISTDatasetLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Structured result returned by ``run_simulation``."""

    experiment_id: str
    n_rounds: int
    n_clients: int
    rounds_history: list[dict[str, Any]] = field(default_factory=list)
    final_clean_accuracy: float | None = None
    final_loss: float | None = None
    total_wall_time_s: float = 0.0
    experiment_path: str | None = None


# ---------------------------------------------------------------------------
# Thin ClientProxy shim (wraps MNISTFlowerClient for FedAvg strategy)
# ---------------------------------------------------------------------------


class _InProcessClientProxy(ClientProxy):
    """Wraps an MNISTFlowerClient as a Flower ClientProxy.

    Translates Flower ``Ins``/``Res`` message objects to the ``NumPyClient``
    interface and back, allowing the ``SentinelFedAvg`` strategy (which expects
    ``ClientProxy`` objects) to work without a real gRPC transport.
    """

    def __init__(self, cid: str, client: MNISTFlowerClient) -> None:
        super().__init__(cid)
        self._client = client

    def get_properties(self, ins: Any, timeout: float | None, group_id: int | None) -> Any:
        raise NotImplementedError("Properties not used in this simulation.")

    def get_parameters(self, ins: GetParametersIns, timeout: float | None, group_id: int | None) -> Any:
        raise NotImplementedError("Parameters fetched directly in round loop.")

    def fit(self, ins: FitIns, timeout: float | None, group_id: int | None) -> FitRes:
        params_np = parameters_to_ndarrays(ins.parameters)
        updated_params_np, n_examples, metrics = self._client.fit(
            params_np, config=dict(ins.config)
        )
        return FitRes(
            status=_ok_status(),
            parameters=ndarrays_to_parameters(updated_params_np),
            num_examples=n_examples,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns, timeout: float | None, group_id: int | None) -> EvaluateRes:
        params_np = parameters_to_ndarrays(ins.parameters)
        loss, n_examples, metrics = self._client.evaluate(
            params_np, config=dict(ins.config)
        )
        return EvaluateRes(
            status=_ok_status(),
            loss=float(loss),
            num_examples=n_examples,
            metrics=metrics,
        )

    def reconnect(self, ins: Any, timeout: float | None, group_id: int | None) -> Any:
        raise NotImplementedError


def _ok_status():
    """Return a Flower OK status object."""
    from flwr.common import Code, Status
    return Status(code=Code.OK, message="")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_simulation(
    config: Configuration | None = None,
    config_path: str | Path | None = None,
    experiment_id: str | None = None,
    experiments_dir: str | Path = "experiments",
    data_dir: str | Path = "datasets",
    log_sink: str | None = None,
    verbose: bool = True,
) -> SimulationResult:
    """Run a federated learning simulation with MNIST and FedAvg.

    Args:
        config: Pre-built Configuration object.  If ``None``, loaded from
            ``config_path``.
        config_path: Path to a YAML config file.
        experiment_id: Identifier for this experiment (auto-generated if None).
        experiments_dir: Root directory for experiment artefacts.
        data_dir: Directory for MNIST download cache.
        log_sink: Override for log destination.
        verbose: If True, emit INFO-level console logs.

    Returns:
        ``SimulationResult`` with per-round metrics and final accuracy.
    """
    # ── 1. Configuration ───────────────────────────────────────────────
    if config is None:
        if config_path is None:
            raise ValueError("Either config or config_path must be provided.")
        config = load_config(config_path)

    if experiment_id is None:
        import uuid
        experiment_id = f"flower_baseline_{uuid.uuid4().hex[:8]}"

    experiments_path = Path(experiments_dir)
    experiments_path.mkdir(parents=True, exist_ok=True)

    # ── 2. Logging ─────────────────────────────────────────────────────
    exp_log_path = experiments_path / f"{experiment_id}.jsonl"
    sink = log_sink or str(exp_log_path)
    sentinel_logger = make_logger(
        log_level="INFO" if verbose else "WARNING",
        log_sink=sink,
    )
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )

    sentinel_logger.log(
        "L1",
        "simulation_start",
        {
            "experiment_id": experiment_id,
            "n_clients": config.n_clients,
            "n_rounds": config.n_rounds,
            "local_epochs": config.local_epochs,
            "seed": config.seed,
        },
    )
    logger.info(
        "Starting simulation: experiment=%s n_clients=%d n_rounds=%d",
        experiment_id, config.n_clients, config.n_rounds,
    )

    # ── 3. Reproducibility ─────────────────────────────────────────────
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ── 4. Dataset ─────────────────────────────────────────────────────
    mnist_loader = MNISTDatasetLoader(
        data_dir=data_dir, dirichlet_alpha=0.5, seed=config.seed,
    )
    partitions = mnist_loader.load_client_partitions(config.n_clients, config)
    X_val, y_val = mnist_loader.load_clean_holdout()
    logger.info("Dataset loaded: %d partitions, holdout=%d", len(partitions), len(y_val))

    # ── 5. Model Registry ──────────────────────────────────────────────
    registry_dir = Path(config.model_registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry = FileModelRegistry(registry_dir)

    # ── 6. Global model + initial parameters ───────────────────────────
    global_model = SimpleCNN()
    current_params_np: list[np.ndarray] = get_model_parameters(global_model)
    current_params: Parameters = ndarrays_to_parameters(current_params_np)

    # ── 7. Build client proxies ────────────────────────────────────────
    def _make_client(idx: int) -> MNISTFlowerClient:
        X_train, y_train = partitions[idx]
        n = len(X_train)
        split = max(1, int(n * 0.9))
        return MNISTFlowerClient(
            client_id=f"client_{idx:02d}",
            train_data=(X_train[:split], y_train[:split]),
            val_data=(X_train[split:], y_train[split:]),
            local_epochs=config.local_epochs,
            learning_rate=config.local_lr,
            batch_size=32,
            device="cpu",
            sentinel_logger=sentinel_logger,
        )

    proxies: list[_InProcessClientProxy] = [
        _InProcessClientProxy(str(i), _make_client(i))
        for i in range(config.n_clients)
    ]

    # ── 8. Strategy ────────────────────────────────────────────────────
    eval_model = SimpleCNN()
    min_fit = max(2, int(config.n_clients * 0.5))
    strategy = SentinelFedAvg(
        sentinel_logger=sentinel_logger,
        eval_model=eval_model,
        eval_fn_data=(X_val, y_val),
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=min_fit,
        min_evaluate_clients=max(2, min_fit // 2),
        min_available_clients=min_fit,
        initial_parameters=current_params,
    )

    # ── 9. Round loop ──────────────────────────────────────────────────
    t_start = time.perf_counter()
    rounds_history: list[dict[str, Any]] = []
    all_centralized_losses: list[tuple[int, float]] = []
    all_centralized_metrics: dict[str, list[tuple[int, float]]] = {}

    rng = np.random.default_rng(config.seed)

    for rnd in range(1, config.n_rounds + 1):
        logger.info("── Round %d/%d ──", rnd, config.n_rounds)
        round_cfg = {"round": rnd}

        # ── 9a. Fit ────────────────────────────────────────────────────
        # Select all clients (fraction_fit=1.0)
        fit_ins = FitIns(parameters=current_params, config=round_cfg)
        fit_results: list[tuple[ClientProxy, FitRes]] = []
        for proxy in proxies:
            res = proxy.fit(fit_ins, timeout=None, group_id=0)
            fit_results.append((proxy, res))

        new_params, _ = strategy.aggregate_fit(rnd, fit_results, [])
        if new_params is not None:
            current_params = new_params

        # ── 9b. Evaluate (federated) ───────────────────────────────────
        n_eval = max(2, config.n_clients // 2)
        eval_proxies = [proxies[i] for i in rng.choice(
            len(proxies), size=min(n_eval, len(proxies)), replace=False
        )]
        eval_ins = EvaluateIns(parameters=current_params, config=round_cfg)
        eval_results: list[tuple[ClientProxy, EvaluateRes]] = []
        for proxy in eval_proxies:
            res = proxy.evaluate(eval_ins, timeout=None, group_id=0)
            eval_results.append((proxy, res))

        strategy.aggregate_evaluate(rnd, eval_results, [])

        # ── 9c. Centralized eval ───────────────────────────────────────
        params_np = parameters_to_ndarrays(current_params)
        # Call strategy's internal evaluate_fn if it exists
        if strategy.evaluate_fn is not None:
            central_loss, central_metrics = strategy.evaluate_fn(rnd, params_np, {})
            all_centralized_losses.append((rnd, central_loss))
            for k, v in central_metrics.items():
                all_centralized_metrics.setdefault(k, []).append((rnd, float(v)))
            rounds_history.append({
                "round": rnd,
                "centralized_loss": round(float(central_loss), 6),
                **{k: round(float(v), 6) for k, v in central_metrics.items()},
            })
            logger.info(
                "Round %d: loss=%.4f %s",
                rnd, central_loss,
                " ".join(f"{k}={v:.4f}" for k, v in central_metrics.items()),
            )
        else:
            rounds_history.append({"round": rnd})

        # ── 9d. Save checkpoint ────────────────────────────────────────
        try:
            params_np_list = parameters_to_ndarrays(current_params)
            state_to_save = {"params": [p.tolist() for p in params_np_list], "round": rnd}
            meta = ModelMetadata(round_num=rnd, architecture="mnist_simplecnn_v1")
            registry.save(rnd, state_to_save, meta)
        except Exception as exc:  # broad catch: checkpoint failure must not abort training
            logger.warning("Round %d: checkpoint save failed: %s", rnd, exc)

    t_elapsed = time.perf_counter() - t_start
    logger.info("Simulation complete in %.1fs.", t_elapsed)

    # ── 10. Final metrics ──────────────────────────────────────────────
    final_accuracy: float | None = None
    final_loss: float | None = None
    if all_centralized_metrics.get("clean_accuracy"):
        final_accuracy = float(all_centralized_metrics["clean_accuracy"][-1][1])
    if all_centralized_losses:
        final_loss = float(all_centralized_losses[-1][1])

    result = SimulationResult(
        experiment_id=experiment_id,
        n_rounds=config.n_rounds,
        n_clients=config.n_clients,
        rounds_history=rounds_history,
        final_clean_accuracy=final_accuracy,
        final_loss=final_loss,
        total_wall_time_s=round(t_elapsed, 2),
    )

    out_path = experiments_path / f"{experiment_id}.json"
    _write_result_json(result, out_path)
    result.experiment_path = str(out_path)

    sentinel_logger.log(
        "L1",
        "simulation_complete",
        {
            "experiment_id": experiment_id,
            "final_clean_accuracy": final_accuracy,
            "final_loss": final_loss,
            "total_wall_time_s": t_elapsed,
        },
    )
    logger.info(
        "Results written to %s  final_accuracy=%s",
        out_path,
        f"{final_accuracy * 100:.2f}%" if final_accuracy is not None else "N/A",
    )
    return result


# ---------------------------------------------------------------------------
# History-extraction helpers (used by tests)
# ---------------------------------------------------------------------------


def _extract_rounds(history: Any) -> list[dict[str, Any]]:
    """Extract per-round metrics from a Flower History-like object."""
    rounds = []
    losses_centralized = dict(history.losses_centralized or [])
    metrics_centralized: dict[int, dict[str, float]] = {}
    for key, vals in (history.metrics_centralized or {}).items():
        for rnd, v in vals:
            metrics_centralized.setdefault(rnd, {})[key] = v

    all_rounds = sorted(
        set(list(losses_centralized.keys()) + list(metrics_centralized.keys()))
    )
    for rnd in all_rounds:
        entry: dict[str, Any] = {"round": rnd}
        if rnd in losses_centralized:
            entry["centralized_loss"] = round(float(losses_centralized[rnd]), 6)
        entry.update(
            {k: round(float(v), 6) for k, v in metrics_centralized.get(rnd, {}).items()}
        )
        rounds.append(entry)
    return rounds


def _extract_final_metrics(
    history: Any,
) -> tuple[float | None, float | None]:
    """Pull the final accuracy and loss from a Flower History-like object."""
    accuracy = None
    loss = None
    if history.metrics_centralized and "clean_accuracy" in history.metrics_centralized:
        vals = history.metrics_centralized["clean_accuracy"]
        if vals:
            accuracy = float(vals[-1][1])
    if history.losses_centralized:
        loss = float(history.losses_centralized[-1][1])
    return accuracy, loss


def _write_result_json(result: SimulationResult, path: Path) -> None:
    """Serialise SimulationResult to a JSON file."""
    data = {
        "experiment_id": result.experiment_id,
        "n_rounds": result.n_rounds,
        "n_clients": result.n_clients,
        "final_clean_accuracy": result.final_clean_accuracy,
        "final_loss": result.final_loss,
        "total_wall_time_s": result.total_wall_time_s,
        "rounds": result.rounds_history,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Milestone 5: run_simulation_with_guard
# ---------------------------------------------------------------------------


def run_simulation_with_guard(
    config: Configuration | None = None,
    config_path: str | Path | None = None,
    experiment_id: str | None = None,
    experiments_dir: str | Path = "experiments",
    data_dir: str | Path = "datasets",
    log_sink: str | None = None,
    verbose: bool = True,
    exclude_flagged_clients: bool = False,
) -> SimulationResult:
    """Run FL simulation with the L1 Update Guard active each round.

    Extends ``run_simulation()`` by wiring ``UpdateGuard`` and
    ``SentinelFedAvgWithGuard`` into the aggregation loop.  Per-round
    anomaly reports are logged to the StructuredLogger and the Trust Ledger.

    All other behaviour (dataset loading, model checkpointing, centralized
    evaluation) is identical to ``run_simulation()``.

    Args:
        config: Pre-built Configuration object. If None, loaded from
            ``config_path``.
        config_path: Path to a YAML config file.
        experiment_id: Identifier for this experiment (auto-generated if None).
        experiments_dir: Root directory for experiment artefacts.
        data_dir: Directory for dataset download cache.
        log_sink: Override for log destination.
        verbose: If True, emit INFO-level console logs.
        exclude_flagged_clients: Pass-through to ``UpdateGuard``; if True,
            norm outliers and colluders are excluded from aggregation.

    Returns:
        ``SimulationResult`` with per-round metrics.
    """
    from ai.detection.trust_ledger import FileTrustLedger
    from ai.detection.update_guard import UpdateGuard
    from ai.fl_engine.strategy import SentinelFedAvgWithGuard

    # ── 1. Configuration ───────────────────────────────────────────────
    if config is None:
        if config_path is None:
            raise ValueError("Either config or config_path must be provided.")
        config = load_config(config_path)

    if experiment_id is None:
        import uuid
        experiment_id = f"guard_{uuid.uuid4().hex[:8]}"

    experiments_path = Path(experiments_dir)
    experiments_path.mkdir(parents=True, exist_ok=True)

    # ── 2. Logging ─────────────────────────────────────────────────────
    exp_log_path = experiments_path / f"{experiment_id}.jsonl"
    sink = log_sink or str(exp_log_path)
    sentinel_logger = make_logger(
        log_level="INFO" if verbose else "WARNING",
        log_sink=sink,
    )
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
    logger.info(
        "Starting guarded simulation: experiment=%s n_clients=%d n_rounds=%d",
        experiment_id, config.n_clients, config.n_rounds,
    )

    # ── 3. Reproducibility ─────────────────────────────────────────────
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ── 4. Dataset ─────────────────────────────────────────────────────
    mnist_loader = MNISTDatasetLoader(
        data_dir=data_dir, dirichlet_alpha=0.5, seed=config.seed,
    )
    partitions = mnist_loader.load_client_partitions(config.n_clients, config)
    X_val, y_val = mnist_loader.load_clean_holdout()

    # ── 5. Model Registry ──────────────────────────────────────────────
    registry_dir = Path(config.model_registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry = FileModelRegistry(registry_dir)

    # ── 6. Global model + initial parameters ───────────────────────────
    global_model = SimpleCNN()
    current_params_np: list[np.ndarray] = get_model_parameters(global_model)
    current_params: Parameters = ndarrays_to_parameters(current_params_np)

    # ── 7. Update Guard ─────────────────────────────────────────────────
    ledger_path = experiments_path / f"{experiment_id}_trust_ledger.jsonl"
    ledger = FileTrustLedger(ledger_path=ledger_path, decay_rate=0.1)
    update_guard = UpdateGuard.from_config(
        config=config,
        sentinel_logger=sentinel_logger,
        ledger=ledger,
    )
    # Override the exclude flag with the caller's argument
    update_guard._exclude_flagged = exclude_flagged_clients
    logger.info("UpdateGuard wired: exclude_flagged=%s", exclude_flagged_clients)

    # ── 8. Build client proxies ────────────────────────────────────────
    def _make_client(idx: int) -> MNISTFlowerClient:
        X_train, y_train = partitions[idx]
        n = len(X_train)
        split = max(1, int(n * 0.9))
        return MNISTFlowerClient(
            client_id=f"client_{idx:02d}",
            train_data=(X_train[:split], y_train[:split]),
            val_data=(X_train[split:], y_train[split:]),
            local_epochs=config.local_epochs,
            learning_rate=config.local_lr,
            batch_size=32,
            device="cpu",
            sentinel_logger=sentinel_logger,
        )

    proxies: list[_InProcessClientProxy] = [
        _InProcessClientProxy(str(i), _make_client(i))
        for i in range(config.n_clients)
    ]

    # ── 9. Strategy ────────────────────────────────────────────────────
    eval_model = SimpleCNN()
    min_fit = max(2, int(config.n_clients * 0.5))
    strategy = SentinelFedAvgWithGuard(
        update_guard=update_guard,
        initial_params=current_params_np,
        sentinel_logger=sentinel_logger,
        eval_model=eval_model,
        eval_fn_data=(X_val, y_val),
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=min_fit,
        min_evaluate_clients=max(2, min_fit // 2),
        min_available_clients=min_fit,
        initial_parameters=current_params,
    )

    # ── 10. Round loop ──────────────────────────────────────────────────
    t_start = time.perf_counter()
    rounds_history: list[dict] = []
    all_centralized_losses: list[tuple[int, float]] = []
    all_centralized_metrics: dict[str, list[tuple[int, float]]] = {}
    rng = np.random.default_rng(config.seed)

    for rnd in range(1, config.n_rounds + 1):
        logger.info("── Round %d/%d ──", rnd, config.n_rounds)
        round_cfg = {"round": rnd}

        # Fit
        fit_ins = FitIns(parameters=current_params, config=round_cfg)
        fit_results: list[tuple[ClientProxy, FitRes]] = [
            (proxy, proxy.fit(fit_ins, timeout=None, group_id=0))
            for proxy in proxies
        ]
        new_params, _ = strategy.aggregate_fit(rnd, fit_results, [])
        if new_params is not None:
            current_params = new_params

        # Decay trust scores each round
        ledger.decay_scores(rnd)

        # Evaluate (federated)
        n_eval = max(2, config.n_clients // 2)
        eval_proxies = [proxies[i] for i in rng.choice(
            len(proxies), size=min(n_eval, len(proxies)), replace=False
        )]
        eval_ins = EvaluateIns(parameters=current_params, config=round_cfg)
        eval_results: list[tuple[ClientProxy, EvaluateRes]] = [
            (proxy, proxy.evaluate(eval_ins, timeout=None, group_id=0))
            for proxy in eval_proxies
        ]
        strategy.aggregate_evaluate(rnd, eval_results, [])

        # Centralized eval
        params_np = parameters_to_ndarrays(current_params)
        if strategy.evaluate_fn is not None:
            central_loss, central_metrics = strategy.evaluate_fn(rnd, params_np, {})
            all_centralized_losses.append((rnd, central_loss))
            for k, v in central_metrics.items():
                all_centralized_metrics.setdefault(k, []).append((rnd, float(v)))
            rounds_history.append({
                "round": rnd,
                "centralized_loss": round(float(central_loss), 6),
                **{k: round(float(v), 6) for k, v in central_metrics.items()},
            })
            logger.info(
                "Round %d: loss=%.4f %s",
                rnd, central_loss,
                " ".join(f"{k}={v:.4f}" for k, v in central_metrics.items()),
            )
        else:
            rounds_history.append({"round": rnd})

        # Checkpoint
        try:
            params_np_list = parameters_to_ndarrays(current_params)
            state_to_save = {"params": [p.tolist() for p in params_np_list], "round": rnd}
            meta = ModelMetadata(round_num=rnd, architecture="mnist_simplecnn_v1")
            registry.save(rnd, state_to_save, meta)
        except Exception as exc:
            logger.warning("Round %d: checkpoint save failed: %s", rnd, exc)

    t_elapsed = time.perf_counter() - t_start
    logger.info("Guarded simulation complete in %.1fs.", t_elapsed)

    final_accuracy = None
    final_loss = None
    if all_centralized_metrics.get("clean_accuracy"):
        final_accuracy = float(all_centralized_metrics["clean_accuracy"][-1][1])
    if all_centralized_losses:
        final_loss = float(all_centralized_losses[-1][1])

    result = SimulationResult(
        experiment_id=experiment_id,
        n_rounds=config.n_rounds,
        n_clients=config.n_clients,
        rounds_history=rounds_history,
        final_clean_accuracy=final_accuracy,
        final_loss=final_loss,
        total_wall_time_s=round(t_elapsed, 2),
    )
    out_path = experiments_path / f"{experiment_id}.json"
    _write_result_json(result, out_path)
    result.experiment_path = str(out_path)

    # Log trust ledger summary
    all_trust = ledger.get_all_scores()
    sentinel_logger.log(
        "L1",
        "simulation_complete_with_guard",
        {
            "experiment_id": experiment_id,
            "final_clean_accuracy": final_accuracy,
            "final_loss": final_loss,
            "total_wall_time_s": t_elapsed,
            "trust_scores": {ts.subject_id: ts.score for ts in all_trust},
        },
    )
    logger.info(
        "Guard simulation results written to %s  final_accuracy=%s",
        out_path,
        f"{final_accuracy * 100:.2f}%" if final_accuracy else "N/A",
    )
    return result
