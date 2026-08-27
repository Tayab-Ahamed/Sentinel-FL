#!/usr/bin/env python
"""
scripts/run_attack.py — Attack engine CLI for SENTINEL-FL Milestone 4.

Runs a configurable BadNets image attack across multiple FL rounds,
evaluates ASR and clean accuracy each round, and saves visualisation
artifacts to disk.

Usage::

    python scripts/run_attack.py --dataset mnist --rounds 5 --clients 6
    python scripts/run_attack.py --dataset cifar10 --rounds 3 --poison-frac 0.20
    python scripts/run_attack.py --config configs/attack.yaml --dataset mnist

Output::
    experiments/attack_eval/
        run_<timestamp>/
            round_reports.jsonl    — per-round attack statistics
            poisoned_grid.png      — clean vs. poisoned sample grid
            asr_curve.png          — ASR & C-Acc over rounds
            summary.json           — final metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Ensure project root is on path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai.attacks.asr_evaluator import AttackSuccessRateEvaluator
from ai.attacks.attack_report import AttackEvalResult
from ai.attacks.badnets import BadNetsImageAttack
from ai.attacks.triggers import TriggerPattern
from ai.attacks.visualizer import PoisonedSampleVisualizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SENTINEL-FL BadNets image attack evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    p.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
        default="mnist",
        help="Dataset to use for the attack experiment.",
    )
    p.add_argument(
        "--data-dir",
        default="datasets",
        help="Root directory for dataset downloads and cache.",
    )

    # FL topology
    p.add_argument("--rounds", type=int, default=5, help="Number of FL rounds.")
    p.add_argument("--clients", type=int, default=6, help="Number of FL clients.")

    # Attack parameters
    p.add_argument("--target-label", type=int, default=0, help="Backdoor target class.")
    p.add_argument(
        "--poison-frac",
        type=float,
        default=0.15,
        help="Fraction of each malicious client's data to poison.",
    )
    p.add_argument(
        "--malicious",
        type=int,
        nargs="+",
        default=[2, 5],
        help="Malicious client indices (space-separated).",
    )

    # Trigger parameters
    p.add_argument(
        "--trigger-shape",
        choices=["square", "cross", "checkerboard", "random_noise"],
        default="square",
        help="Trigger pattern shape.",
    )
    p.add_argument("--trigger-size", type=int, default=4, help="Trigger patch size (pixels).")
    p.add_argument(
        "--trigger-location",
        choices=["bottom_right", "top_left", "top_right", "bottom_left", "center"],
        default="bottom_right",
        help="Trigger placement on the image.",
    )
    p.add_argument("--trigger-color", type=float, default=1.0, help="Trigger pixel intensity.")
    p.add_argument("--trigger-opacity", type=float, default=1.0, help="Trigger opacity [0,1].")

    # Output
    p.add_argument(
        "--output-dir",
        default="experiments/attack_eval",
        help="Root directory for experiment output.",
    )
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return p


# ---------------------------------------------------------------------------
# Data loading (mocked for CLI demo; real loaders plug in here)
# ---------------------------------------------------------------------------


def _load_fake_data(
    dataset: str,
    n_clients: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Generate synthetic data shaped like ``dataset`` for CLI demo.

    Returns:
        ``(client_partitions, X_eval, y_eval)``
    """
    rng = np.random.default_rng(seed)

    if dataset == "mnist":
        C, H, W, n_classes = 1, 28, 28, 10
        n_train_per_client = 100
        n_eval = 200
    else:  # cifar10
        C, H, W, n_classes = 3, 32, 32, 10
        n_train_per_client = 80
        n_eval = 160

    partitions = []
    for _ in range(n_clients):
        X = rng.standard_normal((n_train_per_client, C, H, W)).astype(np.float32)
        y = rng.integers(0, n_classes, size=n_train_per_client).astype(np.int64)
        partitions.append((X, y))

    X_eval = rng.standard_normal((n_eval, C, H, W)).astype(np.float32)
    y_eval = rng.integers(0, n_classes, size=n_eval).astype(np.int64)

    logger.info(
        "Loaded %s data: %d clients, %d eval samples (C=%d H=%d W=%d)",
        dataset,
        n_clients,
        n_eval,
        C,
        H,
        W,
    )
    return partitions, X_eval, y_eval


def _load_real_data(
    dataset: str,
    n_clients: int,
    data_dir: str,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Load real dataset via DatasetRegistry."""
    from types import SimpleNamespace

    from ai.training.datasets.registry import DatasetRegistry

    cfg = SimpleNamespace(
        seed=seed,
        data_dir=data_dir,
        synthetic=SimpleNamespace(dirichlet_alpha=0.5),
    )
    loader = DatasetRegistry.get_loader(dataset, cfg)
    partitions = loader.load_client_partitions(n_clients, cfg)
    X_eval, y_eval = loader.load_evaluation_set()
    return partitions, X_eval, y_eval


# ---------------------------------------------------------------------------
# Stub global model for CLI demo (not trained — for structural demonstration)
# ---------------------------------------------------------------------------


def _make_stub_model(dataset: str):
    """Return an untrained CNN stub for structural demonstration."""
    import torch
    import torch.nn as nn

    class StubCNN(nn.Module):
        """Minimal CNN stub that returns random-ish logits."""

        def __init__(self, in_channels: int, n_classes: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(4)
            self.fc = nn.Linear(8 * 4 * 4, n_classes)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            return self.fc(x.flatten(1))

    in_ch = 1 if dataset == "mnist" else 3
    return StubCNN(in_ch, 10)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_attack(args: argparse.Namespace) -> None:
    """Execute the attack experiment loop."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SENTINEL-FL Attack Experiment")
    logger.info("  Dataset:  %s", args.dataset)
    logger.info("  Rounds:   %d", args.rounds)
    logger.info("  Clients:  %d", args.clients)
    logger.info("  Target:   label=%d", args.target_label)
    logger.info(
        "  Trigger:  %s %dx%d @ %s (opacity=%.1f)",
        args.trigger_shape,
        args.trigger_size,
        args.trigger_size,
        args.trigger_location,
        args.trigger_opacity,
    )
    logger.info("  Malicious clients: %s", args.malicious)
    logger.info("  Output:   %s", run_dir)
    logger.info("=" * 60)

    # ── Build attack objects ─────────────────────────────────────────────
    pattern = TriggerPattern(
        shape=args.trigger_shape,
        size=args.trigger_size,
        location=args.trigger_location,
        color=args.trigger_color,
        opacity=args.trigger_opacity,
    )
    attacker = BadNetsImageAttack(
        target_label=args.target_label,
        poison_fraction=args.poison_frac,
        malicious_client_indices=args.malicious,
        pattern=pattern,
        seed=args.seed,
    )
    evaluator = AttackSuccessRateEvaluator(device="cpu")
    visualizer = PoisonedSampleVisualizer(dpi=100)

    # ── Load data ────────────────────────────────────────────────────────
    try:
        client_partitions, X_eval, y_eval = _load_real_data(
            args.dataset, args.clients, args.data_dir, args.seed
        )
        logger.info("Loaded real %s dataset.", args.dataset)
    except Exception as exc:
        logger.warning("Could not load real dataset (%s); using synthetic stub.", exc)
        client_partitions, X_eval, y_eval = _load_fake_data(args.dataset, args.clients, args.seed)

    # ── Stub global model ─────────────────────────────────────────────────
    model = _make_stub_model(args.dataset)

    # ── Round reports log ─────────────────────────────────────────────────
    reports_path = run_dir / "round_reports.jsonl"
    eval_results: list[AttackEvalResult] = []
    poisoned_grid_saved = False

    # ── FL simulation loop ────────────────────────────────────────────────
    for round_num in range(args.rounds):
        logger.info("── Round %d/%d ──────────────────────────────────", round_num + 1, args.rounds)

        all_X_clean: list[np.ndarray] = []
        all_X_poisoned: list[np.ndarray] = []
        all_mask: list[np.ndarray] = []

        for client_idx, (X_c, y_c) in enumerate(client_partitions):
            client_id = f"client_{client_idx:02d}"
            from types import SimpleNamespace

            cfg_stub = SimpleNamespace(seed=args.seed)
            X_p, _y_p, mask = attacker.poison_client_data(X_c, y_c, client_id, round_num, cfg_stub)
            all_X_clean.append(X_c)
            all_X_poisoned.append(X_p)
            all_mask.append(mask)

        # Concatenate across clients for visualisation
        X_clean_cat = np.concatenate(all_X_clean, axis=0)
        X_pois_cat = np.concatenate(all_X_poisoned, axis=0)
        mask_cat = np.concatenate(all_mask, axis=0)

        # Save poisoned grid on first round with actual poison
        if not poisoned_grid_saved and mask_cat.any():
            fig = visualizer.plot_sample_grid(
                X_clean_cat, X_pois_cat, mask_cat, n_cols=8, n_samples=16
            )
            visualizer.save_figure(fig, run_dir / "poisoned_grid.png")
            poisoned_grid_saved = True
            logger.info("Saved poisoned sample grid → %s", run_dir / "poisoned_grid.png")

        # Save trigger pattern visualisation (once)
        if round_num == 0:
            in_shape = (1, 28, 28) if args.dataset == "mnist" else (3, 32, 32)
            fig = visualizer.plot_trigger_pattern(pattern, input_shape=in_shape)
            visualizer.save_figure(fig, run_dir / "trigger_pattern.png")
            logger.info("Saved trigger pattern → %s", run_dir / "trigger_pattern.png")

        # Evaluate ASR
        eval_result = evaluator.evaluate_round(
            model=model,
            attacker=attacker,
            X_clean_eval=X_eval,
            y_clean_eval=y_eval,
            round_num=round_num,
        )
        eval_results.append(eval_result)
        logger.info(eval_result.summary())

        # Append to JSONL log
        with open(reports_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(eval_result.to_dict()) + "\n")

    # ── Post-experiment visualisations ────────────────────────────────────
    round_nums = [r.round_num for r in eval_results]
    asr_vals = [r.asr for r in eval_results]
    acc_vals = [r.clean_acc for r in eval_results]

    fig = visualizer.plot_asr_curve(round_nums, asr_vals, acc_vals)
    visualizer.save_figure(fig, run_dir / "asr_curve.png")
    logger.info("Saved ASR curve → %s", run_dir / "asr_curve.png")

    # ── Summary JSON ──────────────────────────────────────────────────────
    final = eval_results[-1] if eval_results else None
    summary = {
        "dataset": args.dataset,
        "rounds": args.rounds,
        "n_clients": args.clients,
        "malicious_clients": args.malicious,
        "target_label": args.target_label,
        "poison_fraction": args.poison_frac,
        "trigger": {
            "shape": args.trigger_shape,
            "size": args.trigger_size,
            "location": args.trigger_location,
            "color": args.trigger_color,
            "opacity": args.trigger_opacity,
        },
        "final_asr": final.asr if final else None,
        "final_clean_acc": final.clean_acc if final else None,
        "output_dir": str(run_dir),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("=" * 60)
    logger.info("Experiment complete.")
    logger.info("  Final ASR:     %.1f%%", (final.asr or 0) * 100)
    logger.info("  Final C-Acc:   %.1f%%", (final.clean_acc or 0) * 100)
    logger.info("  Artifacts:     %s", run_dir)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_attack(args)


if __name__ == "__main__":
    main()
