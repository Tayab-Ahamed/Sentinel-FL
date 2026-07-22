#!/usr/bin/env python3
"""
scripts/run_flower.py — CLI entry point for the SENTINEL-FL Flower simulation.

Usage:
    python scripts/run_flower.py [--config configs/flower.yaml] [--rounds N]
                                  [--clients N] [--epochs N]
                                  [--experiment-id ID]
                                  [--experiments-dir experiments]
                                  [--data-dir datasets]
                                  [--quiet]

All CLI flags override the values loaded from the config YAML.

Exit codes:
    0 — simulation completed successfully
    1 — configuration error or simulation failure
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.fl_core.config import load_config
from ai.fl_engine.simulation import run_simulation


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SENTINEL-FL Flower federated learning simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/flower.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override n_rounds from the config.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=None,
        help="Override n_clients from the config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override local_epochs from the config.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Unique ID for this run. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Directory for experiment artefacts (JSON results, checkpoints).",
    )
    parser.add_argument(
        "--data-dir",
        default="datasets",
        help="Directory for MNIST download cache.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO-level console output (WARNING only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns exit code."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load and optionally override configuration ────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"ERROR: Failed to load config: {exc}", file=sys.stderr)
        return 1

    # Apply CLI overrides (immutable Pydantic model → copy with field changes)
    overrides: dict[str, object] = {}
    if args.rounds is not None:
        overrides["n_rounds"] = args.rounds
    if args.clients is not None:
        overrides["n_clients"] = args.clients
    if args.epochs is not None:
        overrides["local_epochs"] = args.epochs

    if overrides:
        config = config.model_copy(update=overrides)

    # ── Run simulation ────────────────────────────────────────────────────
    print(
        f"\n{'='*60}\n"
        f"  SENTINEL-FL -- Flower Baseline\n"
        f"  clients={config.n_clients}  rounds={config.n_rounds}  "
        f"local_epochs={config.local_epochs}  lr={config.local_lr}\n"
        f"{'='*60}\n"
    )

    try:
        result = run_simulation(
            config=config,
            experiment_id=args.experiment_id,
            experiments_dir=args.experiments_dir,
            data_dir=args.data_dir,
            verbose=not args.quiet,
        )
    except Exception as exc:
        logging.exception("Simulation failed: %s", exc)
        return 1

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Experiment ID : {result.experiment_id}")
    print(f"  Rounds        : {result.n_rounds}")
    print(f"  Clients       : {result.n_clients}")
    print(f"  Wall time     : {result.total_wall_time_s:.1f}s")
    if result.final_clean_accuracy is not None:
        print(f"  Final C-Acc   : {result.final_clean_accuracy * 100:.2f}%")
    if result.final_loss is not None:
        print(f"  Final loss    : {result.final_loss:.4f}")
    print(f"  Results JSON  : {result.experiment_path}")
    print(f"{'='*60}\n")

    # Print per-round accuracy table
    if result.rounds_history:
        print(f"  {'Round':>5}  {'C-Acc':>8}  {'Loss':>8}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}")
        for row in result.rounds_history:
            acc = row.get("clean_accuracy", row.get("centralized_eval_clean_accuracy"))
            loss = row.get("centralized_loss", "-")
            acc_str = f"{float(acc) * 100:.2f}%" if acc is not None else "    -"
            loss_str = f"{float(loss):.4f}" if isinstance(loss, (int, float)) else "      -"
            print(f"  {row['round']:>5}  {acc_str:>8}  {loss_str:>8}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
