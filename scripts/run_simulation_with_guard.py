#!/usr/bin/env python
"""
scripts/run_simulation_with_guard.py — SENTINEL-FL Update Guard simulation CLI.

Runs the full FL simulation with the L1 Update Guard active, printing
per-round norms, collusion clusters, anomaly scores, and trust rankings.

Usage::

    python scripts/run_simulation_with_guard.py --rounds 3 --clients 6
    python scripts/run_simulation_with_guard.py --config configs/config.yaml --rounds 5

Output::
    experiments/<experiment_id>.json         — SimulationResult
    experiments/<experiment_id>.jsonl        — structured event log
    experiments/<experiment_id>_trust_ledger.jsonl  — Trust Ledger
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai.fl_core.config import load_config
from ai.fl_engine.simulation import run_simulation_with_guard

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SENTINEL-FL Update Guard FL simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/config.yaml", help="YAML config path.")
    p.add_argument("--rounds", type=int, default=None, help="Override n_rounds.")
    p.add_argument("--clients", type=int, default=None, help="Override n_clients.")
    p.add_argument("--data-dir", default="datasets", help="Dataset cache directory.")
    p.add_argument("--experiments-dir", default="experiments", help="Experiment output root.")
    p.add_argument(
        "--exclude-flagged",
        action="store_true",
        default=False,
        help="Exclude flagged clients from FedAvg aggregation.",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load and optionally override config
    config = load_config(args.config)
    if args.rounds is not None:
        config.n_rounds = args.rounds
    if args.clients is not None:
        config.n_clients = args.clients

    logger.info("=" * 60)
    logger.info("SENTINEL-FL — Update Guard Simulation")
    logger.info("  Rounds:   %d", config.n_rounds)
    logger.info("  Clients:  %d", config.n_clients)
    logger.info("  Exclude flagged: %s", args.exclude_flagged)
    logger.info("=" * 60)

    result = run_simulation_with_guard(
        config=config,
        experiments_dir=args.experiments_dir,
        data_dir=args.data_dir,
        exclude_flagged_clients=args.exclude_flagged,
        verbose=True,
    )

    logger.info("=" * 60)
    logger.info("Simulation complete.")
    logger.info("  Experiment ID:   %s", result.experiment_id)
    logger.info("  Final C-Acc:     %s",
                f"{result.final_clean_accuracy * 100:.2f}%"
                if result.final_clean_accuracy else "N/A")
    logger.info("  Wall time:       %.1fs", result.total_wall_time_s)
    logger.info("  Output:          %s", result.experiment_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
