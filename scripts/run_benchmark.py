"""
scripts/run_benchmark.py — Baseline comparison benchmark runner.

Runs the full ablation matrix (FedAvg / Multi-Krum / Multi-Krum+Guard) across
multiple attack scenarios and prints a comparison table matching the evaluation
plan in BENCHMARK.md.

Usage:
    python scripts/run_benchmark.py [--scenarios all|single|colluding|fragmented]
                                    [--seeds 3]
                                    [--output experiments/benchmark_results.json]

Output:
    - Console: formatted comparison table (mean ± std over seeds)
    - File:    machine-readable JSON (experiments/benchmark_results.json)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai.fl_core.fl_engine import LinearSoftmaxModel, fedavg, local_train, multi_krum
from ai.training.poison import (
    apply_trigger_to_all,
    dirichlet_partition,
    inject_trigger,
    make_dataset,
)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "single_client": {
        "desc": "Single malicious client, BadNets trigger",
        "n_clients": 12,
        "n_rounds": 20,
        "malicious": [2],          # 1/12 = 8%
        "poison_fraction": 0.30,
        "trigger_value": 6.0,
    },
    "colluding_minority": {
        "desc": "Colluding minority (3 clients, individually-mild poisoning)",
        "n_clients": 12,
        "n_rounds": 20,
        "malicious": [2, 5, 9],    # 3/12 = 25%
        "poison_fraction": 0.15,   # mild per-client
        "trigger_value": 6.0,
    },
    "colluding_aggressive": {
        "desc": "Colluding minority (3 clients, aggressive poisoning)",
        "n_clients": 12,
        "n_rounds": 20,
        "malicious": [2, 5, 9],
        "poison_fraction": 0.50,
        "trigger_value": 8.0,
    },
}

STRATEGIES = {
    "fedavg":         "No defense (FedAvg baseline)",
    "multikrum":      "Multi-Krum only",
    "multikrum_guard": "Multi-Krum + L1 Collusion Guard",
}

# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

N_FEATURES = 20
N_CLASSES = 4
N_SAMPLES = 3000
TARGET_CLASS = 0
TRIGGER_BLOCK = slice(0, 3)


def _run_single(strategy: str, scenario: dict, seed: int) -> dict:
    """Run one (strategy, scenario, seed) cell and return metrics dict."""
    malicious = scenario["malicious"]
    n_clients = scenario["n_clients"]
    n_rounds = scenario["n_rounds"]
    poison_fraction = scenario["poison_fraction"]
    trigger_value = scenario["trigger_value"]
    krum_f = len(malicious)
    krum_select = max(1, n_clients - krum_f)

    X, y = make_dataset(N_SAMPLES, N_FEATURES, N_CLASSES, seed=seed)
    split = int(N_SAMPLES * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    X_test_triggered = apply_trigger_to_all(X_test, TRIGGER_BLOCK)

    client_indices = dirichlet_partition(
        len(X_train), n_clients, y_train, N_CLASSES, alpha=0.5, seed=seed + 1
    )

    model_params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
    guard_log: list = []

    t0 = time.perf_counter()
    for rnd in range(n_rounds):
        updates = []
        weights = []
        for cid in range(n_clients):
            idx = client_indices[cid]
            Xc, yc = X_train[idx].copy(), y_train[idx].copy()
            if cid in malicious and len(Xc) > 5:
                Xc, yc, _ = inject_trigger(
                    Xc, yc, TARGET_CLASS, TRIGGER_BLOCK,
                    trigger_value=trigger_value,
                    poison_fraction=poison_fraction,
                    seed=seed * 1000 + rnd * 20 + cid,
                )
            new_p = local_train(model_params, N_FEATURES, N_CLASSES, Xc, yc, epochs=5, lr=0.2)
            updates.append(new_p - model_params)
            weights.append(len(Xc))

        if strategy == "fedavg":
            agg_delta = fedavg(updates, weights)
        else:
            agg_delta, _ = multi_krum(updates, krum_f, krum_select)
            if strategy == "multikrum_guard":
                from ai.detection.update_guard import detect_collusion_clusters
                result = detect_collusion_clusters(
                    updates, agg_delta, sim_threshold=0.85, min_cluster_size=2
                )
                guard_log.append(result["flagged_clusters"])

        model_params = model_params + agg_delta

    elapsed = time.perf_counter() - t0

    m = LinearSoftmaxModel(N_FEATURES, N_CLASSES)
    m.set_params(model_params)
    clean_acc = float((m.predict(X_test) == y_test).mean())
    asr = float((m.predict(X_test_triggered) == TARGET_CLASS).mean())

    guard_detections = sum(
        1 for clusters in guard_log
        if clusters and any(set(cl) & set(malicious) for cl in clusters)
    ) if guard_log else 0

    return {
        "clean_accuracy": clean_acc,
        "attack_success_rate": asr,
        "runtime_seconds": elapsed,
        "guard_correct_detections": guard_detections,
        "guard_total_rounds": n_rounds if strategy == "multikrum_guard" else 0,
    }


def _run_scenario(
    scenario_name: str,
    scenario: dict,
    seeds: list[int],
) -> dict:
    """Run all strategies across all seeds for one scenario."""
    print(f"\n{'─' * 60}")
    print(f"Scenario: {scenario['desc']}")
    print(f"{'─' * 60}")

    results: dict[str, dict] = {}
    for strat_key, strat_desc in STRATEGIES.items():
        runs = []
        for seed in seeds:
            r = _run_single(strat_key, scenario, seed)
            runs.append(r)

        c_acc = [r["clean_accuracy"] for r in runs]
        asrs = [r["attack_success_rate"] for r in runs]
        rts = [r["runtime_seconds"] for r in runs]

        results[strat_key] = {
            "description": strat_desc,
            "clean_accuracy_mean": float(np.mean(c_acc)),
            "clean_accuracy_std": float(np.std(c_acc)),
            "attack_success_rate_mean": float(np.mean(asrs)),
            "attack_success_rate_std": float(np.std(asrs)),
            "runtime_seconds_mean": float(np.mean(rts)),
            "seeds": seeds,
            "runs": runs,
        }

        print(
            f"  {strat_key:<20}  "
            f"C-Acc={np.mean(c_acc):.3f}±{np.std(c_acc):.3f}  "
            f"ASR={np.mean(asrs):.3f}±{np.std(asrs):.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_summary(all_results: dict) -> None:
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    header = f"{'Strategy':<22} {'Scenario':<22} {'C-Acc':>8} {'ASR':>8}"
    print(header)
    print("-" * 70)
    for scen, scen_res in all_results.items():
        for strat, res in scen_res.items():
            print(
                f"  {strat:<20} {scen:<22} "
                f"{res['clean_accuracy_mean']:>7.3f} "
                f"{res['attack_success_rate_mean']:>7.3f}"
            )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL-FL benchmark runner")
    parser.add_argument(
        "--scenarios",
        default="all",
        choices=["all", "single_client", "colluding_minority", "colluding_aggressive"],
        help="Which scenario(s) to run",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument(
        "--output",
        default="experiments/benchmark_results.json",
        help="Output path for machine-readable results",
    )
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    scenarios_to_run = (
        SCENARIOS if args.scenarios == "all" else {args.scenarios: SCENARIOS[args.scenarios]}
    )

    print(f"SENTINEL-FL Benchmark — {len(scenarios_to_run)} scenario(s), {args.seeds} seed(s)")

    all_results: dict[str, dict] = {}
    for scen_name, scen_def in scenarios_to_run.items():
        all_results[scen_name] = _run_scenario(scen_name, scen_def, seeds)

    _print_summary(all_results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
