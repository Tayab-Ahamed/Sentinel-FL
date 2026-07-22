"""
scripts/run_demo.py — end-to-end proof-of-concept.

Runs three federated training configurations on the same synthetic, poisoned,
non-IID dataset and reports Clean Accuracy / Attack Success Rate for each, then runs
the L3 Runtime Sentinel (STRIP) on the resulting Multi-Krum+CollusionGuard model.

Configs:
  A) fedavg           — no defense (baseline, expected to be highly vulnerable)
  B) multikrum         — Multi-Krum only (baseline robust aggregation)
  C) multikrum+guard   — Multi-Krum + L1 collusion clustering (this project's addition)

Attack: 3 of 12 clients collude, each poisoning a MODEST fraction (15%) of their local
data with the same trigger+target-label — individually mild enough to often survive
Multi-Krum's per-client filtering, which is exactly the fragmented-collusion gap this
project targets (see RESEARCH.md §3, ARCHITECTURE.md §2.1).

Run: python3 scripts/run_demo.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai.detection.runtime_sentinel import calibrate_boundary, detect
from ai.detection.update_guard import detect_collusion_clusters
from ai.fl_core.fl_engine import LinearSoftmaxModel, fedavg, local_train, multi_krum
from ai.training.poison import (
    apply_trigger_to_all,
    dirichlet_partition,
    inject_trigger,
    make_dataset,
)

N_FEATURES = 20
N_CLASSES = 4
N_CLIENTS = 12
N_SAMPLES = 3000
TARGET_CLASS = 0
TRIGGER_BLOCK = slice(0, 3)
MALICIOUS_CLIENTS = [2, 5, 9]  # colluding, same trigger + target
ROUNDS = 20
ASSUMED_MALICIOUS_F = 3  # Multi-Krum's f parameter
KRUM_SELECT = N_CLIENTS - ASSUMED_MALICIOUS_F  # how many updates Multi-Krum averages


def run_round(strategy: str, model_params: np.ndarray, client_data, guard_log=None):
    client_updates = []
    weights = []
    for _cid, (X, y) in enumerate(client_data):
        new_params = local_train(model_params, N_FEATURES, N_CLASSES, X, y,
                                  epochs=5, lr=0.2)
        client_updates.append(new_params - model_params)
        weights.append(len(X))

    if strategy == "fedavg":
        agg_delta = fedavg(client_updates, weights)
        selected = list(range(N_CLIENTS))
    elif strategy in ("multikrum", "multikrum+guard"):
        agg_delta, selected = multi_krum(client_updates, ASSUMED_MALICIOUS_F, KRUM_SELECT)
        if strategy == "multikrum+guard" and guard_log is not None:
            result = detect_collusion_clusters(client_updates, agg_delta,
                                                sim_threshold=0.85, min_cluster_size=2)
            guard_log.append(result["flagged_clusters"])
    else:
        raise ValueError(strategy)

    return model_params + agg_delta, selected


def evaluate(model_params, X_test, y_test, X_test_triggered):
    model = LinearSoftmaxModel(N_FEATURES, N_CLASSES)
    model.set_params(model_params)
    clean_acc = float((model.predict(X_test) == y_test).mean())
    triggered_preds = model.predict(X_test_triggered)
    asr = float((triggered_preds == TARGET_CLASS).mean())
    return clean_acc, asr, model


def main():
    X, y = make_dataset(N_SAMPLES, N_FEATURES, N_CLASSES, seed=42)
    split = int(N_SAMPLES * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    X_test_triggered = apply_trigger_to_all(X_test, TRIGGER_BLOCK)

    client_indices = dirichlet_partition(len(X_train), N_CLIENTS, y_train, N_CLASSES,
                                          alpha=0.5, seed=7)

    results = {}
    for strategy in ["fedavg", "multikrum", "multikrum+guard"]:
        model_params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
        guard_log = [] if strategy == "multikrum+guard" else None

        for rnd in range(ROUNDS):
            client_data = []
            for cid in range(N_CLIENTS):
                idx = client_indices[cid]
                Xc, yc = X_train[idx].copy(), y_train[idx].copy()
                if cid in MALICIOUS_CLIENTS and len(Xc) > 5:
                    Xc, yc, _ = inject_trigger(Xc, yc, TARGET_CLASS, TRIGGER_BLOCK,
                                                trigger_value=6.0, poison_fraction=0.15,
                                                seed=100 + rnd * 10 + cid)
                client_data.append((Xc, yc))
            model_params, selected = run_round(strategy, model_params, client_data,
                                                guard_log)

        clean_acc, asr, final_model = evaluate(model_params, X_test, y_test,
                                                 X_test_triggered)
        entry = {"clean_accuracy": clean_acc, "attack_success_rate": asr}

        if strategy == "multikrum+guard":
            flags = [c for c in guard_log if c]
            correctly_flagged_rounds = sum(
                1 for clusters in flags
                if any(set(cl) & set(MALICIOUS_CLIENTS) for cl in clusters)
            )
            entry["rounds_with_any_cluster_flag"] = len(flags)
            entry["rounds_correctly_flagging_malicious_cluster"] = correctly_flagged_rounds
            entry["total_rounds"] = ROUNDS

            # L3 Runtime Sentinel on the final defended model
            clean_pool = X_train[:200]
            boundary = calibrate_boundary(final_model, clean_pool, X_test[:100],
                                           target_frr=0.02, n_perturb=30)
            flagged_clean, scores_clean = detect(final_model, X_test[:100], clean_pool,
                                                  boundary, n_perturb=30)
            flagged_trig, scores_trig = detect(final_model, X_test_triggered[:100],
                                                clean_pool, boundary, n_perturb=30)
            entry["strip_frr_on_clean"] = float(flagged_clean.mean())
            entry["strip_detection_rate_on_triggered"] = float(flagged_trig.mean())
            entry["strip_boundary"] = boundary
            entry["strip_mean_entropy_clean"] = float(np.mean(scores_clean))
            entry["strip_mean_entropy_triggered"] = float(np.mean(scores_trig))

        results[strategy] = entry
        print(f"[{strategy}] clean_acc={clean_acc:.3f}  ASR={asr:.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "experiments",
                             "demo_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
