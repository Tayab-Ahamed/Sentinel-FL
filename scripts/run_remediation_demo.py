"""
scripts/run_remediation_demo.py — end-to-end L5 Remediation proof-of-concept.

Demonstrates the challenge's *Remediation* half:

  1. Train an UNDEFENDED FedAvg global model on the same synthetic, colluding-
     BadNets dataset used by ``run_demo.py`` — producing a strongly backdoored
     model (ASR near 1.0).
  2. Simulate the L2 audit output (the reversed trigger for the target label).
  3. Run the Remediation Engine three ways and record ASR / clean-accuracy
     before and after for each:
       A) rollback   — restore a pre-infection checkpoint
       B) unlearning — targeted fine-tuning on the reversed trigger (no registry)
       C) full escalation policy from configs/remediation.yaml
  4. Write ``experiments/remediation_results.json`` (served by the dashboard's
     remediation panel via GET /api/v1/experiments/demo/remediation).

Run: python3 scripts/run_remediation_demo.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai.fl_core.fl_engine import LinearSoftmaxModel, fedavg, local_train
from ai.fl_core.model_registry import FileModelRegistry
from ai.fl_core.schemas import AuditReport, DetectionResult, ModelMetadata, ReversedTrigger
from ai.remediation import AttestationLedger, LinearSoftmaxAdapter, RemediationEngine
from ai.remediation.triggers import trigger_from_block
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
TRIGGER_VALUE = 6.0
MALICIOUS_CLIENTS = [2, 5, 9]
ROUNDS = 20
INFECTION_ROUND = 8  # malicious clients begin colluding here
SEED = 42


def _train_fedavg(X_train, y_train, client_indices, registry=None):
    """Train an undefended FedAvg model; checkpoint clean rounds to the registry."""
    params = LinearSoftmaxModel(N_FEATURES, N_CLASSES).get_params()
    for rnd in range(ROUNDS):
        updates, weights = [], []
        for cid in range(N_CLIENTS):
            Xc = X_train[client_indices[cid]].copy()
            yc = y_train[client_indices[cid]].copy()
            if cid in MALICIOUS_CLIENTS and rnd >= INFECTION_ROUND and len(Xc) > 5:
                Xc, yc, _ = inject_trigger(
                    Xc, yc, TARGET_CLASS, TRIGGER_BLOCK,
                    trigger_value=TRIGGER_VALUE, poison_fraction=0.5,
                    seed=100 + rnd * 10 + cid,
                )
            new = local_train(params, N_FEATURES, N_CLASSES, Xc, yc, epochs=5, lr=0.2)
            updates.append(new - params)
            weights.append(len(Xc))
        params = params + fedavg(updates, weights)
        # Checkpoint every round BEFORE infection (these are the clean rollback targets).
        if registry is not None and rnd < INFECTION_ROUND:
            registry.save(
                rnd, params,
                ModelMetadata(round_num=rnd, architecture="linear_softmax_v0"),
            )
    return params


def _simulated_audit() -> AuditReport:
    """Stand-in for the L2 Model Auditor output (reversed trigger for TARGET)."""
    vec = trigger_from_block(N_FEATURES, TRIGGER_BLOCK, TRIGGER_VALUE)
    det = DetectionResult(
        detector_name="neural_cleanse_audit", layer="L2", subject_id=str(TARGET_CLASS),
        score=float(np.abs(vec).sum()), flagged=True, boundary=2.0,
        round_num=INFECTION_ROUND + 5, explanation="minimal reversed trigger recovered",
    )
    return AuditReport(
        round_num=INFECTION_ROUND + 5,
        per_label_results=[det],
        flagged_labels=[TARGET_CLASS],
        reversed_triggers=[ReversedTrigger(
            label=TARGET_CLASS, trigger_representation=vec.tolist(),
            l1_norm=float(np.abs(vec).sum()),
        )],
    )


def main():
    X, y = make_dataset(N_SAMPLES, N_FEATURES, N_CLASSES, seed=SEED)
    split = int(N_SAMPLES * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    X_test_trig = apply_trigger_to_all(X_test, TRIGGER_BLOCK, TRIGGER_VALUE)
    client_indices = dirichlet_partition(len(X_train), N_CLIENTS, y_train, N_CLASSES, alpha=0.5, seed=7)

    adapter = LinearSoftmaxAdapter(N_FEATURES, N_CLASSES)
    audit = _simulated_audit()

    reports = []
    remediated: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as td:
        registry = FileModelRegistry(os.path.join(td, "ckpts"))
        poisoned = _train_fedavg(X_train, y_train, client_indices, registry=registry)
        asr0 = float((adapter.predict(poisoned, X_test_trig) == TARGET_CLASS).mean())
        cacc0 = float((adapter.predict(poisoned, X_test) == y_test).mean())
        print(f"[undefended]  clean_acc={cacc0:.3f}  ASR={asr0:.3f}")

        # A) rollback only
        eng_a = RemediationEngine(adapter, registry=registry, asr_threshold=0.3, strategies=("rollback",))
        params_a, rep_a = eng_a.remediate(poisoned, audit, X_test, y_test, X_test_trig, TARGET_CLASS,
                                   suspected_infection_round=INFECTION_ROUND, raise_on_failure=False)
        reports.append(("rollback_only", rep_a))
        remediated["rollback_only"] = params_a

        # B) unlearning only (no registry)
        eng_b = RemediationEngine(adapter, registry=None, asr_threshold=0.3, strategies=("unlearning",),
                                  unlearning_epochs=30, unlearning_lr=0.2)
        params_b, rep_b = eng_b.remediate(poisoned, audit, X_test, y_test, X_test_trig, TARGET_CLASS,
                                   raise_on_failure=False)
        reports.append(("unlearning_only", rep_b))
        remediated["unlearning_only"] = params_b

        # C) full escalation policy
        eng_c = RemediationEngine(adapter, registry=registry, asr_threshold=0.3,
                                  unlearning_epochs=30, unlearning_lr=0.2)
        params_c, rep_c = eng_c.remediate(poisoned, audit, X_test, y_test, X_test_trig, TARGET_CLASS,
                                   suspected_infection_round=INFECTION_ROUND, raise_on_failure=False)
        reports.append(("full_escalation", rep_c))
        remediated["full_escalation"] = params_c

    out = {
        "experiment_id": "demo",
        "undefended": {"clean_accuracy": cacc0, "attack_success_rate": asr0},
        "reports": [dict(scenario=name, **rep.model_dump()) for name, rep in reports],
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "experiments", "remediation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ------------------------------------------------------------------
    # Issue a tamper-evident, hash-chained Remediation Attestation ledger.
    # Every repair is bound to before/after model fingerprints and signed.
    # ------------------------------------------------------------------
    chain_path = os.path.join(os.path.dirname(__file__), "..", "experiments", "attestation_chain.jsonl")
    if os.path.exists(chain_path):
        os.remove(chain_path)
    ledger = AttestationLedger(chain_path, secret_key="sentinel-fl-demo-key")
    print("\nRemediation Attestation Certificates")
    print("-" * 60)
    for name, rep in reports:
        cert = ledger.append(rep, poisoned, remediated[name])
        print(f"  {name:16s} {cert.certificate_id}  "
              f"ASR-Δ={cert.asr_reduction:.3f}  hash={cert.content_hash[:12]}…")
    print(f"  chain length={len(ledger)}  verify_chain={ledger.verify_chain()}")
    print(f"  ledger written to {chain_path}")

    print()
    for name, rep in reports:
        print(f"[{name}] strategy={rep.strategy_succeeded} "
              f"ASR {rep.asr_before:.3f}->{rep.asr_after:.3f} "
              f"C-Acc {rep.clean_accuracy_before:.3f}->{rep.clean_accuracy_after:.3f} "
              f"success={rep.success}")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
