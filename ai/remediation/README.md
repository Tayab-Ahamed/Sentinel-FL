# ai/remediation/ — L5 Remediation Engine

**Purpose**: the Remediation Engine — designed in `../ARCHITECTURE.md` §7.4 and
**implemented**. It closes the detect→mitigate loop: once L2 confirms a backdoor, this
layer repairs the model and *proves* the repair worked before anything is redeployed.

## Contents

| File | Role |
|---|---|
| `remediation_engine.py` | Orchestrator. Runs the ordered escalation policy, measures ASR / clean accuracy before & after each step, accepts the first step that meets the acceptance criteria, emits a `RemediationReport`, and logs an L5 entry to the Trust Ledger. |
| `rollback.py` | Restore the last clean checkpoint strictly before the suspected infection round via the Model Registry. |
| `unlearning.py` | Targeted unlearning — fine-tune on L2's reversed trigger stamped on clean data with correct labels. |
| `pruning.py` | Fine-pruning — zero the trigger-carrying weight channels (capped), then recover clean accuracy. Inspired by `BackdoorBench/defense/fp.py` (`../RESEARCH.md` §4.1, §4.4). |
| `adapters.py` | `ModelAdapter` protocol + `LinearSoftmaxAdapter`, so strategies are model-agnostic (a torch adapter drops in later). |
| `triggers.py` | Helpers to coerce L2 reversed-trigger representations into vectors, build masks, and stamp triggers onto data. |

## Acceptance criteria

A step is accepted only when **both** hold:
- `asr_after ≤ remediation_asr_threshold` (default `0.2`), and
- `clean_accuracy_after ≥ clean_accuracy_before − remediation_max_clean_accuracy_drop`
  (default drop `0.1`).

If every configured strategy fails, the engine raises `RemediationFailedError` (report
attached) or returns the least-bad candidate with `manual_review_required=True` — it
never silently ships a backdoored model.

## Usage

```python
from ai.remediation import RemediationEngine, LinearSoftmaxAdapter

engine = RemediationEngine.from_config(
    LinearSoftmaxAdapter(n_features, n_classes), config, registry=registry, ledger=ledger,
)
repaired_params, report = engine.remediate(
    params, audit_report, X_clean, y_clean, X_triggered, target_label,
    suspected_infection_round=round_num,
)
```

See `../scripts/run_remediation_demo.py` for a runnable end-to-end example and
`../configs/remediation.yaml` for tunables.

## Attestation (tamper-evident proof of repair)

`attestation.py` turns each repair into a signed, hash-chained **Remediation Attestation
Certificate** — binding the before/after model SHA-256 fingerprints, measured ASR/accuracy,
and the winning strategy, linked to the previous certificate to form an append-only ledger.

```python
from ai.remediation import AttestationLedger, verify_certificate

ledger = AttestationLedger("experiments/attestation_chain.jsonl", secret_key="...")
cert = ledger.append(report, params_before, params_after)
verify_certificate(cert, secret_key="...")   # True; tamper -> False
ledger.verify_chain()                         # verifies the whole chain
```

Pure stdlib (`hashlib`/`hmac`/`json`) — no extra dependencies. Covered by
`tests/test_attestation.py`.

**Dependencies**: `ai/detection/` (consumes `AuditReport`), Model Registry
(`ai/fl_core/model_registry.py`), L4 Trust Ledger (`ai/detection/trust_ledger.py`).
