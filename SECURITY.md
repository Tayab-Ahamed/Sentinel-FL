# Security Policy & Threat Model

## Reporting a vulnerability

Please report security issues privately to the Team 010 repository maintainers rather than
opening a public issue containing exploit details. Include the affected component, minimal
reproduction, impact, and any proposed mitigation. Do not include real user/client data.

## Threat model

### Protected assets

- integrity and utility of the global federated model;
- confidentiality of operator signing keys and checkpoints;
- integrity of trust-ledger, experiment, and remediation evidence;
- availability of the aggregation/remediation service.

### Adversary capabilities considered

- controls a subset of federated clients;
- modifies local data, labels, and model updates;
- coordinates updates across colluding clients;
- uses a targeted BadNets-style trigger;
- varies poison rate and trigger strength;
- attempts to preserve clean accuracy while maximizing source-only ASR.

### Out of scope / future validation

- compromise of the central server or CI administrator;
- theft of the HMAC attestation secret;
- arbitrary code execution through malicious serialized model formats;
- poisoning of the trusted clean calibration set;
- privacy attacks such as gradient inversion or membership inference;
- denial-of-service from unbounded client payloads;
- formal Byzantine tolerance beyond each aggregator's mathematical assumptions.

## Security controls

| Control | Purpose |
|---|---|
| L1 residual/collusion analysis | Detect coordinated update geometry |
| Multi-Krum | Limit outlier influence during aggregation |
| L2 trigger reconstruction | Confirm and explain model-level backdoors |
| L4 Trust Ledger | Preserve cross-layer evidence and reputation |
| L5 remediation | Roll back, unlearn, or fine-prune compromised models |
| Dual acceptance gate | Require low ASR and retained clean accuracy |
| Manual-review fail-safe | Refuse silent redeployment on failed repair |
| HMAC hash-chain attestation | Detect altered remediation evidence |
| Model fingerprints | Bind certificates to before/after parameters |
| Adaptive red-team CI gate | Prevent regressions across attacker powers |
| Release evidence manifest | Hash and verify competition artifacts |

## Secure deployment guidance

1. Never use the demo key in production. Load the HMAC secret from a protected secret manager.
2. Run the API behind authenticated TLS; do not expose experiment mutation endpoints publicly.
3. Store model checkpoints and ledgers in append-only/versioned storage with least privilege.
4. Validate tensor counts, shapes, dtypes, finite values, and payload sizes before aggregation.
5. Treat PyTorch pickle checkpoints as trusted-only; prefer parameter-only formats.
6. Keep a physically/logically independent clean calibration set.
7. Verify `experiments/release_evidence.json` and the attestation chain before deployment.
8. Trigger human review whenever the acceptance gate or chain verification fails.

## Automated security regression

```bash
python scripts/run_red_team_matrix.py --quick
python scripts/verify_release.py
```

CI requires the quick adaptive matrix to achieve 100% remediation acceptance and worst
post-remediation source-only ASR ≤ 0.10. The full committed matrix covers 24 scenarios.

## Honest security posture

SENTINEL-FL is defense-in-depth, not a proof of immunity. The full red-team matrix includes
an extreme scenario where Multi-Krum + Guard still reaches ASR `0.942`; the system remains
safe in that experiment because L5 repairs the model to `0.000`. This result is disclosed to
show why lifecycle remediation is necessary—not hidden to make aggregation look perfect.
