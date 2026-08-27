# SENTINEL-FL Model & System Card

## System identity

| Field | Value |
|---|---|
| Name | SENTINEL-FL — Federated Backdoor Immune System |
| Release | 0.2.0 Open-Source Research & Benchmark Release |
| Primary task | Detect, explain, remediate, and attest backdoor attacks in distributed/federated ML |
| Supported backends | NumPy linear-softmax reference; PyTorch CNN via `TorchModelAdapter` |
| License | MIT |

## Intended use

SENTINEL-FL is a research and engineering framework for studying **targeted backdoor
attacks in federated learning**, especially attacks coordinated across multiple clients.
It provides five defensive layers, an escalating remediation engine, and tamper-evident
repair attestations.

It is intended for:

- reproducible security research and defensive benchmarking;
- offline analysis of federated client updates and global models;
- demonstrating detect → explain → repair → verify workflows;
- extension to custom and benchmark datasets.

It is **not** a drop-in safety guarantee for clinical, financial, critical-infrastructure,
or other high-impact production decisions without independent validation.

## Inputs and outputs

- **Inputs:** client model deltas, clean calibration/evaluation data, triggered evaluation
  data, optional recovered trigger, model checkpoints.
- **Outputs:** robust aggregate, client trust scores, alerts, recovered trigger evidence,
  remediation report, repaired model parameters, signed attestation certificate.
- **Privacy note:** the current prototype assumes a server-side clean calibration holdout.
  Raw client examples are not required by the L1 update-analysis layer.

## Performance evidence

### Deterministic reference demo (seed 42)

- Undefended model reaches high source-only ASR under colluding BadNets poisoning.
- Multi-Krum lowers ASR while retaining clean accuracy.
- L5 rollback and targeted unlearning drive source-only ASR from **1.000 to 0.000** with
  clean accuracy retained at **1.000** in the reference remediation run.

### Adaptive red-team matrix

The committed full matrix varies malicious-client count (`1`, `3`, `5` of `12`), local
poison fraction (`8%`, `20%`), trigger strength (`4`, `7`), and seed (`7`, `42`):

- 24 deterministic scenarios;
- worst undefended source-only ASR: **0.993**;
- worst Multi-Krum + Guard source-only ASR: **0.942** under extreme collusion;
- worst post-L5 source-only ASR: **0.000**;
- post-L5 remediation acceptance: **24/24 (100%)**;
- clean accuracy after remediation: **1.000** on this synthetic reference task.

See `experiments/red_team/RED_TEAM_REPORT.md`. These are Phase-0 synthetic results—not a
claim of identical performance on the official image dataset.

## Metric definition

**Attack Success Rate is source-only:** among triggered inputs whose true clean label is
not already the target class, the fraction predicted as the attacker’s target. This avoids
the artificial target-class-prevalence floor created by evaluating target-class examples.

A remediation is accepted only when both conditions hold:

1. ASR is at or below the configured threshold; and
2. clean accuracy drops by no more than the configured maximum.

Otherwise, the engine sets `manual_review_required` and does not silently approve the model.

## Limitations and failure modes

- Current baseline evidence uses synthetic Gaussian features and a linear model; the
  PyTorch path has unit coverage and is ready for benchmark image datasets (e.g. CIFAR-10, MNIST).
- Multi-Krum degrades at high Byzantine fractions; the red-team matrix intentionally shows
  a `0.942` worst defended ASR before L5 rather than hiding it.
- The L1 collusion detector requires at least two coordinated clients; a lone attacker is
  handled by robust aggregation/model auditing rather than “collusion” clustering.
- Trigger unlearning depends on L2 recovering a sufficiently representative trigger.
- Adaptive, clean-label, semantic, dynamic, and distributed-trigger attacks require further
  official-dataset evaluation.
- HMAC attestation proves integrity/authenticity under a protected operator key; it does not
  by itself prove the evaluation dataset was unbiased.

## Human oversight

Manual review is mandatory when all remediation strategies fail, clean utility falls beyond
tolerance, trigger evidence is absent/ambiguous, or attestation verification fails.

## Reproducibility

```bash
python scripts/run_demo.py
python scripts/run_remediation_demo.py
python scripts/run_red_team_matrix.py
python scripts/generate_charts.py
python scripts/verify_release.py
```

All reference stochastic paths use explicit seeds. Release evidence and SHA-256 artifact
hashes are written to `experiments/release_evidence.json`.
