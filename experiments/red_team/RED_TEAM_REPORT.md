# SENTINEL-FL Adaptive Red-Team Report

**Generated:** 2026-07-25T17:05:16.476319+00:00

**Matrix:** 24 deterministic scenarios · 12 clients · 8 rounds each

**Threat dimensions:** malicious-client count × poison fraction × trigger strength × seed

> This is not a cherry-picked benchmark. It intentionally varies attacker power and reports
> means, 95% bootstrap confidence intervals, and the worst observed case.

## Executive result

| Metric | Mean | 95% CI of mean | Worst observed |
|---|---:|---:|---:|
| Undefended FedAvg ASR | 0.473 | [0.331, 0.614] | 0.993 |
| Multi-Krum + Guard ASR | 0.107 | [0.032, 0.204] | 0.942 |
| **After L5 remediation ASR** | **0.000** | **[0.000, 0.000]** | **0.000** |
| Clean accuracy after L5 | 1.000 | [1.000, 1.000] | 1.000 |

- **Remediation acceptance rate:** 100.0%
- **Mean collusion-detection rate:** 47.9%
- **Worst post-remediation scenario:** `m1-p08-t4-s7`

![Adaptive threat heatmap](red_team_heatmap.png)

## Per-scenario evidence

| Scenario | Malicious | Poison | Trigger | FedAvg ASR | Defended ASR | L5 ASR | Clean acc | Guard detect | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `m1-p08-t4-s7` | 1/12 | 8% | 4 | 0.000 | 0.000 | **0.000** | 1.000 | 0% | ✅ |
| `m1-p08-t4-s42` | 1/12 | 8% | 4 | 0.265 | 0.000 | **0.000** | 1.000 | 0% | ✅ |
| `m1-p08-t7-s7` | 1/12 | 8% | 7 | 0.004 | 0.004 | **0.000** | 1.000 | 0% | ✅ |
| `m1-p08-t7-s42` | 1/12 | 8% | 7 | 0.576 | 0.000 | **0.000** | 1.000 | 0% | ✅ |
| `m1-p20-t4-s7` | 1/12 | 20% | 4 | 0.000 | 0.000 | **0.000** | 1.000 | 0% | ✅ |
| `m1-p20-t4-s42` | 1/12 | 20% | 4 | 0.537 | 0.000 | **0.000** | 1.000 | 0% | ✅ |
| `m1-p20-t7-s7` | 1/12 | 20% | 7 | 0.033 | 0.004 | **0.000** | 1.000 | 0% | ✅ |
| `m1-p20-t7-s42` | 1/12 | 20% | 7 | 0.829 | 0.000 | **0.000** | 1.000 | 0% | ✅ |
| `m3-p08-t4-s7` | 3/12 | 8% | 4 | 0.000 | 0.000 | **0.000** | 1.000 | 38% | ✅ |
| `m3-p08-t4-s42` | 3/12 | 8% | 4 | 0.463 | 0.000 | **0.000** | 1.000 | 62% | ✅ |
| `m3-p08-t7-s7` | 3/12 | 8% | 7 | 0.015 | 0.015 | **0.000** | 1.000 | 75% | ✅ |
| `m3-p08-t7-s42` | 3/12 | 8% | 7 | 0.763 | 0.023 | **0.000** | 1.000 | 88% | ✅ |
| `m3-p20-t4-s7` | 3/12 | 20% | 4 | 0.004 | 0.000 | **0.000** | 1.000 | 38% | ✅ |
| `m3-p20-t4-s42` | 3/12 | 20% | 4 | 0.700 | 0.000 | **0.000** | 1.000 | 75% | ✅ |
| `m3-p20-t7-s7` | 3/12 | 20% | 7 | 0.189 | 0.087 | **0.000** | 1.000 | 75% | ✅ |
| `m3-p20-t7-s42` | 3/12 | 20% | 7 | 0.946 | 0.089 | **0.000** | 1.000 | 88% | ✅ |
| `m5-p08-t4-s7` | 5/12 | 8% | 4 | 0.385 | 0.007 | **0.000** | 1.000 | 88% | ✅ |
| `m5-p08-t4-s42` | 5/12 | 8% | 4 | 0.494 | 0.019 | **0.000** | 1.000 | 62% | ✅ |
| `m5-p08-t7-s7` | 5/12 | 8% | 7 | 0.858 | 0.265 | **0.000** | 1.000 | 88% | ✅ |
| `m5-p08-t7-s42` | 5/12 | 8% | 7 | 0.802 | 0.093 | **0.000** | 1.000 | 88% | ✅ |
| `m5-p20-t4-s7` | 5/12 | 20% | 4 | 0.771 | 0.280 | **0.000** | 1.000 | 88% | ✅ |
| `m5-p20-t4-s42` | 5/12 | 20% | 4 | 0.747 | 0.179 | **0.000** | 1.000 | 62% | ✅ |
| `m5-p20-t7-s7` | 5/12 | 20% | 7 | 0.993 | 0.560 | **0.000** | 1.000 | 88% | ✅ |
| `m5-p20-t7-s42` | 5/12 | 20% | 7 | 0.973 | 0.942 | **0.000** | 1.000 | 50% | ✅ |


## Methodology and limitations

- Synthetic Gaussian-blob Phase-0 data, deterministic seeds; no claims are made that these
  numbers transfer unchanged to the official image dataset.
- Multi-Krum's assumed Byzantine count equals the actual malicious count (capped by its
  mathematical client-count constraint).
- L5 receives the recovered trigger representation, as it would from L2 Model Auditor.
- ASR is **source-only**: samples whose clean label is already the target class are excluded.
- Acceptance requires source-only ASR ≤ 0.10 and clean-accuracy drop ≤ 0.10.
- Run the PyTorch/official-dataset benchmark before final competition submission; this matrix
  is a regression and systems-evidence suite, not a replacement for official evaluation.

Reproduce with `python scripts/run_red_team_matrix.py`.
