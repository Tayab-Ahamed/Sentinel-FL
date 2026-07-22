# BENCHMARK.md — Evaluation Plan

Cross-references: metric definitions live in `SCHEMAS.md#EvaluationResult`; the
Phase-0 sanity numbers already produced are in `RESULTS.md` — this document is the plan
for the *real* benchmark run in Phase 1–2 (`IMPLEMENTATION_PLAN.md`), not a repeat of
already-reported results.

## 1. Baselines Compared

| Baseline | Role in comparison | Source |
|---|---|---|
| No defense (plain FedAvg) | Lower bound | `ai/fl_core/fl_engine.py` |
| BadNets (attack, not defense) | The attack under test, not a comparison point — listed because it defines the ASR the other rows are measured against | `RESEARCH.md` §1.2 background |
| Krum / Multi-Krum | Aggregation-level baseline | Flower's `serverapp.strategy` (`RESEARCH.md` §4.2) |
| Trimmed Mean | Aggregation-level baseline | Flower's `FedTrimmedAvg` |
| Median (coordinate-wise) | Aggregation-level baseline | FedML `defense/` (`RESEARCH.md` §4.3) |
| FoolsGold | Collusion-specific baseline — the closest prior art to L1's novel piece | FedML `foolsgold_defense.py` |
| STRIP | Runtime detection baseline | Own implementation, cross-checked against `BackdoorBench/detection_infer/strip.py` |
| Neural Cleanse | Offline audit baseline | `BackdoorBench/defense/nc.py` |
| Fine-Pruning | Mitigation baseline | `BackdoorBench/defense/fp.py` |
| **Sentinel-FL (full)** | This project, L1+L2+L3+L4 | `ARCHITECTURE.md` |
| **Sentinel-FL (ablated)** | Each layer toggled off individually | isolates marginal contribution |

## 2. Datasets

- **Phase 0**: synthetic Gaussian-blob data (`DATASETS.md`) — sanity checks only,
  already run (`RESULTS.md`); not part of the final benchmark table.
- **Phase 1**: official GSC26 Challenge 1 dataset, once released.
- **Phase 2 (stretch, if time allows)**: CIFAR-10 federated partitions, for
  comparability with the exact datasets STRIP/Neural Cleanse/BackdoorBench report
  numbers on, strengthening the "how does this compare to published baselines"
  argument even if the official dataset differs.

## 3. Attack Scenarios

1. **Single malicious client, BadNets trigger** — the case every baseline already
   handles well; expected result is parity across all rows, establishing that
   Sentinel-FL doesn't regress the easy case.
2. **Colluding minority (3+ clients, shared trigger, individually-mild poisoning
   fraction)** — the headline scenario (`RESULTS.md` Phase 0 version). Expected
   differentiation: Multi-Krum alone partial, FoolsGold better, Sentinel-FL best or
   comparable to FoolsGold with an explainability advantage.
3. **Source-label-specific (partial) backdoor** — targets STRIP's documented blind
   spot; expected: STRIP-alone misses it, Sentinel-FL's L2 audit (Neural-Cleanse-style)
   catches it, demonstrating the cross-layer value proposition.
4. **Large/highly visible trigger** — targets Neural Cleanse's documented blind spot;
   expected: L2 misses it, L3 (STRIP-style) catches it — the complementary-blind-spot
   argument from `NOVELTY.md` §2.3, tested rather than asserted.
5. **Entropy-manipulation adaptive attack** — targets STRIP's documented weakness
   directly; expected: L3's Signal 1 (entropy) alone fails, Signal 2 (activation
   consistency) still catches it.

## 4. Metrics (full definitions in `SCHEMAS.md#EvaluationResult`)

- Clean Accuracy (C-Acc)
- Attack Success Rate (ASR)
- Robust Accuracy (R-Acc)
- False Acceptance Rate / False Rejection Rate (L3)
- Detection latency (ms/input for L3; rounds-to-detect for L1/L2)
- Communication cost (bytes transferred per round — Sentinel-FL adds L1 residual-
  similarity computation server-side only, so this should be ~identical to plain
  Multi-Krum; verifying this is itself a result worth reporting)
- Memory (peak server-side RAM during L1 pairwise-similarity computation, which is
  O(n_clients²) — worth reporting explicitly since it's the one place this design adds
  overhead over vanilla Multi-Krum)
- Scalability: wall-clock time for L1/L2/L3 as `n_clients` and `n_labels` scale up,
  since Neural-Cleanse-style audits are documented to scale with label count
  (`RESEARCH.md` §1.2) — reporting this honestly, not hiding the cost.

## 5. Reporting Format

One results table per attack scenario (§3), rows = baselines (§1), columns = metrics
(§4), generated from `MetricsCollector.compute(...)` (`INTERFACES.md`) into
`experiments/<scenario>/results.json`, rendered by the dashboard's metric-timeseries
and reputation-heatmap views (`API.md` §4–5) for the live presentation described in
`NOVELTY.md` §6.

## 6. Statistical Rigor

- Each (attack scenario × baseline) cell run with **at least 3 random seeds**;
  report mean ± std, not a single run — the Phase 0 results in `RESULTS.md` are
  explicitly single-seed sanity checks and are not held to this bar.
- Non-IID skew (Dirichlet α) swept across at least 2 settings, since BackdoorBench's
  own findings show defense effectiveness is architecture- and setup-dependent
  (`RESEARCH.md` §1.3) — the same caution applies to non-IID skew here, and is worth
  demonstrating rather than assuming a single setting generalizes.
