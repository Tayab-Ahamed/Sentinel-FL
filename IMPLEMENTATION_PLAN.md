# IMPLEMENTATION_PLAN.md

Scoped against the real GSC26 calendar: challenge opens **13 July**, baseline-activity
deadline **28 July**, close **9 August**. ~4 weeks total, solo effort.

## Phase 0 — Now through 13 July (prep, no official data yet)
- [ ] Stand up FL simulator using **Flower** (`flwrlabs/flower`) with a synthetic
      partition of CIFAR-10 across 10–20 simulated clients (non-IID via Dirichlet split).
- [ ] Implement BadNets-style square-trigger poisoning on a subset of simulated clients
      as the baseline attack to develop against.
- [ ] Implement L1 Update Guard: Multi-Krum baseline + collusion clustering module.
- [ ] Implement L3 Runtime Sentinel: STRIP-style entropy detector as baseline, then add
      the activation-consistency second signal.
- [ ] Get the evaluation pipeline (C-Acc, ASR, R-Acc, FAR/FRR) running end-to-end on
      synthetic data so it's a drop-in once real data lands.

## Phase 1 — 13 July to ~18 July (real data onboarding)
- [ ] Swap synthetic partitions for the official challenge dataset/spec.
- [ ] Re-validate L1/L3 against the real attack variants specified in the official
      problem statement (adjust if the official threat model differs from BadNets).
- [ ] Implement L2 Model Auditor (Neural-Cleanse-style reverse engineering +
      within-cohort relative outlier test).

## Phase 2 — ~18 July to 28 July (baseline activity deadline)
- [ ] Full ablation matrix: each layer on/off, isolating marginal novelty contribution.
- [ ] Hit whatever "baseline activity" the 13 July email defines, to secure the second
      token allotment.
- [ ] Dashboard v1: static JSON-driven views for reputation heatmap, ASR/C-Acc charts,
      audit report viewer.

## Phase 3 — 28 July to 9 August (polish + submission)
- [ ] Fragmented-trigger collusion attack + defense demo — the headline novelty result.
- [ ] Write up quantitative results vs. STRIP-only and Neural-Cleanse-only baselines.
- [ ] Record presentation / prepare oral defense materials.
- [ ] Final submission, repo cleanup, README pass.

## Week-by-Week Milestones (maps onto Phases 0–3 above; this is the literal weekly
breakdown requested for Antigravity execution — each week's tasks are the T-numbers
from the backlog below)

### Week 1 (through ~16 July — spans Phase 0 prep + Phase 1 onboarding)
- **Deliverables**: T1–T4, T6 (FL simulator, partitioner, BadNets injector, Multi-Krum
  integration, STRIP baseline) + real dataset swapped in once released 13 July.
- **Dependencies**: none blocking Phase 0 items; dataset-dependent items block on the
  13 July 3pm ET release.
- **Testing**: unit tests per `TESTING.md` §2 for every module landed this week;
  `scripts/run_demo.py`-equivalent integration test passing on synthetic data by end
  of week (already true — see `RESULTS.md`).
- **Success criteria**: FedAvg/Multi-Krum comparison reproduces the qualitative
  Phase-0 result (Multi-Krum meaningfully reduces ASR vs. FedAvg) on whatever dataset
  is active.

### Week 2 (~17–23 July)
- **Deliverables**: T5, T7, T8 (collusion clustering, activation-consistency signal,
  L3 fusion classifier), plus FoolsGold comparison baseline wired in
  (`BENCHMARK.md` §1).
- **Dependencies**: T5 depends on T4; T8 depends on T6+T7.
- **Testing**: `detect_collusion_clusters()` unit tests (`TESTING.md` §2) plus a new
  integration test for the fragmented-collusion scenario (`BENCHMARK.md` §3.2).
- **Success criteria**: fragmented-collusion scenario shows a measurable ASR or
  detection-rate improvement for Sentinel-FL's L1 over Multi-Krum-alone, OR a
  documented, honest negative result explaining why not (per `RESULTS.md`'s standard
  of reporting real numbers either way).

### Week 3 (~24–30 July, spans the 28 July baseline-activity deadline)
- **Deliverables**: T9, T10 (Neural-Cleanse-style L2 audit + within-cohort relative
  outlier test), T11 (Trust Ledger), Remediation Engine v1
  (`ARCHITECTURE.md` §7.4). **Delivered beyond scope**: all three mitigation paths
  shipped — rollback **and** targeted unlearning **and** fine-pruning — behind a
  self-verifying escalation policy with `manual_review_required` fallback, a REST API,
  a runnable demo, and a dedicated test suite (`ai/remediation/`,
  `tests/test_remediation.py`).
- **Dependencies**: T11 depends on T5+T8+T10 all emitting `DetectionResult`s.
- **Testing**: L2 audit unit tests against a known-infected vs. known-clean toy model;
  Remediation Engine integration test (rollback reduces ASR back to pre-attack
  baseline on a synthetic known-round infection).
- **Success criteria**: whatever the official "baseline activity" requirement turns out
  to be (per the 13 July email), satisfied and confirmed for the bonus token
  allotment.

### Week 4 (~31 July–9 August, submission)
- **Deliverables**: T12–T15 (evaluation pipeline polish, dashboard v1, full ablation
  matrix per `BENCHMARK.md`, writeup + oral defense materials).
- **Dependencies**: T13 (dashboard) depends on T11+T12; T14 (ablation) depends on
  everything above being stable.
- **Testing**: full `BENCHMARK.md` matrix run at least once end-to-end before the
  submission deadline, with results checked into `experiments/`.
- **Success criteria**: submission package complete — repo, results, presentation
  materials — by 9 August 11:59pm ET.

## Task Backlog

| ID | Task | Depends on | Owner |
|---|---|---|---|
| T1 | Flower FL simulator scaffold | — | you |
| T2 | Dirichlet non-IID partitioner | T1 | you |
| T3 | BadNets trigger injector | T2 | you |
| T4 | Multi-Krum aggregation | T1 | you |
| T5 | Collusion clustering (residual cosine graph) | T4 | you |
| T6 | STRIP entropy detector | — | you |
| T7 | Activation-consistency detector | T6 | you |
| T8 | Fusion classifier (R3) | T6, T7 | you |
| T9 | Neural-Cleanse-style reverse engineering | — | you |
| T10 | Within-cohort relative outlier test | T9 | you |
| T11 | Trust Ledger schema + storage | T5, T8, T10 | you |
| T12 | Evaluation pipeline (metrics) | T3 | you |
| T13 | Dashboard (static JSON-driven) | T11, T12 | you |
| T14 | Ablation experiments | T4–T13 | you |
| T15 | Writeup + oral defense prep | T14 | you |

## Compute Budget Notes
- L2 audits are the expensive step (per-label optimization). Run every N=5–10 rounds,
  not every round, and use the low-cost early-termination heuristic from Neural Cleanse
  for anything beyond ~10 labels.
- Everything else (L1, L3) is cheap enough to run every round / every inference on a
  single GPU or even CPU for CIFAR-10-scale models.
