# RESEARCH.md — Literature Analysis & Gap Identification

Challenge 1: **Backdoor Attacks in Distributed Machine Learning and their Remediation**
IEEE Computer Society Global Student Challenge 2026, Team 010.

This document analyzes the three core references provided, then derives the research
gaps that motivate the original system in `ARCHITECTURE.md`. It intentionally does not
restate methods verbatim — see each paper for full derivations and results tables.

---

## 1. Paper Analysis

### 1.1 STRIP (Gao et al., ACSAC 2019)

**Problem.** Detecting whether a *deployed* model is trojaned, and whether a specific
*incoming input* carries a trigger, without access to poisoned samples.

**Core idea.** Input-agnostic triggers force any perturbed version of a trojaned input
toward the same target class. STRIP linearly blends each incoming input with several
held-out clean images and measures the Shannon entropy of the resulting prediction
distribution. Low entropy (predictions collapse to one class regardless of perturbation)
flags the input as trojaned; high entropy indicates a benign input.

**Strengths.**
- Black-box, run-time, architecture-agnostic — only needs input/output access.
- Insensitive to trigger size (unlike optimization-based reverse engineering), so it
  catches large/visible triggers that defeat Neural Cleanse-style methods.
- Very cheap: a handful of blended forward passes per input (single-digit milliseconds
  reported for lightweight models).

**Limitations (author-acknowledged and derived).**
- Assumes a single dominant, input-agnostic trigger. Source-label-specific (partial)
  backdoors — where the trigger only fires for a subset of source classes — produce
  entropy distributions that overlap with clean inputs and evade detection unless the
  defender already has trojaned samples to calibrate against.
- An adaptive "entropy manipulation" attacker can train the model so poisoned samples
  themselves produce high-entropy perturbed predictions, closing the entropy gap. The
  paper's own countermeasure (flagging an abnormally-high-variance *clean* entropy
  distribution) is a secondary signal, not a primary detector.
- Says nothing about *distributed/federated* training — it defends a single already-
  trained model at inference time, with no notion of multiple contributors.
- Detection boundary calibration assumes clean inputs are normally distributed and
  available in quantity; in a federated client with very little local data this is shaky.

### 1.2 Neural Cleanse (Wang et al., IEEE S&P 2019)

**Problem.** Given a trained model, decide if it is backdoored, identify the target
label, and reverse-engineer an approximation of the trigger — all offline, before
deployment.

**Core idea.** For each candidate target label, solve an optimization problem for the
minimal patch (mask + pattern) that forces misclassification of arbitrary clean inputs
into that label. A genuinely backdoored label needs a dramatically smaller patch (L1
norm) than any legitimate label. Outlier detection (MAD-based anomaly index) over the
per-label L1 norms flags the backdoored label. The reversed trigger then feeds two
mitigation paths: neuron pruning at the most trigger-responsive layer, or fine-tuning
("unlearning") on data stamped with the reversed trigger but correct labels.

**Strengths.**
- Doesn't need any trojaned samples — works from a clean validation set alone.
- Provides an actual mitigation path (pruning/unlearning), not just detection.
- Scales to large label counts via an early-termination heuristic that prunes the
  candidate-label search after a handful of optimization iterations.

**Limitations (author-acknowledged and derived).**
- Cost scales with the number of labels; full per-label optimization is expensive
  without the low-cost heuristic.
- Detection degrades once trigger size grows large enough that the reversed-trigger L1
  norm blends into the uninfected-label distribution — a large enough trigger evades
  outlier detection even though STRIP would still catch it.
- Effectiveness drops sharply once many labels are simultaneously backdoored (the
  method's own experiments show detection failing once roughly a third of labels are
  infected), because the anomaly signal is relative to other labels.
- Like STRIP, this is a single-model, centralized-training assumption. It doesn't model
  an attacker who controls a small fraction of federated clients and only needs to bias
  the aggregated global model, not a single fully-owned training pipeline.
- Reverse-engineered triggers only approximate the original (different position/shape
  for optimization-based attacks such as the Trojan Attack method), which caps
  mitigation effectiveness against attacks that don't resemble simple patch triggers.

### 1.3 BackdoorBench (Wu et al., NeurIPS 2022 Datasets & Benchmarks)

**Problem.** The backdoor learning literature had no standardized, reproducible way to
compare attacks against defenses, leading to inflated or non-comparable claims.

**Core contribution.** A modular codebase (attack module, defense module, evaluation
module) implementing 8 attacks × 9 defenses, evaluated across 5 poisoning ratios, 5
model architectures, and 4 datasets — about 8,000 attack/defense pairs — with metrics
clean accuracy (C-Acc), attack success rate (ASR), and robust accuracy (R-Acc).

**Key empirical findings relevant to this project.**
- Higher poisoning ratio does not monotonically help the attacker: several defenses
  (fine-tuning-family methods, ANP) become *more* effective at higher poisoning ratios
  because the poisoned/clean signal separates more cleanly — a counter-intuitive result
  that argues for stealthy, low-ratio attacks being the harder defense case.
- Defense effectiveness is highly architecture-dependent: a defense that neutralizes a
  backdoor in one backbone (e.g., PreAct-ResNet18) can fail on another (e.g.,
  EfficientNet-B3) under the identical attack and poisoning ratio.
- No single defense dominates across all eight attack types in the benchmark; each
  defense encodes a specific assumption (activation-pattern separability, loss-curve
  separability, Lipschitz-constant separability, etc.) that particular attacks are
  designed to violate.

**Limitation relevant here.** BackdoorBench, like STRIP and Neural Cleanse, evaluates
centralized training only. It has no federated learning attack/defense pairs, and no
notion of aggregation-level defense (Krum, trimmed mean, coordinate-wise median, norm
clipping) that is standard in the federated robustness literature.

---

## 2. State of the Art Review (Synthesis)

**Current best methods, by category:**
- *Run-time input filtering*: STRIP-style perturbation/entropy methods — cheap,
  architecture-agnostic, but blind to partial/source-specific triggers and vulnerable to
  entropy-manipulation adaptive attacks.
- *Offline model auditing*: Neural Cleanse-style trigger reverse engineering and
  outlier detection — strong when one or a few labels are infected with small triggers,
  weak against large triggers or many simultaneously infected labels.
- *Training-time / data-level*: activation clustering, spectral signatures, anti-
  backdoor learning (loss-gap-based isolation) — effective when the defender controls
  the training loop, which is exactly what federated learning denies the server.
- *Aggregation-level (federated-specific, not covered by the three references above but
  established in the wider FL robustness literature)*: Krum/Multi-Krum, trimmed mean,
  coordinate-wise median, and norm-clipping aggregation rules that bound the influence
  any single client update can have on the global model.

**Current benchmark datasets/models:** CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet, MNIST;
ResNet/VGG/EfficientNet/MobileNet/DenseNet family models; BadNets/Blended/SIG/WaNet/
Input-aware as canonical attacks.

**Current evaluation metrics:** Clean Accuracy (C-Acc), Attack Success Rate (ASR),
Robust Accuracy (R-Acc = correct-original-label rate on poisoned inputs), plus
detection-side FAR/FRR and anomaly index.

## 3. Identified Gaps and Opportunity for Novelty

1. **No paper here combines run-time input filtering with federated aggregation-level
   defense in one pipeline.** STRIP defends a deployed model; Neural Cleanse audits a
   trained model; neither addresses *how the model got poisoned during distributed
   training in the first place*. A federated setting needs a defense at three points —
   per-round client-update level, periodic global-model audit level, and run-time
   input level — and no single reference method spans all three.
2. **Distributed collusion is under-addressed.** A single stealthy trigger split across
   multiple colluding clients (each contributing a partial, individually-innocuous
   perturbation) does not trip single-client anomaly detection (Krum-style) and does
   not present as a clean single-label outlier to Neural Cleanse until the fragments
   combine in the aggregated model. This is a concrete, buildable, and demoable novel
   attack + matching defense angle.
3. **Adaptive-attack blind spots are complementary, not identical.** STRIP is defeated
   by source-label-specific triggers; Neural Cleanse is defeated by large triggers and
   many-label infection. An ensemble that runs both, cross-validated against each
   other's confidence, closes gaps that neither method closes alone — and this
   combination is not evaluated in any of the three references.
4. **Explainability of *why* an update was flagged is missing across all three papers.**
   Anomaly indices and entropy scores are single numbers; none of the references surface
   *which features/neurons/update-directions* drove the flag in a way a non-expert
   competition judge can inspect. This is a practical, presentation-relevant gap.

These four gaps directly define the system in `ARCHITECTURE.md`.

---

## 4. Repository Analysis (cloned and inspected directly)

The two zip files originally provided were curated paper-link lists, not code. The
three GitHub links provided since are real, substantial codebases and were cloned and
inspected. This section documents what's actually in them, since it changes several
build-vs-reuse decisions in `ARCHITECTURE.md`.

### 4.1 `SCLBD/BackdoorBench`

**Goal.** Reference implementation backing the BackdoorBench paper (§1.3).

**Structure that matters here:**
- `attack/` — one file per attack (`badnet.py`, `blended.py`, `wanet.py`, `sig.py`, …),
  each implementing a `prototype.py`-derived interface.
- `defense/` — one file per defense, including `nc.py` (Neural Cleanse), `ac.py`
  (Activation Clustering), `anp.py`, `abl.py`, `spectral.py`, `fp.py` (fine-pruning) —
  i.e., essentially every defense referenced in the Neural Cleanse/BackdoorBench papers,
  as real, runnable PyTorch code, not just described in prose.
- `detection_infer/strip.py` and `detection_pretrain/strip.py` — two STRIP
  implementations (inference-time and pretrain-time variants), useful as a correctness
  cross-check for our own `ai/detection/runtime_sentinel.py`.
- `utils/aggregate_block/`, `utils/bd_dataset.py` — data loading and poisoned-dataset
  construction utilities.
- No federated learning code anywhere in the repo — every attack/defense pair assumes
  a single centralized training loop. This confirms the gap identified in §3: none of
  these reference implementations touch the distributed setting this challenge asks for.

**Reusable components (conceptually, not by copying code):** the attack/defense
plugin interface pattern (`prototype.py` base class) is a clean template for how
`ai/detection/` modules in this project register themselves with the L1–L4 pipeline.

**Not reused:** none of BackdoorBench's training loop or centralized defense code is
directly applicable, since our threat model is federated, not centralized.

### 4.2 `flwrlabs/flower`

**Goal.** Production federated learning framework — this project's actual FL backend
(see `TECH_STACK.md`).

**What's directly reusable, discovered by inspection:**
- `flwr/serverapp/strategy/multikrum.py` and `krum.py` — Flower **already ships a
  production Multi-Krum strategy** (Blanchard et al. 2017), plus `bulyan.py`
  (Byzantine-robust aggregation) and `fedtrimmedavg.py` (trimmed mean). This changes a
  build decision from the first draft of this project: **L1's base robust aggregator
  should be Flower's built-in `MultiKrum`/`FedTrimmedAvg` strategy, not a hand-rolled
  reimplementation.** The NumPy `multi_krum()` in `ai/fl_core/fl_engine.py` in this
  repo exists only as a dependency-free proof-of-concept matching Flower's documented
  semantics; the PyTorch/production path should subclass Flower's strategy directly.
- `flwr/server/strategy/aggregate.py` — reference aggregation math (weighted average,
  trimmed mean, Krum distance computation) worth cross-checking our implementation
  against.
- `baselines/flanders/` — a Flower baseline specifically about *detecting malicious
  clients via update-history analysis*, conceptually adjacent to this project's L1
  collusion clustering; worth reading before finalizing L1's design, though not
  inspected line-by-line here due to scope.

### 4.3 `FedML-AI/FedML`

**Goal.** Broader federated learning platform (simulation, cross-device, cross-silo,
IoT) with an extensive `core/security/defense/` module.

**Most important finding:** `defense/foolsgold_defense.py` implements **FoolsGold**
(Fung et al., RAID 2020) — a defense specifically designed for **Sybil/colluding
attackers** in federated learning. It computes pairwise cosine similarity between
clients' historical gradient contributions (using the last layer as the "importance
feature") and down-weights clients whose contributions are suspiciously similar to
another client's, with a "pardoning" step to avoid punishing similarity caused by
class imbalance rather than collusion.

**This is prior art directly relevant to the "novel" collusion-detection angle claimed
in the first draft of `ARCHITECTURE.md`, and the design has been revised accordingly**
(see `ARCHITECTURE.md` §2.1 and the module docstring in
`ai/detection/update_guard.py`). The honest positioning is: FoolsGold already solves
sybil-style collusion at the *reweighting* level. This project's L1 addition is
narrower and complementary — clustering *residuals after Multi-Krum's own filtering*
(catching what survives Krum specifically) and feeding a bounded score into a shared,
auditable Trust Ledger (L4) that also incorporates L2/L3 signals, rather than acting as
a standalone reweighting scheme. A rigorous comparison against FoolsGold head-to-head
is now in the evaluation plan (`IMPLEMENTATION_PLAN.md`) rather than being skipped.

**Other reusable defense modules found:** `krum_defense.py`,
`coordinate_wise_trimmed_mean_defense.py`, `geometric_median_defense.py`,
`norm_diff_clipping_defense.py`, `robust_learning_rate_defense.py` — all implement
`defense_base.BaseDefenseMethod`, a clean `defend_before_aggregation` /
`defend_after_aggregation` interface. This interface pattern is worth mirroring in
`ai/detection/` for consistency with an ecosystem a judge may already know.

### 4.4 Revised build-vs-reuse table

| Component | First-draft plan | Revised, after reading the actual repos |
|---|---|---|
| Base robust aggregation | Hand-rolled Multi-Krum | **Reuse Flower's `MultiKrum`/`FedTrimmedAvg` strategy** in the PyTorch path |
| Collusion detection | Presented as fully novel | **Positioned relative to FoolsGold** (FedML) — narrower, complementary, benchmarked against it, not presented as unprecedented |
| STRIP baseline | Reimplement from paper only | Reimplemented from paper (`ai/detection/runtime_sentinel.py`), **cross-checked against `BackdoorBench/detection_infer/strip.py`** for correctness |
| Neural Cleanse baseline (L2) | Reimplement from paper | Can reuse `BackdoorBench/defense/nc.py` directly for the centralized-audit step, since L2 audits the aggregated global model centrally regardless of how training was distributed |
