# NOVELTY.md — Why Sentinel-FL

This document exists to answer one question directly: **why should judges select this
project over a straightforward "run STRIP + Neural Cleanse on the challenge dataset"
submission?** It cross-references `RESEARCH.md` (literature grounding) and
`ARCHITECTURE.md` (system design) rather than repeating them.

## 1. State of the Art, and Where It Stops

| Approach | What it defends | What it does not defend |
|---|---|---|
| STRIP (`RESEARCH.md` §1.1) | A single deployed model, per-input, at inference time | How the model got poisoned; source-label-specific triggers; entropy-manipulation adaptive attacks |
| Neural Cleanse (`RESEARCH.md` §1.2) | A single trained model, offline, before deployment | Large triggers; many-simultaneously-infected labels; the federated training process itself |
| BackdoorBench defenses (`RESEARCH.md` §1.3, §4.1) | Centralized training pipelines, one defense at a time | Anything federated — there is no FL code in the repository at all |
| Flower Multi-Krum / FedTrimmedAvg (`RESEARCH.md` §4.2) | Per-round outlier client updates | Colluding groups where no single update is an outlier |
| FedML FoolsGold (`RESEARCH.md` §4.3) | Sybil/colluding clients, via raw-gradient similarity | Everything downstream of aggregation — nothing at the trained-model or inference-time level |

**The pattern**: every one of these is a point solution for one stage of the ML
lifecycle (client update → aggregation → trained model → inference). None of them
compose with each other, and none of them share evidence — a Neural Cleanse audit
finding and a STRIP detection on the same label are, in the reference implementations,
two numbers in two unrelated log files.

## 2. Research Gaps (from RESEARCH.md §3, restated as claims)

1. No reference implementation defends all three lifecycle stages (update, model,
   inference) in one coordinated system.
2. Fragmented/collusive triggers — where several clients each contribute an
   individually-mild poisoning signal that only becomes dangerous once combined — are
   not the primary target of any single reference method. FoolsGold is the closest,
   but operates purely at the raw-gradient level with no downstream corroboration.
3. STRIP and Neural Cleanse have *complementary*, not overlapping, blind spots (large
   triggers defeat Neural Cleanse; source-label-specific triggers defeat STRIP) — but no
   reference implementation cross-validates one against the other.
4. None of the reference methods produce a human-inspectable *reason* for a flag beyond
   a single anomaly score or entropy number.

## 3. What Sentinel-FL Actually Contributes

**Contribution 1 — Lifecycle-spanning defense with a shared evidence store.**
L1 (Update Guard), L2 (Model Auditor), and L3 (Runtime Sentinel) are not independent
tools run separately — they write to a common L4 Trust Ledger keyed by client ID and
label ID, and each layer can consume the others' history (e.g., L2 prioritizes
auditing labels that L3 has been flagging at inference time — `ARCHITECTURE.md` §2.2).
This composition, not any single detector, is the contribution.

**Contribution 2 — Residual collusion clustering as a second-order signal.**
Positioned honestly against FoolsGold (`RESEARCH.md` §4.3): rather than replacing
Multi-Krum's or FoolsGold's per-round filtering, L1's collusion clustering operates on
*residuals after* Multi-Krum has already acted, specifically hunting for what survives
Krum's own filter. It is evaluated head-to-head against FoolsGold in
`BENCHMARK.md`, not assumed superior.

**Contribution 3 — Two-signal runtime fusion.**
L3 fuses STRIP's entropy signal with a second, independent activation-consistency
signal (`ARCHITECTURE.md` §2.3), so an adaptive attacker optimizing to defeat one
signal (e.g., STRIP's documented entropy-manipulation weakness) still has to defeat the
other, which relies on internal representations rather than output entropy.

**Contribution 4 — Evidence, not just a score.**
Every flag at every layer is stored with the feature vector and a human-readable
reason string (`ARCHITECTURE.md` §2.4), so a judge — or a real operator — can inspect
*why* something was flagged, not just that it was.

## 4. What Sentinel-FL Does Not Claim

- It does not claim to beat Multi-Krum, FoolsGold, STRIP, or Neural Cleanse on their
  own home turf (single-outlier robustness, sybil detection, single-model inference
  detection, single-model auditing respectively) — the honest expectation, tested in
  `BENCHMARK.md`, is parity on those cases and a measurable edge specifically on the
  fragmented-collusion and cross-validated-detection scenarios those methods don't
  target.
- The Phase 0 results in `RESULTS.md` are proof-of-concept sanity checks on synthetic
  data with a linear model, not competition-grade numbers. Real numbers come from
  Phase 1–2 of `IMPLEMENTATION_PLAN.md` once the official dataset and a CNN are in use.

## 5. Expected Scientific Contribution

A small, honestly-scoped one: an empirical answer to "does layering a residual-based
collusion signal behind Multi-Krum, and cross-validating STRIP against Neural Cleanse
findings, measurably close the fragmented-trigger gap that neither defense closes
alone?" — with an ablation matrix (`BENCHMARK.md`) as the evidence either way. This is
the kind of question a `Computer` magazine-style short paper or workshop note could be
built from — not a claim of a fundamentally new defense paradigm.

## 6. Expected Competition Impact

Judging criteria emphasize quantitative performance, novelty, and presentation
(`RESEARCH.md` intro context). This project's presentation angle is a live ablation
demo: toggle L1/L2/L3 on and off against the fragmented-collusion attack and watch ASR
change in real time via the dashboard (`ARCHITECTURE.md` §5) — a clearer story for a
judge than a single accuracy table, and one that visibly demonstrates the composed-
system contribution rather than just re-running a published detector.
