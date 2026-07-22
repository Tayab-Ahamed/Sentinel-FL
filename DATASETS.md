# DATASETS.md

## Phase 0 (pre-13 July) — synthetic development set
- **CIFAR-10**, partitioned across 10–20 simulated FL clients using a Dirichlet(α)
  distribution to create realistic non-IID splits (small α = more skewed, harder for
  aggregation-level defenses — used to stress-test L1).
- Trigger: BadNets-style opaque white square, bottom-right corner, ~1–4% of image area,
  injected into a configurable fraction of one or more clients' local data.

## Phase 1 (from 13 July) — official challenge dataset
- To be swapped in once the official GSC26 Challenge 1 dataset/spec is released
  (per the 13 July 3pm ET email). All partitioning and injection scripts in
  `datasets/` are written against a generic `(x, y, client_id)` interface so the swap
  should require no changes to `ai/detection/` or `ai/fl_core/`.

## Held-out sets
- Clean validation set (server-side, used by L2 Model Auditor and L3 calibration) —
  must never contain poisoned samples; kept separate from all client partitions.
- Synthetic trojaned test set (attacker-simulated, used only for evaluation metrics
  ASR/R-Acc/FAR/FRR — never exposed to the defense pipeline during training).
