# ai/fl_core/

**Purpose**: the federated learning engine — client/server round loop and
aggregation. See `../ARCHITECTURE.md` §7.1–7.2 and `../INTERFACES.md#Aggregator`.

**Contents**:
- `fl_engine.py` — NumPy reference implementation (FedAvg, Multi-Krum), used by
  `scripts/run_demo.py`. Dependency-free proof-of-concept, see `../RESULTS.md`.
- `flower_app.py` (planned, Week 1–2) — production PyTorch path subclassing Flower's
  `MultiKrum`/`FedTrimmedAvg` strategies (`../RESEARCH.md` §4.2).

**Dependencies**: `ai/training/` (data loading), consumed by `ai/detection/` (L1).

**Not this module's job**: any detection/defense logic — that's `ai/detection/`.
