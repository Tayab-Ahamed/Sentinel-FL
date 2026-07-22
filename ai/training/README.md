# ai/training/

**Purpose**: dataset generation/partitioning and attack simulation for evaluation. See
`../DATASETS.md` and `../INTERFACES.md#AttackSimulator`.

**Contents**:
- `poison.py` — synthetic Phase 0 data, Dirichlet partitioning, BadNets-style trigger
  injection (`../RESULTS.md` documents its output).
- `official_dataset_loader.py` (planned, Week 1, blocked on 13 July release) —
  Phase 1 `DatasetLoader` implementation for the official challenge data.

**Dependencies**: none within `ai/` (this is the data source everything else consumes).

**Important boundary**: poisoning/attack-simulation ground truth (`AttackReport`,
`../SCHEMAS.md`) must never be passed into `ai/detection/` — only used by
`ai/fl_core/`'s evaluation step, after the fact, to compute ASR.
