# ai/detection/

**Purpose**: all four defense layers (L1–L4). See `../ARCHITECTURE.md` §2 for the
full design.

**Contents**:
- `update_guard.py` — L1 collusion clustering (§2.1)
- `runtime_sentinel.py` — L3 STRIP-style entropy detector (§2.3)
- `model_auditor.py` (planned, Week 3) — L2 Neural-Cleanse-style audit (§2.2)
- `trust_ledger.py` (planned, Week 3) — L4 (§2.4)
- `activation_consistency.py` (planned, Week 2) — L3 Signal 2 (§2.3)

**Dependencies**: `ai/fl_core/` (receives client updates and models to inspect).

**Interface contract**: every module here implements `Detector` or
`DefenseStrategy` from `../INTERFACES.md` — do not add a detector that bypasses
this contract, since the dashboard/API depend on it.
