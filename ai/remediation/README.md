# ai/remediation/

**Purpose**: the Remediation Engine — designed in `../ARCHITECTURE.md` §7.4, **not
yet implemented** (`../IMPLEMENTATION_PLAN.md` Week 3).

**Planned contents**:
- `rollback.py` — thin wrapper over `ModelRegistry.rollback_to()` (`../INTERFACES.md`)
- `unlearning.py` — targeted fine-tuning on L2's reversed trigger, mirroring
  `BackdoorBench/defense/fp.py`'s approach (see `../RESEARCH.md` §4.1, §4.4)

**Dependencies**: `ai/detection/` (consumes `AuditReport`), Model Registry
(part of `backend/services/`).
