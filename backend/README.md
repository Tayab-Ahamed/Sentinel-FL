# backend/

**Purpose**: FastAPI backend serving experiment artifacts to the dashboard. See
`API.md` for the full endpoint contract and `../ARCHITECTURE.md` §5 for how it fits
the overall system.

**Contents (planned, not yet implemented — see IMPLEMENTATION_PLAN.md Week 4)**:
- `main.py` — FastAPI app, routes per `API.md`
- `services/` — implementations of `MetricsCollector`/`Visualizer` (`../INTERFACES.md`)
- `models/` — pydantic models mirroring `../SCHEMAS.md`

**Dependencies**: `ai/` (reads its logged/checkpointed output, never imports its
training code directly), `experiments/` (data source).

**Future extension**: authentication layer per `API.md` §10, if ever multi-tenant.
