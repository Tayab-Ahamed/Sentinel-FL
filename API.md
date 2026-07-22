# API.md — Backend REST API

Backend: FastAPI (`TECH_STACK.md`), serving pre-computed experiment artifacts to the
dashboard (`ARCHITECTURE.md` §5). No live training is triggered through this API during
a judged demo — see `DEPLOYMENT` note in `ARCHITECTURE.md`. All endpoints are read-only
except `POST /experiments/run`, used only for local development.

Base path: `/api/v1`. All responses are JSON. All list endpoints support
`?limit=` and `?offset=` (default `limit=50`).

Authentication: none in Phase 0/1 (local/offline demo). See §9 for the Phase 2+ note.

---

## 1. `GET /experiments`

**Purpose**: list all recorded experiments (config + attack + defense-layer
combinations, see `SCHEMAS.md#Experiment`).

- **Response 200**: `{ "experiments": Experiment[] }`
- **Errors**: none (empty list if none exist)

**Example response**
```json
{ "experiments": [
  { "experiment_id": "exp_001", "dataset_phase": "phase0_synthetic",
    "layers_enabled": ["L1"], "result": { "clean_accuracy": 1.0,
    "attack_success_rate": 0.258 } }
]}
```

---

## 2. `GET /experiments/{experiment_id}`

**Purpose**: full detail for one experiment, including config and full
`EvaluationResult`.

- **Path params**: `experiment_id: str`
- **Response 200**: `Experiment` (full object, `SCHEMAS.md#Experiment`)
- **Response 404**: `{ "error": "experiment_not_found", "experiment_id": "..." }`

---

## 3. `GET /experiments/{experiment_id}/rounds`

**Purpose**: per-round timeline data for the dashboard's reputation heatmap and
metric charts.

- **Response 200**: `{ "rounds": TrainingRound[] }` (`SCHEMAS.md#TrainingRound`)
- **Response 404**: same as above

---

## 4. `GET /experiments/{experiment_id}/reputation-heatmap`

**Purpose**: backs `Visualizer.reputation_heatmap` (`INTERFACES.md`) — client × round
matrix of trust scores.

- **Response 200**:
```json
{
  "client_ids": ["client_00", "client_01", "..."],
  "rounds": [0, 1, 2, "..."],
  "scores": [[0.1, 0.1, 0.9], ["..."]]
}
```
- **Validation**: `experiment_id` must exist and have `L1` in `layers_enabled`,
  otherwise `Response 400: { "error": "layer_not_enabled", "layer": "L1" }`

---

## 5. `GET /experiments/{experiment_id}/metrics?names=clean_accuracy,attack_success_rate`

**Purpose**: backs `Visualizer.metric_timeseries`.

- **Query params**: `names` — comma-separated metric names (`SCHEMAS.md#Metric`)
- **Response 200**: `{ "series": { "<metric_name>": Metric[] } }`
- **Response 400**: `{ "error": "unknown_metric", "name": "..." }` if any requested
  name isn't a recognized metric

---

## 6. `GET /experiments/{experiment_id}/audits/{round_num}`

**Purpose**: backs `Visualizer.audit_report` — L2's reversed triggers and per-label
anomaly scores for a given audited round.

- **Response 200**: `AuditReport` (`SCHEMAS.md#AuditReport`)
- **Response 404**: `{ "error": "audit_not_found", "round_num": ... }` — e.g. requested
  a round that wasn't an audit round (see `audit_interval_rounds` in
  `SCHEMAS.md#Configuration`)

---

## 7. `GET /trust-ledger/{entry_id}`

**Purpose**: backs `Visualizer.explainability_drilldown` — the human-readable reason
and raw evidence behind a specific flag.

- **Response 200**: `TrustLedgerEntry` (`SCHEMAS.md#TrustLedgerEntry`)
- **Response 404**: `{ "error": "entry_not_found" }`

---

## 8. `POST /experiments/run` (local development only)

**Purpose**: trigger `scripts/run_demo.py`-equivalent run with a given config. Not
exposed in the deployed demo build (see the Deployment Diagram in
`ARCHITECTURE.md` §9).

- **Request body**: `Configuration` (`SCHEMAS.md#Configuration`)
- **Response 202**: `{ "experiment_id": "...", "status": "queued" }`
- **Response 400**: `{ "error": "invalid_config", "details": [...] }` — field-level
  validation errors (e.g. `collusion_sim_threshold` outside `[0,1]`)
- **Response 409**: `{ "error": "experiment_already_running" }` — single-run-at-a-time
  in the reference implementation; concurrency is intentionally out of scope — not
  needed for a judged demo that serves pre-computed results

---

## 9. Error Handling Convention

All error responses follow `{ "error": "<snake_case_code>", ...context }` with a
matching HTTP status. No endpoint returns partial/malformed JSON on error — errors are
caught at the router level and normalized before response.

## 10. Authentication (forward-looking note, not implemented in the challenge build)

Not required for the competition deliverable (local/offline, single-user demo). If this
became a multi-tenant deployment, the natural extension is an API key per FL
deployment, validated on `POST /experiments/run` only (read endpoints stay open for a
public leaderboard use case) — noted here so a future implementer doesn't have to
re-derive this decision, not because it's built now.
