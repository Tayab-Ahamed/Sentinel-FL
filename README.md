# SENTINEL-FL — Federated Backdoor Immune System

<div align="center">

**IEEE Computer Society Global Student Challenge 2026 · Challenge 1 · Team 010**

[![CI](https://github.com/GSC26-Team010/sentinel-fl/actions/workflows/ci.yml/badge.svg)](https://github.com/GSC26-Team010/sentinel-fl/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen)](#test-results)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

SENTINEL-FL is a **four-layer, lifecycle-spanning federated learning security system**
that defends against backdoor attacks coordinated across colluding clients — a class of
attack that no single existing defense (STRIP, Neural Cleanse, Multi-Krum, FoolsGold)
handles end-to-end.

```
      ┌─────────────────────────────────────────────────────────┐
      │                   SENTINEL-FL Defense Stack              │
      │                                                           │
      │  L1 Update Guard    L2 Model Auditor    L3 Runtime Sentinel │
      │  ─────────────      ──────────────      ───────────────── │
      │  Collusion          Neural Cleanse       STRIP entropy +  │
      │  clustering on      style offline        activation       │
      │  per-round          trigger reverse      consistency      │
      │  residuals          engineering          fusion           │
      │       │                  │                    │           │
      │       └──────────────────┴────────────────────┘           │
      │                          │                                │
      │              L4 Trust Ledger  ←  Shared evidence store   │
      │              per-client reputation + explainable reasons  │
      └─────────────────────────────────────────────────────────┘
```

**Key novelty:** the four layers share a common evidence store (L4 Trust Ledger) so
each layer can consume the others' history — L2 prioritises auditing labels that L3 has
been flagging, L1 cluster detections are corroborated by L4 reputation scores.
No existing implementation composes these defences in one system.

See [`NOVELTY.md`](NOVELTY.md) for the full competition differentiation argument and
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the complete system design.

---

## Quick Start

### Option 1 — Python (recommended for development)

```bash
# 1. Clone and install
git clone <repo-url>
cd GSC26-Challenge1-010
pip install -e ".[dev]"

# 2. Verify installation (exits 0 on success)
python scripts/verify_install.py

# 3. Run the end-to-end demo
python scripts/run_demo.py
#   → experiments/demo_results.json
```

### Option 2 — Docker

```bash
# Build and run the backend API
docker build -t sentinel-fl .
docker run -p 8000:8000 sentinel-fl

# Health check
curl http://localhost:8000/api/health
# → {"status":"ok","service":"SENTINEL-FL","version":"0.1.0"}
```

### Option 3 — Docker Compose (backend + frontend)

```bash
# Backend only
docker-compose up api

# Backend + React dashboard
docker-compose --profile frontend up
```

---

## Phase 0 Results (Reproducible, Seed=42)

Run `python scripts/run_demo.py` to regenerate these numbers exactly:

| Strategy | Clean Accuracy | Attack Success Rate | vs. No-Defense |
|---|---|---|---|
| FedAvg (no defense) | **100.0%** | **98.9%** | baseline |
| Multi-Krum only | 100.0% | 25.8% | −73.1 pp ASR |
| Multi-Krum + L1 Guard | 100.0% | 25.8% | −73.1 pp ASR |
| L3 STRIP (on defended model) | FRR: 2.0% | Detection: 25.0% | n/a |

**Interpretation:** Multi-Krum reduces ASR by 73 percentage points. L1's collusion guard
correctly flagged a malicious cluster in 9/20 rounds (logging to the Trust Ledger).
STRIP's 25% detection rate is expected on a 4-class linear model — see [`RESULTS.md`](RESULTS.md).

> Full Phase 0 output: [`experiments/demo_results.json`](experiments/demo_results.json)

---

## Dashboard

The React + FastAPI dashboard provides real-time visualization of:
- Client trust scores and reputation heatmap
- Round-by-round ASR / accuracy curves
- Attack visualization (trigger patterns, poisoned samples)
- Alert feed (flagged clients/inputs with reasons)
- Configuration panel

### API Endpoints (FastAPI, auto-documented at `/docs`)

```bash
# Start the backend
uvicorn backend.main:app --port 8000

# Key endpoints
GET  /api/health                              # service health
GET  /api/v1/experiments                      # list experiments
GET  /api/v1/experiments/{id}                 # experiment summary
GET  /api/v1/experiments/{id}/rounds          # per-round metrics
GET  /api/v1/experiments/{id}/reputation-heatmap  # trust score matrix
GET  /api/v1/experiments/{id}/alerts          # flagged events
GET  /api/v1/experiments/{id}/clients         # client reputation list
POST /api/v1/experiments/run                  # launch new experiment
```

Full API specification: [`API.md`](API.md)

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Screenshots: [`docs/SCREENSHOTS.md`](docs/SCREENSHOTS.md)

---

## Test Suite

```bash
# Full suite (976 tests, ~90s)
pytest -m "not slow and not benchmark"

# With coverage report
pytest -m "not slow and not benchmark" --cov=ai --cov=backend

# Unit tests only (fast)
pytest -m unit

# Integration + pipeline tests
pytest -m integration
```

### Results (Milestone 12)

| Metric | Value |
|---|---|
| Tests passed | **976** |
| Failures | **0** |
| Total coverage | **84.22%** |
| Coverage threshold | 60% (exceeded by +24 pp) |

Coverage by area: `ai/fl_core/` 95–100% · `ai/attacks/` 82–100% · `ai/detection/` 55–100% · `backend/routers/` 78–98%

---

## Project Structure

```
GSC26-Challenge1-010/
├── ai/
│   ├── fl_core/           # Schemas, FL engine (FedAvg/Multi-Krum), config, logger
│   ├── fl_engine/         # Flower-compatible simulation harness & client
│   ├── attacks/           # BadNets, image-domain attacks, ASR evaluator, visualizer
│   ├── detection/         # L1 UpdateGuard · L2 ModelAuditor · L3 RuntimeSentinel · L4 TrustLedger
│   ├── evaluation/        # Metrics engine, benchmark reporter, JSON-lines collector
│   ├── explainability/    # SHAP explainer, feature importance, chart generator
│   ├── models/            # MNIST CNN model definition
│   ├── remediation/       # Pruning / unlearning mitigation (post-detection)
│   └── training/          # Poison injection, Dirichlet partitioning, dataset loaders
├── backend/
│   ├── main.py            # FastAPI app entry point
│   ├── routers/           # experiments.py, visualizer.py
│   ├── services/          # experiment_service.py, visualizer_service.py
│   └── dependencies.py    # shared FastAPI dependency injection
├── frontend/              # React + Vite dashboard (TypeScript)
├── configs/               # YAML configuration files (default, attack, baselines, …)
├── scripts/
│   ├── run_demo.py        # End-to-end proof-of-concept (→ experiments/demo_results.json)
│   ├── run_attack.py      # Standalone attack simulation runner
│   ├── run_benchmark.py   # Baseline comparison benchmark runner
│   ├── run_flower.py      # Flower FL round harness
│   ├── run_simulation_with_guard.py  # Full pipeline simulation
│   └── verify_install.py  # Installation sanity check (exits 0 on success)
├── tests/                 # 976 tests across 9 files
├── experiments/           # Run outputs and checkpoints (demo_results.json committed)
├── datasets/              # Dataset loaders (raw data NOT committed)
├── docs/                  # Documentation and screenshot placeholders
├── docker/                # Docker helper files
├── .github/workflows/     # CI: lint, test, demo, API import check
├── Dockerfile             # Multi-stage production image
├── docker-compose.yml     # Local dev: API + frontend
├── pyproject.toml         # Build config, ruff, pytest, coverage
├── requirements.txt       # Pinned runtime dependencies
└── .env.example           # Environment variable reference

Key documents:
  ARCHITECTURE.md   Full 4-layer system design + subsystem reference
  NOVELTY.md        Competition differentiation argument
  RESEARCH.md       Literature analysis and gap identification (STRIP, NC, BackdoorBench, Flower, FedML)
  RESULTS.md        Actual Phase 0 results + Milestone 12 test metrics
  BENCHMARK.md      Evaluation plan vs. all baselines
  API.md            REST API specification
  SCHEMAS.md        All data objects, fields, types, example JSON
  INTERFACES.md     Abstract contracts for every pluggable module
  REPORT.md         Final comprehensive project report (judges' document)
```

---

## Installation from a Fresh Clone

```bash
# Requirements: Python 3.11+ (no GPU needed for Phase 0)
git clone <repo-url>
cd GSC26-Challenge1-010

# Install (editable, includes dev/test tools)
pip install -e ".[dev]"

# Verify everything works
python scripts/verify_install.py
# → SENTINEL-FL install OK — all checks passed

# Run the end-to-end demo (~20s)
python scripts/run_demo.py
# → [fedavg]          clean_acc=1.000  ASR=0.989
# → [multikrum]        clean_acc=1.000  ASR=0.258
# → [multikrum+guard]  clean_acc=1.000  ASR=0.258

# Run tests
pytest -m "not slow and not benchmark" -q
# → 976 passed in ~90s

# Lint check
ruff check .
```

> **Phase 1 (PyTorch/Flower):** `pip install -e ".[phase1]"` — requires PyTorch ≥ 2.2.
> GPU recommended but not required. See [`TECH_STACK.md`](TECH_STACK.md).

---

## Reproducibility

All stochastic functions accept an explicit `seed` parameter — there is no hidden global
RNG state. The default seed is `42` (see `configs/default.yaml` and `.env.example`).

```bash
# Re-run the demo and compare against committed results
python scripts/run_demo.py
python -c "
import json, pathlib
r = json.loads(pathlib.Path('experiments/demo_results.json').read_text())
print('FedAvg  ASR:', r['fedavg']['attack_success_rate'])
print('Krum    ASR:', r['multikrum']['attack_success_rate'])
print('Guard   ASR:', r['multikrum+guard']['attack_success_rate'])
"
```

Expected output (seed=42, deterministic):
```
FedAvg  ASR: 0.9888888...
Krum    ASR: 0.2577...
Guard   ASR: 0.2577...
```

---

## Literature Referenced

| Source | Relevance |
|---|---|
| Chen et al. 2017 — BadNets | The primary attack model (`ai/attacks/badnets.py`) |
| Gao et al. 2019 — STRIP | L3 Runtime Sentinel Signal 1 (`ai/detection/runtime_sentinel.py`) |
| Wang et al. 2019 — Neural Cleanse | L2 Model Auditor (`ai/detection/model_auditor.py`) |
| Blanchard et al. 2017 — Multi-Krum | Aggregation backbone (`ai/fl_core/fl_engine.py`) |
| Fung et al. 2020 — FoolsGold | L1 prior art (evaluated in `BENCHMARK.md`) |
| BackdoorBench (SCLBD) | Cross-validation reference (not vendored) |
| Flower (flwrlabs) | Phase 1 FL framework (`ai/fl_engine/`) |

Full analysis in [`RESEARCH.md`](RESEARCH.md).

---

## License

MIT — see [`LICENSE`](LICENSE).
