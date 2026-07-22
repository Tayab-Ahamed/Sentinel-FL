# TESTING.md — Test Strategy and Results

## 1. Test Tiers

| Tier | Scope | Tooling | Frequency |
|---|---|---|---|
| **Smoke** | Every module imports and can be instantiated | pytest | Every commit |
| **Unit** | Each module in isolation, pure functions | pytest | Every commit |
| **Integration** | L1→L4 pipeline on synthetic data | pytest | Every commit |
| **Security** | Adversarial payloads, boundary violations | pytest | Every commit |
| **Performance** | Time-budget benchmarks (not correctness) | pytest | Every commit |
| **Benchmark** | Full `BENCHMARK.md` ablation matrix (multi-seed) | `run_benchmark.py` | Manual / scheduled |
| **Slow** | Actual Flower FL rounds (network required) | pytest `-m slow` | Manual only |

## 2. Running Tests

```bash
# All tests except Flower network and benchmark (recommended, ~90s)
pytest -m "not slow and not benchmark" -q

# With coverage report
pytest -m "not slow and not benchmark" --cov=ai --cov=backend --cov-report=term-missing

# Unit tests only (fastest, ~10s)
pytest -m unit -q

# Integration + pipeline tests
pytest -m integration -q

# Security tests
pytest tests/test_security.py -v

# Performance budget tests
pytest tests/test_performance.py -v
```

## 3. Test Files

| File | Type | Count | Description |
|---|---|---|---|
| `tests/test_smoke.py` | Smoke | 40+ | Import + instantiation for every public class |
| `tests/test_security.py` | Security | 30+ | Adversarial payloads, path traversal, overflow |
| `tests/test_performance.py` | Performance | 12 | Time-budget benchmarks for hot paths |
| `tests/test_backend_api.py` | Integration | 30+ | All FastAPI endpoints with realistic data |
| `tests/test_integration_pipeline.py` | Integration | 6 | Full L1→L4 pipeline end-to-end |
| `tests/test_fl_engine_core.py` | Unit | ~20 | FedAvg, Multi-Krum, local_train correctness |
| `tests/test_model_registry.py` | Unit | 20 | Checkpoint save/load/retention/rollback |
| `tests/test_dataset_loader.py` | Unit | 19 | Partition/load logic, dev_mode, caching |
| `tests/test_attacks_visualizer.py` | Unit | 17 | Plot generation, file output, error handling |

## 4. Milestone 12 — Final Results

| Metric | Value |
|---|---|
| **Tests passed** | **976** |
| **Failures** | **0** |
| **Total coverage** | **84.22%** |
| Coverage threshold (`fail_under`) | 60% |
| Margin | **+24.2 pp** |

## 5. Reproducibility Conventions

- **Explicit seeds everywhere.** Every stochastic function (`make_dataset`,
  `inject_trigger`, `dirichlet_partition`, `local_train`, `multi_krum`) accepts
  a `seed` parameter. The global NumPy RNG is never called unseeded.

- **Fixed-seed integration tests.** `test_integration_pipeline.py` uses `seed=42`
  throughout. Adding a regression fixture (`experiments/*/results.json`) is the next
  step for regression testing.

- **Environment variables.** CI sets `SENTINEL_LOG_LEVEL=WARNING` and
  `SENTINEL_SEED=42` to suppress log noise and pin the default seed.

## 6. What Is Not Tested (by design)

| Area | Reason |
|---|---|
| `ai/fl_engine/simulation.py` | Requires live Flower network — `slow` marker, run manually |
| `ai/fl_engine/strategy.py` | Same — Flower strategy internals tested upstream |
| Load / stress testing of API | Out of scope for local single-user demo |
| Docker build correctness | Verified manually; CI verifies import not container |

## 7. Key APIs and Contracts Verified by Tests

The following non-obvious API contracts were validated by the test suite and documented
for future contributors:

- `TrustLedgerEntry.reason` is a required string (not `explanation` or `event_type`)
- `FileTrustLedger.query(q)` requires a `TrustLedgerQuery` argument (not keyword args)
- `UpdateGuard.process_round(round_num, client_ids, deltas)` — `ledger` goes to `__init__`
- `AttackConfig.malicious_client_indices` is `list[int]` (not `malicious_client_ids`)
- `precision_recall_f1(y_true, y_pred, pos_label=1)` uses `pos_label` (not `positive_label`)
- `ReputationEngine.client_reputation_report(client_id)` returns a dict (no `get_reputation`)
- Backend health endpoint: `/api/health` (prefix `/api`, not `/api/v1`)
- Heatmap route: `/api/v1/experiments/{id}/reputation-heatmap` (full name)

## 8. CI Integration

Tests run automatically via `.github/workflows/ci.yml` on every push to `main`/`develop`:

```
lint   → ruff check + ruff format --check
test   → pytest on Python 3.11 + 3.12
verify → python scripts/verify_install.py
demo   → python scripts/run_demo.py + ASR sanity assertions
api    → FastAPI import + /api/health TestClient check
```
