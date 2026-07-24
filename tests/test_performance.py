"""
tests/test_performance.py — Performance and benchmark tests.

Validates that core operations complete within expected time budgets and
that memory usage stays bounded under realistic workload sizes.

Marked as 'benchmark' — run with: pytest -m benchmark
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Time budget constants (seconds) — generous for CI
# ---------------------------------------------------------------------------
_T_FEDAVG_1K = 0.5  # fedavg over 1000-dim vectors, 20 clients
_T_MULTIKRUM_1K = 2.0  # multi_krum over same
_T_LOCAL_TRAIN = 1.0  # one client local_train (5 epochs)
_T_GUARD_ROUND = 2.0  # UpdateGuard.process_round, 12 clients
_T_LEDGER_WRITE = 1.0  # 100 ledger writes
_T_LEDGER_QUERY = 0.5  # query 1000-entry ledger
_T_NORM_CALC = 0.2  # compute norms for 20 clients × 5000 params
_T_PARTITION = 0.5  # Dirichlet partition of 10k samples


# ---------------------------------------------------------------------------
# FL Engine benchmarks
# ---------------------------------------------------------------------------


class TestFLEngineBenchmarks:
    @pytest.mark.benchmark
    def test_fedavg_speed(self):
        from ai.fl_core.fl_engine import fedavg

        dim = 1000
        rng = np.random.default_rng(0)
        updates = [rng.standard_normal(dim) for _ in range(20)]
        weights = list(range(1, 21))

        t0 = time.perf_counter()
        for _ in range(100):
            fedavg(updates, weights)
        elapsed = time.perf_counter() - t0

        assert elapsed < _T_FEDAVG_1K, f"fedavg x100 took {elapsed:.3f}s (limit {_T_FEDAVG_1K}s)"

    @pytest.mark.benchmark
    def test_multi_krum_speed(self):
        from ai.fl_core.fl_engine import multi_krum

        dim = 1000
        rng = np.random.default_rng(0)
        updates = [rng.standard_normal(dim) for _ in range(12)]

        t0 = time.perf_counter()
        for _ in range(10):
            multi_krum(updates, num_malicious_assumed=3, num_to_select=9)
        elapsed = time.perf_counter() - t0

        assert elapsed < _T_MULTIKRUM_1K, (
            f"multi_krum x10 took {elapsed:.3f}s (limit {_T_MULTIKRUM_1K}s)"
        )

    @pytest.mark.benchmark
    def test_local_train_speed(self):
        from ai.fl_core.fl_engine import LinearSoftmaxModel, local_train

        n, nf, nc = 500, 20, 5
        rng = np.random.default_rng(1)
        X = rng.standard_normal((n, nf)).astype(np.float32)
        y = rng.integers(0, nc, n)
        m = LinearSoftmaxModel(nf, nc)
        p = m.get_params()

        t0 = time.perf_counter()
        for _ in range(5):
            local_train(p, nf, nc, X, y, epochs=5, lr=0.1)
        elapsed = time.perf_counter() - t0

        assert elapsed < _T_LOCAL_TRAIN, (
            f"local_train x5 took {elapsed:.3f}s (limit {_T_LOCAL_TRAIN}s)"
        )


# ---------------------------------------------------------------------------
# Detection benchmarks
# ---------------------------------------------------------------------------


class TestDetectionBenchmarks:
    @pytest.mark.benchmark
    def test_norm_calculation_speed(self):
        from ai.detection.norm_calculator import compute_norms

        dim = 5000
        rng = np.random.default_rng(2)
        updates = [rng.standard_normal(dim) for _ in range(20)]

        t0 = time.perf_counter()
        for _ in range(50):
            compute_norms(updates)
        elapsed = time.perf_counter() - t0

        assert elapsed < _T_NORM_CALC * 50, f"compute_norms x50 took {elapsed:.3f}s"

    @pytest.mark.benchmark
    def test_update_guard_round_speed(self, tmp_path):
        from ai.detection.update_guard import UpdateGuard

        dim = 500
        rng = np.random.default_rng(3)
        updates = [rng.standard_normal(dim) for _ in range(12)]
        client_ids = [f"c_{i:02d}" for i in range(12)]
        guard = UpdateGuard()

        t0 = time.perf_counter()
        for rnd in range(20):
            guard.process_round(rnd, client_ids, updates)
        elapsed = time.perf_counter() - t0

        assert elapsed < _T_GUARD_ROUND, (
            f"20 guard rounds took {elapsed:.3f}s (limit {_T_GUARD_ROUND}s)"
        )

    @pytest.mark.benchmark
    def test_ledger_write_throughput(self, tmp_path):
        from ai.detection.trust_ledger import FileTrustLedger
        from ai.fl_core.schemas import TrustLedgerEntry

        ledger = FileTrustLedger(tmp_path / "throughput.jsonl", warm_start=False)

        t0 = time.perf_counter()
        for i in range(100):
            entry = TrustLedgerEntry(
                subject_type="client",
                subject_id=f"c_{i % 10:02d}",
                round_num=i,
                layer_id="L1",
                score=min(float(i % 10) / 10.0, 1.0),
                reason=f"Test entry {i}",
            )
            ledger.add_entry(entry)
        elapsed = time.perf_counter() - t0

        assert elapsed < _T_LEDGER_WRITE, (
            f"100 ledger writes took {elapsed:.3f}s (limit {_T_LEDGER_WRITE}s)"
        )

    @pytest.mark.benchmark
    def test_ledger_query_speed(self, tmp_path):
        from ai.detection.trust_ledger import FileTrustLedger
        from ai.fl_core.schemas import TrustLedgerEntry, TrustLedgerQuery

        ledger = FileTrustLedger(tmp_path / "query_ledger.jsonl", warm_start=False)
        # Write 500 entries
        for i in range(500):
            entry = TrustLedgerEntry(
                subject_type="client",
                subject_id=f"c_{i % 20:02d}",
                round_num=i % 50,
                layer_id="L1",
                score=0.5,
                reason="bench",
            )
            ledger.add_entry(entry)

        t0 = time.perf_counter()
        _ = ledger.query(TrustLedgerQuery())
        elapsed = time.perf_counter() - t0

        assert elapsed < _T_LEDGER_QUERY, (
            f"Query of 500-entry ledger took {elapsed:.3f}s (limit {_T_LEDGER_QUERY}s)"
        )


# ---------------------------------------------------------------------------
# Training benchmarks
# ---------------------------------------------------------------------------


class TestTrainingBenchmarks:
    @pytest.mark.benchmark
    def test_dirichlet_partition_speed(self):
        from ai.training.poison import dirichlet_partition, make_dataset

        X, y = make_dataset(10_000, 10, 5, seed=0)
        n_train = 8000

        t0 = time.perf_counter()
        for _ in range(5):
            dirichlet_partition(n_train, 20, y[:n_train], 5, alpha=0.5, seed=0)
        elapsed = time.perf_counter() - t0

        assert elapsed < _T_PARTITION, f"partition x5 took {elapsed:.3f}s (limit {_T_PARTITION}s)"

    @pytest.mark.benchmark
    def test_trigger_injection_speed(self):
        from ai.training.poison import inject_trigger

        rng = np.random.default_rng(0)
        X = rng.standard_normal((2000, 10)).astype(np.float32)
        y = rng.integers(0, 4, 2000)

        t0 = time.perf_counter()
        for _ in range(20):
            inject_trigger(X, y, 0, slice(0, 2), trigger_value=5.0, poison_fraction=0.2, seed=0)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.5, f"inject_trigger x20 took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Scalability: increasing client count
# ---------------------------------------------------------------------------


class TestScalability:
    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_clients", [10, 20, 50])
    def test_fedavg_scales_with_clients(self, n_clients):
        from ai.fl_core.fl_engine import fedavg

        rng = np.random.default_rng(0)
        updates = [rng.standard_normal(200) for _ in range(n_clients)]
        weights = [100] * n_clients
        t0 = time.perf_counter()
        fedavg(updates, weights)
        elapsed = time.perf_counter() - t0
        # Should always be sub-second for these sizes
        assert elapsed < 0.1, f"fedavg with {n_clients} clients took {elapsed:.3f}s"

    @pytest.mark.benchmark
    @pytest.mark.parametrize("dim", [100, 1_000, 10_000])
    def test_norm_calculation_scales_with_dim(self, dim):
        from ai.detection.norm_calculator import compute_l2_norms

        rng = np.random.default_rng(0)
        updates = [rng.standard_normal(dim) for _ in range(10)]
        t0 = time.perf_counter()
        compute_l2_norms(updates)
        elapsed = time.perf_counter() - t0
        # Sub-100ms even for 10k dims
        assert elapsed < 0.1, f"compute_l2_norms dim={dim} took {elapsed:.3f}s"
