"""
scripts/verify_install.py — Installation sanity check.

Runs 10 targeted import checks and one mini FL round to verify the full
SENTINEL-FL stack is installed and importable from a fresh clone.

Usage:
    python scripts/verify_install.py

Exit codes:
    0 — all checks passed
    1 — one or more checks failed (error details printed to stderr)
"""

from __future__ import annotations

import os
import sys
import traceback

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_CHECKS: list[tuple[str, str]] = []
_FAILURES: list[str] = []


def _check(name: str):
    """Decorator that registers a check function."""

    def decorator(fn):
        _CHECKS.append((name, fn))
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------


@_check("fl_core.schemas")
def _():
    from ai.fl_core.schemas import (  # noqa: F401
        AttackConfig,
        ClientUpdate,
        Configuration,
        TrustLedgerEntry,
    )


@_check("fl_core.fl_engine")
def _():
    from ai.fl_core.fl_engine import (  # noqa: F401
        LinearSoftmaxModel,
        fedavg,
        local_train,
        multi_krum,
    )


@_check("fl_core.config")
def _():
    from ai.fl_core.config import load_config_from_dict  # noqa: F401


@_check("fl_core.logger")
def _():
    from ai.fl_core.logger import StructuredLogger  # noqa: F401


@_check("detection.update_guard")
def _():
    from ai.detection.update_guard import UpdateGuard  # noqa: F401


@_check("detection.trust_ledger")
def _():
    from ai.detection.trust_ledger import FileTrustLedger  # noqa: F401


@_check("detection.reputation_engine")
def _():
    from ai.detection.reputation_engine import ReputationEngine  # noqa: F401


@_check("training.poison")
def _():
    from ai.training.poison import dirichlet_partition, inject_trigger, make_dataset  # noqa: F401


@_check("evaluation.metrics_engine")
def _():
    from ai.evaluation.metrics_engine import accuracy, attack_success_rate  # noqa: F401


@_check("backend.main (FastAPI app)")
def _():
    from backend.main import app

    assert hasattr(app, "title"), "FastAPI app has no title"


# ---------------------------------------------------------------------------
# Functional check: one mini FL round
# ---------------------------------------------------------------------------


@_check("mini FL round (fedavg + multi_krum)")
def _():
    import numpy as np

    from ai.fl_core.fl_engine import LinearSoftmaxModel, fedavg, local_train, multi_krum
    from ai.training.poison import make_dataset

    rng = np.random.default_rng(42)
    X, y = make_dataset(200, 10, 3, seed=42)
    model = LinearSoftmaxModel(10, 3)
    params = model.get_params()

    updates = []
    for _ in range(5):
        idx = rng.choice(len(X), 40, replace=False)
        new_p = local_train(params, 10, 3, X[idx], y[idx], epochs=2, lr=0.1)
        updates.append(new_p - params)

    # FedAvg
    agg = fedavg(updates, [40] * 5)
    assert agg.shape == params.shape, "FedAvg output shape mismatch"

    # Multi-Krum
    agg_mk, selected = multi_krum(updates, num_malicious_assumed=1, num_to_select=4)
    assert agg_mk.shape == params.shape, "Multi-Krum output shape mismatch"
    assert len(selected) == 4, "Multi-Krum selection count wrong"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    print("SENTINEL-FL install check")
    print("=" * 50)

    passed = 0
    for name, fn in _CHECKS:
        try:
            fn()
            print(f"  [OK]   {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}")
            print(f"     {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            _FAILURES.append(name)

    print("=" * 50)
    total = len(_CHECKS)
    if _FAILURES:
        print(
            f"FAILED: {len(_FAILURES)}/{total} checks failed.",
            file=sys.stderr,
        )
        for f in _FAILURES:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"SENTINEL-FL install OK — all {total} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
