"""
ai/training/partitioning.py — IID and non-IID client partitioning algorithms.

Two standalone partitioner classes with no dataset coupling.  Both accept raw
``(X, y)`` arrays and return lists of index arrays — the caller decides whether
to copy the data or keep index-based access.

IID partitioning:
  Randomly shuffle all indices then split into equal-sized chunks.
  Each client gets roughly ``N // n_clients`` samples.

Non-IID (Dirichlet) partitioning:
  For each class, draw proportions from Dirichlet(alpha) and split that
  class's indices across clients accordingly.  Small alpha (e.g. 0.1) gives
  extreme label skew; large alpha (e.g. 100) approximates IID.

Both classes are deterministic given the same ``seed``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

# Minimum samples a partition must contain.  If Dirichlet draws give a client
# zero samples for every class, we guarantee at least this many by random
# redistribution.
_MIN_SAMPLES_PER_CLIENT = 1


class BasePartitioner(ABC):
    """Abstract base for all partitioning strategies."""

    @abstractmethod
    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_clients: int,
        seed: int = 42,
    ) -> list[np.ndarray]:
        """Partition ``(X, y)`` into ``n_clients`` index arrays.

        Args:
            X: Feature array (N, ...).
            y: Label array (N,).
            n_clients: Number of clients.
            seed: Random seed for reproducibility.

        Returns:
            List of ``n_clients`` index arrays (``np.ndarray`` of dtype int64).
            The union of all index arrays equals ``{0, ..., N-1}``.
        """


class IIDPartitioner(BasePartitioner):
    """Uniform random IID partitioning.

    Each client receives approximately ``N // n_clients`` samples chosen
    uniformly at random without replacement.  If ``N`` is not exactly
    divisible by ``n_clients``, the first ``N % n_clients`` clients each
    receive one extra sample.

    Args:
        shuffle: If ``True`` (default), shuffle before splitting.  Setting to
            ``False`` is only useful for unit tests that need predictable splits.
    """

    def __init__(self, shuffle: bool = True) -> None:
        self._shuffle = shuffle

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_clients: int,
        seed: int = 42,
    ) -> list[np.ndarray]:
        """Return equal-sized IID index splits.

        Args:
            X: Feature array (used only for ``len``).
            y: Label array (unused — labels don't affect IID splitting).
            n_clients: Number of clients.
            seed: Random seed.

        Returns:
            List of ``n_clients`` index arrays.
        """
        n = len(X)
        _validate_inputs(n, n_clients)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n) if self._shuffle else np.arange(n, dtype=np.int64)
        splits = list(np.array_split(indices, n_clients))

        sizes = [len(s) for s in splits]
        logger.info(
            "IIDPartitioner: %d clients, sizes min=%d max=%d (seed=%d)",
            n_clients,
            min(sizes),
            max(sizes),
            seed,
        )
        return splits


class DirichletPartitioner(BasePartitioner):
    """Non-IID partitioning via Dirichlet label distribution.

    For each label class, samples a Dirichlet(alpha) vector of proportions
    and assigns that class's indices to clients accordingly.  Guarantees at
    least ``_MIN_SAMPLES_PER_CLIENT`` samples per client by redistributing
    any clients that end up empty.

    Args:
        alpha: Dirichlet concentration parameter.
            - ``alpha < 1``: Highly skewed (each client mostly one class).
            - ``alpha = 1``: Uniform on the simplex.
            - ``alpha >> 1``: Approximates IID.
        n_classes: Number of label classes.  Auto-detected from ``y`` if
            ``None`` is passed to ``partition()``.
    """

    def __init__(self, alpha: float = 0.5, n_classes: int | None = None) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self._alpha = alpha
        self._n_classes = n_classes

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_clients: int,
        seed: int = 42,
    ) -> list[np.ndarray]:
        """Return Dirichlet non-IID index splits.

        Args:
            X: Feature array (used only for ``len``).
            y: Label array — **must** be integer-typed.
            n_clients: Number of clients.
            seed: Random seed (offset by 13 internally to differ from data seed).

        Returns:
            List of ``n_clients`` index arrays (dtype int64).
        """
        n = len(X)
        _validate_inputs(n, n_clients)

        n_classes = self._n_classes if self._n_classes is not None else int(y.max()) + 1
        rng = np.random.default_rng(seed + 13)  # offset so partition seed ≠ data seed

        client_indices: list[list[int]] = [[] for _ in range(n_clients)]

        for cls in range(n_classes):
            idx_cls = np.where(y == cls)[0]
            if len(idx_cls) == 0:
                continue
            rng.shuffle(idx_cls)
            proportions = rng.dirichlet(self._alpha * np.ones(n_clients))
            splits = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]
            for client_i, part in enumerate(np.split(idx_cls, splits)):
                client_indices[client_i].extend(part.tolist())

        result = [np.array(idx, dtype=np.int64) for idx in client_indices]

        # ── Guarantee minimum samples per client ──────────────────────────
        result = _ensure_minimum(result, n, rng, _MIN_SAMPLES_PER_CLIENT)

        sizes = [len(r) for r in result]
        logger.info(
            "DirichletPartitioner: %d clients, alpha=%.3f, sizes min=%d max=%d (seed=%d)",
            n_clients,
            self._alpha,
            min(sizes),
            max(sizes),
            seed,
        )
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _validate_inputs(n: int, n_clients: int) -> None:
    """Raise ValueError for degenerate inputs."""
    if n_clients <= 0:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if n == 0:
        raise ValueError("Dataset is empty (n=0).")
    if n < n_clients:
        raise ValueError(
            f"Cannot partition {n} samples into {n_clients} clients "
            f"(need at least 1 sample per client)."
        )


def _ensure_minimum(
    partitions: list[np.ndarray],
    n_total: int,
    rng: np.random.Generator,
    min_samples: int,
) -> list[np.ndarray]:
    """Redistribute samples so no client partition is below ``min_samples``.

    Takes samples from the largest partitions and gives them to empty ones.
    This is a safety net for extreme Dirichlet draws — it should not affect
    normal runs.

    Args:
        partitions: Current list of index arrays.
        n_total: Total number of samples.
        rng: Random generator for shuffling donor choices.
        min_samples: Minimum required samples per partition.

    Returns:
        Updated list of index arrays.
    """
    all_used: set[int] = set()
    for p in partitions:
        all_used.update(p.tolist())

    for i, part in enumerate(partitions):
        if len(part) < min_samples:
            # Find donor (largest partition with surplus)
            donor_idx = max(
                range(len(partitions)),
                key=lambda j: len(partitions[j]) if j != i else 0,
            )
            donor = partitions[donor_idx]
            if len(donor) <= min_samples:
                logger.warning(
                    "Cannot guarantee min_samples=%d for client %d "
                    "(dataset too small for %d clients).",
                    min_samples,
                    i,
                    len(partitions),
                )
                continue
            take = min_samples - len(part)
            transferred = donor[-take:]
            partitions[donor_idx] = donor[:-take]
            partitions[i] = np.concatenate([part, transferred])
            logger.debug(
                "ensure_minimum: transferred %d samples from client %d to client %d",
                take,
                donor_idx,
                i,
            )
    return partitions
