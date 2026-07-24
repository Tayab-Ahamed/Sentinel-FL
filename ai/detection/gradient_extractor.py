"""
ai/detection/gradient_extractor.py — Gradient / parameter-delta extraction.

Centralises the "Flower FitRes → flat delta vector" transformation so every
downstream component (norm calculator, cosine similarity, anomaly detector)
all work from the same canonical representation.

A *delta* is the signed difference between a client's updated parameters and
the previous global parameters: ``delta_i = params_new_i − params_prev``.
Positive entries indicate the client increased those weights; negative entries
indicate a decrease.  The direction and magnitude of this vector are the
primary signals for L1 anomaly detection.

Design notes:
  - All operations return new arrays; inputs are never mutated.
  - Flat 1-D representation is used throughout L1 (no layer-wise breakdown
    needed at this stage).
  - Flower ``NDArrays`` (list of layer arrays) are always flattened and
    concatenated in the same deterministic order.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Supported norm types
NormType = str  # "l2" | "l1" | "linf"


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------


def extract_delta(
    prev_params: list[np.ndarray],
    new_params: list[np.ndarray],
) -> np.ndarray:
    """Compute the flat parameter delta: ``new − prev``.

    Args:
        prev_params: Previous global model parameters as a list of layer arrays
            (Flower ``NDArrays`` format).
        new_params: Updated client parameters in the same format.

    Returns:
        1-D float32 array of shape ``(total_params,)``.

    Raises:
        ValueError: If the two parameter lists have different structure.
    """
    if len(prev_params) != len(new_params):
        raise ValueError(
            f"Parameter list length mismatch: prev={len(prev_params)}, new={len(new_params)}"
        )
    diff_flat = np.concatenate(
        [
            (n.astype(np.float32) - p.astype(np.float32)).ravel()
            for p, n in zip(prev_params, new_params)
        ]
    )
    return diff_flat


def extract_all_deltas(
    prev_params: list[np.ndarray],
    results: list[tuple[object, object]],  # list[(ClientProxy, FitRes)]
) -> list[np.ndarray]:
    """Extract delta vectors for all clients from a Flower ``aggregate_fit`` result list.

    Args:
        prev_params: Previous global model parameters (``NDArrays`` format).
        results: List of ``(ClientProxy, FitRes)`` pairs from Flower.

    Returns:
        List of flat float32 delta arrays, one per client in the same order
        as ``results``.
    """
    from flwr.common import parameters_to_ndarrays

    deltas: list[np.ndarray] = []
    for _proxy, fit_res in results:
        try:
            new_params = parameters_to_ndarrays(fit_res.parameters)
            delta = extract_delta(prev_params, new_params)
            deltas.append(delta)
        except Exception as exc:
            logger.warning("GradientExtractor: failed to extract delta for a client: %s", exc)
            # Insert a zero delta as placeholder so index alignment is preserved
            total = sum(p.size for p in prev_params)
            deltas.append(np.zeros(total, dtype=np.float32))
    return deltas


def flatten_params(params: list[np.ndarray]) -> np.ndarray:
    """Flatten a list of parameter arrays into a single 1-D float32 vector.

    Args:
        params: Flower ``NDArrays`` (list of layer arrays).

    Returns:
        1-D float32 array.
    """
    return np.concatenate([p.astype(np.float32).ravel() for p in params])


def normalize_delta(
    delta: np.ndarray,
    norm_type: NormType = "l2",
) -> np.ndarray:
    """Return a unit-normalized copy of a delta vector.

    Args:
        delta: 1-D float32 delta array.
        norm_type: Normalization type — ``"l2"``, ``"l1"``, or ``"linf"``.

    Returns:
        Normalized 1-D float32 array.  Returns a zero vector if the norm is
        effectively zero (norm < 1e-12).
    """
    if norm_type == "l2":
        n = float(np.linalg.norm(delta))
    elif norm_type == "l1":
        n = float(np.sum(np.abs(delta)))
    elif norm_type == "linf":
        n = float(np.max(np.abs(delta)))
    else:
        raise ValueError(f"Unknown norm_type: {norm_type!r}. Use 'l2', 'l1', or 'linf'.")

    if n < 1e-12:
        logger.debug("normalize_delta: near-zero norm (%.2e); returning zero vector.", n)
        return np.zeros_like(delta, dtype=np.float32)
    return (delta / n).astype(np.float32)


def normalize_all_deltas(
    deltas: list[np.ndarray],
    norm_type: NormType = "l2",
) -> list[np.ndarray]:
    """Normalize a list of delta vectors.

    Args:
        deltas: Per-client delta vectors.
        norm_type: Normalization type.

    Returns:
        List of normalized delta vectors.
    """
    return [normalize_delta(d, norm_type) for d in deltas]


# ---------------------------------------------------------------------------
# GradientExtractor class (stateful wrapper — holds prev_params across rounds)
# ---------------------------------------------------------------------------


class GradientExtractor:
    """Stateful extractor that tracks global parameters across FL rounds.

    Maintains a copy of the previous round's global parameters so that
    ``extract_round_deltas()`` can be called after each ``aggregate_fit``
    without the caller having to manage state.

    Args:
        initial_params: Initial global model parameters (round 0 / pre-training).
        norm_type: Default norm type for normalization operations.
    """

    def __init__(
        self,
        initial_params: list[np.ndarray],
        norm_type: NormType = "l2",
    ) -> None:
        self._prev_params: list[np.ndarray] = [p.copy() for p in initial_params]
        self._norm_type = norm_type
        self._total_params = sum(p.size for p in initial_params)
        logger.debug("GradientExtractor: initialized with %d total parameters.", self._total_params)

    def extract_round_deltas(
        self,
        results: list[tuple[object, object]],  # list[(ClientProxy, FitRes)]
    ) -> list[np.ndarray]:
        """Extract per-client delta vectors for the current round.

        Must be called BEFORE ``update_params()``.

        Args:
            results: Flower ``(ClientProxy, FitRes)`` pairs.

        Returns:
            List of flat delta vectors (one per client).
        """
        return extract_all_deltas(self._prev_params, results)

    def update_params(self, new_params: list[np.ndarray]) -> None:
        """Update the stored global parameters after aggregation.

        Must be called AFTER ``extract_round_deltas()`` for the same round.

        Args:
            new_params: New global model parameters after aggregation.
        """
        self._prev_params = [p.copy() for p in new_params]
        logger.debug("GradientExtractor: global params updated for next round.")

    @property
    def prev_params(self) -> list[np.ndarray]:
        """The stored previous round's global parameters."""
        return self._prev_params

    @property
    def total_params(self) -> int:
        """Total number of scalar parameters in the model."""
        return self._total_params
