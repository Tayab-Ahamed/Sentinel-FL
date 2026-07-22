"""
poison.py — synthetic federated dataset + BadNets-style trigger injection.

Phase 0 development data (see DATASETS.md): Gaussian-blob multi-class data, partitioned
across simulated clients via a Dirichlet distribution for non-IID skew, matching the
partitioning strategy documented in DATASETS.md. This is swapped for the official
GSC26 dataset in Phase 1 without touching ai/fl_core or ai/detection (both operate on
generic (X, y) arrays).

Trigger: a fixed additive pattern added to a feature sub-block (the feature-vector
analogue of BadNets' bottom-right pixel-patch trigger, Gu et al. 2017 / see
BackdoorBench attack/badnet.py for the canonical image-domain version we mirror the
semantics of here), paired with a label flip to a fixed target class.

The pure functions (``make_dataset``, ``dirichlet_partition``, ``inject_trigger``,
``apply_trigger_to_all``) are the Phase 0 NumPy proof-of-concept and are PRESERVED
UNCHANGED.  ``BadNetsAttackSimulator`` wraps them in the AttackSimulator interface
(INTERFACES.md §AttackSimulator) for use with the config-driven evaluation pipeline.
"""
from __future__ import annotations

import logging

import numpy as np

from ai.fl_core.interfaces import AttackSimulator

logger = logging.getLogger(__name__)


def make_dataset(n_samples: int, n_features: int, n_classes: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(n_classes, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    X = centers[y] + rng.normal(0, 1.0, size=(n_samples, n_features))
    return X, y


def dirichlet_partition(n_samples: int, n_clients: int, y: np.ndarray, n_classes: int,
                         alpha: float = 0.5, seed: int = 0) -> list[np.ndarray]:
    """Returns a list of index arrays, one per client, non-IID via Dirichlet(alpha)."""
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        parts = np.split(idx_c, splits)
        for i, part in enumerate(parts):
            client_indices[i].extend(part.tolist())
    return [np.array(idx) for idx in client_indices]


def inject_trigger(X: np.ndarray, y: np.ndarray, target_class: int,
                    trigger_block: slice, trigger_value: float = 6.0,
                    poison_fraction: float = 0.2, seed: int = 0
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poison a fraction of (X, y) in place-style (returns new arrays). Returns
    (X_poisoned, y_poisoned, poisoned_mask)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    n_poison = int(n * poison_fraction)
    poison_idx = rng.choice(n, size=n_poison, replace=False)
    X_p = X.copy()
    y_p = y.copy()
    X_p[poison_idx, trigger_block] = trigger_value
    y_p[poison_idx] = target_class
    mask = np.zeros(n, dtype=bool)
    mask[poison_idx] = True
    return X_p, y_p, mask


def apply_trigger_to_all(X: np.ndarray, trigger_block: slice,
                          trigger_value: float = 6.0) -> np.ndarray:
    """Stamp every row with the trigger — used to build the ASR evaluation set."""
    X_t = X.copy()
    X_t[:, trigger_block] = trigger_value
    return X_t


# ---------------------------------------------------------------------------
# AttackSimulator implementation
# ---------------------------------------------------------------------------


class BadNetsAttackSimulator(AttackSimulator):
    """AttackSimulator for BadNets-style trigger injection across colluding clients.

    Mirrors the semantics of ``BackdoorBench/attack/badnet.py`` adapted for the
    federated, per-client setting described in INTERFACES.md §AttackSimulator.

    Args:
        target_class: The class the backdoor causes the model to predict.
        trigger_block: Slice of feature indices where the trigger is placed.
        trigger_value: Constant value stamped into the trigger block.
        poison_fraction: Fraction of each malicious client's data to poison.
        malicious_client_indices: Which client indices are malicious.
    """

    name: str = "badnet_colluding"

    def __init__(
        self,
        target_class: int = 0,
        trigger_block: slice = slice(0, 3),
        trigger_value: float = 6.0,
        poison_fraction: float = 0.15,
        malicious_client_indices: list[int] | None = None,
    ) -> None:
        self._target_class = target_class
        self._trigger_block = trigger_block
        self._trigger_value = trigger_value
        self._poison_fraction = poison_fraction
        self._malicious = set(malicious_client_indices or [2, 5, 9])

    @classmethod
    def from_config(cls, config: object) -> BadNetsAttackSimulator:
        """Build a BadNetsAttackSimulator from a Configuration object."""
        attack = getattr(config, "attack", None)
        if attack is None:
            return cls()
        return cls(
            target_class=attack.target_class,
            trigger_block=slice(attack.trigger_block_start, attack.trigger_block_end),
            trigger_value=attack.trigger_value,
            poison_fraction=attack.poison_fraction,
            malicious_client_indices=list(attack.malicious_client_indices),
        )

    def poison_client_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: str,
        round_num: int,
        config: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject trigger into a fraction of this client's data if malicious.

        For honest clients, returns data unchanged and an all-False mask.

        Args:
            X: Feature matrix for this client.
            y: Label vector for this client.
            client_id: Client identifier string (e.g. ``"client_02"``).
            round_num: Current FL round number (used for per-round seed variation).
            config: Active Configuration (not used by this basic injector).

        Returns:
            ``(X_poisoned, y_poisoned, mask)`` where ``mask`` is bool array.
        """
        # Extract numeric index from client_id string (e.g. "client_02" -> 2)
        try:
            cid_int = int(client_id.split("_")[-1])
        except (ValueError, IndexError):
            cid_int = hash(client_id) % 1000

        if cid_int not in self._malicious or len(X) <= 5:
            # Honest client or too few samples — return data unchanged
            mask = np.zeros(len(X), dtype=bool)
            return X.copy(), y.copy(), mask

        seed = 100 + round_num * 10 + cid_int
        X_p, y_p, mask = inject_trigger(
            X, y,
            target_class=self._target_class,
            trigger_block=self._trigger_block,
            trigger_value=self._trigger_value,
            poison_fraction=self._poison_fraction,
            seed=seed,
        )
        logger.debug(
            "BadNetsAttackSimulator: poisoned %d/%d samples for client %s round %d",
            int(mask.sum()), len(X), client_id, round_num,
        )
        return X_p, y_p, mask

    def build_trigger_eval_set(self, X_clean: np.ndarray) -> np.ndarray:
        """Stamp the trigger onto every row for ASR evaluation.

        Args:
            X_clean: Clean evaluation features, shape ``(n, n_features)``.

        Returns:
            ``X_triggered`` — same shape with trigger block set to ``trigger_value``.
        """
        return apply_trigger_to_all(X_clean, self._trigger_block, self._trigger_value)
