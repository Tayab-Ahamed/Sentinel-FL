"""
ai/fl_core/interfaces.py — Abstract base classes (contracts) for SENTINEL-FL.

Every interface here corresponds exactly to a section in INTERFACES.md.
Concrete implementations register themselves in configs/registry.yaml.

Language contract: all ABCs use Python's ``abc`` module so instantiating an
incomplete subclass raises TypeError at import time rather than at first call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Forward-declared type aliases (avoid circular imports with schemas.py)
# ---------------------------------------------------------------------------
# These are string annotations resolved at runtime via TYPE_CHECKING or
# direct import inside the concrete implementations.

ClientUpdate = Any  # ai.fl_core.schemas.ClientUpdate
DetectionResult = Any  # ai.fl_core.schemas.DetectionResult
TrustLedgerEntry = Any  # ai.fl_core.schemas.TrustLedgerEntry
ModelMetadata = Any  # ai.fl_core.schemas.ModelMetadata
EvaluationResult = Any  # ai.fl_core.schemas.EvaluationResult
Configuration = Any  # ai.fl_core.schemas.Configuration
AggregationResult = Any  # dict returned by Aggregator.aggregate


# ---------------------------------------------------------------------------
# Detector  (INTERFACES.md §Detector)
# ---------------------------------------------------------------------------


class Detector(ABC):
    """Contract implemented by L2 (Model Auditor) and L3 (Runtime Sentinel) detectors.

    Every concrete detector (STRIP, Neural-Cleanse-style, activation-consistency)
    must implement this interface.  All three methods must be present before the
    object can be instantiated.

    Failure contract:
        - calibration data too small → raise InsufficientCalibrationDataError
        - model incompatible → raise UnsupportedModelError *at registration time*,
          not at score time
    """

    #: Unique identifier, e.g. ``"strip_entropy"``.  Must match the key in registry.yaml.
    name: str
    #: Which defense layer this detector belongs to.
    layer: str  # "L2" | "L3"

    @abstractmethod
    def calibrate(self, clean_reference_data: Any) -> Any:
        """Estimate a detection boundary from clean data only (no trojaned samples needed).

        Args:
            clean_reference_data: Clean validation data (format depends on detector).

        Returns:
            An opaque, JSON-serializable calibration state (e.g. an entropy boundary float).

        Raises:
            InsufficientCalibrationDataError: If the clean set is too small.
        """

    @abstractmethod
    def score(self, input_or_model: Any, calibration_state: Any) -> DetectionResult:
        """Score one input (L3) or one label (L2) against the calibrated boundary.

        Args:
            input_or_model:
                L3 detectors receive a single input tensor.
                L2 detectors receive a trained model + label to audit.
            calibration_state: The value previously returned by ``calibrate()``.

        Returns:
            A DetectionResult (ai.fl_core.schemas.DetectionResult).
        """

    @abstractmethod
    def explain(self, detection_result: DetectionResult) -> str:
        """Return a human-readable reason string for the Trust Ledger.

        Args:
            detection_result: The result to explain.

        Returns:
            A concise, human-readable explanation string.
        """


# ---------------------------------------------------------------------------
# Aggregator  (INTERFACES.md §Aggregator)
# ---------------------------------------------------------------------------


class Aggregator(ABC):
    """Contract implemented by L1's base robust-aggregation rule.

    Wraps or subclasses Flower's strategy classes for the production path;
    the NumPy reference in ``ai/fl_core/fl_engine.py`` implements this same
    contract independently.

    Failure contract:
        - fewer than ``min_clients`` updates received → raise InsufficientClientsError
    """

    #: Unique identifier matching registry.yaml.
    name: str

    @abstractmethod
    def aggregate(self, client_updates: list[ClientUpdate]) -> AggregationResult:
        """Aggregate client updates into a new global model delta.

        Returns a dict with at least:
            - ``"delta"`` (list[float]): the aggregated parameter delta
            - ``"selected_indices"`` (list[int]): client indices included
            - ``"excluded_indices"`` (list[int]): client indices excluded

        Raises:
            InsufficientClientsError: If too few updates are provided.
        """


# ---------------------------------------------------------------------------
# DefenseStrategy  (INTERFACES.md §DefenseStrategy)
# ---------------------------------------------------------------------------


class DefenseStrategy(ABC):
    """Umbrella contract implemented by each defense layer (L1, L2, L3).

    Composes one or more Detectors/Aggregators and writes results to the
    Trust Ledger.  This is the extension point for adding a new layer without
    touching the other three layers.
    """

    #: Layer identifier — ``"L1"``, ``"L2"``, or ``"L3"``.
    layer_id: str

    @abstractmethod
    def process(self, context: Any) -> list[TrustLedgerEntry]:
        """Execute the defense logic for one round or one inference.

        Args:
            context:
                L1/L2 receive a ``RoundContext`` (per-round server-side data).
                L3 receives an ``InferenceContext`` (single input at deploy time).

        Returns:
            A list of TrustLedgerEntry objects to be written to L4.
        """


# ---------------------------------------------------------------------------
# AttackSimulator  (INTERFACES.md §AttackSimulator)
# ---------------------------------------------------------------------------


class AttackSimulator(ABC):
    """Contract for generating poisoned training data in evaluation runs.

    Used in ``experiments/`` to inject backdoors per client.  The mask returned
    by ``poison_client_data`` is ground truth and is **never** exposed to the
    defense pipeline during training.
    """

    #: Unique identifier, e.g. ``"badnet_colluding"``.
    name: str

    @abstractmethod
    def poison_client_data(
        self,
        X: Any,
        y: Any,
        client_id: str,
        round_num: int,
        config: Configuration,
    ) -> tuple[Any, Any, Any]:
        """Inject a backdoor trigger into a fraction of the client's local data.

        Args:
            X: Feature matrix for this client.
            y: Label vector for this client.
            client_id: Identifier of the client being poisoned.
            round_num: Current FL round (allows round-varying attacks).
            config: Active Configuration object.

        Returns:
            ``(X_poisoned, y_poisoned, mask)`` where ``mask`` is a boolean
            array marking which rows were modified.
        """

    @abstractmethod
    def build_trigger_eval_set(self, X_clean: Any) -> Any:
        """Stamp the trigger onto every row of ``X_clean``.

        Used to compute Attack Success Rate (ASR) against the current global model.

        Args:
            X_clean: Clean evaluation features.

        Returns:
            ``X_triggered`` — same shape as ``X_clean`` with trigger applied.
        """


# ---------------------------------------------------------------------------
# ModelRegistry  (INTERFACES.md §ModelRegistry)
# ---------------------------------------------------------------------------


class ModelRegistry(ABC):
    """Contract for storing and retrieving model checkpoints across FL rounds.

    Enables L2 audits to reference "the model as of round N" and supports
    the Remediation Engine's rollback mitigation path.
    """

    @abstractmethod
    def save(self, round_num: int, model_state: Any, metadata: ModelMetadata) -> str:
        """Persist a model checkpoint.

        Args:
            round_num: The FL round that produced this model.
            model_state: Serializable model weights (numpy array or PyTorch state_dict).
            metadata: ModelMetadata for this checkpoint.

        Returns:
            The ``model_id`` (UUID string) of the saved checkpoint.
        """

    @abstractmethod
    def load(self, model_id: str) -> tuple[Any, ModelMetadata]:
        """Load a checkpoint by ID.

        Returns:
            ``(model_state, ModelMetadata)``

        Raises:
            CheckpointNotFoundError: If ``model_id`` does not exist.
        """

    @abstractmethod
    def latest(self) -> str:
        """Return the ``model_id`` of the most recently saved checkpoint.

        Raises:
            CheckpointNotFoundError: If no checkpoints exist yet.
        """

    @abstractmethod
    def rollback_to(self, round_num: int) -> str:
        """Return the ``model_id`` of the checkpoint at ``round_num``.

        The caller must fall back to the nearest earlier checkpoint explicitly
        if the exact round has no checkpoint; never silently pick another round.

        Raises:
            CheckpointNotFoundError: If no checkpoint exists for ``round_num``.
        """


# ---------------------------------------------------------------------------
# DatasetLoader  (INTERFACES.md §DatasetLoader)
# ---------------------------------------------------------------------------


class DatasetLoader(ABC):
    """Abstracts Phase 0 synthetic data vs. Phase 1 official challenge data.

    ``ai/fl_core`` and ``ai/detection`` must never special-case which phase is
    active — they call this interface and remain agnostic.
    """

    @abstractmethod
    def load_client_partitions(
        self, n_clients: int, config: Configuration
    ) -> list[tuple[Any, Any]]:
        """Return a list of ``(X, y)`` tuples, one per client.

        Uses Dirichlet non-IID partitioning in Phase 0.
        """

    @abstractmethod
    def load_clean_holdout(self) -> tuple[Any, Any]:
        """Return ``(X, y)`` for the server-side clean validation set.

        Must never contain poisoned samples; kept separate from all client partitions.
        Used by L2 Model Auditor and L3 calibration.
        """

    @abstractmethod
    def load_evaluation_set(self) -> tuple[Any, Any]:
        """Return ``(X, y)`` for the held-out evaluation set.

        Never seen during training; used only for final metric computation.
        """


# ---------------------------------------------------------------------------
# Logger  (INTERFACES.md §Logger)
# ---------------------------------------------------------------------------


class Logger(ABC):
    """Structured JSON-lines logger shared by all layers.

    Failure contract: ``log()`` must never raise into the calling layer's control
    flow.  Wrap in try/except at the boundary; drop-and-count on failure rather
    than crash a training round over a logging error (ARCHITECTURE.md §7.8).
    """

    @abstractmethod
    def log(self, layer_id: str, event_type: str, payload: dict[str, Any]) -> None:
        """Emit one structured log entry.

        Args:
            layer_id: Which defense layer is logging (``"L1"``, ``"L2"``, etc.).
            event_type: Short snake_case event name (``"client_excluded"``, etc.).
            payload: Event-specific data; see SCHEMAS.md §LogEntry.
        """


# ---------------------------------------------------------------------------
# MetricsCollector  (INTERFACES.md §MetricsCollector)
# ---------------------------------------------------------------------------


class MetricsCollector(ABC):
    """Computes the standard metric set from logged rounds/detections."""

    @abstractmethod
    def compute(self, experiment_id: str) -> EvaluationResult:
        """Compute all metrics for a completed experiment.

        Reads Logger output for ``experiment_id``.  Missing expected log events
        → EvaluationResult fields populated as ``null`` with a ``warnings`` list
        naming which metrics couldn't be computed (ARCHITECTURE.md §7.9).

        Args:
            experiment_id: Identifier of the experiment to evaluate.

        Returns:
            An EvaluationResult (ai.fl_core.schemas.EvaluationResult).
        """


# ---------------------------------------------------------------------------
# Visualizer  (INTERFACES.md §Visualizer)
# ---------------------------------------------------------------------------


class Visualizer(ABC):
    """Backend-to-dashboard boundary (ARCHITECTURE.md §5, API.md)."""

    @abstractmethod
    def reputation_heatmap(self, experiment_id: str) -> dict[str, Any]:
        """Return client × round trust-score matrix for the dashboard heatmap.

        See API.md §4.
        """

    @abstractmethod
    def metric_timeseries(self, experiment_id: str, metric_names: list[str]) -> dict[str, Any]:
        """Return time-series data for the requested metrics.

        See API.md §5.
        """

    @abstractmethod
    def audit_report(self, experiment_id: str, round_num: int) -> dict[str, Any]:
        """Return the L2 audit report for a given round.

        See API.md §6.
        """

    @abstractmethod
    def explainability_drilldown(self, trust_ledger_entry_id: str) -> dict[str, Any]:
        """Return the human-readable reason and raw evidence for a specific flag.

        See API.md §7.
        """
