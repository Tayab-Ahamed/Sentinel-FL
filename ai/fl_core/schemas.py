"""
ai/fl_core/schemas.py — Pydantic v2 data models for SENTINEL-FL.

Every schema here corresponds exactly to a table in SCHEMAS.md.  All objects
are JSON-serializable and used in Logger payloads, API responses, and the
Trust Ledger.  Field-level validation rules are taken directly from SCHEMAS.md.

Import convention:
    from ai.fl_core.schemas import ClientUpdate, ModelMetadata, ...
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


def _new_id() -> str:
    """Return a new UUID4 string."""
    return str(uuid4())


# ---------------------------------------------------------------------------
# ClientUpdate  (SCHEMAS.md §ClientUpdate)
# ---------------------------------------------------------------------------


class ClientUpdate(BaseModel):
    """Produced by a client each FL round; consumed by L1 (Aggregator, Update Guard)."""

    client_id: str = Field(..., description="Unique client identifier.")
    round_num: int = Field(..., ge=0, description="FL round number.")
    delta: list[float] = Field(..., min_length=1, description="Flattened model parameter delta.")
    n_samples: int = Field(..., gt=0, description="Local dataset size used for weighting.")
    timestamp: str = Field(default_factory=_now_iso, description="ISO-8601 UTC timestamp.")
    signature: str | None = Field(None, description="Reserved for Security Layer (Phase 2+).")

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# ModelMetadata  (SCHEMAS.md §ModelMetadata)
# ---------------------------------------------------------------------------


class ModelMetadata(BaseModel):
    """Attached to every checkpoint in the Model Registry."""

    model_id: str = Field(default_factory=_new_id, description="UUID for this checkpoint.")
    round_num: int = Field(..., ge=0)
    architecture: str = Field(..., description="e.g. 'linear_softmax_v0', 'resnet18_cifar'.")
    parent_model_id: str | None = Field(None, description="Parent checkpoint for rollback lineage.")
    clean_accuracy: float | None = Field(None, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# AttackReport  (SCHEMAS.md §AttackReport)
# ---------------------------------------------------------------------------


class AttackReport(BaseModel):
    """Ground-truth record produced by an AttackSimulator.

    Never exposed to the defense pipeline during training; used only to score
    detectors after the fact (see INTERFACES.md#AttackSimulator).
    """

    attack_id: str = Field(default_factory=_new_id)
    attack_type: str = Field(..., description="e.g. 'badnet_colluding'.")
    malicious_client_ids: list[str] = Field(..., description="Ground-truth malicious clients.")
    target_class: int = Field(..., ge=0)
    poison_fraction: float = Field(..., ge=0.0, le=1.0)
    rounds_active: list[int] = Field(..., description="Rounds during which the attack was active.")


# ---------------------------------------------------------------------------
# DetectionResult  (SCHEMAS.md §DetectionResult)
# ---------------------------------------------------------------------------


class DetectionResult(BaseModel):
    """Returned by a Detector.score(...) call (L2 or L3)."""

    detector_name: str = Field(..., description="e.g. 'strip_entropy'.")
    layer: Literal["L2", "L3"] = Field(..., description="Which defense layer produced this.")
    subject_id: str = Field(..., description="Input ID (L3) or label ID (L2).")
    score: float = Field(..., description="Raw detector score (e.g. entropy value, L1 norm).")
    flagged: bool = Field(..., description="True if score crosses the calibrated boundary.")
    boundary: float = Field(..., description="Calibration threshold used.")
    round_num: int | None = Field(None, description="Null for L3 (per-inference, not per-round).")
    explanation: str = Field(
        "", description="Human-readable reason populated by Detector.explain()."
    )


# ---------------------------------------------------------------------------
# TrustScore  (SCHEMAS.md §TrustScore)
# ---------------------------------------------------------------------------


class TrustScore(BaseModel):
    """L4's per-client and per-label running reputation score."""

    subject_type: Literal["client", "label"] = Field(...)
    subject_id: str = Field(...)
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0 = fully trusted, 1 = fully flagged. Decayed over rounds.",
    )
    last_updated_round: int = Field(...)
    contributing_events: list[str] = Field(
        default_factory=list,
        description="TrustLedgerEntry.entry_id values that fed this score.",
    )


# ---------------------------------------------------------------------------
# TrainingRound  (SCHEMAS.md §TrainingRound)
# ---------------------------------------------------------------------------


class TrainingRound(BaseModel):
    """One record per FL round — the backbone of the dashboard timeline."""

    round_num: int = Field(..., ge=0)
    participating_clients: list[str] = Field(...)
    excluded_clients: list[str] = Field(
        default_factory=list,
        description="Excluded by L1 aggregator (e.g. Multi-Krum).",
    )
    flagged_clusters: list[list[str]] = Field(
        default_factory=list,
        description="L1 collusion clusters; each inner list is a group of client IDs.",
    )
    global_model_id: str = Field(..., description="Resulting ModelMetadata.model_id.")
    clean_accuracy: float | None = Field(None, ge=0.0, le=1.0)
    attack_success_rate: float | None = Field(None, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# AuditReport  (SCHEMAS.md §AuditReport)
# ---------------------------------------------------------------------------


class ReversedTrigger(BaseModel):
    """Representation of a reversed trigger produced by L2 per flagged label."""

    label: int = Field(..., ge=0)
    trigger_representation: Any = Field(
        ...,
        description="Numpy array / tensor serialized as a nested list (JSON-safe).",
    )
    l1_norm: float = Field(..., ge=0.0)


class AuditReport(BaseModel):
    """Produced by L2 (Model Auditor) every N rounds."""

    audit_id: str = Field(default_factory=_new_id)
    round_num: int = Field(..., ge=0, description="Which global model was audited.")
    per_label_results: list[DetectionResult] = Field(
        ..., description="One DetectionResult per audited label."
    )
    flagged_labels: list[int] = Field(
        default_factory=list, description="Labels exceeding the anomaly threshold."
    )
    reversed_triggers: list[ReversedTrigger] = Field(
        default_factory=list, description="Reversed trigger for each flagged label."
    )


# ---------------------------------------------------------------------------
# EvaluationResult  (SCHEMAS.md §EvaluationResult)
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Output of MetricsCollector.compute(...) — the standard metric set.

    Fields added in Milestone 9 (all nullable — existing consumers unaffected):
      precision, recall, f1_score, false_positive_rate,
      runtime_seconds, peak_memory_mb.
    """

    experiment_id: str = Field(...)
    # --- Original SCHEMAS.md fields ---
    clean_accuracy: float | None = Field(None, ge=0.0, le=1.0)
    attack_success_rate: float | None = Field(None, ge=0.0, le=1.0)
    robust_accuracy: float | None = Field(None, ge=0.0, le=1.0)
    false_acceptance_rate: float | None = Field(
        None, ge=0.0, le=1.0, description="L3, on held-out trojaned set."
    )
    false_rejection_rate: float | None = Field(
        None, ge=0.0, le=1.0, description="L3, on held-out clean set."
    )
    detection_latency_ms: float | None = Field(None, ge=0.0, description="L3, per-input.")
    communication_cost_bytes: int | None = Field(
        None, ge=0, description="Total bytes transferred across all rounds."
    )
    # --- Milestone 9 additions ---
    precision: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Detection precision: TP / (TP + FP) across all layers.",
    )
    recall: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Detection recall: TP / (TP + FN) across all layers.",
    )
    f1_score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Harmonic mean of precision and recall.",
    )
    false_positive_rate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="FP / (FP + TN) — clean clients/inputs incorrectly flagged.",
    )
    runtime_seconds: float | None = Field(
        None,
        ge=0.0,
        description="Wall-clock runtime of the full experiment (seconds).",
    )
    peak_memory_mb: float | None = Field(
        None,
        ge=0.0,
        description="Peak RSS memory usage during the experiment (MB).",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Metrics that could not be computed; populated with null values.",
    )


# ---------------------------------------------------------------------------
# BaselineComparison  (Milestone 9)
# ---------------------------------------------------------------------------


class BaselineComparison(BaseModel):
    """Comparison of one EvaluationResult against a named baseline method.

    ``delta_metrics`` = our value − baseline value (positive = better for
    accuracy-style metrics; negative = better for error-rate-style metrics).
    ``improvement_percent`` = delta / |baseline| * 100 where baseline != 0.
    """

    baseline_name: str = Field(
        ..., description="Name of the baseline method (key in baselines.yaml)."
    )
    baseline_description: str = Field("", description="Human-readable baseline description.")
    baseline_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw baseline metric values from baselines.yaml.",
    )
    delta_metrics: dict[str, float | None] = Field(
        default_factory=dict,
        description="our_value − baseline_value per metric (None if either is None).",
    )
    improvement_percent: dict[str, float | None] = Field(
        default_factory=dict,
        description="(delta / |baseline|) * 100 per metric (None if baseline == 0 or None).",
    )
    verdict: str = Field(
        ...,
        description="'better', 'worse', or 'mixed' based on key metric deltas.",
    )
    summary: str = Field("", description="Human-readable comparison narrative.")
    generated_at: str = Field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# BenchmarkReport  (Milestone 9)
# ---------------------------------------------------------------------------


class BenchmarkReport(BaseModel):
    """Full structured benchmark output for one experiment.

    Contains the evaluation result, per-round metric breakdown, per-layer
    detection summary, optional baseline comparison, and inline chart artifacts.
    Written as both JSON and Markdown by BenchmarkReporter.
    """

    report_id: str = Field(default_factory=_new_id)
    experiment_id: str = Field(...)
    generated_at: str = Field(default_factory=_now_iso)
    evaluation_result: EvaluationResult = Field(...)
    per_round_metrics: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per-round metric breakdown. Each dict: round_num, clean_accuracy, "
            "attack_success_rate, n_l1_flags, n_l2_flags, n_l3_flags."
        ),
    )
    detection_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-layer flag totals and detection rates.",
    )
    baseline_comparison: BaselineComparison | None = Field(
        None, description="Comparison against a named baseline (optional)."
    )
    chart_artifacts: list[ChartArtifact] = Field(
        default_factory=list,
        description="Inline charts generated for this report.",
    )


# ---------------------------------------------------------------------------
# Experiment  (SCHEMAS.md §Experiment)
# ---------------------------------------------------------------------------


class Experiment(BaseModel):
    """Top-level record tying a config + dataset + attack + defense combination together."""

    experiment_id: str = Field(default_factory=_new_id)
    config_ref: str = Field(..., description="Path under configs/ for the YAML used.")
    dataset_phase: Literal["phase0_synthetic", "phase1_official"] = Field(...)
    layers_enabled: list[Literal["L1", "L2", "L3"]] = Field(
        ...,
        description="Active defense layers. L4 is always on (passive logging).",
    )
    attack_config: AttackReport = Field(...)
    result: EvaluationResult | None = Field(
        None, description="Filled once the experiment completes."
    )
    seeds: dict[str, int] = Field(
        default_factory=dict,
        description="Seed values used per stochastic component for full reproducibility.",
    )


# ---------------------------------------------------------------------------
# Configuration  (SCHEMAS.md §Configuration)
# ---------------------------------------------------------------------------


class SyntheticDataConfig(BaseModel):
    """Sub-config for Phase 0 synthetic dataset generation."""

    n_samples: int = Field(3000, gt=0)
    n_features: int = Field(20, gt=0)
    n_classes: int = Field(4, ge=2)
    dirichlet_alpha: float = Field(0.5, gt=0.0)


class AttackConfig(BaseModel):
    """Sub-config for attack simulation (evaluation only, never seen by defense).

    Phase 0 (synthetic/feature-vector) fields:
      ``target_class``, ``trigger_block_start``, ``trigger_block_end``,
      ``trigger_value``, ``poison_fraction``, ``malicious_client_indices``.

    Phase 1 / Milestone 4 image-domain fields (all optional, with defaults):
      ``target_label``, ``trigger_shape``, ``trigger_size``,
      ``trigger_location``, ``trigger_color``, ``trigger_opacity``,
      ``malicious_client_fraction``, ``poison_non_target_only``.
    """

    # ── Phase 0 fields (kept for backward compat) ────────────────────────
    type: str = Field("badnet_colluding")
    target_class: int = Field(0, ge=0)
    trigger_block_start: int = Field(0, ge=0)
    trigger_block_end: int = Field(3, gt=0)
    trigger_value: float = Field(6.0)
    poison_fraction: float = Field(0.15, ge=0.0, le=1.0)
    malicious_client_indices: list[int] = Field(default_factory=lambda: [2, 5, 9])

    # ── Milestone 4 image-domain fields ──────────────────────────────────
    # target_label mirrors target_class but is the canonical name for image attacks.
    target_label: int = Field(0, ge=0, description="Backdoor target class (image attacks).")

    # Trigger pattern
    trigger_shape: str = Field(
        "square",
        description="square | cross | checkerboard | random_noise",
    )
    trigger_size: int = Field(4, ge=1, description="Trigger patch side length (pixels).")
    trigger_location: str = Field(
        "bottom_right",
        description="bottom_right | top_left | top_right | bottom_left | center",
    )
    trigger_color: float = Field(
        1.0,
        description="Normalised trigger pixel intensity (0.0–1.0).",
    )
    trigger_opacity: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Blend ratio: 1.0 = fully opaque.",
    )
    trigger_seed: int = Field(0, description="Seed for random_noise trigger shape.")

    # Malicious client selection
    malicious_client_fraction: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Fraction of n_clients that are malicious (when indices list is empty).",
    )
    poison_non_target_only: bool = Field(
        True,
        description="If True, only non-target-class samples are candidates for poisoning.",
    )

    @model_validator(mode="after")
    def _block_order(self) -> AttackConfig:
        if self.trigger_block_end <= self.trigger_block_start:
            raise ValueError("trigger_block_end must be > trigger_block_start")
        return self


class Configuration(BaseModel):
    """Root configuration object; loaded from configs/*.yaml and validated at load time.

    Schema validation failure exits the process with a clear field-level error message.
    No partial/default-filled config is ever silently run (ARCHITECTURE.md §7.10).
    """

    # FL topology
    n_clients: int = Field(12, gt=0)
    n_rounds: int = Field(20, gt=0)
    min_clients: int = Field(6, gt=0)

    # Aggregation
    aggregator: str = Field("multi_krum")
    krum_f: int = Field(3, ge=0)
    krum_select: int = Field(9, gt=0)

    # L1 collusion guard
    collusion_sim_threshold: float = Field(0.85, ge=0.0, le=1.0)
    collusion_min_cluster_size: int = Field(2, ge=2)

    # L2 model auditor
    audit_interval_rounds: int = Field(5, gt=0)
    audit_early_termination_threshold: int = Field(10, ge=0)

    # L3 runtime sentinel
    detectors: list[str] = Field(default_factory=lambda: ["strip_entropy"])
    strip_n_perturb: int = Field(50, gt=0)
    strip_target_frr: float = Field(0.02, gt=0.0, lt=1.0)

    # Dataset
    dataset_phase: Literal["phase0_synthetic", "phase1_official"] = Field("phase0_synthetic")
    synthetic: SyntheticDataConfig = Field(default_factory=SyntheticDataConfig)
    attack: AttackConfig = Field(default_factory=AttackConfig)

    # Training
    local_epochs: int = Field(5, gt=0)
    local_lr: float = Field(0.2, gt=0.0)

    # Model registry
    model_registry_dir: str = Field("experiments/checkpoints")
    registry_retention_k: int = Field(10, ge=0)

    # L5 remediation engine (ARCHITECTURE.md §7.4)
    remediation_enabled: bool = Field(True)
    remediation_asr_threshold: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Attack-success-rate at/under which remediation is considered successful.",
    )
    remediation_max_clean_accuracy_drop: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Max tolerated clean-accuracy regression for a remediation step to be accepted.",
    )
    remediation_strategies: list[str] = Field(
        default_factory=lambda: ["rollback", "unlearning", "pruning"],
        description="Ordered remediation escalation policy.",
    )
    unlearning_epochs: int = Field(10, gt=0)
    unlearning_lr: float = Field(0.1, gt=0.0)
    pruning_finetune_epochs: int = Field(5, gt=0)

    # Reproducibility
    seed: int = Field(42)

    # Logging
    log_level: str = Field("INFO")
    log_sink: str = Field("stdout")

    @field_validator("krum_select")
    @classmethod
    def _krum_select_valid(cls, v: int, info: Any) -> int:
        n = info.data.get("n_clients", 12)
        if v > n:
            raise ValueError(f"krum_select ({v}) cannot exceed n_clients ({n})")
        return v

    @field_validator("min_clients")
    @classmethod
    def _min_clients_valid(cls, v: int, info: Any) -> int:
        n = info.data.get("n_clients", 12)
        if v > n:
            raise ValueError(f"min_clients ({v}) cannot exceed n_clients ({n})")
        return v


# ---------------------------------------------------------------------------
# RemediationReport  (SCHEMAS.md §RemediationReport / ARCHITECTURE.md §7.4)
# ---------------------------------------------------------------------------


class RemediationReport(BaseModel):
    """Produced by the L5 Remediation Engine after responding to a confirmed backdoor.

    Captures the full audit trail: what was attempted, which step succeeded, and
    the ASR / clean-accuracy before and after, so a judge (or the dashboard's
    manual-review queue) can verify the recovery.
    """

    remediation_id: str = Field(default_factory=_new_id)
    round_num: int = Field(..., ge=0, description="Round whose global model was remediated.")
    suspected_infection_round: int | None = Field(
        None, ge=0, description="Earliest round believed to be poisoned (rollback target hint)."
    )
    strategies_attempted: list[str] = Field(
        default_factory=list, description="Remediation steps tried, in order."
    )
    strategy_succeeded: str | None = Field(
        None, description="The step that met the acceptance criteria, or None."
    )
    asr_before: float = Field(..., ge=0.0, le=1.0)
    asr_after: float = Field(..., ge=0.0, le=1.0)
    clean_accuracy_before: float = Field(..., ge=0.0, le=1.0)
    clean_accuracy_after: float = Field(..., ge=0.0, le=1.0)
    asr_threshold: float = Field(..., ge=0.0, le=1.0)
    success: bool = Field(...)
    manual_review_required: bool = Field(...)
    reason: str = Field("")
    rolled_back_model_id: str | None = Field(None)
    per_strategy: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: float | None = Field(None, ge=0.0)
    created_at: str = Field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Metric  (SCHEMAS.md §Metric)
# ---------------------------------------------------------------------------


class Metric(BaseModel):
    """Generic time-series data point for the dashboard's metric_timeseries endpoint."""

    metric_name: str = Field(...)
    round_num: int = Field(..., ge=0)
    value: float = Field(...)


# ---------------------------------------------------------------------------
# LogEntry  (SCHEMAS.md §LogEntry)
# ---------------------------------------------------------------------------


class LogEntry(BaseModel):
    """Universal structured log line emitted by every layer."""

    timestamp: str = Field(default_factory=_now_iso)
    layer_id: Literal["L1", "L2", "L3", "L4"] = Field(...)
    event_type: str = Field(
        ...,
        description="e.g. 'client_excluded', 'cluster_flagged', 'input_flagged'.",
    )
    round_num: int | None = Field(None)
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# TrustLedgerEntry  (SCHEMAS.md §TrustLedgerEntry)
# ---------------------------------------------------------------------------


class TrustLedgerEntry(BaseModel):
    """Per-flag record stored by L4 (Trust Ledger)."""

    entry_id: str = Field(default_factory=_new_id)
    layer_id: str = Field(..., description="Which layer produced this entry.")
    subject_type: Literal["client", "label", "input", "model"] = Field(...)
    subject_id: str = Field(...)
    round_num: int | None = Field(None)
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Human-readable explanation from Detector.explain().")
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw feature vector / similarity matrix slice / reversed trigger.",
    )


# ---------------------------------------------------------------------------
# ReputationSnapshot  (Milestone 6 — L4 visualization data)
# ---------------------------------------------------------------------------


class ReputationSnapshot(BaseModel):
    """Point-in-time snapshot of a client's full reputation state.

    Produced by ``FileTrustLedger.export_snapshot()`` and consumed by the
    dashboard's ``reputation_heatmap`` endpoint (API.md §4).

    Contains both the current score and rolling history lists so a single
    API call gives the dashboard all the data it needs to render the
    client's full reputation timeline.
    """

    client_id: str = Field(..., description="Client identifier.")
    round_num: int = Field(..., ge=0, description="The round this snapshot was taken at.")
    trust_score: float = Field(..., ge=0.0, le=1.0, description="Current trust score.")
    contributing_entry_count: int = Field(
        ..., ge=0, description="Number of ledger entries that contributed to this score."
    )
    flagged_by_layers: list[str] = Field(
        default_factory=list,
        description="Layer IDs that have flagged this client (e.g. ['L1', 'L2']).",
    )
    anomaly_score_history: list[float] = Field(
        default_factory=list,
        description="Per-round anomaly scores (most recent last).",
    )
    norm_history: list[float] = Field(
        default_factory=list,
        description="Per-round L2 update norms (most recent last).",
    )
    is_suspicious: bool = Field(
        ..., description="True if trust_score >= configured suspicious_threshold."
    )
    snapshot_timestamp: str = Field(
        default_factory=_now_iso,
        description="ISO-8601 UTC timestamp when this snapshot was produced.",
    )


# ---------------------------------------------------------------------------
# TrustLedgerQuery  (Milestone 6 — structured filter for ledger queries)
# ---------------------------------------------------------------------------


class TrustLedgerQuery(BaseModel):
    """Structured query object for ``FileTrustLedger.query()``.

    All filters are optional; unset fields match everything.
    Multiple filters are ANDed together.

    Example::

        q = TrustLedgerQuery(subject_ids=["client_02"], round_min=3, round_max=10)
        entries = ledger.query(q)
    """

    subject_ids: list[str] | None = Field(
        None, description="If set, only return entries for these subject IDs."
    )
    subject_types: list[str] | None = Field(
        None, description="Filter by 'client', 'label', or 'input'."
    )
    layers: list[str] | None = Field(None, description="Filter by layer ID (e.g. ['L1', 'L2']).")
    round_min: int | None = Field(None, ge=0, description="Inclusive lower round bound.")
    round_max: int | None = Field(None, ge=0, description="Inclusive upper round bound.")
    min_score: float | None = Field(None, ge=0.0, le=1.0, description="Minimum entry score.")
    max_score: float | None = Field(None, ge=0.0, le=1.0, description="Maximum entry score.")
    limit: int | None = Field(
        None, gt=0, description="Maximum number of entries to return (newest first)."
    )

    @field_validator("round_max")
    @classmethod
    def _round_range_valid(cls, v: int | None, info: Any) -> int | None:
        rmin = info.data.get("round_min")
        if v is not None and rmin is not None and v < rmin:
            raise ValueError(f"round_max ({v}) must be >= round_min ({rmin})")
        return v


# ---------------------------------------------------------------------------
# InferenceContext  (Milestone 7 — L3 Runtime Sentinel)
# ---------------------------------------------------------------------------


class InferenceContext(BaseModel):
    """All data RuntimeSentinelStrategy.process() needs for one inference.

    Constructed by the inference service and passed directly to the sentinel.
    Using a schema here (rather than raw dicts) makes the contract explicit
    and allows Pydantic to validate caller code at the boundary.

    ``input_data`` is stored as a flat list of floats so the object is
    JSON-serialisable end-to-end without numpy dependencies at the API layer.
    """

    input_id: str = Field(..., description="Unique identifier for this inference input.")
    input_data: list[float] = Field(
        ...,
        min_length=1,
        description="Flattened feature vector (or flattened image pixels).",
    )
    predicted_class: int = Field(..., ge=0, description="Argmax prediction before defence check.")
    predicted_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Softmax confidence of the predicted class."
    )
    round_num: int | None = Field(
        None, description="The FL round that produced the model used for this inference."
    )
    model_id: str | None = Field(
        None, description="ModelMetadata.model_id of the model serving this request."
    )
    timestamp: str = Field(default_factory=_now_iso, description="ISO-8601 UTC request time.")

    model_config = {"frozen": False, "arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# SentinelAlert  (Milestone 7 — L3 Runtime Sentinel)
# ---------------------------------------------------------------------------


class SentinelAlert(BaseModel):
    """Structured alert produced by RuntimeSentinelStrategy when an input is flagged.

    Feeds L4 (via AlertManager.to_ledger_entry()) and the structured logger.
    Severity is assigned by AlertManager based on configurable thresholds:

        fused_score < low_medium_boundary  → "low"
        low_medium_boundary ≤ score < med_high_boundary → "medium"
        score ≥ med_high_boundary          → "high"
    """

    alert_id: str = Field(default_factory=_new_id, description="UUID for this alert.")
    input_id: str = Field(..., description="The input that triggered the alert.")
    round_num: int | None = Field(None, description="FL round when alert was generated.")
    detector_verdicts: list[DetectionResult] = Field(
        ..., description="One DetectionResult per active L3 detector."
    )
    fused_score: float = Field(
        ..., ge=0.0, le=1.0, description="Fused anomaly score from FusionClassifier."
    )
    flagged: bool = Field(..., description="True if any detector flagged this input.")
    alert_severity: str = Field(
        ...,
        pattern="^(low|medium|high)$",
        description="Severity tier: low | medium | high.",
    )
    confidence_at_flag: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's predicted confidence at the time of flagging.",
    )
    explanation: str = Field(
        "", description="Combined human-readable reason from all detector explanations."
    )
    created_at: str = Field(default_factory=_now_iso)


# ===========================================================================
# Milestone 8 — Explainability schemas
# ===========================================================================


# ---------------------------------------------------------------------------
# SHAPExplanation
# ---------------------------------------------------------------------------


class SHAPExplanation(BaseModel):
    """Per-input SHAP feature attribution produced by SHAPExplainer.

    ``shap_values`` is a list of floats, one per feature, indicating each
    feature's additive contribution to the model's prediction.  Positive
    values push toward the predicted class; negative values push away.
    """

    input_id: str = Field(..., description="Identifier of the explained input.")
    predicted_class: int = Field(..., ge=0)
    base_value: float = Field(..., description="Expected model output (SHAP base value).")
    shap_values: list[float] = Field(..., description="Per-feature SHAP values.")
    feature_names: list[str] = Field(..., description="Feature name for each index in shap_values.")
    top_k_features: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Top-k features sorted by |shap_value| descending. "
            "Each dict has 'name', 'shap_value', 'rank'."
        ),
    )
    method: str = Field(
        "kernel_shap",
        description="Explanation method used (e.g. 'kernel_shap', 'permutation').",
    )
    created_at: str = Field(default_factory=_now_iso)

    @model_validator(mode="after")
    def _lengths_match(self) -> SHAPExplanation:
        if len(self.shap_values) != len(self.feature_names):
            raise ValueError(
                f"shap_values length ({len(self.shap_values)}) must equal "
                f"feature_names length ({len(self.feature_names)})."
            )
        return self


# ---------------------------------------------------------------------------
# FeatureImportanceResult
# ---------------------------------------------------------------------------


class FeatureImportanceResult(BaseModel):
    """Ranked feature importance list (permutation, coefficient, or gradient)."""

    method: str = Field(
        ...,
        description=("How importance was computed: 'permutation', 'coefficient', or 'gradient'."),
    )
    feature_names: list[str] = Field(...)
    importance_scores: list[float] = Field(
        ..., description="Importance score per feature (same order as feature_names)."
    )
    ranked_features: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Features sorted by importance descending. Each dict: name, score, rank.",
    )
    context: str = Field("", description="Optional context string (e.g. 'L1 round 4 client_02').")
    created_at: str = Field(default_factory=_now_iso)

    @model_validator(mode="after")
    def _lengths_match(self) -> FeatureImportanceResult:
        if len(self.importance_scores) != len(self.feature_names):
            raise ValueError("importance_scores and feature_names must have the same length.")
        return self


# ---------------------------------------------------------------------------
# DetectionExplanation
# ---------------------------------------------------------------------------


class DetectionExplanation(BaseModel):
    """Uniform drilldown for one flag event across all detection layers.

    Backs ``Visualizer.explainability_drilldown(trust_ledger_entry_id)``
    (INTERFACES.md) — one type regardless of whether the flag came from
    L1 (gradient clustering), L2 (audit report), or L3 (STRIP / SHAP).
    """

    entry_id: str = Field(..., description="TrustLedgerEntry.entry_id being explained.")
    layer_id: str = Field(..., description="Layer that generated the flag: L1, L2, or L3.")
    subject_id: str = Field(..., description="Client ID, label, or input ID.")
    subject_type: str = Field(..., description="'client', 'label', or 'input'.")
    round_num: int | None = Field(None)
    reason_string: str = Field(..., description="Verbatim from Detector.explain().")
    structured_evidence: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Layer-specific evidence: cosine sims (L1), reversed trigger (L2), SHAP values (L3)."
        ),
    )
    shap_explanation: SHAPExplanation | None = Field(
        None, description="SHAP attribution if available (L3 inputs)."
    )
    feature_importance: FeatureImportanceResult | None = Field(
        None, description="Feature importance if available (L1 gradient analysis)."
    )
    chart_artifacts: list[ChartArtifact] = Field(
        default_factory=list,
        description="Inline chart images for this explanation.",
    )
    created_at: str = Field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# TrustExplanation
# ---------------------------------------------------------------------------


class TrustExplanation(BaseModel):
    """Client reputation narrative with score trajectory and contributing events."""

    client_id: str = Field(...)
    current_score: float = Field(..., ge=0.0, le=1.0)
    is_suspicious: bool = Field(...)
    narrative: str = Field(..., description="Human-readable reputation summary.")
    score_trajectory: list[dict[str, Any]] = Field(
        default_factory=list,
        description=("Per-round score history. Each dict: round_num, score, n_flags, layers."),
    )
    top_contributing_entries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top-5 ledger entries by score contribution.",
    )
    layer_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Number of flags per layer (e.g. {'L1': 3, 'L2': 1}).",
    )
    chart_artifacts: list[ChartArtifact] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# AttackExplanation
# ---------------------------------------------------------------------------


class AttackExplanation(BaseModel):
    """Attack characterisation derived from an AuditReport or AttackReport."""

    attack_type: str = Field(..., description="E.g. 'BadNets', 'blended_injection', 'unknown'.")
    target_label: int | None = Field(None, ge=0)
    trigger_description: str = Field("", description="Human-readable trigger description.")
    poison_fraction: float | None = Field(None, ge=0.0, le=1.0)
    estimated_infection_round: int | None = Field(None, ge=0)
    suspected_clients: list[str] = Field(default_factory=list)
    detection_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregated detection confidence across layers.",
    )
    evidence_summary: str = Field("")
    chart_artifacts: list[ChartArtifact] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# ChartArtifact
# ---------------------------------------------------------------------------


class ChartArtifact(BaseModel):
    """An inline chart image serialised as base64 PNG.

    Produced by ChartGenerator and embedded in explanation objects so a
    single JSON response contains both the structured data and its visual
    representation — no separate file-serving endpoint needed.
    """

    chart_type: str = Field(
        ...,
        description=(
            "Chart type identifier: 'shap_bar', 'feature_importance', "
            "'trust_trajectory', 'reputation_heatmap', 'alert_timeline'."
        ),
    )
    title: str = Field("")
    png_b64: str = Field(..., description="Base64-encoded PNG image bytes.")
    alt_text: str = Field("", description="Accessibility description of the chart.")
    width_px: int = Field(800, gt=0)
    height_px: int = Field(500, gt=0)
    created_at: str = Field(default_factory=_now_iso)
