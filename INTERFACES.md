# INTERFACES.md — Abstract Module Contracts

Purpose: so an implementing agent never has to guess a method signature or where a new
attack/defense/dataset plugs in. These are contracts, not implementations — mirrors the
plugin pattern found in `BackdoorBench/attack/prototype.py` and
`FedML/.../defense_base.py` (see `RESEARCH.md` §4.1, §4.3), adapted for this project's
layered design (`ARCHITECTURE.md`).

Language-agnostic pseudocode; the production implementation is Python/PyTorch
(`TECH_STACK.md`).

---

## Detector

Used by L2 (Model Auditor) and L3 (Runtime Sentinel). Every concrete detector (STRIP,
Neural-Cleanse-style, activation-consistency) implements this.

```
interface Detector:
    name: str                      # unique id, e.g. "strip_entropy"
    layer: Literal["L2", "L3"]

    def calibrate(clean_reference_data) -> CalibrationState
        # Uses ONLY clean data. Must not require trojaned samples.
        # Returns an opaque, serializable calibration state (e.g. entropy boundary).

    def score(input_or_model, calibration_state) -> DetectionResult
        # L3 detectors: input_or_model is a single input tensor.
        # L2 detectors: input_or_model is a trained model + label to audit.
        # Returns a DetectionResult (see SCHEMAS.md).

    def explain(detection_result) -> str
        # Human-readable reason string for the Trust Ledger (L4).
```

**Failure cases**: calibration data too small (< configurable minimum) → raise
`InsufficientCalibrationDataError`, do not silently proceed with an unreliable
boundary. Model incompatible (e.g., no penultimate layer for activation-consistency) →
raise `UnsupportedModelError` at registration time, not at score time.

---

## Aggregator

Used by L1 (Update Guard). Wraps or subclasses Flower's strategy classes
(`RESEARCH.md` §4.2) for the production path; the NumPy reference in
`ai/fl_core/fl_engine.py` implements this same contract independently.

```
interface Aggregator:
    name: str

    def aggregate(client_updates: list[ClientUpdate]) -> AggregationResult
        # Returns the new global model delta AND which client indices were
        # included/excluded, so downstream layers (collusion clustering) can act on
        # the same round's exclusion decisions.
```

**Failure cases**: fewer than `min_clients` updates received → raise
`InsufficientClientsError`, round is aborted and logged, not silently degraded to
plain averaging.

---

## DefenseStrategy

The umbrella contract a **layer** (L1/L2/L3) implements, composing one or more
Detectors/Aggregators and writing to the Trust Ledger. This is the extension point for
adding a new layer without touching the other three.

```
interface DefenseStrategy:
    layer_id: str                  # "L1" | "L2" | "L3" | "L4"

    def process(context: RoundContext | InferenceContext) -> list[TrustLedgerEntry]
        # context type depends on layer_id (L1/L2 get RoundContext, L3 gets
        # InferenceContext — see SCHEMAS.md).
```

---

## AttackSimulator

Used in `experiments/` to generate poisoned training data / trigger a client's local
poisoning behavior for evaluation. Mirrors BackdoorBench's `attack/` module pattern
(`RESEARCH.md` §4.1) but adapted for a per-client, federated setting.

```
interface AttackSimulator:
    name: str                      # e.g. "badnet_colluding"

    def poison_client_data(X, y, client_id, round_num, config) -> (X', y', mask)
        # mask: boolean array marking which samples were poisoned, for evaluation only
        # — never exposed to the defense pipeline.

    def build_trigger_eval_set(X_clean) -> X_triggered
        # Used to compute ASR against the current global model.
```

---

## ModelRegistry

Tracks model checkpoints across FL rounds so L2 audits and the dashboard can reference
"the model as of round N," and so a flagged/rolled-back model can be recovered.

```
interface ModelRegistry:
    def save(round_num: int, model_state, metadata: ModelMetadata) -> model_id
    def load(model_id) -> (model_state, ModelMetadata)
    def latest() -> model_id
    def rollback_to(round_num) -> model_id
        # Used by the Remediation Engine (ARCHITECTURE.md §7) when L2 confirms a
        # backdoor was introduced at a known round.
```

**Failure cases**: `rollback_to` a round with no saved checkpoint → raise
`CheckpointNotFoundError`; caller must fall back to nearest earlier checkpoint
explicitly, never silently.

---

## DatasetLoader

Abstracts Phase 0 synthetic data vs. Phase 1 official challenge data
(`DATASETS.md`) behind one interface so `ai/fl_core` and `ai/detection` never
special-case which phase is active.

```
interface DatasetLoader:
    def load_client_partitions(n_clients, config) -> list[(X, y)]
    def load_clean_holdout() -> (X, y)
    def load_evaluation_set() -> (X, y)   # never seen during training
```

---

## Logger

Structured, JSON-lines logging shared by all layers (`ARCHITECTURE.md` §6).

```
interface Logger:
    def log(layer_id: str, event_type: str, payload: dict) -> None
        # Always includes: timestamp, round_num (if applicable), layer_id, event_type.
        # payload is layer-specific; see SCHEMAS.md#LogEntry.
```

---

## MetricsCollector

Computes the standard metric set (`BENCHMARK.md`) from logged rounds/detections.

```
interface MetricsCollector:
    def compute(experiment_id) -> EvaluationResult
        # Reads Logger output for the given experiment_id, returns the full metric
        # set (C-Acc, ASR, R-Acc, FAR, FRR, detection latency, communication cost).
```

---

## Visualizer

Backend-to-dashboard boundary (`ARCHITECTURE.md` §5, `API.md`).

```
interface Visualizer:
    def reputation_heatmap(experiment_id) -> JSON
    def metric_timeseries(experiment_id, metric_names) -> JSON
    def audit_report(experiment_id, round_num) -> JSON
    def explainability_drilldown(trust_ledger_entry_id) -> JSON
```

---

## Registration Convention

Every concrete implementation of the above interfaces registers itself in
`configs/registry.yaml` under its interface name (e.g. `detectors: [strip_entropy,
activation_consistency, neural_cleanse_audit]`), so layers are composed by
configuration, not by editing layer code. This is the mechanism that keeps `ARCHITECTURE.md`
§2's layer boundaries real in the implementation, not just on paper.
