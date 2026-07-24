"""
ai/evaluation/benchmark_reporter.py — Benchmark report generation.

Generates BenchmarkReport objects and writes them to disk as JSON and/or
Markdown.  Supports comparison against named baselines from baselines.yaml.

Public surface:
    BenchmarkReporter
        generate(experiment_id, evaluation_result, per_round_data,
                 detection_summary, chart_generator, baseline_name)
            → BenchmarkReport
        save_json(report, output_dir)     → Path
        save_markdown(report, output_dir) → Path
        compare_baseline(evaluation_result, baseline_name,
                         baseline_metrics, description)
            → BaselineComparison
        load_baseline(name, baselines_yaml)  → dict | None
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from ai.fl_core.schemas import (
    BaselineComparison,
    BenchmarkReport,
    EvaluationResult,
)

logger = logging.getLogger(__name__)

# Metrics where a HIGHER value is BETTER (higher than baseline = improvement).
_HIGHER_IS_BETTER = {
    "clean_accuracy",
    "attack_success_rate",
    "robust_accuracy",
    "precision",
    "recall",
    "f1_score",
}
# Metrics where a LOWER value is BETTER (lower than baseline = improvement).
_LOWER_IS_BETTER = {
    "false_acceptance_rate",
    "false_rejection_rate",
    "false_positive_rate",
    "detection_latency_ms",
    "communication_cost_bytes",
    "runtime_seconds",
    "peak_memory_mb",
}

# Key metrics used to determine overall verdict.
_VERDICT_METRICS = [
    ("clean_accuracy", "higher"),
    ("attack_success_rate", "lower"),  # lower ASR = better defence
    ("f1_score", "higher"),
]


class BenchmarkReporter:
    """Generate, compare, and persist BenchmarkReport objects.

    Args:
        baselines_yaml: Path to ``configs/baselines.yaml``.
            Defaults to ``configs/baselines.yaml`` relative to CWD.
    """

    def __init__(
        self,
        baselines_yaml: str | Path = "configs/baselines.yaml",
    ) -> None:
        self._baselines_yaml = Path(baselines_yaml)
        self._baselines_cache: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate(
        self,
        experiment_id: str,
        evaluation_result: EvaluationResult,
        per_round_data: list[dict[str, Any]] | None = None,
        detection_summary: dict[str, Any] | None = None,
        chart_generator: Any | None = None,
        baseline_name: str | None = None,
    ) -> BenchmarkReport:
        """Build a BenchmarkReport for one experiment.

        Args:
            experiment_id: The experiment identifier.
            evaluation_result: Pre-computed EvaluationResult.
            per_round_data: Optional per-round metric list.
            detection_summary: Optional per-layer flag totals.
            chart_generator: Optional M8 ChartGenerator for inline charts.
            baseline_name: If set, compare against this baseline.

        Returns:
            BenchmarkReport.
        """
        charts = []

        # Per-round data defaults
        rounds = per_round_data or []

        # Detection summary defaults
        det_summary = detection_summary or {}

        # Generate charts if chart_generator is provided
        if chart_generator is not None and rounds:
            try:
                charts.extend(self._generate_round_charts(chart_generator, rounds))
            except Exception as exc:
                logger.debug("BenchmarkReporter: chart generation failed: %s", exc)

        # Baseline comparison
        comparison: BaselineComparison | None = None
        if baseline_name:
            baseline = self.load_baseline(baseline_name)
            if baseline:
                comparison = self.compare_baseline(
                    evaluation_result,
                    baseline_name,
                    baseline,
                    description=baseline.get("description", ""),
                )
            else:
                logger.warning(
                    "BenchmarkReporter: baseline '%s' not found in %s.",
                    baseline_name,
                    self._baselines_yaml,
                )

        return BenchmarkReport(
            experiment_id=experiment_id,
            evaluation_result=evaluation_result,
            per_round_metrics=rounds,
            detection_summary=det_summary,
            baseline_comparison=comparison,
            chart_artifacts=charts,
        )

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    def compare_baseline(
        self,
        result: EvaluationResult,
        baseline_name: str,
        baseline_metrics: dict[str, Any],
        description: str = "",
    ) -> BaselineComparison:
        """Compare an EvaluationResult against a named baseline.

        Args:
            result: Our experiment's EvaluationResult.
            baseline_name: Identifier for the baseline.
            baseline_metrics: Dict of metric_name → value (may contain nulls).
            description: Human-readable baseline description.

        Returns:
            BaselineComparison with delta, improvement_percent, and verdict.
        """
        result_dict = result.model_dump()
        delta: dict[str, float | None] = {}
        improvement: dict[str, float | None] = {}

        for metric, baseline_val in baseline_metrics.items():
            if metric in ("description",):
                continue
            our_val = result_dict.get(metric)
            if our_val is None or baseline_val is None:
                delta[metric] = None
                improvement[metric] = None
                continue

            our_val = float(our_val)
            baseline_val = float(baseline_val)
            d = our_val - baseline_val
            delta[metric] = round(d, 6)

            if abs(baseline_val) > 1e-9:
                improvement[metric] = round(d / abs(baseline_val) * 100, 2)
            else:
                improvement[metric] = None

        verdict = self._compute_verdict(delta)
        summary = self._build_summary(baseline_name, delta, verdict, result)

        return BaselineComparison(
            baseline_name=baseline_name,
            baseline_description=description,
            baseline_metrics={k: v for k, v in baseline_metrics.items() if k != "description"},
            delta_metrics=delta,
            improvement_percent=improvement,
            verdict=verdict,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_json(
        self,
        report: BenchmarkReport,
        output_dir: str | Path,
    ) -> Path:
        """Write the report as a JSON file.

        Args:
            report: BenchmarkReport to persist.
            output_dir: Directory to write into.

        Returns:
            Path to the written file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"report_{report.experiment_id}.json"
        data = json.loads(report.model_dump_json())
        # Strip large base64 blobs from JSON output (charts are embedded in
        # the object; the JSON report omits them for readability).
        for chart in data.get("chart_artifacts", []):
            chart.pop("png_b64", None)
        fname.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("BenchmarkReporter: saved JSON report to %s", fname)
        return fname

    def save_markdown(
        self,
        report: BenchmarkReport,
        output_dir: str | Path,
    ) -> Path:
        """Write the report as a Markdown file.

        Args:
            report: BenchmarkReport to persist.
            output_dir: Directory to write into.

        Returns:
            Path to the written file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"report_{report.experiment_id}.md"
        fname.write_text(self._render_markdown(report), encoding="utf-8")
        logger.info("BenchmarkReporter: saved Markdown report to %s", fname)
        return fname

    def load_baseline(
        self,
        name: str,
        baselines_yaml: str | Path | None = None,
    ) -> dict[str, Any] | None:
        """Load a named baseline's metrics from baselines.yaml.

        Args:
            name: Baseline name (key under ``baselines:`` in the YAML).
            baselines_yaml: Override the default path.

        Returns:
            Dict of metric values, or None if not found.
        """
        path = Path(baselines_yaml) if baselines_yaml else self._baselines_yaml
        if self._baselines_cache is None or baselines_yaml:
            self._baselines_cache = self._load_yaml_baselines(path)
        return self._baselines_cache.get(name)

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def _render_markdown(self, report: BenchmarkReport) -> str:
        er = report.evaluation_result
        lines: list[str] = []

        lines.append(f"# Benchmark Report — {report.experiment_id}\n")
        lines.append(f"Generated: {report.generated_at}  \n")
        lines.append(f"Report ID: `{report.report_id}`\n")

        # Summary metrics table
        lines.append("\n## Summary Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        metric_rows = [
            ("Clean Accuracy", er.clean_accuracy),
            ("Attack Success Rate", er.attack_success_rate),
            ("Robust Accuracy", er.robust_accuracy),
            ("Precision", er.precision),
            ("Recall", er.recall),
            ("F1 Score", er.f1_score),
            ("False Positive Rate", er.false_positive_rate),
            ("False Acceptance Rate (L3)", er.false_acceptance_rate),
            ("False Rejection Rate (L3)", er.false_rejection_rate),
            ("Detection Latency (ms/input)", er.detection_latency_ms),
            ("Communication Cost (bytes)", er.communication_cost_bytes),
            ("Runtime (seconds)", er.runtime_seconds),
            ("Peak Memory (MB)", er.peak_memory_mb),
        ]
        for name, val in metric_rows:
            val_str = (
                f"{val:.4f}" if isinstance(val, float) else (str(val) if val is not None else "—")
            )
            lines.append(f"| {name} | {val_str} |")

        # Warnings
        if er.warnings:
            lines.append("\n### Warnings\n")
            for w in er.warnings:
                lines.append(f"- ⚠️ {w}")

        # Per-round metrics
        if report.per_round_metrics:
            lines.append("\n## Per-Round Metrics\n")
            keys = list(report.per_round_metrics[0].keys())
            lines.append("| " + " | ".join(str(k) for k in keys) + " |")
            lines.append("|" + "---|" * len(keys))
            for row in report.per_round_metrics:
                vals = []
                for k in keys:
                    v = row.get(k)
                    vals.append(
                        f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else "—"
                    )
                lines.append("| " + " | ".join(vals) + " |")

        # Detection summary
        if report.detection_summary:
            lines.append("\n## Detection Summary\n")
            lines.append("| Layer | Flags |")
            lines.append("|---|---|")
            for layer, count in sorted(report.detection_summary.items()):
                lines.append(f"| {layer} | {count} |")

        # Baseline comparison
        if report.baseline_comparison is not None:
            bc = report.baseline_comparison
            lines.append(f"\n## Baseline Comparison — `{bc.baseline_name}`\n")
            if bc.baseline_description:
                lines.append(f"> {bc.baseline_description}\n")
            lines.append(f"**Verdict: {bc.verdict.upper()}**\n")
            if bc.summary:
                lines.append(f"{bc.summary}\n")
            lines.append("\n| Metric | Ours | Baseline | Δ | Improvement % |")
            lines.append("|---|---|---|---|---|")
            for metric, baseline_val in bc.baseline_metrics.items():
                our_val = report.evaluation_result.model_dump().get(metric)
                d = bc.delta_metrics.get(metric)
                imp = bc.improvement_percent.get(metric)
                ours_str = (
                    f"{float(our_val):.4f}"
                    if isinstance(our_val, float)
                    else (str(our_val) if our_val is not None else "—")
                )
                base_str = (
                    f"{float(baseline_val):.4f}"
                    if isinstance(baseline_val, float)
                    else (str(baseline_val) if baseline_val is not None else "—")
                )
                d_str = f"{d:+.4f}" if isinstance(d, float) else "—"
                imp_str = f"{imp:+.1f}%" if isinstance(imp, float) else "—"
                lines.append(f"| {metric} | {ours_str} | {base_str} | {d_str} | {imp_str} |")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Verdict / summary helpers
    # ------------------------------------------------------------------

    def _compute_verdict(self, delta: dict[str, float | None]) -> str:
        """Determine overall verdict based on key metric deltas."""
        better = 0
        worse = 0
        for metric, direction in _VERDICT_METRICS:
            d = delta.get(metric)
            if d is None:
                continue
            if direction == "higher" and d > 0:
                better += 1
            elif direction == "lower" and d < 0:
                better += 1
            elif direction == "higher" and d < 0:
                worse += 1
            elif direction == "lower" and d > 0:
                worse += 1
        if better > 0 and worse == 0:
            return "better"
        if worse > 0 and better == 0:
            return "worse"
        if better == 0 and worse == 0:
            return "mixed"
        return "mixed"

    def _build_summary(
        self,
        baseline_name: str,
        delta: dict[str, float | None],
        verdict: str,
        result: EvaluationResult,
    ) -> str:
        parts = [f"Comparison against '{baseline_name}': verdict={verdict.upper()}."]
        for metric in ("clean_accuracy", "attack_success_rate", "f1_score"):
            d = delta.get(metric)
            if d is not None:
                direction = "↑" if d > 0 else "↓"
                parts.append(f"{metric}: {direction}{abs(d):.4f}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    def _generate_round_charts(
        self, chart_generator: Any, rounds: list[dict[str, Any]]
    ) -> list[Any]:
        """Generate per-round charts via M8 ChartGenerator."""
        charts = []
        # Build an alert-timeline style chart from per-round data
        alert_data = [
            {
                "round_num": r.get("round_num", 0),
                "alert_severity": (
                    "high"
                    if (r.get("n_l1_flags", 0) + r.get("n_l3_flags", 0)) > 2
                    else "medium"
                    if (r.get("n_l1_flags", 0) + r.get("n_l3_flags", 0)) > 0
                    else "low"
                ),
            }
            for r in rounds
        ]
        if alert_data:
            chart = chart_generator.alert_timeline_chart(
                alert_data, title="Detection Activity per Round"
            )
            charts.append(chart)
        return charts

    # ------------------------------------------------------------------
    # YAML loader
    # ------------------------------------------------------------------

    def _load_yaml_baselines(self, path: Path) -> dict[str, dict[str, Any]]:
        """Load baselines.yaml and return the ``baselines:`` dict."""
        if not path.exists():
            logger.warning("baselines.yaml not found at %s.", path)
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            return dict(data.get("baselines", {}))
        except Exception as exc:
            logger.warning("Failed to load baselines.yaml: %s", exc)
            return {}
