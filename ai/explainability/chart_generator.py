"""
ai/explainability/chart_generator.py — Matplotlib chart generation for SENTINEL-FL.

Produces inline PNG charts (base64-encoded) embedded in explanation objects.
No filesystem side-effects by default — all output lives in ``ChartArtifact``
objects returned to the caller.  Optional ``save_to`` path for disk persistence.

Chart types:
    shap_bar_chart          — horizontal bar chart of SHAP values
    feature_importance_chart — horizontal bar chart of feature importances
    trust_trajectory_chart  — line chart of flag count over rounds
    reputation_heatmap_chart — client × layer heatmap of flag counts
    alert_timeline_chart     — timeline of alert severities over rounds
    save_all                 — batch-write all chart artifacts to disk

Design:
    - Uses ``matplotlib`` with the ``Agg`` backend (no display required).
    - All colours use the project palette: deep-blue (#1E4D7B), amber (#F5A623),
      red (#D0021B), clean-white (#FAFAFA).
    - Falls back gracefully if matplotlib is not installed: returns a
      ``ChartArtifact`` with ``png_b64=""`` and logs a warning.

Public surface:
    ChartGenerator
        shap_bar_chart(shap_explanation, title, top_k)   → ChartArtifact
        feature_importance_chart(importance_result, title, top_k) → ChartArtifact
        trust_trajectory_chart(trust_explanation)        → ChartArtifact
        reputation_heatmap_chart(heatmap_data, title)    → ChartArtifact
        alert_timeline_chart(alerts, title)              → ChartArtifact
        save_all(artifacts, output_dir)                  → list[Path]
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

from ai.fl_core.schemas import (
    ChartArtifact,
    FeatureImportanceResult,
    SHAPExplanation,
    TrustExplanation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project colour palette
# ---------------------------------------------------------------------------

_PALETTE = {
    "blue": "#1E4D7B",
    "blue_light": "#4A90D9",
    "amber": "#F5A623",
    "red": "#D0021B",
    "green": "#417505",
    "bg": "#FAFAFA",
    "grid": "#E8E8E8",
    "text": "#2C2C2C",
}

_MPL_AVAILABLE: bool | None = None


def _check_mpl() -> bool:
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib  # noqa: F401

            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
            logger.warning(
                "ChartGenerator: matplotlib not installed — charts will be empty (png_b64='')."
            )
    return _MPL_AVAILABLE


def _empty_artifact(chart_type: str, title: str, alt_text: str) -> ChartArtifact:
    return ChartArtifact(
        chart_type=chart_type,
        title=title,
        png_b64="",
        alt_text=alt_text,
        width_px=800,
        height_px=500,
    )


def _fig_to_b64(fig: Any, dpi: int = 120) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=_PALETTE["bg"])
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


class ChartGenerator:
    """Matplotlib-based chart factory for SENTINEL-FL explanations.

    Args:
        dpi: Chart resolution.
        top_k: Default number of features to show in bar charts.
        figsize_w: Default figure width (inches).
        figsize_h: Default figure height (inches).
    """

    def __init__(
        self,
        dpi: int = 120,
        top_k: int = 10,
        figsize_w: float = 8.0,
        figsize_h: float = 5.0,
    ) -> None:
        self._dpi = dpi
        self._top_k = top_k
        self._fw = figsize_w
        self._fh = figsize_h

    # ------------------------------------------------------------------
    # SHAP bar chart
    # ------------------------------------------------------------------

    def shap_bar_chart(
        self,
        shap_explanation: SHAPExplanation,
        title: str = "SHAP Feature Attribution",
        top_k: int | None = None,
    ) -> ChartArtifact:
        """Horizontal bar chart of SHAP values, coloured by sign.

        Positive values → blue (push toward predicted class).
        Negative values → red (push away from predicted class).

        Args:
            shap_explanation: SHAPExplanation object.
            title: Chart title.
            top_k: Override default top-k features shown.

        Returns:
            ChartArtifact with inline PNG.
        """
        k = top_k or self._top_k
        if not _check_mpl():
            return _empty_artifact("shap_bar", title, "SHAP bar chart (matplotlib unavailable)")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use pre-computed top_k_features if available, else compute
        items = shap_explanation.top_k_features[:k] if shap_explanation.top_k_features else []
        if not items:
            pairs = sorted(
                zip(shap_explanation.feature_names, shap_explanation.shap_values),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:k]
            items = [{"name": n, "shap_value": v} for n, v in pairs]

        names = [it["name"] for it in items]
        values = [it["shap_value"] for it in items]
        colours = [_PALETTE["blue"] if v >= 0 else _PALETTE["red"] for v in values]

        fig, ax = plt.subplots(figsize=(self._fw, max(self._fh, len(names) * 0.45 + 1.0)))
        fig.patch.set_facecolor(_PALETTE["bg"])
        ax.set_facecolor(_PALETTE["bg"])

        ax.barh(range(len(names)), values, color=colours, edgecolor="none", height=0.65)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9, color=_PALETTE["text"])
        ax.axvline(0, color=_PALETTE["text"], linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value", fontsize=10, color=_PALETTE["text"])
        ax.set_title(title, fontsize=12, color=_PALETTE["text"], pad=12, fontweight="bold")
        ax.tick_params(colors=_PALETTE["text"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=_PALETTE["grid"], linewidth=0.5)

        png_b64 = _fig_to_b64(fig, self._dpi)
        plt.close(fig)

        return ChartArtifact(
            chart_type="shap_bar",
            title=title,
            png_b64=png_b64,
            alt_text=(
                f"SHAP bar chart for input '{shap_explanation.input_id}'. "
                f"Top {len(names)} features shown."
            ),
            width_px=int(self._fw * self._dpi),
            height_px=int(max(self._fh, len(names) * 0.45 + 1.0) * self._dpi),
        )

    # ------------------------------------------------------------------
    # Feature importance chart
    # ------------------------------------------------------------------

    def feature_importance_chart(
        self,
        importance_result: FeatureImportanceResult,
        title: str = "Feature Importance",
        top_k: int | None = None,
    ) -> ChartArtifact:
        """Horizontal bar chart of feature importance scores.

        Args:
            importance_result: FeatureImportanceResult object.
            title: Chart title.
            top_k: Override default top-k features shown.

        Returns:
            ChartArtifact.
        """
        k = top_k or self._top_k
        if not _check_mpl():
            return _empty_artifact("feature_importance", title, "Feature importance chart")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        items = importance_result.ranked_features[:k]
        names = [it["name"] for it in items]
        scores = [it["score"] for it in items]

        fig, ax = plt.subplots(figsize=(self._fw, max(self._fh, len(names) * 0.45 + 1.0)))
        fig.patch.set_facecolor(_PALETTE["bg"])
        ax.set_facecolor(_PALETTE["bg"])

        ax.barh(
            range(len(names)), scores, color=_PALETTE["blue_light"], edgecolor="none", height=0.65
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9, color=_PALETTE["text"])
        ax.set_xlabel(
            f"Importance ({importance_result.method})", fontsize=10, color=_PALETTE["text"]
        )
        ax.set_title(title, fontsize=12, color=_PALETTE["text"], pad=12, fontweight="bold")
        ax.tick_params(colors=_PALETTE["text"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=_PALETTE["grid"], linewidth=0.5)

        png_b64 = _fig_to_b64(fig, self._dpi)
        plt.close(fig)

        return ChartArtifact(
            chart_type="feature_importance",
            title=title,
            png_b64=png_b64,
            alt_text=f"Feature importance chart ({importance_result.method}). Top {len(names)} features.",
            width_px=int(self._fw * self._dpi),
            height_px=int(max(self._fh, len(names) * 0.45 + 1.0) * self._dpi),
        )

    # ------------------------------------------------------------------
    # Trust trajectory chart
    # ------------------------------------------------------------------

    def trust_trajectory_chart(
        self,
        trust_explanation: TrustExplanation,
        title: str | None = None,
    ) -> ChartArtifact:
        """Line chart of per-round flag count over FL rounds.

        Args:
            trust_explanation: TrustExplanation with score_trajectory populated.
            title: Chart title.  Defaults to client ID.

        Returns:
            ChartArtifact.
        """
        traj = trust_explanation.score_trajectory
        t = title or f"Trust Trajectory — {trust_explanation.client_id}"

        if not _check_mpl():
            return _empty_artifact("trust_trajectory", t, "Trust trajectory chart")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = [p["round_num"] for p in traj]
        flags = [p["n_flags"] for p in traj]

        fig, ax = plt.subplots(figsize=(self._fw, self._fh))
        fig.patch.set_facecolor(_PALETTE["bg"])
        ax.set_facecolor(_PALETTE["bg"])

        colour = _PALETTE["red"] if trust_explanation.is_suspicious else _PALETTE["blue"]
        ax.plot(rounds, flags, marker="o", color=colour, linewidth=2, markersize=5)
        ax.fill_between(rounds, flags, alpha=0.15, color=colour)
        ax.set_xlabel("FL Round", fontsize=10, color=_PALETTE["text"])
        ax.set_ylabel("Flags in round", fontsize=10, color=_PALETTE["text"])
        ax.set_title(t, fontsize=12, color=_PALETTE["text"], pad=12, fontweight="bold")
        ax.tick_params(colors=_PALETTE["text"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=_PALETTE["grid"], linewidth=0.5)

        # Annotate current score
        ax.annotate(
            f"Score: {trust_explanation.current_score:.3f}",
            xy=(0.98, 0.96),
            xycoords="axes fraction",
            fontsize=9,
            ha="right",
            color=colour,
        )

        png_b64 = _fig_to_b64(fig, self._dpi)
        plt.close(fig)

        return ChartArtifact(
            chart_type="trust_trajectory",
            title=t,
            png_b64=png_b64,
            alt_text=f"Flag count per round for client '{trust_explanation.client_id}'.",
            width_px=int(self._fw * self._dpi),
            height_px=int(self._fh * self._dpi),
        )

    # ------------------------------------------------------------------
    # Reputation heatmap
    # ------------------------------------------------------------------

    def reputation_heatmap_chart(
        self,
        heatmap_data: dict[str, dict[str, int]],
        title: str = "Client × Layer Flag Heatmap",
    ) -> ChartArtifact:
        """Client × Layer heatmap of flag counts.

        Args:
            heatmap_data: Nested dict ``{client_id: {layer_id: count}}``.
            title: Chart title.

        Returns:
            ChartArtifact.
        """
        if not _check_mpl():
            return _empty_artifact("reputation_heatmap", title, "Reputation heatmap")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        clients = sorted(heatmap_data.keys())
        layers = sorted({lyr for d in heatmap_data.values() for lyr in d})
        matrix = np.array(
            [[heatmap_data[c].get(lyr, 0) for lyr in layers] for c in clients],
            dtype=float,
        )

        fig, ax = plt.subplots(
            figsize=(max(self._fw, len(layers) * 1.2), max(self._fh, len(clients) * 0.6))
        )
        fig.patch.set_facecolor(_PALETTE["bg"])
        ax.set_facecolor(_PALETTE["bg"])

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Flag count")

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=10, color=_PALETTE["text"])
        ax.set_yticks(range(len(clients)))
        ax.set_yticklabels(clients, fontsize=9, color=_PALETTE["text"])
        ax.set_title(title, fontsize=12, color=_PALETTE["text"], pad=12, fontweight="bold")

        # Annotate cells
        for i in range(len(clients)):
            for j in range(len(layers)):
                v = int(matrix[i, j])
                if v > 0:
                    ax.text(
                        j, i, str(v), ha="center", va="center", fontsize=9, color=_PALETTE["text"]
                    )

        png_b64 = _fig_to_b64(fig, self._dpi)
        plt.close(fig)

        return ChartArtifact(
            chart_type="reputation_heatmap",
            title=title,
            png_b64=png_b64,
            alt_text=f"Heatmap of {len(clients)} clients × {len(layers)} layers.",
            width_px=int(max(self._fw, len(layers) * 1.2) * self._dpi),
            height_px=int(max(self._fh, len(clients) * 0.6) * self._dpi),
        )

    # ------------------------------------------------------------------
    # Alert timeline
    # ------------------------------------------------------------------

    def alert_timeline_chart(
        self,
        alerts: list[Any],
        title: str = "Alert Timeline",
    ) -> ChartArtifact:
        """Scatter timeline of alerts coloured by severity.

        Args:
            alerts: List of SentinelAlert objects or dicts with
                ``round_num`` and ``alert_severity``.
            title: Chart title.

        Returns:
            ChartArtifact.
        """
        if not _check_mpl():
            return _empty_artifact("alert_timeline", title, "Alert timeline chart")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        severity_colours = {
            "low": _PALETTE["amber"],
            "medium": _PALETTE["amber"],
            "high": _PALETTE["red"],
        }
        severity_sizes = {"low": 40, "medium": 80, "high": 140}

        data = []
        for a in alerts:
            rnd = a.get("round_num") if isinstance(a, dict) else getattr(a, "round_num", None)
            sev = (
                a.get("alert_severity")
                if isinstance(a, dict)
                else getattr(a, "alert_severity", "low")
            )
            data.append((rnd or 0, sev or "low"))

        fig, ax = plt.subplots(figsize=(self._fw, self._fh * 0.8))
        fig.patch.set_facecolor(_PALETTE["bg"])
        ax.set_facecolor(_PALETTE["bg"])

        # Group by severity for legend
        for severity in ("low", "medium", "high"):
            pts = [(r, i) for i, (r, s) in enumerate(data) if s == severity]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(
                    xs,
                    ys,
                    c=severity_colours[severity],
                    s=severity_sizes[severity],
                    label=severity.capitalize(),
                    alpha=0.85,
                    edgecolors="none",
                )

        ax.set_xlabel("FL Round", fontsize=10, color=_PALETTE["text"])
        ax.set_ylabel("Alert index", fontsize=10, color=_PALETTE["text"])
        ax.set_title(title, fontsize=12, color=_PALETTE["text"], pad=12, fontweight="bold")
        ax.tick_params(colors=_PALETTE["text"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=_PALETTE["grid"], linewidth=0.5)
        if data:
            ax.legend(loc="upper left", frameon=False, fontsize=9)

        png_b64 = _fig_to_b64(fig, self._dpi)
        plt.close(fig)

        return ChartArtifact(
            chart_type="alert_timeline",
            title=title,
            png_b64=png_b64,
            alt_text=f"Timeline of {len(data)} alerts.",
            width_px=int(self._fw * self._dpi),
            height_px=int(self._fh * 0.8 * self._dpi),
        )

    # ------------------------------------------------------------------
    # Batch save
    # ------------------------------------------------------------------

    def save_all(
        self,
        artifacts: list[ChartArtifact],
        output_dir: str | Path,
    ) -> list[Path]:
        """Write all chart artifacts to disk as PNG files.

        Args:
            artifacts: List of ChartArtifact objects.
            output_dir: Directory to write PNG files into.

        Returns:
            List of written file paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for i, art in enumerate(artifacts):
            if not art.png_b64:
                continue
            slug = art.chart_type.replace(" ", "_")
            fname = out / f"{slug}_{i:03d}.png"
            fname.write_bytes(base64.b64decode(art.png_b64))
            written.append(fname)
            logger.debug("ChartGenerator: saved %s", fname)
        return written
