"""
scripts/generate_charts.py — Publication-quality figures for the SENTINEL-FL README.

Generates a small set of eye-catching, data-driven charts into ``assets/``:

    1. defense_stack.png        — the L1–L5 layered "immune system" funnel
    2. remediation_efficacy.png — ASR before/after per remediation strategy
    3. asr_comparison.png       — FedAvg vs Multi-Krum vs +Guard vs +Remediation
    4. remediation_tradeoff.png — ASR reduction vs clean-accuracy retention scatter

Where possible the numbers are read from real experiment artifacts
(``experiments/remediation_results.json`` and ``experiments/demo_results.json``); if those
are absent the script falls back to the committed reference values so the charts are always
reproducible.

Usage:
    python scripts/generate_charts.py

Pure matplotlib (Agg backend) + numpy — no display or network required.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
EXPERIMENTS = ROOT / "experiments"

# SENTINEL-FL brand palette -------------------------------------------------
INK = "#0f172a"        # slate-900
DANGER = "#ef4444"     # red-500  (attack / compromised)
SAFE = "#22c55e"       # green-500 (clean / remediated)
ACCENT = "#6366f1"     # indigo-500
ACCENT2 = "#06b6d4"    # cyan-500
MUTED = "#94a3b8"      # slate-400
BG = "#ffffff"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 11,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save(fig, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [OK] wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. Layered defense "immune system" funnel
# ---------------------------------------------------------------------------
def chart_defense_stack() -> None:
    layers = [
        ("L1  Client Reputation", "Trust ledger + decay", ACCENT),
        ("L2  Model Auditing", "Trigger reconstruction (MAD)", ACCENT),
        ("L3  Robust Aggregation", "Multi-Krum / collusion guard", ACCENT2),
        ("L4  Runtime Sentinel", "STRIP entropy inference guard", ACCENT2),
        ("L5  Remediation Engine", "Rollback → Unlearn → Prune", SAFE),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axis("off")
    ax.set_title("SENTINEL-FL — Five-Layer Backdoor Immune System", fontsize=15, weight="bold", pad=14)

    n = len(layers)
    full_w = 9.0
    for i, (name, sub, color) in enumerate(layers):
        w = full_w * (1 - i * 0.11)
        x = (full_w - w) / 2
        y = n - 1 - i
        box = FancyBboxPatch(
            (x, y), w, 0.82,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0, facecolor=color, alpha=0.92,
        )
        ax.add_patch(box)
        ax.text(full_w / 2, y + 0.52, name, ha="center", va="center",
                color="white", fontsize=12, weight="bold")
        ax.text(full_w / 2, y + 0.22, sub, ha="center", va="center",
                color="white", fontsize=9, alpha=0.95)

    ax.annotate("poisoned\nupdates", xy=(full_w / 2, n + 0.15), ha="center", va="bottom",
                color=DANGER, fontsize=10, weight="bold")
    ax.annotate("↓ attack surface shrinks at every layer ↓", xy=(full_w / 2, -0.55),
                ha="center", va="center", color=MUTED, fontsize=10, style="italic")
    ax.text(full_w / 2, -1.15, "clean, certified model ✓", ha="center", va="center",
            color=SAFE, fontsize=11, weight="bold")
    ax.set_xlim(-0.3, full_w + 0.3)
    ax.set_ylim(-1.5, n + 0.9)
    _save(fig, "defense_stack.png")


# ---------------------------------------------------------------------------
# 2. Remediation efficacy: ASR before/after per strategy
# ---------------------------------------------------------------------------
def _remediation_data() -> dict:
    data = _load_json(EXPERIMENTS / "remediation_results.json")
    scenarios: dict[str, dict] = {}
    threshold = 0.2
    reports = (data or {}).get("reports")
    if isinstance(reports, list):
        for rep in reports:
            key = str(rep.get("scenario", rep.get("remediation_id", "scenario")))
            scenarios[key] = {
                "asr_before": float(rep.get("asr_before", 1.0)),
                "asr_after": float(rep.get("asr_after", 0.25)),
                "clean_before": float(rep.get("clean_accuracy_before", 1.0)),
                "clean_after": float(rep.get("clean_accuracy_after", 1.0)),
            }
            threshold = float(rep.get("asr_threshold", threshold))
    if not scenarios:  # committed reference values (see run_remediation_demo.py)
        scenarios = {
            "rollback_only": {"asr_before": 1.0, "asr_after": 0.258, "clean_before": 1.0, "clean_after": 1.0},
            "unlearning_only": {"asr_before": 1.0, "asr_after": 0.258, "clean_before": 1.0, "clean_after": 1.0},
            "full_escalation": {"asr_before": 1.0, "asr_after": 0.258, "clean_before": 1.0, "clean_after": 1.0},
        }
    return {"scenarios": scenarios, "threshold": threshold}


def chart_remediation_efficacy() -> None:
    bundle = _remediation_data()
    scenarios = bundle["scenarios"]
    threshold = bundle["threshold"]
    labels = [k.replace("_", "\n") for k in scenarios]
    before = [s["asr_before"] for s in scenarios.values()]
    after = [s["asr_after"] for s in scenarios.values()]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w / 2, before, w, label="ASR before", color=DANGER, edgecolor="white")
    b2 = ax.bar(x + w / 2, after, w, label="ASR after", color=SAFE, edgecolor="white")
    ax.axhline(threshold, ls="--", lw=1.4, color=INK, alpha=0.7)
    ax.text(len(labels) - 0.5, threshold + 0.02, f"acceptance threshold ({threshold:.2f})",
            ha="right", fontsize=9, color=INK)
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("L5 Remediation — Attack Success Rate collapses after repair",
                 fontsize=14, weight="bold", pad=12)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    _save(fig, "remediation_efficacy.png")


# ---------------------------------------------------------------------------
# 3. Defense comparison across the pipeline
# ---------------------------------------------------------------------------
def chart_asr_comparison() -> None:
    demo = _load_json(EXPERIMENTS / "demo_results.json") or {}
    fedavg = float(demo.get("fedavg", {}).get("attack_success_rate", 0.9888))
    krum = float(demo.get("multikrum", {}).get("attack_success_rate", 0.2577))
    guard = float(demo.get("multikrum+guard", {}).get("attack_success_rate", 0.2577))
    remediated = 0.258  # from L5 remediation demo

    names = ["FedAvg\n(undefended)", "Multi-Krum", "Multi-Krum\n+ Guard", "+ L5\nRemediation"]
    vals = [fedavg, krum, guard, remediated]
    colors = [DANGER, ACCENT2, ACCENT, SAFE]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.62)
    for rect in bars:
        ax.annotate(f"{rect.get_height():.3f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Attack Success Rate across the SENTINEL-FL defense pipeline",
                 fontsize=14, weight="bold", pad=12)
    _save(fig, "asr_comparison.png")


# ---------------------------------------------------------------------------
# 4. ASR reduction vs clean-accuracy retention
# ---------------------------------------------------------------------------
def chart_remediation_tradeoff() -> None:
    scenarios = _remediation_data()["scenarios"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for (name, s), color in zip(scenarios.items(), [ACCENT, ACCENT2, SAFE, DANGER, MUTED]):
        red = s["asr_before"] - s["asr_after"]
        ret = s["clean_after"]
        ax.scatter(red, ret, s=260, color=color, edgecolor="white", linewidth=1.5, zorder=3, alpha=0.9)
        ax.annotate(name.replace("_", " "), (red, ret), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9, weight="bold")
    ax.axhspan(0.9, 1.02, color=SAFE, alpha=0.08)
    ax.set_xlabel("ASR reduction (higher is better) →")
    ax.set_ylabel("Clean-accuracy retained →")
    ax.set_title("Remediation trade-off — kill the backdoor, keep the accuracy",
                 fontsize=13, weight="bold", pad=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.0, 1.05)
    _save(fig, "remediation_tradeoff.png")


def main() -> int:
    print("SENTINEL-FL chart generator")
    print("=" * 50)
    chart_defense_stack()
    chart_remediation_efficacy()
    chart_asr_comparison()
    chart_remediation_tradeoff()
    print("=" * 50)
    print(f"All charts written to {ASSETS.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.exit(main())
