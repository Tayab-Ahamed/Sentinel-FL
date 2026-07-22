# docs/SCREENSHOTS.md — Dashboard Screenshot Guide

This document describes each screenshot that should appear in `docs/screenshots/`
to support the competition submission and README.

Screenshots should be captured after running the full demo:
```bash
python scripts/run_demo.py
uvicorn backend.main:app --port 8000
# Navigate to http://localhost:5173 after: cd frontend && npm run dev
```

---

## Screenshots

### 1. `dashboard_overview.png`
**What to show:** The main dashboard landing page with:
- Active experiment summary card (round count, status)
- Client trust score panel (coloured badges: green/amber/red)
- Live accuracy / ASR chart (two lines across 20 rounds)
- Alert feed with at least one flagged event

**Capture:** After running a `multikrum+guard` experiment for all 20 rounds.

---

### 2. `trust_score_heatmap.png`
**What to show:** The reputation heatmap view:
- X-axis: FL rounds (0–19)
- Y-axis: client IDs (c_00 – c_11)
- Cells: colour-coded trust score (blue=high, red=low)
- Malicious clients (c_02, c_05, c_09) should visibly drift toward red over rounds

**Endpoint:** `GET /api/v1/experiments/{id}/reputation-heatmap`

---

### 3. `asr_curve.png`
**What to show:** The per-round ASR comparison chart:
- Three lines: FedAvg (high, ~99%), Multi-Krum (~26%), Multi-Krum+Guard (~26%)
- X-axis: FL round (0–19)
- Y-axis: Attack Success Rate (0–1)
- Annotation marking the 73 pp reduction from FedAvg → Krum

**Data source:** `experiments/demo_results.json` round-by-round metrics

---

### 4. `attack_visualization.png`
**What to show:** The attack visualization panel:
- Side-by-side: clean sample vs. poisoned sample (trigger highlighted)
- Trigger heatmap (features 0–2 set to 6.0)
- Confidence bars: clean prediction vs. backdoor prediction

---

### 5. `explainability_panel.png`
**What to show:** The SHAP / feature importance view:
- Bar chart of top-10 feature importance scores
- Features 0–2 (the trigger block) should dominate on poisoned inputs
- Human-readable reason string from the TrustLedger entry

---

### 6. `alert_feed.png`
**What to show:** The alert/event feed:
- At least 3 L1 collusion-cluster alerts (round, client cluster, score)
- One L3 input-flagged alert (input_id, entropy score, boundary)
- Severity badges (low / medium / high)

---

## Placeholder Files

Until real screenshots are captured, the following placeholder files are committed:

| File | Status |
|---|---|
| `docs/screenshots/dashboard_overview.png` | 📋 Placeholder |
| `docs/screenshots/trust_score_heatmap.png` | 📋 Placeholder |
| `docs/screenshots/asr_curve.png` | 📋 Placeholder |
| `docs/screenshots/attack_visualization.png` | 📋 Placeholder |
| `docs/screenshots/explainability_panel.png` | 📋 Placeholder |
| `docs/screenshots/alert_feed.png` | 📋 Placeholder |

To replace a placeholder: capture the screenshot and overwrite the file.
The README references these paths directly.
