# assets/

Generated, publication-quality figures embedded in the root [`README.md`](../README.md).

| File | Description |
|---|---|
| `defense_stack.png` | The five-layer (L1–L5) backdoor "immune system" funnel |
| `asr_comparison.png` | Attack-success-rate across the defense pipeline (FedAvg → Multi-Krum → +Guard → +L5) |
| `remediation_efficacy.png` | ASR before/after remediation, per strategy, vs. acceptance threshold |
| `remediation_tradeoff.png` | ASR reduction vs. clean-accuracy retention scatter |

## Regenerate

All charts are **data-driven** — they read the real experiment artifacts in
`experiments/` when present and fall back to committed reference values otherwise:

```bash
python scripts/run_remediation_demo.py   # refresh experiments/*.json
python scripts/generate_charts.py        # rewrite assets/*.png
```

Pure matplotlib (Agg backend) + numpy — no display or network required.
