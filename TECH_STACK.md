# TECH_STACK.md

| Layer | Choice | Why |
|---|---|---|
| FL simulation | Flower (`flwr`) | Actively maintained, simulates many clients on one machine, easy to swap in a real multi-process deployment later |
| Model framework | PyTorch | Ecosystem fit for activation-hooking (needed for L3 signal 2) and gradient access (needed for L1) |
| Aggregation baselines | Custom Multi-Krum / trimmed-mean (small, implement directly — no heavy dependency needed) | Transparent, easy to extend with collusion clustering |
| L2 optimization | PyTorch autograd, Adam | Matches Neural Cleanse's own optimizer choice, well understood |
| Backend API | FastAPI | Lightweight, serves precomputed JSON artifacts to the dashboard |
| Dashboard | React + Recharts | Matches this environment's supported artifact stack; no build-step surprises |
| Experiment tracking | Plain JSON-lines logs + a `experiments/` folder convention | No external service dependency (avoids demo-day network risk) |

Deliberately avoided: heavyweight experiment-tracking services (wandb/mlflow servers)
and real multi-node deployment infra — unnecessary risk for a 4-week solo competition
timeline where the deliverable is a reproducible local/simulated demo.
