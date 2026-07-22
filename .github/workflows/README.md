# .github/workflows/

CI pipeline: runs `TESTING.md` unit + integration tiers on every push. Planned file:
`ci.yml` (Week 1). Benchmark tier (`BENCHMARK.md`) is intentionally NOT run in CI —
too slow for per-commit feedback; triggered manually per `TESTING.md` §1.
