# Offline evaluation

Run the credential-free comparison from the repository root:

```bash
.venv/bin/python -m evals.runner --corpus evals/agent_routing_cases.jsonl \
  --json evals/results/hybrid_directory.json --report evals/results/report.md
```

Run the concise reviewer demo with:

```bash
.venv/bin/python -m evals.demo
```

The runner evaluates three breadth strategies on the same checked-in 20-case
development and 20-case test partitions. The test partition is reproducible,
not a sealed or provably uninspected set:

1. `current_full_roster_exact_name_proxy` measures full-roster prompt growth and
   uses a deliberately limited exact-name routing proxy. Top-5 recall and MRR
   are not applicable because fixture order is not a retrieval ranking.
2. `recency_only_top_five` checks whether a cheap hot cache is sufficient.
3. `hybrid_directory` runs the production deterministic retriever and router.

It separately compares unbounded full-history rendering with the bounded
summary-plus-recent-episode policy at 10, 100, 1,000, and 10,000 entries.

Outputs:

- `results/baseline.json` and `results/baseline.md`: unchanged OpenPoke prompt
  growth captured before the solution.
- `results/hybrid_directory.json`: complete machine-readable observations,
  metrics, configuration, evaluated commit, overall and split hashes, exact
  command, deterministic seed status, environment, and targets.
- `results/report.md`: reviewer-readable comparison and failure analysis.

The exact-name baseline is explicitly a proxy: reproducing current live-model
routing would require a pinned provider/model and repeated nondeterministic
runs. The offline harness does not use an LLM judge or claim end-to-end Gmail
task success.
