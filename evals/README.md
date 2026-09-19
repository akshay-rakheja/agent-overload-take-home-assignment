# Offline evaluation

Run the credential-free comparison from the repository root:

```bash
.venv/bin/python -m evals.runner
```

The runner evaluates three breadth strategies on the same 20-case development
and 20-case held-out splits:

1. `current_full_roster_exact_name_proxy` measures full-roster prompt growth and
   uses a deliberately limited exact-name routing proxy.
2. `recency_only_top_five` checks whether a cheap hot cache is sufficient.
3. `hybrid_directory` runs the production deterministic retriever and router.

It separately compares unbounded full-history rendering with the bounded
summary-plus-recent-episode policy at 10, 100, 1,000, and 10,000 entries.

Outputs:

- `results/baseline.json` and `results/baseline.md`: unchanged OpenPoke prompt
  growth captured before the solution.
- `results/hybrid_directory.json`: complete machine-readable observations,
  metrics, configuration, commit, corpus revision, environment, and targets.
- `results/report.md`: reviewer-readable comparison and failure analysis.

The exact-name baseline is explicitly a proxy: reproducing current live-model
routing would require a pinned provider/model and repeated nondeterministic
runs. The offline harness does not use an LLM judge or claim end-to-end Gmail
task success.
