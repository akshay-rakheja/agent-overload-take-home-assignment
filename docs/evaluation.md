# Evaluation methodology and results

## What is evaluated

The harness measures two axes separately so one improvement cannot hide a
regression in the other.

### Breadth strategies

1. **Current full-roster exact-name proxy** renders every identity and reuses
   only when a complete display name appears verbatim. This measures current
   prompt growth, but it is deliberately labeled a proxy for model routing.
2. **Recency-only top-five** selects the five most recent hot identities, then
   applies the same deterministic relevance/router policy. It tests whether a
   much cheaper cache is sufficient.
3. **Hybrid directory** evaluates the production local retriever and
   reuse/create/abstain router.

### Depth strategies

1. **Full history** renders every execution-log entry, matching the original
   default behavior.
2. **Bounded context** renders an optional summary plus at most eight recent
   complete episodes under a 12,000-character ceiling.

## Corpus discipline

`evals/agent_routing_cases.jsonl` contains 40 labeled cases across exact and
paraphrased follow-ups, contextual pronouns, old relevant identities, recent
distractors, similar identities, novel work, ambiguity, archived recovery,
Unicode/punctuation, and 10/100/500/1,000-record rosters.

- Twenty development cases were available while setting weights and thresholds.
- Twenty held-out cases were not inspected until the configuration was frozen.
- The final report records the corpus SHA-256 revision. No held-out case was
  relabeled after evaluation.
- Identity-routing labels are deterministic ground truth; no LLM judge is used.

## Metrics

Retrieval and routing are reported independently:

- top-5 recall and mean reciprocal rank;
- final reuse/create/abstain accuracy;
- wrong-agent reuse and duplicate creation rates;
- abstention precision;
- candidate count and rendered prompt characters/bytes;
- held-out case-mix retrieval p50/p95 plus a dedicated 30-run, 1,000-record
  latency benchmark after three warm-ups; and
- failures grouped by category with per-case observations.

Depth reports raw versus prompt-visible characters/bytes, included episodes,
omitted/truncated entries, summary use, render latency, raw-log preservation,
and cross-agent contamination failures.

## Frozen configuration

| Setting | Value |
| --- | ---: |
| Top K | 5 |
| Minimum retrieval score | 0.08 |
| Reuse threshold | 0.34 |
| Ambiguity margin | 0.12 |
| Exact name/alias phrase | +0.70 |
| Token overlap | up to +0.35 |
| Shared phrase | +0.12 |
| Recency | up to +0.05 |
| Prior use | up to +0.04 |
| Dormant lifecycle | -0.03 |
| Archived lifecycle | -0.08 |
| Recent episode limit | 8 |
| Rendered history limit | 12,000 characters |

## Actual held-out results

| Strategy | Top-5 recall | MRR | Accuracy | Wrong reuse | Duplicate creation | Max candidates | Prompt chars mean | Case-mix p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current full-roster proxy | 75.0% | 0.761 | 35.0% | 0.0% | 68.8% | 1,000 | 3,417.6 | 0.228 |
| Recency-only top-five | 62.5% | 0.625 | 70.0% | 0.0% | 37.5% | 2 | 79.3 | 0.236 |
| Hybrid directory | **100.0%** | **1.000** | **100.0%** | **0.0%** | **0.0%** | **2** | **112.8** | **3.365** |

The table's p95 values describe a heterogeneous 20-case held-out mix and are not
used for the scale target. The dedicated benchmark recreates the retriever on
each run against exactly 1,000 records; after three warm-ups, 30 measured runs
produced **7.03 ms p50** and **8.31 ms p95** locally. These are observations,
not service-level guarantees. Immutable record features use a bounded cache,
while changed directory records produce new features.

### Depth scaling

| Raw entries | Full prompt chars | Bounded prompt chars | Included episodes | Omitted entries | Raw log unchanged |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 10 | 1,315 | 1,388 | 3 | 0 | yes |
| 100 | 13,224 | 4,374 | 8 | 68 | yes |
| 1,000 | 132,249 | 4,375 | 8 | 968 | yes |
| 10,000 | 1,322,499 | 4,376 | 8 | 9,968 | yes |

The 10-entry bounded form is slightly larger because it adds the durable
summary; its value is the ceiling as history grows, not compression of already
small histories.

Each depth point is written through the production append-only log store. The
evaluator snapshots both the selected agent's journal and a second sentinel
agent's journal before rendering, then verifies byte-for-byte preservation and
checks that the second agent's sentinel never appears in the selected context.

## Targets and failure analysis

The hybrid passed the proposed held-out targets: at least 95% top-5 recall, at
least 90% decision accuracy, at most 2% wrong reuse, no more than five
candidates, and local 1,000-record p95 below 50 ms on the recorded run.

The report still includes failures rather than showing only the preferred path:

- the current exact-name proxy creates a duplicate on `test-exact-sam`; and
- the same proxy creates new rather than abstaining on ambiguous
  `test-ambiguous-jordan`.

The hybrid had no held-out miss on this corpus revision. That is encouraging,
not proof of generalization beyond the fixture distribution.

## Reproduce

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m evals.runner
```

Inspect `evals/results/hybrid_directory.json` for every observation and
`evals/results/report.md` for the generated comparison. Local latency will vary
slightly between runs.

## What the offline harness cannot prove

- live interaction-agent behavior under an external model;
- end-to-end Gmail correctness or user-perceived task success;
- production latency, concurrency, or provider cost;
- semantic equivalence outside lexical/metadata evidence; or
- quality of a future summary-generating model.

A live extension must pin model/provider/configuration, repeat cases, report
variance, and retain deterministic identity labels rather than using an LLM
judge as the only ground truth.
