# Agent overload evaluation

- Implementation commit: `8ac1aa8e5eca21e52744fedf8166fa25ae17bfeb`
- Corpus SHA-256: `20d2035b14650fcff4eb6b14058bcb41bb37f8fdcf45b2497067b0d0056d7877`
- Development partition SHA-256: `de5c1668299015adf6ffbf62f531860ceac1b7eac0b7abb048009364386c30aa`
- Test partition SHA-256: `bdb5a1b6757ee577e3e897e0391b057b379f5aaa3686f08e4b64088c2731876c`
- Corpus cases: 40 (20 development, 20 checked-in test)
- Exact evaluator command: `.venv/bin/python -m evals.runner --corpus evals/agent_routing_cases.jsonl --json evals/results/hybrid_directory.json --report evals/results/report.md`
- Random seed: not applicable; the harness has no random sampling
- Mode: deterministic offline; no credentials required

## Checked-in test partition results

| Strategy | Top-5 recall | MRR | Decision accuracy | Wrong reuse | Duplicate creation | Max candidates | Prompt chars mean | Case-mix p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_full_roster_exact_name_proxy | n/a | n/a | 35.0% | 0.0% | 68.8% | 1000 | 3417.6 | 0.274 |
| recency_only_top_five | 62.5% | 0.625 | 70.0% | 0.0% | 37.5% | 2 | 254.8 | 0.291 |
| hybrid_directory | 100.0% | 1.000 | 100.0% | 0.0% | 0.0% | 2 | 338.9 | 4.792 |

## Dedicated 1,000-record latency benchmark

Using `AgentDirectory.list_records` after 3 warm-up runs, 30 measured runs produced p50 27.818 ms and p95 45.243 ms. Candidate count remained between 5 and 5.

## Checked-in test partition target assessment

- PASS — top 5 recall at least 95 percent
- PASS — decision accuracy at least 90 percent
- PASS — wrong agent reuse at most 2 percent
- PASS — candidate count at most 5
- PASS — production path p95 below 50 ms

## History-depth scaling

| Raw entries | Full prompt chars | Bounded prompt chars | Included episodes | Omitted entries | Raw log unchanged |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 10 | 1315 | 1388 | 3 | 0 | yes |
| 100 | 13224 | 4374 | 8 | 68 | yes |
| 1000 | 132249 | 4375 | 8 | 968 | yes |
| 10000 | 1322499 | 4376 | 8 | 9968 | yes |

## Failure analysis

- Routing miss: `current_full_roster_exact_name_proxy` on `test-exact-sam` expected `reuse` but produced `create_new`.
- Abstention error: `current_full_roster_exact_name_proxy` on `test-ambiguous-jordan` expected `abstain` but produced `create_new`.
- Hybrid directory had no test-partition routing misses in this corpus revision.
- No proposed test-partition target was missed on this corpus revision.

## What this does not prove

This offline harness does not prove live model behavior, Gmail task success, or production latency. It isolates deterministic identity routing, prompt growth, context bounding, and local execution cost. A live model evaluation must pin the provider/model/configuration and report repeated-run variance.
