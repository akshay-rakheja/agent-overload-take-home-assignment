# Agent overload evaluation

- Implementation commit: `62f022707e17589b1a2da9a193348e77a8b024e8`
- Corpus revision: `c8340d12fa35e9705aa31295eba498c05a8e5cab38738aa4cfbe742e6fef256a`
- Corpus cases: 40 (20 development, 20 held-out)
- Mode: deterministic offline; no credentials required

## Held-out test results

| Strategy | Top-5 recall | MRR | Decision accuracy | Wrong reuse | Duplicate creation | Max candidates | Prompt chars mean | Case-mix p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_full_roster_exact_name_proxy | 75.0% | 0.761 | 35.0% | 0.0% | 68.8% | 1000 | 3417.6 | 0.228 |
| recency_only_top_five | 62.5% | 0.625 | 70.0% | 0.0% | 37.5% | 2 | 79.3 | 0.236 |
| hybrid_directory | 100.0% | 1.000 | 100.0% | 0.0% | 0.0% | 2 | 112.8 | 3.365 |

## Dedicated 1,000-record latency benchmark

After 3 warm-up runs, 30 measured runs produced p50 7.034 ms and p95 8.312 ms. Candidate count remained between 5 and 5.

## Held-out target assessment

- PASS — top 5 recall at least 95 percent
- PASS — decision accuracy at least 90 percent
- PASS — wrong agent reuse at most 2 percent
- PASS — candidate count at most 5
- PASS — retrieval p95 below 50 ms

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
- Hybrid directory had no held-out routing misses in this corpus revision.
- No proposed held-out target was missed on this corpus revision.

## What this does not prove

This offline harness does not prove live model behavior, Gmail task success, or production latency. It isolates deterministic identity routing, prompt growth, context bounding, and local execution cost. A live model evaluation must pin the provider/model/configuration and report repeated-run variance.
