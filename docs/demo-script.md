# Five-minute demo script

## 0:00–0:40 — Frame the two overloads

“OpenPoke has useful persistent execution-agent identities, but it puts every
identity into every interaction prompt and then loads the chosen identity's
entire lifetime log. I bounded both attention surfaces without deleting durable
identity or raw history.”

Show the before/after table at the top of `README.md`.

## 0:40–1:20 — Run the narrative demo

```bash
.venv/bin/python -m evals.demo
```

Point out:

- the current 1,000-name roster is 41,999 prompt characters;
- a 1,000-record query returns one relevant candidate under the hard cap of
  five;
- paraphrased work reuses, novel work creates, and ambiguous work abstains; and
- the 10,000-entry history drops from 1,322,499 to 4,376 prompt characters
  while raw history remains unchanged; these depth values are explicitly labeled
  as a replay from the generated evidence file and evaluated commit.

## 1:20–2:10 — Show the directory and bounded prompt

Open:

- `server/services/execution/models.py`
- `server/services/execution/directory.py`
- `server/agents/interaction_agent/agent.py`

Explain that UUID identity is immutable, names remain display metadata, legacy
name lists migrate in place, and the interaction agent receives at most five
stable-ID candidates with purposes and relevance hints.

## 2:10–2:50 — Show safe decisions and dispatch

Open:

- `server/services/execution/retrieval.py`
- `server/services/execution/routing.py`
- `server/agents/interaction_agent/tools.py`

Explain the separation between retrieval and final routing. Unknown IDs fail
closed, creation requires name plus purpose, and repeated creation calls within
one turn are idempotent by intent token or normalized-name fallback. Distinct
intent tokens permit deliberate multi-create behavior. Stable IDs also isolate
execution logs for names that would otherwise share a filesystem slug.

## 2:50–3:30 — Show bounded memory

Open `server/services/execution/context_policy.py`.

Explain summary + recent complete episodes + explicit omission marker. Mention
the independent episode and character budgets, oversized-entry marker, and
append-only raw-log preservation.

## 3:30–4:25 — Show comparative evidence

Open `evals/results/report.md`.

Compare the three checked-in test-partition breadth strategies, then the depth scale. Call out
the exact-name baseline's routing miss and abstention error. State explicitly
that the full-roster “accuracy” strategy is a deterministic proxy, not a live
model claim.

## 4:25–5:00 — Verification and trade-offs

```bash
.venv/bin/python -m pytest -q
```

Close with the limitations: small synthetic corpus, lexical rather than
embedding retrieval, no new summary generator, and no live Gmail/model eval.
Mention the longer-term task/entity ledger as a deeper architectural direction
and Jev as an optional second-stage router over the already bounded candidate
set.

## Optional Extension: Live Evaluation Lab UI

For interviews with an interactive browser or live paired comparison:
1. Follow [`docs/interview-runbook.md`](interview-runbook.md).
2. Start the lab via `python -m evals.live_lab.cli start`.
3. Open `http://127.0.0.1:3000/lab` to show real-time side-by-side execution, candidate filtering, context bounding, and layer scorecards.
