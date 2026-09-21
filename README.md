# OpenPoke agent overload take-home

This repository extends [Shlok Khemani's OpenPoke](https://github.com/shlokkhemani/openpoke),
an open-source personal assistant with one interaction agent and persistent
execution-agent identities. The take-home solves two compounding forms of agent
overload while preserving that continuity:

| Dimension | Current behavior | Bounded solution |
| --- | --- | --- |
| **Roster breadth** | Every execution-agent name is injected into every interaction turn. | A lifecycle-aware directory retrieves at most five candidates, then explicitly chooses reuse, create-new, or abstain. |
| **History depth** | A reused execution agent loads its complete lifetime log by default. | The selected identity receives an optional durable summary plus a bounded suffix of recent complete episodes. |

Persistent identity remains durable. Execution runtimes remain ephemeral. Raw
logs are never deleted or rewritten by the context policy.

## Measured outcome

The committed credential-free evaluation uses a checked-in 40-case corpus (20
development and 20 test-partition cases) plus deterministic scale fixtures. The
test partition is reproducible, but it is not a sealed set whose cases can be
proven uninspected during development.

- Test-partition hybrid routing: **100% top-5 recall**, **100% decision accuracy**,
  **0% wrong reuse**, and **0% duplicate creation** on this corpus revision.
- A 1,000-identity directory exposes at most five candidates; measured local
  directory-backed retrieval, routing, and production rendering were **27.818
  ms p50 / 45.243 ms p95** across 30 dedicated runs after three warm-ups.
- At 10,000 raw history entries, full rehydration renders **1,322,499
  characters** versus **4,376 characters** for the bounded policy, with eight
  recent episodes and the raw log unchanged.

These numbers do not prove live-model or Gmail task success. The current-system
routing comparison is explicitly an exact-name proxy; an unpinned live model is
not reproducible offline. See [the evaluation methodology](docs/evaluation.md)
and [the generated report](evals/results/report.md).

## Architecture

```text
User turn + bounded conversation context
                 │
                 ▼
        deterministic retrieval
     (directory of arbitrary size)
                 │
                 ▼
       ≤ 5 explainable candidates
                 │
                 ▼
        reuse / create / abstain
                 │
        stable-ID dispatch only
                 │
                 ▼
 summary + recent complete episodes
     (raw append-only log retained)
```

The implementation boundaries and failure behavior are documented in
[architecture.md](docs/architecture.md). The decision record explains why the
solution keeps persistent identities rather than replacing every agent with an
ephemeral task worker.

## Reproduce without credentials

Requirements: Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r server/requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m evals.runner --corpus evals/agent_routing_cases.jsonl \
  --json evals/results/hybrid_directory.json --report evals/results/report.md
.venv/bin/python -m evals.demo
```

The test suite, evaluator, and demo require no OpenRouter, Gmail, or Composio
credentials. The evaluator rewrites:

- `evals/results/hybrid_directory.json` — complete machine-readable metrics,
  observations, evaluated commit, overall and per-partition corpus hashes,
  configuration, exact evaluator command, and environment.
- `evals/results/report.md` — reviewer-readable breadth/depth comparison and
  failure analysis.

The original behavior baseline can be regenerated separately:

```bash
.venv/bin/python -m evals.baseline
```

## Five-minute review

Run `.venv/bin/python -m evals.demo`, then open the generated report. The exact
talk track is in [demo-script.md](docs/demo-script.md). It covers:

1. current all-roster prompt growth;
2. retrieval from 1,000 identities;
3. paraphrased reuse, novel creation, and ambiguous abstention;
4. the hard five-candidate cap;
5. bounded context from 10,000 raw entries; and
6. an honest baseline routing and abstention failure.

## Project map

- `server/services/execution/directory.py` — atomic identity persistence,
  legacy migration, lifecycle, and stable IDs.
- `server/services/execution/retrieval.py` — deterministic hybrid candidate
  scoring and hard top-K bound.
- `server/services/execution/routing.py` — explicit reuse/create/abstain policy.
- `server/services/execution/context_policy.py` — bounded history rendering and
  measurements.
- `server/agents/interaction_agent/` — bounded prompt integration and stable-ID
  dispatch.
- `evals/` — corpus, fixtures, baselines, strategies, metrics, runner, demo,
  and committed results.
- `server/tests/` — credential-free unit, integration, scale, and regression
  coverage.
- `docs/` — architecture, evaluation, decision record, and demo guide.

## Run the original application

The original OpenPoke web application still requires Node.js 18+, npm 9+, an
OpenRouter key, and Composio credentials for Gmail:

```bash
cp .env.example .env
.venv/bin/python -m pip install -r server/requirements.txt
npm install --prefix web
.venv/bin/python -m server.server --reload
npm run dev --prefix web
```

Open `http://127.0.0.1:3000` and connect Gmail from Settings. Optional routing
and context budgets are listed in `.env.example`; the default offline path does
not require a `.env` file.

## Local Evaluation Lab UI

The separate interview route is **http://127.0.0.1:3000/lab**. Both `npm run dev
--prefix web` and `npm run start --prefix web` bind only to `127.0.0.1:3000`.
The original chat remains at `/`. The test tooling requires Node 22.12+ or a
supported newer LTS release (verified with Node 24.19.0).

The browser talks only to explicit `/api/lab` handlers. They forward to the fixed
enhanced origin `http://127.0.0.1:8002/api/v1/lab`, validate strict response
contracts, reject unknown fields/versions and private data, and never forward
browser credentials or arbitrary URLs. Run IDs must be UUIDs. Starts require a
same-origin request and a fresh server `runnable` preflight result. The start body
is the current Python contract: `request_id` plus `scenario_ids`; repetitions are
selected by the server. Failed starts are not automatically retried. A lost start
response locks further submission; interrupted status checks can resume reads
without starting another scenario. Terminal states stop polling.

**Integration boundary:** Tasks 07–10 currently supply scenarios and paired-run
contracts, but do not implement `/lab/preflight`, `/lab/gmail/status`, or
`/lab/gmail/link`. Their frontend contracts are explicitly prospective. The UI
fails closed on these missing routes; it does not manufacture readiness or
perform Gmail/model calls. The backend must supply the server-owned readiness
decision covering both revisions/backends, fixture equivalence, identical model
and configuration, Gmail policy, and budget before the UI can start a run.

The prospective version-1 preflight wire shape is defined in
`web/lib/lab/schema.ts`: `runnable`, `blockers`, `warnings`, `baseline`, `enhanced`,
`fixture_equivalence`, `gmail_safety`, and `budget`, all required. Each system has
`reachable`, `revision`, `model`, and `config_fingerprint`; budget values are
server-provided text. Gmail proxy responses currently accept sanitized status
and message fields only, with no OAuth URL or credential handling; an actual
handoff contract remains future integration work. Detailed evidence panels,
Gmail handoff UI, and full browser/accessibility coverage are Task 12.

```bash
npm test --prefix web
npm run typecheck --prefix web
npm run lint --prefix web
npm run build --prefix web
```

`test:watch` provides watch mode; `test:e2e` is configured for Task 12's future
Playwright suite. Contract tests use `web/tests/fixtures/backend.json`, generated
offline from the current Python models by
`.venv/bin/python web/tests/export_lab_fixtures.py`. This exporter prints sanitized
JSON only and makes no provider or Gmail calls. The preflight fixture is separately
marked prospective and is not evidence of a working backend preflight endpoint.

## Known limitations and future work

- The offline corpus is intentionally small and synthetic; production data will
  contain harder aliases, stale metadata, multilingual text, and adversarial
  ambiguity.
- Lexical retrieval is deterministic and dependency-light but not semantic.
  Embeddings or a probabilistic classifier such as Jev can be added behind the
  existing interfaces and compared with the same checked-in test-partition harness.
- This implementation consumes a durable memory summary but deliberately does
  not build an LLM summarization pipeline or semantic search over old logs.
- The prompt is bounded, but the selected agent's append-only journal is still
  read and parsed linearly before the recent-episode window is selected. A
  production-scale follow-up should add episode offsets or a backward index.
- The longer-term abstraction may be a durable task/entity ledger plus a small
  fixed set of capability workers. That redesign is out of scope for this
  five-day vertical slice.
- Cross-turn batch isolation and concurrency budgets remain an optional issue;
  they are adjacent to, but distinct from, the two measured overload axes.

## License

MIT — see [LICENSE](LICENSE). Upstream OpenPoke authorship is preserved in the
repository history.
