# ADR: Preserve persistent identities behind a bounded Agent Directory

- Status: accepted
- Date: 2026-09-19

## Context

OpenPoke models durable relationships and ongoing tasks as named execution
agents. The names and lifetime logs are useful continuity, but the original
implementation injects every name into every supervisor turn and uses exact
display-name equality for reuse. Reused agents also load their full log.

The take-home needs a code-complete, measurable improvement that remains close
enough to OpenPoke to compare before and after within five days.

## Decision

Keep persistent execution-agent identity, but move it into a versioned,
lifecycle-aware Agent Directory with immutable UUIDs. Retrieve a maximum of
five candidates, route explicitly to reuse/create/abstain, dispatch reuse by
stable ID, and bound the selected identity's prompt-visible log.

Raw logs remain append-only. Execution processes remain ephemeral. Archived
records are recoverable and never automatically deleted.

## Why this option

- It fixes the prompt and choice-set scaling problem directly.
- It preserves relationship continuity and legacy data.
- It creates deterministic seams for tests and comparative evaluation.
- It avoids requiring embeddings, a provider-specific classifier, or live
  credentials for reviewers.
- It can later host semantic retrieval or a probabilistic second-stage router
  without changing dispatch safety.

## Alternatives considered

### Make every worker and identity ephemeral

This bounds runtime lifetime but does not answer where durable person, thread,
approval, reminder, and task state lives. It would be a broader rewrite and
would remove the continuity OpenPoke is demonstrating.

### Keep the roster and only archive old names

Archiving helps prompt size but cannot reliably recover a relevant old identity,
resolve aliases, or distinguish a new task from ambiguity.

### Use embeddings or Jev over every identity

This moves the same unbounded choice set into another model call, adds provider
availability and cost, and weakens reproducibility. Semantic/probabilistic
routing remains viable after deterministic top-K retrieval.

### Replace identities with a task/entity ledger

This is likely the cleanest longer-term abstraction: durable state belongs to
entities and tasks, while a small worker pool supplies capabilities. It is
recorded as future work because it prevents a focused before/after comparison
and exceeds the take-home window.

## Consequences

Positive:

- prompt-visible candidates and history now have hard bounds;
- stable IDs eliminate name ambiguity at dispatch;
- routing behavior is explainable and measurable; and
- migration retains existing logs and caller compatibility.

Trade-offs:

- lexical retrieval will miss semantic relationships absent from metadata;
- directory metadata and summaries must remain useful over time;
- local retrieval still scans all records in this implementation; and
- a 40-case synthetic corpus cannot establish production generalization.

