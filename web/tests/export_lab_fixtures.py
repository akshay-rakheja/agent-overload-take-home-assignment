"""Print deterministic, offline Python serialization for the web contract fixture.

Run from repository root: .venv/bin/python web/tests/export_lab_fixtures.py
No backend, provider, Gmail, environment configuration, or model calls are made.
"""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import NAMESPACE_URL, UUID, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.live_lab.fixture_email import build_fact_manifest, render_fixture_messages
from evals.live_lab.grading import grade_scenario_sequence
from evals.live_lab.raw_observation import BaselineObservation, RawModelCall, RawPhaseTiming
from evals.live_lab.scenarios import load_controlled_scenarios
from server.agents.execution_agent.tasks.search_email import tool as search_email_tool
from server.agents.interaction_agent import agent as interaction_agent
from server.config import Settings
from server.services.evaluation_lab.baseline_adapter import adapt_baseline
from server.services.evaluation_lab.models import TraceContext, TraceEventKind
from server.services.evaluation_lab.orchestrator import (
    PairExecutionRecord,
    PairedRunResult,
    PairedSequenceScorecard,
    PersistedSideOutcome,
    RunHandle,
    RunStatus,
    StartRunRequest,
    RunTransition,
)
from server.services.evaluation_lab.redaction import redact_value
from server.services.evaluation_lab.repetitions import (
    MeasuredSystem,
    OutcomeStatus,
    build_repetition_schedule,
)
from server.services.evaluation_lab.trace import consolidate_trace, emit_trace, trace_scope
from server.services.evaluation_lab.usage import emit_usage_evidence
from server.services.execution.context_policy import ExecutionContextPolicy
from server.services.execution.models import AgentRecord, AgentStatus
from server.services.execution.retrieval import AgentRetriever
from server.services.execution.routing import AgentRouter, RoutingAction


class _CollectingSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class _FixtureDirectory:
    def __init__(self, records: tuple[AgentRecord, ...]) -> None:
        self._records = records

    def list_records(self) -> list[AgentRecord]:
        return list(self._records)


def _available(value):
    return {"availability": "available", "value": value, "reason": None}


def _unavailable(reason: str):
    return {"availability": "unavailable", "value": None, "reason": reason}


def _paired_scorecard(scenario, result, *, pair_id: UUID, repetition: int):
    graded = grade_scenario_sequence(scenario, (result,))
    return PairedSequenceScorecard(
        scenario_id=graded.scenario_id,
        system=graded.system,
        turns=graded.turns,
        identity_continuity=graded.identity_continuity,
        pair_id=pair_id,
        repetition=repetition,
    )


RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
REQUEST_ID = UUID("22222222-2222-4222-8222-222222222222")
ENHANCED_TURN_ID = UUID("55555555-5555-4555-8555-555555555555")
NOW = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
FIXTURE_RUN_KEY = "fixture-run-11111111"

scenarios = load_controlled_scenarios()
scenario = scenarios[0]
schedule = build_repetition_schedule([scenario.scenario_id], repetitions=1)
scheduled = schedule[0]
expected_agent_id = UUID("292943fa-641b-5ae6-a8d2-ab631be77ba8")

baseline_names = (
    "Instagram Security Monitor",
    "AI Video Newsletter Curator",
    "AI Video Receipt Reconciler",
    "Account Security Auditor",
    "Controlled Fixture Researcher",
    "Draft Review Assistant",
    *(f"Fixture Agent {index:03d}" for index in range(7, 101)),
)
baseline_observation = BaselineObservation(
    run_id=str(RUN_ID),
    prompt_xml_sha256="a" * 64,
    prompt_characters=18_420,
    exposed_names=baseline_names,
    roster_before=baseline_names,
    roster_after=baseline_names,
    journal_hashes_before={"instagram-security-monitor.log": "b" * 64},
    journal_hashes_after={"instagram-security-monitor.log": "c" * 64},
    inferred_action="reuse",
    inferred_name="Instagram Security Monitor",
    inference_reason="one observed historical journal append",
    final_response=(
        "reference: SEC-7419; timestamp: 2026-09-18 04:12 UTC; "
        "location: Lisbon; device: Pixel 10; verification phrase: indigo-orbit"
    ),
    raw_model_calls=(
        RawModelCall(
            component="interaction",
            call_id="baseline-fixture-call",
            model="fixture-model-v1",
            elapsed_ms=812.4,
            request_sha256="d" * 64,
            response_sha256="e" * 64,
            message_count=3,
            tool_names=("send_message_to_agent",),
            response_choice_count=1,
            response_tool_call_count=1,
            provider="fixture-provider",
            actual_provider="fixture-provider",
            usage={
                "prompt_tokens": _available(4_210),
                "completion_tokens": _available(148),
                "cached_tokens": _available(0),
                "total_tokens": _available(4_358),
                "provider_cost_usd": _available(0.0142),
                "estimated_cost_usd": _unavailable("estimated cost was not supplied"),
            },
        ),
    ),
    raw_phase_timings=(
        RawPhaseTiming(
            phase="total_run",
            started_monotonic_ns=100,
            finished_monotonic_ns=812_400_100,
            elapsed_ns=812_400_000,
        ),
    ),
    errors=(),
)
baseline = adapt_baseline(baseline_observation)

records = (
    AgentRecord(
        agent_id=expected_agent_id,
        name="Instagram Security Monitor",
        purpose="Review Instagram security notices and account-protection events",
        aliases=("Instagram Security Monitor",),
        status=AgentStatus.HOT,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        last_used_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        use_count=100,
    ),
    AgentRecord(
        agent_id=uuid5(NAMESPACE_URL, "fixture-controlled-researcher"),
        name="Controlled Fixture Researcher",
        purpose="Summarize controlled fixture references",
        aliases=("Fixture Researcher",),
        status=AgentStatus.HOT,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        last_used_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        use_count=100,
    ),
    AgentRecord(
        agent_id=uuid5(NAMESPACE_URL, "fixture-security-auditor"),
        name="Account Security Auditor",
        purpose="Review account security notices",
        aliases=("Security Auditor",),
        status=AgentStatus.DORMANT,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        last_used_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        use_count=100,
    ),
)
directory = _FixtureDirectory(records)
settings = Settings(
    agent_retrieval_top_k=5,
    agent_retrieval_min_score=0.08,
    agent_route_reuse_threshold=0.34,
    agent_route_ambiguity_margin=0.12,
    agent_routing_context_max_characters=4_000,
)
sink = _CollectingSink()
trace_context = TraceContext(
    run_id=RUN_ID,
    turn_id=ENHANCED_TURN_ID,
    system="enhanced",
    revision="enhanced@def5678",
    mode="bounded_directory",
)
query = scenario.turns[0].text
with trace_scope(trace_context, sink):
    emit_trace(
        TraceEventKind.RUN_METADATA,
        {"revision": trace_context.revision, "mode": trace_context.mode, "roster_count": 100},
    )
    with (
        patch.object(interaction_agent, "get_settings", return_value=settings),
        patch.object(
            interaction_agent,
            "AgentRetriever",
            lambda provider: AgentRetriever(provider, now=lambda: NOW, settings=settings),
        ),
        patch.object(
            interaction_agent,
            "AgentRouter",
            lambda: AgentRouter(settings=settings),
        ),
        patch.object(
            interaction_agent,
            "monotonic_phase",
            lambda *_args, **_kwargs: nullcontext(),
        ),
    ):
        candidate_context = interaction_agent.build_candidate_context(
            query,
            "",
            directory=directory,
        )
        interaction_agent.render_agent_candidates(candidate_context)

    assert candidate_context.decision.action is RoutingAction.REUSE
    assert candidate_context.decision.agent_id == expected_agent_id
    assert 0 < len(candidate_context.prompt_candidates) <= 5
    assert [candidate.agent_id for candidate in candidate_context.candidates] == [
        expected_agent_id,
        records[1].agent_id,
        records[2].agent_id,
    ]
    emit_trace(
        TraceEventKind.ROUTING_DECISION,
        {
            "action": candidate_context.decision.action.value,
            "agent_id": str(candidate_context.decision.agent_id),
            "confidence": candidate_context.decision.confidence,
            "reasons": candidate_context.decision.reasons,
            "recommendation": str(candidate_context.decision.agent_id),
        },
    )
    emit_trace(
        TraceEventKind.AUTHORIZATION,
        {"routing_action": "reuse", "authorized_ids": [str(expected_agent_id)]},
    )
    emit_trace(
        TraceEventKind.DISPATCH_ATTEMPT,
        {
            "reference_type": "reuse",
            "requested_agent_id": str(expected_agent_id),
            "routing_action": "reuse",
            "authorized_ids": [str(expected_agent_id)],
            "creation_intent_supplied": False,
            "directory_count_before": 100,
            "directory_count_before_availability": "available",
            "directory_count_before_reason": None,
        },
    )
    emit_trace(
        TraceEventKind.DISPATCH_RESULT,
        {
            "status": "accepted",
            "success": True,
            "code": "ok",
            "directory_count_before": 100,
            "directory_count_before_availability": "available",
            "directory_count_before_reason": None,
            "directory_count_after": 100,
            "directory_count_after_availability": "available",
            "directory_count_after_reason": None,
            "selected_agent_id": str(expected_agent_id),
            "new_agent_created": False,
            "idempotent_creation": False,
            "journal_sha256_before": "f" * 64,
            "journal_sha256_after": "0" * 64,
        },
    )
    emit_trace(
        TraceEventKind.IDENTITY,
        {
            "selected": {
                "agent_id": str(expected_agent_id),
                "name": "Instagram Security Monitor",
                "status": "hot",
            },
            "delta": {
                "directory_count_before": 100,
                "directory_count_before_availability": "available",
                "directory_count_before_reason": None,
                "directory_count_after": 100,
                "directory_count_after_availability": "available",
                "directory_count_after_reason": None,
                "journal_sha256_before": "f" * 64,
                "journal_sha256_after": "0" * 64,
            },
            "idempotent_creation": False,
        },
    )

    gmail_query = scenario.gmail.query
    emit_trace(
        TraceEventKind.GMAIL_EVIDENCE,
        {
            "boundary": "gmail_client",
            "operation_name": "GMAIL_FETCH_EMAILS",
            "stage": "completed",
            "allowed": True,
            "policy_code": "allowed_read_only",
            "sdk_executed": True,
            "result_count": 1,
            "has_more": False,
            "observation_availability": "available",
            "observation_reason": None,
        },
    )
    messages = render_fixture_messages(FIXTURE_RUN_KEY)
    manifest = build_fact_manifest(messages)
    with search_email_tool.fixture_fact_manifest_scope(manifest):
        controlled_fixture_evidence = search_email_tool._controlled_fixture_evidence(
            ["SEC-7419"]
        )
    emit_trace(
        TraceEventKind.GMAIL_EVIDENCE,
        {
            "boundary": "email_search_task",
            "operation_name": "GMAIL_FETCH_EMAILS",
            "stage": "completed",
            "result_count": 1,
            "has_more": False,
            "attachment_count": 0,
            "query_sha256": hashlib.sha256(gmail_query.encode("utf-8")).hexdigest(),
            "fact_ids": ["SEC-7419"],
            "controlled_fixture_evidence": controlled_fixture_evidence,
        },
    )

    history_entries = tuple(
        (
            "agent_request" if index % 2 == 0 else "agent_response",
            f"2026-09-20T12:{index % 60:02d}:00Z",
            f"Fixture history entry {index:05d} for bounded context evidence.",
        )
        for index in range(10_000)
    )
    context = ExecutionContextPolicy(
        max_recent_episodes=8,
        max_characters=4_000,
    ).render(history_entries, memory_summary="Controlled fixture summary.")
    emit_trace(TraceEventKind.CONTEXT_METRICS, asdict(context.metrics))
    emit_trace(
        TraceEventKind.PHASE_TIMING,
        {
            "phase": "retrieval_routing",
            "started_monotonic_ns": 10,
            "finished_monotonic_ns": 19_300_010,
            "elapsed_ns": 19_300_000,
        },
    )
    emit_trace(
        TraceEventKind.PHASE_TIMING,
        {
            "phase": "total_run",
            "started_monotonic_ns": 100,
            "finished_monotonic_ns": 436_800_100,
            "elapsed_ns": 436_800_000,
        },
    )
    emit_usage_evidence(
        {
            "usage": {
                "prompt_tokens": 968,
                "completion_tokens": 126,
                "cached_tokens": 0,
                "total_tokens": 1_094,
                "cost": 0.0048,
            }
        },
        role="interaction",
        call_id="enhanced-fixture-call",
        attempt=0,
    )
    emit_trace(
        TraceEventKind.FINAL_RESPONSE,
        {
            "response": (
                "reference: SEC-7419; timestamp: 2026-09-18 04:12 UTC; "
                "location: Lisbon; device: Pixel 10; verification phrase: indigo-orbit"
            )
        },
    )

enhanced = consolidate_trace(sink.events)
baseline_card = _paired_scorecard(
    scenario,
    baseline,
    pair_id=scheduled.pair_id,
    repetition=scheduled.repetition,
)
enhanced_card = _paired_scorecard(
    scenario,
    enhanced,
    pair_id=scheduled.pair_id,
    repetition=scheduled.repetition,
)

request = StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
evidence_result = PairedRunResult(
    run_id=RUN_ID,
    request=request,
    status=RunStatus.COMPLETE,
    generation=4,
    schedule=schedule,
    pairs=(
        PairExecutionRecord(
            scheduled=scheduled,
            outcomes=(
                PersistedSideOutcome(
                    system=MeasuredSystem.BASELINE,
                    status=OutcomeStatus.SUCCESS,
                    model_id="fixture-model-v1",
                    results=(baseline,),
                ),
                PersistedSideOutcome(
                    system=MeasuredSystem.ENHANCED,
                    status=OutcomeStatus.SUCCESS,
                    model_id="fixture-model-v1",
                    results=(enhanced,),
                ),
            ),
        ),
    ),
    scorecards=(baseline_card, enhanced_card),
    transitions=(
        RunTransition(sequence=1, status=RunStatus.QUEUED, occurred_at=NOW),
        RunTransition(sequence=2, status=RunStatus.COMPLETE, occurred_at=NOW),
    ),
    trace=(),
    created_at=NOW,
    updated_at=NOW,
)

partial_result = PairedRunResult(
    run_id=RUN_ID,
    request=request,
    status=RunStatus.PARTIAL_FAILURE,
    generation=3,
    schedule=schedule,
    pairs=(
        PairExecutionRecord(
            scheduled=scheduled,
            outcomes=(
                PersistedSideOutcome(
                    system=MeasuredSystem.BASELINE,
                    status=OutcomeStatus.SUCCESS,
                    model_id="fixture-model-v1",
                    results=(baseline,),
                ),
                PersistedSideOutcome(
                    system=MeasuredSystem.ENHANCED,
                    status=OutcomeStatus.TIMEOUT,
                    reason="Enhanced side exceeded its time limit.",
                ),
            ),
        ),
    ),
    scorecards=(baseline_card,),
    transitions=(
        RunTransition(sequence=1, status=RunStatus.PARTIAL_FAILURE, occurred_at=NOW),
    ),
    trace=(),
    created_at=NOW,
    updated_at=NOW,
)

listed = {
    "schema_version": 1,
    "scenario_count": len(scenarios),
    "scenarios": [
        {
            "scenario_id": item.scenario_id,
            "track": item.track.value,
            "family": item.family,
            "title": item.title,
            "repetitions": item.repetitions,
            "optional": item.optional,
            "budget_guarded": item.budget_guarded,
            "reset_profile": item.reset_profile.model_dump(mode="json"),
            "turn_count": len(item.turns),
        }
        for item in scenarios
    ],
}

print(
    json.dumps(
        redact_value(
            {
                "scenarios": listed,
                "system_result": baseline.model_dump(mode="json"),
                "handle": RunHandle(
                    run_id=RUN_ID,
                    request_id=REQUEST_ID,
                    status=RunStatus.QUEUED,
                ).model_dump(mode="json"),
                "run": partial_result.model_dump(mode="json"),
                "evidence_run": evidence_result.model_dump(mode="json"),
            }
        ),
        indent=2,
        sort_keys=True,
    )
)
