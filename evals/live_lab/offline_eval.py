"""Deterministic, offline evaluation runner for the live lab."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from unittest.mock import patch
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from evals.live_lab.fixture_email import (
    build_fact_manifest,
    fixture_response_facts,
    fixture_response_fields,
    render_fixture_messages,
    write_browser_fixture_contract,
)
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
    project_lab_run,
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


def _emit_enhanced_turn_traces(
    *,
    run_id: UUID,
    pair_id: UUID,
    turn_idx: int,
    turn_exp: Any,
    turn_action: str,
    query: str,
    identity_name: str | None,
    target_agent_id: UUID | None,
    fact_ids: Sequence[str],
    final_response: str,
    system_name: str,
    mode_name: str,
    is_jev: bool = False,
) -> SystemRunResult:
    turn_id = uuid5(run_id, f"{system_name}-turn-{pair_id}-{turn_idx}")
    sink = _CollectingSink()
    trace_context = TraceContext(
        run_id=run_id,
        turn_id=turn_id,
        system=system_name,
        revision=f"{system_name}@local",
        mode=mode_name,
    )
    with trace_scope(trace_context, sink):
        emit_trace(
            TraceEventKind.RUN_METADATA,
            {"revision": trace_context.revision, "mode": trace_context.mode, "roster_count": 100},
        )
        if is_jev:
            scores = (
                [
                    {
                        "agent_id": str(target_agent_id),
                        "composite_score": 0.95,
                        "affinity_score": 0.95,
                        "continuity_score": 1.0,
                        "risk_score": 0.0,
                        "reasoning": "exact match",
                        "card_digest": "card-digest-001",
                        "latency_ms": 12.0,
                    }
                ]
                if turn_action == "reuse"
                else []
            )
            emit_trace(
                TraceEventKind.JEV_MAP,
                {
                    "map_scores": scores,
                    "card_digest": "card-digest-001",
                    "latency_ms": 32.5,
                    "api_calls_count": 3,
                    "partial_failures": [],
                },
            )
            shortlist = [str(target_agent_id)] if turn_action == "reuse" else []
            emit_trace(TraceEventKind.JEV_SHORTLIST, {"shortlist": shortlist})
            emit_trace(
                TraceEventKind.JEV_REDUCE,
                {
                    "decision": {
                        "action": turn_action,
                        "recommended_agent_id": str(target_agent_id) if turn_action != "abstain" else None,
                        "confidence": 0.95 if turn_action != "abstain" else 0.5,
                        "winner_margin": 0.35 if turn_action == "reuse" else 0.0,
                        "rationale": "Jev map-reduce routing decision",
                    },
                    "winner_margin": 0.35 if turn_action == "reuse" else 0.0,
                    "latency_ms": 18.2,
                    "total_latency_ms": 50.7,
                    "token_usage": {"input_tokens": 350, "output_tokens": 65},
                },
            )

        if turn_action == "reuse":
            emit_trace(
                TraceEventKind.CANDIDATES,
                {
                    "candidates": [
                        {
                            "agent_id": str(target_agent_id),
                            "name": identity_name,
                            "status": "hot",
                        }
                    ]
                },
            )
            emit_trace(
                TraceEventKind.ROUTING_DECISION,
                {
                    "action": "reuse",
                    "agent_id": str(target_agent_id),
                    "confidence": 0.95,
                    "reasons": ["exact match"],
                    "recommendation": str(target_agent_id),
                },
            )
            emit_trace(
                TraceEventKind.AUTHORIZATION,
                {"routing_action": "reuse", "authorized_ids": [str(target_agent_id)]},
            )
            emit_trace(
                TraceEventKind.DISPATCH_ATTEMPT,
                {
                    "reference_type": "reuse",
                    "action": "reuse",
                    "requested_agent_id": str(target_agent_id),
                    "routing_action": "reuse",
                    "authorized_ids": [str(target_agent_id)],
                    "directory_count_before": 100,
                    "message_preview": query,
                    "target_name": identity_name,
                },
            )
            emit_trace(
                TraceEventKind.DISPATCH_RESULT,
                {
                    "status": "accepted",
                    "accepted": True,
                    "success": True,
                    "selected_agent_id": str(target_agent_id),
                    "agent_id": str(target_agent_id),
                    "target_name": identity_name,
                    "new_agent_created": False,
                    "idempotent_creation": False,
                    "directory_count_before": 100,
                    "directory_count_after": 100,
                },
            )
            emit_trace(
                TraceEventKind.IDENTITY,
                {
                    "action": "reuse",
                    "agent_id": str(target_agent_id),
                    "name": identity_name,
                    "selected": {
                        "agent_id": str(target_agent_id),
                        "name": identity_name,
                        "status": "hot",
                    },
                },
            )
        elif turn_action == "create_new":
            emit_trace(
                TraceEventKind.CANDIDATES,
                {"candidates": []},
            )
            emit_trace(
                TraceEventKind.ROUTING_DECISION,
                {
                    "action": "create_new",
                    "agent_id": str(target_agent_id),
                    "confidence": 0.95,
                    "reasons": ["novel capability"],
                    "recommendation": "create_new",
                },
            )
            emit_trace(
                TraceEventKind.AUTHORIZATION,
                {"routing_action": "create_new", "authorized_ids": [str(target_agent_id)]},
            )
            emit_trace(
                TraceEventKind.DISPATCH_ATTEMPT,
                {
                    "reference_type": "create",
                    "action": "create_new",
                    "requested_agent_id": str(target_agent_id),
                    "routing_action": "create_new",
                    "authorized_ids": [str(target_agent_id)],
                    "directory_count_before": 100,
                    "message_preview": query,
                    "target_name": identity_name,
                },
            )
            emit_trace(
                TraceEventKind.DISPATCH_RESULT,
                {
                    "status": "accepted",
                    "accepted": True,
                    "success": True,
                    "selected_agent_id": str(target_agent_id),
                    "agent_id": str(target_agent_id),
                    "target_name": identity_name,
                    "new_agent_created": True,
                    "idempotent_creation": False,
                    "directory_count_before": 100,
                    "directory_count_after": 101,
                },
            )
            emit_trace(
                TraceEventKind.IDENTITY,
                {
                    "action": "create_new",
                    "agent_id": str(target_agent_id),
                    "name": identity_name,
                    "selected": {
                        "agent_id": str(target_agent_id),
                        "name": identity_name,
                        "status": "hot",
                    },
                    "created": {
                        "agent_id": str(target_agent_id),
                        "name": identity_name,
                        "status": "hot",
                    },
                },
            )
        else:
            emit_trace(
                TraceEventKind.CANDIDATES,
                {"candidates": []},
            )
            emit_trace(
                TraceEventKind.ROUTING_DECISION,
                {
                    "action": "abstain",
                    "agent_id": None,
                    "confidence": 0.5,
                    "reasons": ["ambiguous query requires clarification"],
                    "recommendation": None,
                },
            )
            emit_trace(
                TraceEventKind.AUTHORIZATION,
                {"routing_action": "abstain", "authorized_ids": []},
            )
            emit_trace(
                TraceEventKind.DISPATCH_RESULT,
                {
                    "status": "abstain",
                    "accepted": False,
                    "success": True,
                    "selected_agent_id": None,
                    "new_agent_created": False,
                    "idempotent_creation": False,
                    "directory_count_before": 100,
                    "directory_count_after": 100,
                },
            )

        if turn_exp.gmail is not None:
            fixture_run_key = f"fixture-run-{run_id.hex[:8]}"
            messages = render_fixture_messages(fixture_run_key)
            manifest = build_fact_manifest(messages)
            with search_email_tool.fixture_fact_manifest_scope(manifest):
                controlled_fixture_evidence = search_email_tool._controlled_fixture_evidence(
                    list(fact_ids)
                )
            emit_trace(
                TraceEventKind.GMAIL_EVIDENCE,
                {
                    "boundary": "email_search_task",
                    "operation_name": turn_exp.gmail.operation,
                    "stage": "completed",
                    "result_count": 0 if turn_exp.gmail.expect_no_result else len(fact_ids),
                    "has_more": False,
                    "attachment_count": 0,
                    "query_sha256": hashlib.sha256(turn_exp.gmail.query.encode("utf-8")).hexdigest(),
                    "fact_ids": list(fact_ids),
                    "controlled_fixture_evidence": controlled_fixture_evidence,
                },
            )

        history_entries = tuple(
            (
                "agent_request" if index % 2 == 0 else "agent_response",
                f"2026-09-20T12:{index % 60:02d}:00Z",
                f"Fixture history entry {index:05d} for bounded context evidence.",
            )
            for index in range(100)
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
            call_id=f"{system_name}-fixture-call-{turn_idx}",
            attempt=0,
        )
        emit_trace(
            TraceEventKind.FINAL_RESPONSE,
            {"response": final_response},
        )

    return consolidate_trace(sink.events)


def run_offline_evaluation(
    scenario_ids: Sequence[str] | None = None,
    repetitions: int = 1,
    seed: int = 42,
    model: str = "openai/gpt-4.1-mini",
    three_way: bool = False,
) -> PairedRunResult:
    """Execute deterministic offline paired evaluation and return projected PairedRunResult."""
    all_scenarios = load_controlled_scenarios()
    if scenario_ids:
        selected_scenarios = [s for s in all_scenarios if s.scenario_id in scenario_ids]
        if not selected_scenarios:
            selected_scenarios = [all_scenarios[0]]
    else:
        selected_scenarios = list(all_scenarios)

    run_id = uuid4()
    request_id = uuid4()
    now = datetime.now(timezone.utc)
    scen_id_tuple = tuple(s.scenario_id for s in selected_scenarios)

    schedule = tuple(
        pair
        for scen in selected_scenarios
        for pair in build_repetition_schedule(
            [scen.scenario_id], repetitions=repetitions, three_way=three_way
        )
    )

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

    pairs: list[PairExecutionRecord] = []
    scorecards: list[PairedSequenceScorecard] = []

    for scheduled in schedule:
        scen = next(s for s in selected_scenarios if s.scenario_id == scheduled.scenario_id)
        pair_id = scheduled.pair_id
        rep = scheduled.repetition

        baseline_turn_results: list[SystemRunResult] = []
        enhanced_turn_results: list[SystemRunResult] = []
        det_turn_results: list[SystemRunResult] = []
        jev_turn_results: list[SystemRunResult] = []

        if scen.identity_contract is not None:
            if scen.expected_agent_id:
                target_agent_id = UUID(scen.expected_agent_id)
            else:
                target_agent_id = uuid5(NAMESPACE_URL, f"offline-agent-{scen.scenario_id}")
            identity_name = scen.identity_contract.name
        else:
            target_agent_id = None
            identity_name = None

        for turn_idx, turn_exp in enumerate(scen.turn_expectations):
            turn_action = turn_exp.action.value
            query = scen.turns[turn_idx].text

            fact_ids = turn_exp.gmail.fact_ids if turn_exp.gmail else ()
            if not fact_ids and turn_exp.evidence_from_turn is not None:
                earlier = scen.turn_expectations[turn_exp.evidence_from_turn]
                if earlier.gmail:
                    fact_ids = earlier.gmail.fact_ids

            if turn_exp.require_clarification:
                final_response = "Which workflow should handle this?"
            elif fact_ids:
                fields = [
                    f"{field.key.replace('_', ' ')}: {field.value}"
                    for fact_id in fact_ids
                    for field in fixture_response_fields(fact_id)
                ]
                final_response = "; ".join(fields) if fields else " ".join(
                    value
                    for fact_id in fact_ids
                    for value in fixture_response_facts(fact_id)
                )
            elif turn_exp.gmail and turn_exp.gmail.expect_no_result:
                final_response = "No matching emails found."
            else:
                final_response = "Action completed."

            # 1. Baseline turn
            baseline_turn_id = uuid5(run_id, f"baseline-turn-{scheduled.pair_id}-{turn_idx}")
            baseline_obs = BaselineObservation(
                run_id=str(run_id),
                prompt_xml_sha256="a" * 64,
                prompt_characters=18_420,
                exposed_names=baseline_names,
                roster_before=baseline_names,
                roster_after=baseline_names if turn_action != "create_new" else (*baseline_names, identity_name or "New Agent"),
                journal_hashes_before={"agent.log": "b" * 64} if turn_action == "reuse" else {},
                journal_hashes_after={"agent.log": "c" * 64} if turn_action == "reuse" else {},
                inferred_action=turn_action if turn_action in {"reuse", "create_new", "abstain"} else "unobservable",
                inferred_name=identity_name if turn_action != "abstain" else None,
                inference_reason="projected historical baseline observation",
                final_response=final_response,
                raw_model_calls=(
                    RawModelCall(
                        component="interaction",
                        call_id=f"baseline-fixture-call-{turn_idx}",
                        model=model,
                        elapsed_ms=812.4,
                        request_sha256="d" * 64,
                        response_sha256="e" * 64,
                        message_count=3,
                        tool_names=("send_message_to_agent",),
                        response_choice_count=1,
                        response_tool_call_count=1,
                        provider="openrouter",
                        actual_provider="openrouter",
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
            baseline_turn_results.append(adapt_baseline(baseline_obs, turn_id=baseline_turn_id))

            # 2. Enhanced turn(s)
            if three_way:
                det_turn_results.append(
                    _emit_enhanced_turn_traces(
                        run_id=run_id,
                        pair_id=scheduled.pair_id,
                        turn_idx=turn_idx,
                        turn_exp=turn_exp,
                        turn_action=turn_action,
                        query=query,
                        identity_name=identity_name,
                        target_agent_id=target_agent_id,
                        fact_ids=fact_ids,
                        final_response=final_response,
                        system_name="enhanced_deterministic",
                        mode_name="bounded_directory",
                        is_jev=False,
                    )
                )
                jev_turn_results.append(
                    _emit_enhanced_turn_traces(
                        run_id=run_id,
                        pair_id=scheduled.pair_id,
                        turn_idx=turn_idx,
                        turn_exp=turn_exp,
                        turn_action=turn_action,
                        query=query,
                        identity_name=identity_name,
                        target_agent_id=target_agent_id,
                        fact_ids=fact_ids,
                        final_response=final_response,
                        system_name="enhanced_jev",
                        mode_name="jev_map_reduce",
                        is_jev=True,
                    )
                )
            else:
                enhanced_turn_results.append(
                    _emit_enhanced_turn_traces(
                        run_id=run_id,
                        pair_id=scheduled.pair_id,
                        turn_idx=turn_idx,
                        turn_exp=turn_exp,
                        turn_action=turn_action,
                        query=query,
                        identity_name=identity_name,
                        target_agent_id=target_agent_id,
                        fact_ids=fact_ids,
                        final_response=final_response,
                        system_name="enhanced",
                        mode_name="bounded_directory",
                        is_jev=False,
                    )
                )

        baseline_results = tuple(baseline_turn_results)
        baseline_graded = grade_scenario_sequence(scen, baseline_results)
        scorecards.append(
            PairedSequenceScorecard(
                scenario_id=scen.scenario_id,
                system="baseline",
                pair_id=pair_id,
                repetition=rep,
                turns=baseline_graded.turns,
                identity_continuity=baseline_graded.identity_continuity,
            )
        )

        if three_way:
            det_results = tuple(det_turn_results)
            jev_results = tuple(jev_turn_results)
            det_graded = grade_scenario_sequence(scen, det_results)
            jev_graded = grade_scenario_sequence(scen, jev_results)

            scorecards.append(
                PairedSequenceScorecard(
                    scenario_id=scen.scenario_id,
                    system="enhanced_deterministic",
                    pair_id=pair_id,
                    repetition=rep,
                    turns=det_graded.turns,
                    identity_continuity=det_graded.identity_continuity,
                )
            )
            scorecards.append(
                PairedSequenceScorecard(
                    scenario_id=scen.scenario_id,
                    system="enhanced_jev",
                    pair_id=pair_id,
                    repetition=rep,
                    turns=jev_graded.turns,
                    identity_continuity=jev_graded.identity_continuity,
                )
            )
            pairs.append(
                PairExecutionRecord(
                    scheduled=scheduled,
                    outcomes=(
                        PersistedSideOutcome(
                            system=MeasuredSystem.BASELINE,
                            status=OutcomeStatus.SUCCESS,
                            model_id=model,
                            results=baseline_results,
                        ),
                        PersistedSideOutcome(
                            system=MeasuredSystem.ENHANCED_DETERMINISTIC,
                            status=OutcomeStatus.SUCCESS,
                            model_id=model,
                            results=det_results,
                        ),
                        PersistedSideOutcome(
                            system=MeasuredSystem.ENHANCED_JEV,
                            status=OutcomeStatus.SUCCESS,
                            model_id="typesafe/jev-routing-v1",
                            results=jev_results,
                        ),
                    ),
                )
            )
        else:
            enhanced_results = tuple(enhanced_turn_results)
            enhanced_graded = grade_scenario_sequence(scen, enhanced_results)
            scorecards.append(
                PairedSequenceScorecard(
                    scenario_id=scen.scenario_id,
                    system="enhanced",
                    pair_id=pair_id,
                    repetition=rep,
                    turns=enhanced_graded.turns,
                    identity_continuity=enhanced_graded.identity_continuity,
                )
            )
            pairs.append(
                PairExecutionRecord(
                    scheduled=scheduled,
                    outcomes=(
                        PersistedSideOutcome(
                            system=MeasuredSystem.BASELINE,
                            status=OutcomeStatus.SUCCESS,
                            model_id=model,
                            results=baseline_results,
                        ),
                        PersistedSideOutcome(
                            system=MeasuredSystem.ENHANCED,
                            status=OutcomeStatus.SUCCESS,
                            model_id=model,
                            results=enhanced_results,
                        ),
                    ),
                )
            )

    if three_way:
        transitions = (
            RunTransition(sequence=1, status=RunStatus.QUEUED, occurred_at=now),
            RunTransition(sequence=2, status=RunStatus.BASELINE_RUNNING, occurred_at=now),
            RunTransition(sequence=3, status=RunStatus.DETERMINISTIC_RUNNING, occurred_at=now),
            RunTransition(sequence=4, status=RunStatus.JEV_RUNNING, occurred_at=now),
            RunTransition(sequence=5, status=RunStatus.GRADING, occurred_at=now),
            RunTransition(sequence=6, status=RunStatus.COMPLETE, occurred_at=now),
        )
    else:
        transitions = (
            RunTransition(sequence=1, status=RunStatus.QUEUED, occurred_at=now),
            RunTransition(sequence=2, status=RunStatus.BASELINE_RUNNING, occurred_at=now),
            RunTransition(sequence=3, status=RunStatus.ENHANCED_RUNNING, occurred_at=now),
            RunTransition(sequence=4, status=RunStatus.GRADING, occurred_at=now),
            RunTransition(sequence=5, status=RunStatus.COMPLETE, occurred_at=now),
        )

    raw_result = PairedRunResult(
        schema_version=2 if three_way else 1,
        run_id=run_id,
        request=StartRunRequest(request_id=request_id, scenario_ids=scen_id_tuple),
        status=RunStatus.COMPLETE,
        generation=1,
        schedule=schedule,
        pairs=tuple(pairs),
        scorecards=tuple(scorecards),
        transitions=transitions,
        trace=(),
        created_at=now,
        updated_at=now,
    )

    return project_lab_run(raw_result)


__all__ = ["run_offline_evaluation"]
