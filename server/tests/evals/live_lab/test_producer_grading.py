"""Producer-to-grader checks using only evidence emitted by real adapters/traces."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from evals.live_lab import grading as grading_module
from evals.live_lab.fixture_email import fixture_response_facts, fixture_response_fields
from evals.live_lab.grading import GradeStatus, grade_scenario
from evals.live_lab.raw_observation import BaselineObservation
from evals.live_lab.scenarios import load_controlled_scenarios
from server.services.evaluation_lab.baseline_adapter import adapt_baseline
from server.agents.interaction_agent.tools import (
    DispatchContext,
    send_message_to_agent,
)
from server.services.evaluation_lab.models import (
    TraceContext,
    TraceEvent,
    TraceEventKind,
)
from server.services.evaluation_lab.trace import (
    consolidate_trace,
    emit_trace,
    trace_scope,
)
from server.services.execution.directory import AgentDirectory
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.routing import RoutingAction


RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
TURN_ID = UUID("22222222-2222-4222-8222-222222222222")


class _CollectingSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class _FakeBatchManager:
    async def execute_agent(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(success=True)


def _dispatch_and_drain(**kwargs):
    async def run():
        result = send_message_to_agent(**kwargs)
        await asyncio.sleep(0)
        return result

    return asyncio.run(run())


def _scenario(scenario_id: str):
    return next(
        item for item in load_controlled_scenarios() if item.scenario_id == scenario_id
    )


def _field_response(fact_id: str, *values: str) -> str:
    included = set(values) if values else None
    return "; ".join(
        f"{field.key.replace('_', ' ')}: {field.value}"
        for field in fixture_response_fields(fact_id)
        if included is None or field.value in included
    )


def _baseline_observation(scenario_id: str) -> BaselineObservation:
    scenario = _scenario(scenario_id)
    before = tuple(
        name
        for name in (
            "Instagram Security Monitor",
            "AI Video Newsletter Curator",
        )
    )
    after = before
    if scenario.expected.action.value == "create_new":
        after = (*before, scenario.expected_identity_name)
    return BaselineObservation(
        run_id=str(RUN_ID),
        prompt_xml_sha256="a" * 64,
        prompt_characters=100,
        exposed_names=before,
        roster_before=before,
        roster_after=after,
        journal_hashes_before={},
        journal_hashes_after={},
        inferred_action=scenario.expected.action.value,
        inferred_name=(
            None
            if scenario.expected.action.value == "abstain"
            else scenario.expected_identity_name
        ),
        inference_reason="producer-observed roster and journal delta",
        final_response=(
            "Which workflow should handle this?"
            if scenario.expected.require_clarification
            else " ".join(
                value
                for fact_id in scenario.gmail.fact_ids
                for value in fixture_response_facts(fact_id)
            )
        ),
        raw_model_calls=(),
        errors=(),
    )


def _event(sequence: int, kind: TraceEventKind, payload: dict[str, object]) -> TraceEvent:
    return TraceEvent(
        run_id=RUN_ID,
        turn_id=TURN_ID,
        sequence=sequence,
        occurred_at=datetime(2026, 9, 21, 12, 0, tzinfo=UTC),
        system="enhanced",
        kind=kind,
        payload=payload,
    )


def _enhanced_result(scenario_id: str, *, rejected_first: bool = False):
    scenario = _scenario(scenario_id)
    action = scenario.expected.action.value
    stable_id = scenario.expected_agent_id or "33333333-3333-4333-8333-333333333333"
    events: list[tuple[TraceEventKind, dict[str, object]]] = [
        (
            TraceEventKind.CANDIDATES,
            {
                "candidates": (
                    [
                        {
                            "agent_id": stable_id,
                            "name": scenario.expected_identity_name,
                            "status": "hot",
                        }
                    ]
                    if action == "reuse"
                    else []
                )
            },
        ),
        (
            TraceEventKind.ROUTING_DECISION,
            {
                "action": action,
                "agent_id": stable_id if action == "reuse" else None,
                "recommendation": stable_id if action == "reuse" else action,
            },
        ),
        (
            TraceEventKind.AUTHORIZATION,
            {"routing_action": action, "authorized_ids": [stable_id] if action == "reuse" else []},
        ),
    ]
    if action != "abstain":
        before, after = (100, 100) if action == "reuse" else (100, 101)
        if rejected_first:
            events.extend(
                [
                    (
                        TraceEventKind.DISPATCH_ATTEMPT,
                        {
                            "reference_type": "reuse" if action == "reuse" else "create",
                            "requested_agent_id": stable_id if action == "reuse" else None,
                            "routing_action": action,
                            "authorized_ids": [stable_id] if action == "reuse" else [],
                            "directory_count_before": before,
                        },
                    ),
                    (
                        TraceEventKind.DISPATCH_RESULT,
                        {
                            "status": "rejected",
                            "success": False,
                            "code": "routing_not_authorized",
                            "directory_count_before": before,
                            "directory_count_after": before,
                        },
                    ),
                ]
            )
        events.extend(
            [
                (
                    TraceEventKind.DISPATCH_ATTEMPT,
                    {
                        "reference_type": "reuse" if action == "reuse" else "create",
                        "requested_agent_id": stable_id if action == "reuse" else None,
                        "routing_action": action,
                        "authorized_ids": [stable_id] if action == "reuse" else [],
                        "directory_count_before": before,
                    },
                ),
                (
                    TraceEventKind.DISPATCH_RESULT,
                    {
                        "status": "accepted",
                        "success": True,
                        "selected_agent_id": stable_id,
                        "new_agent_created": action == "create_new",
                        "idempotent_creation": False,
                        "directory_count_before": before,
                        "directory_count_after": after,
                    },
                ),
                (
                    TraceEventKind.IDENTITY,
                    {
                        "selected": {
                            "agent_id": stable_id,
                            "name": scenario.expected_identity_name,
                            "status": "hot",
                        },
                        **(
                            {
                                "created": {
                                    "agent_id": stable_id,
                                    "name": scenario.expected_identity_name,
                                    "status": "hot",
                                }
                            }
                            if action == "create_new"
                            else {}
                        ),
                        "delta": {
                            "directory_count_before": before,
                            "directory_count_after": after,
                        },
                        "idempotent_creation": False,
                    },
                ),
            ]
        )
    if action != "abstain":
        query_hash = hashlib.sha256(scenario.gmail.query.encode()).hexdigest()
        events.extend(
            [
                (
                    TraceEventKind.GMAIL_EVIDENCE,
                    {
                        "operation_name": scenario.gmail.operation,
                        "stage": "allowed",
                        "allowed": True,
                        "policy_code": "allowed_read_only",
                        "sdk_executed": True,
                    },
                ),
                (
                    TraceEventKind.GMAIL_EVIDENCE,
                    {
                        "operation_name": scenario.gmail.operation,
                        "stage": "completed",
                        "query_sha256": query_hash,
                        "result_count": 1,
                        "has_more": False,
                        "fact_ids": list(scenario.gmail.fact_ids),
                    },
                ),
                (
                    TraceEventKind.CONTEXT_METRICS,
                    {
                        "raw_entry_count": 4,
                        "rendered_characters": 400,
                        "omitted_entry_count": 0,
                        "included_episode_count": 1,
                    },
                ),
            ]
        )
    response = (
        "Which workflow should handle this?"
        if action == "abstain"
        else "; ".join(_field_response(fact_id) for fact_id in scenario.gmail.fact_ids)
    )
    events.append((TraceEventKind.FINAL_RESPONSE, {"response": response}))
    return scenario, consolidate_trace(
        [_event(index, kind, payload) for index, (kind, payload) in enumerate(events, 1)]
    )


def test_baseline_adapter_reuse_and_creation_grade_observed_names_and_deltas() -> None:
    for scenario_id in ("exact-instagram-security", "novel-calendar-creation"):
        scenario = _scenario(scenario_id)
        card = grade_scenario(scenario, adapt_baseline(_baseline_observation(scenario_id)))
        assert card.routing.status is GradeStatus.PASS
        assert card.identity.status is GradeStatus.PASS
        assert card.duplicate.status is GradeStatus.PASS
        assert card.gmail_safety.status is GradeStatus.NOT_APPLICABLE
        assert card.response.status is GradeStatus.MISSING


def test_enhanced_trace_reuse_creation_and_abstention_grade_real_shapes() -> None:
    reuse, reuse_result = _enhanced_result("exact-instagram-security")
    reuse_card = grade_scenario(reuse, reuse_result)
    assert all(layer.status is GradeStatus.PASS for layer in reuse_card.layers.values())

    create, create_result = _enhanced_result("novel-calendar-creation")
    create_card = grade_scenario(create, create_result)
    assert create_card.routing.status is GradeStatus.PASS
    assert create_card.identity.status is GradeStatus.PASS
    assert create_card.duplicate.status is GradeStatus.PASS

    abstain, abstain_result = _enhanced_result("ambiguous-creator-clarification")
    abstain_card = grade_scenario(abstain, abstain_result)
    assert abstain_card.routing.status is GradeStatus.PASS
    assert abstain_card.response.status is GradeStatus.PASS
    assert abstain_card.identity.status is GradeStatus.NOT_APPLICABLE
    assert abstain_card.duplicate.status is GradeStatus.NOT_APPLICABLE
    assert abstain_card.gmail_safety.status is GradeStatus.NOT_APPLICABLE
    assert abstain_card.context.status is GradeStatus.NOT_APPLICABLE


def test_earlier_rejected_dispatch_remains_a_routing_contradiction() -> None:
    scenario, result = _enhanced_result(
        "exact-instagram-security", rejected_first=True
    )

    card = grade_scenario(scenario, result)

    assert card.routing.status is GradeStatus.CONTRADICTORY


def _gmail_mutation_event(stage: str) -> dict[str, object]:
    return {
        "operation_name": "GMAIL_SEND_EMAIL",
        "stage": stage,
        "allowed": stage == "completed",
        "policy_code": (
            "mutating_tool_blocked" if stage == "rejected" else "unexpected_mutation"
        ),
        "sdk_executed": stage == "completed",
    }


def _real_duplicate_sequence(
    tmp_path, *, second_mode: str, first_gmail_mutation_stage: str | None = None
):
    scenario = _scenario("duplicate-clipweaver-prevention")
    run_id = uuid4()
    directory = AgentDirectory(tmp_path / "roster.json")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    distractor = None
    if second_mode in {"wrong_id", "unauthorized"}:
        distractor = directory.create(
            name="Unrelated Workflow",
            purpose="Unrelated fabricated work",
        )

    first_sink = _CollectingSink()
    first_context = TraceContext(
        run_id=run_id,
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture-revision",
        mode="fixture",
    )
    with trace_scope(first_context, first_sink):
        emit_trace(
            TraceEventKind.ROUTING_DECISION,
            {"action": "create_new", "agent_id": None, "recommendation": "create_new"},
        )
        emit_trace(
            TraceEventKind.AUTHORIZATION,
            {"routing_action": "create_new", "authorized_ids": []},
        )
        created = _dispatch_and_drain(
            instructions="Create the declared auditor",
            agent_name=scenario.expected_identity_name,
            agent_purpose=scenario.expected_identity_purpose,
            creation_intent_id="clipweaver-auditor",
            dispatch_context=DispatchContext(routing_action=RoutingAction.CREATE_NEW),
            directory=directory,
            log_store=logs,
            batch_manager=_FakeBatchManager(),
        )
        emit_trace(
            TraceEventKind.CONTEXT_METRICS,
            {
                "raw_entry_count": 0,
                "rendered_characters": 0,
                "omitted_entry_count": 0,
                "included_episode_count": 0,
            },
        )
        emit_trace(TraceEventKind.FINAL_RESPONSE, {"response": "Workflow created."})
        if first_gmail_mutation_stage is not None:
            emit_trace(
                TraceEventKind.GMAIL_EVIDENCE,
                _gmail_mutation_event(first_gmail_mutation_stage),
            )
    created_id = created.payload["agent_id"]

    second_sink = _CollectingSink()
    second_context = TraceContext(
        run_id=run_id,
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture-revision",
        mode="fixture",
    )
    with trace_scope(second_context, second_sink):
        if second_mode == "second_create":
            routed_action = RoutingAction.CREATE_NEW
            routed_id = None
            authorized = frozenset()
        elif second_mode == "wrong_id":
            assert distractor is not None
            routed_action = RoutingAction.REUSE
            routed_id = str(distractor.agent_id)
            authorized = frozenset({distractor.agent_id})
        else:
            routed_action = RoutingAction.REUSE
            routed_id = created_id
            authorized = frozenset({UUID(created_id)})
        emit_trace(
            TraceEventKind.ROUTING_DECISION,
            {
                "action": routed_action.value,
                "agent_id": routed_id,
                "recommendation": routed_id or routed_action.value,
            },
        )
        emit_trace(
            TraceEventKind.AUTHORIZATION,
            {
                "routing_action": routed_action.value,
                "authorized_ids": sorted(str(value) for value in authorized),
            },
        )
        if second_mode == "second_create":
            _dispatch_and_drain(
                instructions="Create another auditor",
                agent_name=scenario.expected_identity_name,
                agent_purpose=scenario.expected_identity_purpose,
                creation_intent_id="clipweaver-auditor-second",
                dispatch_context=DispatchContext(routing_action=routed_action),
                directory=directory,
                log_store=logs,
                batch_manager=_FakeBatchManager(),
            )
        else:
            assert distractor is not None or second_mode == "correct"
            dispatched_id = (
                str(distractor.agent_id) if second_mode == "unauthorized" else routed_id
            )
            _dispatch_and_drain(
                instructions="Reuse the auditor",
                agent_id=dispatched_id,
                dispatch_context=DispatchContext(
                    routing_action=routed_action,
                    allowed_agent_ids=authorized,
                ),
                directory=directory,
                log_store=logs,
                batch_manager=_FakeBatchManager(),
            )
        emit_trace(
            TraceEventKind.GMAIL_EVIDENCE,
            {
                "operation_name": scenario.gmail.operation,
                "stage": "allowed",
                "allowed": True,
                "policy_code": "allowed_read_only",
                "sdk_executed": True,
            },
        )
        emit_trace(
            TraceEventKind.GMAIL_EVIDENCE,
            {
                "operation_name": scenario.gmail.operation,
                "stage": "completed",
                "query_sha256": hashlib.sha256(scenario.gmail.query.encode()).hexdigest(),
                "result_count": 1,
                "has_more": False,
                "fact_ids": list(scenario.gmail.fact_ids),
            },
        )
        emit_trace(
            TraceEventKind.CONTEXT_METRICS,
            {
                "raw_entry_count": 4,
                "rendered_characters": 400,
                "omitted_entry_count": 0,
                "included_episode_count": 1,
            },
        )
        emit_trace(
            TraceEventKind.FINAL_RESPONSE,
            {"response": _field_response("CW-8117")},
        )
    return scenario, (
        consolidate_trace(first_sink.events),
        consolidate_trace(second_sink.events),
    )


def test_real_create_then_separate_reuse_turn_passes_sequence_identity_continuity(
    tmp_path,
) -> None:
    scenario, results = _real_duplicate_sequence(tmp_path, second_mode="correct")

    card = grading_module.grade_scenario_sequence(scenario, results)

    assert results[0].turn_id != results[1].turn_id
    assert [turn.routing.status for turn in card.turns] == [
        GradeStatus.PASS,
        GradeStatus.PASS,
    ]
    assert card.identity_continuity.status is GradeStatus.PASS
    assert card.passed is True


def _real_pronoun_sequence(
    tmp_path, *, second_gmail_mutation_stage: str | None = None
):
    scenario = _scenario("pronoun-receipt-follow-up")
    assert scenario.expected_agent_id is not None
    stable_id = UUID(scenario.expected_agent_id)
    run_id = uuid4()
    directory = AgentDirectory(
        tmp_path / "roster.json", id_factory=lambda: stable_id
    )
    record = directory.create(
        name=scenario.expected_identity_name,
        purpose=scenario.expected_identity_purpose,
    )
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    results = []

    for index, final_response in enumerate(
        (
            _field_response("VF-20481"),
            _field_response("VF-20481", "CAD 47.80"),
        )
    ):
        sink = _CollectingSink()
        context = TraceContext(
            run_id=run_id,
            turn_id=uuid4(),
            system="enhanced",
            revision="fixture-revision",
            mode="fixture",
        )
        with trace_scope(context, sink):
            emit_trace(
                TraceEventKind.ROUTING_DECISION,
                {
                    "action": "reuse",
                    "agent_id": str(record.agent_id),
                    "recommendation": str(record.agent_id),
                },
            )
            emit_trace(
                TraceEventKind.AUTHORIZATION,
                {
                    "routing_action": "reuse",
                    "authorized_ids": [str(record.agent_id)],
                },
            )
            _dispatch_and_drain(
                instructions=scenario.turns[index].text,
                agent_id=str(record.agent_id),
                dispatch_context=DispatchContext(
                    routing_action=RoutingAction.REUSE,
                    allowed_agent_ids=frozenset({record.agent_id}),
                ),
                directory=directory,
                log_store=logs,
                batch_manager=_FakeBatchManager(),
            )
            if index == 0:
                gmail = scenario.turn_expectations[index].gmail
                assert gmail is not None
                emit_trace(
                    TraceEventKind.GMAIL_EVIDENCE,
                    {
                        "operation_name": gmail.operation,
                        "stage": "allowed",
                        "allowed": True,
                        "policy_code": "allowed_read_only",
                        "sdk_executed": True,
                    },
                )
                emit_trace(
                    TraceEventKind.GMAIL_EVIDENCE,
                    {
                        "operation_name": gmail.operation,
                        "stage": "completed",
                        "query_sha256": hashlib.sha256(gmail.query.encode()).hexdigest(),
                        "result_count": 1,
                        "has_more": False,
                        "fact_ids": list(gmail.fact_ids),
                    },
                )
            elif second_gmail_mutation_stage is not None:
                emit_trace(
                    TraceEventKind.GMAIL_EVIDENCE,
                    _gmail_mutation_event(second_gmail_mutation_stage),
                )
            emit_trace(
                TraceEventKind.CONTEXT_METRICS,
                {
                    "raw_entry_count": 4,
                    "rendered_characters": 400,
                    "omitted_entry_count": 0,
                    "included_episode_count": 1,
                },
            )
            emit_trace(TraceEventKind.FINAL_RESPONSE, {"response": final_response})
        results.append(consolidate_trace(sink.events))

    return scenario, tuple(results)


def test_real_pronoun_follow_up_reuses_prior_turn_evidence_without_merging_traces(
    tmp_path,
) -> None:
    scenario, results = _real_pronoun_sequence(tmp_path)

    card = grading_module.grade_scenario_sequence(scenario, results)

    assert results[0].turn_id != results[1].turn_id
    assert card.identity_continuity.status is GradeStatus.PASS
    assert [turn.response.status for turn in card.turns] == [
        GradeStatus.PASS,
        GradeStatus.PASS,
    ]
    assert card.passed is True


@pytest.mark.parametrize("stage", ["rejected", "completed"])
def test_no_gmail_create_turn_rejects_observed_mutating_gmail_activity(
    tmp_path, stage: str
) -> None:
    scenario, results = _real_duplicate_sequence(
        tmp_path,
        second_mode="correct",
        first_gmail_mutation_stage=stage,
    )

    card = grading_module.grade_scenario_sequence(scenario, results)

    assert card.turns[0].gmail_safety.status is GradeStatus.CONTRADICTORY
    assert card.passed is False


@pytest.mark.parametrize("stage", ["rejected", "completed"])
def test_prior_evidence_follow_up_rejects_observed_mutating_gmail_activity(
    tmp_path, stage: str
) -> None:
    scenario, results = _real_pronoun_sequence(
        tmp_path, second_gmail_mutation_stage=stage
    )

    card = grading_module.grade_scenario_sequence(scenario, results)

    assert card.turns[1].gmail_safety.status is GradeStatus.CONTRADICTORY
    assert card.passed is False


@pytest.mark.parametrize("second_mode", ["second_create", "wrong_id", "unauthorized"])
def test_real_duplicate_wrong_id_or_unauthorized_second_turn_fails_sequence(
    tmp_path, second_mode: str
) -> None:
    scenario, results = _real_duplicate_sequence(tmp_path, second_mode=second_mode)

    card = grading_module.grade_scenario_sequence(scenario, results)

    assert card.passed is False
    assert (
        card.identity_continuity.status is GradeStatus.FAIL
        or card.turns[1].routing.status is not GradeStatus.PASS
    )
