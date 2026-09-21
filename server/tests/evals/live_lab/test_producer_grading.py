"""Producer-to-grader checks using only evidence emitted by real adapters/traces."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID

from evals.live_lab.fixture_email import fixture_response_facts
from evals.live_lab.grading import GradeStatus, grade_scenario
from evals.live_lab.raw_observation import BaselineObservation
from evals.live_lab.scenarios import load_controlled_scenarios
from server.services.evaluation_lab.baseline_adapter import adapt_baseline
from server.services.evaluation_lab.models import TraceEvent, TraceEventKind
from server.services.evaluation_lab.trace import consolidate_trace


RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
TURN_ID = UUID("22222222-2222-4222-8222-222222222222")


def _scenario(scenario_id: str):
    return next(
        item for item in load_controlled_scenarios() if item.scenario_id == scenario_id
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


def _enhanced_result(
    scenario_id: str, *, repeated_create: bool = False, rejected_first: bool = False
):
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
        if repeated_create:
            events.extend(
                [
                    (
                        TraceEventKind.DISPATCH_ATTEMPT,
                        {
                            "reference_type": "create",
                            "requested_agent_id": None,
                            "routing_action": "create_new",
                            "authorized_ids": [],
                            "directory_count_before": 101,
                        },
                    ),
                    (
                        TraceEventKind.DISPATCH_RESULT,
                        {
                            "status": "accepted",
                            "success": True,
                            "selected_agent_id": stable_id,
                            "new_agent_created": False,
                            "idempotent_creation": True,
                            "directory_count_before": 101,
                            "directory_count_after": 101,
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
                            "delta": {
                                "directory_count_before": 101,
                                "directory_count_after": 101,
                            },
                            "idempotent_creation": True,
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
        else " ".join(
            value
            for fact_id in scenario.gmail.fact_ids
            for value in fixture_response_facts(fact_id)
        )
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


def test_enhanced_repeated_creation_derives_no_duplicate_from_real_identity_events() -> None:
    scenario, result = _enhanced_result(
        "duplicate-clipweaver-prevention", repeated_create=True
    )

    card = grade_scenario(scenario, result)

    assert card.identity.status is GradeStatus.PASS
    assert card.duplicate.status is GradeStatus.PASS
    assert result.identity_delta.value["created_agent_ids"] == [
        "33333333-3333-4333-8333-333333333333"
    ]


def test_earlier_rejected_dispatch_remains_a_routing_contradiction() -> None:
    scenario, result = _enhanced_result(
        "exact-instagram-security", rejected_first=True
    )

    card = grade_scenario(scenario, result)

    assert card.routing.status is GradeStatus.CONTRADICTORY
