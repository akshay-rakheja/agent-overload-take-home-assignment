"""Deterministic, layer-independent Evaluation Lab grading."""

from __future__ import annotations

import hashlib
from uuid import uuid4

from evals.live_lab.fixture_email import fixture_response_facts
from evals.live_lab.grading import GradeStatus, grade_scenario
from evals.live_lab.scenarios import load_controlled_scenarios
from server.services.evaluation_lab.models import Availability, ObservedValue, SystemRunResult


def _available(value):
    return {"availability": Availability.AVAILABLE, "value": value}


def _query_sha256(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _replace(result: SystemRunResult, **updates) -> SystemRunResult:
    payload = result.model_dump(mode="python")
    payload.update(updates)
    return SystemRunResult.model_validate(payload)


def _result(scenario_id: str, *, system: str = "enhanced"):
    scenario = next(item for item in load_controlled_scenarios() if item.scenario_id == scenario_id)
    expected_id = scenario.expected_agent_id
    response = " ".join(
        fact
        for fact_id in scenario.gmail.fact_ids
        for fact in fixture_response_facts(fact_id)
    )
    result = SystemRunResult(
        run_id=uuid4(),
        turn_id=uuid4(),
        system=system,
        decision=_available({"action": scenario.expected.action.value, "agent_id": expected_id}),
        candidates=_available(
            ([{"agent_id": expected_id, "logical_identity": scenario.expected.logical_identity}]
             if expected_id else [])
        ),
        attempted_dispatch=_available({"attempted": True, "reference_type": "reuse"}),
        accepted_dispatch=_available(
            {"status": "accepted", "success": True, "selected_agent_id": expected_id}
        ),
        selected_identity=_available({"agent_id": expected_id}) if expected_id else _available({"logical_identity": scenario.expected.logical_identity}),
        identity_delta=_available(
            {"directory_count_before": 100, "directory_count_after": 100}
        ),
        duplicates=_available([]),
        gmail_evidence=_available(
            [
                {
                    "operation_name": scenario.gmail.operation,
                    "stage": "completed",
                    "allowed": True,
                    "policy_code": "allowed_read_only",
                    "query_sha256": _query_sha256(scenario.gmail.query),
                    "result_count": 1,
                    "fact_ids": list(scenario.gmail.fact_ids),
                    "sdk_executed": True,
                }
            ]
        ),
        final_response=_available(response or "No matching fixture email was found."),
        context_metrics=_available(
            {
                "raw_entry_count": 4,
                "rendered_characters": 400,
                "omitted_entry_count": 0,
                "included_episode_count": 1,
            }
        ),
    )
    return scenario, result


def test_all_six_layers_pass_with_positive_captured_evidence() -> None:
    scenario, result = _result("exact-instagram-security")

    scorecard = grade_scenario(scenario, result)

    assert scorecard.routing.status is GradeStatus.PASS
    assert scorecard.response.status is GradeStatus.PASS
    assert scorecard.gmail_safety.status is GradeStatus.PASS
    assert scorecard.identity.status is GradeStatus.PASS
    assert scorecard.duplicate.status is GradeStatus.PASS
    assert scorecard.context.status is GradeStatus.PASS
    assert scorecard.candidate_rank.availability is Availability.AVAILABLE
    assert scorecard.candidate_rank.value == 1
    assert scorecard.passed is True


def test_routing_and_response_are_independent_and_invented_fact_fails_response() -> None:
    scenario, result = _result("exact-instagram-security")
    result = _replace(result, final_response=_available("The invoice was CAD 312.40."))

    scorecard = grade_scenario(scenario, result)

    assert scorecard.routing.status is GradeStatus.PASS
    assert scorecard.response.status is GradeStatus.FAIL
    assert any("invented" in item for item in scorecard.response.negative_evidence)

    wrong_route = _replace(
        result,
        decision=_available({"action": "create_new", "agent_id": None}),
        final_response=_available(" ".join(fixture_response_facts("SEC-7419"))),
    )
    inverse = grade_scenario(scenario, wrong_route)
    assert inverse.routing.status is GradeStatus.FAIL
    assert inverse.response.status is GradeStatus.PASS


def test_contradictory_gmail_evidence_fails_even_with_a_matching_event() -> None:
    scenario, result = _result("exact-instagram-security")
    evidence = list(result.gmail_evidence.value)
    evidence.append(
        {
            "operation_name": "GMAIL_SEND_EMAIL",
            "stage": "rejected",
            "allowed": False,
            "policy_code": "mutating_tool_blocked",
            "query_sha256": _query_sha256(scenario.gmail.query),
            "result_count": 0,
            "fact_ids": [],
        }
    )

    scorecard = grade_scenario(
        scenario,
        _replace(result, gmail_evidence=_available(evidence)),
    )

    assert scorecard.gmail_safety.status is GradeStatus.CONTRADICTORY
    assert scorecard.gmail_safety.contradictory_evidence
    assert scorecard.response.status is GradeStatus.PASS


def test_missing_evidence_is_not_treated_as_failure_or_success() -> None:
    scenario, result = _result("exact-instagram-security")
    missing = {"availability": Availability.UNAVAILABLE, "reason": "not emitted"}

    scorecard = grade_scenario(
        scenario,
        _replace(
            result,
            decision=missing,
            final_response=missing,
            gmail_evidence=missing,
            selected_identity=missing,
            identity_delta=missing,
            duplicates=missing,
            context_metrics=missing,
        ),
    )

    assert {layer.status for layer in scorecard.layers.values()} == {GradeStatus.MISSING}
    assert all(layer.missing_evidence for layer in scorecard.layers.values())


def test_candidate_rank_is_baseline_na_and_nonreuse_enhanced_na() -> None:
    reuse, baseline = _result("exact-instagram-security", system="baseline")
    baseline_card = grade_scenario(reuse, baseline)
    assert baseline_card.candidate_rank.availability is Availability.NOT_APPLICABLE

    create, enhanced = _result("novel-calendar-creation")
    create_card = grade_scenario(create, enhanced)
    assert create_card.candidate_rank.availability is Availability.NOT_APPLICABLE


def test_ambiguity_requires_no_dispatch_and_explicit_clarification() -> None:
    scenario, result = _result("ambiguous-creator-clarification")
    result = _replace(
        result,
        attempted_dispatch=_available({"attempted": False}),
        accepted_dispatch=_available({"status": "not_attempted"}),
        selected_identity={
            "availability": Availability.NOT_APPLICABLE,
            "reason": "abstained before dispatch",
        },
        identity_delta=_available(
            {"directory_count_before": 100, "directory_count_after": 100}
        ),
        final_response=_available(
            "Which Instagram workflow should handle this: security or engagement?"
        ),
    )

    scorecard = grade_scenario(scenario, result)
    assert scorecard.routing.status is GradeStatus.PASS
    assert scorecard.response.status is GradeStatus.PASS
    assert scorecard.identity.status is GradeStatus.PASS

    dispatched = _replace(result, attempted_dispatch=_available({"attempted": True}))
    assert grade_scenario(scenario, dispatched).routing.status is GradeStatus.FAIL


def test_no_result_requires_query_empty_evidence_explicit_absence_and_no_invention() -> None:
    scenario, result = _result("honest-no-result")
    result = _replace(
        result,
        gmail_evidence=_available(
            [
                {
                    "operation_name": scenario.gmail.operation,
                    "stage": "completed",
                    "allowed": True,
                    "policy_code": "allowed_read_only",
                    "query_sha256": _query_sha256(scenario.gmail.query),
                    "result_count": 0,
                    "fact_ids": [],
                    "sdk_executed": True,
                }
            ]
        ),
        final_response=_available("No matching fixture email was found."),
    )

    assert grade_scenario(scenario, result).response.status is GradeStatus.PASS

    invented = _replace(result, final_response=_available("No result, but it was CAD 47.80."))
    assert grade_scenario(scenario, invented).response.status is GradeStatus.FAIL


def test_duplicate_prevention_uses_stable_identity_and_directory_delta() -> None:
    scenario, result = _result("duplicate-clipweaver-prevention")
    stable_id = "00000000-0000-0000-0000-000000000123"
    result = _replace(
        result,
        selected_identity=_available(
            {"agent_id": stable_id, "logical_identity": "new:clipweaver-auditor"}
        ),
        created_identity=_available(
            {"agent_id": stable_id, "logical_identity": "new:clipweaver-auditor"}
        ),
        identity_delta=_available(
            {
                "directory_count_before": 100,
                "directory_count_after": 101,
                "stable_ids": [stable_id, stable_id],
            }
        ),
        duplicates=_available([]),
    )

    assert grade_scenario(scenario, result).duplicate.status is GradeStatus.PASS

    duplicate = _replace(
        result,
        identity_delta=_available(
            {"directory_count_before": 100, "directory_count_after": 102}
        ),
        duplicates=_available([{"logical_identity": "new:clipweaver-auditor"}]),
    )
    assert grade_scenario(scenario, duplicate).duplicate.status is GradeStatus.FAIL


def test_long_history_context_requires_bounded_rendering_with_omissions() -> None:
    scenario, result = _result("ten-thousand-history-anchor")
    result = _replace(
        result,
        context_metrics=_available(
            {
                "raw_entry_count": 10_000,
                "rendered_characters": 4_000,
                "omitted_entry_count": 9_968,
                "included_episode_count": 8,
            }
        ),
    )
    assert grade_scenario(scenario, result).context.status is GradeStatus.PASS

    unbounded = _replace(
        result,
        context_metrics=_available(
            {
                "raw_entry_count": 10_000,
                "rendered_characters": 200_000,
                "omitted_entry_count": 0,
                "included_episode_count": 2_500,
            }
        ),
    )
    assert grade_scenario(scenario, unbounded).context.status is GradeStatus.FAIL
