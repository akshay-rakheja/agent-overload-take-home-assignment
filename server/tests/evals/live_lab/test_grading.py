"""Deterministic, layer-independent Evaluation Lab grading."""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest

from evals.live_lab.fixture_email import fixture_response_facts, fixture_response_fields
from evals.live_lab.grading import GradeStatus, grade_scenario
from evals.live_lab.scenarios import GmailExpectation, load_controlled_scenarios
from server.services.evaluation_lab.models import Availability, ObservedValue, SystemRunResult


def _available(value):
    return {"availability": Availability.AVAILABLE, "value": value}


def _query_sha256(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _field_response(*fact_ids: str) -> str:
    return "; ".join(
        f"{field.key.replace('_', ' ')}: {field.value}"
        for fact_id in fact_ids
        for field in fixture_response_fields(fact_id)
    )


def _replace(result: SystemRunResult, **updates) -> SystemRunResult:
    payload = result.model_dump(mode="python")
    payload.update(updates)
    return SystemRunResult.model_validate(payload)


def _result(scenario_id: str, *, system: str = "enhanced"):
    scenario = next(item for item in load_controlled_scenarios() if item.scenario_id == scenario_id)
    expected_id = scenario.expected_agent_id
    action = scenario.expected.action.value
    selected_id = expected_id or "00000000-0000-4000-8000-000000000123"
    response = _field_response(*scenario.gmail.fact_ids)
    result = SystemRunResult(
        run_id=uuid4(),
        turn_id=uuid4(),
        system=system,
        decision=_available({"action": scenario.expected.action.value, "agent_id": expected_id}),
        candidates=_available(
            ([{"agent_id": expected_id, "name": scenario.expected_identity_name}]
             if expected_id else [])
        ),
        attempted_dispatch=_available(
            {
                "reference_type": "reuse" if action == "reuse" else "create",
                "requested_agent_id": expected_id,
                "routing_action": action,
                "authorized_ids": [expected_id] if expected_id else [],
            }
        ),
        accepted_dispatch=_available(
            {"status": "accepted", "success": True, "selected_agent_id": selected_id}
        ),
        authorized_ids=_available([expected_id] if expected_id else []),
        selected_identity=(
            _available({"agent_id": selected_id, "name": scenario.expected_identity_name})
        ),
        created_identity=(
            _available({"agent_id": selected_id, "name": scenario.expected_identity_name})
            if action == "create_new"
            else {"availability": Availability.NOT_APPLICABLE, "reason": "not created"}
        ),
        identity_delta=_available(
            {
                "directory_count_before": 100,
                "directory_count_after": 101 if action == "create_new" else 100,
                "created_agent_ids": [selected_id] if action == "create_new" else [],
                "selected_agent_ids": [selected_id] if action != "abstain" else [],
            }
        ),
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


def _result_for_fixture_response(fact_id: str, response: str):
    scenario, result = _result("exact-instagram-security")
    gmail = GmailExpectation(
        operation="GMAIL_FETCH_EMAILS",
        query=f'subject:"[OpenPoke Interview Fixture]" "{fact_id}"',
        fact_ids=(fact_id,),
    )
    scenario = scenario.model_copy(
        update={
            "gmail": gmail,
            "response_assertions": fixture_response_facts(fact_id),
        }
    )
    result = _replace(
        result,
        gmail_evidence=_available(
            [
                {
                    "operation_name": gmail.operation,
                    "stage": "completed",
                    "allowed": True,
                    "policy_code": "allowed_read_only",
                    "query_sha256": _query_sha256(gmail.query),
                    "result_count": 1,
                    "fact_ids": [fact_id],
                    "sdk_executed": True,
                }
            ]
        ),
        final_response=_available(response),
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
        final_response=_available(_field_response("SEC-7419")),
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


def test_no_gmail_turn_fails_when_unexpected_read_only_work_is_observed() -> None:
    scenario, result = _result("novel-calendar-creation")
    scenario = scenario.model_copy(update={"gmail_required": False})

    card = grade_scenario(scenario, result)

    assert card.gmail_safety.status is GradeStatus.FAIL
    assert card.gmail_safety.status is not GradeStatus.NOT_APPLICABLE


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
        authorized_ids=_available([]),
        attempted_dispatch={"availability": Availability.UNAVAILABLE, "reason": "not emitted"},
        accepted_dispatch={"availability": Availability.UNAVAILABLE, "reason": "not emitted"},
        selected_identity={
            "availability": Availability.NOT_APPLICABLE,
            "reason": "abstained before dispatch",
        },
        identity_delta={"availability": Availability.UNAVAILABLE, "reason": "not emitted"},
        gmail_evidence={"availability": Availability.UNAVAILABLE, "reason": "not emitted"},
        context_metrics={"availability": Availability.UNAVAILABLE, "reason": "not emitted"},
        final_response=_available(
            "Which Instagram workflow should handle this: security or engagement?"
        ),
    )

    scorecard = grade_scenario(scenario, result)
    assert scorecard.routing.status is GradeStatus.PASS
    assert scorecard.response.status is GradeStatus.PASS
    assert scorecard.identity.status is GradeStatus.NOT_APPLICABLE
    assert scorecard.gmail_safety.status is GradeStatus.NOT_APPLICABLE

    dispatched = _replace(result, attempted_dispatch=_available({"attempted": True}))
    assert grade_scenario(scenario, dispatched).routing.status is GradeStatus.CONTRADICTORY


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


def test_no_result_rejects_any_nonempty_failed_or_paginated_matching_event() -> None:
    scenario, result = _result("honest-no-result")
    empty = {
        "operation_name": scenario.gmail.operation,
        "stage": "completed",
        "query_sha256": _query_sha256(scenario.gmail.query),
        "result_count": 0,
        "fact_ids": [],
        "has_more": False,
        "sdk_executed": True,
    }
    contradictions = [
        {**empty, "result_count": 1, "fact_ids": ["SEC-7419"]},
        {**empty, "stage": "failed"},
        {**empty, "has_more": True},
    ]
    for contradiction in contradictions:
        card = grade_scenario(
            scenario,
            _replace(
                result,
                gmail_evidence=_available([empty, contradiction]),
                final_response=_available("No matching fixture email was found."),
            ),
        )
        assert card.response.status in {GradeStatus.FAIL, GradeStatus.CONTRADICTORY}


def test_response_rejects_contradictory_expected_fields_and_unknown_facts() -> None:
    scenario, result = _result("exact-instagram-security")
    canonical = _field_response("SEC-7419")
    contradictory = _replace(
        result,
        final_response=_available(
            f"{canonical}. Correction: device was Pixel 9 in Paris, not Lisbon."
        ),
    )
    assert grade_scenario(scenario, contradictory).response.status is GradeStatus.FAIL

    no_result, empty = _result("honest-no-result")
    empty = _replace(
        empty,
        gmail_evidence=_available(
            [
                {
                    "operation_name": no_result.gmail.operation,
                    "stage": "completed",
                    "query_sha256": _query_sha256(no_result.gmail.query),
                    "result_count": 0,
                    "fact_ids": [],
                    "has_more": False,
                    "sdk_executed": True,
                }
            ]
        ),
        final_response=_available(
            "No matching fixture email was found. CAD 999.99, code XYZ-2222."
        ),
    )
    assert grade_scenario(no_result, empty).response.status is GradeStatus.FAIL


@pytest.mark.parametrize(
    "unsupported_assertion",
    [
        "Correction: location was Vancouver.",
        "Correction: device was iPhone 17.",
        "Correction: verification phrase was orange-cloud.",
        "The location was not actually Lisbon.",
    ],
)
def test_response_contract_never_passes_unrecognized_factual_assertions(
    unsupported_assertion: str,
) -> None:
    scenario, result = _result("exact-instagram-security")
    canonical = _field_response("SEC-7419")

    card = grade_scenario(
        scenario,
        _replace(result, final_response=_available(f"{canonical}. {unsupported_assertion}")),
    )

    assert card.response.status is not GradeStatus.PASS


@pytest.mark.parametrize(
    "contradictory_assertion",
    [
        "The location was Pixel 10.",
        "The device was Lisbon.",
        "The verification phrase was Lisbon.",
        "The location was indigo-orbit.",
    ],
)
def test_response_contract_binds_each_value_to_its_declared_field(
    contradictory_assertion: str,
) -> None:
    canonical = (
        "Reference: SEC-7419; dated: 2026-09-18 04:12 UTC; location: Lisbon; "
        "device: Pixel 10; verification phrase: indigo-orbit."
    )
    scenario, result = _result_for_fixture_response(
        "SEC-7419", f"{canonical} {contradictory_assertion}"
    )

    assert grade_scenario(scenario, result).response.status is not GradeStatus.PASS


def test_response_contract_rejects_repeated_field_with_a_conflicting_allowed_value() -> None:
    scenario, result = _result_for_fixture_response(
        "SEC-7419",
        (
            "Reference: SEC-7419; dated: 2026-09-18 04:12 UTC; location: Lisbon; "
            "device: Pixel 10; verification phrase: indigo-orbit. "
            "The location was Pixel 10."
        ),
    )

    assert grade_scenario(scenario, result).response.status is not GradeStatus.PASS


@pytest.mark.parametrize(
    ("fact_id", "response"),
    [
        (
            "SEC-7419",
            "Reference: SEC-7419; sign-in time = 2026-09-18 04:12 UTC; "
            "the location was Lisbon; device is Pixel 10; "
            "verification phrase: indigo-orbit.",
        ),
        (
            "ENG-2284",
            "Reference is ENG-2284. Creator: Aurora Loop; likes = 183 likes; "
            "comments were 27 comments.",
        ),
        (
            "NF-3207",
            "Reference: NF-3207; release was Prism Cut 2.4; "
            "release date = 2026-10-07; feature is Storyboard Lock.",
        ),
        (
            "VF-20481",
            "Reference = VF-20481; product: Pro Render Monthly; "
            "total was CAD 47.80; purchase date is 2026-09-19.",
        ),
        (
            "MS-8820",
            "Reference: MS-8820; session is Temporal Layers; "
            "session time: 2026-10-11 17:30 UTC; reference code = GLASS-52.",
        ),
        (
            "CW-8117",
            "Reference was CW-8117; amount: CAD 312.40; "
            "due date is 2026-10-15; purchase order = PO-4406.",
        ),
        (
            "ARC-1042",
            "Reference: ARC-1042; archive is Cedar Comet; date = 2026-08-29; "
            "checksum prefix was 9f2c7a.",
        ),
        (
            "AMB-6063",
            "Reference: AMB-6063; security category = account-security; "
            "engagement category was engagement-performance.",
        ),
    ],
)
def test_response_contract_accepts_finite_field_bound_forms_for_every_fixture(
    fact_id: str, response: str
) -> None:
    scenario, result = _result_for_fixture_response(fact_id, response)

    assert grade_scenario(scenario, result).response.status is GradeStatus.PASS


@pytest.mark.parametrize(
    "response",
    [
        "No matching email. The login was from Vancouver on iPhone 17.",
        "No matching email. The invoice was USD 900.99.",
        "No matching email. The receipt code is xyz-2222.",
    ],
)
def test_no_result_uses_a_closed_absence_response_form(response: str) -> None:
    scenario, result = _result("honest-no-result")
    result = _replace(
        result,
        gmail_evidence=_available(
            [
                {
                    "operation_name": scenario.gmail.operation,
                    "stage": "completed",
                    "query_sha256": _query_sha256(scenario.gmail.query),
                    "result_count": 0,
                    "fact_ids": [],
                    "has_more": False,
                    "sdk_executed": True,
                }
            ]
        ),
        final_response=_available(response),
    )

    assert grade_scenario(scenario, result).response.status is not GradeStatus.PASS


def test_routing_rejects_rejected_wrong_or_unauthorized_dispatch() -> None:
    scenario, result = _result("exact-instagram-security")
    wrong_id = "00000000-0000-4000-8000-000000000999"
    cases = [
        _replace(result, authorized_ids=_available([])),
        _replace(
            result,
            attempted_dispatch=_available(
                {
                    "reference_type": "reuse",
                    "requested_agent_id": wrong_id,
                    "routing_action": "reuse",
                    "authorized_ids": [scenario.expected_agent_id],
                }
            ),
        ),
        _replace(
            result,
            accepted_dispatch=_available(
                {
                    "status": "rejected",
                    "success": False,
                    "code": "routing_not_authorized",
                }
            ),
        ),
    ]

    for case in cases:
        assert grade_scenario(scenario, case).routing.status in {
            GradeStatus.FAIL,
            GradeStatus.CONTRADICTORY,
        }


def test_abstain_rejects_any_accepted_dispatch_even_without_attempt_event() -> None:
    scenario, result = _result("ambiguous-creator-clarification")
    result = _replace(
        result,
        authorized_ids=_available([]),
        attempted_dispatch={"availability": Availability.UNAVAILABLE, "reason": "not emitted"},
        accepted_dispatch=_available(
            {
                "status": "accepted",
                "success": True,
                "selected_agent_id": "00000000-0000-4000-8000-000000000999",
            }
        ),
        final_response=_available("Which workflow should handle this?"),
    )
    assert grade_scenario(scenario, result).routing.status is GradeStatus.CONTRADICTORY


def test_stable_id_conflict_overrides_matching_logical_or_name_fallback() -> None:
    scenario, result = _result("exact-instagram-security")
    wrong_id = "00000000-0000-4000-8000-000000000999"
    result = _replace(
        result,
        selected_identity=_available(
            {
                "agent_id": wrong_id,
                "logical_identity": scenario.expected.logical_identity,
                "name": scenario.expected_identity_name,
            }
        ),
        candidates=_available(
            [
                {
                    "agent_id": wrong_id,
                    "logical_identity": scenario.expected.logical_identity,
                    "name": scenario.expected_identity_name,
                }
            ]
        ),
    )
    card = grade_scenario(scenario, result)
    assert card.identity.status is GradeStatus.FAIL
    assert card.candidate_rank.availability is Availability.UNAVAILABLE


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
