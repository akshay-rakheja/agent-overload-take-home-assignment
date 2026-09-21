"""Honest adaptation of raw historical observations into shared results."""

from __future__ import annotations

from uuid import UUID

from evals.live_lab.raw_observation import (
    BaselineObservation,
    ObservedError,
    RawModelCall,
)
from server.services.evaluation_lab.baseline_adapter import adapt_baseline
from server.services.evaluation_lab.models import Availability


RUN_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def _observation(
    *,
    action: str = "reuse",
    name: str | None = "Historical Alice",
    final_response: str | None = "Historical response",
) -> BaselineObservation:
    return BaselineObservation(
        run_id=RUN_ID,
        prompt_xml_sha256="a" * 64,
        prompt_characters=412,
        exposed_names=("Historical Alice", "Historical Bob"),
        roster_before=("Historical Alice", "Historical Bob"),
        roster_after=("Historical Alice", "Historical Bob"),
        journal_hashes_before={"historical-alice.log": "b" * 64},
        journal_hashes_after={"historical-alice.log": "c" * 64},
        inferred_action=action,
        inferred_name=name,
        inference_reason="one observed historical journal append",
        final_response=final_response,
        raw_model_calls=(
            RawModelCall(
                component="interaction",
                model="fixture-model",
                elapsed_ms=12.5,
                request_sha256="d" * 64,
                response_sha256="e" * 64,
                message_count=3,
                tool_names=("send_message_to_agent",),
                response_choice_count=1,
                response_tool_call_count=1,
            ),
        ),
        errors=(),
    )


def test_full_roster_baseline_maps_only_observed_or_inferred_facts() -> None:
    result = adapt_baseline(_observation())

    assert result.run_id == UUID(RUN_ID)
    assert result.system == "baseline"
    assert result.roster_count.availability is Availability.AVAILABLE
    assert result.roster_count.value == 2
    assert result.prompt_exposure.value == {
        "prompt_xml_sha256": "a" * 64,
        "prompt_characters": 412,
        "exposed_names": ["Historical Alice", "Historical Bob"],
        "exposed_name_count": 2,
    }
    assert result.decision.availability is Availability.INFERRED
    assert result.decision.value == {
        "action": "reuse",
        "reason": "one observed historical journal append",
    }
    assert result.selected_identity.availability is Availability.INFERRED
    assert result.selected_identity.value == {
        "name": "Historical Alice",
        "reason": "one observed historical journal append",
    }
    assert result.final_response.value == "Historical response"
    assert result.timings.value == [
        {
            "component": "interaction",
            "model": "fixture-model",
            "elapsed_ms": 12.5,
            "request_sha256": "d" * 64,
            "response_sha256": "e" * 64,
            "message_count": 3,
            "tool_names": ["send_message_to_agent"],
            "response_choice_count": 1,
            "response_tool_call_count": 1,
            "error_type": None,
        }
    ]


def test_baseline_does_not_invent_candidates_ranks_scores_ids_or_zeroes() -> None:
    result = adapt_baseline(_observation())

    for field in ("candidates", "recommendation", "authorized_ids", "duplicates"):
        observed = getattr(result, field)
        assert observed.availability is Availability.NOT_APPLICABLE
        assert observed.value is None
    for field in ("attempted_dispatch", "accepted_dispatch", "context_metrics"):
        observed = getattr(result, field)
        assert observed.availability is Availability.UNAVAILABLE
        assert observed.value is None
    assert result.usage.input_tokens.value is None
    assert result.usage.output_tokens.value is None
    assert result.cost.amount.value is None
    rendered = result.model_dump_json()
    assert '"rank"' not in rendered
    assert '"score"' not in rendered
    assert '"agent_id"' not in rendered


def test_create_maps_selected_and_created_names_as_inferred_without_stable_id() -> None:
    observation = _observation(action="create_new", name="Historical Carol")
    observation = observation.model_copy(
        update={
            "roster_after": (
                "Historical Alice",
                "Historical Bob",
                "Historical Carol",
            )
        }
    )

    result = adapt_baseline(observation)

    assert result.created_identity.availability is Availability.INFERRED
    assert result.created_identity.value == {
        "name": "Historical Carol",
        "reason": "one observed historical journal append",
    }
    assert result.identity_delta.availability is Availability.AVAILABLE
    assert result.identity_delta.value == {
        "roster_before_count": 2,
        "roster_after_count": 3,
        "added_names": ["Historical Carol"],
        "removed_names": [],
    }


def test_unobservable_missing_values_and_errors_remain_explicitly_unavailable() -> None:
    observation = _observation(
        action="unobservable",
        name=None,
        final_response=None,
    ).model_copy(
        update={
            "errors": (
                ObservedError(
                    phase="interaction",
                    code="timeout",
                    message="sanitized timeout",
                ),
            )
        }
    )

    first = adapt_baseline(observation)
    second = adapt_baseline(observation)

    assert first == second
    assert first.selected_identity.availability is Availability.UNAVAILABLE
    assert first.created_identity.availability is Availability.UNAVAILABLE
    assert first.final_response.availability is Availability.UNAVAILABLE
    assert first.errors.availability is Availability.AVAILABLE
    assert first.errors.value == [
        {
            "phase": "interaction",
            "code": "timeout",
            "message": "sanitized timeout",
            "late": False,
            "partial": False,
        }
    ]


def test_adapter_redacts_secret_and_mailbox_markers_before_shared_serialization() -> None:
    observation = _observation(
        name="mailbox@example.invalid",
        final_response="Bearer private-baseline-token",
    ).model_copy(
        update={
            "errors": (
                ObservedError(
                    phase="provider",
                    code="provider_failure",
                    message="api_key=private-baseline-key",
                ),
            )
        }
    )

    rendered = adapt_baseline(observation).model_dump_json()

    assert "mailbox@example.invalid" not in rendered
    assert "private-baseline-token" not in rendered
    assert "private-baseline-key" not in rendered
    assert "[REDACTED" in rendered


def test_non_uuid_historical_run_label_gets_deterministic_transport_identity() -> None:
    observation = _observation().model_copy(update={"run_id": "historical-run-label"})

    first = adapt_baseline(observation)
    second = adapt_baseline(observation)

    assert first.run_id == second.run_id
    assert first.turn_id == second.turn_id
    assert first.run_id != first.turn_id
