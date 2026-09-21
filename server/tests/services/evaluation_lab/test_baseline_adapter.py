"""Honest adaptation of raw historical observations into shared results."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

from evals.live_lab import baseline_observer as observer_module
from evals.live_lab.baseline_observer import _failed_observation, run_baseline_turn
from evals.live_lab.raw_observation import (
    BaselineObservation,
    BaselineTurnRequest,
    ObservedError,
    RawModelCall,
    RawPhaseTiming,
)
from server.services.evaluation_lab.baseline_adapter import adapt_baseline
from server.services.evaluation_lab.models import Availability


RUN_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


class _ObserverResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _MissingAfterAndPromptClient:
    def __init__(self, roster_path: Path, **kwargs) -> None:
        del kwargs
        self.roster_path = roster_path
        self.get_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        del args

    async def get(self, url: str) -> _ObserverResponse:
        del url
        self.get_count += 1
        messages = (
            []
            if self.get_count == 1
            else [{"role": "assistant", "content": "done"}]
        )
        return _ObserverResponse({"messages": messages})

    async def post(self, url: str, json: dict[str, object]) -> _ObserverResponse:
        del url, json
        self.roster_path.write_text("{partial", encoding="utf-8")
        return _ObserverResponse({})


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
    assert result.cost.currency.availability is Availability.UNAVAILABLE
    assert result.cost.currency.value is None
    rendered = result.model_dump_json()
    assert '"rank"' not in rendered
    assert '"score"' not in rendered
    assert '"agent_id"' not in rendered


def test_baseline_adapter_carries_direct_named_phases_and_aggregates_all_call_usage() -> None:
    available = lambda value: {
        "availability": "available",
        "value": value,
        "reason": None,
    }
    unavailable = {
        "availability": "unavailable",
        "value": None,
        "reason": "estimated cost was not supplied",
    }
    first = RawModelCall(
        component="interaction",
        model="openai/gpt-4.1-mini",
        elapsed_ms=1,
        request_sha256="1" * 64,
        usage={
            "prompt_tokens": available(10),
            "completion_tokens": available(2),
            "cached_tokens": available(1),
            "total_tokens": available(12),
            "provider_cost_usd": available(0.01),
            "estimated_cost_usd": unavailable,
        },
    )
    second = first.model_copy(
        update={
            "component": "execution",
            "request_sha256": "2" * 64,
            "usage": {
                "prompt_tokens": available(20),
                "completion_tokens": available(3),
                "cached_tokens": available(2),
                "total_tokens": available(23),
                "provider_cost_usd": available(0.02),
                "estimated_cost_usd": unavailable,
            },
        }
    )
    observation = _observation().model_copy(
        update={
            "raw_model_calls": (first, second),
            "raw_phase_timings": (
                RawPhaseTiming(
                    phase="gmail_tool",
                    started_monotonic_ns=100,
                    finished_monotonic_ns=111,
                    elapsed_ns=11,
                ),
                RawPhaseTiming(
                    phase="total_run",
                    started_monotonic_ns=90,
                    finished_monotonic_ns=140,
                    elapsed_ns=50,
                ),
            ),
        }
    )

    result = adapt_baseline(observation)

    assert result.usage.input_tokens.value == 30
    assert result.usage.output_tokens.value == 5
    assert result.usage.cached_tokens.value == 3
    assert result.usage.total_tokens.value == 35
    assert result.cost.amount.value == 0.03
    phases = [item for item in result.timings.value if "phase" in item]
    assert [item["phase"] for item in phases] == ["gmail_tool", "total_run"]


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


def test_real_early_failed_observation_does_not_export_placeholder_zeroes(
    tmp_path,
) -> None:
    request = BaselineTurnRequest(
        run_id="early-failure",
        data_dir=str(tmp_path / "missing-data"),
        event_path=str(tmp_path / "run" / "events.jsonl"),
        user_message="fixture",
    )
    observation = _failed_observation(
        request,
        code="invalid_manifest",
        phase="preflight",
        message="Bearer private-preflight-token",
    )

    result = adapt_baseline(observation)

    for field in ("roster_count", "prompt_exposure", "identity_delta", "timings"):
        observed = getattr(result, field)
        assert observed.availability is Availability.UNAVAILABLE
        assert observed.value is None
        assert observed.reason
    assert "private-preflight-token" not in result.model_dump_json()


def test_real_missing_after_prompt_and_model_evidence_stays_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    execution_dir = data_dir / "execution_agents"
    execution_dir.mkdir(parents=True)
    roster_path = execution_dir / "roster.json"
    roster_path.write_text(json.dumps(["Known"]), encoding="utf-8")
    (execution_dir / "known.log").write_text(
        "<agent_request>before</agent_request>\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "process_context.json").write_text(
        json.dumps({"process_nonce": "adapter-fixture"}),
        encoding="utf-8",
    )
    event_path = run_dir / "events.jsonl"
    monkeypatch.setattr(
        observer_module.httpx,
        "AsyncClient",
        lambda **kwargs: _MissingAfterAndPromptClient(roster_path, **kwargs),
    )
    observation = asyncio.run(
        run_baseline_turn(
            BaselineTurnRequest(
                run_id="missing-after-prompt",
                data_dir=str(data_dir),
                event_path=str(event_path),
                user_message="fixture",
                timeout_seconds=0.1,
            )
        )
    )

    result = adapt_baseline(observation)

    assert {error.phase for error in observation.errors} >= {
        "snapshot_after",
        "events",
    }
    assert result.roster_count.availability is Availability.AVAILABLE
    assert result.roster_count.value == 1
    for field in ("prompt_exposure", "identity_delta", "timings"):
        observed = getattr(result, field)
        assert observed.availability is Availability.UNAVAILABLE
        assert observed.value is None
