from __future__ import annotations

import asyncio
import importlib
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from server.config import ModelRole, Settings
from server.services.evaluation_lab.models import Availability
from server.services.evaluation_lab.repetitions import (
    FALLBACK_MODEL_ID,
    PRIMARY_MODEL_ID,
    CompatibilityFailureCode,
    FullResetEvidence,
    MeasuredSystem,
    MetricObservation,
    OutcomeStatus,
    PairOrder,
    PairResult,
    PreflightUnavailableCode,
    PreflightIncompatibility,
    SideResult,
    aggregate_repetitions,
    build_repetition_schedule,
    run_compatibility_preflight,
)


app_module = importlib.import_module("server.app")


def _success_response(
    model_id: str,
    *,
    provider: str = "fixture-provider",
    arguments: str = '{"resource_name":"people/me"}',
    call_id: str | None = "fixture-call-1",
    call_type: str = "function",
) -> dict[str, object]:
    call: dict[str, object] = {
        "type": call_type,
        "function": {
            "name": "gmail_get_contacts",
            "arguments": arguments,
        },
    }
    if call_id is not None:
        call["id"] = call_id
    return {
        "model": model_id,
        "provider": provider,
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        call
                    ]
                }
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
            "cost": 0.0001,
        },
    }


def _metric(value: str | None, *, reason: str = "missing") -> MetricObservation:
    if value is None:
        return MetricObservation(
            availability=Availability.UNAVAILABLE,
            reason=reason,
        )
    return MetricObservation(
        availability=Availability.AVAILABLE,
        value=Decimal(value),
    )


def _side(
    system: MeasuredSystem,
    *,
    status: OutcomeStatus = OutcomeStatus.SUCCESS,
    value: str | None = "1",
    model_id: str = PRIMARY_MODEL_ID,
) -> SideResult:
    return SideResult(
        system=system,
        status=status,
        model_id=model_id,
        metrics={"latency_ms": _metric(value)},
        attempt_count=1,
        reason=(None if status is OutcomeStatus.SUCCESS else status.value),
    )


def test_default_schedule_has_three_stable_alternating_pairs() -> None:
    first = build_repetition_schedule(["scenario-alpha"])
    second = build_repetition_schedule(["scenario-alpha"])

    assert first == second
    assert [pair.repetition for pair in first] == [1, 2, 3]
    assert [pair.order for pair in first] == [
        PairOrder.BASELINE_THEN_ENHANCED,
        PairOrder.ENHANCED_THEN_BASELINE,
        PairOrder.BASELINE_THEN_ENHANCED,
    ]
    assert len({pair.pair_id for pair in first}) == 3


def test_schedule_ids_do_not_depend_on_scenario_input_order() -> None:
    forward = build_repetition_schedule(["alpha", "beta"])
    reverse = build_repetition_schedule(["beta", "alpha"])

    forward_ids = {
        (pair.scenario_id, pair.repetition): pair.pair_id for pair in forward
    }
    reverse_ids = {
        (pair.scenario_id, pair.repetition): pair.pair_id for pair in reverse
    }
    assert forward_ids == reverse_ids


@pytest.mark.parametrize(
    "status",
    [
        OutcomeStatus.FAILURE,
        OutcomeStatus.TIMEOUT,
        OutcomeStatus.BUDGET_STOP,
        OutcomeStatus.MALFORMED,
        OutcomeStatus.UNAVAILABLE,
    ],
)
def test_pair_result_preserves_each_terminal_failure_kind(status: OutcomeStatus) -> None:
    scheduled = build_repetition_schedule(["alpha"], repetitions=1)[0]
    result = PairResult(
        scheduled=scheduled,
        outcomes=(
            _side(MeasuredSystem.BASELINE, status=status, value=None),
            _side(MeasuredSystem.ENHANCED),
        ),
    )

    assert result.outcomes[0].status is status
    assert result.outcomes[1].status is OutcomeStatus.SUCCESS


def test_partial_pair_is_retained_and_missing_side_is_counted_unavailable() -> None:
    scheduled = build_repetition_schedule(["alpha"], repetitions=1)[0]
    partial = PairResult(
        scheduled=scheduled,
        outcomes=(_side(MeasuredSystem.BASELINE),),
    )

    aggregate = aggregate_repetitions([partial])

    assert aggregate.records == (partial,)
    assert aggregate.partial_pair_count == 1
    assert aggregate.unavailable_record_count == 1
    assert aggregate.outcome_counts[OutcomeStatus.SUCCESS] == 1


def test_aggregate_uses_only_available_values_and_keeps_all_records() -> None:
    schedule = build_repetition_schedule(["alpha"])
    records = (
        PairResult(
            scheduled=schedule[0],
            outcomes=(
                _side(MeasuredSystem.BASELINE, value="1"),
                _side(MeasuredSystem.ENHANCED, value=None),
            ),
        ),
        PairResult(
            scheduled=schedule[1],
            outcomes=(
                _side(MeasuredSystem.BASELINE, value="2"),
                _side(
                    MeasuredSystem.ENHANCED,
                    status=OutcomeStatus.TIMEOUT,
                    value=None,
                ),
            ),
        ),
        PairResult(
            scheduled=schedule[2],
            outcomes=(
                _side(MeasuredSystem.BASELINE, value="100"),
                _side(MeasuredSystem.ENHANCED, value="3"),
            ),
        ),
    )

    aggregate = aggregate_repetitions(records)
    distribution = aggregate.metrics["latency_ms"]

    assert aggregate.records == records
    assert aggregate.failure_count == 1
    assert aggregate.unavailable_record_count == 0
    assert distribution.available_count == 4
    assert distribution.unavailable_count == 2
    assert distribution.minimum == Decimal("1")
    assert distribution.p50 == Decimal("2")
    assert distribution.p95 == Decimal("100")
    assert distribution.maximum == Decimal("100")


def test_side_result_forbids_silent_application_retries() -> None:
    with pytest.raises(ValidationError):
        SideResult(
            system=MeasuredSystem.BASELINE,
            status=OutcomeStatus.SUCCESS,
            model_id=PRIMARY_MODEL_ID,
            attempt_count=2,
        )


def test_aggregate_rejects_any_mixed_model_result_set() -> None:
    schedule = build_repetition_schedule(["alpha"])
    records = [
        PairResult(
            scheduled=schedule[0],
            outcomes=(
                _side(MeasuredSystem.BASELINE, model_id=PRIMARY_MODEL_ID),
                _side(MeasuredSystem.ENHANCED, model_id=PRIMARY_MODEL_ID),
            ),
        ),
        PairResult(
            scheduled=schedule[1],
            outcomes=(
                _side(MeasuredSystem.BASELINE, model_id=FALLBACK_MODEL_ID),
                _side(MeasuredSystem.ENHANCED, model_id=FALLBACK_MODEL_ID),
            ),
        ),
    ]

    with pytest.raises(ValueError, match="mix models"):
        aggregate_repetitions(records)


def test_primary_preflight_records_config_provider_usage_and_read_only_tool_call() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def attempt(config, tool_schema):
        calls.append((config.model_id, tool_schema))
        return _success_response(config.model_id)

    result = asyncio.run(run_compatibility_preflight(attempt=attempt))

    assert result.ready is True
    assert result.selected_model_id == PRIMARY_MODEL_ID
    assert len(result.attempts) == 1
    record = result.attempts[0]
    assert record.compatible is True
    assert record.provider is not None
    assert len(record.provider.label_sha256) == 64
    assert record.usage.prompt_tokens.value == 8
    assert record.tool_name == "gmail_get_contacts"
    assert record.tool_argument_keys == ("resource_name",)
    assert len(record.tool_arguments_sha256 or "") == 64
    serialized = result.model_dump_json()
    assert "fixture-provider" not in serialized
    assert "people/me" not in serialized
    assert calls[0][0] == PRIMARY_MODEL_ID
    assert calls[0][1]["function"]["name"] == "gmail_get_contacts"
    assert set(result.role_models) == {
        MeasuredSystem.BASELINE,
        MeasuredSystem.ENHANCED,
    }
    for system in result.role_models:
        assert set(result.role_models[system]) == set(ModelRole)
        assert set(result.role_models[system].values()) == {PRIMARY_MODEL_ID}


def test_recorded_tool_incompatibility_requires_full_reset_before_fallback() -> None:
    order: list[str] = []

    async def attempt(config, _tool_schema):
        order.append(f"attempt:{config.model_id}")
        if config.model_id == PRIMARY_MODEL_ID:
            raise PreflightIncompatibility(
                CompatibilityFailureCode.TOOL_SCHEMA_INCOMPATIBLE
            )
        return _success_response(config.model_id, provider="Google")

    def reset_both() -> FullResetEvidence:
        order.append("reset")
        return FullResetEvidence(
            baseline_restored=True,
            enhanced_restored=True,
            equivalence_verified=True,
            reset_id="fixture-reset",
        )

    result = asyncio.run(
        run_compatibility_preflight(attempt=attempt, reset_both=reset_both)
    )

    assert order == [
        f"attempt:{PRIMARY_MODEL_ID}",
        "reset",
        f"attempt:{FALLBACK_MODEL_ID}",
    ]
    assert result.ready is True
    assert result.selected_model_id == FALLBACK_MODEL_ID
    assert result.attempts[0].incompatibility is not None
    assert (
        result.attempts[0].incompatibility.code
        is CompatibilityFailureCode.TOOL_SCHEMA_INCOMPATIBLE
    )
    assert result.reset_evidence is not None
    for system in result.role_models:
        assert set(result.role_models[system].values()) == {FALLBACK_MODEL_ID}


def test_explicit_chat_incompatibility_can_select_fallback_only_after_reset() -> None:
    attempts = 0

    async def attempt(config, _tool_schema):
        nonlocal attempts
        attempts += 1
        if config.model_id == PRIMARY_MODEL_ID:
            raise PreflightIncompatibility(
                CompatibilityFailureCode.CHAT_COMPLETIONS_INCOMPATIBLE
            )
        return _success_response(config.model_id)

    result = asyncio.run(
        run_compatibility_preflight(
            attempt=attempt,
            reset_both=lambda: FullResetEvidence(
                baseline_restored=True,
                enhanced_restored=True,
                equivalence_verified=True,
                reset_id="fixture-reset",
            ),
        )
    )

    assert attempts == 2
    assert result.selected_model_id == FALLBACK_MODEL_ID


def test_timeout_is_unavailable_and_cannot_trigger_fallback() -> None:
    attempts = 0

    async def attempt(_config, _tool_schema):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("private timeout detail")

    def forbidden_reset() -> FullResetEvidence:
        pytest.fail("timeout must not select fallback")

    result = asyncio.run(
        run_compatibility_preflight(attempt=attempt, reset_both=forbidden_reset)
    )

    assert attempts == 1
    assert result.ready is False
    assert result.selected_model_id is None
    assert result.attempts[0].incompatibility is None
    assert result.attempts[0].unavailable_code is PreflightUnavailableCode.TRANSPORT
    assert "private timeout detail" not in result.model_dump_json()


@pytest.mark.parametrize(
    "reset",
    [
        FullResetEvidence(
            baseline_restored=False,
            enhanced_restored=True,
            equivalence_verified=True,
            reset_id="bad-reset",
        ),
        FullResetEvidence(
            baseline_restored=True,
            enhanced_restored=True,
            equivalence_verified=False,
            reset_id="bad-reset",
        ),
    ],
)
def test_incomplete_reset_blocks_fallback_attempt(reset: FullResetEvidence) -> None:
    calls = 0

    async def attempt(config, _tool_schema):
        nonlocal calls
        calls += 1
        raise PreflightIncompatibility(
            CompatibilityFailureCode.TOOL_SCHEMA_INCOMPATIBLE
        )

    result = asyncio.run(
        run_compatibility_preflight(attempt=attempt, reset_both=lambda: reset)
    )

    assert calls == 1
    assert result.ready is False
    assert result.selected_model_id is None


@pytest.mark.parametrize(
    "response",
    [
        {"error": {"code": 429, "message": "private rate limit detail"}},
        {"error": {"code": 401, "message": "private auth detail"}},
        {"model": PRIMARY_MODEL_ID, "choices": []},
        {"provider": "fixture-provider", "choices": []},
        ["malformed-envelope"],
    ],
)
def test_provider_errors_missing_identity_and_malformed_envelopes_never_fallback(
    response: object,
) -> None:
    calls = 0

    async def attempt(_config, _tool_schema):
        nonlocal calls
        calls += 1
        return response

    def forbidden_reset() -> FullResetEvidence:
        pytest.fail("unavailable response must not reset or select fallback")

    result = asyncio.run(
        run_compatibility_preflight(attempt=attempt, reset_both=forbidden_reset)
    )

    assert calls == 1
    assert result.ready is False
    assert result.attempts[0].incompatibility is None
    assert result.attempts[0].unavailable_code is not None
    assert "private rate limit detail" not in result.model_dump_json()
    assert "private auth detail" not in result.model_dump_json()


@pytest.mark.parametrize(
    "response",
    [
        _success_response(PRIMARY_MODEL_ID, call_id=None),
        _success_response(PRIMARY_MODEL_ID, call_type="tool"),
        _success_response(PRIMARY_MODEL_ID, arguments="not-json"),
        {
            **_success_response(PRIMARY_MODEL_ID),
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "fixture-call-1",
                                "type": "function",
                                "function": {
                                    "name": "gmail_get_contacts",
                                    "arguments": {"resource_name": "people/me"},
                                },
                            }
                        ]
                    }
                }
            ],
        },
    ],
)
def test_tool_call_wire_envelope_must_be_complete_and_arguments_json_string(
    response: dict[str, object],
) -> None:
    async def attempt(_config, _tool_schema):
        return response

    result = asyncio.run(run_compatibility_preflight(attempt=attempt))

    assert result.ready is False
    assert result.attempts[0].incompatibility is None
    assert (
        result.attempts[0].unavailable_code
        is PreflightUnavailableCode.MALFORMED_TOOL_CALL
    )


def test_preflight_serialization_never_persists_raw_provider_or_argument_values() -> None:
    private_provider = "provider-private-marker.example"
    private_token = "private-page-token-marker"

    async def attempt(config, _tool_schema):
        return _success_response(
            config.model_id,
            provider=private_provider,
            arguments=(
                '{"resource_name":"people/me","page_token":"'
                + private_token
                + '"}'
            ),
        )

    result = asyncio.run(run_compatibility_preflight(attempt=attempt))
    serialized = result.model_dump_json()

    assert result.ready is True
    assert private_provider not in serialized
    assert private_token not in serialized
    assert "people/me" not in serialized


def test_lab_mode_disables_background_services_and_automatic_summarization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Service:
        def __init__(self, name: str) -> None:
            self.name = name

        async def start(self) -> None:
            events.append(f"start:{self.name}")

        async def stop(self) -> None:
            events.append(f"stop:{self.name}")

    settings = Settings(
        lab_enabled=True,
        lab_model=PRIMARY_MODEL_ID,
        lab_composio_user_id="fixture-user",
        server_host="127.0.0.1",
        conversation_summary_threshold=100,
    )
    monkeypatch.setattr(app_module, "_settings", settings)
    monkeypatch.setattr(app_module, "get_trigger_scheduler", lambda: Service("scheduler"))
    monkeypatch.setattr(app_module, "get_important_email_watcher", lambda: Service("watcher"))

    asyncio.run(app_module._start_trigger_scheduler())
    asyncio.run(app_module._stop_trigger_scheduler())

    assert events == []
    assert settings.summarization_enabled is True
    assert settings.automatic_summarization_enabled is False


def test_ordinary_mode_keeps_background_services_enabled() -> None:
    settings = Settings(lab_enabled=False, conversation_summary_threshold=100)

    assert settings.background_model_activity_enabled is True
    assert settings.automatic_summarization_enabled is True
