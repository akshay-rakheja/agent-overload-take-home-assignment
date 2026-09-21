from __future__ import annotations

import math
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from server.services.evaluation_lab.models import (
    Availability,
    ObservedValue,
    TraceEvent,
    TraceEventKind,
)
from server.services.evaluation_lab.trace import consolidate_trace


RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
TURN_ID = UUID("22222222-2222-4222-8222-222222222222")


def _event(
    *,
    kind: TraceEventKind = TraceEventKind.RUN_METADATA,
    sequence: int = 1,
    run_id: UUID = RUN_ID,
    turn_id: UUID = TURN_ID,
    system: str = "enhanced",
) -> TraceEvent:
    return TraceEvent(
        schema_version=1,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        occurred_at=datetime(2026, 9, 21, 12, 0, tzinfo=UTC),
        system=system,
        kind=kind,
        payload={"available": True, "count": sequence},
    )


@pytest.mark.parametrize("availability", list(Availability))
def test_observed_value_round_trips_every_availability(availability: Availability) -> None:
    value = ObservedValue[dict[str, bool]](
        availability=availability,
        value={"observed": True} if availability is Availability.AVAILABLE else None,
        reason=None if availability is Availability.AVAILABLE else "not emitted",
    )

    restored = TypeAdapter(ObservedValue[dict[str, bool]]).validate_json(
        value.model_dump_json()
    )

    assert restored == value


@pytest.mark.parametrize("kind", list(TraceEventKind))
def test_trace_event_round_trips_every_discriminator(kind: TraceEventKind) -> None:
    event = _event(kind=kind)

    restored = TraceEvent.model_validate_json(event.model_dump_json())

    assert restored == event
    assert restored.kind is kind


def test_trace_event_rejects_unknown_schema_version() -> None:
    payload = _event().model_dump(mode="json")
    payload["schema_version"] = 2

    with pytest.raises(ValidationError, match="schema_version"):
        TraceEvent.model_validate(payload)


def test_trace_event_rejects_coerced_sequence_values() -> None:
    payload = _event().model_dump(mode="python")
    payload["sequence"] = "1"

    with pytest.raises(ValidationError, match="sequence"):
        TraceEvent.model_validate(payload)


def test_trace_event_payload_is_deeply_immutable_and_detached() -> None:
    source = {"nested": {"ids": ["fixture_fact_001"]}}
    event = TraceEvent(
        **_event().model_dump(exclude={"payload"}),
        payload=source,
    )
    source["nested"]["ids"].append("outside-mutation")

    with pytest.raises(TypeError):
        event.payload["new"] = "mutation"
    with pytest.raises((AttributeError, TypeError)):
        event.payload["nested"]["ids"].append("inside-mutation")
    assert event.model_dump(mode="json")["payload"] == {
        "nested": {"ids": ["fixture_fact_001"]}
    }


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_trace_event_rejects_non_finite_json_values(value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        TraceEvent(
            **_event().model_dump(exclude={"payload"}),
            payload={"metric": value},
        )


@pytest.mark.parametrize(
    "availability", [Availability.NOT_APPLICABLE, Availability.UNAVAILABLE]
)
def test_unsupported_observed_values_cannot_encode_invented_zero(
    availability: Availability,
) -> None:
    with pytest.raises(ValidationError, match="must not carry a value"):
        ObservedValue[int](availability=availability, value=0, reason="unsupported")


@pytest.mark.parametrize("sequences", [(1, 1), (2, 1), (1, 3)])
def test_consolidation_rejects_duplicate_non_monotonic_or_gapped_sequences(
    sequences: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        consolidate_trace([_event(sequence=value) for value in sequences])


@pytest.mark.parametrize(
    "second",
    [
        _event(sequence=2, run_id=UUID("33333333-3333-4333-8333-333333333333")),
        _event(sequence=2, turn_id=UUID("44444444-4444-4444-8444-444444444444")),
        _event(sequence=2, system="baseline"),
    ],
)
def test_consolidation_rejects_mixed_run_turn_or_system(second: TraceEvent) -> None:
    with pytest.raises(ValueError, match="same run, turn, and system"):
        consolidate_trace([_event(), second])


def test_run_metadata_maps_emitted_facts_without_inventing_missing_values() -> None:
    result = consolidate_trace(
        [
            TraceEvent(
                **_event().model_dump(exclude={"payload"}),
                payload={"revision": "enhanced", "mode": "lab", "roster_count": 5},
            )
        ]
    )

    assert result.revision.value == "enhanced"
    assert result.mode.value == "lab"
    assert result.roster_count.value == 5
    assert result.candidates.availability is Availability.UNAVAILABLE
    assert result.candidates.value is None
    assert result.usage.total_tokens.availability is Availability.UNAVAILABLE
    assert result.usage.total_tokens.value is None
    assert result.cost.amount.availability is Availability.UNAVAILABLE
    assert result.cost.amount.value is None


def test_consolidation_redacts_directly_constructed_events_before_export() -> None:
    result = consolidate_trace(
        [
            TraceEvent(
                **_event().model_dump(exclude={"payload"}),
                payload={"revision": "Bearer direct-construction-secret", "mode": "lab"},
            )
        ]
    )

    assert result.revision.value == "[REDACTED]"


def test_consolidation_revalidates_copied_event_schema_version() -> None:
    bypassed = _event().model_copy(update={"schema_version": 2})

    with pytest.raises(ValidationError, match="schema_version"):
        consolidate_trace([bypassed])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_consolidation_rejects_copied_non_finite_payloads(value: float) -> None:
    bypassed = _event().model_copy(update={"payload": {"metric": value}})

    with pytest.raises(ValueError, match="finite|JSON"):
        consolidate_trace([bypassed])
