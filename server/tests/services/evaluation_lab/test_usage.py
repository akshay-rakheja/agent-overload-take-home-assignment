from __future__ import annotations

from uuid import uuid4

from server.services.evaluation_lab.models import Availability, TraceContext, TraceEventKind
from server.services.evaluation_lab.trace import trace_scope
from server.services.evaluation_lab.usage import (
    PhaseName,
    UsageRecord,
    monotonic_phase,
    normalize_usage,
)


class _Sink:
    def __init__(self, clock=None) -> None:
        self.events = []
        self.clock = clock

    def emit(self, event) -> None:
        if self.clock is not None:
            self.clock["now"] += 1_000_000
        self.events.append(event)


def _context() -> TraceContext:
    return TraceContext(
        run_id=uuid4(),
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture",
        mode="offline",
    )


def test_normalize_top_level_usage_and_provider_cost() -> None:
    warnings: list[str] = []
    record = normalize_usage(
        {
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "cached_tokens": 3,
                "total_tokens": 18,
                "cost": 0.00042,
            }
        },
        warnings=warnings,
    )

    assert isinstance(record, UsageRecord)
    assert record.prompt_tokens.value == 13
    assert record.completion_tokens.value == 5
    assert record.cached_tokens.value == 3
    assert record.total_tokens.value == 18
    assert record.provider_cost_usd.value == 0.00042
    assert record.estimated_cost_usd.availability is Availability.UNAVAILABLE
    assert record.estimated_cost_usd.value is None
    assert warnings == []


def test_normalize_nested_usage_and_nested_cached_tokens() -> None:
    record = normalize_usage(
        {
            "data": {
                "usage": {
                    "input_tokens": 21,
                    "output_tokens": 8,
                    "total_tokens": 29,
                    "prompt_tokens_details": {"cached_tokens": 9},
                },
                "cost": 0.001,
            }
        },
        estimated_cost_usd=0.0009,
    )

    assert record.prompt_tokens.value == 21
    assert record.completion_tokens.value == 8
    assert record.cached_tokens.value == 9
    assert record.total_tokens.value == 29
    assert record.provider_cost_usd.value == 0.001
    assert record.estimated_cost_usd.value == 0.0009


def test_missing_usage_is_unavailable_never_zero() -> None:
    record = normalize_usage({"choices": []})

    for field in UsageRecord.model_fields:
        observed = getattr(record, field)
        assert observed.availability is Availability.UNAVAILABLE
        assert observed.value is None


def test_malformed_usage_and_error_shapes_remain_unavailable() -> None:
    warnings: list[str] = []
    malformed = normalize_usage(
        {
            "usage": {
                "prompt_tokens": "12",
                "completion_tokens": -1,
                "cached_tokens": True,
                "total_tokens": 11,
                "cost": "free",
            }
        },
        warnings=warnings,
    )
    error = normalize_usage({"error": {"code": 429, "message": "private"}})

    assert malformed.prompt_tokens.availability is Availability.UNAVAILABLE
    assert malformed.completion_tokens.availability is Availability.UNAVAILABLE
    assert malformed.cached_tokens.availability is Availability.UNAVAILABLE
    assert malformed.provider_cost_usd.availability is Availability.UNAVAILABLE
    assert len(warnings) == 4
    assert all(
        getattr(error, field).availability is Availability.UNAVAILABLE
        for field in UsageRecord.model_fields
    )
    assert "error" in (error.prompt_tokens.reason or "")


def test_total_mismatch_is_preserved_with_warning() -> None:
    warnings: list[str] = []
    record = normalize_usage(
        {
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 10,
            }
        },
        warnings=warnings,
    )

    assert record.total_tokens.value == 10
    assert warnings == ["total_tokens mismatch: provider=10 calculated=9"]


def test_named_phase_uses_monotonic_duration_without_trace_sink_time() -> None:
    clock = {"now": 100}
    sink = _Sink(clock)

    with trace_scope(_context(), sink):
        with monotonic_phase(PhaseName.INTERACTION_MODEL, clock=lambda: clock["now"]):
            clock["now"] += 25

    timing = [event for event in sink.events if event.kind is TraceEventKind.PHASE_TIMING]
    assert len(timing) == 1
    assert timing[0].payload == {
        "phase": "interaction_model",
        "started_monotonic_ns": 100,
        "finished_monotonic_ns": 125,
        "elapsed_ns": 25,
    }
