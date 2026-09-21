"""Provider-neutral model usage evidence and monotonic phase measurement."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from time import monotonic_ns
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from .models import Availability, ObservedValue, TraceEventKind
from .trace import TraceTiming, emit_trace, trace_timing


class PhaseName(str, Enum):
    RETRIEVAL_ROUTING = "retrieval_routing"
    INTERACTION_MODEL = "interaction_model"
    EXECUTION_MODEL = "execution_model"
    GMAIL_TOOL = "gmail_tool"
    TOTAL_RUN = "total_run"


def _missing(reason: str) -> ObservedValue[Any]:
    return ObservedValue(availability=Availability.UNAVAILABLE, reason=reason)


class UsageRecord(BaseModel):
    """Normalized evidence; absence is explicit and never coerced to zero."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt_tokens: ObservedValue[int] = Field(
        default_factory=lambda: _missing("provider usage was not emitted")
    )
    completion_tokens: ObservedValue[int] = Field(
        default_factory=lambda: _missing("provider usage was not emitted")
    )
    cached_tokens: ObservedValue[int] = Field(
        default_factory=lambda: _missing("provider usage was not emitted")
    )
    total_tokens: ObservedValue[int] = Field(
        default_factory=lambda: _missing("provider usage was not emitted")
    )
    provider_cost_usd: ObservedValue[float] = Field(
        default_factory=lambda: _missing("provider cost was not emitted")
    )
    estimated_cost_usd: ObservedValue[float] = Field(
        default_factory=lambda: _missing("estimated cost was not supplied")
    )


def _record(**values: ObservedValue[Any]) -> UsageRecord:
    return UsageRecord(**{key: value.model_dump() for key, value in values.items()})


def _at_path(value: object, path: tuple[str, ...]) -> object | None:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first(value: object, *paths: tuple[str, ...]) -> object | None:
    for path in paths:
        found = _at_path(value, path)
        if found is not None:
            return found
    return None


def _token_value(
    value: object,
    name: str,
    warnings: list[str],
    *,
    missing_reason: str,
) -> ObservedValue[int]:
    if value is None:
        return _missing(missing_reason)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        warnings.append(f"{name} is malformed")
        return _missing(f"provider {name} was malformed")
    return ObservedValue(availability=Availability.AVAILABLE, value=value)


def _cost_value(
    value: object,
    name: str,
    warnings: list[str],
    *,
    missing_reason: str,
) -> ObservedValue[float]:
    if value is None:
        return _missing(missing_reason)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        warnings.append(f"{name} is malformed")
        return _missing(f"{name} was malformed")
    return ObservedValue(availability=Availability.AVAILABLE, value=float(value))


def normalize_usage(
    response: object,
    *,
    estimated_cost_usd: float | None = None,
    warnings: list[str] | None = None,
) -> UsageRecord:
    """Normalize documented OpenRouter/OpenAI-compatible response shapes."""

    observed_warnings = warnings if warnings is not None else []
    if isinstance(response, Mapping) and response.get("error") is not None:
        reason = "provider returned an error response"
        return _record(
            prompt_tokens=_missing(reason),
            completion_tokens=_missing(reason),
            cached_tokens=_missing(reason),
            total_tokens=_missing(reason),
            provider_cost_usd=_missing(reason),
            estimated_cost_usd=_cost_value(
                estimated_cost_usd,
                "estimated_cost_usd",
                observed_warnings,
                missing_reason="estimated cost was not supplied",
            ),
        )

    usage = _first(
        response,
        ("usage",),
        ("data", "usage"),
        ("response", "usage"),
        ("result", "usage"),
    )
    usage_mapping: Mapping[str, object] = usage if isinstance(usage, Mapping) else {}
    usage_reason = (
        "provider usage was malformed"
        if usage is not None and not isinstance(usage, Mapping)
        else "provider usage was not emitted"
    )
    prompt = _token_value(
        _first(usage_mapping, ("prompt_tokens",), ("input_tokens",)),
        "prompt_tokens",
        observed_warnings,
        missing_reason=usage_reason,
    )
    completion = _token_value(
        _first(usage_mapping, ("completion_tokens",), ("output_tokens",)),
        "completion_tokens",
        observed_warnings,
        missing_reason=usage_reason,
    )
    cached = _token_value(
        _first(
            usage_mapping,
            ("cached_tokens",),
            ("prompt_tokens_details", "cached_tokens"),
            ("input_tokens_details", "cached_tokens"),
            ("cache_read_input_tokens",),
        ),
        "cached_tokens",
        observed_warnings,
        missing_reason=usage_reason,
    )
    total = _token_value(
        _first(usage_mapping, ("total_tokens",)),
        "total_tokens",
        observed_warnings,
        missing_reason=usage_reason,
    )
    provider_cost_source = _first(
        usage_mapping,
        ("cost",),
        ("provider_cost",),
        ("cost_usd",),
        ("total_cost",),
    )
    if provider_cost_source is None:
        provider_cost_source = _first(
            response,
            ("cost",),
            ("data", "cost"),
            ("provider_cost",),
        )
    provider_cost = _cost_value(
        provider_cost_source,
        "provider_cost_usd",
        observed_warnings,
        missing_reason="provider cost was not emitted",
    )
    estimated_cost = _cost_value(
        estimated_cost_usd,
        "estimated_cost_usd",
        observed_warnings,
        missing_reason="estimated cost was not supplied",
    )

    if (
        prompt.availability is Availability.AVAILABLE
        and completion.availability is Availability.AVAILABLE
        and total.availability is Availability.AVAILABLE
    ):
        expected = int(prompt.value) + int(completion.value)
        if total.value != expected:
            observed_warnings.append(
                f"total_tokens mismatch: provider={total.value} calculated={expected}"
            )

    return _record(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        total_tokens=total,
        provider_cost_usd=provider_cost,
        estimated_cost_usd=estimated_cost,
    )


def emit_usage_evidence(response: object, *, role: str) -> UsageRecord:
    """Emit normalized model evidence plus non-fatal consistency warnings."""

    warnings: list[str] = []
    record = normalize_usage(response, warnings=warnings)
    emit_trace(
        TraceEventKind.USAGE,
        {"role": role, **record.model_dump(mode="json")},
    )
    if record.provider_cost_usd.availability is Availability.AVAILABLE:
        emit_trace(
            TraceEventKind.COST,
            {
                "role": role,
                "provider_cost_usd": record.provider_cost_usd.model_dump(mode="json"),
                "estimated_cost_usd": record.estimated_cost_usd.model_dump(mode="json"),
            },
        )
    for warning in warnings:
        emit_trace(
            TraceEventKind.OBSERVABILITY_WARNING,
            {"boundary": "model_usage", "role": role, "warning": warning},
        )
    return record


@contextmanager
def monotonic_phase(
    phase: PhaseName | str,
    *,
    clock: Callable[[], int] = monotonic_ns,
) -> Iterator[TraceTiming]:
    """Measure one named phase and emit only after freezing its endpoint."""

    phase_name = phase.value if isinstance(phase, PhaseName) else str(phase)
    with trace_timing(clock) as timing:
        yield timing
    emit_trace(
        TraceEventKind.PHASE_TIMING,
        {
            "phase": phase_name,
            "started_monotonic_ns": timing.started_monotonic_ns,
            "finished_monotonic_ns": timing.finished_monotonic_ns,
            "elapsed_ns": timing.elapsed_ns,
        },
    )


measure_phase = monotonic_phase


__all__ = [
    "PhaseName",
    "UsageRecord",
    "emit_usage_evidence",
    "measure_phase",
    "monotonic_phase",
    "normalize_usage",
]
