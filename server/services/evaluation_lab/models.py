"""Typed, versioned contracts for privacy-safe Evaluation Lab traces."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Generic, Literal, Mapping, TypeVar
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})  # type: ignore[return-value]
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)  # type: ignore[return-value]
    return value


def _thaw_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value  # type: ignore[return-value]


class Availability(str, Enum):
    AVAILABLE = "available"
    INFERRED = "inferred"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


ObservedT = TypeVar("ObservedT")


class ObservedValue(_FrozenModel, Generic[ObservedT]):
    availability: Availability
    value: ObservedT | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _unsupported_values_are_absent(self) -> "ObservedValue[ObservedT]":
        if self.availability in {
            Availability.NOT_APPLICABLE,
            Availability.UNAVAILABLE,
        } and self.value is not None:
            raise ValueError(f"{self.availability.value} observations must not carry a value")
        return self


class TraceEventKind(str, Enum):
    RUN_METADATA = "run_metadata"
    ROSTER_SNAPSHOT = "roster_snapshot"
    PROMPT_EXPOSURE = "prompt_exposure"
    CANDIDATES = "candidates"
    ROUTING_DECISION = "routing_decision"
    AUTHORIZATION = "authorization"
    DISPATCH_ATTEMPT = "dispatch_attempt"
    DISPATCH_RESULT = "dispatch_result"
    IDENTITY = "identity"
    GMAIL_EVIDENCE = "gmail_evidence"
    FINAL_RESPONSE = "final_response"
    CONTEXT_METRICS = "context_metrics"
    PHASE_TIMING = "phase_timing"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    USAGE = "usage"
    COST = "cost"
    ERROR = "error"
    OBSERVABILITY_WARNING = "observability_warning"


class TraceEvent(_FrozenModel):
    schema_version: Literal[1] = 1
    run_id: UUID
    turn_id: UUID
    sequence: int = Field(ge=1)
    occurred_at: datetime
    system: Literal["baseline", "enhanced"]
    kind: TraceEventKind
    payload: Mapping[str, JsonValue]

    @field_validator("payload", mode="after")
    @classmethod
    def _freeze_payload(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )

    @field_serializer("payload")
    def _serialize_payload(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        thawed = _thaw_json(value)
        assert isinstance(thawed, dict)
        return thawed

    @model_validator(mode="after")
    def _validate_identity_and_time(self) -> "TraceEvent":
        if self.run_id == self.turn_id:
            raise ValueError("run_id and turn_id must identify different scopes")
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return self


class TraceContext(_FrozenModel):
    run_id: UUID
    turn_id: UUID
    system: Literal["baseline", "enhanced"]
    revision: str
    mode: str

    @model_validator(mode="after")
    def _validate_identity(self) -> "TraceContext":
        if self.run_id == self.turn_id:
            raise ValueError("run_id and turn_id must identify different scopes")
        return self


def unavailable(reason: str = "not emitted") -> ObservedValue[JsonValue]:
    return ObservedValue(availability=Availability.UNAVAILABLE, reason=reason)


class UsagePlaceholder(_FrozenModel):
    input_tokens: ObservedValue[int] = Field(
        default_factory=lambda: ObservedValue(
            availability=Availability.UNAVAILABLE, reason="not emitted"
        )
    )
    output_tokens: ObservedValue[int] = Field(
        default_factory=lambda: ObservedValue(
            availability=Availability.UNAVAILABLE, reason="not emitted"
        )
    )
    cached_tokens: ObservedValue[int] = Field(
        default_factory=lambda: ObservedValue(
            availability=Availability.UNAVAILABLE, reason="not emitted"
        )
    )
    total_tokens: ObservedValue[int] = Field(
        default_factory=lambda: ObservedValue(
            availability=Availability.UNAVAILABLE, reason="not emitted"
        )
    )


class CostPlaceholder(_FrozenModel):
    amount: ObservedValue[float] = Field(
        default_factory=lambda: ObservedValue(
            availability=Availability.UNAVAILABLE, reason="not emitted"
        )
    )
    currency: ObservedValue[str] = Field(
        default_factory=lambda: ObservedValue(
            availability=Availability.UNAVAILABLE, reason="not emitted"
        )
    )


def _missing() -> ObservedValue[JsonValue]:
    return unavailable()


class SystemRunResult(_FrozenModel):
    schema_version: Literal[1] = 1
    run_id: UUID
    turn_id: UUID
    system: Literal["baseline", "enhanced"]
    revision: ObservedValue[str] = Field(default_factory=_missing)
    mode: ObservedValue[str] = Field(default_factory=_missing)
    roster_count: ObservedValue[int] = Field(default_factory=_missing)
    prompt_exposure: ObservedValue[JsonValue] = Field(default_factory=_missing)
    candidates: ObservedValue[list[JsonValue]] = Field(default_factory=_missing)
    decision: ObservedValue[JsonValue] = Field(default_factory=_missing)
    recommendation: ObservedValue[JsonValue] = Field(default_factory=_missing)
    authorized_ids: ObservedValue[list[str]] = Field(default_factory=_missing)
    attempted_dispatch: ObservedValue[JsonValue] = Field(default_factory=_missing)
    accepted_dispatch: ObservedValue[JsonValue] = Field(default_factory=_missing)
    selected_identity: ObservedValue[JsonValue] = Field(default_factory=_missing)
    created_identity: ObservedValue[JsonValue] = Field(default_factory=_missing)
    identity_delta: ObservedValue[JsonValue] = Field(default_factory=_missing)
    duplicates: ObservedValue[list[JsonValue]] = Field(default_factory=_missing)
    gmail_evidence: ObservedValue[list[JsonValue]] = Field(default_factory=_missing)
    final_response: ObservedValue[str] = Field(default_factory=_missing)
    context_metrics: ObservedValue[JsonValue] = Field(default_factory=_missing)
    timings: ObservedValue[list[JsonValue]] = Field(default_factory=_missing)
    usage: UsagePlaceholder = Field(default_factory=UsagePlaceholder)
    cost: CostPlaceholder = Field(default_factory=CostPlaceholder)
    errors: ObservedValue[list[JsonValue]] = Field(default_factory=_missing)
    availability_metadata: dict[str, Availability] = Field(default_factory=dict)


class TraceRunStatus(_FrozenModel):
    run_id: UUID
    event_count: int = Field(ge=0)
    last_sequence: int | None = Field(default=None, ge=1)
    complete: bool


class TraceResponse(_FrozenModel):
    events: tuple[TraceEvent, ...]
    result: SystemRunResult


__all__ = [
    "Availability",
    "CostPlaceholder",
    "ObservedValue",
    "SystemRunResult",
    "TraceContext",
    "TraceEvent",
    "TraceEventKind",
    "TraceResponse",
    "TraceRunStatus",
    "UsagePlaceholder",
]
