"""Deterministic paired repetitions and non-measured compatibility preflight."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from ...config import ModelCallConfig, ModelRole
from .models import Availability
from .policy import LabToolPolicy
from .usage import UsageRecord, normalize_usage


PRIMARY_MODEL_ID = "openai/gpt-4.1-mini"
FALLBACK_MODEL_ID = "google/gemini-2.5-flash"
_SCHEDULE_NAMESPACE = UUID("90a44066-93a3-5c13-923e-a1c4b227b8b1")


class MeasuredSystem(str, Enum):
    BASELINE = "baseline"
    ENHANCED = "enhanced"
    ENHANCED_DETERMINISTIC = "enhanced_deterministic"
    ENHANCED_JEV = "enhanced_jev"


class PairOrder(str, Enum):
    BASELINE_THEN_ENHANCED = "baseline_then_enhanced"
    ENHANCED_THEN_BASELINE = "enhanced_then_baseline"
    BASELINE_DETERMINISTIC_JEV = "baseline_deterministic_jev"
    DETERMINISTIC_JEV_BASELINE = "deterministic_jev_baseline"
    JEV_BASELINE_DETERMINISTIC = "jev_baseline_deterministic"
    BASELINE_JEV_DETERMINISTIC = "baseline_jev_deterministic"
    DETERMINISTIC_BASELINE_JEV = "deterministic_baseline_jev"
    JEV_DETERMINISTIC_BASELINE = "jev_deterministic_baseline"

    @property
    def systems(self) -> tuple[MeasuredSystem, ...]:
        if self is PairOrder.BASELINE_THEN_ENHANCED:
            return MeasuredSystem.BASELINE, MeasuredSystem.ENHANCED
        if self is PairOrder.ENHANCED_THEN_BASELINE:
            return MeasuredSystem.ENHANCED, MeasuredSystem.BASELINE
        if self is PairOrder.BASELINE_DETERMINISTIC_JEV:
            return (
                MeasuredSystem.BASELINE,
                MeasuredSystem.ENHANCED_DETERMINISTIC,
                MeasuredSystem.ENHANCED_JEV,
            )
        if self is PairOrder.DETERMINISTIC_JEV_BASELINE:
            return (
                MeasuredSystem.ENHANCED_DETERMINISTIC,
                MeasuredSystem.ENHANCED_JEV,
                MeasuredSystem.BASELINE,
            )
        if self is PairOrder.JEV_BASELINE_DETERMINISTIC:
            return (
                MeasuredSystem.ENHANCED_JEV,
                MeasuredSystem.BASELINE,
                MeasuredSystem.ENHANCED_DETERMINISTIC,
            )
        if self is PairOrder.BASELINE_JEV_DETERMINISTIC:
            return (
                MeasuredSystem.BASELINE,
                MeasuredSystem.ENHANCED_JEV,
                MeasuredSystem.ENHANCED_DETERMINISTIC,
            )
        if self is PairOrder.DETERMINISTIC_BASELINE_JEV:
            return (
                MeasuredSystem.ENHANCED_DETERMINISTIC,
                MeasuredSystem.BASELINE,
                MeasuredSystem.ENHANCED_JEV,
            )
        return (
            MeasuredSystem.ENHANCED_JEV,
            MeasuredSystem.ENHANCED_DETERMINISTIC,
            MeasuredSystem.BASELINE,
        )


class OutcomeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    BUDGET_STOP = "budget_stop"
    MALFORMED = "malformed"
    UNAVAILABLE = "unavailable"


class ScheduledPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pair_id: UUID
    scenario_id: str
    repetition: int = Field(ge=1)
    order: PairOrder

    @field_validator("scenario_id")
    @classmethod
    def _valid_scenario(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 200:
            raise ValueError("scenario_id must be bounded non-empty text")
        return normalized


THREE_WAY_CYCLE = (
    PairOrder.BASELINE_DETERMINISTIC_JEV,
    PairOrder.DETERMINISTIC_JEV_BASELINE,
    PairOrder.JEV_BASELINE_DETERMINISTIC,
    PairOrder.BASELINE_JEV_DETERMINISTIC,
    PairOrder.DETERMINISTIC_BASELINE_JEV,
    PairOrder.JEV_DETERMINISTIC_BASELINE,
)


def build_repetition_schedule(
    scenario_ids: Sequence[str], repetitions: int = 3, three_way: bool = False
) -> tuple[ScheduledPair, ...]:
    """Build stable IDs independent of caller ordering and alternate each repeat."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    normalized = [scenario.strip() for scenario in scenario_ids]
    if any(not scenario for scenario in normalized):
        raise ValueError("scenario ids must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("scenario ids must be unique")
    schedule: list[ScheduledPair] = []
    for scenario_id in normalized:
        for repetition in range(1, repetitions + 1):
            if three_way:
                order = THREE_WAY_CYCLE[(repetition - 1) % len(THREE_WAY_CYCLE)]
            else:
                order = (
                    PairOrder.BASELINE_THEN_ENHANCED
                    if repetition % 2 == 1
                    else PairOrder.ENHANCED_THEN_BASELINE
                )
            version_tag = "v2" if three_way else "v1"
            schedule.append(
                ScheduledPair(
                    pair_id=uuid5(
                        _SCHEDULE_NAMESPACE,
                        f"scheduled-pair:{version_tag}:{scenario_id}:{repetition}",
                    ),
                    scenario_id=scenario_id,
                    repetition=repetition,
                    order=order,
                )
            )
    return tuple(schedule)


class MetricObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    availability: Availability
    value: Decimal | None = None
    reason: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _finite_decimal(cls, value: object):
        if value is None:
            return None
        try:
            result = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("metric must be Decimal-compatible") from exc
        if not result.is_finite():
            raise ValueError("metric must be finite")
        return result

    @field_validator("reason")
    @classmethod
    def _bounded_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or len(normalized) > 256:
            raise ValueError("reason must be bounded non-empty text")
        return normalized

    @model_validator(mode="after")
    def _coherent(self) -> "MetricObservation":
        if self.availability is Availability.AVAILABLE and self.value is None:
            raise ValueError("available metric requires a value")
        if self.availability is not Availability.AVAILABLE and self.value is not None:
            raise ValueError("unavailable metric cannot carry a value")
        return self


class SideResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    system: MeasuredSystem
    status: OutcomeStatus
    model_id: str | None = None
    metrics: Mapping[str, MetricObservation] = Field(default_factory=dict)
    attempt_count: Literal[1] = 1
    reason: str | None = None

    @field_validator("model_id")
    @classmethod
    def _valid_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if (
            not normalized
            or "/" not in normalized
            or normalized.startswith("/")
            or normalized.endswith("/")
            or any(character.isspace() for character in normalized)
        ):
            raise ValueError("model_id must be a provider/model identifier")
        return normalized

    @field_validator("metrics", mode="after")
    @classmethod
    def _freeze_metrics(
        cls, value: Mapping[str, MetricObservation]
    ) -> Mapping[str, MetricObservation]:
        normalized: dict[str, MetricObservation] = {}
        for key, observation in value.items():
            name = str(key).strip()
            if not name or len(name) > 128:
                raise ValueError("metric names must be bounded non-empty text")
            normalized[name] = observation
        return MappingProxyType(normalized)

    @field_serializer("metrics")
    def _serialize_metrics(
        self, value: Mapping[str, MetricObservation]
    ) -> dict[str, MetricObservation]:
        return dict(value)

    @field_validator("reason")
    @classmethod
    def _valid_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or len(normalized) > 256:
            raise ValueError("outcome reason must be bounded non-empty text")
        return normalized

    @model_validator(mode="after")
    def _coherent(self) -> "SideResult":
        if self.status is OutcomeStatus.SUCCESS:
            if self.model_id is None:
                raise ValueError("successful measured result requires a model_id")
            if self.reason is not None:
                raise ValueError("successful result cannot carry a failure reason")
        elif self.reason is None:
            raise ValueError("non-success result requires a reason")
        return self


class PairResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scheduled: ScheduledPair
    outcomes: tuple[SideResult, ...]

    @model_validator(mode="after")
    def _unique_sides(self) -> "PairResult":
        if len(self.outcomes) > 2:
            raise ValueError("a pair can contain at most two system outcomes")
        systems = [outcome.system for outcome in self.outcomes]
        if len(set(systems)) != len(systems):
            raise ValueError("a pair cannot contain duplicate system outcomes")
        return self


class MetricDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    available_count: int = Field(ge=0)
    unavailable_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    minimum: Decimal | None = None
    p50: Decimal | None = None
    p95: Decimal | None = None
    maximum: Decimal | None = None


class RepetitionAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    records: tuple[PairResult, ...]
    metrics: Mapping[str, MetricDistribution]
    outcome_counts: Mapping[OutcomeStatus, int]
    failure_count: int = Field(ge=0)
    unavailable_record_count: int = Field(ge=0)
    partial_pair_count: int = Field(ge=0)
    model_id: str | None = None


def _nearest_rank(values: Sequence[Decimal], percentile: Decimal) -> Decimal:
    rank = max(1, math.ceil(float(percentile * len(values))))
    return values[rank - 1]


def aggregate_repetitions(records: Sequence[PairResult]) -> RepetitionAggregate:
    """Aggregate available observations without discarding any terminal record."""

    preserved = tuple(records)
    pair_ids = [record.scheduled.pair_id for record in preserved]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("repetition results contain duplicate pair ids")
    model_ids = {
        outcome.model_id
        for record in preserved
        for outcome in record.outcomes
        if outcome.model_id is not None
    }
    if len(model_ids) > 1:
        raise ValueError("a measured result set cannot mix models")

    outcome_counts = {status: 0 for status in OutcomeStatus}
    failure_count = 0
    unavailable_records = 0
    partial_pairs = 0
    metric_names: set[str] = set()
    for record in preserved:
        metric_names.update(
            name for outcome in record.outcomes for name in outcome.metrics
        )
        missing_sides = len(record.scheduled.order.systems) - len(record.outcomes)
        if missing_sides:
            partial_pairs += 1
            unavailable_records += missing_sides
        for outcome in record.outcomes:
            outcome_counts[outcome.status] += 1
            if outcome.status is OutcomeStatus.UNAVAILABLE:
                unavailable_records += 1
            elif outcome.status is not OutcomeStatus.SUCCESS:
                failure_count += 1

    distributions: dict[str, MetricDistribution] = {}
    for metric_name in sorted(metric_names):
        available: list[Decimal] = []
        unavailable = 0
        metric_failures = 0
        for record in preserved:
            outcomes = {outcome.system: outcome for outcome in record.outcomes}
            for system in record.scheduled.order.systems:
                outcome = outcomes.get(system)
                if outcome is None:
                    unavailable += 1
                    continue
                observation = outcome.metrics.get(metric_name)
                if (
                    observation is None
                    or observation.availability is not Availability.AVAILABLE
                    or observation.value is None
                ):
                    unavailable += 1
                else:
                    available.append(observation.value)
                if outcome.status not in {
                    OutcomeStatus.SUCCESS,
                    OutcomeStatus.UNAVAILABLE,
                }:
                    metric_failures += 1
        available.sort()
        distributions[metric_name] = MetricDistribution(
            available_count=len(available),
            unavailable_count=unavailable,
            failure_count=metric_failures,
            minimum=available[0] if available else None,
            p50=_nearest_rank(available, Decimal("0.50")) if available else None,
            p95=_nearest_rank(available, Decimal("0.95")) if available else None,
            maximum=available[-1] if available else None,
        )

    return RepetitionAggregate(
        records=preserved,
        metrics=distributions,
        outcome_counts=outcome_counts,
        failure_count=failure_count,
        unavailable_record_count=unavailable_records,
        partial_pair_count=partial_pairs,
        model_id=next(iter(model_ids)) if model_ids else None,
    )


READ_ONLY_PREFLIGHT_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "gmail_get_contacts",
        "description": "Compatibility-only schema probe for a read-only Gmail tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_name": {"type": "string"},
                "person_fields": {"type": "string"},
                "include_other_contacts": {"type": "boolean"},
                "page_token": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
}


class CompatibilityFailureCode(str, Enum):
    CHAT_COMPLETIONS_INCOMPATIBLE = "chat_completions_incompatible"
    TOOL_SCHEMA_INCOMPATIBLE = "tool_schema_incompatible"


_FAILURE_REASONS = {
    CompatibilityFailureCode.CHAT_COMPLETIONS_INCOMPATIBLE: (
        "provider recorded an incompatible chat-completions interface"
    ),
    CompatibilityFailureCode.TOOL_SCHEMA_INCOMPATIBLE: (
        "provider did not return the required schema-valid read-only tool call"
    ),
}


class PreflightIncompatibility(RuntimeError):
    """Typed, sanitized signal that is authorized to trigger the fallback."""

    def __init__(self, code: CompatibilityFailureCode) -> None:
        self.code = code
        super().__init__(_FAILURE_REASONS[code])


class RecordedIncompatibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: CompatibilityFailureCode
    reason: str


class PreflightUnavailableCode(str, Enum):
    TRANSPORT = "transport_unavailable"
    PROVIDER_ERROR = "provider_error"
    RESPONSE_IDENTITY = "response_identity_unavailable"
    MALFORMED_ENVELOPE = "malformed_response_envelope"
    MALFORMED_TOOL_CALL = "malformed_tool_call_envelope"


class ProviderEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label_sha256: str

    @field_validator("label_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        normalized = value.casefold()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("provider evidence must be a SHA-256 digest")
        return normalized


class CompatibilityAttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())
    model_id: str
    config: ModelCallConfig
    provider: ProviderEvidence | None = None
    usage: UsageRecord
    compatible: bool
    tool_name: str | None = None
    tool_argument_keys: tuple[str, ...] | None = None
    tool_arguments_sha256: str | None = None
    incompatibility: RecordedIncompatibility | None = None
    unavailable_code: PreflightUnavailableCode | None = None

    @field_validator("tool_arguments_sha256")
    @classmethod
    def _valid_argument_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.casefold()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("tool argument evidence must be a SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def _coherent(self) -> "CompatibilityAttemptRecord":
        if self.compatible:
            if (
                self.provider is None
                or self.tool_name != "gmail_get_contacts"
                or self.tool_argument_keys is None
                or self.tool_arguments_sha256 is None
                or self.incompatibility is not None
                or self.unavailable_code is not None
            ):
                raise ValueError("compatible preflight is missing required evidence")
        elif (
            self.incompatibility is not None
            and self.unavailable_code is not None
        ):
            raise ValueError("preflight failure cannot be both incompatible and unavailable")
        elif self.incompatibility is None and self.unavailable_code is None:
            raise ValueError("unready preflight requires a typed terminal fact")
        return self


class FullResetEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    baseline_restored: bool
    enhanced_restored: bool
    equivalence_verified: bool
    reset_id: str

    @field_validator("reset_id")
    @classmethod
    def _valid_reset_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("reset_id must be bounded non-empty text")
        return normalized

    @property
    def complete(self) -> bool:
        return (
            self.baseline_restored
            and self.enhanced_restored
            and self.equivalence_verified
        )


class CompatibilityPreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())
    ready: bool
    selected_model_id: str | None = None
    attempts: tuple[CompatibilityAttemptRecord, ...]
    reset_evidence: FullResetEvidence | None = None
    role_models: Mapping[MeasuredSystem, Mapping[ModelRole, str]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _coherent_selection(self) -> "CompatibilityPreflightResult":
        if not self.ready:
            if self.selected_model_id is not None or self.role_models:
                raise ValueError("unready preflight cannot select measured models")
            return self
        if self.selected_model_id not in {PRIMARY_MODEL_ID, FALLBACK_MODEL_ID}:
            raise ValueError("ready preflight must select an approved model")
        if set(self.role_models) not in (
            {MeasuredSystem.BASELINE, MeasuredSystem.ENHANCED},
            set(MeasuredSystem),
        ):
            raise ValueError("both measured systems must be pinned")
        for system in self.role_models:
            roles = self.role_models[system]
            if set(roles) != set(ModelRole):
                raise ValueError("all five roles must be pinned")
            if set(roles.values()) != {self.selected_model_id}:
                raise ValueError("a measured role set cannot mix models")
        if self.selected_model_id == FALLBACK_MODEL_ID:
            if self.reset_evidence is None or not self.reset_evidence.complete:
                raise ValueError("fallback selection requires a verified full reset")
            if not self.attempts or self.attempts[0].incompatibility is None:
                raise ValueError("fallback selection requires recorded incompatibility")
        return self


def _measured_config(model_id: str) -> ModelCallConfig:
    return ModelCallConfig(
        model_id=model_id,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1000,
        timeout_seconds=60.0,
        max_retries=0,
    )


def _pinned_roles(
    model_id: str,
    systems: Sequence[MeasuredSystem] = (MeasuredSystem.BASELINE, MeasuredSystem.ENHANCED),
) -> dict[MeasuredSystem, dict[ModelRole, str]]:
    return {
        system: {role: model_id for role in ModelRole}
        for system in systems
    }


def _incompatibility_record(
    model_id: str,
    config: ModelCallConfig,
    code: CompatibilityFailureCode,
    *,
    response: object = None,
) -> CompatibilityAttemptRecord:
    return CompatibilityAttemptRecord(
        model_id=model_id,
        config=config,
        usage=normalize_usage(response),
        compatible=False,
        incompatibility=RecordedIncompatibility(
            code=code,
            reason=_FAILURE_REASONS[code],
        ),
    )


def _provider_evidence(provider: object) -> ProviderEvidence | None:
    if not isinstance(provider, str):
        return None
    label = provider.strip()
    if (
        not label
        or len(label) > 128
        or any(not character.isprintable() for character in label)
    ):
        return None
    return ProviderEvidence(
        label_sha256=hashlib.sha256(label.encode("utf-8")).hexdigest()
    )


def _validate_tool_arguments(arguments: object) -> dict[str, str | bool] | None:
    if not isinstance(arguments, str):
        return None

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        arguments = json.loads(arguments, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(arguments, Mapping):
        return None
    types: dict[str, type] = {
        "resource_name": str,
        "person_fields": str,
        "include_other_contacts": bool,
        "page_token": str,
    }
    normalized: dict[str, str | bool] = {}
    for key, value in arguments.items():
        if (
            key not in types
            or not isinstance(value, types[key])
            or (types[key] is not bool and isinstance(value, bool))
        ):
            return None
        if isinstance(value, str) and (
            len(value) > 2048 or any(not character.isprintable() for character in value)
        ):
            return None
        normalized[str(key)] = value
    if (
        "resource_name" in normalized
        and normalized["resource_name"] != "people/me"
    ):
        return None
    return normalized


def _unavailable_record(
    model_id: str,
    config: ModelCallConfig,
    code: PreflightUnavailableCode,
    *,
    response: object = None,
) -> CompatibilityAttemptRecord:
    return CompatibilityAttemptRecord(
        model_id=model_id,
        config=config,
        usage=normalize_usage(response),
        compatible=False,
        unavailable_code=code,
    )


def _validate_preflight_response(
    response: object, *, model_id: str, config: ModelCallConfig
) -> CompatibilityAttemptRecord:
    if not isinstance(response, Mapping):
        return _unavailable_record(
            model_id,
            config,
            PreflightUnavailableCode.MALFORMED_ENVELOPE,
            response=response,
        )
    if response.get("error") is not None:
        return _unavailable_record(
            model_id,
            config,
            PreflightUnavailableCode.PROVIDER_ERROR,
            response=response,
        )
    response_model = response.get("model")
    provider = _provider_evidence(response.get("provider"))
    if response_model != model_id or provider is None:
        return _unavailable_record(
            model_id,
            config,
            PreflightUnavailableCode.RESPONSE_IDENTITY,
            response=response,
        )
    choices = response.get("choices")
    calls: object = None
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            calls = message.get("tool_calls")
    valid_call: Mapping[str, object] | None = None
    if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], Mapping):
        valid_call = calls[0]
    call_id = valid_call.get("id") if valid_call is not None else None
    call_type = valid_call.get("type") if valid_call is not None else None
    function = valid_call.get("function") if valid_call is not None else None
    name = function.get("name") if isinstance(function, Mapping) else None
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    normalized_arguments = _validate_tool_arguments(arguments)
    decision = LabToolPolicy().decide_model_tool(str(name or ""))
    if (
        call_type != "function"
        or not isinstance(call_id, str)
        or not call_id.strip()
        or len(call_id.strip()) > 128
        or any(not character.isprintable() for character in call_id)
        or name != "gmail_get_contacts"
        or normalized_arguments is None
        or not decision.allowed
    ):
        return _unavailable_record(
            model_id,
            config,
            PreflightUnavailableCode.MALFORMED_TOOL_CALL,
            response=response,
        )
    canonical_arguments = json.dumps(
        normalized_arguments,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return CompatibilityAttemptRecord(
        model_id=model_id,
        config=config,
        provider=provider,
        usage=normalize_usage(response),
        compatible=True,
        tool_name=name,
        tool_argument_keys=tuple(sorted(normalized_arguments)),
        tool_arguments_sha256=hashlib.sha256(canonical_arguments).hexdigest(),
    )


async def _attempt_once(
    model_id: str,
    attempt: Callable[
        [ModelCallConfig, Mapping[str, object]],
        Awaitable[Mapping[str, object]],
    ],
) -> CompatibilityAttemptRecord:
    config = _measured_config(model_id)
    try:
        response = await attempt(config, READ_ONLY_PREFLIGHT_TOOL)
    except PreflightIncompatibility as exc:
        return _incompatibility_record(model_id, config, exc.code)
    except Exception:
        return _unavailable_record(
            model_id,
            config,
            PreflightUnavailableCode.TRANSPORT,
        )
    return _validate_preflight_response(response, model_id=model_id, config=config)


async def run_compatibility_preflight(
    *,
    attempt: Callable[
        [ModelCallConfig, Mapping[str, object]],
        Awaitable[Mapping[str, object]],
    ],
    reset_both: Callable[[], FullResetEvidence | Awaitable[FullResetEvidence]] | None = None,
) -> CompatibilityPreflightResult:
    """Run one primary probe and only fall back after recorded incompatibility."""

    primary = await _attempt_once(PRIMARY_MODEL_ID, attempt)
    if primary.compatible:
        return CompatibilityPreflightResult(
            ready=True,
            selected_model_id=PRIMARY_MODEL_ID,
            attempts=(primary,),
            role_models=_pinned_roles(PRIMARY_MODEL_ID),
        )
    if primary.incompatibility is None or reset_both is None:
        return CompatibilityPreflightResult(ready=False, attempts=(primary,))

    reset_result = reset_both()
    reset = await reset_result if inspect.isawaitable(reset_result) else reset_result
    if not isinstance(reset, FullResetEvidence):
        raise TypeError("reset_both must return FullResetEvidence")
    if not reset.complete:
        return CompatibilityPreflightResult(
            ready=False,
            attempts=(primary,),
            reset_evidence=reset,
        )

    fallback = await _attempt_once(FALLBACK_MODEL_ID, attempt)
    if not fallback.compatible:
        return CompatibilityPreflightResult(
            ready=False,
            attempts=(primary, fallback),
            reset_evidence=reset,
        )
    return CompatibilityPreflightResult(
        ready=True,
        selected_model_id=FALLBACK_MODEL_ID,
        attempts=(primary, fallback),
        reset_evidence=reset,
        role_models=_pinned_roles(FALLBACK_MODEL_ID),
    )


__all__ = [
    "CompatibilityAttemptRecord",
    "CompatibilityFailureCode",
    "CompatibilityPreflightResult",
    "FALLBACK_MODEL_ID",
    "FullResetEvidence",
    "MeasuredSystem",
    "MetricDistribution",
    "MetricObservation",
    "OutcomeStatus",
    "PRIMARY_MODEL_ID",
    "PairOrder",
    "PairResult",
    "PreflightIncompatibility",
    "PreflightUnavailableCode",
    "ProviderEvidence",
    "READ_ONLY_PREFLIGHT_TOOL",
    "RepetitionAggregate",
    "ScheduledPair",
    "SideResult",
    "aggregate_repetitions",
    "build_repetition_schedule",
    "run_compatibility_preflight",
]
