"""Honest mapping from historical observations to the shared result schema."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from evals.live_lab.raw_observation import BaselineObservation

from .models import (
    Availability,
    CostPlaceholder,
    ObservedValue,
    SystemRunResult,
    UsagePlaceholder,
)
from .redaction import redact_value
from .usage import UsageRecord


def _available(value: Any) -> ObservedValue[Any]:
    return ObservedValue(
        availability=Availability.AVAILABLE,
        value=redact_value(value),
    )


def _inferred(value: Any, reason: str) -> ObservedValue[Any]:
    return ObservedValue(
        availability=Availability.INFERRED,
        value=redact_value(value),
        reason=str(redact_value(reason)),
    )


def _not_applicable(reason: str) -> ObservedValue[Any]:
    return ObservedValue(
        availability=Availability.NOT_APPLICABLE,
        reason=str(redact_value(reason)),
    )


def _unavailable(reason: str) -> ObservedValue[Any]:
    return ObservedValue(
        availability=Availability.UNAVAILABLE,
        reason=str(redact_value(reason)),
    )


def _ordered_multiset_delta(
    before: tuple[str, ...], after: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    remaining_before = Counter(before)
    added: list[str] = []
    for name in after:
        if remaining_before[name]:
            remaining_before[name] -= 1
        else:
            added.append(name)

    remaining_after = Counter(after)
    removed: list[str] = []
    for name in before:
        if remaining_after[name]:
            remaining_after[name] -= 1
        else:
            removed.append(name)
    return added, removed


def _aggregate_observed(values: list[ObservedValue[Any]], fact: str) -> ObservedValue[Any]:
    if not values:
        return _unavailable(f"baseline {fact} was not observed")
    if not all(value.availability is Availability.AVAILABLE for value in values):
        return _unavailable(f"one or more baseline model calls omitted {fact}")
    return ObservedValue(
        availability=Availability.AVAILABLE,
        value=sum(value.value for value in values if value.value is not None),
    )


def _aggregate_usage(observation: BaselineObservation) -> tuple[UsagePlaceholder, CostPlaceholder]:
    records = [
        UsageRecord.model_validate_json(
            json.dumps(call.usage, allow_nan=False, sort_keys=True, separators=(",", ":"))
        )
        for call in observation.raw_model_calls
    ]
    usage = UsagePlaceholder(
        input_tokens=_aggregate_observed(
            [record.prompt_tokens for record in records], "prompt tokens"
        ).model_dump(),
        output_tokens=_aggregate_observed(
            [record.completion_tokens for record in records], "completion tokens"
        ).model_dump(),
        cached_tokens=_aggregate_observed(
            [record.cached_tokens for record in records], "cached tokens"
        ).model_dump(),
        total_tokens=_aggregate_observed(
            [record.total_tokens for record in records], "total tokens"
        ).model_dump(),
    )
    costs = [record.provider_cost_usd for record in records]
    aggregate_cost = _aggregate_observed(costs, "provider cost")
    cost = CostPlaceholder(
        amount=aggregate_cost.model_dump(),
        currency=(
            ObservedValue(
                availability=Availability.AVAILABLE,
                value="USD",
            ).model_dump()
            if aggregate_cost.availability is Availability.AVAILABLE
            else _unavailable("baseline provider cost currency was not observed").model_dump()
        ),
    )
    return usage, cost


def adapt_baseline(observation: BaselineObservation) -> SystemRunResult:
    """Adapt direct baseline evidence without projecting enhanced-only concepts."""

    try:
        run_id = UUID(observation.run_id)
    except ValueError:
        run_id = uuid5(
            NAMESPACE_URL,
            f"openpoke-baseline-run:{observation.run_id}",
        )
    turn_id = uuid5(NAMESPACE_URL, f"openpoke-baseline-turn:{observation.run_id}")
    unsupported_ranking = "historical full-roster selection has no ranked candidate set"
    unsupported_identity = "historical baseline has no enhanced stable identity contract"
    unsupported_authorization = "historical baseline has no deterministic authorization set"
    dispatch_unavailable = "raw baseline observation does not expose exact dispatch acceptance"
    failed_placeholder = (
        observation.inference_reason
        == "baseline observation failed before trustworthy evidence was available"
    )
    before_available = not failed_placeholder or bool(
        observation.roster_before or observation.journal_hashes_before
    )
    after_available = not failed_placeholder and not any(
        error.phase == "snapshot_after" for error in observation.errors
    )
    prompt_available = bool(observation.prompt_xml_sha256)
    timings_available = bool(observation.raw_model_calls or observation.raw_phase_timings)

    selected_identity: ObservedValue[Any]
    if (
        observation.inferred_action in {"reuse", "create_new"}
        and observation.inferred_name is not None
    ):
        selected_identity = _inferred(
            {
                "name": observation.inferred_name,
                "reason": observation.inference_reason,
            },
            observation.inference_reason,
        )
    elif observation.inferred_action == "abstain":
        selected_identity = _not_applicable("baseline inferred no dispatch")
    else:
        selected_identity = _unavailable(observation.inference_reason)

    if (
        observation.inferred_action == "create_new"
        and observation.inferred_name is not None
    ):
        created_identity = _inferred(
            {
                "name": observation.inferred_name,
                "reason": observation.inference_reason,
            },
            observation.inference_reason,
        )
    elif observation.inferred_action == "unobservable":
        created_identity = _unavailable(observation.inference_reason)
    else:
        created_identity = _not_applicable(
            "baseline did not infer one new historical roster name"
        )

    added_names, removed_names = _ordered_multiset_delta(
        observation.roster_before, observation.roster_after
    )
    timings = [
        {
            **call.model_dump(mode="json", exclude_defaults=True),
            "error_type": call.error_type,
        }
        for call in observation.raw_model_calls
    ]
    timings.extend(timing.model_dump(mode="json") for timing in observation.raw_phase_timings)
    errors = [error.model_dump(mode="json") for error in observation.errors]
    values: dict[str, Any] = {
        "revision": _unavailable("baseline observation does not carry revision metadata"),
        "mode": _inferred(
            "full_roster",
            "historical baseline observer captures the full-roster selection path",
        ),
        "roster_count": (
            _available(len(observation.roster_before))
            if before_available
            else _unavailable("baseline pre-turn roster evidence was not observed")
        ),
        "prompt_exposure": (
            _available(
                {
                    "prompt_xml_sha256": observation.prompt_xml_sha256,
                    "prompt_characters": observation.prompt_characters,
                    "exposed_names": list(observation.exposed_names),
                    "exposed_name_count": len(observation.exposed_names),
                }
            )
            if prompt_available
            else _unavailable("baseline interaction prompt evidence was not observed")
        ),
        "candidates": _not_applicable(unsupported_ranking),
        "decision": _inferred(
            {
                "action": observation.inferred_action,
                "reason": observation.inference_reason,
            },
            observation.inference_reason,
        ),
        "recommendation": _not_applicable(unsupported_ranking),
        "authorized_ids": _not_applicable(unsupported_authorization),
        "attempted_dispatch": _unavailable(dispatch_unavailable),
        "accepted_dispatch": _unavailable(dispatch_unavailable),
        "selected_identity": selected_identity,
        "created_identity": created_identity,
        "identity_delta": (
            _available(
                {
                    "roster_before_count": len(observation.roster_before),
                    "roster_after_count": len(observation.roster_after),
                    "added_names": added_names,
                    "removed_names": removed_names,
                }
            )
            if before_available and after_available
            else _unavailable("baseline post-turn roster evidence was not observed")
        ),
        "duplicates": _not_applicable(unsupported_identity),
        "gmail_evidence": _unavailable(
            "raw baseline observation has no normalized Gmail evidence"
        ),
        "final_response": (
            _available(observation.final_response)
            if observation.final_response is not None
            else _unavailable("baseline final response was not observed")
        ),
        "context_metrics": _unavailable(
            "historical baseline has no bounded enhanced context metrics"
        ),
        "timings": (
            _available(timings)
            if timings_available
            else _unavailable("baseline model timing evidence was not observed")
        ),
        "errors": _available(errors),
    }
    usage, cost = _aggregate_usage(observation)
    result = SystemRunResult(
        run_id=run_id,
        turn_id=turn_id,
        system="baseline",
        usage=usage,
        cost=cost,
        **{
            key: value.model_dump() if isinstance(value, ObservedValue) else value
            for key, value in values.items()
        },
    )
    availability = {
        field: observed.availability
        for field, observed in result.__dict__.items()
        if isinstance(observed, ObservedValue)
    }
    return result.model_copy(update={"availability_metadata": availability})


__all__ = ["adapt_baseline"]
