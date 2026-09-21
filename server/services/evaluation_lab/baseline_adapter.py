"""Honest mapping from historical observations to the shared result schema."""

from __future__ import annotations

from collections import Counter
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from evals.live_lab.raw_observation import BaselineObservation

from .models import (
    Availability,
    ObservedValue,
    SystemRunResult,
)
from .redaction import redact_value


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
    timings_available = bool(observation.raw_model_calls)

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
    timings = [call.model_dump(mode="json") for call in observation.raw_model_calls]
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
    result = SystemRunResult(
        run_id=run_id,
        turn_id=turn_id,
        system="baseline",
        **values,
    )
    availability = {
        field: observed.availability
        for field, observed in result.__dict__.items()
        if isinstance(observed, ObservedValue)
    }
    return result.model_copy(update={"availability_metadata": availability})


__all__ = ["adapt_baseline"]
