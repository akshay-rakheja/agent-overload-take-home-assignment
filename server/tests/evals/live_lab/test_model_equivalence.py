from __future__ import annotations

import pytest

from evals.live_lab.model_equivalence import (
    ModelConfigurationEvidence,
    compare_model_equivalence,
)
from server.config import ModelRole


def _evidence(role: ModelRole, **changes) -> ModelConfigurationEvidence:
    values = {
        "role": role,
        "model_id": "openai/gpt-4.1-mini",
        "provider": "OpenAI",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1000,
        "seed_requested": None,
        "seed_acknowledged": None,
        "timeout_seconds": 60.0,
        "max_retries": 0,
    }
    values.update(changes)
    return ModelConfigurationEvidence(**values)


def _all_roles() -> tuple[ModelConfigurationEvidence, ...]:
    return tuple(_evidence(role) for role in ModelRole)


def test_all_five_exact_role_configurations_are_equivalent() -> None:
    report = compare_model_equivalence(_all_roles(), _all_roles())

    assert report.equivalent is True
    assert all(report.checks.values())
    assert report.differences == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "openai/gpt-4.1"),
        ("provider", "Azure"),
        ("temperature", 0.2),
        ("top_p", 0.9),
        ("max_tokens", 999),
        ("seed_requested", 1313),
        ("seed_acknowledged", True),
        ("timeout_seconds", 59.0),
        ("max_retries", 1),
    ],
)
def test_any_configuration_or_seed_state_difference_is_ungradeable(
    field: str, value: object
) -> None:
    baseline = _all_roles()
    enhanced = tuple(
        _evidence(role, **({field: value} if role is ModelRole.EXECUTION else {}))
        for role in ModelRole
    )

    report = compare_model_equivalence(baseline, enhanced)

    assert report.equivalent is False
    assert report.checks["execution"] is False
    assert report.differences == (f"execution.{field}",)


def test_missing_or_duplicate_role_rejects_the_pair() -> None:
    complete = _all_roles()

    missing = compare_model_equivalence(complete[:-1], complete)
    duplicate = compare_model_equivalence((*complete, complete[0]), complete)

    assert missing.equivalent is False
    assert "baseline.missing.classifier" in missing.differences
    assert duplicate.equivalent is False
    assert "baseline.duplicate.interaction" in duplicate.differences
