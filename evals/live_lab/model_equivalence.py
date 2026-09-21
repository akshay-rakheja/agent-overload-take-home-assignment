"""Fail-closed equivalence checks for model-backed evaluation roles."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from server.config import ModelRole

from .contracts import EquivalenceReport


class ModelConfigurationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())
    role: ModelRole
    model_id: str
    provider: str
    temperature: float
    top_p: float
    max_tokens: int = Field(gt=0)
    seed_requested: int | None = None
    seed_acknowledged: bool | None = None
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(default=0, ge=0)


_FIELDS = (
    "model_id",
    "provider",
    "temperature",
    "top_p",
    "max_tokens",
    "seed_requested",
    "seed_acknowledged",
    "timeout_seconds",
    "max_retries",
)


def _index(
    side: str, evidence: Iterable[ModelConfigurationEvidence]
) -> tuple[dict[ModelRole, ModelConfigurationEvidence], list[str]]:
    indexed: dict[ModelRole, ModelConfigurationEvidence] = {}
    differences: list[str] = []
    seen: set[ModelRole] = set()
    for item in evidence:
        if item.role in seen:
            differences.append(f"{side}.duplicate.{item.role.value}")
            continue
        seen.add(item.role)
        indexed[item.role] = item
    for role in ModelRole:
        if role not in indexed:
            differences.append(f"{side}.missing.{role.value}")
    return indexed, differences


def compare_model_equivalence(
    baseline: Iterable[ModelConfigurationEvidence],
    enhanced: Iterable[ModelConfigurationEvidence],
) -> EquivalenceReport:
    """Reject grading unless every role and controlled field matches."""

    baseline_by_role, differences = _index("baseline", baseline)
    enhanced_by_role, enhanced_differences = _index("enhanced", enhanced)
    differences.extend(enhanced_differences)
    checks: dict[str, bool] = {}
    for role in ModelRole:
        left = baseline_by_role.get(role)
        right = enhanced_by_role.get(role)
        role_matches = left is not None and right is not None
        if left is not None and right is not None:
            for field in _FIELDS:
                if getattr(left, field) != getattr(right, field):
                    differences.append(f"{role.value}.{field}")
                    role_matches = False
        checks[role.value] = role_matches
    return EquivalenceReport(
        equivalent=not differences and all(checks.values()),
        checks=checks,
        differences=tuple(differences),
    )


compare_model_configurations = compare_model_equivalence


__all__ = [
    "ModelConfigurationEvidence",
    "compare_model_configurations",
    "compare_model_equivalence",
]
