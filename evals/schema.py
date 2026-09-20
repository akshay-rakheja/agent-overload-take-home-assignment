"""Validated schema and loading rules for routing evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


RoutingSplit = Literal["development", "test"]
RoutingAction = Literal["reuse", "create_new", "abstain"]
AgentStatus = Literal["hot", "dormant", "archived"]

REQUIRED_CATEGORIES = {
    "exact_named_follow_up",
    "paraphrased_follow_up",
    "contextual_pronoun",
    "old_relevant_identity",
    "recent_irrelevant_distractor",
    "semantically_similar_identities",
    "novel_task",
    "ambiguous_task",
    "archived_identity_recovery",
    "unicode_punctuation_variant",
}


class AgentFixture(BaseModel):
    """Portable agent record used by the offline routing corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    purpose: str = ""
    aliases: list[str] = Field(default_factory=list)
    status: AgentStatus = "hot"
    last_used_at: str = "2026-01-01T00:00:00Z"
    use_count: int = Field(default=0, ge=0)
    memory_summary: str = ""


class SyntheticRosterReference(BaseModel):
    """Recipe for a deterministic roster with an optional authored target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int = Field(ge=1, le=100_000)
    target: AgentFixture | None = None
    target_index: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def target_fits_roster(self) -> "SyntheticRosterReference":
        if self.target is not None and self.target_index >= self.size:
            raise ValueError("target_index must be smaller than synthetic roster size")
        return self


class RoutingCase(BaseModel):
    """One labeled reuse/create/abstain decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    split: RoutingSplit
    category: str = Field(min_length=1)
    query: str = Field(min_length=1)
    conversation_context: str = ""
    agents: list[AgentFixture] | None = None
    synthetic_roster: SyntheticRosterReference | None = None
    expected_action: RoutingAction
    expected_agent_id: str | None = None
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources_and_expected_target(self) -> "RoutingCase":
        if (self.agents is None) == (self.synthetic_roster is None):
            raise ValueError("exactly one of agents or synthetic_roster is required")
        if self.expected_action == "reuse" and not self.expected_agent_id:
            raise ValueError("expected_agent_id is required when expected_action is reuse")
        if self.expected_action != "reuse" and self.expected_agent_id is not None:
            raise ValueError("expected_agent_id is only valid when expected_action is reuse")

        available_ids = {agent.agent_id for agent in self.agents or []}
        if self.synthetic_roster and self.synthetic_roster.target:
            available_ids.add(self.synthetic_roster.target.agent_id)
        if self.expected_agent_id and self.expected_agent_id not in available_ids:
            raise ValueError("expected_agent_id must identify an agent supplied by the case")
        return self


def validate_corpus(cases: list[RoutingCase]) -> list[RoutingCase]:
    """Enforce split, identity, and coverage rules across a complete corpus."""

    if len(cases) < 40:
        raise ValueError("routing corpus must contain at least 40 cases")

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("routing corpus contains duplicate case_id values")

    for split in ("development", "test"):
        count = sum(case.split == split for case in cases)
        if count < 20:
            raise ValueError(f"routing corpus requires at least 20 {split} cases; found {count}")

    categories = {case.category for case in cases}
    missing_categories = REQUIRED_CATEGORIES - categories
    if missing_categories:
        raise ValueError(f"routing corpus is missing categories: {sorted(missing_categories)}")

    return cases


def load_routing_corpus(path: Path) -> list[RoutingCase]:
    """Load JSONL cases and report the exact invalid line on failure."""

    cases: list[RoutingCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            cases.append(RoutingCase.model_validate_json(raw_line))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid routing case at {path}:{line_number}: {exc}") from exc
    return validate_corpus(cases)

