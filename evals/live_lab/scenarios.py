"""Predeclared controlled and exploratory Evaluation Lab scenarios."""

from __future__ import annotations

import argparse
import json
import re
from enum import Enum
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from server.services.evaluation_lab.repetitions import ScheduledPair, build_repetition_schedule

from .contracts import ExpectedAction
from .fixture_email import fixture_fact_ids


class ScenarioTrack(str, Enum):
    CONTROLLED = "controlled"
    NATURAL = "natural"


REQUIRED_CONTROLLED_FAMILIES = frozenset(
    {
        "exact_named_reuse",
        "paraphrased_reuse",
        "pronoun_follow_up",
        "ambiguity_abstention",
        "novel_creation",
        "duplicate_prevention",
        "old_relevant_recent_distractor",
        "honest_no_result",
        "hundred_agent_overload",
        "ten_thousand_history",
        "instagram_security_vs_engagement",
        "similar_video_ambiguity",
        "dormant_archived_recovery",
        "thousand_agent_overload",
    }
)
_KNOWN_IDENTITIES = frozenset(
    {
        "instagram-security",
        "instagram-engagement",
        "ai-video-newsletter",
        "ai-video-receipts",
        "slug-space",
        "slug-hyphen",
        "same-name-retention",
        "same-name-partners",
        "unicode-accented",
        "unicode-plain",
        "new:calendar-workflow",
        "new:clipweaver-auditor",
    }
)
_BANNED = re.compile(
    r"[\w.+-]+@[\w.-]+|https?://|\boauth\b|\bbearer\b|"
    r"api[_ -]?key|access[_ -]?token|client[_ -]?secret|authorization[_ -]?code",
    re.IGNORECASE,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResetProfile(_FrozenModel):
    profile_id: str
    fixture_seed: int
    roster_size: Literal[10, 100, 500, 1000]
    history_entries: int = Field(ge=0)


RESET_PROFILES: Mapping[str, ResetProfile] = MappingProxyType(
    {
        "standard-100": ResetProfile(
            profile_id="standard-100",
            fixture_seed=1313,
            roster_size=100,
            history_entries=10_000,
        ),
        "scale-1000": ResetProfile(
            profile_id="scale-1000",
            fixture_seed=1313,
            roster_size=1000,
            history_entries=10_000,
        ),
    }
)


class ScenarioTurn(_FrozenModel):
    text: str

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 2_000:
            raise ValueError("turn text must be bounded nonempty text")
        return normalized


class GmailExpectation(_FrozenModel):
    operation: Literal["GMAIL_FETCH_EMAILS"]
    query: str
    fact_ids: tuple[str, ...] = ()
    expect_no_result: bool = False

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 500:
            raise ValueError("Gmail query must be bounded nonempty text")
        return normalized

    @model_validator(mode="after")
    def _no_result_has_no_facts(self) -> "GmailExpectation":
        if self.expect_no_result and self.fact_ids:
            raise ValueError("no-result Gmail expectation cannot declare facts")
        return self


class ScenarioOutcome(_FrozenModel):
    action: ExpectedAction
    logical_identity: str | None = None
    require_clarification: bool = False

    @model_validator(mode="after")
    def _action_identity(self) -> "ScenarioOutcome":
        if self.action in {ExpectedAction.REUSE, ExpectedAction.CREATE_NEW}:
            if self.logical_identity is None:
                raise ValueError("reuse/create action requires a logical identity")
        elif self.logical_identity is not None:
            raise ValueError("abstain action forbids a logical identity")
        if self.action is not ExpectedAction.ABSTAIN and self.require_clarification:
            raise ValueError("only abstain may require clarification")
        return self


class ScenarioDefinition(_FrozenModel):
    scenario_id: str
    track: Literal[ScenarioTrack.CONTROLLED] = ScenarioTrack.CONTROLLED
    family: str
    title: str
    turns: tuple[ScenarioTurn, ...]
    expected: ScenarioOutcome
    gmail: GmailExpectation
    reset_profile: ResetProfile
    repetitions: int = Field(ge=3)
    optional: bool = False
    budget_guarded: bool = False

    @field_validator("scenario_id", "family", "title")
    @classmethod
    def _bounded_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 200:
            raise ValueError("scenario labels must be bounded nonempty text")
        return normalized

    @model_validator(mode="after")
    def _coherent(self) -> "ScenarioDefinition":
        if not self.turns:
            raise ValueError("scenario turns must be nonempty")
        if self.expected.logical_identity not in _KNOWN_IDENTITIES | {None}:
            raise ValueError("unknown logical identity")
        unknown_facts = set(self.gmail.fact_ids) - fixture_fact_ids()
        if unknown_facts:
            raise ValueError(f"unknown fact ids: {sorted(unknown_facts)}")
        if self.optional != self.budget_guarded:
            raise ValueError("optional controlled scenarios must be budget guarded")
        if self.reset_profile.roster_size == 1000 and not self.optional:
            raise ValueError("1,000-agent scenario must be optional and budget guarded")
        return self

    @property
    def expected_agent_id(self) -> str | None:
        identity = self.expected.logical_identity
        if identity is None or identity.startswith("new:"):
            return None
        return str(
            uuid5(
                NAMESPACE_URL,
                f"openpoke-live-lab:{self.reset_profile.fixture_seed}:{identity}",
            )
        )


class _RawScenario(_FrozenModel):
    scenario_id: str
    track: Literal[ScenarioTrack.CONTROLLED] = ScenarioTrack.CONTROLLED
    family: str
    title: str
    turns: tuple[ScenarioTurn, ...]
    expected: ScenarioOutcome
    gmail: GmailExpectation
    reset_profile: str
    repetitions: int
    optional: bool = False
    budget_guarded: bool = False


class _ScenarioDocument(_FrozenModel):
    schema_version: Literal[1]
    scenarios: tuple[_RawScenario, ...]


def _default_controlled_path() -> Path:
    return Path(str(files("evals.live_lab").joinpath("scenarios", "controlled.json")))


def load_controlled_scenarios(path: Path | None = None) -> tuple[ScenarioDefinition, ...]:
    """Load and fail closed on any malformed or incomplete controlled matrix."""

    source = Path(path) if path is not None else _default_controlled_path()
    rendered = source.read_text(encoding="utf-8")
    if _BANNED.search(rendered):
        raise ValueError("controlled scenario contains banned address, secret, or auth URL pattern")
    try:
        document = _ScenarioDocument.model_validate_json(rendered)
    except ValidationError as exc:
        raise ValueError(f"invalid controlled scenario action, identity, repetition, or shape: {exc}") from exc

    ids = [item.scenario_id for item in document.scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate controlled scenario id")
    scenarios: list[ScenarioDefinition] = []
    for raw in document.scenarios:
        profile = RESET_PROFILES.get(raw.reset_profile)
        if profile is None:
            raise ValueError(f"unknown reset profile: {raw.reset_profile}")
        try:
            scenarios.append(
                ScenarioDefinition(
                    **raw.model_dump(exclude={"reset_profile"}),
                    reset_profile=profile,
                )
            )
        except ValidationError as exc:
            raise ValueError(f"invalid scenario identity, fact, repetition, or profile: {exc}") from exc

    families = {item.family for item in scenarios}
    if families != REQUIRED_CONTROLLED_FAMILIES:
        missing = sorted(REQUIRED_CONTROLLED_FAMILIES - families)
        extra = sorted(families - REQUIRED_CONTROLLED_FAMILIES)
        raise ValueError(f"controlled family coverage mismatch; missing={missing}, extra={extra}")
    covered_facts = {fact for item in scenarios for fact in item.gmail.fact_ids}
    if covered_facts != fixture_fact_ids():
        raise ValueError("controlled fact coverage must include the exact fabricated fact manifest")
    thousand = [item for item in scenarios if item.reset_profile.roster_size == 1000]
    if len(thousand) != 1 or not thousand[0].optional or not thousand[0].budget_guarded:
        raise ValueError("exactly one budget-guarded optional 1,000-agent scenario is required")
    return tuple(scenarios)


def predeclare_controlled_scenarios() -> tuple[ScheduledPair, ...]:
    """Freeze every scenario/repetition pair before either runtime executes."""

    pairs: list[ScheduledPair] = []
    for scenario in load_controlled_scenarios():
        pairs.extend(build_repetition_schedule((scenario.scenario_id,), scenario.repetitions))
    return tuple(pairs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate",))
    parser.add_argument("--path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scenarios = load_controlled_scenarios(args.path)
    pairs = predeclare_controlled_scenarios() if args.path is None else tuple(
        pair
        for scenario in scenarios
        for pair in build_repetition_schedule((scenario.scenario_id,), scenario.repetitions)
    )
    print(
        json.dumps(
            {
                "controlled_repetitions": len(pairs),
                "fact_count": len(fixture_fact_ids()),
                "scenario_count": len(scenarios),
                "valid": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
