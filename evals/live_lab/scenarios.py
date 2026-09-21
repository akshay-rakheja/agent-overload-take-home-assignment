"""Predeclared controlled and exploratory Evaluation Lab scenarios."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping as MappingABC
from enum import Enum
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from server.services.evaluation_lab.repetitions import ScheduledPair, build_repetition_schedule
from server.services.evaluation_lab.redaction import contains_secret_material

from .contracts import ExpectedAction
from .fixture_email import fixture_fact_ids, fixture_response_facts
from .fixtures import fixture_agent_contract


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
_BANNED = re.compile(
    r"[\w.+-]+@[\w.-]+|https?://|\boauth\b|\bbearer\b|"
    r"api[_ -]?key|access[_ -]?token|client[_ -]?secret|authorization[_ -]?code",
    re.IGNORECASE,
)


def _validate_safe_tree(value: object) -> None:
    if isinstance(value, BaseModel):
        _validate_safe_tree(value.model_dump(mode="python"))
        return
    if isinstance(value, MappingABC):
        for key, item in value.items():
            _validate_safe_tree(str(key))
            _validate_safe_tree(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _validate_safe_tree(item)
        return
    if isinstance(value, str) and (
        _BANNED.search(value) or contains_secret_material(value)
    ):
        raise ValueError("controlled scenario contains banned address, secret, or auth URL pattern")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _reject_banned_nested_values(cls, value: object) -> object:
        _validate_safe_tree(value)
        return value


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


class ScenarioTurnExpectation(_FrozenModel):
    action: ExpectedAction
    logical_identity: str | None = None
    require_clarification: bool = False
    gmail: GmailExpectation | None = None
    response_assertions: tuple[str, ...] = ()
    evidence_from_turn: int | None = Field(default=None, ge=0)

    @field_validator("response_assertions")
    @classmethod
    def _bounded_assertions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or len(item) > 200 for item in normalized):
            raise ValueError("response assertions must be bounded nonempty values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("response assertions must be unique")
        return normalized

    @model_validator(mode="after")
    def _turn_contract(self) -> "ScenarioTurnExpectation":
        if self.action in {ExpectedAction.REUSE, ExpectedAction.CREATE_NEW}:
            if self.logical_identity is None:
                raise ValueError("reuse/create turn requires a logical identity")
        elif self.logical_identity is not None:
            raise ValueError("abstain turn forbids a logical identity")
        if self.action is not ExpectedAction.ABSTAIN and self.require_clarification:
            raise ValueError("only abstain turns may require clarification")
        if self.gmail is not None and self.evidence_from_turn is not None:
            raise ValueError("turn evidence must be current Gmail or an earlier turn, not both")
        if (
            self.response_assertions
            and self.gmail is None
            and self.evidence_from_turn is None
        ):
            raise ValueError("factual response assertions require declared evidence")
        if self.gmail is not None and self.gmail.expect_no_result and self.response_assertions:
            raise ValueError("no-result turns cannot declare factual response assertions")
        return self

    @property
    def outcome(self) -> ScenarioOutcome:
        return ScenarioOutcome(
            action=self.action,
            logical_identity=self.logical_identity,
            require_clarification=self.require_clarification,
        )


class ScenarioIdentityContract(_FrozenModel):
    logical_identity: str
    name: str
    purpose: str
    preexisting: bool
    agent_id: str | None = None

    @model_validator(mode="after")
    def _stable_id_only_for_preexisting(self) -> "ScenarioIdentityContract":
        if self.preexisting != (self.agent_id is not None):
            raise ValueError("only preexisting fixture identities carry a stable agent ID")
        return self


class ScenarioDefinition(_FrozenModel):
    scenario_id: str
    track: Literal[ScenarioTrack.CONTROLLED] = ScenarioTrack.CONTROLLED
    family: str
    title: str
    turns: tuple[ScenarioTurn, ...]
    turn_expectations: tuple[ScenarioTurnExpectation, ...]
    expected: ScenarioOutcome
    identity_contract: ScenarioIdentityContract | None = None
    gmail: GmailExpectation
    reset_profile: ResetProfile
    repetitions: int = Field(ge=3)
    optional: bool = False
    budget_guarded: bool = False
    gmail_required: bool = True
    response_assertions: tuple[str, ...] | None = None

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
        if len(self.turn_expectations) != len(self.turns):
            raise ValueError("every scenario turn requires exactly one predeclared expectation")
        if self.expected.action is ExpectedAction.ABSTAIN:
            if self.identity_contract is not None:
                raise ValueError("abstain scenario cannot carry an identity contract")
        else:
            contract = self.identity_contract
            if contract is None or contract.logical_identity != self.expected.logical_identity:
                raise ValueError("scenario identity must resolve through the selected reset contract")
            if self.expected.action is ExpectedAction.REUSE and not contract.preexisting:
                raise ValueError("reuse identity must exist in the selected reset manifest")
            if self.expected.action is ExpectedAction.CREATE_NEW and contract.preexisting:
                raise ValueError("create identity must be novel to the selected reset manifest")
        unknown_facts = set(self.gmail.fact_ids) - fixture_fact_ids()
        if unknown_facts:
            raise ValueError(f"unknown fact ids: {sorted(unknown_facts)}")
        if self.optional != self.budget_guarded:
            raise ValueError("optional controlled scenarios must be budget guarded")
        if self.reset_profile.roster_size == 1000 and not self.optional:
            raise ValueError("1,000-agent scenario must be optional and budget guarded")
        if self.family == "thousand_agent_overload":
            if self.reset_profile.profile_id != "scale-1000":
                raise ValueError("thousand-agent family requires the scale-1000 profile")
        elif self.reset_profile.profile_id != "standard-100":
            raise ValueError("non-thousand controlled families require the standard-100 profile")
        return self

    @property
    def expected_agent_id(self) -> str | None:
        return self.identity_contract.agent_id if self.identity_contract is not None else None

    @property
    def expected_identity_name(self) -> str | None:
        return self.identity_contract.name if self.identity_contract is not None else None

    @property
    def expected_identity_purpose(self) -> str | None:
        return self.identity_contract.purpose if self.identity_contract is not None else None


class _RawScenario(_FrozenModel):
    scenario_id: str
    track: Literal[ScenarioTrack.CONTROLLED] = ScenarioTrack.CONTROLLED
    family: str
    title: str
    turns: tuple[ScenarioTurn, ...]
    turn_expectations: tuple[ScenarioTurnExpectation, ...]
    expected: ScenarioOutcome
    gmail: GmailExpectation
    reset_profile: str
    repetitions: int
    optional: bool = False
    budget_guarded: bool = False


class _NovelIdentity(_FrozenModel):
    logical_identity: str
    name: str
    purpose: str

    @field_validator("logical_identity", "name", "purpose")
    @classmethod
    def _bounded_contract_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 200:
            raise ValueError("novel identity contract text must be bounded and nonempty")
        return normalized

    @model_validator(mode="after")
    def _novel_prefix(self) -> "_NovelIdentity":
        if not self.logical_identity.startswith("new:"):
            raise ValueError("novel identity keys must start with new:")
        return self


class _ScenarioDocument(_FrozenModel):
    schema_version: Literal[1]
    novel_identities: tuple[_NovelIdentity, ...]
    scenarios: tuple[_RawScenario, ...]


def _default_controlled_path() -> Path:
    return Path(str(files("evals.live_lab").joinpath("scenarios", "controlled.json")))


def load_controlled_scenarios(path: Path | None = None) -> tuple[ScenarioDefinition, ...]:
    """Load and fail closed on any malformed or incomplete controlled matrix."""

    source = Path(path) if path is not None else _default_controlled_path()
    rendered = source.read_text(encoding="utf-8")
    try:
        document = _ScenarioDocument.model_validate_json(rendered)
    except ValidationError as exc:
        raise ValueError(f"invalid controlled scenario action, identity, repetition, or shape: {exc}") from exc

    ids = [item.scenario_id for item in document.scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate controlled scenario id")
    novel_ids = [item.logical_identity for item in document.novel_identities]
    if len(novel_ids) != len(set(novel_ids)):
        raise ValueError("duplicate novel identity contract")
    novel_contracts = {item.logical_identity: item for item in document.novel_identities}
    referenced_novel_ids = {
        item.expected.logical_identity
        for item in document.scenarios
        if item.expected.action is ExpectedAction.CREATE_NEW
    }
    if referenced_novel_ids != set(novel_contracts):
        raise ValueError("novel identity contracts must exactly match controlled create scenarios")
    if len({item.name.casefold() for item in document.novel_identities}) != len(document.novel_identities):
        raise ValueError("novel identity names must be unique")
    scenarios: list[ScenarioDefinition] = []
    for raw in document.scenarios:
        profile = RESET_PROFILES.get(raw.reset_profile)
        if profile is None:
            raise ValueError(f"unknown reset profile: {raw.reset_profile}")
        fixture_contracts = fixture_agent_contract(
            seed=profile.fixture_seed, roster_size=profile.roster_size
        )
        fixture_names = {item.name.casefold() for item in fixture_contracts.values()}
        logical_identity = raw.expected.logical_identity
        identity_contract: ScenarioIdentityContract | None = None
        if raw.expected.action is ExpectedAction.REUSE:
            fixture = fixture_contracts.get(logical_identity or "")
            if fixture is None:
                raise ValueError("reuse identity must exist in selected reset profile")
            identity_contract = ScenarioIdentityContract(
                logical_identity=fixture.logical_identity,
                name=fixture.name,
                purpose=fixture.purpose,
                preexisting=True,
                agent_id=fixture.agent_id,
            )
        elif raw.expected.action is ExpectedAction.CREATE_NEW:
            novel = novel_contracts.get(logical_identity or "")
            if (
                novel is None
                or logical_identity in fixture_contracts
                or novel.name.casefold() in fixture_names
            ):
                raise ValueError("create identity must be declared novel and absent from reset profile")
            identity_contract = ScenarioIdentityContract(
                logical_identity=novel.logical_identity,
                name=novel.name,
                purpose=novel.purpose,
                preexisting=False,
            )
        prior_created: set[str] = set()
        for turn_index, turn in enumerate(raw.turn_expectations):
            turn_identity = turn.logical_identity or ""
            if turn.action is ExpectedAction.REUSE:
                if turn_identity not in fixture_contracts and turn_identity not in prior_created:
                    raise ValueError(
                        "reuse turn identity must exist in reset or an earlier create turn"
                    )
            elif turn.action is ExpectedAction.CREATE_NEW:
                if turn_identity not in novel_contracts or turn_identity in fixture_contracts:
                    raise ValueError("create turn identity must use a declared novel contract")
                prior_created.add(turn_identity)
            if turn.evidence_from_turn is not None:
                if turn.evidence_from_turn >= turn_index:
                    raise ValueError("turn evidence must reference an earlier turn")
                source = raw.turn_expectations[turn.evidence_from_turn]
                if source.gmail is None:
                    raise ValueError("referenced evidence turn must declare Gmail evidence")
                allowed_assertions = {
                    value
                    for fact_id in source.gmail.fact_ids
                    for value in fixture_response_facts(fact_id)
                }
            elif turn.gmail is not None:
                allowed_assertions = {
                    value
                    for fact_id in turn.gmail.fact_ids
                    for value in fixture_response_facts(fact_id)
                }
            else:
                allowed_assertions = set()
            if not set(turn.response_assertions).issubset(allowed_assertions):
                raise ValueError(
                    "turn response assertions must come from its declared fixture evidence"
                )
        try:
            scenarios.append(
                ScenarioDefinition(
                    **raw.model_dump(exclude={"reset_profile"}),
                    reset_profile=profile,
                    identity_contract=identity_contract,
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
