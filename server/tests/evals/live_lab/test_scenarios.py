"""Strict contracts for predeclared controlled Evaluation Lab scenarios."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from evals.live_lab.fixture_email import fixture_fact_ids
from evals.live_lab.scenarios import (
    GmailExpectation,
    REQUIRED_CONTROLLED_FAMILIES,
    RESET_PROFILES,
    ScenarioDefinition,
    ScenarioOutcome,
    ScenarioTrack,
    ScenarioTurn,
    load_controlled_scenarios,
    predeclare_controlled_scenarios,
)
from evals.live_lab.contracts import ExpectedAction
from pydantic import ValidationError


def _raw_controlled() -> dict[str, object]:
    path = Path("evals/live_lab/scenarios/controlled.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "controlled.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_controlled_matrix_predeclares_every_required_axis() -> None:
    scenarios = load_controlled_scenarios()

    assert scenarios
    assert {scenario.family for scenario in scenarios} == REQUIRED_CONTROLLED_FAMILIES
    assert {scenario.track for scenario in scenarios} == {ScenarioTrack.CONTROLLED}
    assert all(scenario.repetitions >= 3 for scenario in scenarios)
    assert all(scenario.turns for scenario in scenarios)
    assert all(scenario.gmail.operation == "GMAIL_FETCH_EMAILS" for scenario in scenarios)
    assert all(scenario.gmail.query.strip() for scenario in scenarios)
    assert {fact for scenario in scenarios for fact in scenario.gmail.fact_ids} == fixture_fact_ids()

    hundred = [scenario for scenario in scenarios if scenario.reset_profile.roster_size == 100]
    depth = [scenario for scenario in scenarios if scenario.reset_profile.history_entries == 10_000]
    thousand = [scenario for scenario in scenarios if scenario.reset_profile.roster_size == 1_000]
    assert hundred
    assert depth
    assert len(thousand) == 1
    assert thousand[0].optional and thousand[0].budget_guarded


def test_predeclaration_is_stable_and_expands_minimum_repetitions() -> None:
    first = predeclare_controlled_scenarios()
    second = predeclare_controlled_scenarios()

    assert first == second
    assert len(first) == sum(item.repetitions for item in load_controlled_scenarios())
    assert len({item.pair_id for item in first}) == len(first)


def test_every_turn_has_an_explicit_action_identity_and_response_evidence_contract() -> None:
    scenarios = load_controlled_scenarios()

    assert all(
        len(scenario.turn_expectations) == len(scenario.turns)
        for scenario in scenarios
    )
    duplicate = next(
        item for item in scenarios if item.scenario_id == "duplicate-clipweaver-prevention"
    )
    assert [item.action.value for item in duplicate.turn_expectations] == [
        "create_new",
        "reuse",
    ]
    assert [item.logical_identity for item in duplicate.turn_expectations] == [
        "new:clipweaver-auditor",
        "new:clipweaver-auditor",
    ]
    assert duplicate.turn_expectations[0].gmail is None
    assert duplicate.turn_expectations[1].gmail == duplicate.gmail

    pronoun = next(
        item for item in scenarios if item.scenario_id == "pronoun-receipt-follow-up"
    )
    assert [item.action.value for item in pronoun.turn_expectations] == [
        "reuse",
        "reuse",
    ]
    assert pronoun.turn_expectations[1].evidence_from_turn == 0
    assert pronoun.turn_expectations[1].response_assertions == ("CAD 47.80",)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda raw: raw["scenarios"].append(deepcopy(raw["scenarios"][0])), "duplicate"),
        (lambda raw: raw["scenarios"][0]["expected"].update(action="delete"), "action"),
        (
            lambda raw: raw["scenarios"][0]["expected"].update(
                action="abstain", logical_identity="instagram-security"
            ),
            "identity",
        ),
        (lambda raw: raw["scenarios"][0]["gmail"].update(fact_ids=["UNKNOWN-1"]), "fact"),
        (lambda raw: raw["scenarios"][0].update(reset_profile="not-a-profile"), "profile"),
        (lambda raw: raw["scenarios"][0].update(repetitions=2), "repetition"),
        (lambda raw: raw["scenarios"][0]["turns"].append({"text": "user@example.com"}), "banned"),
        (
            lambda raw: raw["scenarios"][0]["turns"].append(
                {"text": "https://accounts.google.com/o/oauth2/auth"}
            ),
            "banned",
        ),
    ],
)
def test_loader_fails_closed_on_invalid_predeclarations(tmp_path, mutate, match: str) -> None:
    raw = _raw_controlled()
    mutate(raw)

    with pytest.raises(ValueError, match=match):
        load_controlled_scenarios(_write(tmp_path, raw))


@pytest.mark.parametrize(
    ("encoded", "match"),
    [
        ("user\\u0040example.com", "banned"),
        ("\\u0068ttps:\\/\\/example.com", "banned"),
        ("CLIENT\\u005fSeCrEt = value", "banned"),
        ("Bearer\\u0020opaque-token", "banned"),
        ("sk-fabricatedreview123456789", "banned"),
        ("password=review-secret-value", "banned"),
        ("token: fabricated-token-value", "banned"),
        ("key = fabricated-key-value", "banned"),
    ],
)
def test_loader_rejects_banned_content_after_json_decoding(
    tmp_path: Path, encoded: str, match: str
) -> None:
    raw = _raw_controlled()
    rendered = json.dumps(raw).replace(
        "Ask Instagram Security Monitor to summarize fixture SEC-7419.",
        encoded,
    )
    path = tmp_path / "controlled.json"
    path.write_text(rendered, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_controlled_scenarios(path)


def test_direct_scenario_construction_rejects_nested_banned_content() -> None:
    with pytest.raises(ValidationError, match="banned"):
        ScenarioDefinition(
            scenario_id="direct-invalid",
            family="exact_named_reuse",
            title="Direct invalid",
            turns=(ScenarioTurn(text="nested user@example.com"),),
            expected=ScenarioOutcome(
                action=ExpectedAction.REUSE,
                logical_identity="instagram-security",
            ),
            gmail=GmailExpectation(
                operation="GMAIL_FETCH_EMAILS",
                query='subject:"[OpenPoke Interview Fixture]" "SEC-7419"',
                fact_ids=("SEC-7419",),
            ),
            reset_profile=RESET_PROFILES["standard-100"],
            repetitions=3,
        )


@pytest.mark.parametrize(
    "secret",
    [
        "sk-fabricatedreview123456789",
        "password=review-secret-value",
        "token: fabricated-token-value",
        "key = fabricated-key-value",
    ],
)
def test_direct_turn_construction_rejects_recognizable_secret_material(
    secret: str,
) -> None:
    with pytest.raises(ValidationError, match="banned|secret"):
        ScenarioTurn(text=secret)


@pytest.mark.parametrize(
    ("action", "logical_identity"),
    [
        ("reuse", "new:calendar-workflow"),
        ("create_new", "instagram-security"),
    ],
)
def test_loader_rejects_crossed_action_identity_contracts(
    tmp_path: Path, action: str, logical_identity: str
) -> None:
    raw = _raw_controlled()
    raw["scenarios"][0]["expected"] = {
        "action": action,
        "logical_identity": logical_identity,
    }

    with pytest.raises(ValueError, match="identity|reuse|create"):
        load_controlled_scenarios(_write(tmp_path, raw))


def test_loader_rejects_family_crossed_to_wrong_reset_profile(tmp_path: Path) -> None:
    raw = _raw_controlled()
    raw["scenarios"][0]["reset_profile"] = "scale-1000"
    raw["scenarios"][0]["optional"] = True
    raw["scenarios"][0]["budget_guarded"] = True

    with pytest.raises(ValueError, match="profile"):
        load_controlled_scenarios(_write(tmp_path, raw))


def test_validation_cli_accepts_only_the_tracked_matrix() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "evals.live_lab.scenarios", "validate"],
        cwd=Path(__file__).resolve().parents[4],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert payload["scenario_count"] == len(load_controlled_scenarios())
    assert payload["controlled_repetitions"] >= payload["scenario_count"] * 3
