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
    REQUIRED_CONTROLLED_FAMILIES,
    ScenarioTrack,
    load_controlled_scenarios,
    predeclare_controlled_scenarios,
)


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
