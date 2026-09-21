"""Deterministic, fabricated fixtures for the live evaluation lab."""

from __future__ import annotations

import hashlib
import re

import pytest

from evals.live_lab.contracts import ExpectedAction
from evals.live_lab.fixtures import build_fixture_manifest, serialize_manifest


def test_manifest_is_seed_deterministic() -> None:
    first = serialize_manifest(build_fixture_manifest(seed=1313, roster_size=100))
    second = serialize_manifest(build_fixture_manifest(seed=1313, roster_size=100))

    assert first == second
    assert first.endswith(b"\n")


def test_manifest_changes_when_seed_changes() -> None:
    first = serialize_manifest(build_fixture_manifest(seed=1313, roster_size=100))
    second = serialize_manifest(build_fixture_manifest(seed=1314, roster_size=100))

    assert first != second


@pytest.mark.parametrize("roster_size", [10, 100, 500, 1_000])
def test_roster_sizes_are_exact(roster_size: int) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=roster_size)

    assert manifest.roster_size == roster_size
    assert len(manifest.agents) == roster_size
    assert len({agent.logical_id for agent in manifest.agents}) == roster_size


def test_default_manifest_contains_required_confusables_and_lifecycle_cases() -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=100)
    agents = list(manifest.agents)

    assert any("Instagram" in item.name and "security" in item.purpose.lower() for item in agents)
    assert any("Instagram" in item.name and "engagement" in item.purpose.lower() for item in agents)
    assert any("AI Video" in item.name and "newsletter" in item.purpose.lower() for item in agents)
    assert any("AI Video" in item.name and "receipt" in item.purpose.lower() for item in agents)
    campaign_desks = [item for item in agents if item.name == "Campaign Desk"]
    assert len(campaign_desks) == 2
    assert len({item.purpose for item in campaign_desks}) == 2
    assert {item.name for item in agents}.issuperset({"A B", "A-B", "José Research", "Jose Research"})
    assert {item.status for item in agents}.issuperset({"hot", "dormant", "archived"})
    assert any("OLD-RELEVANT" in entry.payload for item in agents for entry in item.journal_entries)
    assert any("RECENT-DISTRACTOR" in entry.payload for item in agents for entry in item.journal_entries)
    assert sum(item.name.startswith("Unrelated Workflow") for item in agents) >= 80


def test_every_scenario_has_an_explicit_valid_expectation() -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=100)
    logical_ids = {agent.logical_id for agent in manifest.agents}

    assert manifest.scenarios
    assert len({scenario.scenario_id for scenario in manifest.scenarios}) == len(manifest.scenarios)
    for scenario in manifest.scenarios:
        assert isinstance(scenario.expected_action, ExpectedAction)
        assert scenario.expected_gmail_fact_ids == ()
        if scenario.expected_action is ExpectedAction.REUSE:
            assert scenario.expected_logical_identity in logical_ids
        else:
            assert scenario.expected_logical_identity is None


def test_default_manifest_golden_sha_and_contains_no_live_data_patterns() -> None:
    rendered = serialize_manifest(build_fixture_manifest(seed=1313, roster_size=100))

    assert hashlib.sha256(rendered).hexdigest() == (
        "2aa92eb06ccd933753c3118880a1c2d23a1490ef398aa56c05e11f6a9e9514c5"
    )
    assert not re.search(rb"https?://|oauth|bearer|api[_-]?key|@(?:gmail|googlemail)\.", rendered, re.I)
