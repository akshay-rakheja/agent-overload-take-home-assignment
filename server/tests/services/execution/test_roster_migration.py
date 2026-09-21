"""Compatibility and migration tests for legacy name-only rosters."""

from __future__ import annotations

import json

from evals.live_lab.fixtures import build_fixture_manifest, materialize_baseline
from server.services.execution.directory import AgentDirectory
from server.services.execution.roster import AgentRoster


def test_legacy_name_list_migrates_without_loss_and_is_idempotent(tmp_path) -> None:
    path = tmp_path / "roster.json"
    names = ["Alice correspondence", "Alice correspondence", "José — résumé review"]
    path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = tmp_path / "alice-correspondence.log"
    original_log = b"<agent_request>keep me</agent_request>\n"
    log_path.write_bytes(original_log)

    first = AgentDirectory(path).list_records()
    migrated_bytes = path.read_bytes()
    second = AgentDirectory(path).list_records()

    assert [record.name for record in first] == names
    assert len({record.agent_id for record in first}) == 3
    assert first[0].legacy_storage_key is None
    assert first[1].legacy_storage_key is None
    assert first[2].legacy_storage_key == "José — résumé review"
    assert first == second
    assert path.read_bytes() == migrated_bytes
    assert log_path.read_bytes() == original_log


def test_legacy_log_ownership_uses_actual_filesystem_slug(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_text(
        json.dumps(["José", "Jose", "A B", "A-B"], ensure_ascii=False),
        encoding="utf-8",
    )

    records = AgentDirectory(path).list_records()

    assert [record.legacy_storage_key for record in records] == [
        "José",
        "Jose",
        None,
        None,
    ]


def test_later_explicit_claim_is_reserved_before_deterministic_backfill(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(["A B", "A-B"]), encoding="utf-8")
    AgentDirectory(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["agents"][0].pop("legacy_storage_key")
    payload["agents"][1]["legacy_storage_key"] = "A-B"
    path.write_text(json.dumps(payload), encoding="utf-8")

    first, explicit_claimant = AgentDirectory(path).list_records()

    assert first.legacy_storage_key is None
    assert explicit_claimant.legacy_storage_key == "A-B"


def test_conflicting_explicit_legacy_claims_are_quarantined(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(["A B", "A-B"]), encoding="utf-8")
    AgentDirectory(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["agents"][0]["legacy_storage_key"] = "A B"
    payload["agents"][1]["legacy_storage_key"] = "A-B"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = AgentDirectory(path).list_records()

    assert [record.legacy_storage_key for record in records] == [None, None]


def test_previous_structured_migration_backfills_only_deterministic_legacy_ids(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(["Alice", "Bob"]), encoding="utf-8")
    AgentDirectory(path)
    migrated_payload = json.loads(path.read_text(encoding="utf-8"))
    for item in migrated_payload["agents"]:
        item.pop("legacy_storage_key")
    migrated_payload["agents"][1]["agent_id"] = "00000000-0000-0000-0000-000000000001"
    path.write_text(json.dumps(migrated_payload), encoding="utf-8")

    records = AgentDirectory(path).list_records()

    assert records[0].legacy_storage_key == "Alice"
    assert records[1].legacy_storage_key is None
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert all("legacy_storage_key" in item for item in persisted["agents"])


def test_compatibility_roster_preserves_existing_name_api(tmp_path) -> None:
    path = tmp_path / "roster.json"
    roster = AgentRoster(path)

    roster.add_agent("Alice correspondence")
    roster.add_agent("Alice correspondence")
    roster.add_agent("Bob invoices")

    assert roster.get_agents() == ["Alice correspondence", "Bob invoices"]
    assert AgentRoster(path).get_agents() == ["Alice correspondence", "Bob invoices"]


def test_compatibility_roster_clear_empties_directory(tmp_path) -> None:
    path = tmp_path / "roster.json"
    roster = AgentRoster(path)
    roster.add_agent("Alice correspondence")

    roster.clear()

    assert roster.get_agents() == []
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"agents": [], "schema_version": 1}


def test_generated_legacy_fixture_migration_preserves_all_raw_journal_bytes(tmp_path) -> None:
    data_dir = tmp_path / "server" / "data"
    manifest = build_fixture_manifest(seed=1313, roster_size=100)
    materialize_baseline(manifest, data_dir)
    execution_dir = data_dir / "execution_agents"
    before = {
        path.relative_to(execution_dir): path.read_bytes()
        for path in sorted(execution_dir.glob("*.log"))
    }

    records = AgentDirectory(execution_dir / "roster.json").list_records()
    after = {
        path.relative_to(execution_dir): path.read_bytes()
        for path in sorted(execution_dir.glob("*.log"))
    }

    assert len(records) == 100
    assert before == after
    assert [record.legacy_storage_key for record in records if record.name in {"A B", "A-B"}] == [
        None,
        None,
    ]
    same_name = [record for record in records if record.name == "Campaign Desk"]
    assert len(same_name) == 2
    assert len({record.agent_id for record in same_name}) == 2
    assert all(record.legacy_storage_key is None for record in same_name)
