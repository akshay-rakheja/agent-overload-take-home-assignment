"""Compatibility and migration tests for legacy name-only rosters."""

from __future__ import annotations

import json

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
    assert first[0].legacy_storage_key == "Alice correspondence"
    assert first[1].legacy_storage_key is None
    assert first[2].legacy_storage_key == "José — résumé review"
    assert first == second
    assert path.read_bytes() == migrated_bytes
    assert log_path.read_bytes() == original_log


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
