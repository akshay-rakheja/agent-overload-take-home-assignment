"""Behavioral contract for the lifecycle-aware Agent Directory."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from server.services.execution.directory import (
    AgentDirectory,
    DirectoryCorruptError,
    UnknownAgentError,
)
from server.services.execution.models import AgentStatus, normalize_agent_text


UTC = timezone.utc


def test_create_persists_stable_identity_across_restart(tmp_path) -> None:
    path = tmp_path / "roster.json"
    now = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    directory = AgentDirectory(path, clock=lambda: now)

    created = directory.create(
        name="José — résumé review",
        purpose="Coordinate résumé review with José",
        aliases=["José", "Jose"],
    )
    reloaded = AgentDirectory(path, clock=lambda: now).require(created.agent_id)

    assert isinstance(created.agent_id, UUID)
    assert reloaded == created
    assert created.status is AgentStatus.HOT
    assert created.created_at == now
    assert created.last_used_at == now
    assert created.schema_version == 1


def test_lifecycle_transitions_and_successful_dispatch_metadata_use_injected_clock(tmp_path) -> None:
    path = tmp_path / "roster.json"
    current = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    directory = AgentDirectory(path, clock=lambda: current)
    created = directory.create(name="Alice", purpose="Track Alice correspondence")

    current += timedelta(days=1)
    dormant = directory.transition(created.agent_id, AgentStatus.DORMANT)
    assert dormant.status is AgentStatus.DORMANT

    current += timedelta(seconds=1)
    used = directory.mark_used(created.agent_id)
    assert used.status is AgentStatus.HOT
    assert used.last_used_at == current
    assert used.use_count == 1

    archived = directory.transition(created.agent_id, AgentStatus.ARCHIVED)
    assert archived.status is AgentStatus.ARCHIVED
    assert directory.require(created.agent_id).status is AgentStatus.ARCHIVED


def test_unknown_stable_id_fails_closed(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")

    with pytest.raises(UnknownAgentError, match="Unknown execution agent"):
        directory.require(UUID("00000000-0000-0000-0000-000000000001"))


def test_alias_normalization_preserves_display_text_without_merging_duplicates(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    first = directory.create(
        name="José — résumé review",
        purpose="Review one résumé",
        aliases=["José", "O’Connor & Sons"],
    )
    second = directory.create(
        name="José — contract review",
        purpose="Review one contract",
        aliases=["Jose", "OConnor and Sons"],
    )

    assert normalize_agent_text("José") == normalize_agent_text("Jose")
    assert normalize_agent_text("Maya's answer") == "maya answer"
    assert normalize_agent_text("Alice-correspondence") == "alice correspondence"
    assert normalize_agent_text("O’Connor & Sons") == "oconnor and sons"
    assert first.aliases == ("José", "O’Connor & Sons")
    assert second.aliases == ("Jose", "OConnor and Sons")
    assert {record.agent_id for record in directory.find_by_alias("Jose")} == {
        first.agent_id,
        second.agent_id,
    }


def test_archived_records_remain_searchable(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    record = directory.create(name="Atlas audit", purpose="Collect audit evidence", aliases=["Atlas"])
    directory.transition(record.agent_id, AgentStatus.ARCHIVED)

    matches = directory.find_by_alias("atlas")

    assert [match.agent_id for match in matches] == [record.agent_id]
    assert matches[0].status is AgentStatus.ARCHIVED


def test_malformed_directory_fails_without_overwriting_source(tmp_path) -> None:
    path = tmp_path / "roster.json"
    malformed = b'{"schema_version": 1, "agents": ['
    path.write_bytes(malformed)

    with pytest.raises(DirectoryCorruptError, match="Could not parse"):
        AgentDirectory(path)

    assert path.read_bytes() == malformed


def test_concurrent_creates_leave_complete_valid_json(tmp_path) -> None:
    path = tmp_path / "roster.json"

    def create(index: int) -> None:
        AgentDirectory(path).create(
            name=f"Agent {index}",
            purpose=f"Handle task {index}",
            aliases=[f"task-{index}"],
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(create, range(40)))

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = AgentDirectory(path).list_records()
    assert payload["schema_version"] == 1
    assert len(payload["agents"]) == 40
    assert len(records) == 40
    assert len({record.agent_id for record in records}) == 40
