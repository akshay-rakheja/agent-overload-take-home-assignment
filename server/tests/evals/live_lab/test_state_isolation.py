"""Byte equivalence and isolated reset behavior for live-lab state."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evals.live_lab.fixtures import (
    build_fixture_manifest,
    materialize_baseline,
    materialize_enhanced,
)
from evals.live_lab.state import (
    compare_logical_state,
    create_snapshot,
    fingerprint_state,
    restore_snapshot,
)
from server.services.execution.directory import AgentDirectory
from server.services.execution.log_store import ExecutionAgentLogStore, execution_log_slug


def _materialized_pair(tmp_path: Path):
    manifest = build_fixture_manifest(seed=1313, roster_size=100)
    baseline_dir = tmp_path / "baseline" / "server" / "data"
    enhanced_dir = tmp_path / "enhanced" / "server" / "data"
    baseline = materialize_baseline(manifest, baseline_dir)
    enhanced = materialize_enhanced(manifest, enhanced_dir)
    return manifest, baseline_dir, enhanced_dir, baseline, enhanced


def _journal_bytes(data_dir: Path) -> dict[str, bytes]:
    execution_dir = data_dir / "execution_agents"
    return {
        path.name: path.read_bytes()
        for path in sorted(execution_dir.glob("*.log"))
    }


def test_materializers_have_equivalent_journals_and_intentionally_different_rosters(tmp_path) -> None:
    manifest, baseline_dir, enhanced_dir, baseline, enhanced = _materialized_pair(tmp_path)

    baseline_roster = json.loads((baseline_dir / "execution_agents" / "roster.json").read_text())
    enhanced_roster = json.loads((enhanced_dir / "execution_agents" / "roster.json").read_text())
    assert isinstance(baseline_roster, list)
    assert isinstance(enhanced_roster, dict)
    assert baseline_roster == [agent.name for agent in manifest.agents]
    assert _journal_bytes(baseline_dir) == _journal_bytes(enhanced_dir)
    assert baseline.roster_sha256 != enhanced.roster_sha256
    assert compare_logical_state(baseline, enhanced).equivalent


def test_enhanced_migration_preserves_selected_and_cross_agent_sentinel_bytes(tmp_path) -> None:
    manifest, baseline_dir, enhanced_dir, _, _ = _materialized_pair(tmp_path)
    selected = next(agent for agent in manifest.agents if agent.name == "AI Video Newsletter Curator")
    sentinel = next(agent for agent in manifest.agents if agent.name == "Jose Research")

    for agent in (selected, sentinel):
        relative = Path("execution_agents") / f"{execution_log_slug(agent.name)}.log"
        assert (enhanced_dir / relative).read_bytes() == (baseline_dir / relative).read_bytes()


def test_slug_collision_is_not_cross_attached(tmp_path) -> None:
    _, _, enhanced_dir, _, _ = _materialized_pair(tmp_path)
    execution_dir = enhanced_dir / "execution_agents"
    records = AgentDirectory(execution_dir / "roster.json").list_records()
    colliding = [record for record in records if record.name in {"A B", "A-B"}]
    raw_collision = (execution_dir / "a-b.log").read_bytes()

    assert len(colliding) == 2
    assert len({record.agent_id for record in colliding}) == 2
    assert [record.legacy_storage_key for record in colliding] == [None, None]
    assert b"SPACE-COLLISION-SENTINEL" in raw_collision
    assert b"HYPHEN-COLLISION-SENTINEL" in raw_collision


def test_unicode_equivalents_keep_storage_ownership(tmp_path) -> None:
    _, _, enhanced_dir, _, _ = _materialized_pair(tmp_path)
    execution_dir = enhanced_dir / "execution_agents"
    records = AgentDirectory(execution_dir / "roster.json").list_records()
    by_name = {record.name: record for record in records}

    assert by_name["José Research"].legacy_storage_key == "José Research"
    assert by_name["Jose Research"].legacy_storage_key == "Jose Research"
    assert b"ACCENTED-UNICODE-SENTINEL" in (execution_dir / "josé-research.log").read_bytes()
    assert b"PLAIN-UNICODE-SENTINEL" in (execution_dir / "jose-research.log").read_bytes()


def test_same_name_different_purpose_remains_distinct(tmp_path) -> None:
    manifest, _, enhanced_dir, _, enhanced = _materialized_pair(tmp_path)
    logical = [agent for agent in manifest.agents if agent.name == "Campaign Desk"]
    records = [
        record
        for record in AgentDirectory(enhanced_dir / "execution_agents" / "roster.json").list_records()
        if record.name == "Campaign Desk"
    ]

    assert len(logical) == 2
    assert len({agent.purpose for agent in logical}) == 2
    assert len(records) == 2
    assert len({record.agent_id for record in records}) == 2
    assert [record.purpose for record in records] == [agent.purpose for agent in logical]
    assert [record.status.value for record in records] == [agent.status for agent in logical]
    assert [record.legacy_storage_key for record in records] == [None, None]
    assert enhanced.roster_count == 100

    depth_record = next(
        record
        for record in AgentDirectory(enhanced_dir / "execution_agents" / "roster.json").list_records()
        if record.name == "AI Video Newsletter Curator"
    )
    assert depth_record.memory_summary.startswith("Durable summary:")


def test_10000_entry_depth_fixture_has_complete_episodes_and_isolated_sentinel(tmp_path) -> None:
    manifest, baseline_dir, _, baseline, _ = _materialized_pair(tmp_path)
    depth = next(agent for agent in manifest.agents if agent.name == "AI Video Newsletter Curator")
    depth_store = ExecutionAgentLogStore(baseline_dir / "execution_agents")
    entries = list(depth_store.iter_entries(depth.name))
    tags = [entry[0] for entry in entries]
    raw_depth = depth_store.read_raw_bytes(depth.name)

    assert len(depth.journal_entries) == 10_000
    assert len(entries) == 10_000
    assert tags == ["agent_request", "agent_action", "tool_response", "agent_response"] * 2_500
    assert b"DEPTH-START-SENTINEL" in raw_depth
    assert b"DEPTH-END-SENTINEL" in raw_depth
    assert b"CROSS-AGENT-SENTINEL" not in raw_depth
    assert baseline.journal_entries == 10_396
    assert all(baseline.sentinel_checks.values())


@pytest.mark.parametrize("mutation", ["append", "truncate"])
def test_restore_recovers_exact_fingerprint_after_partial_writes(tmp_path, mutation: str) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=10)
    allowed_root = tmp_path / "runtime"
    data_dir = allowed_root / "baseline" / "server" / "data"
    pristine = materialize_baseline(manifest, data_dir)
    snapshot = create_snapshot(data_dir, tmp_path / "snapshots" / "baseline")
    journal = next((data_dir / "execution_agents").glob("*.log"))
    if mutation == "append":
        with journal.open("ab") as handle:
            handle.write(b"<agent_request>partial")
    else:
        journal.write_bytes(journal.read_bytes()[:17])

    restored = restore_snapshot(snapshot, data_dir, allowed_root=allowed_root)

    assert restored == pristine
    assert fingerprint_state(data_dir) == pristine


def test_pristine_and_repeated_reset_are_idempotent(tmp_path) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=10)
    allowed_root = tmp_path / "runtime"
    data_dir = allowed_root / "enhanced" / "server" / "data"
    pristine = materialize_enhanced(manifest, data_dir)
    snapshot = create_snapshot(data_dir, tmp_path / "snapshots" / "enhanced")

    first = restore_snapshot(snapshot, data_dir, allowed_root=allowed_root)
    second = restore_snapshot(snapshot, data_dir, allowed_root=allowed_root)

    assert first == pristine
    assert second == pristine


def test_simultaneous_roots_reset_independently(tmp_path) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=10)
    allowed_root = tmp_path / "runtime"
    left = allowed_root / "left" / "server" / "data"
    right = allowed_root / "right" / "server" / "data"
    left_pristine = materialize_baseline(manifest, left)
    right_pristine = materialize_enhanced(manifest, right)
    left_snapshot = create_snapshot(left, tmp_path / "snapshots" / "left")
    right_snapshot = create_snapshot(right, tmp_path / "snapshots" / "right")
    (left / "execution_agents" / "a-b.log").write_bytes(b"left damaged")
    (right / "execution_agents" / "a-b.log").write_bytes(b"right damaged")

    with ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(restore_snapshot, left_snapshot, left, allowed_root=allowed_root)
        right_future = executor.submit(restore_snapshot, right_snapshot, right, allowed_root=allowed_root)

    assert left_future.result() == left_pristine
    assert right_future.result() == right_pristine
    right_before = _journal_bytes(right)
    restore_snapshot(left_snapshot, left, allowed_root=allowed_root)
    assert _journal_bytes(right) == right_before


def test_restore_rejects_symlink_escape_without_mutating_outside_tree(tmp_path) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=10)
    source = tmp_path / "source" / "server" / "data"
    materialize_baseline(manifest, source)
    snapshot = create_snapshot(source, tmp_path / "snapshots" / "baseline")
    allowed_root = tmp_path / "runtime"
    allowed_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker.bin"
    marker.write_bytes(b"do not mutate")
    (allowed_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside allowed_root"):
        restore_snapshot(snapshot, allowed_root / "escape" / "data", allowed_root=allowed_root)

    assert marker.read_bytes() == b"do not mutate"


@pytest.mark.parametrize(
    "target_kind",
    ["outside", "allowed-root", "filesystem-root", "home", "repository-root"],
)
def test_restore_rejects_unsafe_target_before_mutation(tmp_path, target_kind: str) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=10)
    source = tmp_path / "source" / "server" / "data"
    materialize_baseline(manifest, source)
    snapshot = create_snapshot(source, tmp_path / "snapshots" / "baseline")
    allowed_root = tmp_path / "runtime"
    allowed_root.mkdir()
    targets = {
        "outside": tmp_path / "outside-target",
        "allowed-root": allowed_root,
        "filesystem-root": Path("/"),
        "home": Path.home(),
        "repository-root": Path(__file__).resolve().parents[4],
    }

    with pytest.raises(ValueError):
        restore_snapshot(snapshot, targets[target_kind], allowed_root=allowed_root)

    assert not (tmp_path / "outside-target").exists()


def test_cli_prepare_fingerprint_and_reset_use_explicit_paths(tmp_path) -> None:
    output_root = tmp_path / "cli-runtime"
    prepare = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.live_lab.cli",
            "prepare",
            "--seed",
            "1313",
            "--size",
            "10",
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    prepared = json.loads(prepare.stdout)
    baseline_dir = output_root / "baseline" / "server" / "data"
    fingerprint = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.live_lab.cli",
            "fingerprint",
            "--data-dir",
            str(baseline_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fingerprinted = json.loads(fingerprint.stdout)
    journal = next((baseline_dir / "execution_agents").glob("*.log"))
    with journal.open("ab") as handle:
        handle.write(b"<agent_request>interrupted")
    reset = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.live_lab.cli",
            "reset",
            "--snapshot-dir",
            str(output_root / "snapshots" / "baseline"),
            "--data-dir",
            str(baseline_dir),
            "--allowed-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    reset_payload = json.loads(reset.stdout)

    assert prepared["equivalent"] is True
    assert prepared["baseline"]["roster_count"] == 10
    assert fingerprinted["raw_journal_digest"] == prepared["baseline"]["raw_journal_digest"]
    assert reset_payload["before"]["raw_journal_digest"] != reset_payload["after"]["raw_journal_digest"]
    assert reset_payload["after"]["raw_journal_digest"] == prepared["baseline"]["raw_journal_digest"]
    assert "SENTINEL" not in prepare.stdout + fingerprint.stdout + reset.stdout
