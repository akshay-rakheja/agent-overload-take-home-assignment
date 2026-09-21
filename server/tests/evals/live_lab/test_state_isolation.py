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
from evals.live_lab import state as state_module
from evals.live_lab.state import (
    compare_logical_state,
    create_snapshot,
    fingerprint_state,
    restore_snapshot,
)
from server.services.execution.directory import AgentDirectory
from server.services.execution.context_policy import ExecutionContextPolicy
from server.services.execution.log_store import ExecutionAgentLogStore, execution_log_slug
from server.agents.execution_agent.agent import ExecutionAgent


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


def test_same_name_histories_are_quarantined_from_both_runtime_identities(tmp_path) -> None:
    _, _, enhanced_dir, _, fingerprint = _materialized_pair(tmp_path)
    execution_dir = enhanced_dir / "execution_agents"
    directory = AgentDirectory(execution_dir / "roster.json")
    records = [record for record in directory.list_records() if record.name == "Campaign Desk"]
    store = ExecutionAgentLogStore(execution_dir)
    ambiguous_before = store.read_raw_bytes("Campaign Desk")

    prompts = [
        ExecutionAgent(
            record.name,
            storage_key=str(record.agent_id),
            agent_id=str(record.agent_id),
            legacy_storage_key=record.legacy_storage_key,
            log_store=store,
            directory=directory,
            context_policy=ExecutionContextPolicy(
                max_recent_episodes=4,
                max_characters=2_000,
            ),
        ).build_system_prompt_with_history()
        for record in records
    ]

    assert len(records) == 2
    assert all(record.legacy_storage_key is None for record in records)
    assert b"RETENTION-CAMPAIGN-SENTINEL" in ambiguous_before
    assert b"PARTNER-CAMPAIGN-SENTINEL" in ambiguous_before
    assert all("RETENTION-CAMPAIGN-SENTINEL" not in prompt for prompt in prompts)
    assert all("PARTNER-CAMPAIGN-SENTINEL" not in prompt for prompt in prompts)
    assert store.read_raw_bytes("Campaign Desk") == ambiguous_before
    assert fingerprint.sentinel_checks["same_name_retention"]
    assert fingerprint.sentinel_checks["same_name_partners"]


@pytest.mark.parametrize("field", ["agent_id", "purpose", "status", "aliases", "legacy_storage_key"])
def test_actual_enhanced_roster_tamper_breaks_logical_equivalence(tmp_path, field: str) -> None:
    _, _, enhanced_dir, baseline, _ = _materialized_pair(tmp_path)
    roster_path = enhanced_dir / "execution_agents" / "roster.json"
    payload = json.loads(roster_path.read_text(encoding="utf-8"))
    record = next(
        item for item in payload["agents"] if item["name"] == "Instagram Security Monitor"
    )
    replacements = {
        "agent_id": "00000000-0000-0000-0000-000000000001",
        "purpose": "tampered purpose",
        "status": "archived",
        "aliases": ["tampered alias"],
        "legacy_storage_key": "tampered owner",
    }
    record[field] = replacements[field]
    roster_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    tampered = fingerprint_state(enhanced_dir)

    assert tampered.logical_identity_digest != baseline.logical_identity_digest
    assert not compare_logical_state(baseline, tampered).equivalent


def test_collision_quarantine_tamper_breaks_logical_equivalence(tmp_path) -> None:
    _, _, enhanced_dir, baseline, _ = _materialized_pair(tmp_path)
    roster_path = enhanced_dir / "execution_agents" / "roster.json"
    payload = json.loads(roster_path.read_text(encoding="utf-8"))
    collision = next(item for item in payload["agents"] if item["name"] == "A B")
    collision["legacy_storage_key"] = "A B"
    roster_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    tampered = fingerprint_state(enhanced_dir)

    assert tampered.logical_identity_digest != baseline.logical_identity_digest
    assert not compare_logical_state(baseline, tampered).equivalent


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


def test_cross_attached_sentinel_fails_expected_owner_check(tmp_path) -> None:
    _, _, enhanced_dir, baseline, _ = _materialized_pair(tmp_path)
    execution_dir = enhanced_dir / "execution_agents"
    wrong_owner = execution_dir / "instagram-security-monitor.log"
    with wrong_owner.open("ab") as handle:
        handle.write(b"<agent_response>CROSS-AGENT-SENTINEL</agent_response>\n")

    tampered = fingerprint_state(enhanced_dir)

    assert tampered.sentinel_checks["cross_agent"] is False
    assert not compare_logical_state(baseline, tampered).equivalent


def test_equivalence_requires_all_owner_scoped_sentinel_checks_to_pass(tmp_path) -> None:
    _, _, _, baseline, _ = _materialized_pair(tmp_path)
    failed_checks = dict(baseline.sentinel_checks)
    failed_checks["cross_agent"] = False
    invalid = baseline.model_copy(update={"sentinel_checks": failed_checks})

    assert not compare_logical_state(invalid, invalid).equivalent


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

    with pytest.raises(ValueError, match="symlink"):
        restore_snapshot(snapshot, allowed_root / "escape" / "data", allowed_root=allowed_root)

    assert marker.read_bytes() == b"do not mutate"


def test_restore_rejects_in_root_symlink_target_without_mutation(tmp_path) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=10)
    source = tmp_path / "source" / "server" / "data"
    materialize_baseline(manifest, source)
    snapshot = create_snapshot(source, tmp_path / "snapshots" / "baseline")
    allowed_root = tmp_path / "runtime"
    allowed_root.mkdir()
    real_target = allowed_root / "real-data"
    real_target.mkdir()
    marker = real_target / "do-not-replace.bin"
    marker.write_bytes(b"preserve in-root target")
    linked_target = allowed_root / "linked-data"
    linked_target.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        restore_snapshot(snapshot, linked_target, allowed_root=allowed_root)

    assert marker.read_bytes() == b"preserve in-root target"
    assert linked_target.is_symlink()


def test_restore_revalidates_target_after_path_swap_without_mutating_link_target(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=10)
    source = tmp_path / "source" / "server" / "data"
    materialize_baseline(manifest, source)
    snapshot = create_snapshot(source, tmp_path / "snapshots" / "baseline")
    allowed_root = tmp_path / "runtime"
    target = allowed_root / "slot" / "server" / "data"
    materialize_enhanced(manifest, target)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "do-not-mutate.bin"
    marker.write_bytes(b"outside remains unchanged")
    displaced = target.with_name("data-displaced")
    original_copy = state_module._copy_snapshot_bytes

    def copy_then_swap(*args, **kwargs) -> None:
        original_copy(*args, **kwargs)
        target.rename(displaced)
        target.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(state_module, "_copy_snapshot_bytes", copy_then_swap)

    with pytest.raises(ValueError, match="symlink"):
        restore_snapshot(snapshot, target, allowed_root=allowed_root)

    assert marker.read_bytes() == b"outside remains unchanged"
    assert target.is_symlink()


def test_restore_stages_through_pinned_parent_when_parent_path_is_replaced(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=10)
    source = tmp_path / "source" / "server" / "data"
    materialize_baseline(manifest, source)
    snapshot = create_snapshot(source, tmp_path / "snapshots" / "baseline")
    allowed_root = tmp_path / "runtime"
    target = allowed_root / "slot" / "server" / "data"
    pinned_fingerprint = materialize_enhanced(manifest, target)
    pinned_parent = target.parent.with_name("server-pinned")
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "do-not-write-here.bin"
    marker.write_bytes(b"outside remains byte-exact")
    original_copy = state_module._copy_snapshot_bytes

    def copy_after_parent_swap(*args, **kwargs) -> None:
        target.parent.rename(pinned_parent)
        target.parent.symlink_to(outside, target_is_directory=True)
        original_copy(*args, **kwargs)

    monkeypatch.setattr(state_module, "_copy_snapshot_bytes", copy_after_parent_swap)

    with pytest.raises(ValueError, match="symlink|parent changed"):
        restore_snapshot(snapshot, target, allowed_root=allowed_root)

    assert list(outside.iterdir()) == [marker]
    assert marker.read_bytes() == b"outside remains byte-exact"
    assert fingerprint_state(pinned_parent / "data") == pinned_fingerprint
    assert target.parent.is_symlink()


def test_restore_rolls_back_exchange_when_ordinary_target_replaces_pinned_target(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = build_fixture_manifest(seed=1313, roster_size=10)
    source = tmp_path / "source" / "server" / "data"
    materialize_baseline(manifest, source)
    snapshot = create_snapshot(source, tmp_path / "snapshots" / "baseline")
    allowed_root = tmp_path / "runtime"
    target = allowed_root / "slot" / "server" / "data"
    pinned_fingerprint = materialize_enhanced(manifest, target)
    displaced = target.with_name("data-pinned")
    unrelated_marker = "unrelated.bin"
    unrelated_bytes = b"ordinary replacement must survive"
    original_exchange = state_module._atomic_exchange_at
    replaced = False

    def exchange_after_target_replacement(parent_fd: int, left_name: str, right_name: str) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            target.rename(displaced)
            target.mkdir()
            (target / unrelated_marker).write_bytes(unrelated_bytes)
        original_exchange(parent_fd, left_name, right_name)

    monkeypatch.setattr(state_module, "_atomic_exchange_at", exchange_after_target_replacement)

    with pytest.raises(RuntimeError, match="changed during atomic exchange"):
        restore_snapshot(snapshot, target, allowed_root=allowed_root)

    assert (target / unrelated_marker).read_bytes() == unrelated_bytes
    assert fingerprint_state(displaced) == pinned_fingerprint


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
